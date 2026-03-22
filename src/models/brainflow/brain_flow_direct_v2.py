"""BrainFlow Direct v2 — CSFM-Inspired Flow Matching in fMRI Space.

Key features:
  - Factored Cross-Modal Fusion (per-timestep M×M attention)
  - CSFM: Condition-dependent source x₀ = μ(context) + ε·σ(context)
  - Per-subject output heads (SubjectLayers)
  - Align loss (μ → target) + var-KLD loss (regularize σ)

Architecture:
  Context (B, T, [mod_dims...])
    → FactoredCrossModalFusion → context_encoded (B, T, hidden_dim)

  SourcePredictor:
    context_encoded → pool → MLP → μ, log_var → x₀ (learned source)

  VelocityNet:
    x_t (B, V) → Input Proj → h (B, hidden_dim)
    t → SinPosEmb → MLP → t_emb + SubjectEmb
    8× CrossAttentionResBlock(h, t_emb, context_encoded)
    → LayerNorm → SubjectLayers → v_pred (B, V)

Usage:
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct_v2.yaml
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver


# =============================================================================
# Building Blocks
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t * 1000.0
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SubjectLayers(nn.Module):
    """Per-subject linear output head (from TRIBE/Brain-Diffuser)."""

    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels)) if bias else None

        self.weights.data.normal_(0, 1.0 / in_channels ** 0.5)
        if self.bias is not None:
            self.bias.data.normal_(0, 1.0 / in_channels ** 0.5)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        w = self.weights[subject_ids]
        out = torch.einsum("bd,bdo->bo", x, w)
        if self.bias is not None:
            b = self.bias[subject_ids]
            out = out + b
        return out


# =============================================================================
# CrossAttentionResBlock — FiLM (time) + FFN + CrossAttn (context)
# =============================================================================

class CrossAttentionResBlock(nn.Module):
    """Residual block with FiLM (time) + cross-attention (processed context)."""

    def __init__(self, dim: int, time_dim: int, context_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.film = nn.Linear(time_dim, dim * 2)

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale_shift = self.film(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.ffn(h)

        q = self.norm_q(x).unsqueeze(1)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = x + attn_out.squeeze(1)

        return x


# =============================================================================
# FactoredCrossModalFusion — Efficient Cross-Modal Attention
# =============================================================================

class FactoredCrossModalFusion(nn.Module):
    """Factored Cross-Modal Attention Fusion.

    Per-timestep M×M cross-modal attention instead of full T*M self-attention.
    ~11× faster with comparable cross-modal reasoning.
    """

    def __init__(
        self,
        modality_dims: list[int],
        hidden_dim: int = 1024,
        max_seq_len: int = 11,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.modality_dropout = modality_dropout

        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for dim in modality_dims
        ])

        self.modality_emb = nn.Parameter(
            torch.randn(1, 1, self.n_modalities, hidden_dim) * 0.02
        )

        self.cross_modal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        B, T = modality_features[0].shape[:2]
        T = min(T, self.max_seq_len)

        projected = []
        for feat, proj in zip(modality_features, self.projectors):
            projected.append(proj(feat[:, :T]))

        stacked = torch.stack(projected, dim=2)
        stacked = stacked + self.modality_emb

        if self.training and self.modality_dropout > 0:
            keep_mask = (
                torch.rand(B, 1, self.n_modalities, 1, device=stacked.device)
                > self.modality_dropout
            ).float()
            all_dropped = (keep_mask.sum(dim=2, keepdim=True) == 0).float()
            keep_mask[:, :, 0:1, :] = torch.max(
                keep_mask[:, :, 0:1, :], all_dropped
            )
            stacked = stacked * keep_mask

        x = stacked.reshape(B * T, self.n_modalities, self.hidden_dim)

        for layer in self.cross_modal_layers:
            x = layer(x)

        x = x.mean(dim=1)
        context = x.reshape(B, T, self.hidden_dim)
        return self.final_norm(context)


# =============================================================================
# SourcePredictor — CSFM-Inspired Condition-Dependent Source
# =============================================================================

class SourcePredictor(nn.Module):
    """Predicts condition-dependent source x₀ for flow matching (CSFM).

    Instead of x₀ = 0 or x₀ ~ N(0,I), learns x₀ = μ(context) + ε·σ(context).
    This gives the ODE a head-start: shorter flow → easier velocity → better PCC.

    Architecture:
        context (B, T, D) → mean-pool → MLP → μ (B, output_dim)
                                             → log_var (B, output_dim) [optional]
        x₀ = μ + ε·σ  (training)
        x₀ = μ        (inference)
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        n_subjects: int = 4,
        use_variational: bool = True,
    ):
        super().__init__()
        self.use_variational = use_variational

        # Temporal pooling + hidden projection
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-subject mean prediction (different subjects → different fMRI distributions)
        self.mean_head = SubjectLayers(hidden_dim, output_dim, n_subjects, bias=True)

        # Shared log-variance head (variance structure similar across subjects)
        if use_variational:
            self.log_var_head = nn.Linear(hidden_dim, output_dim)
            # Init small variance: exp(-2) ≈ 0.135 std
            nn.init.normal_(self.log_var_head.weight, std=1e-4)
            nn.init.constant_(self.log_var_head.bias, -2.0)

    def forward(
        self,
        context: torch.Tensor,
        subject_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            context:     (B, T, hidden_dim) fused context from fusion block.
            subject_ids: (B,) long tensor of subject indices.

        Returns:
            x0:      (B, output_dim) sampled source point.
            mu:      (B, output_dim) source mean.
            log_var: (B, output_dim) or None.
        """
        # Mean-pool over temporal dimension
        h = context.mean(dim=1)  # (B, hidden_dim)
        h = self.pool_proj(h)

        if subject_ids is None:
            subject_ids = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)

        mu = self.mean_head(h, subject_ids)  # (B, output_dim)

        if self.use_variational:
            log_var = self.log_var_head(h)
            log_var = torch.clamp(log_var, min=-10.0, max=2.0)

            if self.training:
                std = torch.exp(0.5 * log_var)
                x0 = mu + torch.randn_like(mu) * std
            else:
                x0 = mu
        else:
            log_var = None
            x0 = mu

        return x0, mu, log_var


# =============================================================================
# DirectVelocityNetV2 — Velocity Network with Factored Fusion
# =============================================================================

class DirectVelocityNetV2(nn.Module):
    """Velocity network v2 with Factored Cross-Modal Fusion.

    Architecture:
        1. FactoredCrossModalFusion: per-modality proj → cross-modal attn → (B, T, D)
        2. Input: x_t (B, output_dim) → Linear → (B, hidden_dim)
        3. Time: sinusoidal → MLP → + subject embedding → t_emb
        4. N blocks of CrossAttentionResBlock(h, t_emb, context_encoded)
        5. SubjectLayers output head → (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 1024,
        modality_dims: list[int] = None,
        n_blocks: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        cross_modal_n_layers: int = 2,
        max_seq_len: int = 11,
        n_subjects: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dims = modality_dims or [1408]
        self.n_modalities = len(self.modality_dims)

        # --- Factored Cross-Modal Fusion ---
        self.fusion_block = FactoredCrossModalFusion(
            modality_dims=self.modality_dims,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            n_layers=cross_modal_n_layers,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        # --- Temporal positional embedding for context (after fusion) ---
        self.context_pos_emb = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # --- Time embedding MLP ---
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Subject Embedding ---
        self.subject_emb = nn.Embedding(n_subjects, hidden_dim)

        # --- Velocity blocks ---
        self.blocks = nn.ModuleList([
            CrossAttentionResBlock(hidden_dim, hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # --- Output ---
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.subject_output = SubjectLayers(hidden_dim, output_dim, n_subjects, bias=True)

        # Zero-init output for stable start
        nn.init.constant_(self.subject_output.weights, 0)
        if self.subject_output.bias is not None:
            nn.init.constant_(self.subject_output.bias, 0)

    def encode_context_from_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode concatenated context tensor via fusion block.

        Args:
            cond: (B, T, total_context_dim) pre-concatenated modality features.

        Returns:
            context_encoded: (B, T, hidden_dim) fused context.
        """
        splits = []
        offset = 0
        for dim in self.modality_dims:
            splits.append(cond[:, :, offset:offset + dim])
            offset += dim
        context = self.fusion_block(splits)

        # Add temporal positional embedding
        T = context.shape[1]
        context = context + self.context_pos_emb[:, :T, :]
        return context

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        pre_encoded_context: torch.Tensor = None,
        subject_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x:                    (B, output_dim) noisy fMRI at time t.
            t:                    (B,) or scalar, timestep in [0, 1].
            cond:                 (B, T, total_dim) concatenated context (fallback).
            pre_encoded_context:  (B, T, hidden_dim) already-fused context (skip re-encoding).
            subject_ids:          (B,) long tensor of subject indices.

        Returns:
            v_pred: (B, output_dim) predicted velocity.
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # --- Encode context (skip if pre-encoded) ---
        if pre_encoded_context is not None:
            context_encoded = pre_encoded_context
        elif cond is not None:
            context_encoded = self.encode_context_from_cond(cond)
        else:
            context_encoded = torch.zeros(
                x.shape[0], 1, self.hidden_dim, device=x.device, dtype=x.dtype
            )

        # --- Embeddings ---
        t_emb = self.time_mlp(SinusoidalPosEmb(self.hidden_dim)(t))

        if subject_ids is not None:
            t_emb = t_emb + self.subject_emb(subject_ids)

        # --- Forward through velocity blocks ---
        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h, t_emb, context_encoded)

        # --- Output ---
        h = self.final_norm(h)
        if subject_ids is not None:
            return self.subject_output(h, subject_ids)
        else:
            avg_w = self.subject_output.weights.mean(dim=0)
            out = h @ avg_w
            if self.subject_output.bias is not None:
                out = out + self.subject_output.bias.mean(dim=0)
            return out


# =============================================================================
# BrainFlowDirectV2 — Top-level model with CSFM
# =============================================================================

class BrainFlowDirectV2(nn.Module):
    """Direct fMRI flow matching v2 with CSFM-inspired learned source.

    Key features:
      - Factored Cross-Modal Fusion (efficient multi-modality encoding)
      - Condition-dependent source: x₀ = μ(ctx) + ε·σ(ctx) instead of zeros
      - Losses: flow_loss + align_loss(μ, target) + var_kld_loss(σ)
      - SubjectLayers per-subject output heads
    """

    def __init__(
        self,
        output_dim: int = 1000,
        velocity_net_params: dict = None,
        n_subjects: int = 4,
        source_predictor_params: dict = None,
        source_mode: str = "csfm",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.source_mode = source_mode  # "csfm", "gaussian", "zero"

        # Velocity network
        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg.setdefault("n_subjects", n_subjects)
        self.velocity_net = DirectVelocityNetV2(**vn_cfg)

        # Source predictor (only for CSFM mode)
        self.source_predictor = None
        self.align_weight = 0.0
        self.kld_weight = 0.0
        if source_mode == "csfm":
            sp_cfg = dict(source_predictor_params or {})
            self.align_weight = sp_cfg.pop("align_weight", 1.0)
            self.kld_weight = sp_cfg.pop("kld_weight", 0.1)
            sp_cfg.setdefault("hidden_dim", vn_cfg.get("hidden_dim", 1024))
            sp_cfg.setdefault("output_dim", output_dim)
            sp_cfg.setdefault("n_subjects", n_subjects)
            self.source_predictor = SourcePredictor(**sp_cfg)

        # OT-CFM path
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log parameters
        vn_params = sum(p.numel() for p in self.velocity_net.parameters())
        sp_params = sum(p.numel() for p in self.source_predictor.parameters()) if self.source_predictor else 0
        print(f"[BrainFlowDirectV2] source_mode={source_mode}")
        print(f"[BrainFlowDirectV2] VelocityNet: {vn_params:,} params")
        if self.source_predictor:
            print(f"[BrainFlowDirectV2] SourcePredictor: {sp_params:,} params")
            print(f"[BrainFlowDirectV2] Loss weights: align={self.align_weight}, kld={self.kld_weight}")
        print(f"[BrainFlowDirectV2] Total: {vn_params + sp_params:,} params")

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Compute CSFM losses: flow + align + var-KLD.

        Args:
            context: (B, T, total_dim) concatenated multimodal context.
            target:  (B, output_dim) ground truth fMRI.
            subject_ids: (B,) long tensor of subject indices.

        Returns:
            dict with keys: total_loss, flow_loss, align_loss, kld_loss.
        """
        # 1. Encode context (shared between source predictor and velocity net)
        context_encoded = self.velocity_net.encode_context_from_cond(context)

        # 2. Determine source x₀ based on mode
        x_1 = target
        if self.source_mode == "csfm":
            x_0, mu, log_var = self.source_predictor(context_encoded, subject_ids)
        elif self.source_mode == "gaussian":
            x_0 = torch.randn_like(x_1)
            mu, log_var = None, None
        else:  # "zero"
            x_0 = torch.zeros_like(x_1)
            mu, log_var = None, None

        # 3. Flow matching
        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        v_pred = self.velocity_net(
            x=sample_info.x_t,
            t=sample_info.t,
            pre_encoded_context=context_encoded,
            subject_ids=subject_ids,
        )

        flow_loss = F.mse_loss(v_pred, sample_info.dx_t)

        # 4. Align loss: push source mean close to target → shorter flow
        if mu is not None:
            align_loss = F.mse_loss(mu, target)
        else:
            align_loss = torch.tensor(0.0, device=target.device)

        # 5. Var-KLD loss: regularize variance (only σ terms, no μ²)
        #    KL(q || N(0,1)) variance part = 0.5 * (σ² - log(σ²) - 1)
        if log_var is not None:
            kld_loss = 0.5 * (torch.exp(log_var) - log_var - 1.0).mean()
        else:
            kld_loss = torch.tensor(0.0, device=target.device)

        # 6. Total loss
        total_loss = flow_loss + self.align_weight * align_loss + self.kld_weight * kld_loss

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "align_loss": align_loss,
            "kld_loss": kld_loss,
        }

    @torch.inference_mode()
    def synthesise(
        self,
        context: torch.Tensor,
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
        subject_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE from learned source.

        Uses SourcePredictor to generate x_init = μ(context) instead of zeros.
        Then solves ODE with pre-encoded context (computed once).

        Args:
            context:        (B, T, total_dim) concatenated context.
            n_timesteps:    Number of ODE solver steps.
            solver_method:  ODE solver method.
            subject_ids:    (B,) long tensor of subject indices.

        Returns:
            fmri_pred: (B, output_dim) predicted fMRI.
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        # 1. Encode context once (shared)
        context_encoded = self.velocity_net.encode_context_from_cond(context)

        # 2. Source based on mode
        if self.source_mode == "csfm":
            x_init, _, _ = self.source_predictor(context_encoded, subject_ids)
        else:
            # For gaussian/zero modes, always use zeros at inference
            # (deterministic ODE → optimal PCC)
            x_init = torch.zeros(B, self.output_dim, device=device, dtype=dtype)

        # 3. ODE solve from learned source with pre-encoded context
        solver = ODESolver(velocity_model=self.velocity_net)
        T = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)

        fmri_pred = solver.sample(
            time_grid=T,
            x_init=x_init,
            method=solver_method,
            step_size=1.0 / n_timesteps,
            return_intermediates=False,
            pre_encoded_context=context_encoded,
            subject_ids=subject_ids,
        )

        return fmri_pred
