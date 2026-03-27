"""BrainFlow Direct v3 — Flow Matching + Aux Reg + Contrastive in fMRI Space.

Key features (NSD-inspired improvements):
  - MultiTokenFusion: per-modality proj + modality embeddings → preserve token identity
  - Temporal Self-Attention: model temporal dependencies in context
  - Auxiliary Regression Head: direct fMRI supervision (DETACHED from encoder)
  - Contrastive Branch: InfoNCE loss in 256-D projected space
  - Classifier-Free Guidance (CFG) at inference
  - 4 SimpleFiLMBlocks for velocity estimation

Architecture:
  Context (B, T, [mod_dims...])
    → MultiTokenFusion: per-mod proj + modality_emb → concat → (B, T, hidden)
    → Temporal Self-Attention (2 layers) → refined context

  Training:
    context_encoded (shared) → flow branch (gradient flows to encoder)
    context_encoded.detach() → pool → RegressionHead → fmri_reg
    fmri_reg → contrastive projections → InfoNCE loss
    total_loss = flow_loss + reg_weight * reg_loss + cont_weight * cont_loss

  Inference:
    x_init = 0 → ODE solve with CFG guidance → fmri_pred

Usage:
    python src/train_brainflow_direct.py --config src/configs/brainflow_nsd.yaml
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
# MultiTokenFusion — NSD-style per-modality token preservation
# =============================================================================

class MultiTokenFusion(nn.Module):
    """NSD-style fusion: per-modality projection + modality embeddings.

    Unlike LinearFusion (concat on feature dim → destroys modality identity),
    this module projects each modality to hidden_dim and adds a learnable
    modality embedding so downstream attention can distinguish token origins.

    Output: (B, T, hidden_dim) — modalities are concatenated along feature dim
    AFTER individual projection, preserving modality-aware information via embeddings.
    """

    def __init__(
        self,
        modality_dims: list[int],
        hidden_dim: int = 1024,
        proj_dim: int = 256,
        max_seq_len: int = 11,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.max_seq_len = max_seq_len
        self.modality_dropout = modality_dropout

        # Per-modality projection: mod_dim → hidden_dim (direct, like NSD)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for dim in modality_dims
        ])

        # Learnable modality embeddings (NSD improvement)
        # Each modality gets a unique embedding so attention can distinguish origins
        self.modality_emb = nn.Parameter(
            torch.randn(self.n_modalities, hidden_dim) * 0.02
        )

        # Output projection: hidden_dim → hidden_dim (with dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: List of M tensors, each (B, T, mod_dim_i).

        Returns:
            context: (B, T, hidden_dim) fused context.
        """
        B, T = modality_features[0].shape[:2]
        T = min(T, self.max_seq_len)

        # Project each modality and add modality embedding
        projected = []
        for i, (feat, proj) in enumerate(zip(modality_features, self.projectors)):
            h = proj(feat[:, :T])  # (B, T, hidden_dim)
            h = h + self.modality_emb[i]  # broadcast (hidden_dim,) → (B, T, hidden_dim)
            projected.append(h)

        # Modality dropout during training
        if self.training and self.modality_dropout > 0:
            keep_mask = (
                torch.rand(B, 1, self.n_modalities, device=projected[0].device)
                > self.modality_dropout
            )
            # Ensure at least one modality is kept
            all_dropped = (keep_mask.sum(dim=2, keepdim=True) == 0)
            keep_mask[:, :, 0:1] = torch.max(keep_mask[:, :, 0:1], all_dropped)

            for i in range(self.n_modalities):
                projected[i] = projected[i] * keep_mask[:, :, i:i+1]

        # Average across modalities (preserves temporal structure)
        # Each projected[i] is (B, T, hidden_dim) with modality_emb baked in
        x = torch.stack(projected, dim=0).mean(dim=0)  # (B, T, hidden_dim)

        # Output projection
        context = self.output_proj(x)  # (B, T, hidden_dim)
        return context


# =============================================================================
# SimpleFiLMBlock — FiLM (time) + FFN + CrossAttn (no Time-Adaptive KV)
# =============================================================================

class SimpleFiLMBlock(nn.Module):
    """Simplified residual block: FiLM + FFN + cross-attention."""

    def __init__(self, dim: int, time_dim: int, context_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # FiLM: time → scale/shift for hidden state
        self.film = nn.Linear(time_dim, dim * 2)

        # FFN
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Cross-attention (Q=hidden, KV=context)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # FiLM conditioning
        scale_shift = self.film(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.ffn(h)

        # Cross-attention (standard, no time-adaptive KV)
        q = self.norm_q(x).unsqueeze(1)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = x + attn_out.squeeze(1)

        return x


# =============================================================================
# VelocityNet — Velocity Network with Temporal Self-Attention
# =============================================================================

class VelocityNet(nn.Module):
    """Velocity network v3 with temporal self-attention context refinement.

    Architecture:
      1. LinearFusion: per-modality proj → concat → (B, T, hidden)
      2. Temporal Self-Attention: 2-layer TransformerEncoder for temporal reasoning
      3. Input: x_t (B, V) → Linear → (B, hidden)
      4. Time: sinusoidal → MLP → + subject embedding → t_emb
      5. N blocks of SimpleFiLMBlock(h, t_emb, context)
      6. SubjectLayers output head → (B, V)
    """

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 1024,
        modality_dims: list[int] = None,
        proj_dim: int = 256,
        n_blocks: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        max_seq_len: int = 31,
        n_subjects: int = 4,
        temporal_attn_layers: int = 2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dims = modality_dims or [1408]

        # --- MultiTokenFusion (NSD-style) ---
        self.fusion_block = MultiTokenFusion(
            modality_dims=self.modality_dims,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

        # --- Temporal positional embedding ---
        self.context_pos_emb = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # --- Temporal Self-Attention (#5: model temporal dependencies) ---
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=temporal_attn_layers,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
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
            SimpleFiLMBlock(hidden_dim, hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # --- Output ---
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Zero-init output for stable start
        nn.init.constant_(self.output_layer.weight, 0)
        nn.init.constant_(self.output_layer.bias, 0)

    def encode_context_from_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode concatenated context tensor via linear fusion + temporal attention.

        Args:
            cond: (B, T, total_context_dim) pre-concatenated modality features.

        Returns:
            context_encoded: (B, T, hidden_dim) fused + temporally refined context.
        """
        # Split into per-modality tensors
        splits = []
        offset = 0
        for dim in self.modality_dims:
            splits.append(cond[:, :, offset:offset + dim])
            offset += dim

        # Linear fusion: (B, T, hidden_dim)
        context = self.fusion_block(splits)

        # Add temporal positional embedding
        T = context.shape[1]
        context = context + self.context_pos_emb[:, :T, :]

        # Temporal self-attention: model sequential patterns
        context = self.temporal_attn(context)
        context = self.temporal_norm(context)

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
            cond:                 (B, T, total_dim) concatenated context.
            pre_encoded_context:  (B, T, hidden_dim) already-fused context.
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
        return self.output_layer(h)


# =============================================================================
# BrainFlow — Top-level Model with Auxiliary Regression + CFG
# =============================================================================

def info_nce_loss(z_pred, z_target, temperature=0.07):
    """Bidirectional InfoNCE loss in projected space.

    Args:
        z_pred:   (B, D) L2-normalized prediction embeddings.
        z_target: (B, D) L2-normalized target embeddings.
        temperature: softmax temperature.

    Returns:
        loss: scalar, average of pred→target and target→pred NCE.
    """
    logits = z_pred @ z_target.T / temperature  # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.T, labels)
    return (loss_p2t + loss_t2p) / 2


class BrainFlow(nn.Module):
    """Flow matching v3 with auxiliary regression, contrastive, and CFG.

    NSD-inspired improvements:
      - MultiTokenFusion: modality embeddings preserve token identity
      - Gradient isolation: .detach() prevents reg from conflicting with flow
      - Contrastive branch: InfoNCE in 256-D for ranking signal
      - Deeper regression head: 4-layer MLP
      - CFG at inference (trained with 10% context dropout)
    """

    def __init__(
        self,
        output_dim: int = 1000,
        velocity_net_params: dict = None,
        n_subjects: int = 4,
        reg_weight: float = 1.0,
        cont_weight: float = 0.1,
        cont_dim: int = 256,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.reg_weight = reg_weight
        self.cont_weight = cont_weight

        # Velocity network
        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg.setdefault("n_subjects", n_subjects)
        self.velocity_net = VelocityNet(**vn_cfg)

        # --- Deepened Regression Head (NSD: 4-layer instead of 2) ---
        hidden_dim = vn_cfg.get("hidden_dim", 1024)
        reg_hidden = hidden_dim * 2  # 1024 → 2048 intermediate
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, reg_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(reg_hidden, reg_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(reg_hidden, reg_hidden),
            nn.GELU(),
        )
        self.reg_output = nn.Linear(reg_hidden, output_dim)

        # --- Contrastive Projection Heads (NSD improvement) ---
        self.contrastive_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cont_dim),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cont_dim),
        )

        # OT-CFM path
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log parameters
        vn_params = sum(p.numel() for p in self.velocity_net.parameters())
        reg_params = sum(p.numel() for p in self.reg_head.parameters()) + \
                     sum(p.numel() for p in self.reg_output.parameters())
        cont_params = sum(p.numel() for p in self.contrastive_proj.parameters()) + \
                      sum(p.numel() for p in self.target_proj.parameters())
        total = vn_params + reg_params + cont_params
        print(f"[BrainFlow] VelocityNet: {vn_params:,} params")
        print(f"[BrainFlow] RegressionHead: {reg_params:,} params")
        print(f"[BrainFlow] ContrastiveHeads: {cont_params:,} params")
        print(f"[BrainFlow] Total: {total:,} params "
              f"(reg_w={reg_weight}, cont_w={cont_weight})")

    def compute_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
        starting_distribution: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Compute flow + regression + contrastive loss.

        NSD-style gradient isolation:
          - Flow branch: gradient flows through encoder (updates fusion + DiT)
          - Reg branch: .detach() prevents conflict (updates reg head + proj only)

        Args:
            context: (B, T, total_dim) concatenated multimodal context.
            target:  (B, output_dim) ground truth fMRI.
            subject_ids: (B,) long tensor of subject indices.

        Returns:
            dict with keys: total_loss, flow_loss, align_loss, cont_loss, kld_loss.
        """
        # 1. Encode context once (shared)
        context_encoded = self.velocity_net.encode_context_from_cond(context)

        # 2. Regression branch with gradient isolation (NSD improvement)
        # .detach() prevents regression from pulling the shared encoder
        ctx_detached = context_encoded.detach()  # ⛔ no gradient to fusion
        ctx_pooled = ctx_detached.mean(dim=1)  # (B, hidden)
        reg_hidden = self.reg_head(ctx_pooled)
        fmri_pred = self.reg_output(reg_hidden)

        reg_loss = F.mse_loss(fmri_pred, target)

        # 3. Contrastive loss in 256-D projected space (NSD improvement)
        z_pred = F.normalize(self.contrastive_proj(fmri_pred), dim=-1)
        z_target = F.normalize(self.target_proj(target), dim=-1)
        cont_loss = info_nce_loss(z_pred, z_target)

        # 4. Flow matching source distribution (x_0)
        x_1 = target
        if starting_distribution is not None:
            x_0 = starting_distribution
        else:
            x_0 = torch.randn_like(x_1)

        # 5. Flow matching (gradient flows to encoder)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        v_pred = self.velocity_net(
            x=sample_info.x_t,
            t=sample_info.t,
            pre_encoded_context=context_encoded,  # ✅ gradient flows
            subject_ids=subject_ids,
        )

        flow_loss = F.mse_loss(v_pred, sample_info.dx_t)

        # 6. Total loss
        total_loss = flow_loss + self.reg_weight * reg_loss + self.cont_weight * cont_loss

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "align_loss": reg_loss,    # compat key
            "cont_loss": cont_loss,
            "kld_loss": torch.tensor(0.0, device=target.device),
        }

    @torch.inference_mode()
    def synthesise(
        self,
        context: torch.Tensor,
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
        subject_ids: torch.Tensor = None,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE from zeros, optionally with CFG.

        Classifier-Free Guidance (when cfg_scale > 0):
          v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

        Args:
            context:        (B, T, total_dim) concatenated context.
            n_timesteps:    Number of ODE solver steps.
            solver_method:  ODE solver method.
            subject_ids:    (B,) long tensor of subject indices.
            cfg_scale:      CFG guidance scale (0 = no guidance).

        Returns:
            fmri_pred: (B, output_dim) predicted fMRI.
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        # 1. Encode context once
        context_encoded = self.velocity_net.encode_context_from_cond(context)

        # 2. Start from zeros (deterministic → optimal PCC)
        x_init = torch.zeros(B, self.output_dim, device=device, dtype=dtype)

        if cfg_scale > 0:
            # CFG: also compute unconditional context (zeros)
            uncond_context = torch.zeros_like(context)
            uncond_encoded = self.velocity_net.encode_context_from_cond(uncond_context)

            # Manual ODE solve with CFG
            T_grid = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)
            dt = 1.0 / (n_timesteps - 1)
            x = x_init

            for i in range(n_timesteps - 1):
                t_val = T_grid[i]
                t_batch = t_val.expand(B)

                v_cond = self.velocity_net(
                    x=x, t=t_batch,
                    pre_encoded_context=context_encoded,
                    subject_ids=subject_ids,
                )
                v_uncond = self.velocity_net(
                    x=x, t=t_batch,
                    pre_encoded_context=uncond_encoded,
                    subject_ids=subject_ids,
                )
                v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
                x = x + dt * v_guided

            return x
        else:
            # Standard ODE solve (no CFG)
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
