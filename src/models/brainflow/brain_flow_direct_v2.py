"""BrainFlow Direct v2 — TRIBE-Inspired Flow Matching in fMRI Space.

Improvements over v1:
  - Per-modality MLP projectors (supports N modalities of different dims)
  - Modality dropout (0.3) for robustness
  - Temporal Transformer Encoder to refine context before cross-attention
  - SubjectLayers output head (per-subject linear projection)
  - NO Global Flattened Context MLP (removed to prevent overfitting)

Architecture:
  Context (B, T, [mod_dims...])
    → Per-modality Projectors + Modality Dropout
    → Concatenate → (B, T, hidden_dim)
    → Temporal Transformer Encoder (4 layers)
    → context_encoded (B, T, hidden_dim)

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
    """Per-subject linear output head (from TRIBE/Brain-Diffuser).

    Each subject gets its own weight matrix for the final projection,
    allowing subject-specific adaptation while sharing the backbone.
    """

    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels)) if bias else None

        # Xavier-like init
        self.weights.data.normal_(0, 1.0 / in_channels ** 0.5)
        if self.bias is not None:
            self.bias.data.normal_(0, 1.0 / in_channels ** 0.5)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D_in) hidden state
            subject_ids: (B,) long tensor of subject indices
        Returns:
            (B, D_out) per-subject projected output
        """
        # Gather per-subject weights
        w = self.weights[subject_ids]  # (B, D_in, D_out)
        out = torch.einsum("bd,bdo->bo", x, w)
        if self.bias is not None:
            b = self.bias[subject_ids]  # (B, D_out)
            out = out + b
        return out


# =============================================================================
# CrossAttentionResBlock — FiLM (time) + FFN + CrossAttn (context)
# =============================================================================

class CrossAttentionResBlock(nn.Module):
    """Residual block with FiLM (time) + cross-attention (processed context).

    Architecture per block:
        h = FiLM(LayerNorm(x), t_emb)     — time-conditioned modulation
        h = x + FFN(h)                     — residual MLP
        h = h + CrossAttn(h, context)      — attend to transformer-processed context
    """

    def __init__(self, dim: int, time_dim: int, context_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # FiLM from timestep
        self.film = nn.Linear(time_dim, dim * 2)

        # FFN with pre-norm
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Cross-attention to processed context (same dim now!)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, D) hidden state
            t_emb:   (B, D_t) timestep embedding
            context: (B, T, C) transformer-processed context
        """
        # 1. FiLM modulated FFN
        scale_shift = self.film(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.ffn(h)

        # 2. Cross-attention to context
        q = self.norm_q(x).unsqueeze(1)           # (B, 1, D)
        kv = self.norm_kv(context)                 # (B, T, C)
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, D)
        x = x + attn_out.squeeze(1)               # (B, D)

        return x


# =============================================================================
# TokenizedFusionBlock — All-to-All Modality-Time Fusion
# =============================================================================

class TokenizedFusionBlock(nn.Module):
    """Tokenized All-to-All Modality-Time Fusion Block.
    
    Treats each (modality, time) pair as a distinct token.
    1. Projects all modalities to a large common context_dim (e.g., 2048).
    2. Adds separate temporal and modality positional embeddings.
    3. Applies Modality Dropout.
    4. Runs a Full Attention Transformer over the flattened (T * M) sequence.
    """
    def __init__(
        self, 
        modality_dims: list[int], 
        context_dim: int = 2048, 
        max_seq_len: int = 11, 
        n_heads: int = 8, 
        depth: int = 4, 
        dropout: float = 0.1,
        modality_dropout: float = 0.3
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.n_modalities = len(modality_dims)
        self.context_dim = context_dim
        self.max_seq_len = max_seq_len
        self.modality_dropout = modality_dropout

        # Per-modality projectors to common context_dim
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU()
            ) for dim in modality_dims
        ])

        # Learnable Embeddings coordinates
        self.time_emb = nn.Parameter(torch.randn(1, max_seq_len, 1, context_dim) * 0.02)
        self.modality_emb = nn.Parameter(torch.randn(1, 1, self.n_modalities, context_dim) * 0.02)

        # Transformer Encoder for all-to-all cross-modal and cross-time fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=n_heads,
            dim_feedforward=context_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        B, T = modality_features[0].shape[:2]
        
        # 1. Project all to common D
        projected = []
        for feat, proj in zip(modality_features, self.projectors):
            projected.append(proj(feat))  # (B, T, context_dim)
            
        # 2. Stack building the modality axis -> (B, T, M, context_dim)
        stacked = torch.stack(projected, dim=2)
        
        # 3. Add Positional Coordinates
        T_safe = min(T, self.max_seq_len)
        t_emb = self.time_emb[:, :T_safe, :, :]
        if T > self.max_seq_len:
            stacked = stacked[:, :self.max_seq_len, :, :]
        
        stacked = stacked + t_emb + self.modality_emb
        
        # 4. Modality Dropout
        if self.training and self.modality_dropout > 0:
            keep_mask = (torch.rand(B, 1, self.n_modalities, 1, device=stacked.device) > self.modality_dropout).float()
            # Ensure at least one modality survives per sample
            all_dropped = (keep_mask.sum(dim=2, keepdim=True) == 0).float()
            keep_mask[:, :, 0:1, :] = torch.max(keep_mask[:, :, 0:1, :], all_dropped)
            stacked = stacked * keep_mask

        # 5. Flatten to sequence of length T_safe * M tokens
        flattened = stacked.view(B, T_safe * self.n_modalities, self.context_dim)
        
        # 6. Deep Transformer Fusion
        fused = self.transformer(flattened)
        return fused


# =============================================================================
# DirectVelocityNetV2 — TRIBE-Inspired VelocityNet
# =============================================================================

class DirectVelocityNetV2(nn.Module):
    """Velocity network v2 with Tokenized Modality-Time Fusion.

    Architecture:
        1. TokenizedFusionBlock: Project all mods to context_dim, add coords, Full-Attn.
        2. Input: x_t (B, output_dim) → Linear → (B, hidden_dim)
        3. Time: sinusoidal → MLP → + subject embedding → t_emb
        4. N blocks of CrossAttentionResBlock(h, t_emb, context_encoded)
        5. SubjectLayers output head → (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 1024,
        context_dim: int = 2048,
        modality_dims: list[int] = None,
        n_blocks: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        context_transformer_depth: int = 4,
        max_seq_len: int = 11,
        n_subjects: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.modality_dims = modality_dims or [1408]
        self.n_modalities = len(self.modality_dims)

        # --- Tokenized Fusion Block ---
        self.fusion_block = TokenizedFusionBlock(
            modality_dims=self.modality_dims,
            context_dim=context_dim,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            depth=context_transformer_depth,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

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

        # --- Subject Embedding (injected into t_emb) ---
        self.subject_emb = nn.Embedding(n_subjects, hidden_dim)

        # --- Velocity blocks: FiLM + CrossAttn ---
        self.blocks = nn.ModuleList([
            CrossAttentionResBlock(hidden_dim, hidden_dim, self.context_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # --- Output: SubjectLayers (per-subject linear head) ---
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.subject_output = SubjectLayers(hidden_dim, output_dim, n_subjects, bias=True)

        # Zero-init the subject output for stable start
        nn.init.constant_(self.subject_output.weights, 0)
        if self.subject_output.bias is not None:
            nn.init.constant_(self.subject_output.bias, 0)

    def _encode_context(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """Tokenize and fuse modality features.

        Args:
            modality_features: list of (B, T, mod_dim_i) tensors

        Returns:
            context_encoded: (B, T * M, context_dim) fused context sequence
        """
        return self.fusion_block(modality_features)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        modality_features: list[torch.Tensor] = None,
        subject_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x:                  (B, output_dim) noisy fMRI at time t.
            t:                  (B,) or scalar, timestep in [0, 1].
            cond:               (B, T, total_context_dim) pre-concatenated context (fallback).
            modality_features:  list of (B, T, mod_dim_i) per-modality tensors.
            subject_ids:        (B,) long tensor of subject indices.

        Returns:
            v_pred: (B, output_dim) predicted velocity.
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # --- Encode context ---
        if modality_features is not None:
            context_encoded = self._encode_context(modality_features)
        elif cond is not None:
            # Fallback: split concatenated context into modalities
            splits = []
            offset = 0
            for dim in self.modality_dims:
                splits.append(cond[:, :, offset:offset + dim])
                offset += dim
            context_encoded = self._encode_context(splits)
        else:
            context_encoded = torch.zeros(
                x.shape[0], 1, self.context_dim, device=x.device, dtype=x.dtype
            )

        # --- Embeddings ---
        t_emb = self.time_mlp(SinusoidalPosEmb(self.hidden_dim)(t))  # (B, hidden_dim)

        if subject_ids is not None:
            t_emb = t_emb + self.subject_emb(subject_ids)

        # --- Forward through velocity blocks ---
        h = self.input_proj(x)  # (B, hidden_dim)

        for block in self.blocks:
            h = block(h, t_emb, context_encoded)

        # --- Output with SubjectLayers ---
        h = self.final_norm(h)
        if subject_ids is not None:
            return self.subject_output(h, subject_ids)  # (B, output_dim)
        else:
            # Average across subjects for unconditional
            avg_w = self.subject_output.weights.mean(dim=0)  # (D_in, D_out)
            out = h @ avg_w
            if self.subject_output.bias is not None:
                out = out + self.subject_output.bias.mean(dim=0)
            return out


# =============================================================================
# BrainFlowDirectV2 — Top-level model
# =============================================================================

class BrainFlowDirectV2(nn.Module):
    """Direct fMRI flow matching v2 with TRIBE-inspired conditioning.

    Key improvements over v1:
      - Per-modality projectors with modality dropout
      - Temporal transformer encoder for context
      - SubjectLayers output head
      - No Global Flattened Context MLP
    """

    def __init__(
        self,
        output_dim: int = 1000,
        velocity_net_params: dict = None,
        n_subjects: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Velocity network
        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg.setdefault("n_subjects", n_subjects)
        self.velocity_net = DirectVelocityNetV2(**vn_cfg)

        # OT-CFM path
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log parameters
        n_params = sum(p.numel() for p in self.velocity_net.parameters())
        print(f"[BrainFlowDirectV2] VelocityNet: {n_params:,} params")

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
        modality_features: list[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute OT-CFM loss in fMRI space.

        Args:
            context:            (B, T, total_dim) concatenated context (fallback).
            target:             (B, output_dim) ground truth fMRI.
            subject_ids:        (B,) long tensor of subject indices.
            modality_features:  list of (B, T, mod_dim_i) per-modality tensors.

        Returns:
            loss: scalar MSE(v_pred, dx_t).
        """
        x_1 = target
        x_0 = torch.zeros_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        v_pred = self.velocity_net(
            x=sample_info.x_t,
            t=sample_info.t,
            cond=context,
            modality_features=modality_features,
            subject_ids=subject_ids,
        )

        return F.mse_loss(v_pred, sample_info.dx_t)

    @torch.inference_mode()
    def synthesise(
        self,
        context: torch.Tensor,
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
        subject_ids: torch.Tensor = None,
        modality_features: list[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE with context conditioning.

        Args:
            context:            (B, T, total_dim) concatenated context (fallback).
            n_timesteps:        Number of ODE solver steps.
            solver_method:      ODE solver method.
            subject_ids:        (B,) long tensor of subject indices.
            modality_features:  list of (B, T, mod_dim_i) per-modality tensors.

        Returns:
            fmri_pred: (B, output_dim) predicted fMRI.
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        x_init = torch.zeros(B, self.output_dim, device=device, dtype=dtype)

        solver = ODESolver(velocity_model=self.velocity_net)
        T = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)

        fmri_pred = solver.sample(
            time_grid=T,
            x_init=x_init,
            method=solver_method,
            step_size=1.0 / n_timesteps,
            return_intermediates=False,
            cond=context,
            modality_features=modality_features,
            subject_ids=subject_ids,
        )

        return fmri_pred
