"""BrainFlow Direct v3 — Flow Matching + Aux Reg + Contrastive in fMRI Space.

Key features:
  - MultiTokenFusion: per-modality proj + modality embeddings; ``fusion_mode=concat``
    (TRIBE-style) concatenates per-modality projections then maps to ``hidden_dim``.
    ``fusion_mode=mean`` preserves the legacy average fusion.
  - Temporal Self-Attention over fused context tokens
  - Optional shared latent + SubjectLayers: velocity/regression end in ``latent_dim``,
    then per-subject linear maps to voxels (``use_subject_head=True``, default).
  - Auxiliary regression (encoder detached) + InfoNCE + CFG at inference

Architecture (default, Proposal A):
  Context → MultiTokenFusion (concat) → (B,T,H) → temporal Transformer
  Velocity trunk → latent_head (B,H)→(B,L) → SubjectLayers → (B,V)
  Reg: pool(detach) → reg_head → reg_output → (B,L) → same SubjectLayers → fmri_pred

Usage:
    python src/train_brainflow.py --config src/configs/brainflow.yaml
"""

import math
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver


# =============================================================================
# InDI (Inversion by Direct Iteration) — training target x_1 - x_t, recover dx/dt
# =============================================================================

def recover_velocity_indi(
    u: torch.Tensor,
    t: torch.Tensor,
    min_denom: float = 1e-3,
) -> torch.Tensor:
    """Map InDI-trained output u ≈ (1-t)(x_1-x_0) to OT velocity v = x_1 - x_0.

    ``t`` may be scalar (ODE solvers) or (B,) matching batch of ``u`` (B, D).
    """
    if t.dim() == 0:
        denom = (1.0 - t).clamp(min=min_denom)
        return u / denom
    denom = (1.0 - t).clamp(min=min_denom)
    while denom.dim() < u.dim():
        denom = denom.unsqueeze(-1)
    return u / denom


class IndiVelocityCallable:
    """Wraps ``VelocityNet`` for ODE integration without duplicating parameters."""

    def __init__(self, net: nn.Module, min_denom: float = 1e-3):
        self.net = net
        self.min_denom = min_denom

    def __call__(self, *, x, t, **kwargs):
        u = self.net(x=x, t=t, **kwargs)
        return recover_velocity_indi(u, t, self.min_denom)


def flow_train_time_sample(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sqrt_bias_end: bool,
) -> torch.Tensor:
    """Sample t in [0, 1]. If ``sqrt_bias_end``, use t = sqrt(U) for more mass near t→1."""
    u = torch.rand(batch_size, device=device, dtype=dtype)
    if sqrt_bias_end:
        return torch.sqrt(u)
    return u


# =============================================================================
# Tensor Flow Matching — per-dimension adaptive schedule
# =============================================================================

def tensor_warp_schedule(
    gamma: torch.Tensor, t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension exponential time-warping schedule.

    lambda_t = (exp(gamma * t) - 1) / (exp(gamma) - 1)
    d_lambda  = gamma * exp(gamma * t) / (exp(gamma) - 1)

    Boundary conditions: lambda(0)=0, lambda(1)=1 for any gamma.
    When |gamma| -> 0 the limit is the standard OT schedule: lambda_t=t, d_lambda=1.

    Args:
        gamma: (B, D) per-dimension speed parameters (clamped internally).
        t:     (B,)   timestep in [0, 1].

    Returns:
        lambda_t:   (B, D) interpolation coefficients.
        d_lambda_t: (B, D) time-derivatives of lambda_t.
    """
    gamma = gamma.clamp(-5.0, 5.0)
    t = t.unsqueeze(-1)                       # (B, 1)

    exp_gt = torch.exp(gamma * t)             # (B, D)
    exp_g = torch.exp(gamma)                  # (B, D)
    denom = exp_g - 1.0                       # (B, D)

    lambda_t = (exp_gt - 1.0) / denom
    d_lambda_t = gamma * exp_gt / denom

    small = gamma.abs() < 1e-4
    lambda_t = torch.where(small, t.expand_as(gamma), lambda_t)
    d_lambda_t = torch.where(small, torch.ones_like(gamma), d_lambda_t)

    return lambda_t, d_lambda_t


class TimeWarpNet(nn.Module):
    """Predicts per-dimension (or per-group) speed parameters gamma from context.

    Zero-initialized output layer so gamma=0 at init, recovering standard OT.
    """

    def __init__(
        self,
        context_dim: int,
        output_dim: int,
        warp_hidden_dim: int = 512,
        n_groups: int | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_groups = n_groups

        out_features = n_groups if n_groups else output_dim
        if n_groups is not None:
            assert output_dim % n_groups == 0, (
                f"output_dim ({output_dim}) must be divisible by n_groups ({n_groups})"
            )
            self.group_size = output_dim // n_groups
        else:
            self.group_size = 1

        self.net = nn.Sequential(
            nn.Linear(context_dim, warp_hidden_dim),
            nn.GELU(),
            nn.Linear(warp_hidden_dim, warp_hidden_dim),
            nn.GELU(),
            nn.Linear(warp_hidden_dim, out_features),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_pooled: (B, context_dim) mean-pooled context.
        Returns:
            gamma: (B, output_dim) per-dimension speed parameters.
        """
        gamma = self.net(context_pooled)          # (B, out_features)
        if self.n_groups is not None:
            gamma = gamma.repeat_interleave(self.group_size, dim=-1)
        return gamma


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


class NetworkSubjectLayers(nn.Module):
    """Per-network SubjectLayers: 7 independent per-subject linear heads.

    Each Yeo functional network gets its own SubjectLayers mapping
    latent_dim -> n_parcels_k for each subject independently.
    Output is concatenated in network order to produce (B, total_output_dim).

    Schaefer 1000Par7Net parcels are ordered: LH networks then RH networks,
    each hemisphere in order: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default.
    """

    SCHAEFER_7NET_PER_HEMI = [75, 74, 66, 68, 38, 61, 118]  # sum = 500
    NETWORK_NAMES = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

    def __init__(self, in_channels: int, n_subjects: int, network_counts: list[int] | None = None):
        super().__init__()
        if network_counts is None:
            # Default: Schaefer 1000 parcels, 7 networks, both hemispheres
            network_counts = [2 * c for c in self.SCHAEFER_7NET_PER_HEMI]
        self.network_counts = network_counts
        self.total_output_dim = sum(network_counts)
        self.n_networks = len(network_counts)

        self.heads = nn.ModuleList([
            SubjectLayers(in_channels, n_k, n_subjects)
            for n_k in network_counts
        ])

        print(f"  [NetworkSubjectLayers] {self.n_networks} heads: "
              f"{list(zip(self.NETWORK_NAMES[:self.n_networks], network_counts))} "
              f"= {self.total_output_dim} total voxels")

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Run all network heads and concatenate outputs.

        Args:
            x: (B, in_channels) shared latent representation.
            subject_ids: (B,) subject index.

        Returns:
            (B, total_output_dim) concatenated per-network predictions.
        """
        parts = [head(x, subject_ids) for head in self.heads]
        return torch.cat(parts, dim=-1)


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Precomputes sin/cos rotation frequencies for RoPE."""

    def __init__(self, dim: int, max_seq_len: int = 128, theta: float = 10000.0):
        super().__init__()
        # dim is head_dim (hidden_dim // n_heads)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute for max_seq_len
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # (T, dim//2)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)  # (T, dim//2)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)  # (T, dim//2)

    def forward(self, seq_len: int):
        """Return (cos, sin) each of shape (seq_len, dim//2)."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to x of shape (B, H, T, D).

    cos, sin: (T, D//2) — broadcast over B, H.
    Splits x into even/odd pairs and applies 2D rotation.
    """
    # x: (B, H, T, D), cos/sin: (T, D//2) -> (1, 1, T, D//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE on Q/K (pre-norm architecture)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer with RoPE attention.

        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape

        # --- Self-attention with RoPE ---
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv_proj(x_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q, K (not V)
        cos, sin = self.rotary_emb(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention (uses FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = residual + attn_out

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        return x


# =============================================================================
# MultiTokenFusion — NSD-style per-modality token preservation
# =============================================================================

class MultiTokenFusion(nn.Module):
    """Per-modality projection + embeddings, then mean or concat (TRIBE-style) fusion.

    ``fusion_mode="mean"``: project each modality to ``hidden_dim``, mean pool,
    ``output_proj`` (legacy).

    ``fusion_mode="concat"``: project each to ``fusion_proj_dim``, add modality_emb
    in that space, concatenate on the feature axis → ``(B,T,M*P)``, then
    ``fusion_to_hidden`` → ``(B,T,hidden_dim)``.
    """

    def __init__(
        self,
        modality_dims: list[int],
        hidden_dim: int = 1024,
        proj_dim: int = 256,
        max_seq_len: int = 11,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        fusion_mode: str = "concat",
        fusion_proj_dim: int = 384,
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.max_seq_len = max_seq_len
        self.modality_dropout = modality_dropout
        self.fusion_mode = fusion_mode
        self.fusion_proj_dim = fusion_proj_dim

        if fusion_mode not in ("mean", "concat"):
            raise ValueError(f"fusion_mode must be 'mean' or 'concat', got {fusion_mode!r}")

        if fusion_mode == "mean":
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for dim in modality_dims
            ])
            self.modality_emb = nn.Parameter(
                torch.randn(self.n_modalities, hidden_dim) * 0.02
            )
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fusion_to_hidden = None
            self.concat_dim = None
        else:
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, fusion_proj_dim),
                    nn.LayerNorm(fusion_proj_dim),
                    nn.GELU(),
                )
                for dim in modality_dims
            ])
            self.modality_emb = nn.Parameter(
                torch.randn(self.n_modalities, fusion_proj_dim) * 0.02
            )
            self.concat_dim = self.n_modalities * fusion_proj_dim
            self.fusion_to_hidden = nn.Sequential(
                nn.Linear(self.concat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.output_proj = None

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: List of M tensors, each (B, T, mod_dim_i).

        Returns:
            context: (B, T, hidden_dim) fused context.
        """
        B, T = modality_features[0].shape[:2]
        T = min(T, self.max_seq_len)

        projected = []
        for i, (feat, proj) in enumerate(zip(modality_features, self.projectors)):
            h = proj(feat[:, :T])
            h = h + self.modality_emb[i]
            projected.append(h)

        if self.training and self.modality_dropout > 0:
            keep_mask = (
                torch.rand(B, 1, self.n_modalities, device=projected[0].device)
                > self.modality_dropout
            )
            all_dropped = (keep_mask.sum(dim=2, keepdim=True) == 0)
            keep_mask[:, :, 0:1] = torch.max(keep_mask[:, :, 0:1], all_dropped)

            for i in range(self.n_modalities):
                projected[i] = projected[i] * keep_mask[:, :, i:i+1]

        if self.fusion_mode == "mean":
            x = torch.stack(projected, dim=0).mean(dim=0)
            return self.output_proj(x)

        x = torch.cat(projected, dim=-1)
        return self.fusion_to_hidden(x)


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
    """Velocity network: temporal context encoder + FiLM/DiT-style blocks.

    With ``use_subject_head=True`` (Proposal A / TRIBE-style): time MLP has no
    subject add; trunk maps to ``latent_dim`` then ``subject_layers`` → voxels.
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
        fusion_mode: str = "concat",
        fusion_proj_dim: int = 384,
        use_subject_head: bool = True,
        latent_dim: int | None = None,
        gradient_checkpointing: bool = False,
        network_head: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dims = modality_dims or [1408]
        self.use_subject_head = use_subject_head
        self.latent_dim = latent_dim if latent_dim is not None else hidden_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.use_rope = use_rope
        self.network_head = network_head and use_subject_head

        self.fusion_block = MultiTokenFusion(
            modality_dims=self.modality_dims,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            modality_dropout=modality_dropout,
            fusion_mode=fusion_mode,
            fusion_proj_dim=fusion_proj_dim,
        )

        # --- Temporal positional encoding ---
        if use_rope:
            head_dim = hidden_dim // n_heads
            self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)
            self.context_pos_emb = None  # no absolute PE when using RoPE
            rope_layers = nn.ModuleList([
                RoPETransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    rotary_emb=self.rotary_emb,
                )
                for _ in range(temporal_attn_layers)
            ])
            self.temporal_attn = rope_layers
        else:
            self.context_pos_emb = nn.Parameter(
                torch.randn(1, max_seq_len, hidden_dim) * 0.02
            )
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

        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        self.time_embed = SinusoidalPosEmb(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_subject_head:
            if self.network_head:
                self.subject_layers = NetworkSubjectLayers(self.latent_dim, n_subjects)
            else:
                self.subject_layers = SubjectLayers(self.latent_dim, output_dim, n_subjects)
            self.subject_emb = None
        else:
            self.subject_layers = None
            self.subject_emb = nn.Embedding(n_subjects, hidden_dim)

        self.blocks = nn.ModuleList([
            SimpleFiLMBlock(hidden_dim, hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        if use_subject_head:
            self.latent_head = nn.Linear(hidden_dim, self.latent_dim)
            nn.init.constant_(self.latent_head.weight, 0)
            nn.init.constant_(self.latent_head.bias, 0)
            self.output_layer = None
        else:
            self.latent_head = None
            self.output_layer = nn.Linear(hidden_dim, output_dim)
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

        # Add absolute positional embedding (only when NOT using RoPE)
        if self.context_pos_emb is not None:
            T = context.shape[1]
            context = context + self.context_pos_emb[:, :T, :]

        # Temporal attention (RoPE layers apply position internally)
        if self.use_rope:
            for layer in self.temporal_attn:
                if self.gradient_checkpointing and self.training:
                    context = checkpoint(layer, context, use_reentrant=False)
                else:
                    context = layer(context)
            context = self.temporal_norm(context)
        else:
            def _temporal_fwd(x):
                return self.temporal_norm(self.temporal_attn(x))
            if self.gradient_checkpointing and self.training:
                context = checkpoint(_temporal_fwd, context, use_reentrant=False)
            else:
                context = _temporal_fwd(context)

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

        t_emb = self.time_mlp(self.time_embed(t))

        if not self.use_subject_head and self.subject_emb is not None and subject_ids is not None:
            t_emb = t_emb + self.subject_emb(subject_ids)

        h = self.input_proj(x)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, t_emb, context_encoded, use_reentrant=False)
            else:
                h = block(h, t_emb, context_encoded)

        h = self.final_norm(h)
        if self.use_subject_head:
            z = self.latent_head(h)
            if subject_ids is None:
                subject_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self.subject_layers(z, subject_ids)
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
        tensor_fm_params: dict = None,
        indi_flow_matching: bool = False,
        indi_train_time_sqrt: bool = False,
        indi_min_denom: float = 1e-3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.reg_weight = reg_weight
        self.cont_weight = cont_weight
        self.indi_flow_matching = indi_flow_matching
        self.indi_train_time_sqrt = indi_train_time_sqrt
        self.indi_min_denom = indi_min_denom

        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg.setdefault("n_subjects", n_subjects)
        self.velocity_net = VelocityNet(**vn_cfg)

        hidden_dim = vn_cfg.get("hidden_dim", 1024)
        use_subject_head = vn_cfg.get("use_subject_head", True)
        latent_dim = vn_cfg.get("latent_dim", hidden_dim)

        # --- Tensor Flow Matching (optional) ---
        self.use_tensor_fm = tensor_fm_params is not None
        if indi_flow_matching and self.use_tensor_fm:
            warnings.warn(
                "indi_flow_matching is ignored when tensor_fm is enabled "
                "(InDI recovery is only defined for linear OT paths).",
                stacklevel=2,
            )
        self._indi_effective = indi_flow_matching and not self.use_tensor_fm

        if self.use_tensor_fm:
            tfm = dict(tensor_fm_params)
            self.gamma_reg_weight = tfm.pop("gamma_reg_weight", 0.01)
            self.time_warp_net = TimeWarpNet(
                context_dim=hidden_dim,
                output_dim=output_dim,
                **tfm,
            )
        else:
            self.gamma_reg_weight = 0.0

        reg_hidden = hidden_dim * 2
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
        reg_out_dim = latent_dim if use_subject_head else output_dim
        self.reg_output = nn.Linear(reg_hidden, reg_out_dim)

        # --- Contrastive Projection Heads ---
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

        # OT-CFM path (fallback when TensorFM disabled)
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log parameters
        vn_params = sum(p.numel() for p in self.velocity_net.parameters())
        reg_params = sum(p.numel() for p in self.reg_head.parameters()) + \
                     sum(p.numel() for p in self.reg_output.parameters())
        cont_params = sum(p.numel() for p in self.contrastive_proj.parameters()) + \
                      sum(p.numel() for p in self.target_proj.parameters())
        warp_params = sum(p.numel() for p in self.time_warp_net.parameters()) if self.use_tensor_fm else 0
        total = vn_params + reg_params + cont_params + warp_params
        print(f"[BrainFlow] VelocityNet: {vn_params:,} params")
        print(f"[BrainFlow] RegressionHead: {reg_params:,} params")
        print(f"[BrainFlow] ContrastiveHeads: {cont_params:,} params")
        if self.use_tensor_fm:
            print(f"[BrainFlow] TimeWarpNet: {warp_params:,} params (gamma_reg={self.gamma_reg_weight})")
        print(f"[BrainFlow] Total: {total:,} params "
              f"(reg_w={reg_weight}, cont_w={cont_weight}, tensor_fm={self.use_tensor_fm}, "
              f"indi={self._indi_effective}, indi_t_sqrt={indi_train_time_sqrt})")

    def compute_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
        starting_distribution: torch.Tensor = None,
        skip_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute flow + regression + contrastive loss.

        NSD-style gradient isolation:
          - Flow branch: gradient flows through encoder (updates fusion + DiT)
          - Reg branch: .detach() prevents conflict (updates reg head + proj only)

        Args:
            context: (B, T, total_dim) concatenated multimodal context.
            target:  (B, output_dim) ground truth fMRI.
            subject_ids: (B,) long tensor of subject indices.
            skip_aux: if True, skip regression and contrastive losses (used when
                      context is zeroed out for CFG unconditional training).

        Returns:
            dict with keys: total_loss, flow_loss, align_loss, cont_loss, gamma_reg.
        """
        # 1. Encode context once (shared)
        context_encoded = self.velocity_net.encode_context_from_cond(context)

        # 2. Regression branch with gradient isolation (NSD improvement)
        # .detach() prevents regression from pulling the shared encoder
        _zero = torch.tensor(0.0, device=target.device)
        if skip_aux:
            reg_loss = _zero
            cont_loss = _zero
        else:
            ctx_detached = context_encoded.detach()  # ⛔ no gradient to fusion
            ctx_pooled_reg = ctx_detached.mean(dim=1)  # (B, hidden)
            reg_hidden = self.reg_head(ctx_pooled_reg)
            latent_reg = self.reg_output(reg_hidden)
            if self.velocity_net.use_subject_head:
                if subject_ids is None:
                    subject_ids = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
                fmri_pred = self.velocity_net.subject_layers(latent_reg, subject_ids)
            else:
                fmri_pred = latent_reg

            reg_loss = F.mse_loss(fmri_pred, target)

            # 3. Contrastive loss in 256-D projected space (NSD improvement)
            # .detach() on fmri_pred prevents contrastive gradient from leaking
            # through shared SubjectLayers into the velocity flow branch.
            z_pred = F.normalize(self.contrastive_proj(fmri_pred.detach()), dim=-1)
            z_target = F.normalize(self.target_proj(target), dim=-1)
            cont_loss = info_nce_loss(z_pred, z_target)

        # 4. Flow matching source distribution (x_0)
        x_1 = target
        if starting_distribution is not None:
            x_0 = starting_distribution
        else:
            x_0 = torch.randn_like(x_1)

        # 5. Flow matching (gradient flows to encoder)
        gamma_reg = _zero

        if self.use_tensor_fm:
            t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
            # Use NON-detached pooling so TimeWarpNet gradients flow through encoder
            ctx_pooled_flow = context_encoded.mean(dim=1)  # ✅ gradient flows
            gamma = self.time_warp_net(ctx_pooled_flow)     # (B, D)
            lambda_t, d_lambda_t = tensor_warp_schedule(gamma, t)

            x_t = lambda_t * x_1 + (1.0 - lambda_t) * x_0
            target_velocity = d_lambda_t * (x_1 - x_0)

            v_pred = self.velocity_net(
                x=x_t, t=t,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )
            flow_loss = F.mse_loss(v_pred, target_velocity)

            gamma_reg = gamma.pow(2).mean()
        else:
            t = flow_train_time_sample(
                x_1.shape[0],
                x_1.device,
                x_1.dtype,
                sqrt_bias_end=self.indi_train_time_sqrt and self._indi_effective,
            )
            sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

            v_pred = self.velocity_net(
                x=sample_info.x_t,
                t=sample_info.t,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )
            if self._indi_effective:
                target_velocity = x_1 - sample_info.x_t
            else:
                target_velocity = sample_info.dx_t
            flow_loss = F.mse_loss(v_pred, target_velocity)

        # 6. Total loss
        total_loss = (flow_loss
                      + self.reg_weight * reg_loss
                      + self.cont_weight * cont_loss
                      + self.gamma_reg_weight * gamma_reg)

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "align_loss": reg_loss,
            "cont_loss": cont_loss,
            "gamma_reg": gamma_reg,
        }

    def _build_time_grid(
        self,
        n_timesteps: int,
        device: torch.device,
        dtype: torch.dtype,
        time_grid_warp: str | None,
    ) -> torch.Tensor:
        """Monotone grid in [0, 1]. ``sqrt`` uses T = sqrt(s) for finer steps near t→1."""
        if n_timesteps < 2:
            n_timesteps = 2
        s = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)
        if time_grid_warp is None or time_grid_warp in ("", "none", "linear"):
            return s
        if time_grid_warp == "sqrt":
            return torch.sqrt(s)
        raise ValueError(
            f"time_grid_warp must be None, 'linear', or 'sqrt', got {time_grid_warp!r}"
        )

    def _velocity_for_ode(self):
        """OT-CFM velocity for integration; InDI-trained nets need u/(1-t) recovery."""
        if self._indi_effective:
            return IndiVelocityCallable(self.velocity_net, self.indi_min_denom)
        return self.velocity_net

    @torch.inference_mode()
    def synthesise(
        self,
        context: torch.Tensor,
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
        subject_ids: torch.Tensor = None,
        cfg_scale: float = 0.0,
        starting_distribution: torch.Tensor = None,
        temperature: float = 0.0,
        time_grid_warp: str | None = None,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE, optionally with CFG.

        Classifier-Free Guidance (when cfg_scale > 0):
          v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

        Args:
            context:                (B, T, total_dim) concatenated context.
            n_timesteps:            Number of ODE solver steps.
            solver_method:          ODE solver method.
            subject_ids:            (B,) long tensor of subject indices.
            cfg_scale:              CFG guidance scale (0 = no guidance).
            starting_distribution:  (B, output_dim) custom x_0 for Residual FM.
                                    If None, uses low-temperature Gaussian.
            temperature:            Std-dev of starting noise (0 = deterministic zeros,
                                    1 = full Gaussian, 0.1-0.5 = low-temperature).
            time_grid_warp:         ``None``/``linear`` = uniform; ``sqrt`` = ``sqrt(linspace)``
                                    for smaller Δt near t→1 (pairs with InDI inference).

        Returns:
            fmri_pred: (B, output_dim) predicted fMRI.
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        context_encoded = self.velocity_net.encode_context_from_cond(context)

        if starting_distribution is not None:
            x_init = starting_distribution.to(device=device, dtype=dtype)
        elif temperature > 0:
            x_init = temperature * torch.randn(B, self.output_dim, device=device, dtype=dtype)
        else:
            x_init = torch.zeros(B, self.output_dim, device=device, dtype=dtype)

        T_grid = self._build_time_grid(n_timesteps, device, dtype, time_grid_warp)
        n_steps = T_grid.shape[0]
        deltas = T_grid[1:] - T_grid[:-1]
        ode_step_size = float(deltas.min().clamp_min(torch.tensor(1e-6, device=device, dtype=dtype)))

        if cfg_scale > 0:
            uncond_context = torch.zeros_like(context)
            uncond_encoded = self.velocity_net.encode_context_from_cond(uncond_context)

            x = x_init
            for i in range(n_steps - 1):
                t_val = T_grid[i]
                dt_i = T_grid[i + 1] - T_grid[i]
                t_batch = t_val.expand(B)

                u_cond = self.velocity_net(
                    x=x, t=t_batch,
                    pre_encoded_context=context_encoded,
                    subject_ids=subject_ids,
                )
                u_uncond = self.velocity_net(
                    x=x, t=t_batch,
                    pre_encoded_context=uncond_encoded,
                    subject_ids=subject_ids,
                )
                u_guided = u_uncond + cfg_scale * (u_cond - u_uncond)
                if self._indi_effective:
                    v_guided = recover_velocity_indi(u_guided, t_val, self.indi_min_denom)
                else:
                    v_guided = u_guided
                x = x + dt_i * v_guided

            return x

        solver = ODESolver(velocity_model=self._velocity_for_ode())
        fmri_pred = solver.sample(
            time_grid=T_grid,
            x_init=x_init,
            method=solver_method,
            step_size=ode_step_size,
            return_intermediates=False,
            pre_encoded_context=context_encoded,
            subject_ids=subject_ids,
        )

        return fmri_pred
