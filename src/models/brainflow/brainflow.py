"""BrainFlow — Seq2Seq Flow Matching with multitoken context fusion.

Seq2seq (``n_target_trs`` > 1, default):
    - MultiTokenFusion over modality-preserved context streams
    - RoPE temporal encoder over full context TRs (no causal mask)
    - Temporal slice to ``n_target_trs`` tokens aligned with fMRI targets
    - Decoder: DiT-X (cross-attention + AdaLN-Zero) or DiT-1D (additive + AdaLN-Zero)
    - Optional 7 Yeo network heads (``network_head``) with optional zero-init

Decoder types:
    ``decoder_type='ditx'`` (default): DiT-X from ManiFlow — 9-param AdaLN-Zero
        modulating Self-Attention + Cross-Attention + FFN. Context is queried at every
        block via cross-attention with RoPE on Q/K for temporal alignment.
    ``decoder_type='dit1d'``: Legacy 1D-DiT — 6-param AdaLN-Zero on Self-Attention + FFN.
        Context is added once to x_t_emb before entering the block stack.
    ``use_dit_decoder: false``: FiLM + cross-attention (oldest legacy path).

Usage:
        python src/train_brainflow.py --config src/configs/brainflow.yaml
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
        w = self.weights[subject_ids]  # (B, in_channels, out_channels)
        if x.dim() == 3:
            # Seq2seq mode: (B, T, in_channels) → (B, T, out_channels)
            out = torch.einsum("btd,bdo->bto", x, w)
            if self.bias is not None:
                out = out + self.bias[subject_ids].unsqueeze(1)  # (B, 1, out_channels)
        else:
            # Single-step mode: (B, in_channels) → (B, out_channels)
            out = torch.einsum("bd,bdo->bo", x, w)
            if self.bias is not None:
                out = out + self.bias[subject_ids]
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

    def __init__(
        self,
        in_channels: int,
        n_subjects: int,
        network_counts: list[int] | None = None,
        zero_init: bool = False,
    ):
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
        if zero_init:
            for head in self.heads:
                nn.init.zeros_(head.weights)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

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
        self.attn_dropout_p = dropout

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
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = residual + attn_out

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-style modulation: x * (1 + scale) + shift. x is (B,T,D); shift/scale are (B,D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiT1DBlock(nn.Module):
    """1D DiT block: AdaLN-Zero + RoPE self-attention + FFN (temporal consistency on T_target)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

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

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(
            6, dim=-1
        )

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv_proj(x_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary_emb(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_norm2)
        return x


class CrossAttention(nn.Module):
    """Cross-attention with RoPE on Q/K for temporally-aligned context.

    Adapted from ManiFlow DiT-X. Q comes from target/action tokens,
    K/V from context tokens. RoPE is applied when both sequences share
    the same temporal axis (e.g. BrainFlow's 50-token aligned windows).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        rotary_emb: RotaryEmbedding = None,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend from x (query) to context (key/value).

        Args:
            x:       (B, T_q, D) target/action tokens.
            context: (B, T_kv, D) context tokens.

        Returns:
            (B, T_q, D) attended output.
        """
        B, T_q, D = x.shape
        _, T_kv, _ = context.shape

        q = self.q_proj(x).reshape(B, T_q, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(B, T_kv, 2, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Apply RoPE to Q and K when context is temporally aligned with target
        if self.rotary_emb is not None:
            cos_q, sin_q = self.rotary_emb(T_q)
            q = apply_rotary_emb(q, cos_q, sin_q)
            cos_k, sin_k = self.rotary_emb(T_kv)
            k = apply_rotary_emb(k, cos_k, sin_k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T_q, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        return attn_out


class DiTXBlock(nn.Module):
    """DiT-X block: AdaLN-Zero on Self-Attention + Cross-Attention + FFN.

    Adapted from ManiFlow. Key difference from DiT1DBlock:
    - 9 modulation parameters (3×3) instead of 6 (3×2)
    - Cross-attention at every block with AdaLN-Zero gating
    - Context is queried every layer (not added once at start)
    - RoPE on both Self-Attention and Cross-Attention Q/K
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        # Layer norms (no affine — modulated by AdaLN)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)  # self-attn
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)  # cross-attn
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)  # FFN

        # AdaLN-Zero: 9 modulation params (shift, scale, gate) × 3 sub-layers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 9 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # Self-attention
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # Cross-attention (Q from target, K/V from context)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            rotary_emb=rotary_emb,  # RoPE on Q/K for temporal alignment
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """DiT-X forward: Self-Attn → Cross-Attn → FFN, all with AdaLN-Zero.

        Args:
            x:       (B, T, D) target tokens (fMRI states).
            t_emb:   (B, D) time embedding.
            context: (B, T_ctx, D) encoded context tokens.

        Returns:
            (B, T, D) updated tokens.
        """
        B, T, D = x.shape

        # Generate 9 modulation parameters from time embedding
        modulation = self.adaLN_modulation(t_emb)  # (B, 9*D)
        chunks = modulation.chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # 1. Self-Attention with AdaLN-Zero
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv_proj(x_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary_emb(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. Cross-Attention with AdaLN-Zero
        x_norm_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_out = self.cross_attn(x_norm_cross, context)
        x = x + gate_cross.unsqueeze(1) * cross_out

        # 3. FFN with AdaLN-Zero
        x_norm_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_norm_mlp)

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
            intermediate_dim = 2048
            self.fusion_to_hidden = nn.Sequential(
                nn.Linear(self.concat_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
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
        # FiLM conditioning — broadcast over T_target dim if x is a sequence
        scale_shift = self.film(t_emb)                    # (B, D*2)
        scale, shift = scale_shift.chunk(2, dim=-1)       # (B, D) each
        is_seq = x.dim() == 3
        if is_seq:
            scale = scale.unsqueeze(1)                    # (B, 1, D) → broadcast over T_target
            shift = shift.unsqueeze(1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.ffn(h)

        # Cross-attention: Q=x (1 or T_target tokens), KV=context (T_ctx tokens)
        q = self.norm_q(x)                                # (B, D) or (B, T_target, D)
        kv = self.norm_kv(context)                        # (B, T_ctx, D)
        if not is_seq:
            attn_out, _ = self.cross_attn(q.unsqueeze(1), kv, kv)
            x = x + attn_out.squeeze(1)
        else:
            attn_out, _ = self.cross_attn(q, kv, kv)     # (B, T_target, D)
            x = x + attn_out

        return x


# =============================================================================
# VelocityNet — Temporal encoder + DiT (seq2seq) or FiLM + cross-attn (legacy)
# =============================================================================

class VelocityNet(nn.Module):
    """Velocity network with multitoken context encoder, optional temporal slice, DiT/DiT-X/FiLM trunk."""

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
        context_trs: int | None = None,
        n_subjects: int = 4,
        temporal_attn_layers: int = 2,
        fusion_mode: str = "concat",
        fusion_proj_dim: int = 384,
        use_subject_head: bool = True,
        latent_dim: int | None = None,
        gradient_checkpointing: bool = False,
        network_head: bool = False,
        use_rope: bool = False,
        n_target_trs: int = 1,
        context_encoder: str = "multitoken",
        use_dit_decoder: bool = True,
        dit_num_blocks: int | None = None,
        decoder_type: str = "ditx",
        zero_init_network_heads: bool = False,
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
        self.n_target_trs = n_target_trs
        self.context_encoder = context_encoder
        self.context_trs = int(context_trs) if context_trs is not None else int(max_seq_len)

        if context_encoder != "multitoken":
            raise ValueError(
                "context_encoder must be 'multitoken'. "
                f"Got {context_encoder!r}. Flat encoder was removed in this version."
            )

        use_dit = bool(use_dit_decoder and n_target_trs > 1)
        self._use_dit = use_dit
        self._decoder_type = decoder_type if use_dit else "film"
        dit_depth = dit_num_blocks if dit_num_blocks is not None else n_blocks

        # Learned positional embeddings:
        # - FiLM legacy path: used for seq2seq
        # - DiT-X path: used instead of additive context (context goes to cross-attn)
        # - DiT-1D legacy path: not needed (context added to x_t_emb)
        if n_target_trs > 1 and not use_dit:
            self.target_pos_emb = nn.Parameter(
                torch.randn(1, n_target_trs, hidden_dim) * 0.02
            )
        elif use_dit and decoder_type == "ditx":
            # DiT-X needs pos emb on target tokens (context goes via cross-attn)
            self.target_pos_emb = nn.Parameter(
                torch.randn(1, n_target_trs, hidden_dim) * 0.02
            )
        else:
            self.target_pos_emb = None

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

        enc_max_len = max(self.context_trs, max_seq_len)
        if use_rope:
            head_dim = hidden_dim // n_heads
            self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=enc_max_len)
            self.context_pos_emb = None
            self.temporal_attn = nn.ModuleList([
                RoPETransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    rotary_emb=self.rotary_emb,
                )
                for _ in range(temporal_attn_layers)
            ])
        else:
            self.context_pos_emb = nn.Parameter(
                torch.randn(1, enc_max_len, hidden_dim) * 0.02
            )
            self.rotary_emb = None
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
                self.subject_layers = NetworkSubjectLayers(
                    self.latent_dim,
                    n_subjects,
                    zero_init=zero_init_network_heads,
                )
            else:
                self.subject_layers = SubjectLayers(self.latent_dim, output_dim, n_subjects)
            self.subject_emb = None
        else:
            self.subject_layers = None
            self.subject_emb = nn.Embedding(n_subjects, hidden_dim)

        if use_dit:
            dec_max = max(n_target_trs, 64)
            head_dim_d = hidden_dim // n_heads
            self.rotary_emb_decoder = RotaryEmbedding(head_dim_d, max_seq_len=dec_max)
            if decoder_type == "ditx":
                self.dit_blocks = nn.ModuleList([
                    DiTXBlock(
                        d_model=hidden_dim,
                        nhead=n_heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=dropout,
                        time_dim=hidden_dim,
                        rotary_emb=self.rotary_emb_decoder,
                    )
                    for _ in range(dit_depth)
                ])
                print(f"  [VelocityNet] DiT-X decoder: {dit_depth} DiTXBlock "
                      f"(9-param AdaLN-Zero, cross-attention w/ RoPE)")
            else:  # dit1d legacy
                self.dit_blocks = nn.ModuleList([
                    DiT1DBlock(
                        d_model=hidden_dim,
                        nhead=n_heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=dropout,
                        time_dim=hidden_dim,
                        rotary_emb=self.rotary_emb_decoder,
                    )
                    for _ in range(dit_depth)
                ])
                print(f"  [VelocityNet] DiT-1D decoder: {dit_depth} DiT1DBlock "
                      f"(6-param AdaLN-Zero, additive context)")
            self.blocks = None
        else:
            self.rotary_emb_decoder = None
            self.dit_blocks = None
            self.blocks = nn.ModuleList([
                SimpleFiLMBlock(hidden_dim, hidden_dim, hidden_dim, n_heads, dropout)
                for _ in range(n_blocks)
            ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        if use_subject_head:
            self.latent_head = nn.Linear(hidden_dim, self.latent_dim)
            # Removed zero initialization on latent_head back to standard PyTorch initialization.
            # This prevents Double-Zero Initialization Gradient Death when NetworkSubjectLayers is also zero-initialized.
            self.output_layer = None
        else:
            self.latent_head = None
            self.output_layer = nn.Linear(hidden_dim, output_dim)
            nn.init.constant_(self.output_layer.weight, 0)
            nn.init.constant_(self.output_layer.bias, 0)

    def encode_context_from_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode context: multitoken fusion → temporal encoder → optional slice to ``n_target_trs``."""
        splits = []
        offset = 0
        for dim in self.modality_dims:
            splits.append(cond[:, :, offset:offset + dim])
            offset += dim
        context = self.fusion_block(splits)

        if self.context_pos_emb is not None:
            Tc = context.shape[1]
            context = context + self.context_pos_emb[:, :Tc, :]

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

        if self._use_dit:
            slice_start = (self.context_trs - self.n_target_trs) // 2
            context = context[:, slice_start : slice_start + self.n_target_trs, :]
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
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        if pre_encoded_context is not None:
            context_encoded = pre_encoded_context
        elif cond is not None:
            context_encoded = self.encode_context_from_cond(cond)
        else:
            if self._use_dit:
                tlen = self.n_target_trs
            else:
                tlen = 1
            context_encoded = torch.zeros(
                x.shape[0], tlen, self.hidden_dim, device=x.device, dtype=x.dtype
            )

        t_emb = self.time_mlp(self.time_embed(t))

        if not self.use_subject_head and self.subject_emb is not None and subject_ids is not None:
            t_emb = t_emb + self.subject_emb(subject_ids)

        h = self.input_proj(x)

        if self._use_dit:
            if self._decoder_type == "ditx":
                # DiT-X: additive context shortcut (baseline) + cross-attn (refinement)
                # The additive path provides context from epoch 0 (identical to dit1d).
                # Cross-attention gates start at 0 (AdaLN-Zero) and gradually open,
                # adding per-layer, per-position context refinement on top.
                h = h + context_encoded
                if self.target_pos_emb is not None:
                    T_h = h.shape[1]
                    h = h + self.target_pos_emb[:, :T_h, :]
                for block in self.dit_blocks:
                    if self.gradient_checkpointing and self.training:
                        h = checkpoint(block, h, t_emb, context_encoded, use_reentrant=False)
                    else:
                        h = block(h, t_emb, context_encoded)
            else:
                # DiT-1D legacy: additive context fusion, self-attn only
                h = h + context_encoded
                for block in self.dit_blocks:
                    if self.gradient_checkpointing and self.training:
                        h = checkpoint(block, h, t_emb, use_reentrant=False)
                    else:
                        h = block(h, t_emb)
        else:
            if x.dim() == 3 and self.target_pos_emb is not None:
                T = h.shape[1]
                h = h + self.target_pos_emb[:, :T, :]
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


class AECNN_HRF_Source(nn.Module):
    """
    Condition-dependent Source Generator using Biological HRF Prior.
    Compresses temporal context into neural events and filters through an HRF conv layer
    to create a biological base distribution (mu_phi) and variance (sigma_phi).
    """
    def __init__(self, context_dim: int, latent_dim: int, hrf_kernel_size: int = 12):
        super().__init__()
        # Step 1: Neural Event Extractor
        self.neural_event_net = nn.Sequential(
            nn.Conv1d(context_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, latent_dim, kernel_size=3, padding=1),
            nn.Sigmoid()  # Soft binarization to simulate neural firing
        )
        
        # Step 2: HRF Convolutional Filter
        # Linear Conv1D without activation simulating hemodynamic convolution
        self.hrf_filter = nn.Conv1d(
            latent_dim, latent_dim, 
            kernel_size=hrf_kernel_size, 
            padding='same', 
            groups=latent_dim
        )
        
        # Sigma Predictor (Variance)
        self.sigma_net = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softplus()  # Ensure strictly positive variance
        )

    def forward(self, context_sequence: torch.Tensor, context_pooled: torch.Tensor):
        # context_sequence: [B, C, T]
        # context_pooled: [B, C]
        
        neural_events = self.neural_event_net(context_sequence) # [B, latent_dim, T]
        mu_phi = self.hrf_filter(neural_events)                # [B, latent_dim, T]
        mu_phi = mu_phi.transpose(1, 2)                        # [B, T, latent_dim]
        
        sigma_phi = self.sigma_net(context_pooled).unsqueeze(1) # [B, 1, 1]
        
        return mu_phi, sigma_phi



class BrainFlow(nn.Module):
    """Flow matching with auxiliary regression, optional contrastive loss, and CFG.

    Context encoding is handled inside ``VelocityNet`` (multitoken fusion,
    temporal encoder, optional slice to ``n_target_trs`` for seq2seq DiT).
    Regression pools detached context over time (aligned slice when using DiT).
    """

    def __init__(
        self,
        output_dim: int = 1000,
        velocity_net_params: dict = None,
        n_subjects: int = 4,

        tensor_fm_params: dict = None,
        indi_flow_matching: bool = False,
        indi_train_time_sqrt: bool = False,
        indi_min_denom: float = 1e-3,
        use_csfm: bool = False,
        csfm_var_reg_weight: float = 0.1,
        csfm_align_weight: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.indi_flow_matching = indi_flow_matching
        self.indi_train_time_sqrt = indi_train_time_sqrt
        self.indi_min_denom = indi_min_denom
        self.use_csfm = use_csfm
        self.csfm_var_reg_weight = csfm_var_reg_weight
        self.csfm_align_weight = csfm_align_weight

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

        if self.use_csfm:
            self.hrf_source = AECNN_HRF_Source(
                context_dim=hidden_dim,
                latent_dim=latent_dim,
                hrf_kernel_size=vn_cfg.get("hrf_kernel_size", 12)
            )

        # Log parameters
        vn_params = sum(p.numel() for p in self.velocity_net.parameters())
        csfm_params = sum(p.numel() for p in self.hrf_source.parameters()) if self.use_csfm else 0
        warp_params = sum(p.numel() for p in self.time_warp_net.parameters()) if self.use_tensor_fm else 0
        
        print(f"[BrainFlow] VelocityNet: {vn_params:,} params")
        if self.use_csfm:
            print(f"  + HRF Source: {csfm_params:,} params")
        print(f"Total params: {vn_params + csfm_params:,} "
              f"(csfm={self.use_csfm}, tensor_fm={self.use_tensor_fm}, "
              f"modality_dims={vn_cfg.get('modality_dims')})")

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
            context: (B, T_ctx, total_dim) concatenated multimodal context.
            target:  (B, output_dim) or (B, n_target_trs, output_dim) ground-truth fMRI.
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
        csfm_var_reg_loss = _zero
        csfm_align_loss = _zero
        x_0_csfm = None

        if skip_aux:
            reg_loss = _zero
            cont_loss = _zero
        else:
            if self.use_csfm:
                ctx_detached = context_encoded.detach()  # ⛔ no gradient to fusion
                ctx_transposed = ctx_detached.transpose(1, 2)
                ctx_pooled_reg = ctx_detached.mean(dim=1)
                
                mu_phi_latent, sigma_phi = self.hrf_source(ctx_transposed, ctx_pooled_reg)
                
                if self.velocity_net.use_subject_head:
                    if subject_ids is None:
                        subject_ids = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
                    fmri_pred = self.velocity_net.subject_layers(mu_phi_latent, subject_ids)
                else:
                    fmri_pred = mu_phi_latent

                epsilon = torch.randn_like(target)
                x_0_csfm = fmri_pred + sigma_phi * epsilon
                
                # CSFM Losses
                csfm_var_reg_loss = torch.mean(sigma_phi**2 - torch.log(sigma_phi**2 + 1e-8) - 1.0)
                
                # Cosine alignment between the clean base distribution (mu_phi) and target
                cos_sim = F.cosine_similarity(fmri_pred.flatten(1), target.flatten(1), dim=1)
                csfm_align_loss = torch.mean(1.0 - cos_sim)

        # 4. Flow matching source distribution (x_0)
        x_1 = target
        if self.use_csfm and not skip_aux:
            x_0 = x_0_csfm
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
            # Manual OT interpolation — supports both (B, V) and (B, T, V) shapes
            t_bc = t
            while t_bc.dim() < x_1.dim():
                t_bc = t_bc.unsqueeze(-1)          # (B, 1, 1) for seq mode
            x_t = t_bc * x_1 + (1.0 - t_bc) * x_0

            if self._indi_effective:
                target_velocity = x_1 - x_t       # InDI: residual = (1-t)*(x_1-x_0)
            else:
                target_velocity = x_1 - x_0       # OT-CFM: constant velocity

            v_pred = self.velocity_net(
                x=x_t,
                t=t,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )
            flow_loss = F.mse_loss(v_pred, target_velocity)

        # 6. Total loss
        total_loss = flow_loss
        if self.use_csfm:
            total_loss = total_loss + self.csfm_var_reg_weight * csfm_var_reg_loss
            total_loss = total_loss + self.csfm_align_weight * csfm_align_loss
            
        total_loss = total_loss + self.gamma_reg_weight * gamma_reg

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "align_loss": csfm_align_loss if self.use_csfm else _zero,
            "var_reg_loss": csfm_var_reg_loss if self.use_csfm else _zero,
            "gamma_reg": gamma_reg,
        }

    def _build_time_grid(
        self,
        n_timesteps: int,
        device: torch.device,
        dtype: torch.dtype,
        time_grid_warp: str | None,
        max_t: float = 1.0,
    ) -> torch.Tensor:
        """Monotone grid in [0, 1]. ``sqrt`` uses T = sqrt(s) for finer steps near t→1."""
        if n_timesteps < 2:
            n_timesteps = 2
        max_t = float(max(0.0, min(1.0, max_t)))
        s = torch.linspace(0, max_t, n_timesteps, device=device, dtype=dtype)
        if time_grid_warp is None or time_grid_warp in ("", "none", "linear"):
            return s
        if time_grid_warp == "sqrt":
            return torch.sqrt(torch.clamp(s / max(max_t, 1e-8), min=0.0, max=1.0)) * max_t
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
        temperature: float = 0.0,
        time_grid_warp: str | None = None,
        time_grid_max: float = 1.0,
        final_jump: bool = False,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE. Supports single-TR and seq2seq modes.

        Single-step: returns (B, output_dim).
        Seq2seq:     returns (B, n_target_trs, output_dim).

        Args:
            context:               (B, T_ctx, total_dim) concatenated context.
            n_timesteps:           Number of ODE steps.
            solver_method:         ``'midpoint'`` (default) or ``'euler'``.
            subject_ids:           (B,) subject indices.
            cfg_scale:             CFG guidance scale (0 = disabled).
            temperature:           Std-dev of starting noise (0 = zero init).
            time_grid_warp:        ``'sqrt'`` = finer steps near t→1.
            time_grid_max:         Upper integration bound in [0,1] to avoid t→1 singularity.
            final_jump:            If True and time_grid_max<1, perform a final residual jump.

        Returns:
            fmri_pred: (B, output_dim) or (B, n_target_trs, output_dim).
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype
        n_target = getattr(self.velocity_net, 'n_target_trs', 1)

        # --- Encode context ---
        context_encoded = self.velocity_net.encode_context_from_cond(context)
        uncond_encoded = None
        if cfg_scale > 0:
            uncond_encoded = self.velocity_net.encode_context_from_cond(
                torch.zeros_like(context)
            )

        # --- Initialise x_0 ---
        if self.use_csfm:
            ctx_transposed = context_encoded.transpose(1, 2)
            ctx_pooled = context_encoded.mean(dim=1)
            mu_phi_latent, _ = self.hrf_source(ctx_transposed, ctx_pooled)
            
            if self.velocity_net.use_subject_head:
                if subject_ids is None:
                    subject_ids = torch.zeros(B, dtype=torch.long, device=device)
                mu_phi_fmri = self.velocity_net.subject_layers(mu_phi_latent, subject_ids)
            else:
                mu_phi_fmri = mu_phi_latent
            x = mu_phi_fmri.to(device=device, dtype=dtype)
        elif temperature > 0:
            shape = (B, n_target, self.output_dim) if n_target > 1 else (B, self.output_dim)
            x = temperature * torch.randn(*shape, device=device, dtype=dtype)
        else:
            shape = (B, n_target, self.output_dim) if n_target > 1 else (B, self.output_dim)
            x = torch.zeros(*shape, device=device, dtype=dtype)

        # --- Build time grid ---
        T_grid = self._build_time_grid(
            n_timesteps,
            device,
            dtype,
            time_grid_warp,
            max_t=time_grid_max,
        )

        # --- Velocity helper (handles InDI recovery) ---
        def _vel(x_in, t_scalar, ctx_enc):
            t_b = t_scalar.reshape(1).expand(B)
            u = self.velocity_net(
                x=x_in, t=t_b,
                pre_encoded_context=ctx_enc,
                subject_ids=subject_ids,
            )
            if self._indi_effective:
                u = recover_velocity_indi(u, t_scalar, self.indi_min_denom)
            return u

        # --- Manual ODE integration (works for any x shape) ---
        for i in range(len(T_grid) - 1):
            t_i = T_grid[i]
            dt = T_grid[i + 1] - T_grid[i]

            if solver_method == "midpoint":
                # Stage 1: velocity at t_i
                v1 = _vel(x, t_i, context_encoded)
                if cfg_scale > 0:
                    v1 = _vel(x, t_i, uncond_encoded) + cfg_scale * (v1 - _vel(x, t_i, uncond_encoded))
                # Stage 2: velocity at t_i + dt/2
                x_mid = x + 0.5 * dt * v1
                t_mid = t_i + 0.5 * dt
                v2 = _vel(x_mid, t_mid, context_encoded)
                if cfg_scale > 0:
                    v2 = _vel(x_mid, t_mid, uncond_encoded) + cfg_scale * (
                        v2 - _vel(x_mid, t_mid, uncond_encoded)
                    )
                x = x + dt * v2
            else:  # euler
                v = _vel(x, t_i, context_encoded)
                if cfg_scale > 0:
                    v = _vel(x, t_i, uncond_encoded) + cfg_scale * (v - _vel(x, t_i, uncond_encoded))
                x = x + dt * v

        # Optional final step for singularity-avoidance schedules that stop before t=1.
        if final_jump and float(T_grid[-1].item()) < 1.0:
            t_last = T_grid[-1]
            t_b = t_last.reshape(1).expand(B)
            u_last = self.velocity_net(
                x=x,
                t=t_b,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )
            if cfg_scale > 0:
                u_uncond = self.velocity_net(
                    x=x,
                    t=t_b,
                    pre_encoded_context=uncond_encoded,
                    subject_ids=subject_ids,
                )
                u_last = u_uncond + cfg_scale * (u_last - u_uncond)

            if self._indi_effective:
                # InDI predicts residual u ≈ x_1 - x_t at late t.
                x = x + u_last
            else:
                # Non-InDI models already predict velocity, so integrate remaining interval.
                x = x + (1.0 - t_last) * u_last

        return x
