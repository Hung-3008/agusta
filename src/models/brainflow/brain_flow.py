"""BrainFlow: Latent-Space Conditional Flow Matching.

Generates fMRI VAE latent vectors (B, T, Z) from noise, conditioned on
PCA-reduced stimulus features and subject embeddings.

Key design choices (Matcha-TTS + TRIBE inspired):
  - Operates in VAE latent space (Z=256) instead of voxel space (V=1000)
  - Channel concatenation: [z_t; cond] instead of additive conditioning
  - DiT-style adaLN: time+subject injected into EVERY transformer layer
  - TRIBE SubjectLayers: per-subject linear transforms for conditioning & output
  - Zero-initialized output for identity flow at init

Architecture:
    PCA cond (B,T,D) → ConditioningMLP → SubjectLayer → cond (B,T,P)
    [z_t (B,T,Z); cond (B,T,P)] → input_proj → (B,T,H)
    t (B,) + subject → MLP → per-layer (shift, scale, gate)
    → N × AdaLNTransformerBlock (self-attn + FFN + adaLN)
    → final_norm (adaLN) → SubjectLayer output → v_θ (B,T,Z)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building blocks
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for flow time t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# =============================================================================
# Conditioning MLP
# =============================================================================

class ConditioningMLP(nn.Module):
    """Project PCA features → conditioning dim for channel concatenation.

    2-layer MLP with LayerNorm + GELU.
    Processes each TR independently (applied to (B, T, D_cond)).
    """

    def __init__(self, cond_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D_cond) → (B, T, P)."""
        return self.net(x)


# =============================================================================
# TRIBE-style Subject Layers (per-subject linear transform)
# =============================================================================

class SubjectLayers(nn.Module):
    """Per-subject linear transform (TRIBE-style).

    Each subject gets its own weight matrix W_s: (in_dim, out_dim).
    For a given subject s, computes: y = x @ W_s + b_s.

    This is much more expressive than a shared embedding + adaLN because
    it allows full channel mixing per subject (a 2D matrix vs a 1D vector).

    Adapted from TRIBE (Meta) to work with BrainFlow's (B, T, C) tensors
    instead of the original (B, C, T) format.

    Parameters
    ----------
    in_dim : int
    out_dim : int
    n_subjects : int
    bias : bool
    init_id : bool
        If True, initialize weights as identity (requires in_dim == out_dim).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_subjects: int,
        bias: bool = True,
        init_id: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_subjects = n_subjects

        # (N, in_dim, out_dim) — one linear transform per subject
        self.weights = nn.Parameter(torch.empty(n_subjects, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_subjects, out_dim)) if bias else None

        if init_id:
            assert in_dim == out_dim, "init_id requires in_dim == out_dim"
            # Identity init: each subject starts as pass-through
            self.weights.data[:] = torch.eye(in_dim)[None]
        else:
            nn.init.xavier_uniform_(self.weights.data.view(-1, in_dim, out_dim))

    def forward(
        self,
        x: torch.Tensor,           # (B, T, C_in)
        subject_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Apply per-subject linear transform.

        Returns (B, T, C_out)
        """
        # Select weights for each sample in batch
        W = self.weights[subject_id]        # (B, C_in, C_out)
        # Batched matmul: (B, T, C_in) @ (B, C_in, C_out) → (B, T, C_out)
        out = torch.bmm(x, W)
        if self.bias is not None:
            out = out + self.bias[subject_id].unsqueeze(1)  # (B, 1, C_out)
        return out

    def __repr__(self) -> str:
        return (f"SubjectLayers({self.in_dim}, {self.out_dim}, "
                f"n_subjects={self.n_subjects})")


# =============================================================================
# DiT-style Adaptive LayerNorm Transformer Block
# =============================================================================

class AdaLNTransformerBlock(nn.Module):
    """Transformer block with Adaptive Layer Norm (DiT-style).

    Instead of standard Pre-LN, uses adaLN where the normalization
    shift/scale/gate are conditioned on flow time + subject.
    This injects time information at EVERY layer (like Matcha-TTS
    injects time into every ResNet block).

    Flow:
        adaLN_1(h, γ₁, β₁) → Self-Attention → h + gate₁ · attn_out
        adaLN_2(h, γ₂, β₂) → FFN            → h + gate₂ · ffn_out

    Parameters
    ----------
    hidden_dim : int
    n_heads : int
    dropout : float
    ffn_mult : int
        FFN expansion factor.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # adaLN norms (no learnable affine — shift/scale come from conditioning)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        ffn_dim = hidden_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # adaLN modulation: 6 values per token (shift1, scale1, gate1, shift2, scale2, gate2)
        # These are projected from the global conditioning embedding
        # Initialized so that shift=0, scale=1, gate=1 at init (identity)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Zero-init the linear layer so outputs start at 0
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        h: torch.Tensor,       # (B, T, H)
        c: torch.Tensor,       # (B, H) — global conditioning (time + subject)
    ) -> torch.Tensor:
        # Compute modulation parameters from conditioning
        mod = self.adaLN_modulation(c)  # (B, 6*H)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)  # each (B, H)

        # --- Self-Attention block ---
        # adaLN: norm then apply learned shift/scale
        h_norm = self.norm1(h)
        h_norm = h_norm * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)  # (B, T, H)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm)
        h = h + gate1.unsqueeze(1) * attn_out  # gated residual

        # --- FFN block ---
        h_norm = self.norm2(h)
        h_norm = h_norm * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        ffn_out = self.ffn(h_norm)
        h = h + gate2.unsqueeze(1) * ffn_out  # gated residual

        return h


# =============================================================================
# Velocity Transformer (DiT-style)
# =============================================================================

class VelocityTransformer(nn.Module):
    """Velocity field v_θ(z_t, t, cond) with DiT-style temporal transformer.

    Key improvements over previous version:
    1. Channel concatenation: [z_t; cond] instead of z_t + cond
    2. DiT-style adaLN: time+subject modulates every layer
    3. Gated residuals for stable training

    Parameters
    ----------
    latent_dim : int
        VAE latent dimension (Z).
    cond_proj_dim : int
        Projected conditioning dimension for concatenation.
    hidden_dim : int
        Transformer d_model.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    dropout : float
    time_embed_dim : int
        Dimension of sinusoidal time embedding.
    max_seq_len : int
        Maximum sequence length for positional embedding.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        cond_proj_dim: int = 256,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Time embedding: sinusoidal → MLP → hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: concat [z_t; cond] → hidden_dim
        # (Matcha-TTS style: concat noise and condition along channel dim)
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + cond_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Learnable temporal positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # DiT-style transformer blocks (adaLN in every layer)
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final adaLN + output projection
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # shift + scale
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        # Shared output projection (before subject-specific layer)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Zero-init output layer: velocity ≈ 0 at initialization (identity flow)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,      # (B, T, Z) noisy latent sequence
        t: torch.Tensor,         # (B,) flow time
        cond: torch.Tensor,      # (B, T, P) projected conditioning
        c_global: torch.Tensor,  # (B, H) global conditioning (time + subject)
    ) -> torch.Tensor:
        """Predict velocity v(z_t, t, cond).

        Returns
        -------
        v : (B, T, Z) — predicted velocity field in latent space
        """
        B, T, Z = z_t.shape

        # Channel concatenation: [z_t; cond] (Matcha-TTS style)
        h = torch.cat([z_t, cond], dim=-1)  # (B, T, Z+P)
        h = self.input_proj(h)  # (B, T, H)

        # Add positional embedding
        h = h + self.pos_embed[:, :T, :]  # (B, T, H)

        # DiT blocks: each layer receives time+subject conditioning
        for block in self.blocks:
            h = block(h, c_global)  # (B, T, H)

        # Final adaLN + shared output projection
        final_mod = self.final_adaLN(c_global)  # (B, 2*H)
        shift, scale = final_mod.chunk(2, dim=-1)
        h = self.final_norm(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        v = self.output_proj(h)  # (B, T, Z)

        return v


# =============================================================================
# BrainFlow CFM (top-level model)
# =============================================================================

class BrainFlowCFM(nn.Module):
    """BrainFlow: Latent-Space Conditional Flow Matching.

    Generates fMRI VAE latent sequences conditioned on PCA stimulus features.
    Uses DiT-style temporal transformer with adaLN conditioning.

    Key improvements (Matcha-TTS inspired):
    1. Channel concat [z_t; cond] instead of additive
    2. adaLN: time+subject modulates every transformer layer
    3. Gated residuals for training stability

    Parameters
    ----------
    latent_dim : int
        VAE latent dimension.
    cond_dim : int
        PCA conditioning dimension.
    hidden_dim : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    n_subjects : int
        Number of subjects for subject embedding.
    dropout : float
    time_embed_dim : int
    cond_proj_dim : int
        Projected conditioning dimension for concat with z_t.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        cond_dim: int = 3750,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        n_subjects: int = 1,   # kept for API compat, no longer used
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        cond_proj_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Conditioning: PCA features → projected dim for concat
        self.cond_mlp = ConditioningMLP(cond_dim, cond_proj_dim, dropout=dropout)

        # Velocity network (DiT-style)
        self.velocity_net = VelocityTransformer(
            latent_dim=latent_dim,
            cond_proj_dim=cond_proj_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        )

    def _build_global_cond(self, t: torch.Tensor) -> torch.Tensor:
        """Build global conditioning vector from time embedding only.

        Returns (B, H) vector used by adaLN in every transformer layer.
        """
        return self.velocity_net.time_embed(t)  # (B, H)

    def forward(
        self,
        pca_features: torch.Tensor,  # (B, T, D_cond)
        t: torch.Tensor,             # (B,)
        z_t: torch.Tensor,           # (B, T, Z)
        subject_id: Optional[torch.Tensor] = None,  # kept for API compat, unused
    ) -> torch.Tensor:
        """Predict velocity v_θ(z_t, t | pca_features)."""
        cond = self.cond_mlp(pca_features)      # (B, T, P)
        c_global = self._build_global_cond(t)  # (B, H)
        return self.velocity_net(z_t, t, cond, c_global)  # (B, T, Z)

    @torch.no_grad()
    def sample(
        self,
        pca_features: torch.Tensor,  # (B, T, D_cond)
        subject_id: Optional[torch.Tensor] = None,  # unused, kept for API compat
        n_steps: int = 50,
        method: str = "euler",
    ) -> torch.Tensor:
        """Generate latent sequence from noise via ODE integration."""
        B, T, _ = pca_features.shape
        device = pca_features.device

        cond = self.cond_mlp(pca_features)  # (B, T, P) — encode once
        z = torch.randn(B, T, self.latent_dim, device=device)

        dt = 1.0 / n_steps
        ts = torch.linspace(0, 1 - dt, n_steps, device=device)

        for t_val in ts:
            t = torch.full((B,), t_val, device=device)
            c_global = self._build_global_cond(t)
            if method == "euler":
                z = z + self.velocity_net(z, t, cond, c_global) * dt
            elif method == "midpoint":
                v1 = self.velocity_net(z, t, cond, c_global)
                z_mid = z + v1 * (dt / 2)
                t_mid = torch.full((B,), t_val + dt / 2, device=device)
                c_mid = self._build_global_cond(t_mid)
                z = z + self.velocity_net(z_mid, t_mid, cond, c_mid) * dt
            else:
                raise ValueError(f"Unknown method: {method}")
        return z

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"BrainFlowCFM("
            f"latent_dim={self.latent_dim}, "
            f"cond_dim={self.cond_dim}, "
            f"params={n_params:,})"
        )
