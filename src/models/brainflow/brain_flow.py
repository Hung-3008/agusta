"""BrainFlow: Conditional Flow Matching for fMRI Generation.

Replaces TRIBE's direct prediction with a generative flow matching approach:
  Features → Encoder → Conditioning → Velocity Network → ODE → fMRI

Uses torchcfm for CFM training logic and torchdyn for ODE integration.

Changes from v1:
  - FeatureEncoder returns both global vector (B, H) and sequence (B, T, H)
  - VelocityDiTBlock uses cross-attention to encoder features + adaLN FFN
  - adaLN gate initialized to small non-zero value (not zero)
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
    """Sinusoidal positional embedding for flow time t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


class VelocityDiTBlock(nn.Module):
    """DiT block with cross-attention to encoder features + adaLN-modulated FFN.

    Each block:
      1. Cross-attention: h attends to encoder feature sequence (B, T, ctx_dim)
      2. FFN with adaptive LayerNorm modulation from conditioning vector

    Key difference from old AdaLNZeroBlock:
      - Cross-attention provides rich conditioning even at initialization
      - FFN gate initialized to small non-zero value (0.1) instead of 0,
        so FFN contributes from the start
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        context_dim: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Cross-attention to encoder features
        self.norm_x = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True,
        )

        # FFN with adaLN modulation
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim),
        )
        ff_dim = dim * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

        # Initialize adaLN: shift=0, scale=0, gate=0.1 (NOT zero!)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        # Set gate bias to 0.1 so FFN has effect from the start
        with torch.no_grad():
            self.adaLN_modulation[-1].bias[2 * dim:].fill_(0.1)

    def forward(
        self,
        x: torch.Tensor,       # (B, dim)
        c: torch.Tensor,       # (B, cond_dim) modulation vector
        context: torch.Tensor,  # (B, T, context_dim) encoder features
    ) -> torch.Tensor:
        # Cross-attention: h queries encoder feature sequence
        x_q = self.norm_x(x).unsqueeze(1)   # (B, 1, dim)
        ctx = self.norm_ctx(context)          # (B, T, context_dim)
        attn_out, _ = self.cross_attn(x_q, ctx, ctx)  # (B, 1, dim)
        x = x + attn_out.squeeze(1)          # (B, dim)

        # FFN with adaLN modulation
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = self.norm_ffn(x) * (1 + scale) + shift
        h = self.ffn(h)
        return x + gate * h


# =============================================================================
# Feature Encoder (adapted from TRIBE)
# =============================================================================

class FeatureEncoder(nn.Module):
    """Encode multimodal features into conditioning vectors.

    Adapted from TRIBE's FmriEncoder: per-modality MLP projectors → concat
    → Transformer encoder → mean-pool + full sequence.

    Returns:
        c_global : (B, H) — mean-pooled conditioning for time embedding
        c_seq    : (B, T, H) — full sequence for cross-attention in velocity net
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        hidden_dim: int = 1024,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        modality_dropout: float = 0.3,
        ff_dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout

        n_modalities = len(modality_dims)
        proj_dim = hidden_dim // n_modalities

        # Per-modality MLP projectors
        self.projectors = nn.ModuleDict()
        for mod, in_dim in modality_dims.items():
            self.projectors[mod] = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            )

        # Actual concat dim (may not perfectly divide)
        concat_dim = proj_dim * n_modalities
        self.input_proj = nn.Linear(concat_dim, hidden_dim) if concat_dim != hidden_dim else nn.Identity()

        # Learnable temporal positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=ff_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode multimodal features → global vector + sequence.

        Parameters
        ----------
        features : dict[str, Tensor]
            {modality: (B, T, D)} — batched feature sequences.

        Returns
        -------
        c_global : Tensor (B, hidden_dim)
            Global conditioning vector (mean-pooled).
        c_seq : Tensor (B, T, hidden_dim)
            Full temporal sequence for cross-attention.
        """
        B = next(iter(features.values())).shape[0]
        T = next(iter(features.values())).shape[1]
        device = next(iter(features.values())).device

        # Modality dropout: randomly zero-out modalities during training
        active_mods = list(self.modality_dims.keys())
        if self.training and self.modality_dropout > 0:
            to_drop = []
            for mod in active_mods:
                if torch.rand(1).item() < self.modality_dropout:
                    to_drop.append(mod)
            # Keep at least one
            if len(to_drop) == len(active_mods):
                to_drop.pop()
            active_mods = [m for m in active_mods if m not in to_drop]

        tensors = []
        proj_dim = self.hidden_dim // len(self.modality_dims)
        for mod in self.modality_dims.keys():
            if mod in active_mods and mod in features:
                # features[mod]: (B, T, D)
                x = features[mod].float()
                # Handle case where T might differ per modality (omni keep_tokens)
                if x.shape[1] != T:
                    # Resample to T via interpolation
                    x = x.permute(0, 2, 1)  # (B, D, T')
                    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
                    x = x.permute(0, 2, 1)  # (B, T, D)
                x = self.projectors[mod](x)   # (B, T, proj_dim)
            else:
                x = torch.zeros(B, T, proj_dim, device=device)
            tensors.append(x)

        # Concat modalities → (B, T, concat_dim)
        x = torch.cat(tensors, dim=-1)
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Add positional embedding
        x = x + self.pos_embed[:, :T, :]

        # Transformer encoder
        x = self.transformer(x)  # (B, T, hidden_dim)
        x = self.output_norm(x)

        # Return both global (mean-pooled) and sequence
        c_global = x.mean(dim=1)  # (B, hidden_dim)
        c_seq = x                  # (B, T, hidden_dim)
        return c_global, c_seq


# =============================================================================
# Subject Layers (per-subject linear decoder, identical to TRIBE)
# =============================================================================

class SubjectLayers(nn.Module):
    """Per-subject linear projection layer.

    Each subject has its own weight matrix to map from shared hidden
    representation to subject-specific fMRI space. This captures anatomical
    differences between subjects that a single shared layer cannot.

    Identical to TRIBE's SubjectLayers (Meta AI, 2024).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        bias: bool = True,
        init_id: bool = False,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels)) if bias else None

        if init_id:
            if in_channels != out_channels:
                raise ValueError("in_channels must equal out_channels for identity init.")
            self.weights.data[:] = torch.eye(in_channels)[None]
            if self.bias is not None:
                self.bias.data.zero_()
        else:
            self.weights.data.normal_()
            if self.bias is not None:
                self.bias.data.normal_()
        # Scale init by 1/sqrt(in_channels) for stable activations
        self.weights.data *= 1.0 / in_channels ** 0.5
        if self.bias is not None:
            self.bias.data *= 1.0 / in_channels ** 0.5

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        """Apply per-subject linear projection.

        Parameters
        ----------
        x : (B, in_channels)
        subjects : (B,) integer subject indices

        Returns
        -------
        out : (B, out_channels)
        """
        # x: (B, C), weights: (N, C, D)
        # Gather per-sample weight matrix
        weights = self.weights.index_select(0, subjects)  # (B, C, D)
        out = torch.einsum("bc,bcd->bd", x, weights)       # (B, D)
        if self.bias is not None:
            bias = self.bias.index_select(0, subjects)     # (B, D)
            out = out + bias
        return out

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C} → {D}, n_subjects={S})"


# =============================================================================
# Velocity Network
# =============================================================================

class VelocityNetwork(nn.Module):
    """Velocity field v_θ(x_t, t, c) for flow matching.

    Predicts the velocity that transports Gaussian noise → fMRI.
    Uses cross-attention to encoder feature sequence for rich conditioning.

    Parameters
    ----------
    n_voxels : int
        Number of fMRI voxels (output dimensionality).
    cond_dim : int
        Conditioning vector dimension (from FeatureEncoder).
    hidden_dim : int
        Hidden dimension for DiT blocks.
    n_blocks : int
        Number of DiT blocks.
    time_embed_dim : int
        Dimension of sinusoidal time embedding.
    n_subjects : int
        Number of subjects for per-subject embedding.
    n_heads : int
        Number of attention heads for cross-attention.
    """

    def __init__(
        self,
        n_voxels: int = 1000,
        cond_dim: int = 1024,
        hidden_dim: int = 2048,
        n_blocks: int = 8,
        time_embed_dim: int = 256,
        n_subjects: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_voxels = n_voxels

        # Time embedding: sinusoidal → MLP
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Subject embedding (optional)
        self.subject_embed = nn.Embedding(n_subjects, cond_dim)

        # Merge condition + time → modulation vector
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: x_t (n_voxels) → hidden_dim
        self.input_proj = nn.Linear(n_voxels, hidden_dim)

        # Stack of DiT blocks with cross-attention
        self.blocks = nn.ModuleList([
            VelocityDiTBlock(
                dim=hidden_dim,
                cond_dim=hidden_dim,
                context_dim=cond_dim,  # encoder hidden dim
                n_heads=n_heads,
                ff_mult=4,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])

        # Output projection: per-subject SubjectLayers (like TRIBE)
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.output_proj = SubjectLayers(
            in_channels=hidden_dim,
            out_channels=n_voxels,
            n_subjects=n_subjects,
            bias=True,
        )

    def forward(
        self,
        x_t: torch.Tensor,       # (B, V) noisy fMRI
        t: torch.Tensor,          # (B,) flow time
        c_global: torch.Tensor,   # (B, cond_dim) global conditioning
        c_seq: torch.Tensor,      # (B, T, cond_dim) encoder feature sequence
        subject_id: Optional[torch.Tensor] = None,  # (B,) subject indices
    ) -> torch.Tensor:
        """Predict velocity v(x_t, t, c).

        Returns
        -------
        v : Tensor (B, V) — predicted velocity field
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_embed_dim)

        # Subject embedding
        if subject_id is not None:
            c_global = c_global + self.subject_embed(subject_id)

        # Merge condition + time → modulation vector
        mod = self.cond_mlp(torch.cat([c_global, t_emb], dim=-1))  # (B, hidden_dim)

        # Input projection
        h = self.input_proj(x_t)  # (B, hidden_dim)

        # DiT blocks with cross-attention to encoder features
        for block in self.blocks:
            h = block(h, mod, c_seq)

        # Output: per-subject projection
        h = self.output_norm(h)
        if subject_id is None:
            weights_mean = self.output_proj.weights.mean(0)
            v = h @ weights_mean
            if self.output_proj.bias is not None:
                v = v + self.output_proj.bias.mean(0)
        else:
            v = self.output_proj(h, subject_id)
        return v


# =============================================================================
# BrainFlow: top-level model
# =============================================================================

class BrainFlowCFM(nn.Module):
    """BrainFlow: Conditional Flow Matching for fMRI generation.

    Combines a multimodal FeatureEncoder with a VelocityNetwork to learn
    a flow from Gaussian noise → fMRI, conditioned on stimulus features.
    """

    def __init__(
        self,
        # Feature Encoder config
        modality_dims: dict[str, int],
        encoder_hidden_dim: int = 1024,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        modality_dropout: float = 0.3,
        # Velocity Network config
        n_voxels: int = 1000,
        velocity_hidden_dim: int = 2048,
        n_velocity_blocks: int = 8,
        time_embed_dim: int = 256,
        n_subjects: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = FeatureEncoder(
            modality_dims=modality_dims,
            hidden_dim=encoder_hidden_dim,
            n_transformer_layers=n_transformer_layers,
            n_heads=n_heads,
            modality_dropout=modality_dropout,
            ff_dropout=dropout,
        )

        self.velocity_net = VelocityNetwork(
            n_voxels=n_voxels,
            cond_dim=encoder_hidden_dim,
            hidden_dim=velocity_hidden_dim,
            n_blocks=n_velocity_blocks,
            time_embed_dim=time_embed_dim,
            n_subjects=n_subjects,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.n_voxels = n_voxels

    def forward(
        self,
        features: dict[str, torch.Tensor],  # {mod: (B, T, D)}
        t: torch.Tensor,                    # (B,)
        x_t: torch.Tensor,                  # (B, V)
        subject_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity v_θ(x_t, t | features).

        Used during training with torchcfm:
            t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
            v_t = model(features, t, x_t, subject_id)
            loss = MSE(v_t, u_t)
        """
        c_global, c_seq = self.encoder(features)  # (B, H), (B, T, H)
        v = self.velocity_net(x_t, t, c_global, c_seq, subject_id)  # (B, V)
        return v

    @torch.no_grad()
    def sample(
        self,
        features: dict[str, torch.Tensor],
        subject_id: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        method: str = "euler",
    ) -> torch.Tensor:
        """Generate fMRI from noise via ODE integration.

        Parameters
        ----------
        features : dict[str, Tensor]
            {mod: (B, T, D)} batched features.
        subject_id : Tensor (B,), optional
        n_steps : int
            Number of integration steps.
        method : str
            'euler' (simple) or 'midpoint' (more accurate).

        Returns
        -------
        x1 : Tensor (B, V) — generated fMRI prediction
        """
        B = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device

        # Encode features once
        c_global, c_seq = self.encoder(features)  # (B, H), (B, T, H)

        # Start from Gaussian noise
        x = torch.randn(B, self.n_voxels, device=device)

        # Euler ODE integration from t=0 to t=1
        dt = 1.0 / n_steps
        ts = torch.linspace(0, 1 - dt, n_steps, device=device)

        for t_val in ts:
            t = torch.full((B,), t_val, device=device)
            if method == "euler":
                v = self.velocity_net(x, t, c_global, c_seq, subject_id)
                x = x + v * dt
            elif method == "midpoint":
                # Midpoint method (2nd order)
                v1 = self.velocity_net(x, t, c_global, c_seq, subject_id)
                x_mid = x + v1 * (dt / 2)
                t_mid = torch.full((B,), t_val + dt / 2, device=device)
                v2 = self.velocity_net(x_mid, t_mid, c_global, c_seq, subject_id)
                x = x + v2 * dt
            else:
                raise ValueError(f"Unknown method: {method}")

        return x
