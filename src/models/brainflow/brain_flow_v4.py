"""BrainFlow v4: Latent-Space Conditional Flow Matching.

Generates fMRI VAE latent vectors (B, T, Z) from noise, conditioned on
PCA-reduced stimulus features and subject embeddings.

Key differences from v3:
  - Operates in VAE latent space (Z=256) instead of voxel space (V=1000)
  - No multimodal FeatureEncoder — uses precomputed PCA features
  - VelocityTransformer with temporal self-attention across TRs
  - Simple additive conditioning (no DiT adaLN or cross-attention)
  - Zero-initialized output for identity flow at init

Architecture:
    PCA cond (B,T,D) → ConditioningMLP → (B,T,H)
    + subject_embed + time_embed
    z_t (B,T,Z) → input_proj → (B,T,H)
    → add conditioning
    → TransformerEncoder (self-attention across T)
    → output_proj → v_θ (B,T,Z)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


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
    """Project PCA features → hidden dim.

    Simple 2-layer MLP with LayerNorm + GELU.
    Processes each TR independently (applied to (B, T, D_cond)).
    """

    def __init__(self, cond_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D_cond) → (B, T, H)"""
        return self.net(x)


# =============================================================================
# Velocity Transformer
# =============================================================================

class VelocityTransformer(nn.Module):
    """Velocity field v_θ(z_t, t, cond) with temporal self-attention.

    Processes the full sequence of TRs jointly, allowing the model to learn
    temporal dynamics in the latent space (HRF, temporal smoothness, etc.).

    Parameters
    ----------
    latent_dim : int
        VAE latent dimension (Z).
    hidden_dim : int
        Transformer d_model.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    dropout : float
    time_embed_dim : int
        Dimension of sinusoidal time embedding.
    max_seq_len : int
        Maximum sequence length for positional embedding.
    """

    def __init__(
        self,
        latent_dim: int = 256,
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

        # Time embedding: sinusoidal → MLP
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: z_t (Z) → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Learnable temporal positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder: self-attention across T positions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Output projection: hidden_dim → Z
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Zero-init output layer: velocity ≈ 0 at initialization (identity flow)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,      # (B, T, Z) noisy latent sequence
        t: torch.Tensor,         # (B,) flow time
        cond: torch.Tensor,      # (B, T, H) conditioning from CondMLP
    ) -> torch.Tensor:
        """Predict velocity v(z_t, t, cond).

        Returns
        -------
        v : (B, T, Z) — predicted velocity field in latent space
        """
        B, T, Z = z_t.shape

        # Project z_t to hidden dim
        h = self.input_proj(z_t)  # (B, T, H)

        # Add conditioning (additive — simple and effective)
        h = h + cond  # (B, T, H)

        # Add time embedding (broadcast across T)
        t_emb = self.time_embed(t)  # (B, H)
        h = h + t_emb.unsqueeze(1)  # (B, T, H)

        # Add positional embedding
        h = h + self.pos_embed[:, :T, :]  # (B, T, H)

        # Transformer: self-attention across T (temporal modeling)
        h = self.transformer(h)  # (B, T, H)

        # Output projection
        h = self.output_norm(h)
        v = self.output_proj(h)  # (B, T, Z)

        return v


# =============================================================================
# BrainFlow CFM v4 (top-level model)
# =============================================================================

class BrainFlowCFM_v4(nn.Module):
    """BrainFlow v4: Latent-Space Conditional Flow Matching.

    Generates fMRI VAE latent sequences conditioned on PCA stimulus features.
    Uses temporal self-attention to capture inter-TR dynamics.

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
    """

    def __init__(
        self,
        latent_dim: int = 256,
        cond_dim: int = 3750,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        n_subjects: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Conditioning: PCA features → hidden dim
        self.cond_mlp = ConditioningMLP(cond_dim, hidden_dim, dropout=dropout)

        # Subject embedding
        self.subject_embed = nn.Embedding(n_subjects, hidden_dim)

        # Velocity network
        self.velocity_net = VelocityTransformer(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        )

    def forward(
        self,
        pca_features: torch.Tensor,  # (B, T, D_cond)
        t: torch.Tensor,             # (B,)
        z_t: torch.Tensor,           # (B, T, Z)
        subject_id: Optional[torch.Tensor] = None,  # (B,)
    ) -> torch.Tensor:
        """Predict velocity v_θ(z_t, t | pca_features).

        Used during training with CFM:
            t, z_t, u_t = FM.sample_location_and_conditional_flow(z0, z1)
            v_t = model(pca_features, t, z_t, subject_id)
            loss = MSE(v_t, u_t)
        """
        # Conditioning
        cond = self.cond_mlp(pca_features)  # (B, T, H)

        # Add subject embedding
        if subject_id is not None:
            cond = cond + self.subject_embed(subject_id).unsqueeze(1)  # (B, T, H)

        # Predict velocity
        v = self.velocity_net(z_t, t, cond)  # (B, T, Z)
        return v

    @torch.no_grad()
    def sample(
        self,
        pca_features: torch.Tensor,  # (B, T, D_cond)
        subject_id: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        method: str = "euler",
    ) -> torch.Tensor:
        """Generate latent sequence from noise via ODE integration.

        Parameters
        ----------
        pca_features : (B, T, D_cond)
        subject_id : (B,)
        n_steps : int
        method : 'euler' or 'midpoint'

        Returns
        -------
        z1 : (B, T, Z) — generated latent sequence
        """
        B, T, _ = pca_features.shape
        device = pca_features.device

        # Encode conditioning once
        cond = self.cond_mlp(pca_features)  # (B, T, H)
        if subject_id is not None:
            cond = cond + self.subject_embed(subject_id).unsqueeze(1)

        # Start from Gaussian noise
        z = torch.randn(B, T, self.latent_dim, device=device)

        # ODE integration t=0 → t=1
        dt = 1.0 / n_steps
        ts = torch.linspace(0, 1 - dt, n_steps, device=device)

        for t_val in ts:
            t = torch.full((B,), t_val, device=device)
            if method == "euler":
                v = self.velocity_net(z, t, cond)
                z = z + v * dt
            elif method == "midpoint":
                v1 = self.velocity_net(z, t, cond)
                z_mid = z + v1 * (dt / 2)
                t_mid = torch.full((B,), t_val + dt / 2, device=device)
                v2 = self.velocity_net(z_mid, t_mid, cond)
                z = z + v2 * dt
            else:
                raise ValueError(f"Unknown method: {method}")

        return z

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"BrainFlowCFM_v4("
            f"latent_dim={self.latent_dim}, "
            f"cond_dim={self.cond_dim}, "
            f"params={n_params:,})"
        )
