"""BrainFlow MOTFM-VAE — OT-CFM in VAE Latent Space.

Instead of directly predicting 1000-dim fMRI, this model learns to generate
64-dim VAE latent vectors via flow matching.  At inference the generated
latent is decoded by a frozen VAE decoder back to 1000-dim fMRI.

OT-CFM (Optimal Transport Conditional Flow Matching):
  - Path:  x_t = (1-t)*x_0 + t*x_1
  - Target velocity:  dx = x_1 - x_0
  - Loss:  MSE(v_pred, dx)
  - Inference: ODE solve from t=0 → t=1  (in latent space)

Encoder output provides cross-attention conditioning to velocity network.
"""

import math

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
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


# =============================================================================
# Temporal Fusion Encoder (unchanged from MOTFM)
# =============================================================================

class TemporalFusionEncoder(nn.Module):
    """Multimodal temporal encoder: many TRs → context sequence.

    Input:  {mod_name: (B, T_ctx, D_mod)}  where T_ctx = n_context + 1
    Output: (B, T_ctx, hidden_dim)           context sequence for cross-attn
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        projector_depth: int = 2,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.n_modalities = len(modality_dims)

        # Per-modality MLP projectors
        self.projectors = nn.ModuleDict()
        for mod_name, raw_dim in modality_dims.items():
            layers = [
                nn.Linear(raw_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            for _ in range(projector_depth - 1):
                layers.append(ResidualBlock(hidden_dim, dropout))
            self.projectors[mod_name] = nn.Sequential(*layers)

        # Fusion: concat across modalities → project
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.n_modalities * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode multimodal features → context sequence (B, T, H)."""
        B = None
        T = None
        all_tokens = []

        # Modality dropout
        active_modalities = list(self.modality_dims.keys())
        if self.training and self.modality_dropout > 0:
            drop_mask = torch.rand(self.n_modalities) < self.modality_dropout
            if drop_mask.all():
                keep_idx = torch.randint(0, self.n_modalities, (1,)).item()
                drop_mask[keep_idx] = False
            active_modalities = [
                m for m, drop in zip(self.modality_dims.keys(), drop_mask) if not drop
            ]

        for mod_name in self.modality_dims.keys():
            if mod_name in modality_features and mod_name in active_modalities:
                x = modality_features[mod_name]
                if B is None:
                    B = x.shape[0]
                    T = x.shape[1]
                tokens = self.projectors[mod_name](x)
            else:
                if B is None:
                    for v in modality_features.values():
                        B, T = v.shape[0], v.shape[1]
                        break
                tokens = torch.zeros(B, T, self.hidden_dim,
                                     device=next(self.parameters()).device,
                                     dtype=next(self.parameters()).dtype)
            all_tokens.append(tokens)

        # Concat + project per TR
        x_cat = torch.cat(all_tokens, dim=-1)
        fused = self.fusion_proj(x_cat)

        # Temporal Transformer
        out = self.temporal_transformer(fused)
        out = self.final_norm(out)

        return out  # (B, T, H)


# =============================================================================
# Latent Velocity Network — FiLM + Per-Block Cross-Attention
# =============================================================================
# Each residual block has:
#   1. FiLM modulation from timestep embedding (time conditioning)
#   2. Cross-attention to FULL encoder output (B, T, C) (content conditioning)
#   3. Feed-forward residual MLP
# This avoids the information bottleneck of compressing 11×1024 → 512.
# =============================================================================


class CrossAttentionResBlock(nn.Module):
    """Residual block with FiLM (time) + cross-attention (context).

    Architecture per block:
        h = FiLM(LayerNorm(x), t_emb)     — time-conditioned modulation
        h = x + FFN(h)                     — residual MLP
        h = h + CrossAttn(h, context)      — attend to full encoder output
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        context_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # FiLM from timestep only (not mixed with context)
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

        # Cross-attention to full encoder context
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, D) hidden state
            t_emb:   (B, D_t) timestep embedding
            context: (B, T, C) full encoder output
        """
        # 1. FiLM modulated FFN
        scale_shift = self.film(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.ffn(h)

        # 2. Cross-attention to encoder context
        q = self.norm_q(x).unsqueeze(1)       # (B, 1, D)
        kv = self.norm_kv(context)             # (B, T, C)
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, D)
        x = x + attn_out.squeeze(1)           # (B, D)

        return x


class LatentVelocityNet(nn.Module):
    """Velocity network with FiLM time conditioning + per-block cross-attention.

    Architecture:
        1. Input: z_t (B, Z) → Linear → (B, D)
        2. Time: t → sinusoidal → MLP → (B, D)
        3. N blocks, each with:
           - FiLM(time_emb) → FFN residual
           - CrossAttn(hidden, encoder_output)
        4. Output: LayerNorm → Linear → (B, Z)

    Args:
        latent_dim:   Dimension of latent space (default 64).
        hidden_dim:   Internal MLP dimension (default 512).
        context_dim:  Dimension of encoder context sequence.
        n_blocks:     Number of cross-attention residual blocks.
        n_heads:      Attention heads per block.
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 512,
        context_dim: int = 1024,
        n_blocks: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Residual blocks with FiLM + cross-attention
        self.blocks = nn.ModuleList([
            CrossAttentionResBlock(hidden_dim, hidden_dim, context_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # Output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, latent_dim) noisy latent at time t.
            t:    (B,) or scalar, timestep in [0, 1].
            cond: (B, T, context_dim) full encoder output.

        Returns:
            v_pred: (B, latent_dim) predicted velocity.
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # Embeddings
        t_emb = self.time_embed(t)  # (B, D)

        # Prepare context
        if cond is not None and cond.dim() == 2:
            cond = cond.unsqueeze(1)
        if cond is None:
            cond = torch.zeros(x.shape[0], 1, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Forward
        h = self.input_proj(x)  # (B, D)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = self.final_norm(h)
        return self.output_proj(h)  # (B, latent_dim)


# =============================================================================
# Top-level BrainFlowMOTFM_VAE
# =============================================================================

class BrainFlowMOTFM_VAE(nn.Module):
    """BrainFlow with MOTFM architecture operating in VAE latent space.

    Combines:
    - TemporalFusionEncoder (from BrainFlow) for multimodal conditioning
    - LatentVelocityNet (MLP+AdaLN+CrossAttn) as velocity network
    - AffineProbPath + CondOTScheduler (Meta flow_matching) for OT-CFM
    - ODESolver (Meta flow_matching) for inference
    - Frozen fMRI_VAE_v5 decoder for latent → fMRI conversion at inference

    Training target: pre-computed 64-dim VAE latent vectors.
    Inference: ODE solve → decode via frozen VAE decoder → 1000-dim fMRI.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        latent_dim: int = 64,
        encoder_params: dict = None,
        velocity_net_params: dict = None,
        encoder: nn.Module = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # 1. Encoder — use external encoder if provided, else create default
        if encoder is not None:
            self.encoder = encoder
            cond_dim = encoder.hidden_dim
        else:
            self.encoder = TemporalFusionEncoder(
                modality_dims=modality_dims,
                **(encoder_params or {}),
            )
            cond_dim = (encoder_params or {}).get("hidden_dim", 512)

        # 2. Latent velocity network
        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("latent_dim", latent_dim)
        vn_cfg.setdefault("context_dim", cond_dim)
        self.velocity_net = LatentVelocityNet(**vn_cfg)

        # 3. OT-CFM path (same as MOTFM)
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # 4. VAE decoder (loaded separately, frozen)
        self.vae_decoder = None  # Set externally after init

        # Log parameters
        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_vn = sum(p.numel() for p in self.velocity_net.parameters())
        print(f"[BrainFlowMOTFM_VAE] Encoder: {n_enc:,} params, "
              f"VelocityNet: {n_vn:,} params, "
              f"Total: {n_enc + n_vn:,} params (excl. frozen VAE decoder)")

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OT-CFM loss in latent space.

        Args:
            modality_features: dict of (B, T, D_mod) multimodal features.
            target: (B, latent_dim) pre-computed VAE latent vector.

        Returns:
            loss: scalar MSE(v_pred, dx_t).
        """
        # Encode multimodal features → context
        context = self.encoder(modality_features)  # (B, T, H)

        # OT-CFM: sample path in latent space
        x_1 = target   # (B, Z)
        x_0 = torch.randn_like(x_1)  # N(0, I)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # Predict velocity
        v_pred = self.velocity_net(
            x=sample_info.x_t,
            t=sample_info.t,
            cond=context,
        )

        # MSE loss
        return F.mse_loss(v_pred, sample_info.dx_t)

    @torch.inference_mode()
    def synthesise(
        self,
        modality_features: dict[str, torch.Tensor],
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE in latent space then decoding.

        Returns:
            fmri_pred: (B, n_voxels) predicted fMRI.
        """
        context = self.encoder(modality_features)  # (B, T, H)
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        # Start from noise in latent space
        x_init = torch.randn(B, self.latent_dim, device=device, dtype=dtype)

        # ODE solve
        solver = ODESolver(velocity_model=self.velocity_net)
        T = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)

        z_pred = solver.sample(
            time_grid=T,
            x_init=x_init,
            method=solver_method,
            step_size=1.0 / n_timesteps,
            return_intermediates=False,
            cond=context,
        )  # (B, latent_dim)

        # Decode latent → fMRI using frozen VAE decoder
        if self.vae_decoder is not None:
            # VAE decoder expects (B, T, Z) with subject_id
            z_for_decode = z_pred.unsqueeze(1)  # (B, 1, Z)
            subject_id = torch.zeros(B, dtype=torch.long, device=device)
            fmri_pred = self.vae_decoder(z_for_decode, subject_id)  # (B, 1, V)
            return fmri_pred.squeeze(1)  # (B, V)
        else:
            # No VAE decoder loaded — return raw latent
            return z_pred

    @torch.inference_mode()
    def synthesise_latent(
        self,
        modality_features: dict[str, torch.Tensor],
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
    ) -> torch.Tensor:
        """Generate latent only (without VAE decoding)."""
        context = self.encoder(modality_features)
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        x_init = torch.randn(B, self.latent_dim, device=device, dtype=dtype)
        solver = ODESolver(velocity_model=self.velocity_net)
        T = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)

        return solver.sample(
            time_grid=T,
            x_init=x_init,
            method=solver_method,
            step_size=1.0 / n_timesteps,
            return_intermediates=False,
            cond=context,
        )
