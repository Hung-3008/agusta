"""BrainFlow MOTFM — Flow Matching with MONAI UNet backbone.

Adapted from MOTFM (Medical Optimal Transport Flow Matching).
Uses Meta's flow_matching library for OT-CFM path + ODE solver.
Uses MONAI's DiffusionModelUNet (spatial_dims=2) as velocity network.

fMRI (1000 voxels) is reshaped to pseudo-2D (1, 32, 32) with 24 padding.
Encoder output provides cross-attention conditioning.

Architecture:
  - TemporalFusionEncoder: multimodal features → context (B, H)
  - MergedModel (from MOTFM): wraps DiffusionModelUNet
  - AffineProbPath + CondOTScheduler: OT-CFM path
  - ODESolver: inference sampling
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from generative.networks.nets import DiffusionModelUNet
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver


# =============================================================================
# Pseudo-2D reshape helpers
# =============================================================================
# fMRI 1000 voxels → pad to 1024 → reshape (1, 32, 32)
FMRI_DIM = 1000
PAD_DIM = 1024  # 32 * 32
IMG_H, IMG_W = 32, 32


def fmri_to_2d(x: torch.Tensor) -> torch.Tensor:
    """(B, 1000) → (B, 1, 32, 32) with zero-padding."""
    B = x.shape[0]
    x_pad = F.pad(x, (0, PAD_DIM - FMRI_DIM))  # (B, 1024)
    return x_pad.reshape(B, 1, IMG_H, IMG_W)


def fmri_from_2d(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, 32, 32) → (B, 1000) by cropping."""
    B = x.shape[0]
    return x.reshape(B, -1)[:, :FMRI_DIM]


# =============================================================================
# Building Blocks (from existing BrainFlow)
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
# Temporal Fusion Encoder (from existing BrainFlow — unchanged)
# =============================================================================

class TemporalFusionEncoder(nn.Module):
    """Multimodal temporal encoder: many TRs → 1 context vector.

    Input:  {mod_name: (B, T_ctx, D_mod)}  where T_ctx = n_context + 1
    Output: (B, hidden_dim)                  single fused context vector
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
        """Encode multimodal features → single context vector (B, H)."""
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

        # Take LAST token (= current TR, most recent info)
        context = out[:, -1, :]
        return context


# =============================================================================
# MergedModel (adapted from MOTFM — no ControlNet)
# =============================================================================

class MergedModel(nn.Module):
    """UNet wrapper adapted from MOTFM.

    Wraps MONAI DiffusionModelUNet. Scales continuous t ∈ [0,1] to
    discrete timesteps for the UNet's sinusoidal embedding.
    No ControlNet (no mask conditioning for fMRI task).
    """

    def __init__(self, unet: DiffusionModelUNet, max_timestep: int = 1000):
        super().__init__()
        self.unet = unet
        self.max_timestep = max_timestep

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        **kwargs,  # absorb extra kwargs from ODESolver
    ):
        """
        Args:
            x: (B, 1, H, W) pseudo-2D fMRI.
            t: timesteps in [0,1], will be scaled to [0, max_timestep - 1].
            cond: (B, 1, cross_attention_dim) context for cross-attention.
        """
        # Scale continuous t → discrete timesteps (from MOTFM)
        t = t * (self.max_timestep - 1)
        t = t.floor().long()

        if t.dim() == 0:
            t = t.expand(x.shape[0])

        if cond is not None and cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, D)

        output = self.unet(x=x, timesteps=t, context=cond)
        return output


# =============================================================================
# Top-level BrainFlowMOTFM
# =============================================================================

class BrainFlowMOTFM(nn.Module):
    """BrainFlow with MOTFM architecture.

    Combines:
    - TemporalFusionEncoder (from BrainFlow) for multimodal conditioning
    - DiffusionModelUNet (from MOTFM/MONAI) as velocity network
    - AffineProbPath + CondOTScheduler (from Meta flow_matching) for OT-CFM
    - ODESolver (from Meta flow_matching) for inference

    fMRI (1000-d) is reshaped to pseudo-2D (1, 32, 32) for the UNet.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 1000,
        encoder_params: dict = None,
        unet_params: dict = None,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.cfg_drop_prob = cfg_drop_prob

        # 1. Encoder (from existing BrainFlow)
        self.encoder = TemporalFusionEncoder(
            modality_dims=modality_dims,
            **(encoder_params or {}),
        )
        cond_dim = (encoder_params or {}).get("hidden_dim", 512)

        # 2. UNet velocity network (from MOTFM)
        unet_cfg = dict(unet_params or {})
        unet_cfg.setdefault("spatial_dims", 2)
        unet_cfg.setdefault("in_channels", 1)
        unet_cfg.setdefault("out_channels", 1)
        unet_cfg.setdefault("with_conditioning", True)
        unet_cfg.setdefault("cross_attention_dim", cond_dim)

        max_timestep = unet_cfg.pop("max_timestep", 1000)
        unet = DiffusionModelUNet(**unet_cfg)
        self.velocity_net = MergedModel(unet=unet, max_timestep=max_timestep)

        # 3. OT-CFM path (from Meta flow_matching — same as MOTFM)
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log parameters
        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_unet = sum(p.numel() for p in self.velocity_net.parameters())
        print(f"[BrainFlowMOTFM] Encoder: {n_enc:,} params, UNet: {n_unet:,} params, "
              f"Total: {n_enc + n_unet:,} params")

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OT-CFM loss (training).

        Same logic as MOTFM trainer.py:_compute_loss()

        Args:
            modality_features: {mod: (B, T_ctx, D_mod)}
            target: (B, D) — ground truth fMRI (1000-d)
        """
        # Encode multimodal features → context
        context = self.encoder(modality_features)  # (B, H)

        # CFG dropout: zero out context randomly
        if self.training and self.cfg_drop_prob > 0.0:
            B = context.shape[0]
            cfg_keep = (torch.rand(B, 1, device=context.device) > self.cfg_drop_prob).float()
            context = context * cfg_keep

        # Reshape fMRI to pseudo-2D for UNet
        x_1 = fmri_to_2d(target)  # (B, 1, 32, 32)

        # OT-CFM: sample from noise → data path (from MOTFM)
        x_0 = torch.randn_like(x_1)  # N(0, I)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # Predict velocity
        v_pred = self.velocity_net(
            x=sample_info.x_t,
            t=sample_info.t,
            cond=context,
        )

        # MSE loss (from MOTFM)
        return F.mse_loss(v_pred, sample_info.dx_t)

    @torch.inference_mode()
    def synthesise(
        self,
        modality_features: dict[str, torch.Tensor],
        n_timesteps: int = 50,
        guidance_scale: float = 1.0,
        solver_method: str = "midpoint",
    ) -> torch.Tensor:
        """Generate fMRI using ODESolver (from MOTFM).

        Args:
            modality_features: {mod: (B, T_ctx, D_mod)}
            n_timesteps: number of solver time points
            guidance_scale: classifier-free guidance scale
            solver_method: ODE solver method ("euler", "midpoint", "rk4")

        Returns: (B, D) — generated fMRI (1000-d)
        """
        context = self.encoder(modality_features)  # (B, H)
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        # Start from noise
        x_init = torch.randn(B, 1, IMG_H, IMG_W, device=device, dtype=dtype)

        if guidance_scale != 1.0:
            # Classifier-free guidance: manual Euler
            context_uncond = torch.zeros_like(context)
            t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)

            x = x_init
            for step in range(1, len(t_span)):
                dt = t_span[step] - t_span[step - 1]
                t_batch = t_span[step - 1].expand(B)

                v_cond = self.velocity_net(x, t_batch, cond=context)
                v_uncond = self.velocity_net(x, t_batch, cond=context_uncond)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

                x = x + dt * v

            return fmri_from_2d(x)
        else:
            # Standard ODESolver (from MOTFM)
            solver = ODESolver(velocity_model=self.velocity_net)
            T = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)

            sol = solver.sample(
                time_grid=T,
                x_init=x_init,
                method=solver_method,
                step_size=1.0 / n_timesteps,
                return_intermediates=False,
                cond=context,
            )

            # sol is the final state (B, 1, H, W)
            return fmri_from_2d(sol)
