"""BrainFlow CFM — Many-to-One Flow Matching.

Predicts 1 fMRI latent at time t from 11 TRs of multimodal features
(10 past TRs + 1 current TR).

Architecture:
  - TemporalFusionEncoder: per-modality projectors → concat per TR →
    fusion proj → Temporal Transformer → pool → context vector (B, H).
  - MLPFlowHead: ResidualMLP velocity net + additive conditioning.
    No self-attention needed since output is a single vector.
  - OT-CFM loss with Euler ODE solver.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Temporal Fusion Encoder — 11 TRs → 1 context vector
# =============================================================================

class TemporalFusionEncoder(nn.Module):
    """Multimodal temporal encoder: many TRs → 1 context vector.

    Input:  {mod_name: (B, T_ctx, D_mod)}  where T_ctx = n_context + 1
    Output: (B, hidden_dim)                  single fused context vector

    Architecture:
        1. Per-modality MLP projectors: (B, T_ctx, D_mod) → (B, T_ctx, H)
        2. Concat across modalities per TR + linear projection: (B, T_ctx, H)
        3. Temporal Transformer: self-attention across TRs
        4. Pool (last token) → (B, H)
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
        modality_features: dict[str, torch.Tensor],  # {mod: (B, T_ctx, D_mod)}
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
        x_cat = torch.cat(all_tokens, dim=-1)      # (B, T_ctx, H * N_mod)
        fused = self.fusion_proj(x_cat)             # (B, T_ctx, H)

        # Temporal Transformer
        out = self.temporal_transformer(fused)       # (B, T_ctx, H)
        out = self.final_norm(out)

        # Take LAST token (= current TR, most recent info)
        context = out[:, -1, :]                      # (B, H)
        return context


# =============================================================================
# MLP Flow Head — FiLM conditioning from BOTH time AND context at every layer
# =============================================================================

class FiLMResidualBlock(nn.Module):
    """Residual MLP block with FiLM modulation from external conditioning."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        # FiLM: conditioning → (scale, shift) for this block
        self.film = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, D), cond: (B, cond_dim)"""
        h = self.norm(x)
        # Apply FiLM modulation from conditioning BEFORE residual MLP
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        return x + self.net(h)


class MLPFlowHead(nn.Module):
    """MLP velocity network with FiLM conditioning from both time AND context.

    Context is injected at EVERY layer via FiLM modulation, not just as
    an additive bias at the output. This makes the model strongly conditional.

    Architecture:
      time → time_mlp → time_emb
      context → cond_mlp → cond_emb
      combined = time_emb + cond_emb  → FiLM at every block

      x_t → proj_in → [N × FiLMResidualBlock(combined)] → proj_out → v_θ
    """

    def __init__(
        self,
        latent_dim: int = 64,
        context_dim: int = 512,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,     # unused, kept for config compatibility
        time_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        cond_dim = hidden_dim  # internal conditioning dimension

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Context embedding (projects encoder output to conditioning space)
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # In/Out
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, latent_dim)
        # NO zero-init — allow model to learn from the start

        # FiLM Residual blocks (conditioned on time + context at every layer)
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(hidden_dim, cond_dim, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,        # (B, Z)
        t: torch.Tensor,           # (B,)
        context: torch.Tensor,     # (B, H)
    ) -> torch.Tensor:
        """Returns v_θ (B, Z)"""
        # Combined conditioning: time + context
        t_emb = self.time_mlp(t)       # (B, cond_dim)
        c_emb = self.cond_mlp(context) # (B, cond_dim)
        cond = t_emb + c_emb           # (B, cond_dim)

        # MLP with FiLM at every layer
        x = self.proj_in(x_t)
        for block in self.blocks:
            x = block(x, cond)         # FiLM modulation at each layer!

        v_t = self.proj_out(self.norm_out(x))
        return v_t


# =============================================================================
# OT-CFM Wrapper
# =============================================================================

class SimpleCFM(nn.Module):
    """OT-CFM with MLPFlowHead. Operates on single vectors (B, Z)."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        velocity_net_params: dict,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sigma_min = sigma_min

        self.estimator = MLPFlowHead(
            latent_dim=latent_dim,
            context_dim=context_dim,
            **velocity_net_params,
        )

    @torch.inference_mode()
    def forward(
        self,
        context: torch.Tensor,
        n_timesteps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        context_uncond: torch.Tensor = None,
    ) -> torch.Tensor:
        """Euler ODE sampling. Returns (B, Z)."""
        B = context.shape[0]
        device, dtype = context.device, context.dtype

        x = torch.randn(B, self.latent_dim, device=device, dtype=dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)

        t_val = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_val
            t_batch = t_val.expand(B)

            if guidance_scale != 1.0 and context_uncond is not None:
                v_cond = self.estimator(x, t_batch, context)
                v_uncond = self.estimator(x, t_batch, context_uncond)
                dphi_dt = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                dphi_dt = self.estimator(x, t_batch, context)

            x = x + dt * dphi_dt
            t_val = t_span[step]
        return x  # (B, Z)

    def compute_loss(
        self,
        x1: torch.Tensor,        # (B, Z) — target latent
        context: torch.Tensor,    # (B, H)
    ) -> torch.Tensor:
        """OT-CFM MSE loss."""
        B, D = x1.shape
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        z = torch.randn_like(x1)
        t_exp = t.unsqueeze(1)  # (B, 1)

        y = (1 - (1 - self.sigma_min) * t_exp) * z + t_exp * x1
        u = x1 - (1 - self.sigma_min) * z

        v_pred = self.estimator(y, t, context)
        loss = F.mse_loss(v_pred, u)
        return loss


# =============================================================================
# Top-level BrainFlowCFM
# =============================================================================

class BrainFlowCFM(nn.Module):
    """BrainFlow CFM — Many-to-One flow matching.

    Predicts 1 fMRI latent from N TRs of multimodal features.

    Components:
    1. TemporalFusionEncoder: N TR features → 1 context vector
    2. SimpleCFM: MLP velocity net + additive conditioning → 1 latent
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        latent_dim: int,
        encoder_params: dict,
        velocity_net_params: dict = None,
        cfm_params: dict = None,
        cfg_drop_prob: float = 0.1,
        n_voxels: int = 1000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cfg_drop_prob = cfg_drop_prob

        self.encoder = TemporalFusionEncoder(
            modality_dims=modality_dims,
            **encoder_params,
        )
        cond_dim = encoder_params.get("hidden_dim", 512)

        cfm_p = cfm_params or {"sigma_min": 1e-4}
        self.decoder = SimpleCFM(
            latent_dim=latent_dim,
            context_dim=cond_dim,
            velocity_net_params=velocity_net_params or {},
            sigma_min=cfm_p.get("sigma_min", 1e-4),
        )

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod: (B, T_ctx, D_mod)}
        latent: torch.Tensor,                          # (B, Z) — 1 target latent
    ) -> torch.Tensor:
        """Compute CFM loss for a single target TR."""
        context = self.encoder(modality_features)       # (B, H)

        # CFG dropout
        if self.training and self.cfg_drop_prob > 0.0:
            B = context.shape[0]
            cfg_keep = (torch.rand(B, 1, device=context.device) > self.cfg_drop_prob).float()
            context = context * cfg_keep

        loss = self.decoder.compute_loss(x1=latent, context=context)
        return loss

    @torch.inference_mode()
    def synthesise(
        self,
        modality_features: dict[str, torch.Tensor],
        n_timesteps: int = 20,
        temperature: float = 0.0,
        guidance_scale: float = 1.0,
        n_ensembles: int = 1,
    ) -> torch.Tensor:
        """Generate 1 fMRI latent from multimodal features.

        Returns: (B, Z)
        """
        context = self.encoder(modality_features)
        context_uncond = torch.zeros_like(context)

        if temperature <= 0.0:
            n_ensembles = 1

        z_acc = 0.0
        for _ in range(n_ensembles):
            z_sample = self.decoder(
                context=context,
                n_timesteps=n_timesteps,
                temperature=max(temperature, 1e-6),
                guidance_scale=guidance_scale,
                context_uncond=context_uncond,
            )
            z_acc = z_acc + z_sample

        return z_acc / n_ensembles  # (B, Z)
