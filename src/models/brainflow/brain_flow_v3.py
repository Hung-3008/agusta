"""BrainFlow CFM v3 — Deep-Flow Inspired Architecture.

Key changes from v2:
  - Early Fusion Encoder: per-modality MLP projectors → concat tokens →
    Transformer → Ego-Centric Cross Attention (adapted from Deep-Flow).
  - Residual MLP FlowHead with concat conditioning (no AdaLN-Zero).
  - Intent-preserving skip connection: condition bypass Transformer → FlowHead.
  - No Latent-CFM (simplified pipeline).

Reference: Deep-Flow (Guillen-Perez, 2026) — FlowMatchingTrajectoryAnomaly
"""

import math
from typing import Optional

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
        """Args: t: (B,) scalar timesteps in [0, 1]."""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: LN → Linear → GELU → Dropout → Linear + skip."""

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
# Early Fusion Encoder (adapted from Deep-Flow SceneEncoder)
# =============================================================================

class EarlyFusionEncoder(nn.Module):
    """Multimodal encoder: per-modality projectors → early fusion → Transformer → Cross-Attention.

    Adapted from Deep-Flow's SceneEncoder:
    - Stage A: Per-modality MLP projectors (like Agent/Map/Goal encoders)
    - Stage B: Transformer encoder for global fusion (like 4× Transformer Layer)
    - Stage C: Ego-centric cross attention (CLS query → scene)

    Key adaptation from driving → fMRI domain:
    - Deep-Flow tokens are spatial (32 agents + 256 map + 1 goal = 289 tokens)
    - V3 tokens are temporal (T timesteps per modality, concatenated)
    - Deep-Flow "Ego" = Agent[0]. V3 "Ego" = learnable CLS token.
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

        # --- Stage A: Per-modality MLP projectors ---
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

        # Learnable CLS token (replaces Deep-Flow's Ego Agent[0])
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # --- Stage B: Transformer Encoder (Global Fusion) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,  # Pre-Norm for stability (same as Deep-Flow)
        )
        self.global_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Stage C: Ego-Centric Cross Attention ---
        self.ego_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod_name: (B, T, D_mod)}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode multimodal features.

        Returns:
            context_vector: (B, hidden_dim) — fused context from cross-attention
            skip_emb: (B, hidden_dim) — pre-transformer mean embedding (skip connection)
        """
        B = None
        all_tokens = []

        # Determine which modalities to keep
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
                x = modality_features[mod_name]  # (B, T, D_mod)
                if B is None:
                    B = x.shape[0]
                tokens = self.projectors[mod_name](x)  # (B, T, hidden_dim)
            else:
                if B is None:
                    for v in modality_features.values():
                        B = v.shape[0]
                        break
                T = next(iter(modality_features.values())).shape[1]
                tokens = torch.zeros(B, T, self.hidden_dim,
                                     device=next(self.parameters()).device,
                                     dtype=next(self.parameters()).dtype)
            all_tokens.append(tokens)

        # --- Skip connection: mean-pool BEFORE Transformer ---
        # Analogous to Deep-Flow's Goal Skip — preserves raw modality info
        pre_transformer = torch.cat(all_tokens, dim=1)  # (B, T * N_mods, H)
        skip_emb = pre_transformer.mean(dim=1)  # (B, H)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        all_tokens_with_cls = torch.cat([cls, pre_transformer], dim=1)  # (B, 1 + T*N, H)

        # --- Global Transformer Fusion ---
        global_tokens = self.global_transformer(all_tokens_with_cls)  # (B, 1+T*N, H)

        # --- Ego-Centric Cross Attention ---
        # CLS token (index 0) queries the entire scene
        ego_query = global_tokens[:, 0:1, :]  # (B, 1, H)
        context_vector, _ = self.ego_cross_attn(
            query=ego_query,
            key=global_tokens,
            value=global_tokens,
        )
        context_vector = self.final_norm(context_vector.squeeze(1))  # (B, H)

        return context_vector, skip_emb


# =============================================================================
# Residual MLP Flow Head (adapted from Deep-Flow FlowHead)
# =============================================================================

class ResidualMLPFlowHead(nn.Module):
    """Velocity network using concat conditioning + Residual MLP.

    Adapted from Deep-Flow's FlowHead:
    Input = Concat(z_t, time_emb, context, context_skip)
    → Linear → 5× ResidualBlock → Linear → v_θ

    Applied per-timestep: each TR is processed independently by the MLP.
    The Transformer encoder has already captured temporal dependencies.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        time_dim: int = 128,
        context_dim: int = 512,
        hidden_dim: int = 512,
        n_blocks: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Input = z_t + time_emb + context + context_skip
        input_dim = latent_dim + time_dim + context_dim + context_dim

        # Time embedding: Sinusoidal → MLP (same as Deep-Flow)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )

        # Main network: concat → ResidualMLPs → velocity
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, latent_dim))

        # Small init on final layer for stable start
        nn.init.normal_(layers[-1].weight, std=0.02)
        nn.init.zeros_(layers[-1].bias)

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,        # (B, T, latent_dim) or (B, latent_dim)
        t: torch.Tensor,           # (B,) timestep
        context: torch.Tensor,     # (B, context_dim)
        skip_emb: torch.Tensor,    # (B, context_dim) — skip connection
    ) -> torch.Tensor:
        """Predict velocity field v_θ(z_t, t, context, skip).

        Returns: (B, T, latent_dim) or (B, latent_dim)
        """
        has_time_dim = x_t.ndim == 3
        if has_time_dim:
            B, T, D = x_t.shape
            x_t = x_t.reshape(B * T, D)

        t_emb = self.time_mlp(t)  # (B, time_dim)

        if has_time_dim:
            # Expand context/skip/time to match B*T
            t_emb = t_emb.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            context = context.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            skip_emb = skip_emb.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        # Concat everything (Deep-Flow style)
        feat = torch.cat([x_t, t_emb, context, skip_emb], dim=-1)
        v_t = self.net(feat)

        if has_time_dim:
            v_t = v_t.reshape(B, T, -1)

        return v_t


# =============================================================================
# OT-CFM Wrapper
# =============================================================================

class ResidualMLPCFM(nn.Module):
    """OT-CFM with Residual MLP velocity estimator.

    Drop-in replacement for DiTCFM from V2.
    """

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

        self.estimator = ResidualMLPFlowHead(
            latent_dim=latent_dim,
            context_dim=context_dim,
            **velocity_net_params,
        )

    @torch.inference_mode()
    def forward(
        self,
        n_output_trs: int,
        context: torch.Tensor,        # (B, H)
        skip_emb: torch.Tensor,       # (B, H)
        mask: torch.Tensor = None,     # (B, 1, T)
        n_timesteps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        context_uncond: torch.Tensor = None,  # (B, H) for CFG
        skip_uncond: torch.Tensor = None,     # (B, H) for CFG
    ) -> torch.Tensor:
        """ODE sampling (Euler solver) with optional CFG.

        Returns: (B, T, latent_dim)
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        # Start from noise
        z = torch.randn(B, n_output_trs, self.latent_dim, device=device, dtype=dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)

        t_val = t_span[0]
        x = z

        for step in range(1, len(t_span)):
            dt = t_span[step] - t_val
            t_batch = t_val.expand(B)

            if guidance_scale != 1.0 and context_uncond is not None:
                # CFG: v = v_uncond + scale * (v_cond - v_uncond)
                v_cond = self.estimator(x, t_batch, context, skip_emb)
                v_uncond = self.estimator(x, t_batch, context_uncond, skip_uncond)
                dphi_dt = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                dphi_dt = self.estimator(x, t_batch, context, skip_emb)

            x = x + dt * dphi_dt
            t_val = t_span[step]

        if mask is not None:
            x = x * mask.transpose(1, 2)  # (B, T, 1)

        return x  # (B, T, latent_dim)

    def compute_loss(
        self,
        x1: torch.Tensor,          # (B, T, latent_dim) — target
        mask: torch.Tensor,         # (B, 1, T)
        context: torch.Tensor,      # (B, H)
        skip_emb: torch.Tensor,     # (B, H)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute OT-CFM loss.

        Returns: (loss, y) where y is the interpolated noisy sample.
        """
        B, T, D = x1.shape

        # Random timestep per sample
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)

        # Sample noise
        z = torch.randn_like(x1)

        # Interpolate: y(t) = (1 - (1-σ_min)t)z + t·x1
        t_exp = t.view(B, 1, 1)  # (B, 1, 1)
        y = (1 - (1 - self.sigma_min) * t_exp) * z + t_exp * x1

        # Target velocity: u = x1 - (1-σ_min)·z
        u = x1 - (1 - self.sigma_min) * z

        # Apply mask
        mask_T = mask.transpose(1, 2)  # (B, T, 1)
        u = u * mask_T

        # Predict velocity
        v_pred = self.estimator(y, t, context, skip_emb)

        # Masked MSE loss
        loss = F.mse_loss(v_pred * mask_T, u, reduction="sum") / (
            torch.sum(mask) * D
        )
        return loss, y


# =============================================================================
# Cross-Attention Transformer Flow Head (Matcha-TTS / VoiceBox inspired)
# =============================================================================

class CrossAttnBlock(nn.Module):
    """One Transformer layer: Pre-Norm Self-Attn → Cross-Attn → FFN.

    x (noisy latents) self-attends for temporal coherence, then cross-attends
    into the conditioning context tokens [context_tok, skip_tok].
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H)  kv: (B, N_ctx, H) — context+skip tokens."""
        # Temporal self-attention
        x_n = self.norm1(x)
        sa_out, _ = self.self_attn(x_n, x_n, x_n)
        x = x + sa_out
        # Conditional cross-attention
        x_n = self.norm2(x)
        ca_out, _ = self.cross_attn(query=x_n, key=kv, value=kv)
        x = x + ca_out
        # Feed-forward
        x = x + self.ffn(self.norm3(x))
        return x


class UDiTFlowHead(nn.Module):
    """Vastly scaled-up U-Net Diffusion Transformer (1D) for Flow Matching.

    Architecture (U-DiT 1D):
      x_t → FiLM(t_emb) → proj_in
      → [Down Blocks 0..N-1] (saves skips)
      → [Mid Block]
      → [Up Blocks 0..N-1] (inputs = concat(x, skip_from_down))
      → proj_out → v_θ

    This provides ~25-30M parameters matching the encoder's strength, and
    skip connections allow fast training of the identity flow.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        context_dim: int = 512,
        hidden_dim: int = 512,
        n_layers: int = 4,    # Creates 4 down + 1 mid + 4 up = 9 blocks total
        n_heads: int = 8,
        time_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, hidden_dim * 2),  # FiLM scale + shift
        )

        # Context KV Projection
        self.ctx_proj = nn.Linear(context_dim, hidden_dim)
        self.skip_proj = nn.Linear(context_dim, hidden_dim)

        # In/Out Projections
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # U-Net Blocks (No temporal downsampling for T=10, just feature skips)
        self.down_blocks = nn.ModuleList([
            CrossAttnBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        
        self.mid_block = CrossAttnBlock(hidden_dim, n_heads, dropout)

        # Up blocks need to process concat(current, skip) → 2*hidden_dim
        # so we project it back to hidden_dim before the attention block
        self.up_projs = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_layers)
        ])
        self.up_blocks = nn.ModuleList([
            CrossAttnBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,       # (B, T, Z)
        t: torch.Tensor,          # (B,)
        context: torch.Tensor,    # (B, H_ctx)
        skip_emb: torch.Tensor,   # (B, H_ctx)
    ) -> torch.Tensor:
        """Returns v_θ (B, T, Z)"""
        B, T, _ = x_t.shape

        # 1. Time FiLM
        t_emb = self.time_mlp(t)
        scale, shift = t_emb.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        # 2. KV Tokens
        ctx_tok = self.ctx_proj(context).unsqueeze(1)
        skip_tok = self.skip_proj(skip_emb).unsqueeze(1)
        kv = torch.cat([ctx_tok, skip_tok], dim=1)  # (B, 2, H)

        # 3. Input
        x = self.proj_in(x_t)
        x = x * (1.0 + scale) + shift

        # 4. Down pass
        skips = []
        for block in self.down_blocks:
            x = block(x, kv)
            skips.append(x)

        # 5. Mid pass
        x = self.mid_block(x, kv)

        # 6. Up pass (Reverse skips)
        for proj, block, skip in zip(self.up_projs, self.up_blocks, skips[::-1]):
            x = torch.cat([x, skip], dim=-1)  # (B, T, 2H)
            x = proj(x)                       # (B, T, H)
            x = block(x, kv)                  # (B, T, H)

        # 7. Output
        v_t = self.proj_out(self.norm_out(x))
        return v_t


class UDiTCFM(nn.Module):
    """OT-CFM wrapper for UDiTFlowHead."""

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

        self.estimator = UDiTFlowHead(
            latent_dim=latent_dim,
            context_dim=context_dim,
            **velocity_net_params,
        )

    @torch.inference_mode()
    def forward(
        self,
        n_output_trs: int,
        context: torch.Tensor,
        skip_emb: torch.Tensor,
        mask: torch.Tensor = None,
        n_timesteps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        context_uncond: torch.Tensor = None,
        skip_uncond: torch.Tensor = None,
    ) -> torch.Tensor:
        B = context.shape[0]
        device, dtype = context.device, context.dtype

        x = torch.randn(B, n_output_trs, self.latent_dim, device=device, dtype=dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)

        t_val = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_val
            t_batch = t_val.expand(B)

            if guidance_scale != 1.0 and context_uncond is not None:
                v_cond = self.estimator(x, t_batch, context, skip_emb)
                v_uncond = self.estimator(x, t_batch, context_uncond, skip_uncond)
                dphi_dt = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                dphi_dt = self.estimator(x, t_batch, context, skip_emb)

            x = x + dt * dphi_dt
            t_val = t_span[step]

        if mask is not None:
            x = x * mask.transpose(1, 2)
        return x

    def compute_loss(
        self,
        x1: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
        skip_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x1.shape
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        z = torch.randn_like(x1)
        t_exp = t.view(B, 1, 1)

        y = (1 - (1 - self.sigma_min) * t_exp) * z + t_exp * x1
        u = x1 - (1 - self.sigma_min) * z

        mask_T = mask.transpose(1, 2)
        u = u * mask_T

        v_pred = self.estimator(y, t, context, skip_emb)
        loss = F.mse_loss(v_pred * mask_T, u, reduction="sum") / (torch.sum(mask) * D)
        return loss, y

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        velocity_net_params: dict,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sigma_min  = sigma_min

        self.estimator = UDiTFlowHead(
            latent_dim=latent_dim,
            context_dim=context_dim,
            **velocity_net_params,
        )

    @torch.inference_mode()
    def forward(
        self,
        n_output_trs: int,
        context: torch.Tensor,
        skip_emb: torch.Tensor,
        mask: torch.Tensor = None,
        n_timesteps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        context_uncond: torch.Tensor = None,
        skip_uncond: torch.Tensor = None,
    ) -> torch.Tensor:
        """Euler ODE sampling with optional CFG. Returns (B, T, latent_dim)."""
        B = context.shape[0]
        device, dtype = context.device, context.dtype

        x = torch.randn(B, n_output_trs, self.latent_dim, device=device, dtype=dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)

        t_val = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_val
            t_batch = t_val.expand(B)

            if guidance_scale != 1.0 and context_uncond is not None:
                v_cond   = self.estimator(x, t_batch, context, skip_emb)
                v_uncond = self.estimator(x, t_batch, context_uncond, skip_uncond)
                dphi_dt  = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                dphi_dt = self.estimator(x, t_batch, context, skip_emb)

            x = x + dt * dphi_dt
            t_val = t_span[step]

        if mask is not None:
            x = x * mask.transpose(1, 2)
        return x

    def compute_loss(
        self,
        x1: torch.Tensor,       # (B, T, latent_dim)
        mask: torch.Tensor,      # (B, 1, T)
        context: torch.Tensor,
        skip_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OT-CFM MSE loss. Returns (loss, y)."""
        B, T, D = x1.shape
        t     = torch.rand(B, device=x1.device, dtype=x1.dtype)
        z     = torch.randn_like(x1)
        t_exp = t.view(B, 1, 1)

        y = (1 - (1 - self.sigma_min) * t_exp) * z + t_exp * x1
        u = x1 - (1 - self.sigma_min) * z

        mask_T = mask.transpose(1, 2)  # (B, T, 1)
        u      = u * mask_T

        v_pred = self.estimator(y, t, context, skip_emb)
        loss   = F.mse_loss(v_pred * mask_T, u, reduction="sum") / (
            torch.sum(mask) * D
        )
        return loss, y


# =============================================================================
# Top-level BrainFlowCFM v3
# =============================================================================

class BrainFlowCFMv3(nn.Module):
    """BrainFlow CFM v3 — Deep-Flow inspired architecture.

    Components:
    1. EarlyFusionEncoder: per-modality projectors → concat → Transformer → Cross-Attn
    2. ResidualMLPCFM: concat conditioning + Residual MLP velocity net
    3. fMRI regression head (auxiliary)

    Args:
        modality_dims: dict mapping modality name → feature dimension.
        latent_dim: VAE latent dimension.
        encoder_params: kwargs for EarlyFusionEncoder.
        velocity_net_params: kwargs for ResidualMLPFlowHead.
        cfm_params: CFM hyperparameters.
        cfg_drop_prob: Classifier-free guidance dropout probability.
        n_voxels: Number of fMRI voxels (for regression head).
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

        # Encoder
        self.encoder = EarlyFusionEncoder(
            modality_dims=modality_dims,
            **encoder_params,
        )
        cond_dim = encoder_params.get("hidden_dim", 512)

        # CFM decoder — U-DiT 1D (Scaled Capacity)
        cfm_p = cfm_params or {"sigma_min": 1e-4}
        self.decoder = UDiTCFM(
            latent_dim=latent_dim,
            context_dim=cond_dim,
            velocity_net_params=velocity_net_params or {},
            sigma_min=cfm_p.get("sigma_min", 1e-4),
        )

    def _encode_condition(
        self,
        modality_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode → (context_vector, skip_emb), both (B, H)."""
        return self.encoder(modality_features)

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod: (B, T, D_mod)}
        latent: torch.Tensor,                          # (B, T, Z)
        mask: torch.Tensor = None,                     # (B, 1, T)
        fmri_target: torch.Tensor = None,              # (B, T, V) optional
    ) -> torch.Tensor:
        """Compute CFM loss.

        Returns: diff_loss
        """
        B, T, _ = latent.shape

        if mask is None:
            mask = torch.ones(B, 1, T, device=latent.device, dtype=latent.dtype)

        # 1. Encode conditioning (clean context, no CFG yet)
        context, skip_emb = self._encode_condition(modality_features)

        # 2. Apply CFG dropout ONLY for diff_loss
        context_cfg = context
        skip_emb_cfg = skip_emb
        if self.training and self.cfg_drop_prob > 0.0:
            cfg_keep = (torch.rand(B, 1, device=context.device) > self.cfg_drop_prob).float()
            context_cfg = context * cfg_keep
            skip_emb_cfg = skip_emb * cfg_keep

        # 3. CFM loss (with CFG-dropped context)
        diff_loss, _ = self.decoder.compute_loss(
            x1=latent, mask=mask, context=context_cfg, skip_emb=skip_emb_cfg,
        )

        return diff_loss

    @torch.inference_mode()
    def synthesise(
        self,
        modality_features: dict[str, torch.Tensor],
        n_timesteps: int = 20,
        temperature: float = 0.0,
        guidance_scale: float = 1.0,
        n_output_trs: int = None,
        n_ensembles: int = 1,
    ) -> torch.Tensor:
        """Generate fMRI latents from multimodal features.

        Returns: (B, T, latent_dim)
        """
        ref = next(iter(modality_features.values()))
        B = ref.shape[0]
        device = ref.device

        T_lat = n_output_trs if n_output_trs is not None else ref.shape[1]
        mask = torch.ones(B, 1, T_lat, device=device, dtype=ref.dtype)

        context, skip_emb = self._encode_condition(modality_features)

        # Prepare uncond context for CFG
        context_uncond = torch.zeros_like(context)
        skip_uncond = torch.zeros_like(skip_emb)

        # Force 1 ensemble if deterministic
        if temperature <= 0.0:
            n_ensembles = 1

        z_acc = 0.0
        for _ in range(n_ensembles):
            z_sample = self.decoder(
                n_output_trs=T_lat,
                context=context,
                skip_emb=skip_emb,
                mask=mask,
                n_timesteps=n_timesteps,
                temperature=max(temperature, 1e-6),  # Avoid zero noise
                guidance_scale=guidance_scale,
                context_uncond=context_uncond,
                skip_uncond=skip_uncond,
            )
            z_acc = z_acc + z_sample

        return z_acc / n_ensembles  # (B, T, latent_dim)
