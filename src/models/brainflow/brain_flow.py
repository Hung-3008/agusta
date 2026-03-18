"""BrainFlow CFM — Many-to-One Direct Flow Matching.

Predicts 1 fMRI vector (1000-d) at time t from 11 TRs of multimodal features
(10 past TRs + 1 current TR). Direct fMRI output, no VAE needed.

Architecture:
  - TemporalFusionEncoder: per-modality projectors → concat per TR →
    fusion proj → Temporal Transformer → pool → context vector (B, H).
  - MLPFlowHead: Simple residual MLP velocity net with concat conditioning.
  - µ_cond source: ODE starts from encoder prediction, learns residual.
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
# MLP Flow Head — Simple concat conditioning
# =============================================================================

class MLPFlowHead(nn.Module):
    """Simple MLP velocity network with concatenation conditioning.

    Architecture:
      input = concat[x_t, t_emb, context] → proj_in
      → [N × ResidualBlock] → proj_out → v_θ
    """

    def __init__(
        self,
        output_dim: int = 1000,
        context_dim: int = 512,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,     # unused, kept for config compatibility
        time_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input: concat [x_t (output_dim), t_emb (time_dim), context (context_dim)]
        input_dim = output_dim + time_dim + context_dim
        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        # Simple residual blocks (no conditioning injection — everything is in input)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Output
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x_t: torch.Tensor,        # (B, D)
        t: torch.Tensor,           # (B,)
        context: torch.Tensor,     # (B, H)
    ) -> torch.Tensor:
        """Returns v_θ (B, D)"""
        t_emb = self.time_mlp(t)                     # (B, time_dim)
        inp = torch.cat([x_t, t_emb, context], dim=-1)  # (B, D + time_dim + H)

        x = self.proj_in(inp)                        # (B, hidden_dim)
        for block in self.blocks:
            x = block(x)

        return self.proj_out(self.norm_out(x))        # (B, D)


# =============================================================================
# DiT-1D: Patch-based Transformer Velocity Net
# =============================================================================

class AdaLNZero(nn.Module):
    """Adaptive LayerNorm Zero — modulates features based on conditioning.

    Given conditioning c (e.g., time embedding), produces:
        scale (γ), shift (β), gate (α) for modulating attention/FFN output.
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Produce 6 values: γ1, β1, α1 (for attn) + γ2, β2, α2 (for FFN)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, d_model * 6),
        )
        # Initialize gate (α) to zero → residual starts as identity
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x: (B, N, d_model) — features to modulate
            cond: (B, cond_dim) — conditioning vector (time embedding)
        Returns:
            Tuple of (normed_x_1, gate_1, normed_x_2, gate_2) for attn and FFN
        """
        params = self.proj(cond).unsqueeze(1)  # (B, 1, 6*d_model)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)

        x_norm = self.norm(x)
        x_mod1 = x_norm * (1 + gamma1) + beta1   # for self-attn
        x_mod2 = x_norm * (1 + gamma2) + beta2   # for FFN
        return x_mod1, alpha1, x_mod2, alpha2


class DiTBlock(nn.Module):
    """DiT Transformer block with AdaLN-Zero + optional Cross-Attention.

    Architecture:
        1. AdaLN-Zero → Self-Attention (inter-patch relationships)
        2. LayerNorm → Cross-Attention with context (optional)
        3. AdaLN-Zero → FFN (GELU)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cond_dim: int,
        context_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # AdaLN-Zero modulation from time embedding
        self.adaln = AdaLNZero(d_model, cond_dim)

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.self_attn_drop = nn.Dropout(dropout)

        # Cross-Attention with context
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn_drop = nn.Dropout(dropout)

        # Context projection (context_dim → d_model for K, V)
        self.context_proj = nn.Linear(context_dim, d_model) if context_dim != d_model else nn.Identity()

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,       # (B, N, d_model)
        t_emb: torch.Tensor,    # (B, cond_dim)
        context: torch.Tensor,  # (B, L, context_dim)
    ) -> torch.Tensor:
        # 1. AdaLN-Zero → Self-Attention
        x_mod1, gate1, x_mod2, gate2 = self.adaln(x, t_emb)
        sa_out, _ = self.self_attn(x_mod1, x_mod1, x_mod1)
        x = x + gate1 * self.self_attn_drop(sa_out)

        # 2. Cross-Attention with context
        x_cross = self.cross_norm(x)
        ctx = self.context_proj(context)   # (B, L, d_model)
        ca_out, _ = self.cross_attn(x_cross, ctx, ctx)
        x = x + self.cross_attn_drop(ca_out)

        # 3. AdaLN-Zero → FFN
        # Re-compute modulation for current x (after attn updates)
        _, _, x_mod2, gate2 = self.adaln(x, t_emb)
        x = x + gate2 * self.ffn(x_mod2)

        return x


class DiT1DFlowHead(nn.Module):
    """DiT-1D: Patch-based Transformer velocity network.

    Treats the 1000-dim fMRI vector as a sequence of patches and processes
    with Transformer blocks using AdaLN-Zero (time) + Cross-Attention (context).

    Architecture:
        1. Patch embed: (B, D) → (B, N_patches, d_model) + positional emb
        2. N × DiTBlock (Self-Attn + Cross-Attn + FFN)
        3. Unpatch: (B, N_patches, d_model) → (B, D)
    """

    def __init__(
        self,
        output_dim: int = 1000,
        context_dim: int = 512,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        patch_size: int = 20,
        time_dim: int = 256,
        dropout: float = 0.15,
        **kwargs,  # absorb unused keys from config
    ):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.n_patches = output_dim // patch_size
        assert output_dim % patch_size == 0, \
            f"output_dim ({output_dim}) must be divisible by patch_size ({patch_size})"

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Patch embedding: each patch of `patch_size` voxels → d_model
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, time_dim, context_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final norm + unpatch
        self.final_norm = nn.LayerNorm(d_model)
        self.unpatch_proj = nn.Linear(d_model, patch_size)

        # Zero-initialize output projection for stable start
        nn.init.zeros_(self.unpatch_proj.weight)
        nn.init.zeros_(self.unpatch_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,        # (B, D) — noisy fMRI
        t: torch.Tensor,           # (B,)
        context: torch.Tensor,     # (B, H)
    ) -> torch.Tensor:
        """Returns v_θ (B, D)"""
        B = x_t.shape[0]

        # Time embedding
        t_emb = self.time_mlp(t)   # (B, time_dim)

        # Patchify: (B, D) → (B, N_patches, patch_size) → (B, N_patches, d_model)
        x = x_t.reshape(B, self.n_patches, self.patch_size)
        x = self.patch_proj(x) + self.pos_emb      # (B, N_patches, d_model)

        # Context: (B, H) → (B, 1, H) for cross-attention
        ctx = context.unsqueeze(1)  # (B, 1, H)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, ctx)

        # Unpatch: (B, N_patches, d_model) → (B, N_patches, patch_size) → (B, D)
        x = self.final_norm(x)
        x = self.unpatch_proj(x)                    # (B, N_patches, patch_size)
        return x.reshape(B, self.output_dim)         # (B, D)


# =============================================================================
# OT-CFM Wrapper with adaptive normalization
# =============================================================================

class SimpleCFM(nn.Module):
    """Standard OT-CFM with adaptive fMRI normalization.

    Uses register_buffer('running_std') to track fMRI scale during training.
    fMRI is normalized to std≈1 before flow matching, denormalized after ODE.
    Source: x0 ~ N(mu_cond, I), Target: x1_norm = x1 / running_std.
    """

    def __init__(
        self,
        output_dim: int,
        context_dim: int,
        velocity_net_params: dict,
        sigma_min: float = 1e-4,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.sigma_min = sigma_min
        self.ema_decay = ema_decay

        # Select velocity net architecture based on config
        net_type = velocity_net_params.pop("type", "mlp")
        if net_type == "dit1d":
            self.estimator = DiT1DFlowHead(
                output_dim=output_dim,
                context_dim=context_dim,
                **velocity_net_params,
            )
        else:
            self.estimator = MLPFlowHead(
                output_dim=output_dim,
                context_dim=context_dim,
                **velocity_net_params,
            )

        # Running statistics for adaptive normalization (like BatchNorm)
        self.register_buffer('running_std', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    @torch.inference_mode()
    def forward(
        self,
        mu_cond: torch.Tensor,      # (B, D) — base fMRI prediction
        context: torch.Tensor,      # (B, H) — context for conditioning
        n_timesteps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        context_uncond: torch.Tensor = None,
    ) -> torch.Tensor:
        """Euler ODE sampling. Returns denormalized (B, D)."""
        B = context.shape[0]
        device, dtype = context.device, context.dtype

        # Normalize mu_cond to match target space
        mu_norm = mu_cond / self.running_std.clamp(min=1e-6)

        # Start from N(mu_norm, I * T_temperature)
        z = torch.randn(B, self.output_dim, device=device, dtype=dtype)
        x = mu_norm + z * temperature
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

        # Denormalize: normalized space → fMRI space
        return x * self.running_std

    def compute_loss(
        self,
        x1: torch.Tensor,        # (B, D) — target fMRI (raw)
        mu_cond: torch.Tensor,    # (B, D) — base fMRI prediction (raw, from encoder)
        context: torch.Tensor,    # (B, H) — context for velocity net
    ) -> torch.Tensor:
        """Standard OT-CFM loss with adaptive normalization and N(mu, I) source.

        1. Normalize x1 and mu_cond using running_std
        2. Sample x0 ~ N(mu_norm, I) => x0 = mu_norm + z
        3. Interpolate: x_t = (1-(1-σ)t)·x0 + t·x1_norm
        4. Target velocity: u = x1_norm - (1-σ)·x0
        """
        B, D = x1.shape

        # Update running statistics during training
        if self.training:
            with torch.no_grad():
                batch_std = x1.std().clamp(min=1e-6)
                if self.num_batches_tracked == 0:
                    self.running_std.copy_(batch_std)
                else:
                    self.running_std.mul_(self.ema_decay).add_(
                        batch_std * (1 - self.ema_decay)
                    )
                self.num_batches_tracked += 1

        # Normalize both target and source mean
        x1_norm = x1 / self.running_std.clamp(min=1e-6)
        mu_norm = mu_cond / self.running_std.clamp(min=1e-6)

        # Standard OT-CFM from N(mu_norm, I)
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        z = torch.randn_like(x1_norm)  # N(0, I)
        x0 = mu_norm + z               # N(mu_norm, I)
        t_exp = t.unsqueeze(1)

        y = (1 - (1 - self.sigma_min) * t_exp) * x0 + t_exp * x1_norm
        u = x1_norm - (1 - self.sigma_min) * x0

        v_pred = self.estimator(y, t, context)
        return F.mse_loss(v_pred, u)


# =============================================================================
# Top-level BrainFlowCFM
# =============================================================================

class BrainFlowCFM(nn.Module):
    """BrainFlow CFM — Many-to-One direct flow matching.

    Predicts 1 fMRI vector (1000-d) from N TRs of multimodal features.
    Standard Gaussian CFM with adaptive fMRI normalization.

    Components:
    1. TemporalFusionEncoder: N TR features → context vector (B, H)
    2. SimpleCFM: Gaussian source → velocity net → denormalized fMRI
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 1000,
        encoder_params: dict = None,
        velocity_net_params: dict = None,
        cfm_params: dict = None,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.cfg_drop_prob = cfg_drop_prob

        self.encoder = TemporalFusionEncoder(
            modality_dims=modality_dims,
            **(encoder_params or {}),
        )
        cond_dim = (encoder_params or {}).get("hidden_dim", 512)

        # µ_cond projection: context → initial fMRI prediction
        # Restored: this gives the optimal start point. Must load from Stage 1.
        self.mu_proj = nn.Linear(cond_dim, output_dim)

        cfm_p = cfm_params or {}
        self.decoder = SimpleCFM(
            output_dim=output_dim,
            context_dim=cond_dim,
            velocity_net_params=velocity_net_params or {},
            sigma_min=cfm_p.get("sigma_min", 1e-4),
        )

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod: (B, T_ctx, D_mod)}
        target: torch.Tensor,                          # (B, D) — 1 target fMRI
    ) -> torch.Tensor:
        """Compute CFM loss for a single target TR."""
        context = self.encoder(modality_features)       # (B, H)
        mu_cond = self.mu_proj(context)                 # (B, D)

        # CFG dropout: zero out context (but keeping mu_cond deterministic is fine)
        if self.training and self.cfg_drop_prob > 0.0:
            B = context.shape[0]
            cfg_keep = (torch.rand(B, 1, device=context.device) > self.cfg_drop_prob).float()
            context = context * cfg_keep

        # We detach mu_cond so the pretrained regression head stays frozen!
        return self.decoder.compute_loss(x1=target, mu_cond=mu_cond.detach(), context=context)

    @torch.inference_mode()
    def synthesise(
        self,
        modality_features: dict[str, torch.Tensor],
        n_timesteps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        n_ensembles: int = 1,
    ) -> torch.Tensor:
        """Generate 1 fMRI prediction from multimodal features.

        Returns: (B, D) — denormalized fMRI
        """
        context = self.encoder(modality_features)
        mu_cond = self.mu_proj(context)
        context_uncond = torch.zeros_like(context)

        acc = 0.0
        for _ in range(n_ensembles):
            sample = self.decoder(
                mu_cond=mu_cond,
                context=context,
                n_timesteps=n_timesteps,
                temperature=temperature,
                guidance_scale=guidance_scale,
                context_uncond=context_uncond,
            )
            acc = acc + sample

        return acc / n_ensembles  # (B, D)
