"""BrainFlow CFM v2 — TRIBE-style Encoder + DiT/U-Net Decoder.

Changes from v1:
  - ConditionEncoder replaced with MultiModalConditionEncoder (TRIBE-style):
    Per-modality MLP projectors, modality dropout, Transformer encoder.
    No PCA preprocessing needed — operates on raw extracted features.
  - DiT-style decoder with AdaLN-Zero + Cross-Attention as alternative
    to the U-Net decoder. Both backends supported for ablation.
  - Classifier-Free Guidance (CFG) preserved.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import: U-Net CFM depends on diffusers which may not be installed
# Import only when decoder_type="unet" is used
CFM = None

def _get_cfm_class():
    global CFM
    if CFM is None:
        from src.models.brainflow.components.flow_matching import CFM as _CFM
        CFM = _CFM
    return CFM


from src.models.brainflow.components.rope import RotaryEmbedding, apply_rotary_pos_emb

# =============================================================================
# Custom Transformer Encoder Layer for RoPE
# =============================================================================

class RoPETransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer customized to inject RoPE."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        self.nhead = nhead
        self.head_dim = d_model // nhead

    def forward(self, src, cos, sin):
        # Pre-LN
        x = self.norm1(src)
        B, T, C = x.shape
        
        # Project Q, K, V
        qkv = self.self_attn.in_proj_weight @ x.transpose(1, 2)
        qkv = qkv.transpose(1, 2) + self.self_attn.in_proj_bias
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for RoPE
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2) # (B, H, T, D)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Flash Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.self_attn.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        
        # Output proj
        attn_out = self.self_attn.out_proj(attn_out)
        
        # Residual
        src = src + self.dropout1(attn_out)
        
        # FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


# =============================================================================
# TRIBE-style Condition Encoder
# =============================================================================

class MultiModalConditionEncoder(nn.Module):
    """Encode raw multimodal features into conditioning vectors.

    TRIBE-inspired design:
      1. Per-modality MLP projectors: raw_dim → hidden_dim // N_modalities
      2. Concatenate all modalities → (B, T, hidden_dim)
      3. Transformer encoder for temporal context
      4. Modality dropout for training robustness

    Args:
        modality_dims: dict mapping modality name → input dimension.
            e.g. {"video": 5632, "audio": 1280, "text": 15360, "omni": 3584}
        hidden_dim: Transformer hidden dimension (default: 768).
        n_layers: Number of Transformer encoder layers (default: 4).
        n_heads: Number of attention heads (default: 8).
        dropout: Dropout rate (default: 0.1).
        modality_dropout: Probability of dropping an entire modality (default: 0.2).
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        hidden_dim: int = 768,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: float = 0.2,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.n_modalities = len(modality_dims)

        # Per-modality MLP projectors
        # Each modality projects raw_dim → hidden_dim (bottleneck) → per_mod_dim
        per_mod_dim = hidden_dim // self.n_modalities
        self.projectors = nn.ModuleDict()
        for mod_name, raw_dim in modality_dims.items():
            self.projectors[mod_name] = nn.Sequential(
                nn.Linear(raw_dim, hidden_dim),       # raw → full hidden (large bottleneck)
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, per_mod_dim),    # hidden → per_mod_dim
                nn.LayerNorm(per_mod_dim),
            )

        # Actual hidden dim after concat (handle rounding)
        actual_hidden = per_mod_dim * self.n_modalities

        # Optional linear to map to exact hidden_dim if rounding occurred
        if actual_hidden != hidden_dim:
            self.dim_fix = nn.Linear(actual_hidden, hidden_dim)
        else:
            self.dim_fix = nn.Identity()

        # RoPE embedding (no more absolute pos_embed)
        self.rope = RotaryEmbedding(dim=hidden_dim // n_heads)

        # Transformer encoder for temporal context
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, dropout=dropout
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod_name: (B, T, D_mod)}
    ) -> torch.Tensor:
        """Encode multimodal features.

        Args:
            modality_features: dict mapping modality name → tensor (B, T, D_mod).

        Returns:
            cond_seq: (B, T, hidden_dim) — conditioning sequence for cross-attention.
        """
        B = None
        T = None
        projections = []

        # Determine modalities to drop (at least 1 must remain)
        active_modalities = list(self.modality_dims.keys())
        if self.training and self.modality_dropout > 0:
            drop_mask = torch.rand(self.n_modalities) < self.modality_dropout
            # Ensure at least one modality survives
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
                    B, T = x.shape[0], x.shape[1]
                proj = self.projectors[mod_name](x)  # (B, T, per_mod_dim)
            else:
                # Zero-fill for dropped or missing modalities
                if B is None:
                    # Need at least one modality to determine B, T
                    for v in modality_features.values():
                        B, T = v.shape[0], v.shape[1]
                        break
                per_mod_dim = self.hidden_dim // self.n_modalities
                proj = torch.zeros(B, T, per_mod_dim,
                                   device=next(self.parameters()).device,
                                   dtype=next(self.parameters()).dtype)
            projections.append(proj)

        # Concatenate all modality projections
        x = torch.cat(projections, dim=-1)  # (B, T, actual_hidden)
        x = self.dim_fix(x)                 # (B, T, hidden_dim)

        # Get RoPE frequencies
        cos, sin = self.rope(x, seq_dim=1)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)

        return x  # (B, T, hidden_dim)


# =============================================================================
# DiT Block with AdaLN-Zero + Cross-Attention
# =============================================================================

class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero — DiT-style conditioning.

    Takes a conditioning vector c and produces 6 modulation params:
    (γ₁, β₁, α₁, γ₂, β₂, α₂) for pre-attention and pre-FFN LayerNorms.
    All initialized to zero so the block starts as identity.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim, bias=True),
        )
        # Small-init for faster convergence (not zero-init which blocks learning)
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c: torch.Tensor) -> tuple:
        """
        Args:
            c: (B, cond_dim) or (B, T, cond_dim) — conditioning vector.
        Returns:
            (γ₁, β₁, α₁, γ₂, β₂, α₂), each (B, 1, hidden_dim) or (B, T, hidden_dim)
        """
        params = self.mlp(c)  # (B, 6*H) or (B, T, 6*H)
        if params.ndim == 2:
            params = params.unsqueeze(1)  # (B, 1, 6*H) for broadcast
        return params.chunk(6, dim=-1)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN-Zero + optional Cross-Attention.

    Architecture:
        x = α₁ · SelfAttention(γ₁ · LN(x) + β₁) + x
        x = CrossAttention(LN(x), cond_seq) + x     [if cond_seq provided]
        x = α₂ · FFN(γ₂ · LN(x) + β₂) + x
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # AdaLN-Zero modulation
        self.adaln = AdaLNZero(hidden_dim, cond_dim)

        # Self-attention with RoPE
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = dropout

        # Cross-attention (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            cross_dim = cross_attention_dim or hidden_dim
            self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                kdim=cross_dim,
                vdim=cross_dim,
                dropout=dropout,
                batch_first=True,
            )

        # FFN
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,              # (B, T, H)
        c: torch.Tensor,              # (B, cond_dim) — timestep + global cond
        cos: torch.Tensor = None,     # (1, 1, T, head_dim) — RoPE frequencies
        sin: torch.Tensor = None,     # (1, 1, T, head_dim) — RoPE frequencies
        cond_seq: Optional[torch.Tensor] = None,  # (B, T_cond, H_cond) — for cross-attn
    ) -> torch.Tensor:
        # AdaLN-Zero params
        γ1, β1, α1, γ2, β2, α2 = self.adaln(c)  # each (B, 1, H)

        # 1. Self-Attention with AdaLN modulation and RoPE
        h = self.norm1(x)
        h = (1 + γ1) * h + β1
        
        B, T, C = h.shape
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, n_heads, T, head_dim)
        
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
        attn = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_drop if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(B, T, C)
        h = self.proj(attn)
        x = x + α1 * h

        # 2. Cross-Attention (condition sequence)
        if self.use_cross_attention and cond_seq is not None:
            h = self.norm_cross(x)
            h, _ = self.cross_attn(query=h, key=cond_seq, value=cond_seq)
            x = x + h

        # 3. FFN with AdaLN modulation
        h = self.norm2(x)
        h = (1 + γ2) * h + β2
        h = self.ffn(h)
        x = x + α2 * h

        return x


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar timesteps in [0, 1].
        Returns:
            emb: (B, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


# =============================================================================
# DiT Velocity Network (CFM Decoder)
# =============================================================================

class DiTVelocityNet(nn.Module):
    """DiT-based velocity estimator for Conditional Flow Matching.

    Replaces the U-Net decoder with a stack of DiT blocks.
    Input: noisy latent z_t concatenated with conditioning info.
    Output: predicted velocity v_θ(z_t, t, cond).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        cond_dim: int = 768,
        n_blocks: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection: latent z_t → hidden
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Mu (conditioning) projection: mu in latent space → hidden
        self.mu_proj = nn.Linear(latent_dim, hidden_dim)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)

        # Global condition projection (mean-pooled cond_seq → cond vector)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Total conditioning dim for AdaLN = timestep + global_cond
        adaln_cond_dim = hidden_dim  # combined via addition

        # RoPE embedding (no more absolute pos_embed)
        self.rope = RotaryEmbedding(dim=hidden_dim // n_heads)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                cond_dim=adaln_cond_dim,
                n_heads=n_heads,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
                cross_attention_dim=cond_dim,
            )
            for _ in range(n_blocks)
        ])

        # Output projection: hidden → latent_dim (velocity)
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        # Small init for stable start (NOT zero-init weight, which blocks all gradients!)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,       # (B, latent_dim, T) — noisy latent z_t
        mask: torch.Tensor,     # (B, 1, T) — validity mask
        mu: torch.Tensor,       # (B, latent_dim, T) — NOT used directly (kept for API compat)
        t: torch.Tensor,        # (B,) — flow matching timestep
        spks=None,              # unused, for API compat
        cond=None,              # unused, for API compat
        cond_seq: torch.Tensor = None,  # (B, T_cond, cond_dim) — from encoder
    ) -> torch.Tensor:
        """Predict velocity field.

        Returns:
            velocity: (B, latent_dim, T)
        """
        B, _, T = x.shape

        # Transpose to (B, T, latent_dim)
        x = x.transpose(1, 2)          # (B, T, D)
        mu_T = mu.transpose(1, 2)      # (B, T, D)
        x = self.input_proj(x) + self.mu_proj(mu_T)  # (B, T, H) — add conditioning

        # Get RoPE frequencies
        cos, sin = self.rope(x, seq_dim=1)

        # Conditioning: timestep + per-timestep condition
        t_emb = self.time_embed(t)     # (B, H)
        if cond_seq is not None:
            # Per-timestep conditioning (preserves temporal info)
            c_per_t = self.cond_proj(cond_seq)   # (B, T_cond, H)
            c = t_emb.unsqueeze(1) + c_per_t     # (B, T_cond, H)
        else:
            c = t_emb

        # DiT blocks
        for block in self.blocks:
            x = block(x, c, cos=cos, sin=sin, cond_seq=cond_seq)

        # Output
        x = self.final_norm(x)
        x = self.output_proj(x)        # (B, T, latent_dim)

        # Transpose back to (B, latent_dim, T) and apply mask
        x = x.transpose(1, 2)         # (B, D, T)
        return x * mask


# =============================================================================
# CFM wrapper for DiT
# =============================================================================

class DiTCFM(nn.Module):
    """CFM with DiT velocity estimator.

    Drop-in replacement for the U-Net-based CFM module.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        dit_params: dict,
        cfm_params,
        n_spks: int = 1,
        spk_emb_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sigma_min = getattr(cfm_params, 'sigma_min', 1e-4)

        self.estimator = DiTVelocityNet(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            **dit_params,
        )

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,        # (B, latent_dim, T) — conditioning (for API compat)
        mask: torch.Tensor,       # (B, 1, T)
        n_timesteps: int = 20,
        temperature: float = 1.0,
        cond_seq: torch.Tensor = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """ODE sampling (Euler solver) with optional Classifier-Free Guidance."""
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        t, dt = t_span[0], t_span[1] - t_span[0]
        x = z
        
        B = mu.shape[0]

        for step in range(1, len(t_span)):
            if guidance_scale != 1.0:
                # Batched CFG: [cond, uncond]
                x_both = torch.cat([x, x], dim=0)
                mask_both = torch.cat([mask, mask], dim=0)
                mu_both = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                t_both = t.expand(B * 2)
                if cond_seq is not None:
                    cond_seq_both = torch.cat([cond_seq, torch.zeros_like(cond_seq)], dim=0)
                else:
                    cond_seq_both = None
                
                v_both = self.estimator(x_both, mask_both, mu_both, t_both, cond_seq=cond_seq_both)
                v_cond, v_uncond = v_both[:B], v_both[B:]
                dphi_dt = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                dphi_dt = self.estimator(x, mask, mu, t.expand(B), cond_seq=cond_seq)
                
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x

    def compute_loss(
        self,
        x1: torch.Tensor,        # (B, latent_dim, T) — target
        mask: torch.Tensor,       # (B, 1, T)
        mu: torch.Tensor,         # (B, latent_dim, T) — unused but kept for API
        cond_seq: torch.Tensor = None,
        noise_scale: float = 0.0,  # Pure OT-CFM (no Brownian Bridge stochasticity)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute OT-CFM loss with Brownian Bridge stochasticity."""
        b, _, T = x1.shape

        # Random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)

        # Sample noise
        z = torch.randn_like(x1)

        # Interpolate: y(t) = (1 - (1-σ_min)t)z + t·x1
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        
        # Add Brownian noise scaled by t(1-t) to make the path stochastic
        if noise_scale > 0:
            bridge_std = torch.sqrt(t * (1 - t) + 1e-8)
            epsilon = torch.randn_like(x1)
            y = y + noise_scale * bridge_std * epsilon
            
            # Derivative of the noise term wrt t: c * (1 - 2t) / (2 * sqrt(t(1-t)))
            # Clamp denominator to avoid division by zero near t=0 or t=1
            denom = 2 * torch.clamp(bridge_std, min=1e-5)
            noise_deriv = noise_scale * ((1 - 2 * t) / denom) * epsilon
        else:
            noise_deriv = 0.0
            
        # Target velocity: u = dy/dt
        # Base linear derivative: x1 - (1-σ_min)z
        u = x1 - (1 - self.sigma_min) * z + noise_deriv
        u = u * mask

        # Predict velocity
        v_pred = self.estimator(y, mask, mu, t.squeeze(), cond_seq=cond_seq)

        # Masked MSE loss
        loss = F.mse_loss(v_pred, u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y


# =============================================================================
# Top-level BrainFlowCFM v2
# =============================================================================

class BrainFlowCFMv2(nn.Module):
    """BrainFlow CFM v2 with TRIBE-style encoder and DiT/U-Net decoder.

    Supports both DiT and U-Net decoder backends for ablation study.

    Args:
        modality_dims: dict mapping modality name → feature dimension.
        latent_dim: VAE latent dimension (must match VAE).
        encoder_params: kwargs for MultiModalConditionEncoder.
        decoder_type: "dit" or "unet".
        decoder_params: kwargs for decoder.
        cfm_params: CFM hyperparameters (solver, sigma_min).
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        latent_dim: int,
        encoder_params: dict,
        decoder_type: str = "dit",
        decoder_params: dict = None,
        cfm_params: dict = None,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_type = decoder_type
        self.cfg_drop_prob = cfg_drop_prob

        # Encoder
        self.encoder = MultiModalConditionEncoder(
            modality_dims=modality_dims,
            **encoder_params,
        )
        cond_dim = encoder_params.get("hidden_dim", 768)

        # CFM params object
        class CFMParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        cfm_p = CFMParams(**(cfm_params or {"solver": "euler", "sigma_min": 1e-4}))

        # Conditioning → latent space projection (used by both DiT and U-Net)
        self.cond_to_mu = nn.Linear(cond_dim, latent_dim)

        # Decoder
        if decoder_type == "dit":
            self.decoder = DiTCFM(
                latent_dim=latent_dim,
                cond_dim=cond_dim,
                dit_params=decoder_params or {},
                cfm_params=cfm_p,
            )
        else:
            # U-Net decoder (original Matcha-TTS style)
            self.decoder = _get_cfm_class()(
                in_channels=2 * latent_dim,  # concat [z_t, mu_cond]
                out_channel=latent_dim,
                cfm_params=cfm_p,
                decoder_params=decoder_params or {},
                n_spks=1,
                spk_emb_dim=0,
            )

    def _encode_condition(
        self,
        modality_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode multimodal features → conditioning sequence.

        Returns:
            cond_seq: (B, T, cond_dim)
        """
        return self.encoder(modality_features)

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod: (B, T, D_mod)}
        latent: torch.Tensor,                          # (B, T, Z)
        mask: torch.Tensor = None,                     # (B, 1, T)
    ) -> torch.Tensor:
        """Compute CFM loss.

        Returns:
            loss: scalar CFM loss.
        """
        B, T, _ = latent.shape

        if mask is None:
            mask = torch.ones(B, 1, T, device=latent.device, dtype=latent.dtype)

        # 1. Encode conditioning
        cond_seq = self._encode_condition(modality_features)  # (B, T, H_cond)

        # CFG dropout: zero out entire conditioning vector for random samples
        cfg_keep = None
        if self.training and self.cfg_drop_prob > 0.0:
            # shape (B, 1, 1) broadcasts over T and cond_dim
            cfg_keep = (torch.rand(B, 1, 1, device=cond_seq.device) > self.cfg_drop_prob).float()
            cond_seq = cond_seq * cfg_keep

        # 2. Project conditioning → latent space
        mu_cond = self.cond_to_mu(cond_seq)  # (B, T, Z)
        # Also zero mu_cond for dropped samples (bypass projection bias)
        if cfg_keep is not None:
            mu_cond = mu_cond * cfg_keep
        mu_cond = mu_cond.transpose(1, 2)    # (B, Z, T)

        # 3. CFM loss
        latent_T = latent.transpose(1, 2)  # (B, Z, T)

        if self.decoder_type == "dit":
            diff_loss, _ = self.decoder.compute_loss(
                x1=latent_T, mask=mask, mu=mu_cond, cond_seq=cond_seq,
            )
        else:
            diff_loss, _ = self.decoder.compute_loss(x1=latent_T, mask=mask, mu=mu_cond)

        return diff_loss

    @torch.inference_mode()
    def synthesise(
        self,
        modality_features: dict[str, torch.Tensor],
        n_timesteps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate fMRI latents from multimodal features.

        Args:
            modality_features: dict {mod_name: (B, T, D_mod)}.
            n_timesteps: ODE solver steps.
            temperature: Initial noise variance.
            guidance_scale: CFG scale (>1.0 pushes away from unconditional).

        Returns:
            latent: (B, T, Z) — generated latents.
        """
        ref = next(iter(modality_features.values()))
        B, T = ref.shape[0], ref.shape[1]
        device = ref.device
        mask = torch.ones(B, 1, T, device=device, dtype=ref.dtype)

        cond_seq = self._encode_condition(modality_features)  # (B, T, H_cond)

        if self.decoder_type == "dit":
            mu_cond = self.cond_to_mu(cond_seq).transpose(1, 2)  # (B, Z, T)
            z_1 = self.decoder(
                mu=mu_cond, mask=mask,
                n_timesteps=n_timesteps, temperature=temperature,
                cond_seq=cond_seq, guidance_scale=guidance_scale,
            )
        else:
            mu_cond = self.cond_to_mu(cond_seq).transpose(1, 2)
            z_1 = self.decoder(
                mu=mu_cond, mask=mask,
                n_timesteps=n_timesteps, temperature=temperature,
            )

        return z_1.transpose(1, 2)  # (B, T, Z)
