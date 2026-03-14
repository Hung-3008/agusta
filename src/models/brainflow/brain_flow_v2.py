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
        # Each modality projects to hidden_dim // N so cat gives hidden_dim
        per_mod_dim = hidden_dim // self.n_modalities
        self.projectors = nn.ModuleDict()
        for mod_name, raw_dim in modality_dims.items():
            self.projectors[mod_name] = nn.Sequential(
                nn.Linear(raw_dim, per_mod_dim),
                nn.LayerNorm(per_mod_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(per_mod_dim, per_mod_dim),
                nn.LayerNorm(per_mod_dim),
            )

        # Actual hidden dim after concat (handle rounding)
        actual_hidden = per_mod_dim * self.n_modalities

        # Optional linear to map to exact hidden_dim if rounding occurred
        if actual_hidden != hidden_dim:
            self.dim_fix = nn.Linear(actual_hidden, hidden_dim)
        else:
            self.dim_fix = nn.Identity()

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder for temporal context
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
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

        # Add positional embedding
        x = x + self.pos_embed[:, :T, :]

        # Transformer encoder
        x = self.transformer(x)              # (B, T, hidden_dim)
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
        # Zero-init so block starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c: torch.Tensor) -> tuple:
        """
        Args:
            c: (B, cond_dim) — conditioning vector.
        Returns:
            (γ₁, β₁, α₁, γ₂, β₂, α₂), each (B, 1, hidden_dim)
        """
        params = self.mlp(c).unsqueeze(1)  # (B, 1, 6*H)
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

        # AdaLN-Zero modulation
        self.adaln = AdaLNZero(hidden_dim, cond_dim)

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

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
        cond_seq: Optional[torch.Tensor] = None,  # (B, T_cond, H_cond) — for cross-attn
    ) -> torch.Tensor:
        # AdaLN-Zero params
        γ1, β1, α1, γ2, β2, α2 = self.adaln(c)  # each (B, 1, H)

        # 1. Self-Attention with AdaLN modulation
        h = self.norm1(x)
        h = γ1 * h + β1
        h, _ = self.self_attn(h, h, h)
        x = x + α1 * h

        # 2. Cross-Attention (condition sequence)
        if self.use_cross_attention and cond_seq is not None:
            h = self.norm_cross(x)
            h, _ = self.cross_attn(query=h, key=cond_seq, value=cond_seq)
            x = x + h

        # 3. FFN with AdaLN modulation
        h = self.norm2(x)
        h = γ2 * h + β2
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

        # Positional embedding for latent sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

        # Positional embedding
        x = x + self.pos_embed[:, :T, :]

        # Conditioning: timestep + global condition
        t_emb = self.time_embed(t)     # (B, H)
        if cond_seq is not None:
            # Global condition = mean pool of conditioning sequence
            c_global = cond_seq.mean(dim=1)  # (B, cond_dim)
            c_global = self.cond_proj(c_global)  # (B, H)
            c = t_emb + c_global       # (B, H)
        else:
            c = t_emb

        # DiT blocks
        for block in self.blocks:
            x = block(x, c, cond_seq=cond_seq)

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
    ) -> torch.Tensor:
        """ODE sampling (Euler solver)."""
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        t, dt = t_span[0], t_span[1] - t_span[0]
        x = z

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, cond_seq=cond_seq)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute OT-CFM loss."""
        b, _, T = x1.shape

        # Random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)

        # Sample noise
        z = torch.randn_like(x1)

        # Interpolate: y(t) = (1 - (1-σ_min)t)z + t·x1
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        # Target velocity: u = x1 - (1-σ_min)z
        u = x1 - (1 - self.sigma_min) * z

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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_type = decoder_type

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

        # 2. Project conditioning → latent space
        mu_cond = self.cond_to_mu(cond_seq)  # (B, T, Z)
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
    ) -> torch.Tensor:
        """Generate fMRI latents from multimodal features.

        Args:
            modality_features: dict {mod_name: (B, T, D_mod)}.
            n_timesteps: ODE solver steps.
            temperature: Initial noise variance.

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
                cond_seq=cond_seq,
            )
        else:
            mu_cond = self.cond_to_mu(cond_seq).transpose(1, 2)
            z_1 = self.decoder(
                mu=mu_cond, mask=mask,
                n_timesteps=n_timesteps, temperature=temperature,
            )

        return z_1.transpose(1, 2)  # (B, T, Z)
