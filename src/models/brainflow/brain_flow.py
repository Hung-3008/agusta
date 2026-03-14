"""BrainFlow CFM model (Stage 2).

Learns to generate fMRI VAE latents from multimodal PCA features using
Conditional Flow Matching (CFM), based on Matcha-TTS architecture.
"""

import torch
import torch.nn as nn

from src.models.brainflow.components.flow_matching import CFM


class ConditionEncoder(nn.Module):
    """
    Encodes PCA features into conditioning vectors for the flow matching decoder.
    Input: (B, T, D_feat) -> Output: mu_cond (B, D_latent, T)
    """
    def __init__(
        self,
        feat_dim: int,
        latent_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        # Project PCA features to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Transformer to exchange temporal context within the window
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Output projection to latent_dim
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D_feat)
        Returns:
            mu_cond: (B, D_latent, T)
        """
        x = self.input_proj(features)       # (B, T, H)
        x = self.transformer(x)             # (B, T, H)
        mu_cond = self.output_proj(x)       # (B, T, Z)
        return mu_cond.transpose(1, 2)      # (B, Z, T)


class BrainFlowCFM(nn.Module):
    """
    Top-level Conditional Flow Matching model for fMRI generation.
    Supports Classifier-Free Guidance (CFG) via conditioning dropout.

    During training, conditioning features are randomly zeroed-out with
    probability `cfg_drop_prob`, so the model also learns an unconditional
    distribution (null conditioning = zeros). During inference, the
    unconditional and conditional velocity predictions are blended via
    `guidance_scale` to sharpen the conditioning signal.
    """
    def __init__(
        self,
        feat_dim: int,
        latent_dim: int,
        encoder_params: dict,
        decoder_params: dict,
        cfm_params: dict,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cfg_drop_prob = cfg_drop_prob

        self.encoder = ConditionEncoder(
            feat_dim=feat_dim,
            latent_dim=latent_dim,
            **encoder_params,
        )

        # cfm_params as an object to match expected attribute access in BASECFM
        class CFMParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        cfm_p = CFMParams(**cfm_params)

        self.decoder = CFM(
            in_channels=2 * latent_dim,   # concat [z_t, mu_cond]
            out_channel=latent_dim,
            cfm_params=cfm_p,
            decoder_params=decoder_params,
            n_spks=1,
            spk_emb_dim=0,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,     # (B, T, D_feat)
        latent: torch.Tensor,       # (B, T, Z)
        mask: torch.Tensor = None,  # (B, 1, T)
    ) -> torch.Tensor:
        """Compute CFM loss with CFG dropout."""
        B, T, _ = features.shape

        if mask is None:
            mask = torch.ones(B, 1, T, device=features.device, dtype=features.dtype)

        # 1. Encode conditioning
        mu_cond = self.encoder(features)    # (B, Z, T)

        # 2. CFG dropout: zero out entire conditioning vector for random samples
        if self.training and self.cfg_drop_prob > 0.0:
            # shape (B, 1, 1) broadcasts over Z and T
            keep = (torch.rand(B, 1, 1, device=mu_cond.device) > self.cfg_drop_prob).float()
            mu_cond = mu_cond * keep

        # 3. CFM loss
        latent_T = latent.transpose(1, 2)   # (B, Z, T)
        diff_loss, _ = self.decoder.compute_loss(x1=latent_T, mask=mask, mu=mu_cond)
        return diff_loss

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def synthesise(
        self,
        features: torch.Tensor,
        n_timesteps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        """
        Generate fMRI latents from multimodal features.

        Args:
            features:       (B, T, D_feat)
            n_timesteps:    ODE solver steps
            temperature:    noise variance for initial sample
            guidance_scale: CFG strength.
                1.0  -> pure conditional (no CFG blending)
                >1.0 -> amplify conditional signal away from unconditional

        Returns:
            latent: (B, T, Z)
        """
        B, T, _ = features.shape
        mask = torch.ones(B, 1, T, device=features.device, dtype=features.dtype)

        mu_cond = self.encoder(features)    # (B, Z, T)

        if guidance_scale != 1.0:
            # Unconditional = zeros (same null embedding used during training)
            mu_uncond = torch.zeros_like(mu_cond)

            # Run both in a single batched ODE call for efficiency
            mu_both   = torch.cat([mu_cond, mu_uncond], dim=0)  # (2B, Z, T)
            mask_both = mask.repeat(2, 1, 1)                    # (2B, 1, T)

            z_both = self.decoder(
                mu=mu_both,
                mask=mask_both,
                n_timesteps=n_timesteps,
                temperature=temperature,
            )                                                    # (2B, Z, T)

            z_cond, z_uncond = z_both[:B], z_both[B:]
            # CFG formula: push away from unconditional baseline
            z_1 = z_uncond + guidance_scale * (z_cond - z_uncond)
        else:
            z_1 = self.decoder(
                mu=mu_cond,
                mask=mask,
                n_timesteps=n_timesteps,
                temperature=temperature,
            )

        return z_1.transpose(1, 2)   # (B, T, Z)
