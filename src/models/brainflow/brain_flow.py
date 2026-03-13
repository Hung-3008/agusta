"""BrainFlow CFM model (Stage 2).

Learns to generate fMRI VAE latents from multimodal PCA features using
Conditional Flow Matching (CFM), based on Matcha-TTS architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = self.input_proj(features)               # (B, T, H)
        x = self.transformer(x)                     # (B, T, H)
        mu_cond = self.output_proj(x)                 # (B, T, Z)
        
        # Transpose to match UNet-1D channel format: (B, C, T)
        return mu_cond.transpose(1, 2)


class BrainFlowCFM(nn.Module):
    """
    Top-level Conditional Flow Matching model for fMRI generation.
    """
    def __init__(
        self,
        feat_dim: int,
        latent_dim: int,
        encoder_params: dict,
        decoder_params: dict,
        cfm_params: dict,
    ):
        super().__init__()
        self.latent_dim = latent_dim

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
            in_channels=2 * latent_dim,  # Concat z_t and mu_cond
            out_channel=latent_dim,
            cfm_params=cfm_p,
            decoder_params=decoder_params,
            n_spks=1,
            spk_emb_dim=0, # No speaker embedding matching Matcha's n_spks=1 behavior
        )

    def forward(
        self,
        features: torch.Tensor,     # (B, T, D_feat)
        latent: torch.Tensor,       # (B, T, 256)
        mask: torch.Tensor = None,  # (B, 1, T)
    ) -> torch.Tensor:
        """
        Computes the CFM diffusion loss.
        """
        B, T, _ = features.shape

        if mask is None:
            mask = torch.ones(B, 1, T, device=features.device, dtype=features.dtype)

        # 1. Encode conditioning features
        mu_cond = self.encoder(features)            # (B, 256, T)

        # 2. Transpose latent to match channel format
        latent_T = latent.transpose(1, 2)           # (B, 256, T)

        # 3. Compute CFM diff_loss
        diff_loss, _ = self.decoder.compute_loss(
            x1=latent_T,
            mask=mask,
            mu=mu_cond,
        )

        return diff_loss

    @torch.inference_mode()
    def synthesise(
        self,
        features: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generates fMRI latents from multimodal features.
        
        Args:
            features: (B, T, D_feat)
            n_timesteps: Number of steps for Euler ODE solver
            temperature: Variance of initial Gaussian noise
            
        Returns:
            latent: (B, T, 256)
        """
        B, T, _ = features.shape
        mask = torch.ones(B, 1, T, device=features.device, dtype=features.dtype)

        mu_cond = self.encoder(features)            # (B, 256, T)

        # Generate sample tracing the probability flow
        z_1 = self.decoder(
            mu=mu_cond,
            mask=mask,
            n_timesteps=n_timesteps,
            temperature=temperature,
        )
        
        # Return to (B, T, Z) format
        return z_1.transpose(1, 2)
