"""BrainFlow TTS — Matcha-TTS architecture adapted for multimodal → fMRI prediction.

Key differences from original Matcha-TTS:
  - Encoder: MultiModalConditionEncoder (TRIBE-style) instead of TextEncoder
  - No duration predictor, no MAS alignment (temporal alignment via sliding window)
  - Target: VAE latent (B, Z, T) instead of mel-spectrogram (B, n_feats, T)
  - CFG (Classifier-Free Guidance) support
  - Auxiliary fMRI regression head

Architecture:
  multimodal features → TRIBE Encoder → cond_to_mu → mu_cond (B, Z, T)
  CFM Decoder (U-Net 1D): concat[z_t, mu_cond] → velocity v(t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.brainflow.matcha.flow_matching import CFM
from src.models.brainflow.brain_flow_v2 import MultiModalConditionEncoder


class BrainFlowTTS(nn.Module):
    """Matcha-TTS architecture adapted for multimodal → fMRI latent prediction.

    Uses the original Matcha-TTS U-Net 1D decoder (with Transformer/Conformer blocks)
    as the velocity estimator, paired with a TRIBE-style multimodal encoder.

    Args:
        modality_dims: dict mapping modality name → feature dimension.
        latent_dim: VAE latent dimension (e.g. 64).
        encoder_params: kwargs for MultiModalConditionEncoder.
        decoder_params: kwargs for Matcha Decoder (U-Net 1D).
        cfm_params: CFM hyperparameters (solver, sigma_min).
        cfg_drop_prob: Probability of dropping conditioning for CFG.
        n_voxels: Number of fMRI voxels for auxiliary regression.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        latent_dim: int,
        encoder_params: dict,
        decoder_params: dict = None,
        cfm_params: dict = None,
        cfg_drop_prob: float = 0.1,
        n_voxels: int = 1000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cfg_drop_prob = cfg_drop_prob

        # === Encoder (TRIBE-style multimodal) ===
        self.encoder = MultiModalConditionEncoder(
            modality_dims=modality_dims,
            **encoder_params,
        )
        cond_dim = encoder_params.get("hidden_dim", 768)

        # Conditioning → latent space projection
        self.cond_to_mu = nn.Linear(cond_dim, latent_dim)

        # === CFM Decoder (Matcha-TTS U-Net 1D) ===
        class CFMParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        cfm_p = CFMParams(**(cfm_params or {"solver": "euler", "sigma_min": 1e-4}))

        self.decoder = CFM(
            in_channels=2 * latent_dim,   # concat [z_t, mu_cond]
            out_channel=latent_dim,
            cfm_params=cfm_p,
            decoder_params=decoder_params or {},
            n_spks=1,
            spk_emb_dim=0,
        )

        # Direct fMRI regression head (auxiliary, TRIBE-style)
        self.fmri_head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, n_voxels),
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
        fmri_target: torch.Tensor = None,              # (B, T, V) optional
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute CFM loss, prior loss, and optional fMRI regression loss.

        Returns:
            diff_loss, prior_loss, reg_loss
        """
        B, T, _ = latent.shape

        if mask is None:
            mask = torch.ones(B, 1, T, device=latent.device, dtype=latent.dtype)

        # 1. Encode conditioning
        cond_seq = self._encode_condition(modality_features)  # (B, T_feat, H_cond)

        # CFG dropout
        cfg_keep = None
        if self.training and self.cfg_drop_prob > 0.0:
            cfg_keep = (torch.rand(B, 1, 1, device=cond_seq.device) > self.cfg_drop_prob).float()
            cond_seq = cond_seq * cfg_keep

        # 2. Project aligned conditioning → latent space
        cond_aligned = cond_seq[:, -T:, :]  # (B, T_lat, H_cond)
        mu_cond = self.cond_to_mu(cond_aligned)  # (B, T_lat, Z)

        # Prior loss (Matcha-TTS style)
        mask_T = mask.transpose(1, 2)  # (B, T, 1)
        prior_loss = torch.sum(
            0.5 * ((latent - mu_cond) ** 2) * mask_T
        ) / (torch.sum(mask_T) * self.latent_dim)

        # 3. Direct fMRI regression (auxiliary)
        reg_loss = torch.tensor(0.0, device=latent.device)
        if fmri_target is not None:
            fmri_pred = self.fmri_head(cond_aligned)
            reg_loss = F.mse_loss(fmri_pred * mask_T, fmri_target * mask_T)

        # Zero mu_cond for dropped samples
        if cfg_keep is not None:
            mu_cond = mu_cond * cfg_keep
        mu_cond = mu_cond.transpose(1, 2)  # (B, Z, T)

        # 4. CFM loss (U-Net decoder)
        latent_T = latent.transpose(1, 2)  # (B, Z, T)
        diff_loss, _ = self.decoder.compute_loss(x1=latent_T, mask=mask, mu=mu_cond)

        return diff_loss, prior_loss, reg_loss

    @torch.inference_mode()
    def synthesise(
        self,
        modality_features: dict[str, torch.Tensor],
        n_timesteps: int = 20,
        temperature: float = 0.667,
        guidance_scale: float = 1.0,
        n_output_trs: int = None,
    ) -> torch.Tensor:
        """Generate fMRI latents from multimodal features.

        Args:
            modality_features: dict {mod_name: (B, T_feat, D_mod)}.
            n_timesteps: ODE solver steps.
            temperature: Noise scaling (Matcha uses 0.667 by default).
            guidance_scale: CFG scale (>1.0 pushes toward conditional).
            n_output_trs: Number of output TRs.

        Returns:
            latent: (B, T_lat, Z)
        """
        ref = next(iter(modality_features.values()))
        B, T_feat = ref.shape[0], ref.shape[1]
        device = ref.device

        T_lat = n_output_trs if n_output_trs is not None else T_feat
        mask = torch.ones(B, 1, T_lat, device=device, dtype=ref.dtype)

        cond_seq = self._encode_condition(modality_features)
        cond_aligned = cond_seq[:, -T_lat:, :]
        mu_cond = self.cond_to_mu(cond_aligned).transpose(1, 2)  # (B, Z, T_lat)

        if guidance_scale != 1.0:
            # CFG: generate both conditional and unconditional, then blend
            z_cond = self.decoder(mu_cond, mask, n_timesteps, temperature)
            z_uncond = self.decoder(
                torch.zeros_like(mu_cond), mask, n_timesteps, temperature
            )
            z_1 = z_uncond + guidance_scale * (z_cond - z_uncond)
        else:
            z_1 = self.decoder(mu_cond, mask, n_timesteps, temperature)

        return z_1.transpose(1, 2)  # (B, T_lat, Z)
