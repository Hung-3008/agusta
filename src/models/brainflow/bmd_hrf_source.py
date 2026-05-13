"""BMD-specific Source Generator for Conditional Source Flow Matching (CSFM).

Unlike AECNN_HRF_Source which uses Conv1d over temporal context (T >> 1),
this module operates on a single pooled context vector (T=1) from event-related
BMD trials. Instead of simulating hemodynamic convolution, it uses a deeper MLP
to transform the multimodal context into a conditional base distribution
(mu_phi, sigma_phi) for the flow matching source.

Design rationale:
    - BMD is event-related: one 3s clip → one beta vector → no temporal axis
    - Conv1d HRF filter requires T > kernel_size, which is impossible for T=1
    - The MLP-based approach directly maps pooled context to latent mu/sigma
    - Bottleneck architecture acts as a "neural event" compressor (analogous
      to the Conv1d → Sigmoid binarization in the temporal version)
"""

import torch
import torch.nn as nn


class BMD_HRF_Source(nn.Module):
    """Condition-dependent Source Generator for event-related (BMD) data.

    Replaces the temporal Conv1d HRF pipeline with an MLP that maps
    a single pooled context vector to (mu_phi, sigma_phi).

    Architecture:
        context_pooled (B, D) → Encoder MLP → bottleneck (B, latent_dim)
                              → mu_head → mu_phi (B, latent_dim)
                              → sigma_head → sigma_phi (B, 1)

    The bottleneck with Sigmoid mirrors the "neural event extractor" from
    AECNN_HRF_Source, acting as a soft-gating mechanism that selects which
    latent dimensions carry signal before projecting to the base distribution.
    """

    def __init__(self, context_dim: int, latent_dim: int, bottleneck_dim: int = 512):
        super().__init__()

        # Step 1: Neural Event Extractor (MLP analog of Conv1d + Sigmoid)
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )

        # Step 2: Soft gating — analogous to Sigmoid binarization in AECNN
        self.gate = nn.Sequential(
            nn.Linear(bottleneck_dim, latent_dim),
            nn.GELU(),
        )

        # Step 3: Mu head — projects gated features to latent base distribution
        self.mu_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Step 4: Sigma predictor (scalar variance)
        self.sigma_net = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Softplus(),  # Ensure strictly positive variance
        )

    def forward(self, context_pooled: torch.Tensor):
        """Generate conditional source distribution parameters.

        Args:
            context_pooled: (B, D) pooled context from encoded features.

        Returns:
            mu_phi:    (B, latent_dim) mean of the base distribution.
            sigma_phi: (B, 1) std of the base distribution.
        """
        h = self.encoder(context_pooled)       # (B, bottleneck_dim)
        gated = self.gate(h)                   # (B, latent_dim) — soft neural events
        mu_phi = self.mu_head(gated)           # (B, latent_dim)
        sigma_phi = self.sigma_net(context_pooled)  # (B, 1)

        return mu_phi, sigma_phi
