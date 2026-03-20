"""Multimodal Deterministic Autoencoder (DAE).

Simplified version of MoEVAE: no KL loss, no sampling, deterministic encoding.
Produces clean, high-variance latent representations for downstream conditioning.

Architecture:
    Modality features (pre-pooled) → ModalityEncoder → h_mod
    All h_mod → MoE Fusion → z (deterministic)
    z → ModalityDecoder → reconstructed features (per modality)

Training: reconstruction loss only (MSE per modality, averaged).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Per-modality Encoder (deterministic — no mu/logvar split)
# =============================================================================

class ModalityEncoder(nn.Module):
    """Maps pooled modality features → latent vector (deterministic)."""

    def __init__(self, input_dim: int, hidden_dim: int = 1024,
                 latent_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, Z)


# =============================================================================
# MoE Fusion (deterministic — outputs z directly)
# =============================================================================

class MoEFusion(nn.Module):
    """Mixture-of-Experts fusion: concat per-modality latents → fused z."""

    def __init__(self, total_input_dim: int, latent_dim: int = 512,
                 n_experts: int = 4, expert_hidden: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.latent_dim = latent_dim

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(total_input_dim, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, n_experts),
        )

        # Experts → output z directly (no mu/logvar)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_input_dim, expert_hidden),
                nn.LayerNorm(expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, expert_hidden),
                nn.LayerNorm(expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, latent_dim),
            )
            for _ in range(n_experts)
        ])

    def forward(self, modality_latents: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(modality_latents, dim=-1)  # (B, K*Z)
        gate_weights = F.softmax(self.gate(x), dim=-1)  # (B, E)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, E, Z)
        z = (gate_weights.unsqueeze(-1) * expert_outputs).sum(1)  # (B, Z)
        return z


# =============================================================================
# Per-modality Decoder
# =============================================================================

class ModalityDecoder(nn.Module):
    """Reconstructs modality features from latent z."""

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 1024,
                 output_dim: int = 1408, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =============================================================================
# Top-level Multimodal DAE
# =============================================================================

class MultimodalDAE(nn.Module):
    """Multimodal Deterministic Autoencoder.

    No KL, no sampling — purely deterministic encode → decode.
    """

    def __init__(
        self,
        modality_configs: dict,
        latent_dim: int = 512,
        encoder_hidden: int = 1024,
        n_experts: int = 4,
        expert_hidden: int = 1024,
        decoder_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_configs = modality_configs
        self.latent_dim = latent_dim
        self.modality_names = list(modality_configs.keys())
        self.n_modalities = len(self.modality_names)

        # Per-modality encoders
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(
                input_dim=cfg['dim'],
                hidden_dim=encoder_hidden,
                latent_dim=latent_dim,
                dropout=dropout,
            )
            for name, cfg in modality_configs.items()
        })

        # MoE fusion
        total_dim = self.n_modalities * latent_dim
        self.moe_fusion = MoEFusion(
            total_input_dim=total_dim,
            latent_dim=latent_dim,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            dropout=dropout,
        )

        # Per-modality decoders
        self.decoders = nn.ModuleDict({
            name: ModalityDecoder(
                latent_dim=latent_dim,
                hidden_dim=decoder_hidden,
                output_dim=cfg['dim'],
                dropout=dropout,
            )
            for name, cfg in modality_configs.items()
        })

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MultimodalDAE] {self.n_modalities} modalities, "
              f"latent_dim={latent_dim}, n_experts={n_experts}, "
              f"total params: {n_params:,}")

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Encode all modalities → fused z (deterministic).

        Returns:
            z: (B, Z) deterministic latent.
            targets: dict of {mod_name: (B, D_mod)} input features for decoder targets.
        """
        modality_latents = []
        targets = {}

        for mod_name in self.modality_names:
            if mod_name in batch:
                x = batch[mod_name]  # (B, D_mod) — pre-pooled
                targets[mod_name] = x
                h = self.encoders[mod_name](x)  # (B, Z)
                modality_latents.append(h)
            else:
                B = next(iter(batch.values())).shape[0]
                device = next(iter(batch.values())).device
                modality_latents.append(
                    torch.zeros(B, self.latent_dim, device=device)
                )

        z = self.moe_fusion(modality_latents)  # (B, Z)
        return z, targets

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: self.decoders[name](z) for name in self.modality_names}

    def forward(self, batch: dict[str, torch.Tensor]):
        z, targets = self.encode(batch)
        recons = self.decode(z)
        return recons, z, targets

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reconstruction-only loss (MSE per modality, averaged)."""
        recons, z, targets = self.forward(batch)

        recon_losses = {}
        total_recon = 0.0
        n_active = 0

        for mod_name in self.modality_names:
            if mod_name in targets:
                target = targets[mod_name].detach()
                recon_loss = F.mse_loss(recons[mod_name], target)
                recon_losses[f"recon_{mod_name}"] = recon_loss
                total_recon = total_recon + recon_loss
                n_active += 1

        if n_active > 0:
            total_recon = total_recon / n_active

        result = {"loss": total_recon, "recon": total_recon}
        result.update(recon_losses)
        return result

    @torch.no_grad()
    def get_latent(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z, _ = self.encode(batch)
        return z
