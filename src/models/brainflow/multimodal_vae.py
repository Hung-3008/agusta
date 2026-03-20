"""Multimodal MoE-VAE — Encode multi-feature modalities → shared latent space.

Compresses 8 pretrained feature modalities (V-JEPA, Whisper, LLaMA, Qwen,
InternVL, etc.) into a shared 512-dim latent space using:
  - Per-modality Encoders: independent MLP → (μ, logσ²)
  - Mixture-of-Experts Fusion: gated expert network to fuse posteriors
  - Per-modality Decoders: reconstruct each modality from shared z
  - β-VAE with free_bits: prevents posterior collapse

Per-timestep encoding: each TR is encoded independently.

Architecture:
    Modality features (pre-pooled) → ModalityEncoder → (μ_mod, logσ²_mod)
    All posteriors → MoEFusion → (μ_fused, logσ²_fused)
    z ~ N(μ_fused, σ²_fused)
    z → ModalityDecoder → reconstructed features (per modality)

Usage:
    model = MultimodalMoEVAE(modality_configs, latent_dim=512)
    loss_dict = model.loss(batch, beta=0.001)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Weighted Layer Pooling
# =============================================================================

class WeightedLayerPool(nn.Module):
    """Learnable weighted sum across layers (ELMo-style).

    Given features from `n_layers` layers of a pretrained model,
    computes a softmax-normalised weighted sum to produce a single
    feature vector per timestep.

    Parameters
    ----------
    n_layers : int
        Number of layers to pool over.
    """

    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        # Initialise uniform weights (will become equal after softmax)
        self.layer_weights = nn.Parameter(torch.ones(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool across layers.

        Parameters
        ----------
        x : (n_layers, D) or (B, n_layers, D)
            Per-layer features.

        Returns
        -------
        pooled : (D,) or (B, D)
        """
        weights = F.softmax(self.layer_weights, dim=0)

        if x.dim() == 2:
            # (n_layers, D) → (D,)
            return (weights.unsqueeze(-1) * x).sum(0)
        elif x.dim() == 3:
            # (B, n_layers, D) → (B, D)
            return (weights.unsqueeze(0).unsqueeze(-1) * x).sum(1)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

    def get_weights(self) -> torch.Tensor:
        """Return normalised weights (for visualisation)."""
        return F.softmax(self.layer_weights, dim=0).detach()


# =============================================================================
# Per-modality Encoder
# =============================================================================

class ModalityEncoder(nn.Module):
    """Maps a single modality's pooled features → (μ, logσ²).

    Architecture: Linear → LayerNorm → GELU → Linear → LayerNorm → GELU → (μ, logσ²)

    Parameters
    ----------
    input_dim : int
        Dimension of the pooled modality feature.
    hidden_dim : int
        Hidden layer width.
    latent_dim : int
        Output latent dimension (μ and logσ² each have this size).
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self._logvar_clamp = 10.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, D_mod)

        Returns
        -------
        mu     : (B, Z)
        logvar : (B, Z)
        """
        h = self.trunk(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-self._logvar_clamp, self._logvar_clamp)
        return mu, logvar


# =============================================================================
# Mixture-of-Experts Fusion
# =============================================================================

class MoEFusion(nn.Module):
    """Mixture-of-Experts for fusing per-modality posteriors.

    Takes the concatenated means from all modality encoders and produces
    a fused (μ, logσ²) via a gated mixture of expert MLPs.

    Architecture:
        gate:   concat(μ_1, ..., μ_K) → Linear → softmax → (B, E)
        expert: concat(μ_1, ..., μ_K) → MLP → (μ_e, logσ²_e)  [per expert]
        fused:  Σ_e gate_e * (μ_e, logσ²_e)

    Parameters
    ----------
    total_input_dim : int
        Sum of all modality latent dims (= K * latent_dim if equal).
    latent_dim : int
        Output latent dimension.
    n_experts : int
        Number of expert MLPs.
    expert_hidden : int
        Hidden dim in each expert MLP.
    dropout : float
    """

    def __init__(
        self,
        total_input_dim: int,
        latent_dim: int = 512,
        n_experts: int = 4,
        expert_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.latent_dim = latent_dim

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(total_input_dim, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, n_experts),
        )

        # Expert networks
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
                nn.Linear(expert_hidden, latent_dim * 2),  # μ + logσ²
            )
            for _ in range(n_experts)
        ])

    def forward(
        self, modality_mus: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse modality posteriors via MoE.

        Parameters
        ----------
        modality_mus : list of (B, Z) tensors
            Mean vectors from each modality encoder.

        Returns
        -------
        mu_fused     : (B, Z)
        logvar_fused : (B, Z)
        """
        # Concatenate all modality means as input
        x = torch.cat(modality_mus, dim=-1)  # (B, K * Z)

        # Gate weights
        gate_logits = self.gate(x)  # (B, E)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, E)

        # Expert outputs
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, E, 2*Z)

        # Weighted sum
        weighted = (gate_weights.unsqueeze(-1) * expert_outputs).sum(1)  # (B, 2*Z)

        mu_fused = weighted[:, :self.latent_dim]
        logvar_fused = weighted[:, self.latent_dim:].clamp(-10.0, 10.0)

        return mu_fused, logvar_fused


# =============================================================================
# Per-modality Decoder
# =============================================================================

class ModalityDecoder(nn.Module):
    """Reconstructs a single modality's pooled features from latent z.

    Parameters
    ----------
    latent_dim : int
    hidden_dim : int
    output_dim : int
        Dimension of the modality feature to reconstruct.
    dropout : float
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 1408,
        dropout: float = 0.1,
    ):
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
        """
        Parameters
        ----------
        z : (B, Z)

        Returns
        -------
        recon : (B, D_mod)
        """
        return self.net(z)


# =============================================================================
# β annealing schedule
# =============================================================================

def beta_schedule(
    epoch: int,
    beta_start: float = 0.0001,
    beta_max: float = 0.01,
    warmup_epochs: int = 30,
) -> float:
    """Linear β-KL annealing: beta_start → beta_max over warmup_epochs."""
    if warmup_epochs <= 0:
        return beta_max
    progress = min(1.0, epoch / warmup_epochs)
    return beta_start + (beta_max - beta_start) * progress


# =============================================================================
# Top-level Multimodal MoE-VAE
# =============================================================================

class MultimodalMoEVAE(nn.Module):
    """Multimodal Mixture-of-Experts VAE.

    Encodes multiple pretrained feature modalities into a shared latent space
    using per-modality weighted layer pooling, independent encoders, MoE
    posterior fusion, and per-modality decoders.

    Parameters
    ----------
    modality_configs : dict
        {mod_name: {'dim': int, 'n_layers': int}} for each modality.
    latent_dim : int
        Shared latent space dimension (default 512).
    encoder_hidden : int
        Hidden dim for per-modality encoders.
    n_experts : int
        Number of MoE experts.
    expert_hidden : int
        Hidden dim for each expert MLP.
    decoder_hidden : int
        Hidden dim for per-modality decoders.
    dropout : float
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
        free_bits: float = 0.5,
    ):
        super().__init__()
        self.modality_configs = modality_configs
        self.latent_dim = latent_dim
        self.modality_names = list(modality_configs.keys())
        self.n_modalities = len(self.modality_names)
        self.free_bits = free_bits

        # 1. Weighted Layer Pooling (optional, only for multi-layer inputs)
        self.layer_pools = nn.ModuleDict({
            name: WeightedLayerPool(cfg['n_layers'])
            for name, cfg in modality_configs.items()
            if cfg.get('n_layers', 1) > 1
        })

        # 2. Per-modality encoders
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(
                input_dim=cfg['dim'],
                hidden_dim=encoder_hidden,
                latent_dim=latent_dim,
                dropout=dropout,
            )
            for name, cfg in modality_configs.items()
        })

        # 3. MoE Fusion
        total_mu_dim = self.n_modalities * latent_dim
        self.moe_fusion = MoEFusion(
            total_input_dim=total_mu_dim,
            latent_dim=latent_dim,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            dropout=dropout,
        )

        # 4. Per-modality decoders
        self.decoders = nn.ModuleDict({
            name: ModalityDecoder(
                latent_dim=latent_dim,
                hidden_dim=decoder_hidden,
                output_dim=cfg['dim'],
                dropout=dropout,
            )
            for name, cfg in modality_configs.items()
        })

        # Log parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MultimodalMoEVAE] {self.n_modalities} modalities, "
              f"latent_dim={latent_dim}, n_experts={n_experts}, "
              f"total params: {n_params:,}")

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·σ using the reparameterisation trick."""
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Encode all modalities → fused posterior → sample z.

        Supports both pre-pooled (B, D) and multi-layer (B, n_layers, D) inputs.
        Pre-pooled inputs skip WeightedLayerPool.

        Parameters
        ----------
        batch : dict of {mod_name: (B, D) or (B, n_layers, D)}

        Returns
        -------
        z          : (B, Z)  sampled latent
        mu_fused   : (B, Z)
        logvar_fused : (B, Z)
        pooled     : dict of {mod_name: (B, D_mod)}  pooled features (for decoder targets)
        """
        modality_mus = []
        pooled = {}

        for mod_name in self.modality_names:
            if mod_name in batch:
                x = batch[mod_name]
                # Auto-detect: (B, D) = pre-pooled, (B, L, D) = multi-layer
                if x.dim() == 2:
                    x_pooled = x  # already pooled
                elif x.dim() == 3:
                    x_pooled = self.layer_pools[mod_name](x)  # (B, D)
                else:
                    raise ValueError(f"Unexpected input dim {x.dim()} for {mod_name}")
                pooled[mod_name] = x_pooled
                # Per-modality encoding
                mu_mod, logvar_mod = self.encoders[mod_name](x_pooled)
                modality_mus.append(mu_mod)
            else:
                # Missing modality → zero mean (uniform prior contribution)
                B = next(iter(batch.values())).shape[0]
                device = next(iter(batch.values())).device
                modality_mus.append(
                    torch.zeros(B, self.latent_dim, device=device)
                )

        # MoE fusion
        mu_fused, logvar_fused = self.moe_fusion(modality_mus)

        # Sample
        z = self.reparameterize(mu_fused, logvar_fused)

        return z, mu_fused, logvar_fused, pooled

    def decode(
        self, z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Decode latent z → reconstructed features per modality.

        Parameters
        ----------
        z : (B, Z)

        Returns
        -------
        recons : dict of {mod_name: (B, D_mod)}
        """
        return {
            name: self.decoders[name](z)
            for name in self.modality_names
        }

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """Full forward pass.

        Returns
        -------
        recons     : dict of {mod_name: (B, D_mod)}
        mu_fused   : (B, Z)
        logvar_fused : (B, Z)
        pooled     : dict of {mod_name: (B, D_mod)}
        """
        z, mu_fused, logvar_fused, pooled = self.encode(batch)
        recons = self.decode(z)
        return recons, mu_fused, logvar_fused, pooled

    # -----------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I)) with free_bits, sum over dims, mean over batch."""
        # Per-dim KL: (B, Z)
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        # Free bits: clamp per-dim KL to minimum threshold
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        # Sum over latent dims, mean over batch (standard VAE)
        return kl_per_dim.sum(dim=-1).mean()

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        beta: float = 0.001,
    ) -> dict[str, torch.Tensor]:
        """Compute β-VAE loss with per-modality reconstruction.

        Parameters
        ----------
        batch : dict of {mod_name: (B, n_layers, D_mod)}
        beta : float
            KL weight (anneal from small → target).

        Returns
        -------
        dict with keys: 'loss', 'recon', 'kl', and per-modality recon losses.
        """
        recons, mu_fused, logvar_fused, pooled = self.forward(batch)

        # Per-modality reconstruction losses
        recon_losses = {}
        total_recon = 0.0
        n_active = 0
        for mod_name in self.modality_names:
            if mod_name in pooled:
                target = pooled[mod_name].detach()  # stop gradient on targets
                recon_loss = F.mse_loss(recons[mod_name], target)
                recon_losses[f"recon_{mod_name}"] = recon_loss
                total_recon = total_recon + recon_loss
                n_active += 1

        if n_active > 0:
            total_recon = total_recon / n_active  # average across modalities

        # KL divergence
        kl = self.kl_divergence(mu_fused, logvar_fused)

        # Total loss
        total = total_recon + beta * kl

        result = {
            "loss": total,
            "recon": total_recon if isinstance(total_recon, torch.Tensor) else torch.tensor(0.0),
            "kl": kl,
        }
        result.update(recon_losses)
        return result

    # -----------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------

    @torch.no_grad()
    def get_latent(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return deterministic mean latent μ (no sampling)."""
        z, mu, logvar, pooled = self.encode(batch)
        return mu  # (B, Z)

    @torch.no_grad()
    def get_layer_pool_weights(self) -> dict[str, torch.Tensor]:
        """Return normalised layer pooling weights for each modality."""
        return {
            name: pool.get_weights()
            for name, pool in self.layer_pools.items()
        }

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"MultimodalMoEVAE("
            f"modalities={self.n_modalities}, "
            f"latent_dim={self.latent_dim}, "
            f"params={n_params:,})"
        )
