"""fMRI VAE with Subject Embedding.

Stage-1 pretraining model: compress fMRI sequences from all subjects into
a shared low-dimensional latent space (default 512-dim). The learned latent
space is subsequently used as the target space for Stage-2 Flow Matching.

Architecture:
    Encoder: [fMRI (B,T,V) + subject_emb] → MLP → (μ, log_σ²) (B,T,Z)
    Decoder: [z (B,T,Z)   + subject_emb] → MLP → fMRI_recon (B,T,V)
             + lightweight per-subject scale+bias correction at output

Key design decisions:
    - Subject embedding concatenated at both encoder & decoder
      (vs. DemoVAE which only conditions decoder) for better per-subject recon
    - SubjectLayers (from brain_flow.py) for final per-subject affine correction
    - β-VAE annealing: KL weight ramps from 0 → beta_max over warmup_epochs
    - Operates on sequences (B, T, V) so VAE encodes per-TR latents
    - Latent dim is configurable (512 / 256 / 128 …)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building blocks
# =============================================================================

class MLP(nn.Module):
    """Simple MLP with LayerNorm + SiLU activations.

    Parameters
    ----------
    in_dim : int
    hidden_dims : list[int]
        Intermediate layer widths.
    out_dim : int
    dropout : float
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Encoder
# =============================================================================

class fMRI_Encoder(nn.Module):
    """Maps fMRI sequence + subject embedding → (μ, log_σ²) in latent space.

    Parameters
    ----------
    n_voxels : int
        Number of fMRI voxels (input dimension per TR).
    latent_dim : int
        Dimension of the latent code z.
    n_subjects : int
        Number of subjects.
    subject_embed_dim : int
        Dimension of learned subject embedding.
    hidden_dims : list[int]
        Widths of shared MLP trunk.
    dropout : float
    """

    def __init__(
        self,
        n_voxels: int,
        latent_dim: int,
        n_subjects: int,
        subject_embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 1024]

        self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
        in_dim = n_voxels + subject_embed_dim

        # Shared trunk
        self.trunk = MLP(in_dim, hidden_dims, hidden_dims[-1], dropout=dropout)

        # Split heads for mean and log-variance
        self.fc_mu     = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Clamp logvar to avoid numerical blow-up
        self._logvar_clamp = 10.0

    def forward(
        self,
        fmri: torch.Tensor,         # (B, T, V)
        subject_id: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode fMRI sequence to Gaussian parameters.

        Returns
        -------
        mu     : (B, T, Z)
        logvar : (B, T, Z)
        """
        B, T, V = fmri.shape
        s_emb = self.subject_embed(subject_id)          # (B, E)
        s_emb = s_emb.unsqueeze(1).expand(-1, T, -1)   # (B, T, E)

        x = torch.cat([fmri, s_emb], dim=-1)   # (B, T, V+E)
        h = self.trunk(x)                        # (B, T, hidden)

        mu     = self.fc_mu(h)                   # (B, T, Z)
        logvar = self.fc_logvar(h).clamp(-self._logvar_clamp, self._logvar_clamp)
        return mu, logvar


# =============================================================================
# Decoder
# =============================================================================

class fMRI_Decoder(nn.Module):
    """Maps latent z + subject embedding → reconstructed fMRI sequence.

    Uses a lightweight per-subject scale + bias correction at the output
    (instead of SubjectLayers which creates an (B*T, V, V) weight gather
    that is prohibitively large for sequence inputs).

    Parameters
    ----------
    n_voxels : int
    latent_dim : int
    n_subjects : int
    subject_embed_dim : int
    hidden_dims : list[int]
        Widths of MLP trunk (used in reversed order relative to encoder).
    dropout : float
    """

    def __init__(
        self,
        n_voxels: int,
        latent_dim: int,
        n_subjects: int,
        subject_embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 2048]  # mirror of encoder (reversed)

        self.n_voxels = n_voxels
        self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
        in_dim = latent_dim + subject_embed_dim

        # MLP trunk
        self.trunk = MLP(in_dim, hidden_dims, hidden_dims[-1], dropout=dropout)
        self.out_norm = nn.LayerNorm(hidden_dims[-1], elementwise_affine=False)
        self.pre_proj = nn.Linear(hidden_dims[-1], n_voxels)

        # Lightweight per-subject scale + bias: (n_subjects, V) each
        # Much smaller than SubjectLayers (n_subjects × V × V).
        # Initialized to scale=1, bias=0 so starts as identity.
        self.subject_scale = nn.Embedding(n_subjects, n_voxels)
        self.subject_bias  = nn.Embedding(n_subjects, n_voxels)
        nn.init.ones_(self.subject_scale.weight)
        nn.init.zeros_(self.subject_bias.weight)

    def forward(
        self,
        z: torch.Tensor,            # (B, T, Z)
        subject_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Decode latent sequence to fMRI.

        Returns
        -------
        recon : (B, T, V)
        """
        B, T, Z = z.shape
        s_emb = self.subject_embed(subject_id)          # (B, E)
        s_emb = s_emb.unsqueeze(1).expand(-1, T, -1)   # (B, T, E)

        x = torch.cat([z, s_emb], dim=-1)   # (B, T, Z+E)
        h = self.trunk(x)                    # (B, T, hidden)
        h = self.out_norm(h)
        h = self.pre_proj(h)                 # (B, T, V)

        # Per-subject affine correction: broadcast over T
        scale = self.subject_scale(subject_id).unsqueeze(1)  # (B, 1, V)
        bias  = self.subject_bias(subject_id).unsqueeze(1)   # (B, 1, V)
        return h * scale + bias                               # (B, T, V)


# =============================================================================
# VAE (top-level)
# =============================================================================

class fMRI_VAE(nn.Module):
    """Subject-aware fMRI VAE.

    Codec for (B, T, V) fMRI sequences → (B, T, Z) latent codes.

    Parameters
    ----------
    n_voxels : int
        Number of fMRI voxels per TR.
    latent_dim : int
        Latent space dimensionality. Configurable (512 / 256 / 128 …).
    n_subjects : int
        Number of unique subjects.
    subject_embed_dim : int
        Dimension of learned subject embedding (shared by encoder & decoder).
    encoder_hidden_dims : list[int]
        Encoder MLP widths, e.g. [2048, 1024].
    decoder_hidden_dims : list[int] or None
        Decoder MLP widths. Defaults to reversed encoder dims.
    dropout : float
    """

    def __init__(
        self,
        n_voxels: int = 1000,
        latent_dim: int = 512,
        n_subjects: int = 4,
        subject_embed_dim: int = 64,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_voxels   = n_voxels
        self.latent_dim = latent_dim

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [2048, 1024]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = list(reversed(encoder_hidden_dims))

        self.encoder = fMRI_Encoder(
            n_voxels=n_voxels,
            latent_dim=latent_dim,
            n_subjects=n_subjects,
            subject_embed_dim=subject_embed_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
        )
        self.decoder = fMRI_Decoder(
            n_voxels=n_voxels,
            latent_dim=latent_dim,
            n_subjects=n_subjects,
            subject_embed_dim=subject_embed_dim,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Core VAE operations
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·σ using the reparameterisation trick."""
        if not torch.is_grad_enabled():
            return mu   # deterministic at inference
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        fmri: torch.Tensor,         # (B, T, V)
        subject_id: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode fMRI → (z, μ, log_σ²).

        Returns
        -------
        z      : (B, T, Z)   — sampled latent (or μ at eval)
        mu     : (B, T, Z)
        logvar : (B, T, Z)
        """
        mu, logvar = self.encoder(fmri, subject_id)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(
        self,
        z: torch.Tensor,            # (B, T, Z)
        subject_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Decode latent → fMRI reconstruction (B, T, V)."""
        return self.decoder(z, subject_id)

    def forward(
        self,
        fmri: torch.Tensor,         # (B, T, V)
        subject_id: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """End-to-end forward pass.

        Returns
        -------
        recon  : (B, T, V)   — reconstructed fMRI
        mu     : (B, T, Z)
        logvar : (B, T, Z)
        """
        z, mu, logvar = self.encode(fmri, subject_id)
        recon = self.decode(z, subject_id)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-element MSE averaged over (B, T, V)."""
        return F.mse_loss(recon, target, reduction="mean")

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence KL(q(z|x) ‖ N(0,I)) averaged over batch and latent dims.

        KL = -0.5 * mean(1 + log_σ² - μ² - σ²)
        """
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    def loss(
        self,
        fmri: torch.Tensor,
        target: torch.Tensor,
        subject_id: torch.Tensor,
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute β-VAE loss.

        Parameters
        ----------
        fmri : (B, T, V)
        target: (B, T, V)
        subject_id : (B,)
        beta : float
            KL weight (anneal from 0 → beta_max during training).

        Returns
        -------
        dict with keys: 'loss', 'recon', 'kl'
        """
        recon, mu, logvar = self.forward(fmri, subject_id)
        l_recon = self.reconstruction_loss(recon, target)
        l_kl    = self.kl_loss(mu, logvar)
        total   = l_recon + beta * l_kl
        return {"loss": total, "recon": l_recon, "kl": l_kl}

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent(
        self,
        fmri: torch.Tensor,         # (B, T, V)
        subject_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Return deterministic mean latent μ (no sampling). Used for Stage 2."""
        mu, _ = self.encoder(fmri, subject_id)
        return mu   # (B, T, Z)

    @torch.no_grad()
    def reconstruct(
        self,
        fmri: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct fMRI using deterministic μ (eval mode)."""
        mu = self.get_latent(fmri, subject_id)
        return self.decode(mu, subject_id)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"fMRI_VAE("
            f"n_voxels={self.n_voxels}, "
            f"latent_dim={self.latent_dim}, "
            f"params={n_params:,})"
        )


# =============================================================================
# β annealing schedule helper
# =============================================================================

def beta_schedule(
    epoch: int,
    beta_max: float = 0.01,
    warmup_epochs: int = 20,
) -> float:
    """Linear β-KL annealing: 0 → beta_max over warmup_epochs."""
    if warmup_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * epoch / warmup_epochs)


# =============================================================================
# VAE v2  —  Transformer Temporal Encoder/Decoder
# =============================================================================

class fMRI_TransformerEncoder(nn.Module):
    """Transformer-based fMRI encoder.

    Models temporal correlations between TRs before projecting to latent space.
    Subject embedding is injected as an extra learnable token prepended to the
    sequence so attention can attend to it at every TR.

    Architecture
    ------------
    fMRI (B,T,V) → Linear input_proj → (B,T,H)
    subject_token (B,1,H)  ← Linear(subject_emb)
    cat → (B,T+1,H)
    → TransformerEncoder (n_layers, n_heads, ffn=4H, dropout)
    take T output tokens (drop subject token)
    → fc_mu, fc_logvar → (B,T,Z)
    """

    def __init__(
        self,
        n_voxels: int,
        latent_dim: int,
        n_subjects: int,
        subject_embed_dim: int = 64,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection: voxels → hidden
        self.input_proj = nn.Sequential(
            nn.Linear(n_voxels, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Subject embedding → token
        self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
        self.subject_proj  = nn.Linear(subject_embed_dim, hidden_dim)

        # Learnable positional embedding for T+1 positions
        # (register_buffer so it moves with .to(device) automatically)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False,
        )

        # Latent projections
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self._logvar_clamp = 10.0

    def forward(
        self,
        fmri: torch.Tensor,         # (B, T, V)
        subject_id: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, V = fmri.shape

        # Project fMRI to latent dim
        x = self.input_proj(fmri)   # (B, T, H)

        # Subject token
        s_emb = self.subject_embed(subject_id)   # (B, E)
        s_tok = self.subject_proj(s_emb).unsqueeze(1)  # (B, 1, H)

        # Prepend subject token
        x = torch.cat([s_tok, x], dim=1)         # (B, T+1, H)

        # Positional embedding (truncate/pad dynamically)
        pos = self.pos_embed[:, :T + 1, :]
        x = x + pos

        # Transformer
        h = self.transformer(x)     # (B, T+1, H)

        # Drop subject token, keep T fMRI tokens
        h = h[:, 1:, :]             # (B, T, H)

        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-self._logvar_clamp, self._logvar_clamp)
        return mu, logvar


class fMRI_TransformerDecoder(nn.Module):
    """Transformer-based fMRI decoder.

    Decodes latent z (B,T,Z) → fMRI (B,T,V) with temporal context and
    per-subject affine correction.

    Architecture
    ------------
    z (B,T,Z) → Linear latent_proj → (B,T,H)
    subject_token prepended
    → TransformerEncoder (shared architecture, separate weights)
    → output_proj → (B,T,V)
    → per-subject scale+bias
    """

    def __init__(
        self,
        n_voxels: int,
        latent_dim: int,
        n_subjects: int,
        subject_embed_dim: int = 64,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_voxels = n_voxels

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
        self.subject_proj  = nn.Linear(subject_embed_dim, hidden_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            dec_layer, num_layers=n_layers, enable_nested_tensor=False,
        )

        self.output_proj = nn.Linear(hidden_dim, n_voxels)

        # Per-subject affine correction (identity init)
        self.subject_scale = nn.Embedding(n_subjects, n_voxels)
        self.subject_bias  = nn.Embedding(n_subjects, n_voxels)
        nn.init.ones_(self.subject_scale.weight)
        nn.init.zeros_(self.subject_bias.weight)

    def forward(
        self,
        z: torch.Tensor,            # (B, T, Z)
        subject_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        B, T, Z = z.shape

        x = self.latent_proj(z)     # (B, T, H)

        s_emb = self.subject_embed(subject_id)
        s_tok = self.subject_proj(s_emb).unsqueeze(1)  # (B, 1, H)
        x = torch.cat([s_tok, x], dim=1)               # (B, T+1, H)

        pos = self.pos_embed[:, :T + 1, :]
        x = x + pos

        h = self.transformer(x)[:, 1:, :]   # (B, T, H)
        out = self.output_proj(h)            # (B, T, V)

        scale = self.subject_scale(subject_id).unsqueeze(1)  # (B, 1, V)
        bias  = self.subject_bias(subject_id).unsqueeze(1)   # (B, 1, V)
        return out * scale + bias


class fMRI_VAE_v2(nn.Module):
    """Subject-aware fMRI VAE with Transformer temporal encoder/decoder.

    Models temporal structure of BOLD signal (HRF correlations across TRs)
    using full-sequence self-attention. Key improvements over v1 (MLP):

    - Transformer encoder/decoder with pre-LN TransformerEncoderLayer
    - Subject injected as a prepended token (attention can attend to it)
    - Free Bits KL: KL floored at `free_bits` nats/dim to avoid posterior collapse
    - Per-subject scale+bias correction at decoder output

    Parameters
    ----------
    n_voxels : int
    latent_dim : int
    n_subjects : int
    subject_embed_dim : int
    hidden_dim : int        Transformer d_model
    n_heads : int
    n_layers : int          Number of Transformer layers (encoder and decoder each)
    dropout : float
    free_bits : float       KL floor in nats/dim (0 = disabled). Recommended: 1-2.
    """

    def __init__(
        self,
        n_voxels: int = 1000,
        latent_dim: int = 256,
        n_subjects: int = 4,
        subject_embed_dim: int = 64,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        free_bits: float = 2.0,
    ):
        super().__init__()
        self.n_voxels   = n_voxels
        self.latent_dim = latent_dim
        self.free_bits  = free_bits

        self.encoder = fMRI_TransformerEncoder(
            n_voxels=n_voxels, latent_dim=latent_dim,
            n_subjects=n_subjects, subject_embed_dim=subject_embed_dim,
            hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout,
        )
        self.decoder = fMRI_TransformerDecoder(
            n_voxels=n_voxels, latent_dim=latent_dim,
            n_subjects=n_subjects, subject_embed_dim=subject_embed_dim,
            hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Core operations (same API as fMRI_VAE for drop-in compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, fmri, subject_id):
        mu, logvar = self.encoder(fmri, subject_id)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, subject_id):
        return self.decoder(z, subject_id)

    def forward(self, fmri, subject_id):
        z, mu, logvar = self.encode(fmri, subject_id)
        recon = self.decode(z, subject_id)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss with Free Bits
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(recon, target):
        return F.mse_loss(recon, target, reduction="mean")

    def kl_loss(self, mu, logvar):
        """KL with Free Bits floor.

        KL per dim is clamped at `free_bits` nats before averaging.
        This prevents the decoder from ignoring a subset of latent dims
        (posterior collapse) while still allowing easy dims to fully
        collapse toward N(0,I).
        """
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.mean()
        if self.free_bits > 0:
            kl_loss = torch.max(kl_loss, torch.tensor(self.free_bits, device=kl_loss.device))
        return kl_loss

    def loss(self, fmri, target, subject_id, beta=1.0):
        recon, mu, logvar = self.forward(fmri, subject_id)
        l_recon = self.reconstruction_loss(recon, target)
        l_kl    = self.kl_loss(mu, logvar)
        return {"loss": l_recon + beta * l_kl, "recon": l_recon, "kl": l_kl}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent(self, fmri, subject_id):
        mu, _ = self.encoder(fmri, subject_id)
        return mu

    @torch.no_grad()
    def reconstruct(self, fmri, subject_id):
        return self.decode(self.get_latent(fmri, subject_id), subject_id)

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"fMRI_VAE_v2(n_voxels={self.n_voxels}, latent_dim={self.latent_dim}, "
            f"params={n:,})"
        )



# =============================================================================
# VAE v5 — Temporal Convolutional Network (TCN)
# =============================================================================
# Processes Voxels as Channels and Time as the Spatial Dimension.
# Respects the independent spatial nature of fMRI parcels while smoothing dynamics.
# =============================================================================

class TemporalResBlock(nn.Module):
    """1D Convolutional block acting on the temporal dimension."""
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class fMRI_VAE_v5(nn.Module):
    """Temporal Convolutional VAE for fMRI.
    
    Treats each voxel as a channel and convolves across the time dimension `T`.
    This resolves the issue of spatial pooling that destroyed voxel independence.
    
    Architecture:
        Input: (B, T, V) -> Transpose -> (B, V, T)
        Encoder: 1x1 Conv (V -> hidden) -> TemporalResBlocks -> 1x1 Conv to (mu, logvar)
        Latent: (B, Z, T) -> Transpose -> (B, T, Z)
        Decoder: 1x1 Conv (Z -> hidden) -> TemporalResBlocks -> 1x1 Conv (hidden -> V)
        Output: (B, V, T) -> Transpose -> (B, T, V)
    """

    def __init__(
        self,
        n_voxels: int = 1000,
        latent_dim: int = 256,
        hidden_dim: int = 1024,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        free_bits: float = 0.5,
        lambda_pcc: float = 1.0,
        pcc_warmstart_epochs: int = 0,
        use_subject: bool = True,
        n_subjects: int = 4,
        subject_embed_dim: int = 64,
    ):
        super().__init__()
        self.n_voxels = n_voxels
        self.latent_dim = latent_dim
        self.free_bits = free_bits
        self.lambda_pcc = lambda_pcc
        self.pcc_warmstart_epochs = pcc_warmstart_epochs
        self._current_epoch = 0
        self.use_subject = use_subject

        # Encoder: (B, V, T) -> (B, Z, T)
        in_ch = n_voxels + (subject_embed_dim if use_subject else 0)
        self.enc_in = nn.Conv1d(in_ch, hidden_dim, kernel_size=1)
        self.enc_blocks = nn.Sequential(*[
            TemporalResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        self.enc_out_mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.enc_out_lv = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        # Decoder: (B, Z, T) -> (B, V, T)
        dec_in_ch = latent_dim + (subject_embed_dim if use_subject else 0)
        self.dec_in = nn.Conv1d(dec_in_ch, hidden_dim, kernel_size=1)
        self.dec_blocks = nn.Sequential(*[
            TemporalResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        
        if use_subject:
            self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
            self.subject_scale = nn.Embedding(n_subjects, n_voxels)
            self.subject_bias  = nn.Embedding(n_subjects, n_voxels)
            nn.init.ones_(self.subject_scale.weight)
            nn.init.zeros_(self.subject_bias.weight)
            self.dec_out = nn.Conv1d(hidden_dim, n_voxels, kernel_size=1)
        else:
            self.dec_out = nn.Conv1d(hidden_dim, n_voxels, kernel_size=1)
            self.global_scale = nn.Parameter(torch.ones(1, n_voxels, 1))
            self.global_bias  = nn.Parameter(torch.zeros(1, n_voxels, 1))

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, fmri, subject_id):
        # fmri: (B, T, V)
        B, T, V = fmri.shape
        x = fmri.transpose(1, 2) # (B, V, T)
        
        if self.use_subject:
            s_emb = self.subject_embed(subject_id).unsqueeze(-1) # (B, E, 1)
            s_emb = s_emb.expand(-1, -1, T) # (B, E, T)
            x = torch.cat([x, s_emb], dim=1) # (B, V+E, T)

        h = self.enc_in(x)
        h = self.enc_blocks(h)
        mu = self.enc_out_mu(h).transpose(1, 2) # (B, T, Z)
        logvar = self.enc_out_lv(h).clamp(-10.0, 10.0).transpose(1, 2) # (B, T, Z)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, subject_id):
        # z: (B, T, Z)
        B, T, Z = z.shape
        x = z.transpose(1, 2) # (B, Z, T)
        
        if self.use_subject:
            s_emb = self.subject_embed(subject_id).unsqueeze(-1) # (B, E, 1)
            s_emb = s_emb.expand(-1, -1, T) # (B, E, T)
            x = torch.cat([x, s_emb], dim=1) # (B, Z+E, T)

        h = self.dec_in(x)
        h = self.dec_blocks(h)
        out = self.dec_out(h) # (B, V, T)
        
        if self.use_subject:
            scale = self.subject_scale(subject_id).unsqueeze(-1) # (B, V, 1)
            bias = self.subject_bias(subject_id).unsqueeze(-1)   # (B, V, 1)
            out = out * scale + bias
        else:
            out = out * self.global_scale + self.global_bias
            
        return out.transpose(1, 2) # (B, T, V)

    def forward(self, fmri, subject_id):
        z, mu, logvar = self.encode(fmri, subject_id)
        recon = self.decode(z, subject_id)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss: MSE + PCC + KL
    # ------------------------------------------------------------------

    @staticmethod
    def pcc_loss(recon, target):
        N = recon.shape[0] * recon.shape[1]
        r = recon.reshape(N, -1)
        t = target.reshape(N, -1)
        r_zm = r - r.mean(dim=1, keepdim=True)
        t_zm = t - t.mean(dim=1, keepdim=True)
        pcc = F.cosine_similarity(t_zm, r_zm, dim=1).mean()
        return 1.0 - pcc

    def loss(self, fmri, target, subject_id, beta=1.0):
        recon, mu, logvar = self.forward(fmri, subject_id)
        l_recon = F.mse_loss(recon, target, reduction="mean")
        l_pcc = self.pcc_loss(recon, target)
        
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl = kl_per_dim.mean()
        if self.free_bits > 0:
            kl = torch.max(kl, torch.tensor(self.free_bits, device=kl.device))
            
        if self._current_epoch < self.pcc_warmstart_epochs:
            loss = self.lambda_pcc * l_pcc
        else:
            loss = l_recon + self.lambda_pcc * l_pcc + beta * kl
            
        return {
            "loss": loss,
            "recon": l_recon,
            "spatial_pcc": 1.0 - l_pcc,
            "kl": kl
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent(self, fmri, subject_id):
        B, T, V = fmri.shape
        x = fmri.transpose(1, 2)
        if self.use_subject:
            s_emb = self.subject_embed(subject_id).unsqueeze(-1).expand(-1, -1, T)
            x = torch.cat([x, s_emb], dim=1)
        h = self.enc_blocks(self.enc_in(x))
        return self.enc_out_mu(h).transpose(1, 2)

    @torch.no_grad()
    def reconstruct(self, fmri, subject_id):
        return self.decode(self.get_latent(fmri, subject_id), subject_id)

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"fMRI_VAE_v5(n_voxels={self.n_voxels}, latent_dim={self.latent_dim}, "
            f"lambda_pcc={self.lambda_pcc}, use_subject={self.use_subject}, "
            f"params={n:,})"
        )

# =============================================================================
# Factory — build_vae()
# =============================================================================


# =============================================================================
# VAE v3 — Per-TR Spatial Patch ViT with PCC Loss
# =============================================================================

class fMRI_ViT_Encoder(nn.Module):
    """Per-TR spatial patch ViT encoder.

    Key design decisions (adapted from FmriViTVAE reference):
    - Each TR processed independently: input (B*T, V)
    - Voxels divided into spatial patches → Transformer attends across parcels
    - Subject embedding injected as CLS token (same role as class token)
    - Zero-init fc_logvar → starts with posterior ≈ N(0,1)
    - use_subject=False: pure spatial autoencoder, no subject conditioning
    """

    def __init__(
        self,
        n_voxels: int,
        latent_dim: int,
        n_subjects: int,
        subject_embed_dim: int = 64,
        patch_size: int = 50,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_subject: bool = True,
    ):
        super().__init__()
        self.use_subject = use_subject

        # Pad voxels to be divisible by patch_size
        if n_voxels % patch_size != 0:
            self.padded_v = ((n_voxels // patch_size) + 1) * patch_size
        else:
            self.padded_v = n_voxels
        self.pad_len  = self.padded_v - n_voxels
        self.n_patches = self.padded_v // patch_size
        self.patch_size = patch_size

        # Patch projection
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Subject as CLS token (only when use_subject=True)
        if use_subject:
            self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
            self.subject_proj  = nn.Linear(subject_embed_dim, hidden_dim)
        else:
            # Learnable global CLS token (no subject info)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embedding: 1 (CLS) + n_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Zero-init logvar → starts with posterior = N(0,1)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(
        self,
        x: torch.Tensor,           # (N, V)  N = B*T
        subject_id: torch.Tensor,  # (N,)  — may be ignored when use_subject=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, V = x.shape

        # Pad → patchify
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        x = x.view(N, self.n_patches, self.patch_size)   # (N, n_patches, p)
        tokens = self.patch_embed(x)                      # (N, n_patches, H)

        if self.use_subject:
            # Subject CLS token
            s_emb = self.subject_embed(subject_id)            # (N, E)
            s_tok = self.subject_proj(s_emb).unsqueeze(1)     # (N, 1, H)
        else:
            # Global learnable CLS token (no subject info)
            s_tok = self.cls_token.expand(N, -1, -1)          # (N, 1, H)

        tokens = torch.cat([s_tok, tokens], dim=1)        # (N, n_patches+1, H)
        tokens = tokens + self.pos_embed
        h = self.norm(self.transformer(tokens))            # (N, n_patches+1, H)

        cls = h[:, 0]                                     # (N, H) — CLS output
        mu     = self.fc_mu(cls)
        logvar = self.fc_logvar(cls).clamp(-10.0, 10.0)
        return mu, logvar


class fMRI_ViT_Decoder(nn.Module):
    """Per-TR spatial patch ViT decoder.

    Input:  z (N, Z)
    Output: recon (N, V)
    use_subject=False: no per-subject affine correction.
    """

    def __init__(
        self,
        n_voxels: int,
        latent_dim: int,
        n_subjects: int,
        subject_embed_dim: int = 64,
        patch_size: int = 50,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_subject: bool = True,
    ):
        super().__init__()
        self.use_subject = use_subject

        if n_voxels % patch_size != 0:
            padded_v = ((n_voxels // patch_size) + 1) * patch_size
        else:
            padded_v = n_voxels
        self.n_voxels   = n_voxels
        self.pad_len    = padded_v - n_voxels
        self.n_patches  = padded_v // patch_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Expand latent → token sequence
        self.dec_embed = nn.Linear(latent_dim, self.n_patches * hidden_dim)
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, hidden_dim)
        )
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            dec_layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.patch_recover = nn.Linear(hidden_dim, patch_size)

        if use_subject:
            # Per-subject affine correction (identity init)
            self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
            self.subject_proj  = nn.Linear(subject_embed_dim, hidden_dim)
            self.subject_scale = nn.Embedding(n_subjects, n_voxels)
            self.subject_bias  = nn.Embedding(n_subjects, n_voxels)
            nn.init.ones_(self.subject_scale.weight)
            nn.init.zeros_(self.subject_bias.weight)
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            # Global learned scale + bias (shared, subject-agnostic)
            self.global_scale = nn.Parameter(torch.ones(1, n_voxels))
            self.global_bias  = nn.Parameter(torch.zeros(1, n_voxels))

    def forward(
        self,
        z: torch.Tensor,           # (N, Z)
        subject_id: torch.Tensor,  # (N,)  — may be ignored when use_subject=False
    ) -> torch.Tensor:
        N = z.shape[0]

        tokens = self.dec_embed(z).view(N, self.n_patches, self.hidden_dim)
        
        if self.use_subject:
            s_emb = self.subject_embed(subject_id)            # (N, E)
            s_tok = self.subject_proj(s_emb).unsqueeze(1)     # (N, 1, H)
        else:
            s_tok = self.cls_token.expand(N, -1, -1)          # (N, 1, H)
            
        tokens = torch.cat([s_tok, tokens], dim=1)            # (N, n_patches+1, H)
        tokens = tokens + self.dec_pos_embed
        h = self.norm(self.transformer(tokens))               # (N, n_patches+1, H)

        h = h[:, 1:, :]                                       # (N, n_patches, H)
        patches = self.patch_recover(h)                       # (N, n_patches, p)
        out = patches.reshape(N, -1)                  # (N, padded_V)
        if self.pad_len > 0:
            out = out[:, :self.n_voxels]              # (N, V)

        if self.use_subject:
            # Per-subject affine correction
            scale = self.subject_scale(subject_id)    # (N, V)
            bias  = self.subject_bias(subject_id)     # (N, V)
        else:
            # Global scale+bias (broadcast over N)
            scale = self.global_scale.expand(N, -1)   # (N, V)
            bias  = self.global_bias.expand(N, -1)    # (N, V)
        return out * scale + bias


class fMRI_VAE_v3(nn.Module):
    """Per-TR Spatial Patch ViT VAE with PCC Loss.

    Key improvements over v1/v2:
    1. Per-TR processing: (B,T,V) → flatten → (B*T,V) → spatial attention
       Each TR encoded independently; Transformer attends ACROSS VOXELS (spatial)
       not across time (temporal). This is correct since voxels have spatial
       covariance structure, not (unknown) temporal dependencies.
    2. PCC loss: directly optimizes the evaluation metric.
       loss = MSE + lambda_pcc*(1-PCC) + beta*KL
    3. Subject as CLS token: rich subject conditioning via attention.
    4. Zero-init logvar: clean cold start.

    Parameters
    ----------
    n_voxels : int
    latent_dim : int
    n_subjects : int
    subject_embed_dim : int
    patch_size : int       Voxels per patch. n_voxels=1000, patch_size=50 → 20 patches
    hidden_dim : int       Transformer d_model
    n_heads : int
    n_layers : int
    dropout : float
    free_bits : float      KL floor per dim (0=disabled)
    lambda_pcc : float     Weight of PCC loss term (recommended: 0.5–2.0)
    """

    def __init__(
        self,
        n_voxels: int = 1000,
        latent_dim: int = 256,
        n_subjects: int = 4,
        subject_embed_dim: int = 64,
        patch_size: int = 50,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        free_bits: float = 0.5,
        lambda_pcc: float = 1.0,
        use_subject: bool = True,
    ):
        super().__init__()
        self.n_voxels   = n_voxels
        self.latent_dim = latent_dim
        self.free_bits  = free_bits
        self.lambda_pcc = lambda_pcc
        self.use_subject = use_subject

        shared = dict(
            n_voxels=n_voxels, latent_dim=latent_dim,
            n_subjects=n_subjects, subject_embed_dim=subject_embed_dim,
            patch_size=patch_size, hidden_dim=hidden_dim,
            n_heads=n_heads, n_layers=n_layers, dropout=dropout,
            use_subject=use_subject,
        )
        self.encoder = fMRI_ViT_Encoder(**shared)
        self.decoder = fMRI_ViT_Decoder(**shared)

    # ------------------------------------------------------------------
    # Core ops  (API compatible with v1/v2)
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu, logvar):
        if not torch.is_grad_enabled():
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(
        self,
        fmri: torch.Tensor,        # (B, T, V)
        subject_id: torch.Tensor,  # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode fMRI sequence. Returns z, mu, logvar each (B, T, Z)."""
        B, T, V = fmri.shape
        # Flatten to per-TR
        x_flat  = fmri.reshape(B * T, V)                             # (B*T, V)
        sub_flat = subject_id.unsqueeze(1).expand(-1, T).reshape(-1) # (B*T,)

        mu, logvar = self.encoder(x_flat, sub_flat)    # (B*T, Z) each
        z = self.reparameterize(mu, logvar)

        # Reshape back to sequence
        mu     = mu.view(B, T, -1)
        logvar = logvar.view(B, T, -1)
        z      = z.view(B, T, -1)
        return z, mu, logvar

    def decode(
        self,
        z: torch.Tensor,           # (B, T, Z)
        subject_id: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Decode latent sequence. Returns recon (B, T, V)."""
        B, T, Z = z.shape
        z_flat  = z.reshape(B * T, Z)
        sub_flat = subject_id.unsqueeze(1).expand(-1, T).reshape(-1)
        recon_flat = self.decoder(z_flat, sub_flat)   # (B*T, V)
        return recon_flat.view(B, T, self.n_voxels)

    def forward(self, fmri, subject_id):
        z, mu, logvar = self.encode(fmri, subject_id)
        recon = self.decode(z, subject_id)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss: MSE + PCC + KL
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(recon, target):
        return F.mse_loss(recon, target, reduction="mean")

    @staticmethod
    def pcc_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 - mean Pearson correlation across voxels (per-TR sample-wise).

        Computed as cosine similarity of mean-centered signals,
        same as FmriViTVAE reference implementation.

        recon, target: (B, T, V)  → treat each (B*T) as a sample
        """
        N = recon.shape[0] * recon.shape[1]
        r = recon.reshape(N, -1)
        t = target.reshape(N, -1)
        r_zm = r - r.mean(dim=1, keepdim=True)
        t_zm = t - t.mean(dim=1, keepdim=True)
        pcc = F.cosine_similarity(t_zm, r_zm, dim=1).mean()
        return 1.0 - pcc

    def kl_loss(self, mu, logvar):
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.mean()
        if self.free_bits > 0:
            kl_loss = torch.max(kl_loss, torch.tensor(self.free_bits, device=kl_loss.device))
        return kl_loss

    def loss(self, fmri, target, subject_id, beta=1.0):
        recon, mu, logvar = self.forward(fmri, subject_id)
        l_recon = self.reconstruction_loss(recon, target)
        l_pcc   = self.pcc_loss(recon, target)
        l_kl    = self.kl_loss(mu, logvar)
        total   = l_recon + self.lambda_pcc * l_pcc + beta * l_kl
        return {
            "loss":  total,
            "recon": l_recon,
            "spatial_pcc": 1.0 - l_pcc,
            "kl":    l_kl,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent(self, fmri, subject_id):
        B, T, V = fmri.shape
        x_flat  = fmri.reshape(B * T, V)
        sub_flat = subject_id.unsqueeze(1).expand(-1, T).reshape(-1)
        mu, _ = self.encoder(x_flat, sub_flat)
        return mu.view(B, T, -1)

    @torch.no_grad()
    def reconstruct(self, fmri, subject_id):
        return self.decode(self.get_latent(fmri, subject_id), subject_id)

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"fMRI_VAE_v3(n_voxels={self.n_voxels}, latent_dim={self.latent_dim}, "
            f"lambda_pcc={self.lambda_pcc}, use_subject={self.use_subject}, "
            f"params={n:,})"
        )


# =============================================================================
# VAE v4 — SynBrain Conv1D U-Net Architecture
# =============================================================================
# Adapted from SynBrain (https://github.com/MichaelMaiii/SynBrain)
# Self-contained: no external ldm dependency needed.
# =============================================================================

def _nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation."""
    return x * torch.sigmoid(x)


def _normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with auto-adjusted num_groups for small channel counts."""
    if in_channels >= 32:
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels,
                           eps=1e-6, affine=True)
    else:
        return nn.GroupNorm(num_groups=in_channels, num_channels=in_channels,
                           eps=1e-6, affine=True)


class _Downsample1D(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels,
                                  kernel_size=7, stride=2, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            return self.conv(x)
        return F.avg_pool1d(x, kernel_size=2, stride=2)


class _Upsample1D(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.trans_conv = nn.ConvTranspose1d(
            in_channels, in_channels,
            kernel_size=7, stride=2, padding=3, output_padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trans_conv(x)


class _ResnetBlock1D(nn.Module):
    """ResNet block for 1D convolutions (from SynBrain)."""

    def __init__(self, *, in_channels: int, out_channels: int | None = None,
                 conv_shortcut: bool = False, dropout: float = 0.0,
                 temb_channels: int = 0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None):
        h = _nonlinearity(self.norm1(x))
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(_nonlinearity(temb))[:, :, None]
        h = _nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class _AttnBlock1D(nn.Module):
    """Self-attention block for 1D features."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = _normalize(in_channels)
        self.q = nn.Conv1d(in_channels, in_channels, 1)
        self.k = nn.Conv1d(in_channels, in_channels, 1)
        self.v = nn.Conv1d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, l = q.shape
        w_ = torch.bmm(q.permute(0, 2, 1), k) * (c ** -0.5)
        w_ = F.softmax(w_, dim=2)
        h_ = torch.bmm(v, w_.permute(0, 2, 1))
        return x + self.proj_out(h_)


def _make_attn(in_channels: int, attn_type: str = "vanilla") -> nn.Module:
    if attn_type == "vanilla":
        return _AttnBlock1D(in_channels)
    return nn.Identity()


class NeuroEncoder(nn.Module):
    """SynBrain 1D convolutional encoder with multi-resolution ResNet blocks.

    Input:  (B, in_channels, V)    — typically in_channels=1
    Output: (B, z_channels*2, spatial)  when double_z=True
            (B, z_channels,   spatial)  otherwise
    """

    def __init__(self, *, ch: int, out_ch: int, ch_mult: tuple | list = (1, 2, 4, 8),
                 ada_length: int, num_res_blocks: int, num_down_blocks: int,
                 attn_resolutions: list, dropout: float = 0.0,
                 resamp_with_conv: bool = True, in_channels: int,
                 resolution: int, z_channels: int, double_z: bool = True,
                 use_linear_attn: bool = False, attn_type: str = "vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_down_blocks = num_down_blocks

        self.conv_in = nn.Conv1d(in_channels, ch, kernel_size=7, stride=1, padding=3)
        self.ada_maxpool = nn.AdaptiveMaxPool1d(ada_length)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch  # track for mid blocks

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(_ResnetBlock1D(in_channels=block_in,
                                            out_channels=block_out,
                                            dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < num_down_blocks:
                down.downsample = _Downsample1D(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock1D(in_channels=block_in,
                                          out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = _make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = _ResnetBlock1D(in_channels=block_in,
                                          out_channels=block_in, dropout=dropout)

        # Output
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv1d(block_in,
                                  2 * z_channels if double_z else z_channels,
                                  kernel_size=7, stride=1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.ada_maxpool(x)
        hs = [x]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level < self.num_down_blocks:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = _nonlinearity(self.norm_out(h))
        h = self.conv_out(h)
        return h


class NeuroDecoder(nn.Module):
    """SynBrain 1D convolutional decoder with multi-resolution ResNet blocks.

    Input:  (B, z_channels, spatial)
    Output: (B, out_ch, target_length)  — interpolated to original voxel count
    """

    def __init__(self, *, ch: int, out_ch: int, ch_mult: tuple | list = (1, 2, 4, 8),
                 num_res_blocks: int, num_up_blocks: int,
                 attn_resolutions: list, dropout: float = 0.0,
                 resamp_with_conv: bool = True, in_channels: int,
                 resolution: int, z_channels: int, give_pre_end: bool = False,
                 tanh_out: bool = False, use_linear_attn: bool = False,
                 attn_type: str = "vanilla", **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_up_blocks = num_up_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # z to block_in
        self.conv_in = nn.Conv1d(z_channels, block_in,
                                 kernel_size=7, stride=1, padding=3)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock1D(in_channels=block_in,
                                          out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = _make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = _ResnetBlock1D(in_channels=block_in,
                                          out_channels=block_in, dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(_ResnetBlock1D(in_channels=block_in,
                                            out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= self.num_resolutions - num_up_blocks:
                up.upsample = _Upsample1D(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # Output
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, out_ch,
                                  kernel_size=7, stride=1, padding=3)

    def forward(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= self.num_resolutions - self.num_up_blocks:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = F.interpolate(h, size=target_length, mode='linear', align_corners=True)
        h = _nonlinearity(self.norm_out(h))
        h = self.conv_out(h)
        return h


class _ProjectionMLP(nn.Module):
    """MLP projector for mapping between encoder output and latent space (SynBrain)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class fMRI_VAE_v4(nn.Module):
    """SynBrain Conv1D U-Net VAE (adapted for our pipeline).

    Per-TR processing: (B,T,V) → flatten → (B*T,1,V) → Conv1D U-Net
    encoder/decoder with multi-resolution ResNet blocks.

    Key features over v1–v3:
    - 1D convolutional encoder/decoder (not MLP or ViT)
    - Multi-resolution ResNet blocks + optional self-attention
    - AdaptiveMaxPool1d normalizes variable voxel counts to fixed spatial dim
    - F.interpolate restores original voxel dimension at decoder output
    - MLP projectors for μ/logvar (not just Linear heads)
    - PCC loss support for direct metric optimization

    Parameters
    ----------
    n_voxels : int
    latent_dim : int          VAE latent dimension (z_channels in SynBrain)
    ch : int                  Base channel count for Conv1D backbone
    ch_mult : list[int]       Channel multipliers per resolution level
    ada_length : int          Fixed spatial dim after AdaptiveMaxPool1d
    num_res_blocks : int      ResNet blocks per resolution level
    num_down_blocks : int     Number of downsampling levels (encoder)
    num_up_blocks : int       Number of upsampling levels (decoder)
    attn_resolutions : list   Resolutions at which to add attention
    resolution : int          Logical resolution for attention scheduling
    dropout : float
    double_z : bool           If True, encoder outputs 2*z_channels for μ+logvar
    linear_dim : int          Hidden dim of MLP projectors
    embed_dim : int           Output dim of MLP projectors (latent bottleneck)
    free_bits : float         KL floor (0 = disabled)
    lambda_pcc : float        PCC loss weight
    """

    def __init__(
        self,
        n_voxels: int = 1000,
        latent_dim: int = 256,
        ch: int = 128,
        ch_mult: list[int] | None = None,
        ada_length: int = 8192,
        num_res_blocks: int = 2,
        num_down_blocks: int = 1,
        num_up_blocks: int = 1,
        attn_resolutions: list | None = None,
        resolution: int = 512,
        dropout: float = 0.0,
        double_z: bool = False,
        free_bits: float = 0.0,
        lambda_pcc: float = 1.0,
        chunk_size: int = 64,
        attn_type: str = "vanilla",
        pcc_warmstart_epochs: int = 0,
    ):
        super().__init__()
        if ch_mult is None:
            ch_mult = [2, 4, 4]
        if attn_resolutions is None:
            attn_resolutions = []

        self.n_voxels = n_voxels
        self.latent_dim = latent_dim
        self.free_bits = free_bits
        self.lambda_pcc = lambda_pcc
        self.chunk_size = chunk_size
        self.pcc_warmstart_epochs = pcc_warmstart_epochs
        self._current_epoch = 0  # set by training loop

        ddconfig = dict(
            ch=ch, out_ch=1, ch_mult=ch_mult, ada_length=ada_length,
            num_res_blocks=num_res_blocks, num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks, attn_resolutions=attn_resolutions,
            dropout=dropout, in_channels=1, resolution=resolution,
            z_channels=latent_dim, double_z=double_z,
            attn_type=attn_type,
        )

        self.encoder = NeuroEncoder(**ddconfig)
        self.decoder = NeuroDecoder(**ddconfig)

        # Encoder output: (B, z_ch, spatial) where z_ch=latent_dim
        # spatial = ada_length // (2 ** num_down_blocks)
        enc_spatial = ada_length // (2 ** num_down_blocks)
        flat_dim = latent_dim * enc_spatial  # total features after flatten
        self._enc_spatial = enc_spatial

        # Learned bottleneck: flatten (z_ch, spatial) → latent_dim
        # This replaces the destructive mean(dim=-1) pooling
        self.mu_head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, latent_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, latent_dim),
        )
        # Expand latent_dim back to (z_ch, spatial) for decoder
        self.latent_expand = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.GELU(),
            nn.LayerNorm(flat_dim),
        )

    # ------------------------------------------------------------------
    # Core operations (API compatible with v1/v2/v3)
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _encode_flat(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode flat (N, 1, V) → (N, latent_dim) mu and logvar with mini-chunking."""
        N = x.shape[0]
        cs = self.chunk_size
        mu_parts, lv_parts = [], []
        for i in range(0, N, cs):
            chunk_in = x[i:i+cs]
            if self.training:
                h = torch.utils.checkpoint.checkpoint(
                    self._encode_chunk, chunk_in, use_reentrant=False)
            else:
                h = self._encode_chunk(chunk_in)
            mu_parts.append(h[0])
            lv_parts.append(h[1])
        return torch.cat(mu_parts, dim=0), torch.cat(lv_parts, dim=0)

    def _encode_chunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-chunk encode helper (checkpointable)."""
        h = self.encoder(x)                     # (cs, z_ch, spatial)
        h_flat = h.reshape(h.shape[0], -1)      # (cs, z_ch * spatial)
        mu = self.mu_head(h_flat)                # (cs, latent_dim)
        lv = self.logvar_head(h_flat).clamp(-10, 10)
        return mu, lv

    def _decode_flat(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Decode flat (N, latent_dim) → (N, V) with mini-chunking."""
        N = z_flat.shape[0]
        cs = self.chunk_size
        parts = []
        for i in range(0, N, cs):
            chunk = z_flat[i:i+cs]
            if self.training:
                r = torch.utils.checkpoint.checkpoint(
                    self._decode_chunk, chunk, use_reentrant=False)
            else:
                r = self._decode_chunk(chunk)
            parts.append(r)
        return torch.cat(parts, dim=0)

    def _decode_chunk(self, z: torch.Tensor) -> torch.Tensor:
        """Single-chunk decode helper (checkpointable)."""
        h = self.latent_expand(z)               # (cs, z_ch * spatial)
        h = h.view(h.shape[0], self.latent_dim, self._enc_spatial)  # (cs, z_ch, spatial)
        return self.decoder(h, self.n_voxels).squeeze(1)

    def encode(
        self,
        fmri: torch.Tensor,        # (B, T, V)
        subject_id: torch.Tensor,   # (B,) — unused, kept for API compat
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode fMRI sequence → (z, μ, logvar) each (B, T, Z)."""
        B, T, V = fmri.shape
        x = fmri.reshape(B * T, 1, V)
        mu, logvar = self._encode_flat(x)
        z = self.reparameterize(mu, logvar)
        return z.view(B, T, -1), mu.view(B, T, -1), logvar.view(B, T, -1)

    def decode(
        self,
        z: torch.Tensor,           # (B, T, Z)
        subject_id: torch.Tensor,   # (B,) — unused
    ) -> torch.Tensor:
        """Decode latent → fMRI reconstruction (B, T, V)."""
        B, T, Z = z.shape
        z_flat = z.reshape(B * T, Z)
        recon = self._decode_flat(z_flat)
        return recon.view(B, T, self.n_voxels)

    def forward(self, fmri, subject_id):
        z, mu, logvar = self.encode(fmri, subject_id)
        recon = self.decode(z, subject_id)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss: MSE + PCC + KL
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(recon, target):
        return F.mse_loss(recon, target, reduction="mean")

    @staticmethod
    def pcc_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 - mean Pearson correlation (spatial, per-TR)."""
        N = recon.shape[0] * recon.shape[1]
        r = recon.reshape(N, -1)
        t = target.reshape(N, -1)
        r_zm = r - r.mean(dim=1, keepdim=True)
        t_zm = t - t.mean(dim=1, keepdim=True)
        pcc = F.cosine_similarity(t_zm, r_zm, dim=1).mean()
        return 1.0 - pcc

    def kl_loss(self, mu, logvar):
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.mean()
        if self.free_bits > 0:
            kl_loss = torch.max(kl_loss,
                                torch.tensor(self.free_bits, device=kl_loss.device))
        return kl_loss

    def loss(self, fmri, target, subject_id, beta=1.0):
        recon, mu, logvar = self.forward(fmri, subject_id)
        l_recon = self.reconstruction_loss(recon, target)
        l_pcc   = self.pcc_loss(recon, target)
        l_kl    = self.kl_loss(mu, logvar)

        # PCC warmstart: first N epochs, train only PCC loss to escape
        # mean-collapse. MSE rewards constant predictions (predict 0),
        # so we disable it initially to force the model to produce
        # varied outputs first.
        if self._current_epoch < self.pcc_warmstart_epochs:
            total = self.lambda_pcc * l_pcc  # PCC-only
        else:
            total = l_recon + self.lambda_pcc * l_pcc + beta * l_kl

        return {
            "loss":  total,
            "recon": l_recon,
            "spatial_pcc": 1.0 - l_pcc,
            "kl":    l_kl,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent(self, fmri, subject_id):
        B, T, V = fmri.shape
        x = fmri.reshape(B * T, 1, V)
        mu, _ = self._encode_flat(x)
        return mu.view(B, T, -1)

    @torch.no_grad()
    def reconstruct(self, fmri, subject_id):
        return self.decode(self.get_latent(fmri, subject_id), subject_id)

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"fMRI_VAE_v4(n_voxels={self.n_voxels}, latent_dim={self.latent_dim}, "
            f"lambda_pcc={self.lambda_pcc}, params={n:,})"
        )



# =============================================================================
# Factory — build_vae()
# =============================================================================

def build_vae(cfg: dict) -> "fMRI_VAE | fMRI_VAE_v2 | fMRI_VAE_v3 | fMRI_VAE_v4 | fMRI_VAE_v5":
    """Construct VAE from config.

    cfg['vae']['version']:
      1 → fMRI_VAE       (MLP)
      2 → fMRI_VAE_v2    (temporal Transformer)
      3 → fMRI_VAE_v3    (per-TR spatial ViT + PCC loss)
      4 → fMRI_VAE_v4    (SynBrain Conv1D U-Net)
      5 → fMRI_VAE_v5    (Temporal TCN)  ← recommended
    """
    vae_cfg    = cfg["vae"]
    version    = vae_cfg.get("version", 1)
    n_voxels   = cfg["fmri"]["n_voxels"]
    n_subjects = len(cfg["subjects"])

    if version == 5:
        return fMRI_VAE_v5(
            n_voxels          = n_voxels,
            latent_dim        = vae_cfg.get("latent_dim", 256),
            hidden_dim        = vae_cfg.get("hidden_dim", 1024),
            num_res_blocks    = vae_cfg.get("num_res_blocks", 4),
            dropout           = vae_cfg.get("dropout", 0.1),
            free_bits         = vae_cfg.get("free_bits", 0.5),
            lambda_pcc        = vae_cfg.get("lambda_pcc", 1.0),
            pcc_warmstart_epochs = vae_cfg.get("pcc_warmstart_epochs", 0),
            use_subject       = vae_cfg.get("use_subject", True),
            n_subjects        = n_subjects,
            subject_embed_dim = vae_cfg.get("subject_embed_dim", 64),
        )
    elif version == 4:
        return fMRI_VAE_v4(
            n_voxels          = n_voxels,
            latent_dim        = vae_cfg.get("latent_dim", 256),
            ch                = vae_cfg.get("ch", 128),
            ch_mult           = vae_cfg.get("ch_mult", [2, 4, 4]),
            ada_length        = vae_cfg.get("ada_length", 8192),
            num_res_blocks    = vae_cfg.get("num_res_blocks", 2),
            num_down_blocks   = vae_cfg.get("num_down_blocks", 1),
            num_up_blocks     = vae_cfg.get("num_up_blocks", 1),
            attn_resolutions  = vae_cfg.get("attn_resolutions", []),
            resolution        = vae_cfg.get("resolution", 512),
            dropout           = vae_cfg.get("dropout", 0.0),
            double_z          = vae_cfg.get("double_z", False),
            free_bits         = vae_cfg.get("free_bits", 0.0),
            lambda_pcc        = vae_cfg.get("lambda_pcc", 1.0),
            chunk_size        = vae_cfg.get("chunk_size", 64),
            attn_type         = vae_cfg.get("attn_type", "vanilla"),
            pcc_warmstart_epochs = vae_cfg.get("pcc_warmstart_epochs", 0),
        )
    elif version == 3:
        return fMRI_VAE_v3(
            n_voxels          = n_voxels,
            latent_dim        = vae_cfg.get("latent_dim", 256),
            n_subjects        = n_subjects,
            subject_embed_dim = vae_cfg.get("subject_embed_dim", 64),
            patch_size        = vae_cfg.get("patch_size", 50),
            hidden_dim        = vae_cfg.get("hidden_dim", 512),
            n_heads           = vae_cfg.get("n_heads", 8),
            n_layers          = vae_cfg.get("n_layers", 4),
            dropout           = vae_cfg.get("dropout", 0.1),
            free_bits         = vae_cfg.get("free_bits", 0.5),
            lambda_pcc        = vae_cfg.get("lambda_pcc", 1.0),
            use_subject       = vae_cfg.get("use_subject", True),
        )
    elif version == 2:
        return fMRI_VAE_v2(
            n_voxels          = n_voxels,
            latent_dim        = vae_cfg.get("latent_dim", 256),
            n_subjects        = n_subjects,
            subject_embed_dim = vae_cfg.get("subject_embed_dim", 64),
            hidden_dim        = vae_cfg.get("hidden_dim", 512),
            n_heads           = vae_cfg.get("n_heads", 8),
            n_layers          = vae_cfg.get("n_layers", 4),
            dropout           = vae_cfg.get("dropout", 0.1),
            free_bits         = vae_cfg.get("free_bits", 0.5),
        )
    else:
        enc_h = vae_cfg.get("hidden_dims", [2048, 1024])
        return fMRI_VAE(
            n_voxels            = n_voxels,
            latent_dim          = vae_cfg.get("latent_dim", 256),
            n_subjects          = n_subjects,
            subject_embed_dim   = vae_cfg.get("subject_embed_dim", 64),
            encoder_hidden_dims = enc_h,
            decoder_hidden_dims = list(reversed(enc_h)),
            dropout             = vae_cfg.get("dropout", 0.0),
        )

