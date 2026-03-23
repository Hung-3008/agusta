"""fMRI VAE with Subject Embedding.

Stage-1 pretraining model: compress fMRI sequences from all subjects into
a shared low-dimensional latent space (default 256-dim). The learned latent
space is subsequently used as the target space for Stage-2 Flow Matching.

Architecture:
    Encoder: [fMRI (B,V,T) + subject_emb] → TemporalResBlocks → (μ, log_σ²) (B,Z,T)
    Decoder: [z (B,Z,T)   + subject_emb] → TemporalResBlocks → fMRI_recon (B,V,T)
             + lightweight per-subject scale+bias correction at output
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def build_vae(cfg: dict) -> fMRI_VAE_v5:
    """Construct VAE from config.
    Only V5 is supported.
    """
    vae_cfg    = cfg["vae"]
    n_voxels   = cfg["fmri"]["n_voxels"]
    n_subjects = len(cfg["subjects"])

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
