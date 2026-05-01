import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AECNN_HRF_Source(nn.Module):
    """Condition-dependent Source Generator using Biological HRF Prior.

    Architecture (v2):
      1. Neural Event Extractor: Conv1D sequence → latent activations
      2. HRF Convolutional Filter: depthwise Conv1D simulating hemodynamic response
      3. Temporal Self-Attention: single-head attention for global temporal context
      4. Output heads:
           - mu_phi:      per-TR mean    (B, T, latent_dim)
           - log_var_phi: per-TR log-var (B, T, latent_dim)  ← from sequence, not pooled

    Key design rationale:
    - GELU (not Sigmoid): unbounded activations let mu_phi cover full normalised fMRI range.
    - Per-TR log_var: BOLD uncertainty varies across time (rest vs. high-arousal stimulus);
      a single scalar sigma cannot capture this.
    - Temporal self-attention (1 layer, n_heads=4): Conv1D receptive field is only
      ~12 TRs even with HRF kernel; attention allows the source to reason about which
      remote context TRs most influence each target TR.
    - Lightweight design (single attention layer): avoids overfitting with only 4 subjects.
    """

    def __init__(
        self,
        context_dim: int,
        latent_dim: int,
        hrf_kernel_size: int = 12,
        attn_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Step 1: Neural Event Extractor ──────────────────────────────────
        # GELU replaces Sigmoid: allows mu_phi to span full normalised fMRI range.
        self.neural_event_net = nn.Sequential(
            nn.Conv1d(context_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # ── Step 2: HRF Convolutional Filter ─────────────────────────────────
        # Depthwise Conv1D without activation simulating hemodynamic convolution.
        self.hrf_filter = nn.Conv1d(
            latent_dim, latent_dim,
            kernel_size=hrf_kernel_size,
            padding='same',
            groups=latent_dim,
        )

        # ── Step 3: Temporal Self-Attention ──────────────────────────────────
        # Single lightweight transformer layer for global temporal reasoning.
        # Allows the source to model long-range dependencies beyond the ~12-TR
        # receptive field of the HRF filter.
        self.temporal_norm = nn.LayerNorm(latent_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        # ── Step 4a: mu output projection ────────────────────────────────────
        self.mu_proj = nn.Linear(latent_dim, latent_dim)

        # ── Step 4b: Per-TR Log-Variance Predictor ───────────────────────────
        # Computed from the sequence (not pooled) so each TR has independent σ.
        # Separate small Conv1D branch operating on pre-attention features.
        # Bias initialised to 0 so all variances start at exp(0) = 1 (isotropic).
        self.log_var_conv = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(latent_dim // 2, latent_dim, kernel_size=1),
        )
        nn.init.zeros_(self.log_var_conv[-1].bias)

    def forward(self, context_sequence: torch.Tensor, context_pooled: torch.Tensor):
        """Forward pass.

        Args:
            context_sequence: (B, C, T) — transposed context tokens (C = context_dim).
            context_pooled:   (B, C)    — mean-pooled context (unused, kept for
                                          API compatibility with brainflow.py).

        Returns:
            mu_phi:      (B, T, latent_dim) — source mean, per TR.
            log_var_phi: (B, T, latent_dim) — source log-variance, per TR.
        """
        # Step 1+2: Conv feature extraction → HRF filtering
        neural_events = self.neural_event_net(context_sequence)  # (B, D, T)
        hrf_out = self.hrf_filter(neural_events)                  # (B, D, T)

        # Branch A: log_var (computed before attention — captures raw uncertainty)
        log_var_phi = self.log_var_conv(hrf_out)                  # (B, D, T)
        log_var_phi = log_var_phi.transpose(1, 2)                 # (B, T, D)

        # Step 3: Temporal self-attention on sequence
        h = hrf_out.transpose(1, 2)                               # (B, T, D)
        h_norm = self.temporal_norm(h)
        attn_out, _ = self.temporal_attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + self.attn_dropout(attn_out)                       # residual

        # Step 4a: mu projection
        mu_phi = self.mu_proj(h)                                   # (B, T, D)

        return mu_phi, log_var_phi
