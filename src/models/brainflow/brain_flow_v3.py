"""BrainFlow v3 — Direct Regression (TRIBE-inspired).

Key changes from v2:
  - No VAE bottleneck, no CFM/diffusion decoder
  - Direct regression: Features → Transformer Encoder → Linear Head → 1000 voxels
  - Multi-subject support with subject embeddings
  - MSE + differentiable PCC loss

Architecture:
  1. Per-modality MLP projectors (TRIBE-style)
  2. Concatenate + Transformer encoder with RoPE
  3. Subject-conditioned regression head → fMRI voxels
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.brainflow.components.rope import RotaryEmbedding, apply_rotary_pos_emb


# =============================================================================
# Custom Transformer Encoder Layer for RoPE
# =============================================================================

class RoPETransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with RoPE and Pre-LN."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Self-attention projections
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = dropout

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, src, cos, sin):
        # Pre-LN Self-Attention
        x = self.norm1(src)
        B, T, C = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(B, T, C)
        src = src + self.proj(attn)

        # Pre-LN FFN
        src = src + self.ffn(self.norm2(src))
        return src


# =============================================================================
# Multi-Modal Condition Encoder (TRIBE-style, improved from v2)
# =============================================================================

class MultiModalEncoder(nn.Module):
    """Encode raw multimodal features into temporal representations.

    Improvements over v2:
      - Each modality projects to full hidden_dim (not hidden_dim // N)
      - Uses addition fusion instead of concatenation (more parameter efficient)
      - Deeper per-modality projectors (3-layer MLP)

    Args:
        modality_dims: dict mapping modality name → input dimension.
        hidden_dim: Transformer hidden dimension.
        n_layers: Number of Transformer encoder layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        modality_dropout: Probability of dropping an entire modality.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        hidden_dim: int = 1024,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout
        self.n_modalities = len(modality_dims)

        # Per-modality MLP projectors → full hidden_dim each
        self.projectors = nn.ModuleDict()
        for mod_name, raw_dim in modality_dims.items():
            self.projectors[mod_name] = nn.Sequential(
                nn.Linear(raw_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        # RoPE embedding
        self.rope = RotaryEmbedding(dim=hidden_dim // n_heads)

        # Transformer encoder
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod_name: (B, T, D_mod)}
    ) -> torch.Tensor:
        """Encode multimodal features.

        Returns:
            encoded: (B, T, hidden_dim)
        """
        B = None
        T = None

        # Determine which modalities to drop
        active_modalities = list(self.modality_dims.keys())
        if self.training and self.modality_dropout > 0:
            drop_mask = torch.rand(self.n_modalities) < self.modality_dropout
            if drop_mask.all():
                keep_idx = torch.randint(0, self.n_modalities, (1,)).item()
                drop_mask[keep_idx] = False
            active_modalities = [
                m for m, drop in zip(self.modality_dims.keys(), drop_mask) if not drop
            ]

        # Additive fusion: sum up per-modality projections
        x = None
        n_active = 0
        for mod_name in self.modality_dims.keys():
            if mod_name in modality_features and mod_name in active_modalities:
                feat = modality_features[mod_name]  # (B, T, D_mod)
                if B is None:
                    B, T = feat.shape[0], feat.shape[1]
                proj = self.projectors[mod_name](feat)  # (B, T, hidden_dim)
                if x is None:
                    x = proj
                else:
                    x = x + proj
                n_active += 1

        if x is None:
            # Fallback: determine B, T from any available modality
            for v in modality_features.values():
                B, T = v.shape[0], v.shape[1]
                break
            x = torch.zeros(B, T, self.hidden_dim,
                           device=next(self.parameters()).device,
                           dtype=next(self.parameters()).dtype)
            n_active = 1

        # Average by number of active modalities (normalize)
        x = x / max(n_active, 1)

        # RoPE frequencies
        cos, sin = self.rope(x, seq_dim=1)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)

        return x  # (B, T, hidden_dim)


# =============================================================================
# Subject-Conditioned Regression Head
# =============================================================================

class SubjectRegressionHead(nn.Module):
    """Maps encoder output → fMRI voxels with per-subject conditioning.

    Architecture:
        encoded (B, T, H) + subject_embed → MLP → (B, T, n_voxels)
        + per-subject scale + bias correction

    Args:
        hidden_dim: Input dimension from encoder.
        n_voxels: Number of output fMRI voxels (1000).
        n_subjects: Number of subjects.
        subject_embed_dim: Dimension of subject embedding.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        n_voxels: int = 1000,
        n_subjects: int = 4,
        subject_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_voxels = n_voxels

        # Subject embedding
        self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)

        # Regression MLP
        in_dim = hidden_dim + subject_embed_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_voxels),
        )

        # Per-subject scale + bias (identity init)
        self.subject_scale = nn.Embedding(n_subjects, n_voxels)
        self.subject_bias = nn.Embedding(n_subjects, n_voxels)
        nn.init.ones_(self.subject_scale.weight)
        nn.init.zeros_(self.subject_bias.weight)

    def forward(
        self,
        encoded: torch.Tensor,  # (B, T, H)
        subject_id: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Predict fMRI voxels.

        Returns:
            fmri_pred: (B, T, n_voxels)
        """
        B, T, _ = encoded.shape

        # Subject embedding
        s_emb = self.subject_embed(subject_id)  # (B, E)
        s_emb = s_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, E)

        # Concat and predict
        x = torch.cat([encoded, s_emb], dim=-1)  # (B, T, H+E)
        out = self.head(x)  # (B, T, V)

        # Per-subject affine correction
        scale = self.subject_scale(subject_id).unsqueeze(1)  # (B, 1, V)
        bias = self.subject_bias(subject_id).unsqueeze(1)  # (B, 1, V)
        return out * scale + bias  # (B, T, V)


# =============================================================================
# Loss Functions
# =============================================================================

def pearson_corr_loss(pred, target, mask=None):
    """Differentiable Pearson correlation loss (1 - mean PCC).

    Computes per-voxel PCC across time, then averages.

    Args:
        pred: (B, T, V)
        target: (B, T, V)
        mask: optional (B, T) boolean mask for valid timesteps

    Returns:
        loss: scalar (1 - mean_pcc), so minimizing this maximizes PCC
    """
    if mask is not None:
        # Apply mask
        mask_3d = mask.unsqueeze(-1)  # (B, T, 1)
        pred = pred * mask_3d
        target = target * mask_3d
        count = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # (B, 1, 1)
    else:
        count = pred.shape[1]

    # Per-voxel statistics across time dimension
    pred_mean = pred.sum(dim=1, keepdim=True) / count  # (B, 1, V)
    target_mean = target.sum(dim=1, keepdim=True) / count

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    if mask is not None:
        pred_centered = pred_centered * mask_3d
        target_centered = target_centered * mask_3d

    # Covariance and variance
    cov = (pred_centered * target_centered).sum(dim=1)  # (B, V)
    pred_var = (pred_centered ** 2).sum(dim=1)  # (B, V)
    target_var = (target_centered ** 2).sum(dim=1)  # (B, V)

    # Pearson correlation
    denom = torch.sqrt(pred_var * target_var + 1e-8)
    pcc = cov / denom  # (B, V)

    # Average across voxels and batch
    mean_pcc = pcc.mean()

    return 1.0 - mean_pcc


# =============================================================================
# Top-level BrainFlowDirect v3
# =============================================================================

class BrainFlowDirect(nn.Module):
    """BrainFlow v3 — Direct regression from multimodal features to fMRI.

    No VAE bottleneck, no CFM/diffusion. Directly predicts fMRI voxels.

    Args:
        modality_dims: dict mapping modality name → feature dimension.
        n_voxels: Number of fMRI voxels to predict.
        n_subjects: Number of subjects.
        encoder_params: kwargs for MultiModalEncoder.
        head_params: kwargs for SubjectRegressionHead.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_voxels: int = 1000,
        n_subjects: int = 4,
        encoder_params: dict = None,
        head_params: dict = None,
    ):
        super().__init__()
        self.n_voxels = n_voxels
        self.n_subjects = n_subjects

        # Encoder
        enc_params = encoder_params or {}
        self.encoder = MultiModalEncoder(
            modality_dims=modality_dims,
            **enc_params,
        )
        hidden_dim = enc_params.get("hidden_dim", 1024)

        # Regression head
        h_params = head_params or {}
        self.head = SubjectRegressionHead(
            hidden_dim=hidden_dim,
            n_voxels=n_voxels,
            n_subjects=n_subjects,
            **h_params,
        )

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],  # {mod: (B, T, D_mod)}
        subject_id: torch.Tensor,  # (B,)
        target_fmri: torch.Tensor = None,  # (B, T, V) — for loss computation
        valid_lens: torch.Tensor = None,  # (B,) — valid lengths
        pcc_weight: float = 0.5,
    ) -> dict:
        """Forward pass with optional loss computation.

        Returns:
            dict with 'pred' and optionally 'loss', 'mse_loss', 'pcc_loss'
        """
        # Encode
        encoded = self.encoder(modality_features)  # (B, T, H)

        # Predict fMRI
        pred = self.head(encoded, subject_id)  # (B, T, V)

        result = {"pred": pred}

        if target_fmri is not None:
            # Build mask from valid_lens
            B, T, V = pred.shape
            if valid_lens is not None:
                mask = torch.arange(T, device=pred.device).unsqueeze(0) < valid_lens.unsqueeze(1)
                # Apply mask for loss
                mask_3d = mask.unsqueeze(-1).float()
                n_valid = mask_3d.sum().clamp(min=1)

                # MSE loss (masked)
                mse = ((pred - target_fmri) ** 2 * mask_3d).sum() / (n_valid * V)
            else:
                mask = None
                mse = F.mse_loss(pred, target_fmri)

            result["mse_loss"] = mse

            # PCC loss
            if pcc_weight > 0:
                pcc_loss = pearson_corr_loss(pred, target_fmri, mask)
                result["pcc_loss"] = pcc_loss
                result["loss"] = mse + pcc_weight * pcc_loss
            else:
                result["loss"] = mse

        return result

    @torch.inference_mode()
    def predict(
        self,
        modality_features: dict[str, torch.Tensor],
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        """Generate fMRI predictions (inference mode).

        Args:
            modality_features: {mod_name: (B, T, D_mod)}
            subject_id: (B,)

        Returns:
            pred: (B, T, V) — predicted fMRI voxels
        """
        encoded = self.encoder(modality_features)
        return self.head(encoded, subject_id)

    def __repr__(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"BrainFlowDirect("
            f"n_voxels={self.n_voxels}, "
            f"n_subjects={self.n_subjects}, "
            f"params={n_params:,})"
        )
