"""BrainFlow V2: Feature-Prior Conditional Flow Matching for fMRI prediction.

Core design:
    x0 = FeaturePriorEncoder(stimulus_features)   ← deterministic prior
    x1 = true fMRI
    Flow: x0 → x1, conditioned on subject_id
    ut = x1 - x0   ← DETERMINISTIC for same sample

Training:
    loss = CFM_loss(v_θ, ut) + λ * MSE(x0, x1)

Inference:
    x0 = encoder(features)   [deterministic]
    x  = ODE(x0, t=0→1)     [deterministic]
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding for t ∈ [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class AdaLNBlock(nn.Module):
    """Feed-forward block with adaLN-Zero modulation.

    x = x + gate * FFN(modulate(LN(x), shift, scale))
    """
    def __init__(self, dim: int, cond_dim: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * dim))
        ff_dim = dim * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )
        # Zero-init for stable start
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN(c).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ffn(h)


# ─────────────────────────────────────────────────────────────────────────────
# Layer aggregator
# ─────────────────────────────────────────────────────────────────────────────

class LayerWeightedSum(nn.Module):
    """Learned weighted sum over extracted model layers.

    Input:  list of (B, T, dim) tensors, one per layer
    Output: (B, T, dim)
    """
    def __init__(self, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        return sum(w[i] * layers[i] for i in range(len(layers)))


# ─────────────────────────────────────────────────────────────────────────────
# Feature Prior Encoder (hierarchical multi-model fusion)
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePriorEncoder(nn.Module):
    """Encodes multimodal stimulus features into a deterministic fMRI prior x0.

    Architecture (3-level):
      Level 1 — Per-modality layer aggregation + projection
        {mod: (B, T, dim)} → {mod: (B, T, proj_dim)}

      Level 2 — Cross-modal fusion
        Concat all modalities → (B, T, n_mods * proj_dim)
        → Linear → (B, T, hidden_dim)

      Level 3 — Temporal Transformer
        (B, T, hidden_dim) → Transformer → mean-pool → (B, hidden_dim)
        → Linear → (B, n_voxels)   [= x0 prior fMRI]

    Parameters
    ----------
    modality_dims : dict
        {mod_name: feature_dim}, e.g. {"vjepa2": 1408, "whisper": 1280, ...}
    proj_dim : int
        Per-modality projection dimension.
    hidden_dim : int
        Transformer hidden dim.
    n_encoder_layers : int
        Transformer depth.
    n_heads : int
        Attention heads.
    n_voxels : int
        fMRI output dimension.
    modality_dropout : float
        Probability of zeroing out each modality during training.
    ff_dropout : float
        Dropout in Transformer FFN.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        proj_dim: int = 512,
        hidden_dim: int = 1024,
        n_encoder_layers: int = 4,
        n_heads: int = 8,
        n_voxels: int = 1000,
        modality_dropout: float = 0.2,
        ff_dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.modality_dropout = modality_dropout
        self.proj_dim = proj_dim

        # Per-modality: LayerNorm → Linear → GELU → Linear → proj_dim
        self.projectors = nn.ModuleDict()
        for mod, in_dim in modality_dims.items():
            self.projectors[mod] = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            )

        # Cross-modal fusion: concat → linear
        concat_dim = proj_dim * len(modality_dims)
        self.cross_modal_proj = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
        )

        # Learnable temporal positional embedding (max 1024 TRs)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)

        # Temporal Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=ff_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(hidden_dim)

        # Output: mean-pool → linear → n_voxels
        self.head = nn.Linear(hidden_dim, n_voxels)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multimodal features into prior fMRI x0.

        Parameters
        ----------
        features : dict
            {mod: (B, T, dim)}

        Returns
        -------
        x0 : Tensor (B, n_voxels)
        """
        B = next(iter(features.values())).shape[0]
        T = next(iter(features.values())).shape[1]
        device = next(iter(features.values())).device

        # Modality dropout
        active_mods = list(self.modality_dims.keys())
        if self.training and self.modality_dropout > 0:
            to_drop = [m for m in active_mods if torch.rand(1).item() < self.modality_dropout]
            if len(to_drop) == len(active_mods):
                to_drop = to_drop[:-1]  # keep at least one
            active_mods = [m for m in active_mods if m not in to_drop]

        # Level 1: per-modality projection
        proj_list = []
        for mod in self.modality_dims:
            if mod in active_mods and mod in features:
                x = features[mod].float()   # (B, T, dim)
                # Handle potential T mismatch
                if x.shape[1] != T:
                    x = x.permute(0, 2, 1)
                    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
                    x = x.permute(0, 2, 1)
                x = self.projectors[mod](x)  # (B, T, proj_dim)
            else:
                x = torch.zeros(B, T, self.proj_dim, device=device)
            proj_list.append(x)

        # Level 2: cross-modal fusion
        x = torch.cat(proj_list, dim=-1)         # (B, T, n_mods * proj_dim)
        x = self.cross_modal_proj(x)             # (B, T, hidden_dim)

        # Add positional embedding
        x = x + self.pos_embed[:, :T, :]

        # Level 3: temporal Transformer
        x = self.transformer(x)                  # (B, T, hidden_dim)
        x = self.out_norm(x)

        # Mean-pool over time → fMRI prior
        x0 = self.head(x.mean(dim=1))            # (B, n_voxels)
        return x0


# ─────────────────────────────────────────────────────────────────────────────
# Velocity Refiner
# ─────────────────────────────────────────────────────────────────────────────

class VelocityRefiner(nn.Module):
    """Predicts the velocity field for flow matching x0 → x1.

    Conditioned on:
        - flow time t
        - subject_id  (captures individual brain differences)
        - x0          (the feature prior; residual reference)

    Input: x_t ∈ ℝ^V — interpolated fMRI at time t
    Output: velocity v_θ(x_t, t, subject_id, x0) ∈ ℝ^V

    Architecture:
        [x_t, x0] → input_proj → hidden
        [t_emb, sub_emb] → cond_mlp → modulation vector
        N × AdaLN-Zero FFN blocks
        → output_proj → velocity
    """

    def __init__(
        self,
        n_voxels: int = 1000,
        hidden_dim: int = 1024,
        n_blocks: int = 4,
        time_embed_dim: int = 256,
        n_subjects: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_voxels = n_voxels

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Subject embedding (condition)
        sub_dim = time_embed_dim
        self.subject_embed = nn.Embedding(n_subjects, sub_dim)

        # Combined conditioning: t + subject_id → modulation vector
        cond_dim = time_embed_dim + sub_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input: [x_t, x0] concatenated
        self.input_proj = nn.Linear(n_voxels * 2, hidden_dim)

        # AdaLN-Zero blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, hidden_dim, ff_mult=4, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Output
        self.out_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, n_voxels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,           # (B, V)
        t: torch.Tensor,              # (B,)
        subject_id: torch.Tensor,     # (B,)
        x0: torch.Tensor,            # (B, V)  prior reference
    ) -> torch.Tensor:
        # Conditioning
        t_emb = self.time_embed(t)                          # (B, td)
        s_emb = self.subject_embed(subject_id)              # (B, sd)
        cond = self.cond_proj(torch.cat([t_emb, s_emb], dim=-1))  # (B, hidden)

        # Input: x_t + x0 concatenated as residual reference
        h = self.input_proj(torch.cat([x_t, x0], dim=-1))  # (B, hidden)

        # Refine
        for block in self.blocks:
            h = block(h, cond)

        h = self.out_norm(h)
        return self.out_proj(h)                             # (B, V) velocity


# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow V2 — top-level
# ─────────────────────────────────────────────────────────────────────────────

class BrainFlowV2(nn.Module):
    """BrainFlow V2: Feature-Prior Conditional Flow Matching.

    Flow: x0 = encode(features) → x1 = true_fMRI
    conditioned on subject_id for individual brain differences.
    """

    def __init__(
        self,
        # FeaturePriorEncoder
        modality_dims: Dict[str, int],
        proj_dim: int = 512,
        hidden_dim: int = 1024,
        n_encoder_layers: int = 4,
        n_heads: int = 8,
        modality_dropout: float = 0.2,
        # VelocityRefiner
        n_voxels: int = 1000,
        refiner_hidden_dim: int = 1024,
        n_refiner_blocks: int = 4,
        time_embed_dim: int = 256,
        n_subjects: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_voxels = n_voxels

        self.prior_encoder = FeaturePriorEncoder(
            modality_dims=modality_dims,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            n_voxels=n_voxels,
            modality_dropout=modality_dropout,
            ff_dropout=dropout,
        )

        self.velocity_refiner = VelocityRefiner(
            n_voxels=n_voxels,
            hidden_dim=refiner_hidden_dim,
            n_blocks=n_refiner_blocks,
            time_embed_dim=time_embed_dim,
            n_subjects=n_subjects,
            dropout=dropout,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],  # {mod: (B, T, dim)}
        t: torch.Tensor,                    # (B,)
        x_t: torch.Tensor,                  # (B, V) interpolated fMRI
        subject_id: torch.Tensor,           # (B,)
    ):
        """Predict velocity v_θ and return prior x0.

        Returns
        -------
        v  : Tensor (B, V) — predicted velocity
        x0 : Tensor (B, V) — deterministic prior from features
        """
        x0 = self.prior_encoder(features)                             # (B, V)
        v = self.velocity_refiner(x_t, t, subject_id, x0=x0)         # (B, V)
        return v, x0

    @torch.no_grad()
    def sample(
        self,
        features: Dict[str, torch.Tensor],
        subject_id: torch.Tensor,
        n_steps: int = 20,
        method: str = "euler",
    ) -> torch.Tensor:
        """Generate fMRI prediction deterministically.

        Starts from x0 = encode(features), integrates ODE to x1.

        Returns
        -------
        x1 : Tensor (B, V)  — predicted fMRI
        """
        x0 = self.prior_encoder(features)   # (B, V) deterministic
        x = x0.clone()

        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0 - dt, n_steps, device=x.device)

        for t_val in ts:
            t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
            if method == "euler":
                v = self.velocity_refiner(x, t, subject_id, x0=x0)
                x = x + v * dt
            elif method == "midpoint":
                v1 = self.velocity_refiner(x, t, subject_id, x0=x0)
                x_mid = x + v1 * (dt / 2)
                t_mid = t + dt / 2
                v2 = self.velocity_refiner(x_mid, t_mid, subject_id, x0=x0)
                x = x + v2 * dt
            else:
                raise ValueError(f"Unknown method: {method}")

        return x
