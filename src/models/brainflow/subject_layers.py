import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class SubjectLayers(nn.Module):
    """Per-subject linear output head (from TRIBE/Brain-Diffuser)."""

    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels)) if bias else None

        self.weights.data.normal_(0, 1.0 / in_channels ** 0.5)
        if self.bias is not None:
            self.bias.data.normal_(0, 1.0 / in_channels ** 0.5)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        w = self.weights[subject_ids]  # (B, in_channels, out_channels)
        if x.dim() == 3:
            # Seq2seq mode: (B, T, in_channels) → (B, T, out_channels)
            out = torch.einsum("btd,bdo->bto", x, w)
            if self.bias is not None:
                out = out + self.bias[subject_ids].unsqueeze(1)  # (B, 1, out_channels)
        else:
            # Single-step mode: (B, in_channels) → (B, out_channels)
            out = torch.einsum("bd,bdo->bo", x, w)
            if self.bias is not None:
                out = out + self.bias[subject_ids]
        return out


class NetworkSubjectLayers(nn.Module):
    """Per-network SubjectLayers: 7 independent per-subject linear heads.

    Each Yeo functional network gets its own SubjectLayers mapping
    latent_dim -> n_parcels_k for each subject independently.
    Output is concatenated in network order to produce (B, total_output_dim).

    Schaefer 1000Par7Net parcels are ordered: LH networks then RH networks,
    each hemisphere in order: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default.
    """

    SCHAEFER_7NET_PER_HEMI = [75, 74, 66, 68, 38, 61, 118]  # sum = 500
    NETWORK_NAMES = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

    def __init__(
        self,
        in_channels: int,
        n_subjects: int,
        network_counts: list[int] | None = None,
        zero_init: bool = False,
    ):
        super().__init__()
        if network_counts is None:
            # Default: Schaefer 1000 parcels, 7 networks, both hemispheres
            network_counts = [2 * c for c in self.SCHAEFER_7NET_PER_HEMI]
        self.network_counts = network_counts
        self.total_output_dim = sum(network_counts)
        self.n_networks = len(network_counts)

        self.heads = nn.ModuleList([
            SubjectLayers(in_channels, n_k, n_subjects)
            for n_k in network_counts
        ])
        if zero_init:
            for head in self.heads:
                nn.init.zeros_(head.weights)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

        logger.info(
            "NetworkSubjectLayers: %d heads: %s = %d total voxels",
            self.n_networks,
            list(zip(self.NETWORK_NAMES[:self.n_networks], network_counts)),
            self.total_output_dim,
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Run all network heads and concatenate outputs.

        Args:
            x: (B, in_channels) shared latent representation.
            subject_ids: (B,) subject index.

        Returns:
            (B, total_output_dim) concatenated per-network predictions.
        """
        parts = [head(x, subject_ids) for head in self.heads]
        return torch.cat(parts, dim=-1)


class VoxelPersonalityHead(nn.Module):
    """Bilinear factorization of per-subject output head.

    Replaces (n_subjects, latent_dim, n_voxels) per-subject weight tensor
    with a low-rank factorization:

        V  ∈ R^(n_voxels × d_v)             — voxel embeddings (shared across subjects)
        W  ∈ R^(n_subjects × latent_dim × d_v) — per-subject projector
        b  ∈ R^(n_subjects × n_voxels)        — per-subject bias (optional)

        y[s, v] = (z · W[s]) · V[v]^T + b[s, v]

    Voxels live in a learned d_v-dimensional embedding space, so functionally
    similar voxels (same Yeo/Schaefer parcel, nearby cortical surface) naturally
    cluster and share readout dynamics. Subject-specific variance is captured
    by the smaller W tensor instead of duplicating the full readout matrix.

    Param count (with n_subjects=4, latent_dim=1024, n_voxels=1000, d_v=128):
        V:    n_voxels * d_v                    = 128,000
        W:    n_subjects * latent_dim * d_v     = 524,288
        b:    n_subjects * n_voxels             =   4,000
        TOTAL ≈ 656K   vs. NetworkSubjectLayers ≈ 4.1M  (≈ 6× fewer)
    """

    def __init__(
        self,
        latent_dim: int,
        n_voxels: int,
        n_subjects: int,
        voxel_dim: int = 128,
        bias: bool = True,
        zero_init_w: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_voxels = n_voxels
        self.n_subjects = n_subjects
        self.voxel_dim = voxel_dim

        # Voxel embeddings — small random init, will learn parcel structure.
        self.V = nn.Parameter(torch.randn(n_voxels, voxel_dim) * (1.0 / voxel_dim ** 0.5))

        # Per-subject projection. Zero-init keeps warm-started trunk safe:
        # output at step 0 = bias only.
        if zero_init_w:
            self.W = nn.Parameter(torch.zeros(n_subjects, latent_dim, voxel_dim))
        else:
            self.W = nn.Parameter(
                torch.randn(n_subjects, latent_dim, voxel_dim) * (1.0 / latent_dim ** 0.5)
            )

        self.bias = nn.Parameter(torch.zeros(n_subjects, n_voxels)) if bias else None

        logger.info(
            "VoxelPersonalityHead: V(%d,%d) + W(%d,%d,%d) + bias=%s = %d params (vs flat %d)",
            n_voxels, voxel_dim,
            n_subjects, latent_dim, voxel_dim,
            bias,
            n_voxels * voxel_dim + n_subjects * latent_dim * voxel_dim
                + (n_subjects * n_voxels if bias else 0),
            n_subjects * latent_dim * n_voxels,
        )

    def forward(self, z: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        W_s = self.W[subject_ids]  # (B, latent_dim, voxel_dim)
        if z.dim() == 3:
            # Seq2seq: (B, T, latent_dim) → (B, T, voxel_dim) → (B, T, n_voxels)
            q = torch.einsum("btd,bds->bts", z, W_s)
            y = torch.einsum("bts,vs->btv", q, self.V)
            if self.bias is not None:
                y = y + self.bias[subject_ids].unsqueeze(1)
        else:
            # Single-step: (B, latent_dim) → (B, voxel_dim) → (B, n_voxels)
            q = torch.einsum("bd,bds->bs", z, W_s)
            y = torch.einsum("bs,vs->bv", q, self.V)
            if self.bias is not None:
                y = y + self.bias[subject_ids]
        return y
