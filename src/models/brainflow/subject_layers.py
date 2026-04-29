import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SubjectMLPLayers(nn.Module):
    """Per-subject 2-layer MLP output head with nonlinearity (Plan B).

    Adds a GELU nonlinearity between two per-subject linear projections,
    increasing expressivity over the linear-only SubjectLayers.
    w2 is zero-initialized for residual-safe startup (adaLN-Zero compatible).
    """

    def __init__(self, in_channels: int, out_channels: int, n_subjects: int,
                 hidden_mult: float = 1.0, bias: bool = True):
        super().__init__()
        mid = int(in_channels * hidden_mult)
        self.mid = mid
        self.w1 = nn.Parameter(torch.empty(n_subjects, in_channels, mid))
        self.w2 = nn.Parameter(torch.empty(n_subjects, mid, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels)) if bias else None

        # Kaiming-style init for w1 so gradient flows from step 1
        self.w1.data.normal_(0, 1.0 / in_channels ** 0.5)
        # Zero init for w2 → output starts at zero (residual-safe)
        nn.init.zeros_(self.w2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        w1 = self.w1[subject_ids]
        w2 = self.w2[subject_ids]
        if x.dim() == 3:
            h = torch.einsum("btd,bdm->btm", x, w1)
            h = F.gelu(h)
            out = torch.einsum("btm,bmo->bto", h, w2)
            if self.bias is not None:
                out = out + self.bias[subject_ids].unsqueeze(1)
        else:
            h = torch.einsum("bd,bdm->bm", x, w1)
            h = F.gelu(h)
            out = torch.einsum("bm,bmo->bo", h, w2)
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


class NetworkSubjectMLPLayers(nn.Module):
    """Per-network SubjectMLPLayers: 7 independent per-subject MLP heads (Plan B).

    Same structure as NetworkSubjectLayers but uses SubjectMLPLayers (2-layer MLP)
    instead of SubjectLayers (linear) for each Yeo network head.
    """

    SCHAEFER_7NET_PER_HEMI = [75, 74, 66, 68, 38, 61, 118]
    NETWORK_NAMES = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

    def __init__(
        self,
        in_channels: int,
        n_subjects: int,
        network_counts: list[int] | None = None,
        zero_init: bool = False,
        hidden_mult: float = 1.0,
    ):
        super().__init__()
        if network_counts is None:
            network_counts = [2 * c for c in self.SCHAEFER_7NET_PER_HEMI]
        self.network_counts = network_counts
        self.total_output_dim = sum(network_counts)
        self.n_networks = len(network_counts)

        self.heads = nn.ModuleList([
            SubjectMLPLayers(in_channels, n_k, n_subjects, hidden_mult=hidden_mult)
            for n_k in network_counts
        ])
        # w2 is already zero-init by default in SubjectMLPLayers.
        # zero_init flag kept for interface consistency (no extra action needed).

        logger.info(
            "NetworkSubjectMLPLayers: %d heads (hidden_mult=%.1f, mid=%d): %s = %d total voxels",
            self.n_networks,
            hidden_mult,
            int(in_channels * hidden_mult),
            list(zip(self.NETWORK_NAMES[:self.n_networks], network_counts)),
            self.total_output_dim,
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        parts = [head(x, subject_ids) for head in self.heads]
        return torch.cat(parts, dim=-1)


def build_subject_head(
    in_channels: int,
    output_dim: int,
    n_subjects: int,
    network_head: bool = False,
    zero_init: bool = False,
    head_type: str = "linear",
    hidden_mult: float = 1.0,
    network_counts: list[int] | None = None,
) -> nn.Module:
    """Factory to build subject output heads.

    Args:
        in_channels: Latent dimension input.
        output_dim: Total output voxels (used only for flat heads).
        n_subjects: Number of subjects.
        network_head: If True, use 7 Yeo network-split heads.
        zero_init: If True, zero-initialize weights.
        head_type: ``'linear'`` (SubjectLayers) or ``'mlp'`` (SubjectMLPLayers).
        hidden_mult: Hidden layer multiplier for MLP heads (mid = in_channels * hidden_mult).
        network_counts: Optional custom per-network parcel counts.

    Returns:
        nn.Module with ``forward(x, subject_ids)`` interface.
    """
    if head_type == "mlp":
        if network_head:
            return NetworkSubjectMLPLayers(
                in_channels, n_subjects,
                network_counts=network_counts,
                zero_init=zero_init,
                hidden_mult=hidden_mult,
            )
        return SubjectMLPLayers(in_channels, output_dim, n_subjects, hidden_mult=hidden_mult)

    # Default: linear
    if network_head:
        return NetworkSubjectLayers(
            in_channels, n_subjects,
            network_counts=network_counts,
            zero_init=zero_init,
        )
    return SubjectLayers(in_channels, output_dim, n_subjects)
