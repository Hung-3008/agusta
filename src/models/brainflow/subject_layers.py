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


class BMDNetworkSubjectLayers(nn.Module):
    """Per-category SubjectLayers for BOLD Moments Dataset (BMD).

    BMD uses 46 functional ROIs (23 per hemisphere) organized into
    9 functional categories — fundamentally different from Algonauts'
    Schaefer 7-network parcellation.

    Each category gets its own SubjectLayers mapping:
        latent_dim → n_voxels_category for each subject independently.
    Output is concatenated in category order to produce (B, 8335).

    Target voxels MUST be ordered to match this category order.
    Use ``ROI_ORDER`` to extract targets in the correct order.
    """

    # 9 functional categories with their ROI names
    # Each ROI has both 'l' (left) and 'r' (right) hemisphere versions
    CATEGORIES = {
        'EarlyVisual':   ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v', 'V3ab', 'hV4'],
        'ScenePlace':    ['PPA', 'RSC', 'TOS'],
        'Body':          ['EBA'],
        'Face':          ['FFA', 'OFA'],
        'Object':        ['LOC'],
        'Motion':        ['MT'],
        'TemporalSTS':   ['STS'],
        'Parietal':      ['IPS0', 'IPS1-2-3', '7AL'],
        'Somatosensory': ['BA2', 'PFt', 'PFop'],
    }

    CATEGORY_NAMES = list(CATEGORIES.keys())

    # Canonical ROI ordering: grouped by category, l then r within each ROI
    # This MUST match the order used in target extraction
    ROI_ORDER = []
    for _cat, _rois in CATEGORIES.items():
        for _roi in _rois:
            ROI_ORDER.append(f'l{_roi}')
            ROI_ORDER.append(f'r{_roi}')

    def __init__(
        self,
        in_channels: int,
        n_subjects: int,
        network_counts: list[int] | None = None,
        zero_init: bool = False,
    ):
        super().__init__()
        if network_counts is None:
            raise ValueError(
                "BMDNetworkSubjectLayers requires explicit network_counts "
                "(voxel counts per category). Use create_bmd_targets.py with "
                "mode='roi_subset_grouped' to extract targets in the correct order."
            )

        self.network_counts = network_counts
        self.total_output_dim = sum(network_counts)
        self.n_categories = len(network_counts)

        assert self.n_categories == len(self.CATEGORIES), \
            f"Expected {len(self.CATEGORIES)} categories, got {self.n_categories}"

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
            "BMDNetworkSubjectLayers: %d heads: %s = %d total voxels",
            self.n_categories,
            list(zip(self.CATEGORY_NAMES, network_counts)),
            self.total_output_dim,
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Run all category heads and concatenate outputs.

        Args:
            x: (B, in_channels) or (B, T, in_channels) shared latent.
            subject_ids: (B,) subject index.

        Returns:
            (B, total_output_dim) or (B, T, total_output_dim) predictions.
        """
        parts = [head(x, subject_ids) for head in self.heads]
        return torch.cat(parts, dim=-1)
