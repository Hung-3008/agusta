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

        # Precompute and register anatomical indices for correct routing
        lh_offset = 0
        rh_offset = 500
        for i, count in enumerate(self.SCHAEFER_7NET_PER_HEMI):
            lh_idx = list(range(lh_offset, lh_offset + count))
            rh_idx = list(range(rh_offset, rh_offset + count))
            indices = torch.tensor(lh_idx + rh_idx, dtype=torch.long)
            self.register_buffer(f"net_indices_{i}", indices)
            lh_offset += count
            rh_offset += count

        logger.info(
            "NetworkSubjectLayers: %d heads: %s = %d total voxels",
            self.n_networks,
            list(zip(self.NETWORK_NAMES[:self.n_networks], network_counts)),
            self.total_output_dim,
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Run all network heads and route outputs to correct anatomical order.

        Args:
            x: (B, in_channels) or (B, T, in_channels) shared latent representation.
            subject_ids: (B,) subject index.

        Returns:
            Concatenated and anatomically routed per-network predictions.
        """
        parts = [head(x, subject_ids) for head in self.heads]
        
        out_shape = list(parts[0].shape)
        out_shape[-1] = self.total_output_dim
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        
        for i, part in enumerate(parts):
            idx = getattr(self, f"net_indices_{i}")
            if len(out_shape) == 3:
                out[:, :, idx] = part.to(dtype=out.dtype)
            else:
                out[:, idx] = part.to(dtype=out.dtype)
        return out
