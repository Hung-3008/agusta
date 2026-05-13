"""Verify BMDNetworkSubjectLayers + grouped target extraction pipeline.

Usage:
    python scripts/verify_bmd_network_heads.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.brainflow.subject_layers import BMDNetworkSubjectLayers


def main():
    print("=" * 70)
    print("BMDNetworkSubjectLayers VERIFICATION")
    print("=" * 70)

    # 1. Check class constants
    print(f"\nCategories: {len(BMDNetworkSubjectLayers.CATEGORIES)}")
    for cat, rois in BMDNetworkSubjectLayers.CATEGORIES.items():
        print(f"  {cat:20s}: {rois}")

    print(f"\nROI_ORDER ({len(BMDNetworkSubjectLayers.ROI_ORDER)} ROIs):")
    for i, roi in enumerate(BMDNetworkSubjectLayers.ROI_ORDER):
        print(f"  [{i:2d}] {roi}")

    # 2. Instantiate with known BMD counts
    counts = [2398, 1002, 324, 489, 1613, 95, 761, 686, 967]
    print(f"\nnetwork_counts: {counts}")
    print(f"sum: {sum(counts)}")
    assert sum(counts) == 8335, f"Expected 8335, got {sum(counts)}"

    n_subjects = 10
    latent_dim = 768
    model = BMDNetworkSubjectLayers(
        in_channels=latent_dim,
        n_subjects=n_subjects,
        network_counts=counts,
        zero_init=True,
    )

    print(f"\nModel: {model.n_categories} heads, {model.total_output_dim} total output")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

    # 3. Forward pass test
    batch_size = 4
    x = torch.randn(batch_size, latent_dim)
    subject_ids = torch.tensor([0, 1, 2, 3])
    out = model(x, subject_ids)
    print(f"\nForward pass:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == (batch_size, 8335), f"Expected (4, 8335), got {out.shape}"

    # 4. Test seq2seq mode (B, T, D)
    x_seq = torch.randn(batch_size, 1, latent_dim)
    out_seq = model(x_seq, subject_ids)
    print(f"\nSeq2seq forward pass:")
    print(f"  Input:  {x_seq.shape}")
    print(f"  Output: {out_seq.shape}")
    assert out_seq.shape == (batch_size, 1, 8335), f"Expected (4, 1, 8335), got {out_seq.shape}"

    # 5. Check output splits by category
    offset = 0
    print(f"\nOutput dimension mapping:")
    for cat, count in zip(BMDNetworkSubjectLayers.CATEGORY_NAMES, counts):
        print(f"  {cat:20s}: [{offset:5d}:{offset+count:5d}] ({count} voxels)")
        offset += count

    print(f"\n✅ All checks passed!")


if __name__ == "__main__":
    main()
