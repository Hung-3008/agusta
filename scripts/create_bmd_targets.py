"""
Create BMD target data with two atlas mappings:
  1. ROI voxel subset  (~8,335-dim per subject, may vary)
  2. BMDgeneral mask   (~14,672-dim per subject, may vary)

Betas are averaged across repetitions (train: 3 reps, test: 10 reps).
Output: .npy files per subject per split.

Usage:
    python scripts/create_bmd_targets.py
"""

import pickle
import numpy as np
from pathlib import Path
import time
import json

BMD_BASE = Path("Data/BOLDMomentsDataset")
GLM_BASE = BMD_BASE / "derivatives/versionB/MNI152/GLM"
OUT_BASE = BMD_BASE / "targets"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_roi_indices(sub_dir):
    """Extract flat indices for BMDgeneral and all specific ROIs."""
    roi_dir = sub_dir / "ROIs"
    
    # BMDgeneral
    gen_pkl = load_pkl(roi_dir / "ROI-BMDgeneral_indices.pkl")
    bmd_indices = gen_pkl[1].flatten()  # (N, 1) → (N,)
    
    # All specific ROIs (exclude BMDgeneral)
    roi_indices_list = []
    roi_names = []
    for rf in sorted(roi_dir.glob("*.pkl")):
        name = rf.stem.replace("ROI-", "").replace("_indices", "")
        if name == "BMDgeneral":
            continue
        roi_pkl = load_pkl(rf)
        indices = roi_pkl[1].flatten()
        roi_indices_list.append(indices)
        roi_names.append(name)
    
    # Concatenate all ROI indices and deduplicate (preserve order)
    all_roi_indices = np.concatenate(roi_indices_list)
    # Use unique to remove duplicates (voxels shared between ROIs)
    roi_subset_indices = np.unique(all_roi_indices)
    
    return {
        "bmd_general": np.sort(bmd_indices),
        "roi_subset": roi_subset_indices,
        "roi_names": roi_names,
        "roi_per_region": {name: idx for name, idx in zip(roi_names, roi_indices_list)},
    }


def process_subject(sub_name, out_base):
    """Process one subject: extract target data with both atlas mappings."""
    sub_dir = GLM_BASE / sub_name
    betas_dir = sub_dir / "prepared_betas"
    
    print(f"\n{'─' * 60}")
    print(f"Processing {sub_name}...")
    
    # Step 1: Get ROI indices
    t0 = time.time()
    roi_info = get_roi_indices(sub_dir)
    bmd_idx = roi_info["bmd_general"]
    roi_idx = roi_info["roi_subset"]
    print(f"  ROI indices loaded ({time.time()-t0:.1f}s)")
    print(f"    BMDgeneral:  {len(bmd_idx):,} voxels (idx range: {bmd_idx.min()}-{bmd_idx.max()})")
    print(f"    ROI subset:  {len(roi_idx):,} voxels (idx range: {roi_idx.min()}-{roi_idx.max()})")
    
    # Verify ROI subset is a subset of BMDgeneral
    roi_in_bmd = np.isin(roi_idx, bmd_idx)
    if not roi_in_bmd.all():
        n_outside = (~roi_in_bmd).sum()
        print(f"    ⚠️  {n_outside} ROI voxels are OUTSIDE BMDgeneral!")
    else:
        print(f"    ✅ All ROI voxels are within BMDgeneral")
    
    # Step 2: Process train and test
    for split, task, expected_vids, expected_reps in [
        ("train", "train", 1000, 3),
        ("test", "test", 102, 10),
    ]:
        pkl_path = betas_dir / f"{sub_name}_organized_betas_task-{task}_normalized.pkl"
        if not pkl_path.exists():
            print(f"  ❌ {split} betas not found!")
            continue
        
        print(f"\n  Loading {split} betas...", end=" ", flush=True)
        t0 = time.time()
        data = load_pkl(pkl_path)
        betas = data[0]  # (N_videos, N_reps, 108219)
        labels = data[1]  # ['vid0001', ...]
        print(f"done ({time.time()-t0:.1f}s) — shape: {betas.shape}")
        
        assert betas.shape[0] == expected_vids, f"Expected {expected_vids} videos, got {betas.shape[0]}"
        assert betas.shape[1] == expected_reps, f"Expected {expected_reps} reps, got {betas.shape[1]}"
        
        # Average across repetitions
        print(f"  Averaging {expected_reps} repetitions...", end=" ", flush=True)
        betas_avg = betas.mean(axis=1)  # (N_videos, 108219)
        print(f"→ shape: {betas_avg.shape}")
        
        # ── AVG targets ──
        bmd_target = betas_avg[:, bmd_idx].astype(np.float32)
        roi_target = betas_avg[:, roi_idx].astype(np.float32)
        
        bmd_out = out_base / "bmd_general"
        bmd_out.mkdir(parents=True, exist_ok=True)
        np.save(bmd_out / f"{sub_name}_{split}.npy", bmd_target)
        
        roi_out = out_base / "roi_subset"
        roi_out.mkdir(parents=True, exist_ok=True)
        np.save(roi_out / f"{sub_name}_{split}.npy", roi_target)
        
        # ── PER-REP targets ──
        # Reshape (N_vids, N_reps, full) → extract atlas → (N_vids*N_reps, atlas)
        bmd_per_rep = betas[:, :, bmd_idx].reshape(-1, len(bmd_idx)).astype(np.float32)
        roi_per_rep = betas[:, :, roi_idx].reshape(-1, len(roi_idx)).astype(np.float32)
        
        bmd_rep_out = out_base / "bmd_general_per_rep"
        bmd_rep_out.mkdir(parents=True, exist_ok=True)
        np.save(bmd_rep_out / f"{sub_name}_{split}.npy", bmd_per_rep)
        
        roi_rep_out = out_base / "roi_subset_per_rep"
        roi_rep_out.mkdir(parents=True, exist_ok=True)
        np.save(roi_rep_out / f"{sub_name}_{split}.npy", roi_per_rep)
        
        print(f"  Saved:")
        print(f"    [avg] BMDgeneral: {bmd_target.shape}, {bmd_target.nbytes/1024/1024:.1f} MB")
        print(f"    [avg] ROI subset: {roi_target.shape}, {roi_target.nbytes/1024/1024:.1f} MB")
        print(f"    [rep] BMDgeneral: {bmd_per_rep.shape}, {bmd_per_rep.nbytes/1024/1024:.1f} MB")
        print(f"    [rep] ROI subset: {roi_per_rep.shape}, {roi_per_rep.nbytes/1024/1024:.1f} MB")
        
        # Stats
        print(f"  Stats (rep-averaged, {split}):")
        print(f"    BMDgeneral: range=[{bmd_target.min():.4f}, {bmd_target.max():.4f}], "
              f"mean={bmd_target.mean():.4f}, std={bmd_target.std():.4f}")
        print(f"    ROI subset: range=[{roi_target.min():.4f}, {roi_target.max():.4f}], "
              f"mean={roi_target.mean():.4f}, std={roi_target.std():.4f}")
        
        # Free memory
        del betas, betas_avg, data
    
    # Save ROI index mapping for this subject
    meta = {
        "bmd_general_n_voxels": len(bmd_idx),
        "roi_subset_n_voxels": len(roi_idx),
        "roi_names": roi_info["roi_names"],
        "roi_per_region_sizes": {name: len(idx) for name, idx in roi_info["roi_per_region"].items()},
    }
    
    # Save indices as numpy for later use
    np.save(out_base / "bmd_general" / f"{sub_name}_indices.npy", bmd_idx)
    np.save(out_base / "roi_subset" / f"{sub_name}_indices.npy", roi_idx)
    
    return meta


def main():
    print("=" * 70)
    print("BMD TARGET DATA CREATOR")
    print("=" * 70)
    print(f"Output directory: {OUT_BASE}")
    
    subjects = sorted([d.name for d in GLM_BASE.iterdir()
                       if d.is_dir() and d.name.startswith("sub-")])
    print(f"Subjects: {subjects}")
    
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    
    all_meta = {}
    for sub in subjects:
        meta = process_subject(sub, OUT_BASE)
        all_meta[sub] = meta
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n  {'Subject':10s} | {'BMDgeneral':>12s} | {'ROI subset':>12s} | Match?")
    print("  " + "─" * 55)
    
    bmd_sizes = set()
    roi_sizes = set()
    for sub in subjects:
        m = all_meta[sub]
        bmd_n = m["bmd_general_n_voxels"]
        roi_n = m["roi_subset_n_voxels"]
        bmd_sizes.add(bmd_n)
        roi_sizes.add(roi_n)
        match = "✅" if bmd_n == list(all_meta.values())[0]["bmd_general_n_voxels"] else "⚠️"
        print(f"  {sub:10s} | {bmd_n:>12,} | {roi_n:>12,} | {match}")
    
    print(f"\n  Unique BMDgeneral sizes: {bmd_sizes}")
    print(f"  Unique ROI subset sizes: {roi_sizes}")
    
    if len(bmd_sizes) == 1:
        print(f"  ✅ All subjects have SAME BMDgeneral dimension: {bmd_sizes.pop()}")
    else:
        print(f"  ⚠️  BMDgeneral dimensions DIFFER across subjects!")
    
    if len(roi_sizes) == 1:
        print(f"  ✅ All subjects have SAME ROI subset dimension: {roi_sizes.pop()}")
    else:
        print(f"  ⚠️  ROI subset dimensions DIFFER across subjects!")
    
    # Save metadata
    # Convert sets to lists for JSON serialization
    meta_json = {}
    for sub, m in all_meta.items():
        meta_json[sub] = {
            "bmd_general_n_voxels": m["bmd_general_n_voxels"],
            "roi_subset_n_voxels": m["roi_subset_n_voxels"],
            "roi_per_region_sizes": m["roi_per_region_sizes"],
        }
    
    with open(OUT_BASE / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    print(f"\n  Metadata saved to {OUT_BASE / 'metadata.json'}")
    
    # List output files
    print(f"\n  Output files:")
    for d in ["bmd_general", "roi_subset"]:
        target_dir = OUT_BASE / d
        files = sorted(target_dir.glob("*.npy"))
        total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        print(f"    {d}/: {len(files)} files, {total_mb:.0f} MB total")


if __name__ == "__main__":
    main()
