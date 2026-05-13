"""
BMD fMRI Data Inspector v2
==========================
Properly handles nested pickle structures (list of arrays, tuples, etc.)

Usage:
    python scripts/check_bmd_fmri_v2.py
"""

import pickle
import numpy as np
from pathlib import Path
import sys


BMD_BASE = Path("Data/BOLDMomentsDataset")
GLM_BASE = BMD_BASE / "derivatives/versionB/MNI152/GLM"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def describe_obj(obj, depth=0, max_depth=3, prefix=""):
    """Recursively describe a Python object's structure."""
    indent = "    " * depth
    
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ('f', 'i', 'u'):  # numeric
            print(f"{indent}{prefix}ndarray: shape={obj.shape}, dtype={obj.dtype}, "
                  f"range=[{obj.min():.4f}, {obj.max():.4f}], "
                  f"mean={obj.mean():.4f}, NaN={np.isnan(obj).sum() if obj.dtype.kind == 'f' else 0}")
        elif obj.dtype == object:
            print(f"{indent}{prefix}ndarray(object): shape={obj.shape}, "
                  f"element types: {set(type(x).__name__ for x in obj.flat)}")
            if depth < max_depth and obj.size <= 5:
                for i, item in enumerate(obj.flat):
                    describe_obj(item, depth + 1, max_depth, prefix=f"[{i}] ")
        else:
            print(f"{indent}{prefix}ndarray: shape={obj.shape}, dtype={obj.dtype}")
        return obj
    elif isinstance(obj, (list, tuple)):
        type_name = "list" if isinstance(obj, list) else "tuple"
        print(f"{indent}{prefix}{type_name} of {len(obj)} elements:")
        if depth < max_depth:
            for i, item in enumerate(obj):
                if i >= 5:  # limit display
                    print(f"{indent}    ... ({len(obj) - 5} more)")
                    break
                describe_obj(item, depth + 1, max_depth, prefix=f"[{i}] ")
        return obj
    elif isinstance(obj, dict):
        print(f"{indent}{prefix}dict with {len(obj)} keys: {list(obj.keys())[:10]}")
        if depth < max_depth:
            for i, (k, v) in enumerate(obj.items()):
                if i >= 5:
                    print(f"{indent}    ... ({len(obj) - 5} more)")
                    break
                describe_obj(v, depth + 1, max_depth, prefix=f"'{k}': ")
        return obj
    else:
        print(f"{indent}{prefix}{type(obj).__name__}: {repr(obj)[:100]}")
        return obj


def check_rois_deep(sub_dir):
    """Deep inspection of ROI files."""
    roi_dir = sub_dir / "ROIs"
    roi_files = sorted(roi_dir.glob("*.pkl"))
    
    print(f"\n  Total ROI files: {len(roi_files)}")
    
    # First: fully inspect BMDgeneral and one specific ROI
    print(f"\n  ── Deep inspect: BMDgeneral ──")
    gen = load_pkl(roi_dir / "ROI-BMDgeneral_indices.pkl")
    describe_obj(gen, depth=1)
    
    # If it's a list of arrays, figure out what it represents
    if isinstance(gen, (list, tuple)):
        print(f"\n  Interpreting BMDgeneral structure:")
        for i, item in enumerate(gen):
            if isinstance(item, np.ndarray):
                print(f"    Element [{i}]: shape={item.shape}, dtype={item.dtype}, "
                      f"min={item.min()}, max={item.max()}, unique={len(np.unique(item))}")
        
        # Common pattern: list of [x_coords, y_coords, z_coords] or [indices, data, ...]
        if len(gen) == 3 and all(isinstance(g, np.ndarray) for g in gen):
            if all(g.ndim == 1 for g in gen):
                # Likely 3D voxel coordinates
                n_voxels = len(gen[0])
                print(f"\n    → Likely 3D voxel coordinates (x,y,z): {n_voxels} voxels")
            elif gen[0].ndim == 1 and gen[1].ndim >= 1:
                print(f"\n    → Element sizes: {[g.shape for g in gen]}")
    
    print(f"\n  ── Deep inspect: lV1d ──")
    v1d = load_pkl(roi_dir / "ROI-lV1d_indices.pkl")
    describe_obj(v1d, depth=1)
    
    if isinstance(v1d, (list, tuple)):
        for i, item in enumerate(v1d):
            if isinstance(item, np.ndarray):
                print(f"    Element [{i}]: shape={item.shape}, dtype={item.dtype}, "
                      f"min={item.min()}, max={item.max()}, unique={len(np.unique(item))}")
    
    # Now summarize all ROIs
    print(f"\n  ── All ROIs summary ──")
    roi_voxel_counts = {}
    
    for rf in roi_files:
        roi_name = rf.stem.replace("ROI-", "").replace("_indices", "")
        data = load_pkl(rf)
        
        if isinstance(data, (list, tuple)):
            # ROI pkl = (coords_array, flat_indices_array, Nifti1Image)
            # coords_array shape: (N_voxels, 3) — 3D voxel coordinates
            # flat_indices_array shape: (N_voxels, 1) — flattened indices
            first = data[0]
            if isinstance(first, np.ndarray):
                n_voxels = first.shape[0]  # first dim = number of voxels
            else:
                n_voxels = len(data)
        elif isinstance(data, np.ndarray):
            if data.dtype == bool:
                n_voxels = int(data.sum())
            else:
                n_voxels = len(data)
        else:
            n_voxels = -1
        
        roi_voxel_counts[roi_name] = n_voxels
    
    # Print table
    print(f"\n  {'ROI Name':25s} | {'#Voxels':>8s}")
    print("  " + "─" * 40)
    
    general_n = roi_voxel_counts.get("BMDgeneral", 0)
    sum_specific = 0
    
    for name in sorted(roi_voxel_counts.keys()):
        n = roi_voxel_counts[name]
        marker = "🧠" if name == "BMDgeneral" else "  "
        print(f"  {marker}{name:23s} | {n:>8,d}")
        if name != "BMDgeneral":
            sum_specific += n
    
    print(f"\n  📊 BMDgeneral (whole brain mask): {general_n:,} voxels")
    print(f"  📊 Sum of all specific ROIs:      {sum_specific:,} voxels")
    if general_n > 0:
        print(f"  📊 ROI coverage of whole brain:   {sum_specific / general_n * 100:.1f}%")
    
    # Bilateral pairs
    print(f"\n  ── Bilateral ROI pairs ──")
    left_rois = {}
    right_rois = {}
    for n, v in roi_voxel_counts.items():
        if n == "BMDgeneral":
            continue
        if n.startswith("l"):
            left_rois[n[1:]] = v
        elif n.startswith("r"):
            right_rois[n[1:]] = v
    
    all_regions = sorted(set(left_rois.keys()) | set(right_rois.keys()))
    total_bilateral = 0
    for region in all_regions:
        l = left_rois.get(region, 0)
        r = right_rois.get(region, 0)
        total = l + r
        total_bilateral += total
        print(f"    {region:20s}: L={l:>6,} + R={r:>6,} = {total:>7,}")
    print(f"    {'TOTAL':20s}: {total_bilateral:>23,}")
    
    return roi_voxel_counts


def check_betas_deep(sub_dir, sub_name):
    """Deep inspection of betas structure."""
    betas_dir = sub_dir / "prepared_betas"
    
    for label, task in [("train", "train"), ("test", "test")]:
        fpath = betas_dir / f"{sub_name}_organized_betas_task-{task}_normalized.pkl"
        if not fpath.exists():
            print(f"\n  ❌ {label} betas not found")
            continue
        
        sz_mb = fpath.stat().st_size / 1024 / 1024
        print(f"\n  ── {label.upper()} betas ({sz_mb:.0f} MB) ──")
        data = load_pkl(fpath)
        describe_obj(data, depth=0, max_depth=2)
    
    # Noise ceilings
    for nc_file in sorted(betas_dir.glob("*noiseceiling*")):
        print(f"\n  ── {nc_file.name} ──")
        nc = load_pkl(nc_file)
        describe_obj(nc, depth=0, max_depth=2)
    
    # Contrasts
    for ct_file in sorted(betas_dir.glob("*contrasts*")):
        print(f"\n  ── {ct_file.name} ──")
        ct = load_pkl(ct_file)
        describe_obj(ct, depth=0, max_depth=2)


def check_session_files(sub_dir):
    """Check per-session GLM files."""
    sessions = sorted([d.name for d in sub_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")])
    print(f"\n  Sessions: {sessions}")
    
    if sessions:
        ses_dir = sub_dir / sessions[0]
        print(f"\n  ── {sessions[0]} contents ──")
        for f in sorted(ses_dir.iterdir()):
            sz_mb = f.stat().st_size / 1024 / 1024
            try:
                if f.suffix == ".npy":
                    arr = np.load(f, allow_pickle=True)
                    print(f"    {f.name} ({sz_mb:.1f} MB): ", end="")
                    describe_obj(arr, depth=0, max_depth=1)
                elif f.suffix == ".pkl":
                    data = load_pkl(f)
                    print(f"    {f.name} ({sz_mb:.1f} MB): ", end="")
                    describe_obj(data, depth=0, max_depth=1)
            except Exception as e:
                print(f"    {f.name} ({sz_mb:.1f} MB): ERROR - {e}")


def cross_subject_summary():
    """Quick cross-subject comparison."""
    subjects = sorted([d.name for d in GLM_BASE.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    
    print(f"\n  {'Subject':10s} | {'Train MB':>10s} | {'Test MB':>10s} | {'BMDgen voxels':>14s} | {'~Inferred':>10s}")
    print("  " + "─" * 72)
    
    for sub in subjects:
        sub_dir = GLM_BASE / sub
        train_f = sub_dir / "prepared_betas" / f"{sub}_organized_betas_task-train_normalized.pkl"
        test_f = sub_dir / "prepared_betas" / f"{sub}_organized_betas_task-test_normalized.pkl"
        
        train_mb = train_f.stat().st_size / 1024 / 1024 if train_f.exists() else 0
        test_mb = test_f.stat().st_size / 1024 / 1024 if test_f.exists() else 0
        
        gen_f = sub_dir / "ROIs" / "ROI-BMDgeneral_indices.pkl"
        if gen_f.exists():
            gen = load_pkl(gen_f)
            if isinstance(gen, (list, tuple)):
                first = gen[0]
                n_vox = first.shape[0] if isinstance(first, np.ndarray) else len(gen)
            elif isinstance(gen, np.ndarray):
                n_vox = int(gen.sum()) if gen.dtype == bool else len(gen)
            else:
                n_vox = len(gen)
        else:
            n_vox = -1
        
        # Also infer voxels from file size: train = (1000 * reps * n_vox * 8 bytes) + overhead
        # Approximate: n_vox ≈ train_bytes / (1000 * 3 * 8)
        inferred_vox = int(train_f.stat().st_size / (1000 * 3 * 8)) if train_f.exists() else 0
        
        print(f"  {sub:10s} | {train_mb:>10.0f} | {test_mb:>10.0f} | {n_vox:>14,} | ~{inferred_vox:>8,}")


def main():
    print("=" * 80)
    print("BMD fMRI DATA INSPECTOR v2 — Deep Structure Analysis")
    print("=" * 80)
    
    subjects = sorted([d.name for d in GLM_BASE.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    print(f"\nSubjects: {subjects} (total: {len(subjects)})")
    
    sub01 = GLM_BASE / "sub-01"
    
    # ── ROI Analysis ──
    print(f"\n{'=' * 80}")
    print("1. ROI / ATLAS DEEP ANALYSIS (sub-01)")
    print("=" * 80)
    roi_info = check_rois_deep(sub01)
    
    # ── Betas Analysis ──
    print(f"\n{'=' * 80}")
    print("2. PREPARED BETAS DEEP ANALYSIS (sub-01)")
    print("=" * 80)
    check_betas_deep(sub01, "sub-01")
    
    # ── Session Files ──
    print(f"\n{'=' * 80}")
    print("3. SESSION FILES (sub-01)")
    print("=" * 80)
    check_session_files(sub01)
    
    # ── Cross-subject ──
    print(f"\n{'=' * 80}")
    print("4. CROSS-SUBJECT SUMMARY")
    print("=" * 80)
    cross_subject_summary()
    
    # ── Final Summary ──
    general_n = roi_info.get("BMDgeneral", 0)
    specific = {k: v for k, v in roi_info.items() if k != "BMDgeneral"}
    sum_roi = sum(specific.values())
    n_rois = len(specific)
    
    print(f"\n{'=' * 80}")
    print("5. CONCLUSIONS")
    print("=" * 80)
    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │ DATA STRUCTURE                                              │
  ├──────────────────────────────────────────────────────────────┤
  │ Betas (train): (1000, 3_reps, 108219_voxels) float64        │
  │ Betas (test):  (102, 10_reps, 108219_voxels) float64        │
  │ Total brain voxels per subject: 108,219                     │
  │                                                             │
  │ ROI pkl structure:                                          │
  │   [0] coords array:  (N_voxels, 3) int64 — 3D MNI coords   │
  │   [1] flat indices:   (N_voxels, 1) int64 — into 108219-dim │
  │   [2] Nifti1Image:    reference NIfTI header                │
  └──────────────────────────────────────────────────────────────┘
  
  ┌──────────────────────────────────────────────────────────────┐
  │ ATLAS / ROI MAPPING                                         │
  ├──────────────────────────────────────────────────────────────┤
  │ BMDgeneral (whole brain mask): {general_n:>6,} voxels              │
  │ Specific ROIs: {n_rois:>3} ROIs covering {sum_roi:>6,} voxels            │
  │ ROI coverage:  {sum_roi/max(general_n,1)*100:>5.1f}% of whole brain                     │
  └──────────────────────────────────────────────────────────────┘
  
  Available ROIs (23 bilateral pairs):
    Visual early:       V1d/v, V2d/v, V3d/v, V3ab, hV4
    Visual motion:      MT
    Category-selective: FFA, EBA, LOC, OFA, PPA, RSC, TOS
    Higher-order:       STS, IPS0, IPS1-2-3, BA2, PFop, PFt, 7AL
  
  DIMENSIONALITY REDUCTION OPTIONS:
  
  1. ROI-LEVEL (use flat_indices [1] to extract & average)
     108,219 voxels → {n_rois} ROI means → {n_rois}-dim vector
     Pro: interpretable, neuroscience-grounded
     Con: loses within-ROI spatial info
  
  2. ROI VOXEL SUBSET (concatenate ROI voxels)
     108,219 → {sum_roi:,} voxels (only ROI-covered voxels)
     Pro: retains spatial detail in functional regions
     Con: still high-dim, drops non-ROI voxels
  
  3. WHOLE-BRAIN + PCA
     108,219 → K principal components (e.g. K=200-1000)
     Pro: data-driven, max variance preserved
     Con: not neuroscience-interpretable
  
  4. EXTERNAL ATLAS (Schaefer, Glasser HCP-MMP1)
     Apply standard parcellation to 108,219 MNI voxels
     E.g. Schaefer-400: 108,219 → 400 parcels
     Pro: standardized, comparable across studies
     Requires: downloading atlas NIfTI & resampling to match space
""")


if __name__ == "__main__":
    main()
