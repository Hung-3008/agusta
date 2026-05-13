"""
BMD fMRI Data Inspector
=======================
Checks:
1. Prepared betas shape/dtype per subject
2. ROI atlas: names, voxel counts, coverage
3. Whether ROIs can map whole-brain to smaller space
4. Cross-subject consistency

Usage:
    python scripts/check_bmd_fmri.py
"""

import pickle
import numpy as np
from pathlib import Path
import sys
import time

BMD_BASE = Path("Data/BOLDMomentsDataset")
GLM_BASE = BMD_BASE / "derivatives/versionB/MNI152/GLM"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def check_rois(sub_dir):
    """Check all ROI files for a subject."""
    roi_dir = sub_dir / "ROIs"
    if not roi_dir.exists():
        print(f"  ❌ ROIs directory not found")
        return None

    roi_files = sorted(roi_dir.glob("*.pkl"))
    print(f"\n  Total ROI files: {len(roi_files)}")

    roi_info = {}
    for rf in roi_files:
        roi_name = rf.stem.replace("ROI-", "").replace("_indices", "")
        data = load_pkl(rf)
        
        if isinstance(data, np.ndarray):
            if data.dtype == bool:
                n_voxels = int(data.sum())
                total_size = data.shape[0]
                roi_info[roi_name] = {
                    "n_voxels": n_voxels,
                    "type": "bool_mask",
                    "mask_size": total_size,
                }
            else:
                n_voxels = len(data)
                roi_info[roi_name] = {
                    "n_voxels": n_voxels,
                    "type": f"indices_{data.dtype}",
                    "shape": data.shape,
                    "min": int(data.min()),
                    "max": int(data.max()),
                }
        elif isinstance(data, (list, tuple)):
            n_voxels = len(data)
            roi_info[roi_name] = {
                "n_voxels": n_voxels,
                "type": f"list_{type(data[0]).__name__}" if data else "empty_list",
            }
        else:
            roi_info[roi_name] = {
                "n_voxels": -1,
                "type": str(type(data)),
            }

    # Print summary
    print(f"\n  {'ROI Name':25s} | {'#Voxels':>8s} | {'Type':>20s} | Extra")
    print("  " + "─" * 80)
    
    general_voxels = 0
    sum_specific = 0
    
    for name, info in sorted(roi_info.items()):
        extra = ""
        if "min" in info:
            extra = f"idx range: [{info['min']}, {info['max']}]"
        elif "mask_size" in info:
            extra = f"mask total: {info['mask_size']}"
        
        marker = "🧠" if name == "BMDgeneral" else "  "
        print(f"  {marker}{name:23s} | {info['n_voxels']:>8d} | {info['type']:>20s} | {extra}")
        
        if name == "BMDgeneral":
            general_voxels = info["n_voxels"]
        else:
            sum_specific += info["n_voxels"]
    
    print(f"\n  📊 BMDgeneral (whole brain): {general_voxels:,} voxels")
    print(f"  📊 Sum of specific ROIs:     {sum_specific:,} voxels")
    if general_voxels > 0:
        print(f"  📊 ROI coverage:             {sum_specific/general_voxels*100:.1f}% of whole brain")
        print(f"  📊 Compression ratio:        {general_voxels} → {sum_specific} ({general_voxels/max(sum_specific,1):.1f}x)")
    
    # Group by region (left/right pairs)
    print(f"\n  ── Bilateral ROI pairs ──")
    left_rois = {n.replace("l", "", 1): info for n, info in roi_info.items() if n.startswith("l")}
    right_rois = {n.replace("r", "", 1): info for n, info in roi_info.items() if n.startswith("r")}
    all_regions = sorted(set(left_rois.keys()) | set(right_rois.keys()))
    
    total_bilateral = 0
    for region in all_regions:
        l = left_rois.get(region, {}).get("n_voxels", 0)
        r = right_rois.get(region, {}).get("n_voxels", 0)
        total = l + r
        total_bilateral += total
        print(f"    {region:20s}: L={l:5d} + R={r:5d} = {total:6d}")
    
    print(f"    {'TOTAL':20s}: {total_bilateral:>19d}")
    
    return roi_info


def check_betas(sub_dir, sub_name):
    """Check prepared betas for a subject."""
    betas_dir = sub_dir / "prepared_betas"
    if not betas_dir.exists():
        print(f"  ❌ prepared_betas not found")
        return None

    results = {}
    
    # Train betas
    train_file = betas_dir / f"{sub_name}_organized_betas_task-train_normalized.pkl"
    test_file = betas_dir / f"{sub_name}_organized_betas_task-test_normalized.pkl"
    
    for label, fpath in [("train", train_file), ("test", test_file)]:
        if not fpath.exists():
            print(f"  ❌ {label} betas not found: {fpath.name}")
            continue
        
        sz_mb = fpath.stat().st_size / 1024 / 1024
        print(f"\n  Loading {label} betas ({sz_mb:.0f} MB)...", end=" ", flush=True)
        t0 = time.time()
        data = load_pkl(fpath)
        dt = time.time() - t0
        print(f"done ({dt:.1f}s)")
        
        if isinstance(data, np.ndarray):
            print(f"    Shape: {data.shape}")
            print(f"    Dtype: {data.dtype}")
            print(f"    Range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"    Mean:  {data.mean():.4f}")
            print(f"    Std:   {data.std():.4f}")
            print(f"    NaN:   {np.isnan(data).sum()}")
            print(f"    Inf:   {np.isinf(data).sum()}")
            results[label] = {"shape": data.shape, "dtype": str(data.dtype)}
        elif isinstance(data, dict):
            print(f"    Type: dict with keys: {list(data.keys())[:5]}")
            for k, v in list(data.items())[:2]:
                if isinstance(v, np.ndarray):
                    print(f"      '{k}': shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"    Type: {type(data)}")
    
    # Also check noise ceiling
    for nc_file in sorted(betas_dir.glob("*noiseceiling*")):
        print(f"\n  Noise ceiling: {nc_file.name}")
        nc = load_pkl(nc_file)
        if isinstance(nc, np.ndarray):
            print(f"    Shape: {nc.shape}, dtype: {nc.dtype}")
            print(f"    Range: [{nc.min():.4f}, {nc.max():.4f}]")
            print(f"    Mean:  {nc.mean():.4f}")
    
    return results


def check_cross_subject_consistency():
    """Check if all subjects have same voxel count."""
    print(f"\n{'=' * 80}")
    print("CROSS-SUBJECT CONSISTENCY")
    print("=" * 80)
    
    subjects = sorted([d.name for d in GLM_BASE.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    
    sub_voxels = {}
    for sub in subjects:
        sub_dir = GLM_BASE / sub
        
        # Check ROI general
        gen_file = sub_dir / "ROIs" / "ROI-BMDgeneral_indices.pkl"
        if gen_file.exists():
            gen = load_pkl(gen_file)
            if isinstance(gen, np.ndarray):
                if gen.dtype == bool:
                    n = int(gen.sum())
                else:
                    n = len(gen)
            else:
                n = len(gen)
            sub_voxels[sub] = n
            
        # Check betas shape from file size
        train_file = sub_dir / "prepared_betas" / f"{sub}_organized_betas_task-train_normalized.pkl"
        if train_file.exists():
            sz = train_file.stat().st_size / 1024 / 1024
            print(f"  {sub}: BMDgeneral={sub_voxels.get(sub, '?'):>6} voxels, train_betas={sz:.0f} MB")
    
    # Check consistency
    unique_counts = set(sub_voxels.values())
    if len(unique_counts) == 1:
        print(f"\n  ✅ All subjects have same voxel count: {unique_counts.pop():,}")
    else:
        print(f"\n  ⚠️  Voxel counts differ across subjects: {unique_counts}")


def main():
    print("=" * 80)
    print("BMD fMRI DATA INSPECTOR")
    print("=" * 80)
    
    subjects = sorted([d.name for d in GLM_BASE.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    print(f"\nSubjects found: {subjects}")
    print(f"Total: {len(subjects)}")
    
    # ── Detailed inspection on sub-01 ──
    sub01 = GLM_BASE / "sub-01"
    
    print(f"\n{'=' * 80}")
    print("ROI / ATLAS ANALYSIS (sub-01)")
    print("=" * 80)
    roi_info = check_rois(sub01)
    
    print(f"\n{'=' * 80}")
    print("PREPARED BETAS (sub-01)")
    print("=" * 80)
    betas_info = check_betas(sub01, "sub-01")
    
    # ── Cross-subject ──
    check_cross_subject_consistency()
    
    # ── Summary & Recommendations ──
    print(f"\n{'=' * 80}")
    print("SUMMARY & ATLAS MAPPING OPTIONS")
    print("=" * 80)
    
    if roi_info:
        general = roi_info.get("BMDgeneral", {}).get("n_voxels", 0)
        specific_rois = {k: v for k, v in roi_info.items() if k != "BMDgeneral"}
        sum_specific = sum(v["n_voxels"] for v in specific_rois.values())
        n_rois = len(specific_rois)
        
        print(f"""
  ┌─────────────────────────────────────────────────────┐
  │ Whole brain (BMDgeneral):  {general:>6,} voxels           │
  │ ROI-level mapping:         {n_rois:>6} ROIs → {sum_specific:>6,} voxels │
  │ ROI coverage:              {sum_specific/max(general,1)*100:>5.1f}%                 │
  │ Mean voxels per ROI:       {sum_specific//max(n_rois,1):>6,}                   │
  └─────────────────────────────────────────────────────┘
        
  Atlas mapping options for dimensionality reduction:
  
  1. ROI-LEVEL AVERAGING (46 ROIs → 46-dim)
     - Average voxels within each ROI → one value per ROI
     - Extreme compression: {general:,} → 46
     - May lose fine-grained spatial info
  
  2. ROI SUBSET (visual cortex only)
     - V1d, V1v, V2d, V2v, V3d, V3v, V3ab, hV4, MT, LOC, FFA, EBA, OFA, PPA
     - ~28 ROIs (14 bilateral pairs)
     - Good for visual encoding models
  
  3. WHOLE-BRAIN with BMDgeneral mask
     - Use all {general:,} voxels
     - Full information but high-dimensional
  
  4. PCA/DIMENSIONALITY REDUCTION
     - Apply PCA on whole-brain betas
     - Reduce to ~100-500 components
     - Data-driven, preserves max variance
""")


if __name__ == "__main__":
    main()
