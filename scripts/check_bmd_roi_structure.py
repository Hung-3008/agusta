"""Inspect BMD ROI structure and check compatibility with NetworkSubjectLayers.

This script:
1. Lists all 46 functional ROIs with voxel counts
2. Shows how they differ from Algonauts Schaefer 7-network parcellation
3. Checks if a grouped network_head is feasible for BMD
4. Verifies cross-subject consistency

Usage:
    python scripts/check_bmd_roi_structure.py
"""

import pickle
from pathlib import Path
from collections import defaultdict

BMD_DIR = Path("Data/BOLDMomentsDataset")
DERIV_BASE = BMD_DIR / "derivatives" / "versionB" / "MNI152" / "GLM"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_roi_name(filename):
    """Parse 'ROI-lV1d_indices.pkl' → ('l', 'V1d')."""
    stem = filename.replace("ROI-", "").replace("_indices", "")
    if stem.startswith("BMDgeneral"):
        return None, "BMDgeneral"
    hemi = stem[0]  # 'l' or 'r'
    roi = stem[1:]
    return hemi, roi


def main():
    print("=" * 70)
    print("BMD ROI STRUCTURE INSPECTOR")
    print("=" * 70)

    sub = "sub-01"
    roi_dir = DERIV_BASE / sub / "ROIs"
    roi_files = sorted(roi_dir.glob("ROI-*_indices.pkl"))

    print(f"\nSubject: {sub}")
    print(f"ROI dir: {roi_dir}")
    print(f"Total ROI files: {len(roi_files)}")

    # ── 1. Load each ROI ──
    print(f"\n{'Hemi':>4s}  {'ROI Name':20s} | {'Voxels':>6s}")
    print("-" * 40)

    roi_info = {}       # full_name → n_vox
    roi_by_region = defaultdict(dict)  # region → {hemi: n_vox}

    for f in roi_files:
        hemi, region = parse_roi_name(f.stem)
        if region == "BMDgeneral":
            continue
        try:
            data = load_pkl(f)
            n_vox = data[0].shape[0]
            full_name = f"{hemi}{region}"
            roi_info[full_name] = n_vox
            roi_by_region[region][hemi] = n_vox
            print(f"  {hemi:>2s}   {region:20s} | {n_vox:>6d}")
        except Exception as e:
            print(f"  ?    {f.stem:20s} | ERROR: {e}")

    total_vox = sum(roi_info.values())
    print(f"\n  {'TOTAL':25s} | {total_vox:>6d}")
    print(f"  Unique regions: {len(roi_by_region)}")
    print(f"  ROIs per hemisphere: l={sum(1 for v in roi_by_region.values() if 'l' in v)}, "
          f"r={sum(1 for v in roi_by_region.values() if 'r' in v)}")

    # ── 2. Group by functional category ──
    print(f"\n{'=' * 70}")
    print("FUNCTIONAL GROUPING")
    print(f"{'=' * 70}")

    categories = {
        "Early Visual":  ["V1d", "V1v", "V2d", "V2v", "V3d", "V3v", "V3ab", "hV4"],
        "Scene/Place":   ["PPA", "RSC", "TOS"],
        "Body":          ["EBA"],
        "Face":          ["FFA", "OFA"],
        "Object":        ["LOC"],
        "Motion":        ["MT"],
        "Temporal/STS":  ["STS"],
        "Parietal":      ["IPS0", "IPS1-2-3", "7AL"],
        "Somatosensory": ["BA2", "PFt", "PFop"],
    }

    grouped_regions = set()
    category_voxels = {}

    for cat, patterns in categories.items():
        cat_rois = []
        cat_total = 0
        for pat in patterns:
            if pat in roi_by_region:
                for hemi, n_vox in roi_by_region[pat].items():
                    cat_rois.append(f"{hemi}{pat}({n_vox})")
                    cat_total += n_vox
                grouped_regions.add(pat)
        if cat_rois:
            category_voxels[cat] = cat_total
            print(f"\n  {cat} ({cat_total} voxels):")
            print(f"    {', '.join(cat_rois)}")

    # Ungrouped
    ungrouped = {r: roi_by_region[r] for r in roi_by_region if r not in grouped_regions}
    if ungrouped:
        ug_total = sum(sum(h.values()) for h in ungrouped.values())
        print(f"\n  UNGROUPED ({ug_total} voxels):")
        for r, hemis in ungrouped.items():
            parts = [f"{h}{r}({n})" for h, n in hemis.items()]
            print(f"    {', '.join(parts)}")

    # ── 3. Compare with Algonauts ──
    print(f"\n{'=' * 70}")
    print("COMPARISON: Algonauts vs BMD")
    print(f"{'=' * 70}")

    schaefer_7net = {
        "Visual": 150, "SomMot": 148, "DorsAttn": 132,
        "SalVentAttn": 136, "Limbic": 76, "Cont": 122, "Default": 236,
    }

    print(f"\n  {'':20s} {'Algonauts':>12s}  {'BMD':>12s}")
    print(f"  {'-'*50}")
    print(f"  {'Parcellation':20s} {'Schaefer7Net':>12s}  {'46 fROIs':>12s}")
    print(f"  {'Total voxels':20s} {sum(schaefer_7net.values()):>12d}  {total_vox:>12d}")
    print(f"  {'Networks/Groups':20s} {7:>12d}  {len(category_voxels):>12d}")
    print(f"  {'Hemisphere':20s} {'Both(LH+RH)':>12s}  {'Both(l+r)':>12s}")

    print(f"\n  Algonauts network sizes: {list(schaefer_7net.values())}")
    print(f"  BMD group sizes: {list(category_voxels.values())}")

    # ── 4. Possible network_counts for BMD ──
    print(f"\n{'=' * 70}")
    print("NETWORK HEAD OPTIONS FOR BMD")
    print(f"{'=' * 70}")

    print(f"""
  ❌ network_head=true + DEFAULT Schaefer counts
     → Expects {sum(schaefer_7net.values())} output voxels, but BMD has {total_vox}
     → WILL CRASH

  ✅ Option A: network_head=false (RECOMMENDED)
     → Single SubjectLayers(768, {total_vox}, n_subjects)
     → Simple, no grouping needed
     → Params per subject: 768 × {total_vox} = {768*total_vox:,}

  ⚠️  Option B: network_head=true + CUSTOM network_counts
     → {len(category_voxels)} groups: {list(category_voxels.values())}
     → Names: {list(category_voxels.keys())}
     → More structured but requires ROI ordering to match target creation
""")

    # ── 5. Cross-subject consistency ──
    print(f"{'=' * 70}")
    print("CROSS-SUBJECT ROI CONSISTENCY")
    print(f"{'=' * 70}")

    subjects = sorted([d.name for d in DERIV_BASE.iterdir()
                       if d.is_dir() and d.name.startswith("sub-")])

    ref_rois = set(roi_info.keys())

    for subj in subjects:
        s_roi_dir = DERIV_BASE / subj / "ROIs"
        s_rois = {}
        for f in s_roi_dir.glob("ROI-*_indices.pkl"):
            hemi, region = parse_roi_name(f.stem)
            if region == "BMDgeneral":
                continue
            try:
                data = load_pkl(f)
                full_name = f"{hemi}{region}"
                s_rois[full_name] = data[0].shape[0]
            except:
                pass

        same_names = set(s_rois.keys()) == ref_rois
        same_total = sum(s_rois.values())

        # Check per-ROI voxel counts
        mismatches = []
        for name in ref_rois:
            if name in s_rois and s_rois[name] != roi_info[name]:
                mismatches.append(f"{name}: {roi_info[name]}→{s_rois[name]}")

        if same_names and not mismatches:
            print(f"  {subj}: {len(s_rois)} ROIs, {same_total} voxels ✅")
        else:
            print(f"  {subj}: {len(s_rois)} ROIs, {same_total} voxels ⚠️")
            if not same_names:
                missing = ref_rois - set(s_rois.keys())
                extra = set(s_rois.keys()) - ref_rois
                if missing:
                    print(f"    Missing: {missing}")
                if extra:
                    print(f"    Extra: {extra}")
            if mismatches:
                print(f"    Voxel count changes: {mismatches[:5]}")


if __name__ == "__main__":
    main()
