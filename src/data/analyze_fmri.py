"""Analyze fMRI H5 files and compute global normalization statistics.

Usage:
    # Analyze structure
    python src/data/analyze_fmri.py

    # Compute global stats (per-voxel mean/std from training clips)
    python src/data/analyze_fmri.py --compute_global_stats
    python src/data/analyze_fmri.py --compute_global_stats --subjects sub-01 sub-02
"""

import argparse
import os
import h5py
import numpy as np
from pathlib import Path


# Default training splits (matching brain_flow_direct configs)
TRAIN_SPLITS = {
    "friends": ["s1", "s2", "s3", "s4", "s5"],
    "movie10": ["bourne", "figures", "life", "wolf"],
}

EXCLUDED_SAMPLES_START = 5
EXCLUDED_SAMPLES_END = 5


def analyze_h5_file(filepath):
    print(f"\n[{filepath.name}]")
    try:
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            print(f"  Total keys (clips/runs): {len(keys)}")
            if len(keys) > 0:
                print("  Sample keys:", keys[:5])

                # Check the first key's shape
                first_key = keys[0]
                data = f[first_key]
                print(f"  Shape of '{first_key}': {data.shape}")
                print(f"  Data type: {data.dtype}")

            else:
                print("  File is empty or contains no keys.")
    except Exception as e:
        print(f"  Cannot open file. Error: {e}")
        print("  (Note: If this is a DataLad repository, make sure you ran 'datalad get' on the file.)")


def _get_fmri_filepath(fmri_dir, subject, task):
    """Construct path to fMRI h5 file."""
    subj_dir = Path(fmri_dir) / subject / "func"
    atlas_part = "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
    stem = f"{subject}_task-{task}_{atlas_part}"
    if task == "friends":
        return subj_dir / f"{stem}_desc-s123456_bold.h5"
    else:
        return subj_dir / f"{stem}_bold.h5"


def compute_global_stats(fmri_dir: Path, subjects: list[str]):
    """Compute per-voxel mean and std from all training clips using Welford's algorithm.

    Saves: {fmri_dir}/{subject}/stats/global_mean.npy and global_std.npy
    """
    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Computing global stats for {subject}")
        print(f"{'='*60}")

        # Welford's online algorithm for numerically stable mean/var
        n_total = 0
        mean_acc = None
        m2_acc = None

        for task, stim_types in TRAIN_SPLITS.items():
            fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
            if not fmri_path.exists():
                print(f"  [SKIP] {fmri_path} not found")
                continue

            with h5py.File(fmri_path, "r") as f:
                all_keys = list(f.keys())

                for stim_type in stim_types:
                    # Find matching clips for this stim_type
                    # Friends: stim_type 's1' → match keys with 's01e'
                    # Movie10: stim_type 'bourne' → match keys with 'bourne'
                    if task == "friends":
                        # s1 → s01, s2 → s02, etc.
                        season_num = stim_type.replace("s", "")
                        search_pattern = f"s{int(season_num):02d}e"
                        matched_keys = [k for k in all_keys if search_pattern in k]
                    else:
                        matched_keys = [k for k in all_keys if stim_type in k.lower() or stim_type in k]
                    if not matched_keys:
                        print(f"  [SKIP] No clips matching '{stim_type}' in {task}")
                        continue

                    for clip_key in matched_keys:
                        raw = f[clip_key][:].astype(np.float32)

                        # Trim excluded samples
                        end = len(raw) - EXCLUDED_SAMPLES_END if EXCLUDED_SAMPLES_END > 0 else len(raw)
                        data = raw[EXCLUDED_SAMPLES_START:end]  # (n_trs, n_voxels)
                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                        n_trs, n_voxels = data.shape

                        if mean_acc is None:
                            mean_acc = np.zeros(n_voxels, dtype=np.float64)
                            m2_acc = np.zeros(n_voxels, dtype=np.float64)

                        # Welford update per-TR
                        for i in range(n_trs):
                            n_total += 1
                            delta = data[i].astype(np.float64) - mean_acc
                            mean_acc += delta / n_total
                            delta2 = data[i].astype(np.float64) - mean_acc
                            m2_acc += delta * delta2

                        print(f"  [{task}/{stim_type}] {clip_key}: {n_trs} TRs × {n_voxels} voxels")

        if n_total == 0:
            print(f"  [ERROR] No training data found for {subject}!")
            continue

        # Finalize
        global_mean = mean_acc.astype(np.float32)
        global_std = np.sqrt(m2_acc / n_total).astype(np.float32)
        # Prevent division by zero
        global_std = np.where(global_std < 1e-8, 1.0, global_std)

        # Save
        stats_dir = fmri_dir / subject / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        np.save(stats_dir / "global_mean.npy", global_mean)
        np.save(stats_dir / "global_std.npy", global_std)

        print(f"\n  Total TRs: {n_total}")
        print(f"  Mean — min={global_mean.min():.4f}, max={global_mean.max():.4f}, avg={global_mean.mean():.4f}")
        print(f"  Std  — min={global_std.min():.4f}, max={global_std.max():.4f}, avg={global_std.mean():.4f}")
        print(f"  Saved to: {stats_dir}/")


def analyze(fmri_dir: Path):
    print(f"Analyzing fMRI directory: {fmri_dir}")

    subjects = sorted([d.name for d in fmri_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    print(f"Found {len(subjects)} subjects: {subjects}")

    for subj in subjects:
        print(f"\n{'='*50}\nAnalyzing Subject: {subj}\n{'='*50}")
        func_dir = fmri_dir / subj / "func"

        if not func_dir.exists():
            print(f"  No 'func' directory found for {subj}.")
            continue

        h5_files = list(func_dir.glob("*.h5"))
        print(f"  Found {len(h5_files)} .h5 files.")

        for h5 in h5_files:
            analyze_h5_file(h5)


def main():
    parser = argparse.ArgumentParser(description="Analyze fMRI H5 files and compute global stats")
    parser.add_argument("--compute_global_stats", action="store_true",
                        help="Compute per-voxel mean/std from training clips")
    parser.add_argument("--subjects", nargs="+", default=["sub-01"],
                        help="Subjects to process (default: sub-01)")
    parser.add_argument("--fmri_dir", type=str,
                        default="Data/algonauts_2025.competitors/fmri",
                        help="Path to fMRI directory")
    args = parser.parse_args()

    fmri_dir = Path(args.fmri_dir)
    if not fmri_dir.exists():
        print(f"Directory {fmri_dir} does not exist.")
        return

    if args.compute_global_stats:
        compute_global_stats(fmri_dir, args.subjects)
    else:
        analyze(fmri_dir)


if __name__ == "__main__":
    main()
