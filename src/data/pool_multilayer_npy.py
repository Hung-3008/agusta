#!/usr/bin/env python3
"""Convert multilayer NPY (T, L, D) → mean-pooled NPY (T, D).

Reduces ~92 GB multilayer files to ~14 GB mean-pooled files,
small enough to preload into RAM for fast training.

Usage:
    python src/data/pool_multilayer_npy.py
"""

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


INPUT_DIR = Path("Data/features_npy_multilayer")
OUTPUT_DIR = Path("Data/features_npy_pooled")

def main():
    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR}")
        return

    # Discover all modality subdirs
    mod_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(mod_dirs)} modalities: {[d.name for d in mod_dirs]}")

    total_files = 0
    total_in_bytes = 0
    total_out_bytes = 0

    for mod_dir in mod_dirs:
        mod_name = mod_dir.name
        out_mod_dir = OUTPUT_DIR / mod_name

        npy_files = sorted(mod_dir.rglob("*.npy"))
        print(f"\n[{mod_name}] {len(npy_files)} files...")

        for npy_path in tqdm(npy_files, desc=mod_name):
            rel = npy_path.relative_to(mod_dir)
            out_path = out_mod_dir / rel

            if out_path.exists():
                continue

            arr = np.load(npy_path)  # (T, L, D) or (T, D)
            total_in_bytes += npy_path.stat().st_size

            if arr.ndim == 3:
                pooled = arr.mean(axis=1).astype(np.float32)  # (T, D)
            else:
                pooled = arr.astype(np.float32)

            pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, pooled)
            total_out_bytes += out_path.stat().st_size
            total_files += 1

    print(f"\nDone! {total_files} files converted.")
    print(f"  Input:  {total_in_bytes / 1e9:.1f} GB")
    print(f"  Output: {total_out_bytes / 1e9:.1f} GB")
    print(f"  Ratio:  {total_in_bytes / max(total_out_bytes, 1):.1f}x reduction")


if __name__ == "__main__":
    main()
