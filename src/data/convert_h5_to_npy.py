#!/usr/bin/env python3
"""
Convert pre-extracted H5 feature files → per-clip NPY files.

Layer aggregation:
  mean  — mean across all layers (default)
  last  — use last layer only
  cat   — concatenate last N layers (use --n_layers to control N)

For omni (tr_tokens_dim format): mean-pool tokens.

Output structure mirrors H5 but as flat per-clip npy files:
  Data/features_npy/{modality}/{movie_type}_{stimulus_type}/{clip_name}.npy

Shape after conversion:
  mean/last: (dim, n_trs)
  cat:       (N*dim, n_trs)  — N = n_layers
  omni:      (dim, n_trs)    — tokens mean-pooled

Usage:
  python src/data/convert_h5_to_npy.py
  python src/data/convert_h5_to_npy.py --aggregation last
  python src/data/convert_h5_to_npy.py --aggregation cat --n_layers 4
  python src/data/convert_h5_to_npy.py --modalities video audio
  python src/data/convert_h5_to_npy.py --dry_run
"""
import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


MODALITY_CONFIGS = {
    "video": {
        "subdir": "video",
        "h5_key": "video",
        "data_format": "layers_dim_tr",   # (n_layers, dim, n_trs)
    },
    "audio": {
        "subdir": "audio",
        "h5_key": "audio",
        "data_format": "layers_dim_tr",
    },
    "text": {
        "subdir": "text",
        "h5_key": "text",
        "data_format": "layers_dim_tr",
    },
    "omni": {
        "subdir": "omni",
        "h5_key": "omni",
        "data_format": "tr_tokens_dim",   # (n_trs, n_tokens, dim)
    },
}


def convert_clip(data: np.ndarray, data_format: str, aggregation: str,
                 keep_tokens: bool = False, n_layers: int = 4) -> np.ndarray:
    """
    Convert raw H5 clip data to output array.

    Args:
        data: raw array from H5
        data_format: 'layers_dim_tr' or 'tr_tokens_dim'
        aggregation: 'mean', 'last', or 'cat' (only for layers_dim_tr)
        keep_tokens: if True, keep individual tokens for tr_tokens_dim
                     output shape: (n_tokens, dim, n_trs) instead of (dim, n_trs)
        n_layers: number of last layers to concatenate (only for aggregation='cat')

    Returns:
        np.ndarray of shape (dim, n_trs), (N*dim, n_trs), or (n_tokens, dim, n_trs)
    """
    if data_format == "tr_tokens_dim":
        if keep_tokens:
            # (n_trs, n_tokens, dim) → (n_tokens, dim, n_trs)
            return data.transpose(1, 2, 0).astype(np.float32)
        else:
            # (n_trs, n_tokens, dim) → mean-pool tokens → (dim, n_trs)
            pooled = data.mean(axis=1)   # (n_trs, dim)
            return pooled.T.astype(np.float32)  # (dim, n_trs)
    else:
        # layers_dim_tr: (n_layers, dim, n_trs)
        if aggregation == "last":
            return data[-1].astype(np.float32)   # (dim, n_trs)
        elif aggregation == "cat":
            # Concatenate last N layers: (N, dim, T) → (N*dim, T)
            n = min(n_layers, data.shape[0])
            selected = data[-n:]  # last N layers: (N, dim, T)
            # Reshape: (N, dim, T) → (N*dim, T)
            N, D, T = selected.shape
            return selected.reshape(N * D, T).astype(np.float32)
        else:  # mean
            return data.mean(axis=0).astype(np.float32)  # (dim, n_trs)


def convert_modality(
    feat_root: Path,
    out_root: Path,
    modality: str,
    aggregation: str = "mean",
    n_layers: int = 4,
    keep_omni_tokens: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    cfg = MODALITY_CONFIGS[modality]
    in_dir = feat_root / cfg["subdir"]
    out_dir = out_root / cfg["subdir"]

    if not in_dir.exists():
        print(f"  [{modality}] Source directory not found: {in_dir}")
        return {}

    h5_files = sorted(in_dir.glob("*.h5"))
    if not h5_files:
        print(f"  [{modality}] No H5 files found in {in_dir}")
        return {}

    stats = {"files": 0, "clips": 0, "skipped": 0, "bytes_in": 0, "bytes_out": 0}
    t0 = time.time()

    for h5_path in h5_files:
        stem = h5_path.stem   # e.g. "friends_s1_features_vjepa2"
        # Extract movie_type_stimtype from filename
        # Pattern: {movie_type}_{stimulus_type}_features_{model}.h5
        # → save under out_dir / stem / {clip_name}.npy
        clip_out_dir = out_dir / stem

        stats["bytes_in"] += h5_path.stat().st_size

        with h5py.File(h5_path, "r") as f:
            clip_names = list(f.keys())

            for clip_name in clip_names:
                out_path = clip_out_dir / f"{clip_name}.npy"

                if out_path.exists() and not overwrite:
                    stats["skipped"] += 1
                    continue

                if dry_run:
                    raw_shape = f[clip_name][cfg["h5_key"]].shape
                    print(f"    [DRY] {clip_name}: {raw_shape} → {out_path.relative_to(out_root)}")
                    stats["clips"] += 1
                    continue

                try:
                    raw = f[clip_name][cfg["h5_key"]][:].astype(np.float32)
                except Exception as e:
                    print(f"    ERROR reading {clip_name}: {e}")
                    continue

                converted = convert_clip(
                    raw, cfg["data_format"], aggregation,
                    keep_tokens=(keep_omni_tokens and modality == "omni"),
                    n_layers=n_layers,
                )
                # Replace NaN/Inf with 0
                converted = np.nan_to_num(converted, nan=0.0, posinf=0.0, neginf=0.0)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, converted)

                stats["clips"] += 1
                stats["bytes_out"] += out_path.stat().st_size

        stats["files"] += 1

    elapsed = time.time() - t0
    if not dry_run:
        ratio = stats["bytes_in"] / max(stats["bytes_out"], 1)
        print(
            f"  [{modality}] {stats['files']} files, {stats['clips']} clips, "
            f"{stats['skipped']} skipped | "
            f"in={stats['bytes_in']/1e9:.2f}GB out={stats['bytes_out']/1e9:.2f}GB "
            f"ratio={ratio:.1f}x | {elapsed:.1f}s"
        )
    else:
        print(f"  [{modality}] DRY RUN: {stats['clips']} clips would be converted")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert H5 features → per-clip NPY")
    parser.add_argument(
        "--feat_root", default="Data/features",
        help="Input features root directory (default: Data/features)"
    )
    parser.add_argument(
        "--out_root", default="Data/features_npy",
        help="Output NPY root directory (default: Data/features_npy)"
    )
    parser.add_argument(
        "--modalities", nargs="+",
        default=["video", "audio", "text", "omni"],
        choices=list(MODALITY_CONFIGS.keys()),
        help="Which modalities to convert"
    )
    parser.add_argument(
        "--aggregation", default="mean", choices=["mean", "last", "cat"],
        help="Layer aggregation: 'mean' pools all layers, 'last' uses last layer, "
             "'cat' concatenates last N layers (see --n_layers)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show what would be done without writing any files"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing NPY files"
    )
    parser.add_argument(
        "--keep_omni_tokens", action="store_true",
        help="Keep individual omni tokens instead of mean-pooling. "
             "Output shape: (n_tokens, dim, n_trs) instead of (dim, n_trs)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=4,
        help="Number of last layers to concatenate (only used with --aggregation cat)"
    )
    args = parser.parse_args()

    feat_root = Path(args.feat_root)
    out_root = Path(args.out_root)

    print(f"Converting H5 → NPY")
    print(f"  Source:      {feat_root}")
    print(f"  Destination: {out_root}")
    print(f"  Modalities:  {args.modalities}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Keep omni:   {args.keep_omni_tokens}")
    print(f"  Overwrite:   {args.overwrite}")
    print(f"  Dry run:     {args.dry_run}")
    print()

    total_bytes_in = 0
    total_bytes_out = 0

    for modality in args.modalities:
        print(f"Processing [{modality}]...")
        stats = convert_modality(
            feat_root, out_root, modality,
            aggregation=args.aggregation,
            n_layers=args.n_layers,
            keep_omni_tokens=args.keep_omni_tokens,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        total_bytes_in += stats.get("bytes_in", 0)
        total_bytes_out += stats.get("bytes_out", 0)
        print()

    if not args.dry_run and total_bytes_out > 0:
        ratio = total_bytes_in / total_bytes_out
        print(f"Total: {total_bytes_in/1e9:.2f} GB → {total_bytes_out/1e9:.2f} GB  ({ratio:.1f}x reduction)")
        print(f"\nDone! NPY files written to: {out_root}")
        print(f"Set features.dir to '{out_root}' (relative to project root) in data.yaml")


if __name__ == "__main__":
    main()
