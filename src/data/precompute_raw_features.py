"""Precompute raw features from H5 to NPY for fast training.

Reads per-clip H5 feature files from algonauts_2025.features/,
applies layer mean-pooling, resamples to TR grid, and saves as NPY.

Usage:
    python src/data/precompute_raw_features.py --config src/configs/brain_flow_v2.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    load_config,
    load_feature_clip_perfile,
    resample_features_to_tr,
    _get_fmri_filepath,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("precompute_raw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    features_dir = str(Path(PROJECT_ROOT) / cfg["raw_features"]["dir"])
    fmri_dir = str(Path(PROJECT_ROOT) / cfg["data_root"] / cfg["fmri"]["dir"])
    modalities = cfg["raw_features"]["modalities"]
    layer_agg = cfg["raw_features"].get("layer_aggregation", "mean")
    feature_freq = cfg["raw_features"].get("sample_freq", 2.0)
    tr = cfg["fmri"]["tr"]
    excl_start = cfg["fmri"].get("excluded_samples_start", 0)
    excl_end = cfg["fmri"].get("excluded_samples_end", 0)
    subjects = cfg["subjects"]
    ref_subject = subjects[0]

    # Change output dir based on layer aggregation to avoid mixing "mean" and "cat"
    base_npy_dir = cfg.get("raw_features_npy_dir", "Data/raw_features_npy")
    if layer_agg == "cat":
        base_npy_dir = f"{base_npy_dir}_{layer_agg}"
    
    output_dir = Path(PROJECT_ROOT) / base_npy_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_cfg = cfg.get("splits", {})
    total_saved = 0

    for task, task_splits in splits_cfg.items():
        for split_name, stim_types in task_splits.items():
            for stim_type in stim_types:
                # Discover clips from first modality
                first_mod = next(iter(modalities))
                mod_cfg = modalities[first_mod]
                clip_dir = Path(features_dir) / mod_cfg["subdir"] / task / stim_type

                if not clip_dir.exists():
                    logger.warning("Skip %s/%s: dir not found", task, stim_type)
                    continue

                clip_stems = sorted(p.stem for p in clip_dir.glob("*.h5"))
                logger.info("Processing %s/%s: %d clips", task, stim_type, len(clip_stems))

                # Get n_trs from fMRI
                for clip_stem in tqdm(clip_stems, desc=f"{task}/{stim_type}"):
                    # Normalize clip name
                    norm_name = clip_stem
                    prefix = f"{task}_"
                    if norm_name.startswith(prefix):
                        norm_name = norm_name[len(prefix):]

                    # Get target n_trs from fMRI
                    fmri_path = _get_fmri_filepath(fmri_dir, ref_subject, task)
                    n_trs = None
                    if fmri_path.exists():
                        with h5py.File(fmri_path, "r") as f:
                            matched = [k for k in f.keys() if norm_name in k]
                            if matched:
                                n_raw = f[matched[0]].shape[0]
                                n_trs = n_raw - excl_start - excl_end

                    if n_trs is None or n_trs < 1:
                        continue

                    # Process each modality
                    for mod_name, mod_cfg in modalities.items():
                        out_path = output_dir / mod_name / task / stim_type / f"{norm_name}.npy"
                        if out_path.exists():
                            continue

                        try:
                            feat = load_feature_clip_perfile(
                                features_dir, mod_name, mod_cfg,
                                task, stim_type, clip_stem, layer_agg,
                            )
                            # Resample: (1, dim, T_feat) → (1, dim, n_trs)
                            feat = resample_features_to_tr(feat, feature_freq, tr, n_trs)
                            # Save as (n_trs, dim) for compact storage
                            feat_2d = feat[0].T.astype(np.float32)  # (n_trs, dim)

                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(out_path, feat_2d)
                        except Exception as e:
                            logger.warning("Skip %s/%s: %s", mod_name, clip_stem, e)
                            continue

                    total_saved += 1

    logger.info("Done! Saved features for %d clips to %s", total_saved, output_dir)


if __name__ == "__main__":
    main()
