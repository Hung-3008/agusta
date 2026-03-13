"""Precompute PCA-reduced stimulus features for BrainFlow.

Follows the challenge baseline pipeline:
  1. Load raw per-clip features (video/audio/text) from algonauts_2025.features/
  2. Resample from extraction freq to TR grid via interpolation
  3. Fit StandardScaler + PCA on all training data (Friends s1-s6 + Movie10)
  4. Apply HRF delay: shift features by `hrf_delay` TRs
  5. Temporal window: for visual/audio, concat `stimulus_window` consecutive TRs
  6. Concat all modalities → single conditioning vector per target TR
  7. Save as NPY per clip + scaler/PCA params for test-time transform

Usage:
    python src/data/precompute_pca_features.py --config src/configs/brain_flow.yaml
    python src/data/precompute_pca_features.py --config src/configs/brain_flow.yaml --verify
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    load_config,
    resolve_paths,
    load_feature_clip_perfile,
    resample_features_to_tr,
    _get_fmri_filepath,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pca_features")

# Seed for reproducibility (same as baseline)
SEED = 20200220


def _get_clip_n_trs(fmri_dir: str, subject: str, task: str, clip_key: str,
                     excl_start: int, excl_end: int) -> int | None:
    """Get number of TRs for a clip from fMRI h5 file, after trimming."""
    fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
    if not fmri_path.exists():
        return None
    with h5py.File(fmri_path, "r") as f:
        matched = [k for k in f.keys() if clip_key in k]
        if not matched:
            return None
        n_raw = f[matched[0]].shape[0]
    return n_raw - excl_start - excl_end


def _normalize_clip_name(clip_name: str, task: str) -> str:
    """Strip task prefix from clip name to get a consistent normalized name.

    Some modalities prefix the clip name with the task name (e.g. text uses
    ``movie10_bourne01.h5`` while video/audio use ``bourne01.h5``). This
    normalisation ensures all modalities share the same UID key-space so that
    cross-modality intersection works correctly.
    """
    prefix = f"{task}_"
    if clip_name.startswith(prefix):
        return clip_name[len(prefix):]
    return clip_name


def _discover_clips(features_dir: str, modality_cfg: dict, task: str,
                     stim_type: str) -> list[tuple[str, str]]:
    """Discover clip names from per-clip feature directory.

    Returns list of (raw_stem, normalized_name) tuples.
    ``raw_stem`` is the actual filename stem used for loading;
    ``normalized_name`` is the task-prefix-stripped name used as UID.
    """
    subdir = modality_cfg["subdir"]
    clip_dir = Path(features_dir) / subdir / task / stim_type
    if not clip_dir.exists():
        return []
    return sorted(
        (p.stem, _normalize_clip_name(p.stem, task))
        for p in clip_dir.glob("*.h5")
    )


def load_all_features_for_modality(
    cfg: dict,
    modality: str,
    split_filter: str = "train",
) -> dict[str, np.ndarray]:
    """Load raw features for one modality, resampled to TR grid.

    Returns dict: clip_uid → (n_trs, dim)
    """
    features_dir = cfg["_features_dir"]
    fmri_dir = cfg["_fmri_dir"]
    subjects = cfg["subjects"]
    splits_cfg = cfg["splits"]
    mod_cfg = cfg["features"]["modalities"][modality]
    tr = cfg["fmri"]["tr"]
    feature_freq = cfg["features"]["sample_freq"]
    layer_agg = cfg["features"].get("layer_aggregation", "last")
    excl_start = cfg["fmri"].get("excluded_samples_start", 0)
    excl_end = cfg["fmri"].get("excluded_samples_end", 0)

    # Use first subject to get n_trs
    ref_subject = subjects[0]
    result = {}

    for task, task_splits in splits_cfg.items():
        stim_types = task_splits.get(split_filter, [])
        for stim_type in stim_types:
            clips = _discover_clips(features_dir, mod_cfg, task, stim_type)
            for raw_clip_name, norm_clip_name in clips:
                # norm_clip_name has task prefix stripped (e.g. "bourne01" not "movie10_bourne01")
                # fmri_key is used to look up data in the H5 file via substring match
                fmri_key = norm_clip_name

                n_trs = _get_clip_n_trs(fmri_dir, ref_subject, task, fmri_key,
                                         excl_start, excl_end)
                if n_trs is None or n_trs < 1:
                    logger.warning("No fMRI for %s/%s, skipping", task, raw_clip_name)
                    continue

                try:
                    # Load using the raw filename (as it exists on disk)
                    feat = load_feature_clip_perfile(
                        features_dir, modality, mod_cfg,
                        task, stim_type, raw_clip_name, layer_agg,
                    )
                except (FileNotFoundError, KeyError, ValueError) as e:
                    logger.warning("Skip %s/%s: %s", modality, raw_clip_name, e)
                    continue

                # feat: (1, dim, T_feat) → resample to (1, dim, n_trs)
                feat = resample_features_to_tr(feat, feature_freq, tr, n_trs)
                # (1, dim, n_trs) → (n_trs, dim)
                feat = feat[0].T  # (n_trs, dim)
                feat = np.nan_to_num(feat, nan=0.0)

                # Use normalized name as UID so all modalities share the same key
                uid = f"{task}/{stim_type}/{norm_clip_name}"
                result[uid] = feat

    return result


def fit_pca_pipeline(
    all_features: dict[str, np.ndarray],
    n_components: int = 250,
) -> tuple[StandardScaler, PCA]:
    """Fit StandardScaler + PCA on concatenated features."""
    # Concatenate all clips
    all_data = np.concatenate(list(all_features.values()), axis=0)
    logger.info("  Fitting on %d samples, %d dims", all_data.shape[0], all_data.shape[1])

    # Z-score
    scaler = StandardScaler()
    scaler.fit(all_data)
    all_data = scaler.transform(all_data)

    # PCA
    n_comp = min(n_components, all_data.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED)
    pca.fit(all_data)
    logger.info("  PCA: %d → %d dims (%.1f%% variance)",
                all_data.shape[1], n_comp,
                pca.explained_variance_ratio_.sum() * 100)

    return scaler, pca


def apply_hrf_and_window(
    features: dict[str, np.ndarray],
    hrf_delay: int = 3,
    stimulus_window: int = 5,
    is_language: bool = False,
) -> dict[str, np.ndarray]:
    """Apply HRF delay and temporal windowing to features.

    For visual/audio: concat `stimulus_window` consecutive TRs per target TR.
    For language: only single TR (already contextual).

    Returns dict: clip_uid → (n_trs, windowed_dim)
    """
    result = {}
    for uid, feat in features.items():
        n_trs, dim = feat.shape
        windowed = []

        for s in range(n_trs):
            if is_language:
                # Single sample minus HRF delay
                idx = max(0, s - hrf_delay)
                idx = min(idx, n_trs - 1)
                windowed.append(feat[idx].flatten())
            else:
                # Window of stimulus_window samples ending at s - hrf_delay
                if s < (stimulus_window + hrf_delay):
                    idx_start = 0
                    idx_end = stimulus_window
                else:
                    idx_end = s - hrf_delay + 1
                    idx_start = idx_end - stimulus_window

                # Clamp to valid range
                if idx_end > n_trs:
                    idx_end = n_trs
                    idx_start = max(0, idx_end - stimulus_window)

                window = feat[idx_start:idx_end]
                windowed.append(window.flatten())

        result[uid] = np.array(windowed, dtype=np.float32)

    return result


def main():
    parser = argparse.ArgumentParser(description="Precompute PCA features for BrainFlow")
    parser.add_argument("--config", type=str, required=True, help="Path to brainflow config YAML")
    parser.add_argument("--verify", action="store_true", help="Verify output dimensions only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    pca_cfg = cfg.get("pca_features", {})
    n_components = pca_cfg.get("n_components", 250)
    hrf_delay = pca_cfg.get("hrf_delay", 3)
    stimulus_window = pca_cfg.get("stimulus_window", 5)
    output_dir = Path(PROJECT_ROOT) / pca_cfg.get("output_dir", "Data/pca_features")
    modalities = pca_cfg.get("modalities", ["video", "audio", "text"])

    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Step 1: Load + fit PCA per modality (training data only)
    # =====================================================================
    scalers = {}
    pcas = {}
    raw_features = {}  # mod → {uid: (n_trs, dim)}
    pca_features = {}  # mod → {uid: (n_trs, n_comp)}

    for mod in modalities:
        logger.info("=== Loading %s features ===", mod)
        raw_features[mod] = load_all_features_for_modality(cfg, mod, "train")
        logger.info("  Loaded %d clips", len(raw_features[mod]))

        if not raw_features[mod]:
            logger.warning("  No features found for %s, skipping", mod)
            continue

        # Fit PCA
        n_comp = n_components
        # Audio: keep all dims if < n_components (matching baseline)
        sample_dim = next(iter(raw_features[mod].values())).shape[1]
        if mod == "audio" and sample_dim <= n_components:
            n_comp = sample_dim
            logger.info("  Audio dim %d <= n_components, keeping all", sample_dim)

        logger.info("  Fitting PCA for %s...", mod)
        scaler, pca = fit_pca_pipeline(raw_features[mod], n_comp)
        scalers[mod] = scaler
        pcas[mod] = pca

        # Transform
        pca_features[mod] = {}
        for uid, feat in raw_features[mod].items():
            transformed = scaler.transform(feat)
            transformed = pca.transform(transformed).astype(np.float32)
            pca_features[mod][uid] = transformed

    # Save scaler/PCA params
    for mod in modalities:
        if mod not in scalers:
            continue
        params_dir = output_dir / "params" / mod
        params_dir.mkdir(parents=True, exist_ok=True)
        np.save(params_dir / "scaler_mean.npy", scalers[mod].mean_)
        np.save(params_dir / "scaler_scale.npy", scalers[mod].scale_)
        np.save(params_dir / "pca_components.npy", pcas[mod].components_)
        np.save(params_dir / "pca_mean.npy", pcas[mod].mean_)
        np.save(params_dir / "pca_explained_variance_ratio.npy",
                pcas[mod].explained_variance_ratio_)
        logger.info("Saved PCA params for %s to %s", mod, params_dir)

    # =====================================================================
    # Step 2: Apply HRF delay + temporal windowing
    # =====================================================================
    logger.info("=== Applying HRF delay=%d, stimulus_window=%d ===", hrf_delay, stimulus_window)
    windowed_features = {}  # mod → {uid: (n_trs, windowed_dim)}

    for mod in modalities:
        if mod not in pca_features:
            continue
        is_lang = (mod == "text" or mod == "language")
        windowed_features[mod] = apply_hrf_and_window(
            pca_features[mod], hrf_delay, stimulus_window, is_lang,
        )
        sample_dim = next(iter(windowed_features[mod].values())).shape[1]
        logger.info("  %s: windowed dim = %d", mod, sample_dim)

    # =====================================================================
    # Step 3: Concat modalities and save per clip
    # =====================================================================
    # Collect all UIDs that have all modalities present
    all_uids = set()
    for mod in modalities:
        if mod in windowed_features:
            all_uids.update(windowed_features[mod].keys())

    # Only keep UIDs present in ALL modalities
    valid_uids = sorted(all_uids)
    for mod in modalities:
        if mod in windowed_features:
            valid_uids = [u for u in valid_uids if u in windowed_features[mod]]

    logger.info("=== Saving %d clips ===", len(valid_uids))
    total_cond_dim = None

    for uid in tqdm(valid_uids, desc="Saving"):
        parts = []
        for mod in modalities:
            if mod in windowed_features:
                parts.append(windowed_features[mod][uid])
        concat = np.concatenate(parts, axis=-1)  # (n_trs, D_cond)

        if total_cond_dim is None:
            total_cond_dim = concat.shape[1]
            logger.info("Total conditioning dimension: %d", total_cond_dim)

        # Save: output_dir/{task}/{stim_type}/{clip_name}.npy
        save_path = output_dir / f"{uid}.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, concat)

    # =====================================================================
    # Step 4: Process test data with fitted scaler/PCA
    # =====================================================================
    logger.info("=== Processing test data ===")
    for mod in modalities:
        if mod not in scalers:
            continue
        logger.info("  Loading test %s...", mod)
        test_raw = load_all_features_for_modality(cfg, mod, "test")
        if not test_raw:
            logger.info("  No test data for %s", mod)
            continue

        # Transform with fitted params
        test_pca = {}
        for uid, feat in test_raw.items():
            transformed = scalers[mod].transform(feat)
            transformed = pcas[mod].transform(transformed).astype(np.float32)
            test_pca[uid] = transformed

        is_lang = (mod == "text" or mod == "language")
        test_windowed = apply_hrf_and_window(
            test_pca, hrf_delay, stimulus_window, is_lang,
        )

        # Save test features
        for uid in test_windowed:
            if all(uid in windowed_features.get(m, test_windowed)
                   for m in modalities if m != mod):
                pass  # will be concatenated below

        # Store for concat
        if mod not in windowed_features:
            windowed_features[mod] = {}
        windowed_features[mod].update(test_windowed)

    # Concat and save test clips
    test_uids = set()
    for mod in modalities:
        if mod in windowed_features:
            for uid in windowed_features[mod]:
                if uid not in valid_uids:
                    test_uids.add(uid)

    test_valid = sorted(test_uids)
    for mod in modalities:
        if mod in windowed_features:
            test_valid = [u for u in test_valid if u in windowed_features[mod]]

    for uid in tqdm(test_valid, desc="Saving test"):
        parts = []
        for mod in modalities:
            if mod in windowed_features:
                parts.append(windowed_features[mod][uid])
        concat = np.concatenate(parts, axis=-1)
        save_path = output_dir / f"{uid}.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, concat)

    logger.info("=== Done! %d train + %d test clips saved to %s ===",
                len(valid_uids), len(test_valid), output_dir)

    if args.verify:
        logger.info("=== Verification ===")
        for uid in valid_uids[:3]:
            data = np.load(output_dir / f"{uid}.npy")
            logger.info("  %s: shape=%s, range=[%.3f, %.3f]",
                        uid, data.shape, data.min(), data.max())
        logger.info("  Total cond_dim: %d", total_cond_dim)


if __name__ == "__main__":
    main()
