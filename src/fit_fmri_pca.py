"""Fit PCA on training fMRI data and save model for flow matching.

Loads all training fMRI from H5 files (with global stats normalization),
fits PCA(n_components=K), and saves the model + diagnostics.

Usage:
    python src/fit_fmri_pca.py --config src/configs/brain_flow_direct_v3_pca.yaml
    python src/fit_fmri_pca.py --config src/configs/brain_flow_direct_v3_pca.yaml --n_components 100
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_config, _get_fmri_filepath

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("fit_fmri_pca")


def load_all_training_fmri(cfg: dict) -> np.ndarray:
    """Load and concatenate all training fMRI data across subjects and clips.

    Returns: (N_total, n_voxels) float32 array.
    """
    fmri_dir = Path(PROJECT_ROOT) / cfg["data_root"] / cfg["fmri"]["dir"]
    subjects = cfg["subjects"]
    splits_cfg = cfg.get("splits", {})
    excl_start = cfg["fmri"].get("excluded_samples_start", 0)
    excl_end = cfg["fmri"].get("excluded_samples_end", 0)

    # Load global stats for normalization
    use_global_stats = cfg["fmri"].get("use_global_stats", False)
    fmri_stats = {}
    if use_global_stats:
        for subj in subjects:
            stats_dir = fmri_dir / subj / "stats"
            mean_path = stats_dir / "global_mean.npy"
            std_path = stats_dir / "global_std.npy"
            if mean_path.exists() and std_path.exists():
                fmri_stats[subj] = {
                    "mean": np.load(mean_path).astype(np.float32),
                    "std": np.load(std_path).astype(np.float32),
                }
                logger.info("Loaded global stats for %s", subj)

    all_fmri = []
    n_clips = 0

    for task, splits in splits_cfg.items():
        train_stims = splits.get("train", [])
        for stim_type in train_stims:
            for subj in subjects:
                fmri_path = _get_fmri_filepath(str(fmri_dir), subj, task)
                if not fmri_path.exists():
                    logger.warning("fMRI file missing: %s", fmri_path)
                    continue

                with h5py.File(fmri_path, "r") as f:
                    for key in f.keys():
                        # Filter by stim_type (e.g., "s1" should match keys containing "s01")
                        raw = f[key]
                        end = len(raw) - excl_end if excl_end > 0 else len(raw)
                        data = raw[excl_start:end].astype(np.float32)
                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                        # Apply global normalization
                        if use_global_stats and subj in fmri_stats:
                            stats = fmri_stats[subj]
                            data = (data - stats["mean"][None, :]) / stats["std"][None, :]

                        all_fmri.append(data)
                        n_clips += 1

    if not all_fmri:
        raise ValueError("No training fMRI data found! Check config splits.")

    result = np.concatenate(all_fmri, axis=0)
    logger.info("Loaded %d clips → %d total TRs, shape %s", n_clips, result.shape[0], result.shape)
    return result


def main():
    parser = argparse.ArgumentParser(description="Fit PCA on training fMRI data")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--n_components", type=int, default=None,
                        help="Number of PCA components (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # PCA settings
    n_components = args.n_components or cfg.get("pca_dim", 100)
    output_dir = Path(args.output_dir or cfg.get("pca_output_dir", f"outputs/fmri_pca_{n_components}"))
    output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Fitting PCA with %d components ===", n_components)

    # 1. Load all training fMRI
    fmri_data = load_all_training_fmri(cfg)
    logger.info("fMRI data shape: %s (%.2f GB)", fmri_data.shape,
                fmri_data.nbytes / 1e9)

    # 2. Fit PCA
    logger.info("Fitting PCA(n_components=%d)...", n_components)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(fmri_data)

    # 3. Diagnostics
    total_var = pca.explained_variance_ratio_.sum()
    logger.info("Explained variance ratio (top %d): %.4f (%.2f%%)",
                n_components, total_var, total_var * 100)
    logger.info("Top-10 component variances: %s",
                np.round(pca.explained_variance_ratio_[:10], 4))

    # 4. Round-trip sanity check
    n_check = min(1000, fmri_data.shape[0])
    x_check = fmri_data[:n_check]
    x_recon = pca.inverse_transform(pca.transform(x_check))
    recon_mse = np.mean((x_check - x_recon) ** 2)
    recon_corr = np.corrcoef(x_check.ravel(), x_recon.ravel())[0, 1]
    logger.info("Round-trip check (N=%d): MSE=%.6f, Correlation=%.4f",
                n_check, recon_mse, recon_corr)

    # 5. Save
    model_path = output_dir / "pca_model.pkl"
    joblib.dump(pca, model_path)
    logger.info("Saved PCA model to %s", model_path)

    # Save diagnostics
    np.save(output_dir / "explained_variance_ratio.npy", pca.explained_variance_ratio_)
    np.save(output_dir / "components.npy", pca.components_)
    np.save(output_dir / "mean.npy", pca.mean_)

    logger.info("=== Done! PCA model saved to %s ===", output_dir)
    logger.info("  Components shape: %s", pca.components_.shape)
    logger.info("  Mean shape: %s", pca.mean_.shape)
    logger.info("  Total explained variance: %.4f", total_var)


if __name__ == "__main__":
    main()
