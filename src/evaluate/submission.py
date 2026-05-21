"""Submission assembly, sample count loaders, and prediction denormalization."""

import gc
import logging
import zipfile
from pathlib import Path

import numpy as np

log = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_sample_counts(subject: str, session: str) -> dict:
    """Load target sample counts for a subject and session.

    Args:
        subject:  e.g. 'sub-01'
        session:  'friends-s7' or 'ood'

    Returns:
        {episode_or_clip: n_trs}  e.g. {'s07e01a': 460, ...}
    """
    p = (PROJECT_ROOT / "Data" / "algonauts_2025.competitors" / "fmri"
         / subject / "target_sample_number"
         / f"{subject}_{session}_fmri_samples.npy")
    return np.load(p, allow_pickle=True).item()


def denormalize_predictions(pred: np.ndarray, stats: dict, subject: str) -> np.ndarray:
    """Reverse z-score normalization on predictions.

    Args:
        pred:    (T, V) predicted fMRI (z-scored space).
        stats:   {subject: {'mean': (1,V), 'std': (1,V)}} from load_fmri_stats.
        subject: subject name.

    Returns:
        (T, V) denormalized predictions.
    """
    if subject in stats:
        s = stats[subject]
        pred = pred * s["std"] + s["mean"]
    return pred


def save_submission(subj_paths: dict, out_dir: Path, tag: str):
    """Merge per-subject npy files → submission.npy + submission.zip."""
    log.info("\nAssembling %s submission...", tag)
    submission = {s: np.load(p, allow_pickle=True).item() for s, p in subj_paths.items()}
    sub_path = out_dir / "submission.npy"
    np.save(sub_path, submission)
    del submission; gc.collect()

    zip_path = sub_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_path, arcname="submission.npy")

    log.info("submission.zip → %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)
    for p in subj_paths.values():
        p.unlink(missing_ok=True)
    log.info("Temp files cleaned.")
