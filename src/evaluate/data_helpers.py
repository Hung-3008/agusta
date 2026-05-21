"""Data loading helpers for BrainFlow evaluation.

Provides context/fMRI loading, sliding-window builders, and PCC computation.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from src.datasets.directflow_dataset import _get_fmri_filepath

log = logging.getLogger("evaluate")


def _movie_from_clip(clip_key: str) -> str:
    """Extract movie name from OOD clip key: 'chaplin1' → 'chaplin'."""
    return clip_key.rstrip("0123456789")


def load_context_clip(context_dirs: list[Path], task: str, session: str,
                      clip_name: str, expected_dims: list[int] | None = None) -> np.ndarray | None:
    """Load and concatenate multi-modal context for one clip → (T, D_total).

    For task='friends':  looks in {ctx_dir}/friends/{session}/{clip_name}.npy
    For task='ood':      looks in {ctx_dir}/ood/{movie}/{clip_name}.npy
                         where movie = clip_name with trailing digits stripped.
    """
    arrays = []
    for ctx_dir in context_dirs:
        if task == "ood":
            movie = _movie_from_clip(clip_name)
            base = ctx_dir / "ood" / movie
            candidates = [
                base / f"{clip_name}.npy",
                base / f"{clip_name.lstrip(movie)}.npy",  # just the number part
                base / f"task-{clip_name}_video.npy",     # e.g. task-chaplin1_video.npy
                base / f"ood_{clip_name}.npy",            # e.g. ood_chaplin1.npy
                base / f"task-{clip_name}.npy"
            ]
        else:
            base = ctx_dir / task / session
            candidates = [
                base / f"{clip_name}.npy",
                base / f"{clip_name.removeprefix(f'{task}_')}.npy",
                base / f"{task}_{clip_name}.npy",
            ]

        for candidate in candidates:
            if candidate.exists():
                arr = np.load(candidate).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0], -1)
                arrays.append(arr)
                break
        else:
            # Zero-pad missing modality with correct or best-guess dim
            dim = 512
            if expected_dims and len(arrays) < len(expected_dims):
                dim = expected_dims[len(arrays)]
            else:
                for f in (ctx_dir / "ood" / _movie_from_clip(clip_name) if task == "ood" else ctx_dir / task / session).rglob("*.npy"):
                    dim = np.load(f).shape[-1]
                    break
            ref = arrays[0].shape[0] if arrays else 100
            arrays.append(np.zeros((ref, dim), dtype=np.float32))

    if not arrays:
        return None
    t = max(a.shape[0] for a in arrays)
    padded_arrays = []
    for a in arrays:
        if a.shape[0] < t:
            pad_len = t - a.shape[0]
            padded_arrays.append(np.pad(a, ((0, pad_len), (0, 0))))
        else:
            padded_arrays.append(a)
    return np.concatenate(padded_arrays, axis=-1)


def load_fmri_clip(fmri_dir: str, subject: str, task: str, clip_key: str,
                   excl_start: int, excl_end: int,
                   fmri_stats: dict | None) -> np.ndarray | None:
    """Load and (optionally) z-score fMRI for one clip → (T, V)."""
    path = _get_fmri_filepath(fmri_dir, subject, task)
    if not path.exists():
        return None

    key = clip_key
    if clip_key.startswith(f"{task}_"):
        key = clip_key[len(f"{task}_"):]

    with h5py.File(path, "r") as f:
        matches = [k for k in f.keys() if key in k]
        if not matches:
            return None
        raw = f[matches[0]]
        end = len(raw) - excl_end if excl_end > 0 else len(raw)
        data = raw[excl_start:end].astype(np.float32)

    data = np.nan_to_num(data, nan=0.0)
    if fmri_stats and subject in fmri_stats:
        s = fmri_stats[subject]
        data = (data - s["mean"]) / s["std"]
    return data


def load_fmri_stats(fmri_dir: str, subjects: list[str]) -> dict:
    """Load per-subject global mean/std for z-score normalization."""
    stats = {}
    for subj in subjects:
        d = Path(fmri_dir) / subj / "stats"
        mp, sp = d / "global_mean.npy", d / "global_std.npy"
        if mp.exists() and sp.exists():
            stats[subj] = {
                "mean": np.load(mp).astype(np.float32)[None, :],  # (1, V)
                "std":  np.load(sp).astype(np.float32)[None, :],
            }
            log.info("Loaded fMRI stats: %s", subj)
    return stats


def build_seq2seq_windows(ctx: np.ndarray, n_trs: int, context_trs: int,
                          n_target_trs: int, hrf_delay: int, excl_start: int,
                          stride: int) -> list[dict]:
    """Sliding-window builder for S6 evaluation (with HRF shift)."""
    extra_past = (context_trs - n_target_trs) // 2
    windows = []
    for ts in range(0, max(0, n_trs - n_target_trs + 1), stride):
        feat0 = (ts + excl_start) - hrf_delay
        c0 = feat0 - extra_past
        c1 = c0 + context_trs
        chunk = ctx[max(0, c0):min(ctx.shape[0], c1)]
        if chunk.shape[0] < context_trs:
            pb = max(0, -c0)
            pa = context_trs - chunk.shape[0] - pb
            chunk = np.pad(chunk, ((pb, pa), (0, 0)))
        windows.append({"target_start": ts, "context": chunk[:context_trs]})
    return windows


def build_s7_windows(ctx: np.ndarray, n_trs: int, context_trs: int,
                     n_target_trs: int, hrf_delay: int, stride: int) -> list[dict]:
    """Sliding-window builder for S7/OOD submission (with HRF shift)."""
    extra_past = (context_trs - n_target_trs) // 2
    windows = []
    for ts in range(0, max(0, n_trs - n_target_trs + 1), stride):
        feat0 = ts - hrf_delay
        c0 = feat0 - extra_past
        c1 = c0 + context_trs
        chunk = ctx[max(0, c0):min(ctx.shape[0], c1)]
        if chunk.shape[0] < context_trs:
            pb = max(0, -c0)
            pa = context_trs - chunk.shape[0] - pb
            chunk = np.pad(chunk, ((pb, pa), (0, 0)))
        windows.append({"target_start": ts, "context": chunk[:context_trs]})
    return windows


def pcc(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-voxel Pearson correlation → (V,)."""
    p = pred - pred.mean(0, keepdims=True)
    t = target - target.mean(0, keepdims=True)
    cov = (p * t).sum(0)
    std = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    return cov / (std + 1e-8)
