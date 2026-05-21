"""Extended evaluation metrics for BrainFlow.

Provides:
  - Inter-subject noise ceiling (per-voxel PCC upper bound)
  - Noise-ceiling normalized score (fraction of explainable variance)
  - Standard Error of the Mean (SEM) across subjects
  - Per-Yeo-network PCC aggregation using Schaefer 1000Par7Net ordering

All functions are numpy-only and operate on pre-computed arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# ── Schaefer 1000Par7Net network layout ──────────────────────────────
# Parcels are ordered: LH networks [1-500] then RH networks [501-1000],
# each hemisphere in order: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default.
# Per-hemisphere counts (same as NetworkSubjectLayers.SCHAEFER_7NET_PER_HEMI):
_SCHAEFER_7NET_PER_HEMI = [75, 74, 66, 68, 38, 61, 118]  # sum = 500
NETWORK_NAMES = ["Visual", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def _build_network_indices(n_voxels: int = 1000) -> list[np.ndarray]:
    """Return list of 7 index arrays, one per Yeo network (both hemispheres merged).

    Index i contains parcel indices (0-based) belonging to network i.
    """
    if n_voxels != 1000:
        log.warning(
            "network_indices designed for 1000 parcels, got %d — skipping network split",
            n_voxels,
        )
        return []

    indices = []
    lh_offset = 0
    rh_offset = 500
    for count in _SCHAEFER_7NET_PER_HEMI:
        lh_idx = np.arange(lh_offset, lh_offset + count)
        rh_idx = np.arange(rh_offset, rh_offset + count)
        indices.append(np.concatenate([lh_idx, rh_idx]))
        lh_offset += count
        rh_offset += count
    return indices


NETWORK_INDICES = _build_network_indices(1000)


# ── Core metric functions ────────────────────────────────────────────

def per_voxel_pcc(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-voxel Pearson correlation → (V,).

    Args:
        pred:   (T, V) predicted fMRI.
        target: (T, V) ground-truth fMRI.
    """
    p = pred - pred.mean(axis=0, keepdims=True)
    t = target - target.mean(axis=0, keepdims=True)
    cov = (p * t).sum(axis=0)
    std = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
    return cov / (std + 1e-8)


def compute_noise_ceiling(
    fmri_by_subject: dict[str, np.ndarray],
    target_subject: str,
) -> np.ndarray:
    """Inter-subject noise ceiling for one target subject.

    For each source subject s ≠ target, compute per-voxel PCC between
    source and target fMRI (both z-scored). Average across sources.

    Args:
        fmri_by_subject: {subject_name: (T, V)} z-scored fMRI, all aligned
                         to the same clips/TRs.
        target_subject:  name of the target subject.

    Returns:
        ceiling: (V,) per-voxel noise ceiling.
    """
    target = fmri_by_subject[target_subject]
    sources = [s for s in fmri_by_subject if s != target_subject]
    if not sources:
        log.warning("No source subjects for noise ceiling of %s", target_subject)
        return np.zeros(target.shape[1], dtype=np.float32)

    pcc_stack = []
    for src_name in sources:
        src = fmri_by_subject[src_name]
        # Ensure same length
        T = min(target.shape[0], src.shape[0])
        pcc_stack.append(per_voxel_pcc(src[:T], target[:T]))

    ceiling = np.nanmean(np.stack(pcc_stack, axis=0), axis=0)
    return np.nan_to_num(ceiling, nan=0.0).astype(np.float32)


def compute_normalized_score(
    model_pcc: np.ndarray,
    noise_ceiling: np.ndarray,
    clip_low: float = 0.0,
    clip_high: float = 1.0,
) -> np.ndarray:
    """Noise-ceiling normalized score per voxel.

    normalized[v] = model_pcc[v] / noise_ceiling[v], clipped to [clip_low, clip_high].
    Voxels with ceiling ≤ 0 are set to NaN.

    Args:
        model_pcc:     (V,) per-voxel PCC of the model.
        noise_ceiling: (V,) per-voxel noise ceiling.

    Returns:
        normalized: (V,) normalized scores.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = model_pcc / noise_ceiling
    # Mask unreliable voxels
    normalized[noise_ceiling <= 0] = np.nan
    normalized[model_pcc < 0] = 0.0
    return np.clip(normalized, clip_low, clip_high)


def compute_sem(values: np.ndarray) -> float:
    """Standard Error of the Mean.

    Args:
        values: (N,) array of per-subject metric values.

    Returns:
        SEM = std / sqrt(N).
    """
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def per_network_pcc(
    voxel_pcc: np.ndarray,
    network_indices: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """Aggregate per-voxel PCC by Yeo 7-network.

    Args:
        voxel_pcc:       (V,) per-voxel PCC.
        network_indices: list of 7 index arrays; defaults to NETWORK_INDICES.

    Returns:
        {network_name: mean_pcc}
    """
    if network_indices is None:
        network_indices = NETWORK_INDICES
    if not network_indices:
        return {}

    result = {}
    for name, idx in zip(NETWORK_NAMES, network_indices):
        vals = voxel_pcc[idx]
        result[name] = float(np.nanmean(vals))
    return result


# ── Aggregate result container ───────────────────────────────────────

@dataclass
class ExtendedMetrics:
    """Container for all extended metrics for one subject."""

    subject: str

    # Whole-brain PCC
    median_pcc: float = 0.0
    mean_pcc: float = 0.0
    per_voxel_pcc: np.ndarray = field(default_factory=lambda: np.array([]))

    # Noise ceiling
    noise_ceiling: np.ndarray = field(default_factory=lambda: np.array([]))
    median_noise_ceiling: float = 0.0
    mean_noise_ceiling: float = 0.0

    # Normalized scores
    normalized_score: np.ndarray = field(default_factory=lambda: np.array([]))
    median_normalized: float = 0.0
    mean_normalized: float = 0.0

    # Per-network PCC
    network_pcc: dict[str, float] = field(default_factory=dict)

    # Per-network normalized scores
    network_normalized: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict suitable for np.save."""
        return {
            "subject": self.subject,
            "median_pcc": self.median_pcc,
            "mean_pcc": self.mean_pcc,
            "per_voxel_pcc": self.per_voxel_pcc,
            "noise_ceiling": self.noise_ceiling,
            "median_noise_ceiling": self.median_noise_ceiling,
            "mean_noise_ceiling": self.mean_noise_ceiling,
            "normalized_score": self.normalized_score,
            "median_normalized": self.median_normalized,
            "mean_normalized": self.mean_normalized,
            "network_pcc": self.network_pcc,
            "network_normalized": self.network_normalized,
        }


def compute_extended_metrics(
    subject: str,
    model_voxel_pcc: np.ndarray,
    noise_ceiling: np.ndarray | None = None,
) -> ExtendedMetrics:
    """Compute all extended metrics for one subject.

    Args:
        subject:         subject name.
        model_voxel_pcc: (V,) per-voxel PCC from model predictions.
        noise_ceiling:   (V,) per-voxel noise ceiling (None = skip).

    Returns:
        ExtendedMetrics instance.
    """
    m = ExtendedMetrics(subject=subject)
    m.per_voxel_pcc = model_voxel_pcc
    m.median_pcc = float(np.median(model_voxel_pcc))
    m.mean_pcc = float(np.mean(model_voxel_pcc))

    # Per-network PCC
    m.network_pcc = per_network_pcc(model_voxel_pcc)

    if noise_ceiling is not None and len(noise_ceiling) == len(model_voxel_pcc):
        m.noise_ceiling = noise_ceiling
        m.median_noise_ceiling = float(np.median(noise_ceiling))
        m.mean_noise_ceiling = float(np.mean(noise_ceiling))

        norm = compute_normalized_score(model_voxel_pcc, noise_ceiling)
        m.normalized_score = norm
        m.median_normalized = float(np.nanmedian(norm))
        m.mean_normalized = float(np.nanmean(norm))

        # Per-network normalized
        m.network_normalized = per_network_pcc(norm)

    return m


def compute_cross_subject_summary(
    metrics_list: list[ExtendedMetrics],
) -> dict:
    """Compute cross-subject summary with SEM.

    Args:
        metrics_list: list of ExtendedMetrics, one per subject.

    Returns:
        Summary dict with means, SEMs, and per-network breakdown.
    """
    if not metrics_list:
        return {}

    median_pccs = np.array([m.median_pcc for m in metrics_list])
    mean_pccs = np.array([m.mean_pcc for m in metrics_list])
    median_norms = np.array([m.median_normalized for m in metrics_list])
    mean_norms = np.array([m.mean_normalized for m in metrics_list])
    mean_ceilings = np.array([m.mean_noise_ceiling for m in metrics_list])

    summary = {
        "n_subjects": len(metrics_list),
        "subjects": [m.subject for m in metrics_list],
        # Whole-brain
        "avg_median_pcc": float(np.mean(median_pccs)),
        "sem_median_pcc": compute_sem(median_pccs),
        "avg_mean_pcc": float(np.mean(mean_pccs)),
        "sem_mean_pcc": compute_sem(mean_pccs),
        # Noise ceiling
        "avg_mean_noise_ceiling": float(np.mean(mean_ceilings)),
        # Normalized
        "avg_median_normalized": float(np.mean(median_norms)),
        "sem_median_normalized": compute_sem(median_norms),
        "avg_mean_normalized": float(np.mean(mean_norms)),
        "sem_mean_normalized": compute_sem(mean_norms),
    }

    # Per-network aggregation
    all_names = NETWORK_NAMES
    for name in all_names:
        net_pccs = np.array([m.network_pcc.get(name, 0.0) for m in metrics_list])
        net_norms = np.array([m.network_normalized.get(name, 0.0) for m in metrics_list])
        summary[f"net_{name}_mean_pcc"] = float(np.mean(net_pccs))
        summary[f"net_{name}_sem_pcc"] = compute_sem(net_pccs)
        summary[f"net_{name}_mean_norm"] = float(np.mean(net_norms))
        summary[f"net_{name}_sem_norm"] = compute_sem(net_norms)

    return summary
