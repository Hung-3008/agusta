"""Sliding-window dataset for fMRI encoding.

Loads pre-extracted multimodal features (video, audio, text) and fMRI data,
creates sliding windows aligned with HRF timing, and returns PyTorch-compatible
samples for training.

Design principles:
  - No fixed HRF delay — the model learns temporal relationships from context
  - Context window ≈ 14.9s (10 TRs) covers full HRF delay range (4-12s)
  - Stride = 1 TR for dense, overlapping sampling
  - Features resampled from 2Hz extraction rate to TR grid via interpolation
"""

import hashlib
import logging
from tqdm import tqdm
import random
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# LRU Cache with byte-level size limit
# =============================================================================

class LRUCache:
    """Thread-unsafe LRU cache with a hard byte-size cap.

    Entries are numpy arrays; size is tracked via array.nbytes.
    Oldest entries are evicted when total size exceeds max_bytes.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._cache: OrderedDict = OrderedDict()
        self._nbytes: dict = {}
        self._total: int = 0

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)   # mark as recently used
        return self._cache[key]

    def put(self, key, value):
        size = value.nbytes
        if key in self._cache:
            self._total -= self._nbytes.pop(key)
            del self._cache[key]
        # Evict LRU entries until there is room
        while self._cache and self._total + size > self.max_bytes:
            old_key, _ = self._cache.popitem(last=False)
            self._total -= self._nbytes.pop(old_key)
        self._cache[key] = value
        self._nbytes[key] = size
        self._total += size

    def __contains__(self, key):
        return key in self._cache

    @property
    def used_gb(self) -> float:
        return self._total / 1e9

    def clear(self):
        self._cache.clear()
        self._nbytes.clear()
        self._total = 0


# =============================================================================
# Config loading
# =============================================================================

def load_config(config_path: str | Path) -> dict:
    """Load YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, project_root: str | Path) -> dict:
    """Resolve relative paths in config to absolute paths."""
    project_root = Path(project_root)
    cfg = cfg.copy()
    cfg["_project_root"] = str(project_root)
    cfg["_data_root"] = str(project_root / cfg["data_root"])
    cfg["_fmri_dir"] = str(project_root / cfg["data_root"] / cfg["fmri"]["dir"])
    cfg["_features_dir"] = str(project_root / cfg["data_root"] / cfg["features"]["dir"])
    # Optional NPY features directory (faster loading than H5)
    npy_subdir = cfg["features"].get("npy_dir")
    if npy_subdir:
        cfg["_features_npy_dir"] = str(project_root / cfg["data_root"] / npy_subdir)
    else:
        cfg["_features_npy_dir"] = None
    return cfg


# =============================================================================
# fMRI data loading
# =============================================================================

def _get_fmri_filepath(fmri_dir: str, subject: str, task: str) -> Path:
    """Construct path to fMRI h5 file for a subject/task.

    File naming convention from Algonauts 2025:
      friends: {subject}_task-friends_..._desc-s123456_bold.h5
      movie10: {subject}_task-movie10_..._bold.h5
    """
    subj_dir = Path(fmri_dir) / subject / "func"
    atlas_part = "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
    stem = f"{subject}_task-{task}_{atlas_part}"
    if task == "friends":
        return subj_dir / f"{stem}_desc-s123456_bold.h5"
    else:
        return subj_dir / f"{stem}_bold.h5"


def load_fmri_clip(
    fmri_dir: str,
    subject: str,
    task: str,
    clip_key: str,
    standardize: bool = True,
    run: int | None = None,
) -> np.ndarray:
    """Load and preprocess fMRI data for a single clip.

    Parameters
    ----------
    fmri_dir : str
        Path to fMRI root directory.
    subject : str
        Subject ID, e.g. 'sub-01'.
    task : str
        Task name: 'friends' or 'movie10'.
    clip_key : str
        H5 key for the clip, e.g. '01a', 'bourne01'.
    standardize : bool
        Whether to z-score the data per clip.
    run : int or None
        For movie10 life/figures clips with multiple runs, specify 1 or 2.
        None means no run disambiguation (friends or single-run movie10).

    Returns
    -------
    np.ndarray
        Shape (n_voxels, n_trs), float32, z-scored.
    """
    fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
    if not fmri_path.exists():
        raise FileNotFoundError(f"fMRI file not found: {fmri_path}")

    with h5py.File(fmri_path, "r") as f:
        # Find matching key (h5 keys may have prefixes)
        matched = [k for k in f.keys() if clip_key in k]
        # Disambiguate by run if specified
        if run is not None and len(matched) > 1:
            run_str = f"run-{run}"
            matched = [k for k in matched if run_str in k]
        if len(matched) != 1:
            raise ValueError(
                f"Expected 1 match for '{clip_key}' (run={run}) in {fmri_path}, "
                f"got {len(matched)}: {matched}"
            )
        data = f[matched[0]][:].astype(np.float32)

    # Data from h5 is always (n_trs, n_voxels) — transpose to (n_voxels, n_trs)
    data = data.T

    # Replace any NaN/Inf in raw data before normalization
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score normalize per clip (matches nilearn standardize="zscore_sample")
    if standardize:
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        data = (data - mean) / std
        # Clamp any residual NaNs (e.g. all-zero voxels)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data


# =============================================================================
# Feature loading
# =============================================================================

def load_feature_clip(
    features_dir: str,
    modality: str,
    modality_cfg: dict,
    movie_type: str,
    stimulus_type: str,
    clip_name: str,
    layer_aggregation: str = "last",
    keep_tokens: bool = False,
) -> np.ndarray:
    """Load pre-extracted features for a single clip.

    Parameters
    ----------
    features_dir : str
        Root features directory.
    modality : str
        Modality name: 'video', 'audio', 'text', 'omni'.
    modality_cfg : dict
        Modality config from data.yaml.
    movie_type, stimulus_type, clip_name : str
        E.g. 'friends', 's1', 'friends_s01e01a'.
    layer_aggregation : str
        How to aggregate/select across layers: 'last' or 'mean'.
    keep_tokens : bool
        If True and data_format is tr_tokens_dim (omni), keep all tokens
        instead of mean-pooling. Returns (n_tokens, dim, n_trs).

    Returns
    -------
    np.ndarray
        Standard: shape (1, dim, n_timepoints) at sample_freq, float32.
        Omni keep_tokens: shape (n_tokens, dim, n_trs), float32.
    """
    subdir = modality_cfg["subdir"]
    pattern = modality_cfg["file_pattern"]
    h5_key = modality_cfg["h5_key"]
    data_format = modality_cfg.get("data_format", "layers_dim_tr")

    filename = pattern.format(movie_type=movie_type, stimulus_type=stimulus_type)
    h5_path = Path(features_dir) / subdir / filename

    if not h5_path.exists():
        raise FileNotFoundError(f"Feature file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # Try exact clip_name first, then prefixed variants
        # (text extractor uses 'movie10_bourne01' while others use 'bourne01')
        actual_key = None
        candidates = [
            clip_name,
            f"{movie_type}_{clip_name}",   # e.g. 'movie10_bourne01'
        ]
        for cand in candidates:
            if cand in f:
                actual_key = cand
                break
        if actual_key is None:
            raise KeyError(
                f"Clip '{clip_name}' not found in {h5_path}. "
                f"Tried: {candidates}. Available: {list(f.keys())[:5]}..."
            )
        data = f[actual_key][h5_key][:].astype(np.float32)

    if data_format == "tr_tokens_dim":
        # omni: shape (n_trs, n_tokens, dim)
        if keep_tokens:
            # Keep all tokens: (n_trs, n_tokens, dim) → (n_tokens, dim, n_trs)
            data = data.transpose(1, 2, 0)  # (n_tokens, dim, n_trs)
        else:
            # Mean-pool tokens: (n_trs, n_tokens, dim) → (1, dim, n_trs)
            T, N, D = data.shape
            data = data.mean(axis=1)  # (n_trs, dim)
            data = data.T[np.newaxis]  # (1, dim, n_trs)
    else:
        # Standard: (n_layers, dim, n_timepoints)
        # Apply layer aggregation to get (1, dim, n_timepoints)
        if layer_aggregation == "last":
            data = data[-1:, :, :]   # keep last layer: (1, dim, T)
        elif layer_aggregation == "mean":
            data = data.mean(axis=0, keepdims=True)  # (1, dim, T)
        # else: keep all layers as-is

    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # (1, dim, n_timepoints)


def load_feature_clip_npy(
    features_npy_dir: str,
    modality: str,
    modality_cfg: dict,
    movie_type: str,
    stimulus_type: str,
    clip_name: str,
    keep_tokens: bool = False,
) -> np.ndarray | None:
    """Load pre-converted NPY feature for a single clip.

    NPY files are produced by convert_h5_to_npy.py with layer mean-pooling.
    Shape on disk: (dim, n_timepoints), float32.
    Returns (1, dim, n_timepoints) to match H5 loader output, or None if not found.

    For omni with keep_tokens:
      If NPY was converted with --keep_omni_tokens, shape is (n_tokens, dim, n_trs).
      If NPY is old format (dim, n_trs), returns None to trigger H5 fallback.
    """
    subdir = modality_cfg["subdir"]
    pattern = modality_cfg["file_pattern"]
    # stem of H5 filename without extension, e.g. 'friends_s1_features_vjepa2'
    h5_stem = pattern.format(
        movie_type=movie_type, stimulus_type=stimulus_type
    ).replace(".h5", "")

    npy_path = Path(features_npy_dir) / subdir / h5_stem / f"{clip_name}.npy"
    if not npy_path.exists():
        return None

    data = np.load(npy_path)  # (dim, n_trs) or (n_tokens, dim, n_trs)

    if keep_tokens:
        # Expect (n_tokens, dim, n_trs) from --keep_omni_tokens conversion
        if data.ndim == 3:
            return data  # already (n_tokens, dim, n_trs)
        else:
            # Old mean-pooled format (dim, n_trs) — cannot use for keep_tokens
            return None

    return data[np.newaxis]   # (1, dim, n_trs)


def resample_features_to_tr(
    features: np.ndarray,
    feature_freq: float,
    tr: float,
    n_trs: int,
) -> np.ndarray:
    """Resample features from extraction frequency to TR grid.

    Parameters
    ----------
    features : np.ndarray
        Shape (1, dim, n_timepoints_at_feature_freq).
    feature_freq : float
        Original sampling frequency (Hz), e.g. 2.0.
    tr : float
        fMRI repetition time (seconds), e.g. 1.49.
    n_trs : int
        Target number of TRs.

    Returns
    -------
    np.ndarray
        Shape (1, dim, n_trs), resampled to TR grid.
    """
    L, D, T_orig = features.shape

    if T_orig == n_trs:
        return features

    # Use F.interpolate: needs (batch, channels, length)
    # Reshape (L, D, T) → (L*D, 1, T) for 1D interpolation
    feat_tensor = torch.from_numpy(features).reshape(L * D, 1, T_orig)
    resampled = F.interpolate(
        feat_tensor, size=n_trs, mode="linear", align_corners=False
    )
    return resampled.reshape(L, D, n_trs).numpy()


# =============================================================================
# Clip index builder
# =============================================================================

def _clip_name_to_fmri_key(clip_name: str, task: str) -> str:
    """Convert clip stem name to fMRI h5 key.

    Examples:
        friends_s01e01a (task=friends) → s01e01a
        movie10_bourne01 (task=movie10) → bourne01
    """
    if task == "friends":
        # friends_s01e01a → s01e01a
        prefix = "friends_"
        if clip_name.startswith(prefix):
            return clip_name[len(prefix):]
        return clip_name
    else:
        # movie10_bourne01  → bourne01
        prefix = "movie10_"
        if clip_name.startswith(prefix):
            return clip_name[len(prefix):]
        return clip_name


def _deterministic_split(uid: str, val_ratio: float = 0.1) -> str:
    """Deterministically assign a clip to train or val based on hash."""
    hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
    rng = random.Random(hashed)
    return "val" if rng.random() < val_ratio else "train"


def _enumerate_fmri_runs(
    fmri_dir: str, subject: str, task: str, clip_key: str,
) -> list[int | None]:
    """Find which runs exist for a clip_key in the fMRI h5.

    For friends and single-run movie10 clips (bourne, wolf), returns [None].
    For movie10 life/figures clips with run-1/run-2, returns [1, 2] etc.
    Returns empty list if clip not found at all.
    """
    fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
    if not fmri_path.exists():
        return []

    with h5py.File(fmri_path, "r") as f:
        matched = [k for k in f.keys() if clip_key in k]

    if not matched:
        return []
    if len(matched) == 1:
        return [None]  # single match, no run disambiguation needed

    # Multiple matches: extract run numbers
    runs = []
    for k in matched:
        if "run-" in k:
            # e.g. 'ses-006_task-life01_run-1' → 1
            run_str = k.split("run-")[-1]
            try:
                runs.append(int(run_str))
            except ValueError:
                runs.append(None)
        else:
            runs.append(None)
    return sorted(r for r in runs if r is not None) or [None]


def build_clip_index(
    cfg: dict,
) -> list[dict]:
    """Build index of all available (subject, clip) pairs.

    Returns a list of dicts with keys:
        subject, task, movie_type, stimulus_type, clip_name, fmri_key, split, run

    For movie10 clips with multiple fMRI runs (life/figures), each run
    produces a separate index entry (same features, different fMRI scan).
    """
    fmri_dir = cfg["_fmri_dir"]
    features_dir = cfg["_features_dir"]
    subjects = cfg["subjects"]
    splits_cfg = cfg["splits"]
    val_ratio = cfg.get("val_ratio", 0.1)
    modalities = cfg["features"]["modalities"]

    index = []

    for task, task_splits in splits_cfg.items():
        for split_name, stim_types in task_splits.items():
            for stim_type in stim_types:
                # Check which clips have features available
                # Use the first available modality to discover clip names
                first_mod = next(iter(modalities))
                mod_cfg = modalities[first_mod]
                filename = mod_cfg["file_pattern"].format(
                    movie_type=task, stimulus_type=stim_type
                )
                feat_path = Path(features_dir) / mod_cfg["subdir"] / filename

                if not feat_path.exists():
                    logger.warning("Feature file missing, skipping: %s", feat_path)
                    continue

                with h5py.File(feat_path, "r") as f:
                    clip_names = list(f.keys())

                for clip_name in clip_names:
                    fmri_key = _clip_name_to_fmri_key(clip_name, task)

                    # Determine split
                    if split_name == "test":
                        clip_split = "test"
                    else:
                        clip_split = _deterministic_split(
                            clip_name, val_ratio=val_ratio
                        )

                    for subject in subjects:
                        # Verify fMRI exists for this subject
                        fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
                        if not fmri_path.exists():
                            continue

                        # Enumerate runs (movie10 life/figures may have run-1/run-2)
                        if split_name == "test":
                            # Test clips have no fMRI — just add one entry
                            runs = [None]
                        else:
                            runs = _enumerate_fmri_runs(
                                fmri_dir, subject, task, fmri_key)
                            if not runs:
                                continue

                        for run in runs:
                            index.append({
                                "subject": subject,
                                "task": task,
                                "movie_type": task,
                                "stimulus_type": stim_type,
                                "clip_name": clip_name,
                                "fmri_key": fmri_key,
                                "split": clip_split,
                                "run": run,
                            })

    logger.info(
        "Built clip index: %d entries (%d subjects × clips)",
        len(index), len(subjects),
    )
    return index


# =============================================================================
# Dataset
# =============================================================================

class SlidingWindowDataset(Dataset):
    """Sliding-window dataset for fMRI encoding.

    Each sample:
      - Input: `context_trs` timesteps of multimodal features
      - Target: 1 fMRI TR (1000 voxels)
      - Subject ID: integer index

    The sliding window covers ~14.9s (10 TRs) of stimulus features
    preceding each target fMRI TR, allowing the model to learn HRF
    timing flexibly for different brain regions.
    """

    def __init__(
        self,
        cfg: dict,
        split: str = "train",
        modalities: list[str] | None = None,
        max_cache_gb: float = 10.0,
    ):
        """
        Parameters
        ----------
        cfg : dict
            Config dict (from load_config + resolve_paths).
        split : str
            Dataset split: 'train', 'val', or 'test'.
        modalities : list[str] or None
            Which modalities to load. None = all available.
        max_cache_gb : float
            Maximum RAM to use for the LRU data cache (default 10 GB).
            Oldest clip data is evicted when the limit is exceeded.
        """
        self.cfg = cfg
        self.split = split
        self.tr = cfg["fmri"]["tr"]
        self.n_voxels = cfg["fmri"]["n_voxels"]
        self.feature_freq = cfg["features"]["sample_freq"]
        self.layer_aggregation = cfg["features"].get("layer_aggregation", "last")

        sw = cfg["sliding_window"]
        self.context_duration = sw["context_duration"]
        self.stride = sw["stride"]
        self.context_trs = round(self.context_duration / self.tr)

        if modalities is None:
            self.modalities = list(cfg["features"]["modalities"].keys())
        else:
            self.modalities = modalities

        self.fmri_dir = cfg["_fmri_dir"]
        self.features_dir = cfg["_features_dir"]
        self.features_npy_dir = cfg.get("_features_npy_dir")  # None if not configured
        self.standardize = cfg["preprocessing"]["fmri"]["standardize"] == "zscore_sample"

        # Track which modalities use keep_tokens (e.g., omni)
        self._keep_tokens: dict[str, bool] = {}
        for mod in self.modalities:
            mod_cfg = cfg["features"]["modalities"].get(mod, {})
            self._keep_tokens[mod] = mod_cfg.get("keep_tokens", False)

        # Subject name → integer ID mapping
        self.subject_to_id = {
            s: i for i, s in enumerate(cfg["subjects"])
        }

        # Build clip index & sample index
        clip_index = build_clip_index(cfg)
        self.clip_index = [c for c in clip_index if c["split"] == split]

        if not self.clip_index:
            logger.warning("No clips found for split '%s'", split)

        # Build sample index: (clip_idx, target_tr)
        self._samples = self._build_sample_index()
        logger.info(
            "SlidingWindowDataset[%s]: %d samples from %d clips, "
            "context=%d TRs (%.1fs), stride=%.2fs",
            split, len(self._samples), len(self.clip_index),
            self.context_trs, self.context_duration, self.stride,
        )

        # Unified LRU cache for fMRI and features (size-bounded)
        self._cache = LRUCache(max_bytes=int(max_cache_gb * 1e9))
        # RAM store: populated by preload_to_ram() for zero-HDD-IO training
        self._ram_store: dict = {}
        npy_status = f"NPY dir: {self.features_npy_dir}" if self.features_npy_dir else "H5 fallback only"
        logger.info("LRU data cache: max %.1f GB | %s", max_cache_gb, npy_status)

    def preload_to_ram(self) -> None:
        """Preload ALL clip features and fMRI into RAM.

        Features are deduplicated across subjects (same clip = same features),
        reducing memory from ~57GB to ~15GB. Only fMRI varies per subject.
        """
        unique_clips = set()
        for clip_idx, _ in self._samples:
            unique_clips.add(clip_idx)

        total_bytes = 0
        feat_dedup: dict[tuple, np.ndarray] = {}  # (movie_type, stim, clip, mod) → array

        logger.info("Preloading %d clips × %d modalities + fMRI to RAM "
                    "(dedup features across subjects)...",
                    len(unique_clips), len(self.modalities))

        for clip_idx in tqdm(sorted(unique_clips),
                             desc=f"  Preload [{self.split}]", leave=True):
            info = self.clip_index[clip_idx]

            # ── fMRI: unique per subject-clip (no dedup) ──
            fmri_key = ("fmri", clip_idx)
            if fmri_key not in self._ram_store:
                try:
                    data = load_fmri_clip(
                        self.fmri_dir, info["subject"], info["task"],
                        info["fmri_key"], standardize=self.standardize,
                    )
                    self._ram_store[fmri_key] = data
                    total_bytes += data.nbytes
                except Exception as e:
                    logger.warning("Failed to preload fMRI clip %d: %s",
                                   clip_idx, e)

            # ── Features: dedup by (movie_type, stim, clip_name, mod) ──
            dedup_id = (info["movie_type"], info["stimulus_type"],
                        info["clip_name"])
            for mod in self.modalities:
                feat_key = ("feat", clip_idx, mod)
                dedup_full = (*dedup_id, mod)

                if dedup_full in feat_dedup:
                    # Reuse existing array (zero-copy reference)
                    self._ram_store[feat_key] = feat_dedup[dedup_full]
                else:
                    try:
                        data = self._load_features(clip_idx, mod)
                        feat_dedup[dedup_full] = data
                        self._ram_store[feat_key] = data
                        total_bytes += data.nbytes
                    except Exception as e:
                        logger.warning("Failed to preload %s clip %d: %s",
                                       mod, clip_idx, e)

        # Clear LRU cache — data is now in RAM store
        self._cache = LRUCache(max_bytes=0)
        n_dedup = len(feat_dedup)
        logger.info("  Preloaded %.2f GB into RAM (%d entries, %d unique features)",
                    total_bytes / 1e9, len(self._ram_store), n_dedup)

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """Build flattened list of (clip_idx, target_tr) tuples.

        For each clip, we create a sample for every valid target TR.
        A target TR is valid if there are enough preceding TRs
        to fill the context window.
        """
        samples = []
        stride_trs = max(1, round(self.stride / self.tr))

        for clip_idx, clip_info in enumerate(self.clip_index):
            # Get number of TRs from fMRI data
            n_trs = self._get_clip_n_trs(clip_info)
            if n_trs is None:
                continue

            # Target TRs: from context_trs (first valid target) to n_trs-1
            for target_tr in range(self.context_trs, n_trs, stride_trs):
                samples.append((clip_idx, target_tr))

        return samples

    def _get_clip_n_trs(self, clip_info: dict) -> int | None:
        """Get number of TRs for a clip by peeking at fMRI data."""
        try:
            fmri_path = _get_fmri_filepath(
                self.fmri_dir, clip_info["subject"], clip_info["task"]
            )
            run = clip_info.get("run")
            with h5py.File(fmri_path, "r") as f:
                matched = [k for k in f.keys() if clip_info["fmri_key"] in k]
                # Disambiguate by run if specified
                if run is not None and len(matched) > 1:
                    run_str = f"run-{run}"
                    matched = [k for k in matched if run_str in k]
                if len(matched) != 1:
                    return None
                shape = f[matched[0]].shape
                # shape is always (n_trs, n_voxels) from h5
                return shape[0]
        except Exception as e:
            logger.warning("Could not read fMRI for %s: %s", clip_info, e)
            return None

    def _load_fmri(self, clip_idx: int) -> np.ndarray:
        """Load fMRI for clip (RAM store or LRU cached)."""
        key = ("fmri", clip_idx)
        # RAM store takes priority (populated by preload_to_ram)
        if key in self._ram_store:
            return self._ram_store[key]
        cached = self._cache.get(key)
        if cached is None:
            info = self.clip_index[clip_idx]
            cached = load_fmri_clip(
                self.fmri_dir, info["subject"], info["task"],
                info["fmri_key"], standardize=self.standardize,
                run=info.get("run"),
            )
            self._cache.put(key, cached)
        return cached

    def _load_features(self, clip_idx: int, modality: str) -> np.ndarray:
        """Load and resample features for clip (LRU cached).

        Returns array with shape (1, dim, n_trs).
        """
        key = ("feat", clip_idx, modality)
        # RAM store takes priority (populated by preload_to_ram)
        if key in self._ram_store:
            return self._ram_store[key]
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        info = self.clip_index[clip_idx]
        mod_cfg = self.cfg["features"]["modalities"][modality]
        keep_tokens = self._keep_tokens.get(modality, False)

        # ── NPY fast path (pre-converted, already mean-pooled, on TR grid) ──
        if self.features_npy_dir:
            raw = load_feature_clip_npy(
                self.features_npy_dir, modality, mod_cfg,
                info["movie_type"], info["stimulus_type"], info["clip_name"],
                keep_tokens=keep_tokens,
            )  # (1, dim, n_trs) or (n_tokens, dim, n_trs) or None
            if raw is not None:
                # NPY is already on TR grid; just trim/pad to match fMRI n_trs
                fmri = self._load_fmri(clip_idx)
                n_trs = fmri.shape[1]
                T_feat = raw.shape[2]
                if T_feat > n_trs:
                    raw = raw[:, :, :n_trs]
                elif T_feat < n_trs:
                    pad = np.zeros((raw.shape[0], raw.shape[1], n_trs - T_feat), dtype=raw.dtype)
                    raw = np.concatenate([raw, pad], axis=2)
                self._cache.put(key, raw)
                return raw

        # ── H5 fallback (with layer aggregation + TR resampling) ──
        raw = load_feature_clip(
            self.features_dir, modality, mod_cfg,
            info["movie_type"], info["stimulus_type"], info["clip_name"],
            layer_aggregation=self.layer_aggregation,
            keep_tokens=keep_tokens,
        )  # shape: (1, dim, T) or (n_tokens, dim, T) for omni keep_tokens

        fmri = self._load_fmri(clip_idx)
        n_trs = fmri.shape[1]

        data_format = mod_cfg.get("data_format", "layers_dim_tr")
        if data_format == "tr_tokens_dim":
            # omni: already on TR grid after load_feature_clip; trim/pad to n_trs
            T_feat = raw.shape[2]
            if T_feat > n_trs:
                resampled = raw[:, :, :n_trs]
            elif T_feat < n_trs:
                pad = np.zeros(
                    (raw.shape[0], raw.shape[1], n_trs - T_feat), dtype=raw.dtype)
                resampled = np.concatenate([raw, pad], axis=2)
            else:
                resampled = raw
        else:
            # 2Hz features: resample to TR grid
            resampled = resample_features_to_tr(
                raw, self.feature_freq, self.tr, n_trs
            )

        self._cache.put(key, resampled)
        return resampled

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("LRU cache cleared")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns
        -------
        dict with keys:
            'features': dict[str, Tensor]
                {modality: (context_trs, n_layers * dim)} — flattened layers
            'fmri': Tensor (n_voxels,)
                Target fMRI for 1 TR
            'subject_id': Tensor (1,)
                Integer subject index
        """
        clip_idx, target_tr = self._samples[idx]
        clip_info = self.clip_index[clip_idx]

        # Context window: [target_tr - context_trs, target_tr)
        ctx_start = target_tr - self.context_trs
        ctx_end = target_tr

        # Load features for each modality
        features = {}
        for mod in self.modalities:
            try:
                feat = self._load_features(clip_idx, mod)  # (L, D, T)
                keep_tokens = self._keep_tokens.get(mod, False)
                # Crop context window
                feat_window = feat[:, :, ctx_start:ctx_end]  # (L, D, context_trs)

                if keep_tokens:
                    # Omni keep_tokens: (N_tokens, D, ctx) → (ctx * N_tokens, D)
                    # Interleave: for each timestep, all N tokens appear sequentially
                    N, D, T = feat_window.shape
                    # permute to (T, N, D) then reshape to (T*N, D)
                    feat_flat = feat_window.transpose(2, 0, 1).reshape(T * N, D)
                else:
                    # Standard: (L, D, T) → (T, L*D)
                    L, D, T = feat_window.shape
                    feat_flat = feat_window.reshape(L * D, T).T  # (context_trs, L*D)

                features[mod] = torch.from_numpy(feat_flat)
            except (FileNotFoundError, KeyError) as e:
                logger.debug("Missing %s for clip %s: %s", mod, clip_info["clip_name"], e)
                # Return zeros for missing modalities
                if self._keep_tokens.get(mod, False):
                    n_tokens = self.cfg["features"]["modalities"][mod].get("n_tokens", 5)
                    dim = self.cfg["features"]["modalities"][mod]["dim"]
                    features[mod] = torch.zeros(self.context_trs * n_tokens, dim)
                else:
                    features[mod] = torch.zeros(self.context_trs, 1)

        # Load fMRI target (single TR)
        fmri = self._load_fmri(clip_idx)  # (n_voxels, n_trs)
        fmri_target = torch.from_numpy(fmri[:, target_tr].copy())

        # Subject ID
        subject_id = torch.tensor(
            self.subject_to_id[clip_info["subject"]], dtype=torch.long
        )

        return {
            "features": features,
            "fmri": fmri_target,
            "subject_id": subject_id,
        }


# =============================================================================
# DataLoader builder
# =============================================================================

def build_dataloaders(
    config_or_path: str | Path | dict,
    project_root: str | Path | None = None,
    splits: list[str] | None = None,
    modalities: list[str] | None = None,
) -> dict[str, DataLoader]:
    """Build DataLoaders for specified splits.

    Parameters
    ----------
    config_or_path : str, Path, or dict
        Path to YAML config file, or a pre-loaded config dict.
        When a dict is passed, it must contain the same structure as data.yaml
        (data_root, fmri, features, etc.).
    project_root : str, Path, or None
        Project root directory. If None, inferred from config path.
    splits : list[str] or None
        Splits to build. None = ['train', 'val'].
    modalities : list[str] or None
        Modalities to include. None = all.

    Returns
    -------
    dict[str, DataLoader]
        Mapping from split name to DataLoader.
    """
    if isinstance(config_or_path, dict):
        cfg = config_or_path
        if project_root is None:
            raise ValueError(
                "project_root is required when passing a config dict")
        cfg = resolve_paths(cfg, project_root)
    else:
        config_path = Path(config_or_path)
        if project_root is None:
            project_root = config_path.resolve().parent.parent.parent
        cfg = load_config(config_path)
        cfg = resolve_paths(cfg, project_root)

    if splits is None:
        splits = ["train", "val"]

    dl_cfg = cfg.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 64)
    num_workers = dl_cfg.get("num_workers", 0)
    max_cache_gb = dl_cfg.get("max_cache_gb", 10.0)
    pin_memory = dl_cfg.get("pin_memory", True)
    prefetch = dl_cfg.get("prefetch_factor", 2)

    loaders = {}
    for split in splits:
        dataset = SlidingWindowDataset(
            cfg=cfg, split=split, modalities=modalities,
            max_cache_gb=max_cache_gb,
        )
        if len(dataset) == 0:
            logger.warning("Empty dataset for split '%s', skipping", split)
            continue
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch if num_workers > 0 else None,
            drop_last=(split == "train"),
        )

    return loaders


def preload_datasets_to_ram(loaders: dict) -> None:
    """Preload all features and fMRI data into RAM for all datasets in loaders.

    Call this once after build_dataloaders() to eliminate HDD I/O during training.
    Requires sufficient RAM to hold all NPY/fMRI data (~25GB for full dataset).
    """
    for split, loader in loaders.items():
        dataset = loader.dataset
        if hasattr(dataset, 'preload_to_ram'):
            dataset.preload_to_ram()
