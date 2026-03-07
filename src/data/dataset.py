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
import random
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
        H5 key for the clip, e.g. '01a', 'bourne01_run-1'.
    standardize : bool
        Whether to z-score the data per clip.

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
        if len(matched) != 1:
            raise ValueError(
                f"Expected 1 match for '{clip_key}' in {fmri_path}, "
                f"got {len(matched)}: {matched}"
            )
        data = f[matched[0]][:].astype(np.float32)

    # Data from h5 is always (n_trs, n_voxels) — transpose to (n_voxels, n_trs)
    data = data.T

    # Z-score normalize per clip (matches nilearn standardize="zscore_sample")
    if standardize:
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        data = (data - mean) / std

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
) -> np.ndarray:
    """Load pre-extracted features for a single clip.

    Parameters
    ----------
    features_dir : str
        Root features directory.
    modality : str
        Modality name: 'video', 'audio', 'text'.
    modality_cfg : dict
        Modality config from data.yaml.
    movie_type : str
        E.g. 'friends', 'movie10'.
    stimulus_type : str
        E.g. 's1', 'bourne'.
    clip_name : str
        Clip stem name, e.g. 'friends_s01e01a'.

    Returns
    -------
    np.ndarray
        Shape (n_layers, dim, n_timepoints) at 2Hz, float32.
    """
    subdir = modality_cfg["subdir"]
    pattern = modality_cfg["file_pattern"]
    h5_key = modality_cfg["h5_key"]

    filename = pattern.format(movie_type=movie_type, stimulus_type=stimulus_type)
    h5_path = Path(features_dir) / subdir / filename

    if not h5_path.exists():
        raise FileNotFoundError(f"Feature file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if clip_name not in f:
            raise KeyError(f"Clip '{clip_name}' not found in {h5_path}. "
                           f"Available: {list(f.keys())[:5]}...")
        data = f[clip_name][h5_key][:].astype(np.float32)

    return data  # (n_layers, dim, n_timepoints)


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
        Shape (n_layers, dim, n_timepoints_at_feature_freq).
    feature_freq : float
        Original sampling frequency (Hz), e.g. 2.0.
    tr : float
        fMRI repetition time (seconds), e.g. 1.49.
    n_trs : int
        Target number of TRs.

    Returns
    -------
    np.ndarray
        Shape (n_layers, dim, n_trs), resampled to TR grid.
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
        friends_s01e01a (task=friends) → 01a
        movie10_bourne01 (task=movie10) → bourne01
    """
    if task == "friends":
        # friends_s01e01a → extract "01a" (season + episode + chunk part)
        # Format: friends_s{SS}e{EE}{chunk}
        parts = clip_name.split("_s")[-1]  # e.g. "01e01a"
        season = parts[:2]  # "01"
        rest = parts[2:]    # "e01a"
        ep_chunk = rest[1:]  # "01a" (skip 'e')
        return ep_chunk
    else:
        # movie10_bourne01  → bourne01
        prefix = f"movie10_"
        if clip_name.startswith(prefix):
            return clip_name[len(prefix):]
        return clip_name


def _deterministic_split(uid: str, val_ratio: float = 0.1) -> str:
    """Deterministically assign a clip to train or val based on hash."""
    hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
    rng = random.Random(hashed)
    return "val" if rng.random() < val_ratio else "train"


def build_clip_index(
    cfg: dict,
) -> list[dict]:
    """Build index of all available (subject, clip) pairs.

    Returns a list of dicts with keys:
        subject, task, movie_type, stimulus_type, clip_name, fmri_key, split
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

                        index.append({
                            "subject": subject,
                            "task": task,
                            "movie_type": task,
                            "stimulus_type": stim_type,
                            "clip_name": clip_name,
                            "fmri_key": fmri_key,
                            "split": clip_split,
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
        """
        self.cfg = cfg
        self.split = split
        self.tr = cfg["fmri"]["tr"]
        self.n_voxels = cfg["fmri"]["n_voxels"]
        self.feature_freq = cfg["features"]["sample_freq"]

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
        self.standardize = cfg["preprocessing"]["fmri"]["standardize"] == "zscore_sample"

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

        # Caches for loaded data (keyed by clip index)
        self._fmri_cache: dict[int, np.ndarray] = {}
        self._feature_cache: dict[tuple[int, str], np.ndarray] = {}

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
            with h5py.File(fmri_path, "r") as f:
                matched = [k for k in f.keys() if clip_info["fmri_key"] in k]
                if len(matched) != 1:
                    return None
                shape = f[matched[0]].shape
                # shape is always (n_trs, n_voxels) from h5
                return shape[0]
        except Exception as e:
            logger.warning("Could not read fMRI for %s: %s", clip_info, e)
            return None

    def _load_fmri(self, clip_idx: int) -> np.ndarray:
        """Load fMRI for clip (cached)."""
        if clip_idx not in self._fmri_cache:
            info = self.clip_index[clip_idx]
            self._fmri_cache[clip_idx] = load_fmri_clip(
                self.fmri_dir, info["subject"], info["task"],
                info["fmri_key"], standardize=self.standardize,
            )
        return self._fmri_cache[clip_idx]

    def _load_features(self, clip_idx: int, modality: str) -> np.ndarray:
        """Load and resample features for clip (cached).

        Returns array with shape (n_layers, dim, n_trs) for all modalities.
        Handles both 2Hz features and TR-aligned features (omni).
        """
        cache_key = (clip_idx, modality)
        if cache_key not in self._feature_cache:
            info = self.clip_index[clip_idx]
            mod_cfg = self.cfg["features"]["modalities"][modality]
            raw = load_feature_clip(
                self.features_dir, modality, mod_cfg,
                info["movie_type"], info["stimulus_type"], info["clip_name"],
            )

            fmri = self._load_fmri(clip_idx)
            n_trs = fmri.shape[1]

            if mod_cfg.get("tr_aligned", False):
                # Omni features: (n_TRs, n_layers, hidden_dim)
                # Transpose to (n_layers, hidden_dim, n_TRs) for consistency
                if raw.ndim == 3 and raw.shape[0] != raw.shape[2]:
                    resampled = raw.transpose(1, 2, 0)  # (L, D, T)
                else:
                    resampled = raw
                # Trim or pad to match fMRI TRs
                if resampled.shape[2] > n_trs:
                    resampled = resampled[:, :, :n_trs]
                elif resampled.shape[2] < n_trs:
                    pad = np.zeros(
                        (resampled.shape[0], resampled.shape[1],
                         n_trs - resampled.shape[2]),
                        dtype=resampled.dtype,
                    )
                    resampled = np.concatenate([resampled, pad], axis=2)
            else:
                # 2Hz features: (n_layers, dim, n_timepoints) → resample
                resampled = resample_features_to_tr(
                    raw, self.feature_freq, self.tr, n_trs
                )

            self._feature_cache[cache_key] = resampled
        return self._feature_cache[cache_key]

    def clear_cache(self):
        """Clear all cached data."""
        self._fmri_cache.clear()
        self._feature_cache.clear()

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
                # Crop context window
                feat_window = feat[:, :, ctx_start:ctx_end]  # (L, D, context_trs)
                # Flatten layers: (L, D, T) → (T, L*D)
                L, D, T = feat_window.shape
                feat_flat = feat_window.reshape(L * D, T).T  # (context_trs, L*D)
                features[mod] = torch.from_numpy(feat_flat)
            except (FileNotFoundError, KeyError) as e:
                logger.debug("Missing %s for clip %s: %s", mod, clip_info["clip_name"], e)
                # Return zeros for missing modalities
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
    config_path: str | Path,
    project_root: str | Path | None = None,
    splits: list[str] | None = None,
    modalities: list[str] | None = None,
) -> dict[str, DataLoader]:
    """Build DataLoaders for specified splits.

    Parameters
    ----------
    config_path : str or Path
        Path to data.yaml config file.
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
    config_path = Path(config_path)
    if project_root is None:
        # src/configs/data.yaml → project root is 2 levels up from configs
        project_root = config_path.resolve().parent.parent.parent

    cfg = load_config(config_path)
    cfg = resolve_paths(cfg, project_root)

    if splits is None:
        splits = ["train", "val"]

    dl_cfg = cfg.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 64)
    num_workers = dl_cfg.get("num_workers", 4)
    pin_memory = dl_cfg.get("pin_memory", True)
    prefetch = dl_cfg.get("prefetch_factor", 2)

    loaders = {}
    for split in splits:
        dataset = SlidingWindowDataset(
            cfg=cfg, split=split, modalities=modalities,
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
