"""Per-timestep dataset for Multimodal VAE training.

Supports two modes:
  1. NPY mode (fast): loads pre-pooled features from .npy files (T, D)
  2. H5 mode (slow):  loads all layers from .h5 files (T, n_layers, D)

Each sample is a single TR, yielding features for all modalities.
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultimodalVAEDataset(Dataset):
    """Per-timestep feature dataset for Multimodal VAE.

    Parameters
    ----------
    features_dir : str or Path
        Root features directory.
    modalities : dict
        {mod_name: {'subdir': str, 'dim': int, 'n_layers': int}}
    splits_cfg : dict
        {task: {split_name: [stim_types]}}
    split : str
        'train' or 'val'.
    stride : int
        Stride between sampled TRs within a clip.
    mode : str
        'npy' for pre-pooled NPY files, 'h5' for raw multi-layer H5.
    """

    def __init__(
        self,
        features_dir: str | Path,
        modalities: dict,
        splits_cfg: dict,
        split: str = "train",
        stride: int = 1,
        mode: str = "npy",
    ):
        self.features_dir = Path(features_dir)
        self.modalities = modalities
        self.split = split
        self.stride = stride
        self.mode = mode

        # Cache: (mod, task, stim, clip) → np.ndarray
        self._cache = {}

        self.samples = self._build_index(splits_cfg)

    def _build_index(self, splits_cfg: dict) -> list[dict]:
        """Build sample index by enumerating clips from the first modality."""
        samples = []

        first_mod_name = list(self.modalities.keys())[0]
        first_mod_cfg = self.modalities[first_mod_name]

        ext = "*.npy" if self.mode == "npy" else "*.h5"

        for task, splits in splits_cfg.items():
            stim_types = splits.get(self.split, [])
            if not stim_types:
                continue

            for stim_type in stim_types:
                clip_dir = self.features_dir / first_mod_cfg['subdir'] / task / stim_type

                if not clip_dir.exists():
                    logger.warning("Feature dir missing: %s", clip_dir)
                    continue

                clip_stems = sorted(p.stem for p in clip_dir.glob(ext))

                for clip_stem in clip_stems:
                    # Get TR count from first modality
                    n_trs = self._get_clip_length(first_mod_cfg, task, stim_type, clip_stem)
                    if n_trs is None or n_trs == 0:
                        continue

                    for tr_idx in range(0, n_trs, self.stride):
                        samples.append({
                            'task': task,
                            'stim_type': stim_type,
                            'clip_stem': clip_stem,
                            'tr_idx': tr_idx,
                        })

        logger.info("Built %d samples for %s split (mode=%s).", len(samples), self.split, self.mode)
        return samples

    def _get_clip_length(self, mod_cfg, task, stim_type, clip_stem) -> Optional[int]:
        """Get TR count via quick file read."""
        if self.mode == "npy":
            path = self.features_dir / mod_cfg['subdir'] / task / stim_type / f"{clip_stem}.npy"
            if not path.exists():
                return None
            # Read just the header to get shape (fast, no full load)
            try:
                arr = np.load(path, mmap_mode='r')
                return arr.shape[0]
            except Exception:
                return None
        else:
            path = self.features_dir / mod_cfg['subdir'] / task / stim_type / f"{clip_stem}.h5"
            if not path.exists():
                return None
            try:
                with h5py.File(path, 'r') as f:
                    return f[next(iter(f.keys()))].shape[0]
            except Exception:
                return None

    def _load_clip_npy(self, mod_name, mod_cfg, task, stim_type, clip_stem):
        """Load and cache a clip from NPY. Returns (T, D)."""
        cache_key = (mod_name, task, stim_type, clip_stem)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.features_dir / mod_cfg['subdir'] / task / stim_type / f"{clip_stem}.npy"
        if not path.exists():
            return None

        data = np.load(path).astype(np.float32)  # (T, D)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        self._cache[cache_key] = data
        return data

    def _load_clip_h5(self, mod_name, mod_cfg, task, stim_type, clip_stem):
        """Load and cache all layers from H5. Returns (T, n_layers, D)."""
        cache_key = (mod_name, task, stim_type, clip_stem)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.features_dir / mod_cfg['subdir'] / task / stim_type / f"{clip_stem}.h5"
        if not path.exists():
            return None

        try:
            with h5py.File(path, 'r') as f:
                layers = []
                for lk in sorted(f.keys()):
                    arr = f[lk][:].astype(np.float32)
                    if arr.ndim == 3 and arr.shape[1] == 1:
                        arr = arr.squeeze(1)
                    try:
                        if not np.isfinite(float(np.std(arr))) or float(np.std(arr)) == 0:
                            continue
                    except (OverflowError, FloatingPointError):
                        continue
                    layers.append(arr)
                if not layers:
                    return None
                stacked = np.stack(layers, axis=0).transpose(1, 0, 2)  # (T, L, D)
                stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
            self._cache[cache_key] = stacked
            return stacked
        except Exception as e:
            logger.warning("Error loading %s: %s", path, e)
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        info = self.samples[idx]
        task = info['task']
        stim_type = info['stim_type']
        clip_stem = info['clip_stem']
        tr_idx = info['tr_idx']

        result = {}

        for mod_name, mod_cfg in self.modalities.items():
            if self.mode == "npy":
                clip_data = self._load_clip_npy(mod_name, mod_cfg, task, stim_type, clip_stem)
                if clip_data is not None and tr_idx < clip_data.shape[0]:
                    result[mod_name] = torch.from_numpy(clip_data[tr_idx].copy())  # (D,)
                else:
                    result[mod_name] = torch.zeros(mod_cfg['dim'], dtype=torch.float32)
            else:
                clip_data = self._load_clip_h5(mod_name, mod_cfg, task, stim_type, clip_stem)
                if clip_data is not None and tr_idx < clip_data.shape[0]:
                    result[mod_name] = torch.from_numpy(clip_data[tr_idx].copy())  # (n_layers, D)
                else:
                    n_layers = mod_cfg['n_layers']
                    result[mod_name] = torch.zeros(n_layers, mod_cfg['dim'], dtype=torch.float32)

        return result
