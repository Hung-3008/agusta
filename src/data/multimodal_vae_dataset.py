"""Per-timestep dataset for Multimodal VAE training.

Preloads ALL mean-pooled NPY clips into RAM at init for instant
batch loading during training. Each sample is a single TR.

NPY files are expected to be shape (T, D) — mean-pooled across layers.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultimodalVAEDataset(Dataset):
    """Per-timestep feature dataset with full RAM preloading.

    Parameters
    ----------
    features_dir : str or Path
        Root features directory containing modality subdirs.
    modalities : dict
        {mod_name: {'subdir': str, 'dim': int, 'n_layers': int}}
    splits_cfg : dict
        {task: {split_name: [stim_types]}}
    split : str
        'train' or 'val'.
    stride : int
        Stride between sampled TRs within a clip.
    """

    def __init__(
        self,
        features_dir: str | Path,
        modalities: dict,
        splits_cfg: dict,
        split: str = "train",
        stride: int = 1,
        **kwargs,
    ):
        self.features_dir = Path(features_dir)
        self.modalities = modalities
        self.split = split
        self.stride = stride

        # Preload all clips into RAM: {(mod, task, stim, clip): np.ndarray (T, D)}
        self._data = {}
        self.samples = self._build_and_preload(splits_cfg)

    def _build_and_preload(self, splits_cfg: dict) -> list[dict]:
        """Discover clips, preload data, and build sample index in one pass."""
        samples = []

        first_mod_name = list(self.modalities.keys())[0]
        first_mod_cfg = self.modalities[first_mod_name]

        # 1. Discover all clips from first modality
        clip_list = []
        for task, splits in splits_cfg.items():
            stim_types = splits.get(self.split, [])
            if not stim_types:
                continue
            for stim_type in stim_types:
                clip_dir = self.features_dir / first_mod_cfg['subdir'] / task / stim_type
                if not clip_dir.exists():
                    logger.warning("Feature dir missing: %s", clip_dir)
                    continue
                clip_stems = sorted(p.stem for p in clip_dir.glob("*.npy"))
                for cs in clip_stems:
                    clip_list.append((task, stim_type, cs))

        logger.info("[%s] Found %d clips. Preloading all modalities into RAM...",
                     self.split, len(clip_list))

        # 2. Preload all clips for all modalities
        total_bytes = 0
        for mod_name, mod_cfg in self.modalities.items():
            subdir = mod_cfg['subdir']
            loaded = 0
            for task, stim_type, clip_stem in clip_list:
                path = self.features_dir / subdir / task / stim_type / f"{clip_stem}.npy"
                if path.exists():
                    data = np.load(path).astype(np.float32)
                    # Handle multilayer if somehow present
                    if data.ndim == 3:
                        data = data.mean(axis=1)
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    self._data[(mod_name, task, stim_type, clip_stem)] = data
                    total_bytes += data.nbytes
                    loaded += 1
            logger.info("  %s: loaded %d/%d clips", mod_name, loaded, len(clip_list))

        logger.info("  Total preloaded: %.1f GB", total_bytes / 1e9)

        # 3. Build sample index using actual TR counts from preloaded data
        for task, stim_type, clip_stem in clip_list:
            key = (first_mod_name, task, stim_type, clip_stem)
            if key not in self._data:
                continue
            n_trs = self._data[key].shape[0]

            for tr_idx in range(0, n_trs, self.stride):
                samples.append({
                    'task': task,
                    'stim_type': stim_type,
                    'clip_stem': clip_stem,
                    'tr_idx': tr_idx,
                })

        logger.info("[%s] Built %d samples from %d clips.", self.split, len(samples), len(clip_list))
        return samples

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
            key = (mod_name, task, stim_type, clip_stem)
            clip_data = self._data.get(key)

            if clip_data is not None and tr_idx < clip_data.shape[0]:
                result[mod_name] = torch.from_numpy(clip_data[tr_idx].copy())  # (D,)
            else:
                result[mod_name] = torch.zeros(mod_cfg['dim'], dtype=torch.float32)

        return result
