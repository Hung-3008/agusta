"""BrainFlow v4 Dataset: PCA features + online VAE latent extraction.

Loads precomputed PCA features (from precompute_pca_features.py) and
extracts VAE latents on-the-fly from fMRI data.

Each sample:
    pca_features : (T_out, D_cond) float32 — PCA-reduced stimulus features
    latent       : (T_out, Z) float32 — VAE latent targets (online encoded)
    subject_id   : int — subject index

The VAE latents are extracted online with configurable `num_trial`:
  - num_trial=1: deterministic μ (no sampling noise)
  - num_trial>1: sample `num_trial` times and average (reduces VAE noise)
"""

import hashlib
import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class BrainFlowDataset(Dataset):
    """Dataset for BrainFlow v4: PCA features + VAE latent targets.

    Parameters
    ----------
    cfg : dict
        Full config dict (loaded + resolved).
    split : str
        'train' or 'val'.
    vae_model : nn.Module or None
        Pretrained VAE for online latent extraction. If None, loads raw fMRI.
    num_trial : int
        Number of VAE encoding trials to average. 1 = deterministic μ.
    seq_len : int
        Number of TRs per sample.
    stride : int or None
        Stride between consecutive windows. None = seq_len (non-overlapping).
    """

    def __init__(
        self,
        cfg: dict,
        split: str = "train",
        vae_model=None,
        num_trial: int = 1,
        seq_len: int = 10,
        stride: int | None = None,
    ):
        self.cfg = cfg
        self.split = split
        self.vae_model = vae_model
        self.num_trial = num_trial
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.n_voxels = cfg["fmri"]["n_voxels"]
        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end = cfg["fmri"].get("excluded_samples_end", 0)
        self.standardize = cfg["preprocessing"]["fmri"]["standardize"] == "zscore_sample"

        # Paths
        self.fmri_dir = cfg["_fmri_dir"]
        pca_cfg = cfg.get("pca_features", {})
        project_root = cfg["_project_root"]
        self.pca_dir = Path(project_root) / pca_cfg.get("output_dir", "Data/pca_features")

        # Subjects
        self.subjects = cfg["subjects"]
        self.subject_to_id = {s: i for i, s in enumerate(self.subjects)}

        # Val ratio
        self.val_ratio = cfg.get("val_ratio", 0.1)

        # Build clip + sample indices
        self._clips: list[dict] = []
        self._samples: list[tuple[int, int]] = []
        self._fmri_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._pca_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._latent_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_cache = 100

        self._enumerate_clips()

        logger.info(
            "BrainFlowDataset[%s]: %d samples (seq_len=%d, stride=%d) "
            "from %d clips, num_trial=%d",
            split, len(self._samples), seq_len, self.stride,
            len(self._clips), num_trial,
        )

    def _assign_split(self, uid: str) -> str:
        """Deterministic train/val split based on hash."""
        h = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        s = random.Random(h).random()
        return "val" if s >= (1.0 - self.val_ratio) else "train"

    def _fmri_path(self, subject: str, task: str) -> Path:
        subj_dir = Path(self.fmri_dir) / subject / "func"
        atlas = "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
        stem = f"{subject}_task-{task}_{atlas}"
        if task == "friends":
            return subj_dir / f"{stem}_desc-s123456_bold.h5"
        else:
            return subj_dir / f"{stem}_bold.h5"

    def _enumerate_clips(self):
        """Discover clips from PCA features directory."""
        splits_cfg = self.cfg["splits"]

        for task, task_splits in splits_cfg.items():
            for split_name, stim_types in task_splits.items():
                for stim_type in stim_types:
                    pca_dir = self.pca_dir / task / stim_type
                    if not pca_dir.exists():
                        logger.debug("PCA dir missing: %s", pca_dir)
                        continue

                    clip_files = sorted(pca_dir.glob("*.npy"))
                    for pca_path in clip_files:
                        clip_name = pca_path.stem

                        # Determine fMRI key
                        fmri_key = clip_name
                        if task == "friends" and clip_name.startswith("friends_"):
                            fmri_key = clip_name[len("friends_"):]
                        elif task == "movie10" and clip_name.startswith("movie10_"):
                            fmri_key = clip_name[len("movie10_"):]

                        # Check split
                        if split_name == "test":
                            clip_split = "test"
                        else:
                            clip_split = self._assign_split(clip_name)

                        if clip_split != self.split:
                            continue

                        # Per subject
                        for subject in self.subjects:
                            fp = self._fmri_path(subject, task)
                            if not fp.exists():
                                continue

                            # Verify clip exists in fMRI h5
                            try:
                                with h5py.File(fp, "r") as f:
                                    matched = [k for k in f.keys() if fmri_key in k]
                                    if not matched:
                                        continue
                                    n_trs_raw = f[matched[0]].shape[0]
                            except Exception:
                                continue

                            n_trs = n_trs_raw - self.excl_start - self.excl_end
                            if n_trs < self.seq_len:
                                continue

                            # Check PCA feature alignment
                            pca_data = np.load(pca_path)
                            pca_trs = pca_data.shape[0]
                            usable_trs = min(n_trs, pca_trs)
                            if usable_trs < self.seq_len:
                                continue

                            clip_idx = len(self._clips)
                            self._clips.append({
                                "subject": subject,
                                "subject_id": self.subject_to_id[subject],
                                "task": task,
                                "fmri_key": fmri_key,
                                "fmri_path": str(fp),
                                "pca_path": str(pca_path),
                                "n_trs": usable_trs,
                                "fmri_h5_key": matched[0],
                            })

                            # Build windows
                            for start in range(0, usable_trs - self.seq_len + 1, self.stride):
                                self._samples.append((clip_idx, start))

    def _load_fmri(self, clip: dict) -> np.ndarray:
        """Load fMRI clip → (n_trs, V) float32."""
        cache_key = f"{clip['subject']}_{clip['fmri_h5_key']}"
        if cache_key in self._fmri_cache:
            self._fmri_cache.move_to_end(cache_key)
            return self._fmri_cache[cache_key]

        with h5py.File(clip["fmri_path"], "r") as f:
            raw = f[clip["fmri_h5_key"]]
            end = len(raw) - self.excl_end if self.excl_end > 0 else len(raw)
            data = raw[self.excl_start:end].astype(np.float32)

        if self.standardize:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            data = (data - mean) / std

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        self._fmri_cache[cache_key] = data
        if len(self._fmri_cache) > self._max_cache:
            self._fmri_cache.popitem(last=False)
        return data

    def _load_pca(self, clip: dict) -> np.ndarray:
        """Load PCA features → (n_trs, D_cond) float32."""
        path = clip["pca_path"]
        if path in self._pca_cache:
            self._pca_cache.move_to_end(path)
            return self._pca_cache[path]

        data = np.load(path).astype(np.float32)
        self._pca_cache[path] = data
        if len(self._pca_cache) > self._max_cache:
            self._pca_cache.popitem(last=False)
        return data

    def _encode_latent(self, fmri_window: np.ndarray, subject_id: int) -> np.ndarray:
        """Encode fMRI window to VAE latent, with num_trial averaging.

        Parameters
        ----------
        fmri_window : (T, V) float32
        subject_id : int

        Returns
        -------
        latent : (T, Z) float32
        """
        if self.vae_model is None:
            # No VAE — return raw fMRI as target
            return fmri_window

        device = next(self.vae_model.parameters()).device
        fmri_t = torch.from_numpy(fmri_window).unsqueeze(0).to(device)  # (1, T, V)
        sid_t = torch.tensor([subject_id], dtype=torch.long, device=device)  # (1,)

        with torch.no_grad():
            if self.num_trial <= 1:
                # Deterministic: use μ only
                latent = self.vae_model.get_latent(fmri_t, sid_t)  # (1, T, Z)
            else:
                # Multi-trial: sample num_trial times and average
                latents = []
                for _ in range(self.num_trial):
                    self.vae_model.train()  # Enable sampling
                    z, _, _ = self.vae_model.encode(fmri_t, sid_t)  # (1, T, Z)
                    latents.append(z)
                self.vae_model.eval()
                latent = torch.stack(latents).mean(dim=0)  # (1, T, Z)

        return latent.squeeze(0).cpu().numpy()  # (T, Z)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        clip_idx, start = self._samples[idx]
        clip = self._clips[clip_idx]

        # Load PCA features
        pca_full = self._load_pca(clip)
        pca_window = pca_full[start:start + self.seq_len].copy()  # (T, D_cond)

        # Load fMRI and encode to latent
        fmri_full = self._load_fmri(clip)
        fmri_window = fmri_full[start:start + self.seq_len].copy()  # (T, V)
        latent = self._encode_latent(fmri_window, clip["subject_id"])  # (T, Z)

        return {
            "pca_features": torch.from_numpy(pca_window),
            "latent": torch.from_numpy(latent),
            "fmri": torch.from_numpy(fmri_window),
            "subject_id": torch.tensor(clip["subject_id"], dtype=torch.long),
        }


def build_brainflow_dataloaders(
    cfg: dict,
    vae_model=None,
    num_trial: int = 1,
) -> dict[str, DataLoader | None]:
    """Build train/val dataloaders for BrainFlow v4."""
    bf_cfg = cfg.get("brainflow", {})
    dl_cfg = cfg["dataloader"]
    seq_len = bf_cfg.get("seq_len", 10)
    train_stride = bf_cfg.get("train_stride", 1)  # Dense overlap for training

    loaders = {}
    for split in ("train", "val"):
        stride = train_stride if split == "train" else seq_len
        ds = BrainFlowDataset(
            cfg, split=split,
            vae_model=vae_model,
            num_trial=num_trial,
            seq_len=seq_len,
            stride=stride,
        )
        if len(ds) == 0:
            logger.warning("Empty dataset for split '%s'", split)
            loaders[split] = None
            continue
        bs = dl_cfg["batch_size"] if split == "train" else dl_cfg.get("val_batch_size", dl_cfg["batch_size"])
        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=dl_cfg.get("num_workers", 0),
            pin_memory=dl_cfg.get("pin_memory", True),
            drop_last=(split == "train"),
        )
    return loaders
