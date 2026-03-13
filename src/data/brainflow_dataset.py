"""BrainFlow Dataset: PCA features + precomputed or online VAE latent extraction.

Loads precomputed PCA features and VAE latents. Two modes:
  1. **Precomputed (fast)**: Loads latents from NPY files (from precompute_latents.py).
     Allows num_workers > 0 for parallel prefetching. ~3-5 min/epoch.
  2. **Online (slow, fallback)**: Encodes fMRI through VAE in __getitem__.
     Requires num_workers=0 (GPU in dataloader). ~70 min/epoch.

Each sample:
    pca_features : (T_out, D_cond) float32
    latent       : (T_out, Z) float32
    fmri         : (T_out, V) float32 (for validation decoding)
    subject_id   : int
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
    """Dataset for BrainFlow: PCA features + VAE latent targets.

    Parameters
    ----------
    cfg : dict
        Full config dict (loaded + resolved).
    split : str
        'train' or 'val'.
    vae_model : nn.Module or None
        Pretrained VAE for online latent extraction (fallback only).
    num_trial : int
        Number of VAE encoding trials to average (online mode only).
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
        project_root = cfg["_project_root"]
        pca_cfg = cfg.get("pca_features", {})
        self.pca_dir = Path(project_root) / pca_cfg.get("output_dir", "Data/pca_features")
        self.latent_dir = Path(project_root) / cfg.get("latent_dir", "Data/latents")

        # Subjects
        self.subjects = cfg["subjects"]
        self.subject_to_id = {s: i for i, s in enumerate(self.subjects)}

        # Val ratio
        self.val_ratio = cfg.get("val_ratio", 0.1)

        # Caches
        self._fmri_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._pca_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._latent_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_cache = 300

        # Build clip + sample indices
        self._clips: list[dict] = []
        self._samples: list[tuple[int, int]] = []
        self._has_precomputed = False
        self._enumerate_clips()

        logger.info(
            "BrainFlowDataset[%s]: %d samples (seq_len=%d, stride=%d) "
            "from %d clips, precomputed_latents=%s",
            split, len(self._samples), seq_len, self.stride,
            len(self._clips), self._has_precomputed,
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
        first_checked = False

        for task, task_splits in splits_cfg.items():
            for split_name, stim_types in task_splits.items():
                for stim_type in stim_types:
                    pca_dir = self.pca_dir / task / stim_type
                    if not pca_dir.exists():
                        continue

                    clip_files = sorted(pca_dir.glob("*.npy"))
                    for pca_path in clip_files:
                        clip_name = pca_path.stem

                        fmri_key = clip_name
                        if task == "friends" and clip_name.startswith("friends_"):
                            fmri_key = clip_name[len("friends_"):]
                        elif task == "movie10" and clip_name.startswith("movie10_"):
                            fmri_key = clip_name[len("movie10_"):]

                        if split_name == "test":
                            clip_split = "test"
                        else:
                            clip_split = self._assign_split(clip_name)

                        if clip_split != self.split:
                            continue

                        for subject in self.subjects:
                            fp = self._fmri_path(subject, task)
                            if not fp.exists():
                                continue

                            # Check precomputed latent
                            latent_path = (self.latent_dir / subject / task
                                           / stim_type / f"{clip_name}.npy")
                            has_latent = latent_path.exists()
                            if has_latent and not first_checked:
                                self._has_precomputed = True
                                first_checked = True

                            # Verify fMRI exists
                            try:
                                with h5py.File(fp, "r") as f:
                                    matched = [k for k in f.keys()
                                               if fmri_key in k]
                                    if not matched:
                                        continue
                                    n_trs_raw = f[matched[0]].shape[0]
                            except Exception:
                                continue

                            n_trs = n_trs_raw - self.excl_start - self.excl_end

                            # Usable TRs = min(fMRI, PCA, latent)
                            pca_data = np.load(pca_path)
                            usable_trs = min(n_trs, pca_data.shape[0])

                            if has_latent:
                                lat_data = np.load(latent_path)
                                usable_trs = min(usable_trs, lat_data.shape[0])

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
                                "latent_path": str(latent_path)
                                    if has_latent else None,
                                "n_trs": usable_trs,
                                "fmri_h5_key": matched[0],
                            })

                            for start in range(
                                0, usable_trs - self.seq_len + 1,
                                self.stride,
                            ):
                                self._samples.append((clip_idx, start))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_cached(self, cache, key, load_fn):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        data = load_fn()
        cache[key] = data
        if len(cache) > self._max_cache:
            cache.popitem(last=False)
        return data

    def _load_fmri(self, clip):
        cache_key = f"{clip['subject']}_{clip['fmri_h5_key']}"
        def _load():
            with h5py.File(clip["fmri_path"], "r") as f:
                raw = f[clip["fmri_h5_key"]]
                end = len(raw) - self.excl_end if self.excl_end > 0 else len(raw)
                data = raw[self.excl_start:end].astype(np.float32)
            if self.standardize:
                m = data.mean(axis=0, keepdims=True)
                s = data.std(axis=0, keepdims=True)
                s = np.where(s < 1e-8, 1.0, s)
                data = (data - m) / s
            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return self._load_cached(self._fmri_cache, cache_key, _load)

    def _load_pca(self, clip):
        path = clip["pca_path"]
        return self._load_cached(
            self._pca_cache, path,
            lambda: np.load(path).astype(np.float32),
        )

    def _load_latent(self, clip):
        path = clip["latent_path"]
        return self._load_cached(
            self._latent_cache, path,
            lambda: np.load(path).astype(np.float32),
        )

    def _encode_latent_online(self, fmri_window, subject_id):
        """Fallback: encode fMRI through VAE online (slow)."""
        if self.vae_model is None:
            return fmri_window
        device = next(self.vae_model.parameters()).device
        fmri_t = torch.from_numpy(fmri_window).unsqueeze(0).to(device)
        sid_t = torch.tensor([subject_id], dtype=torch.long, device=device)
        with torch.no_grad():
            if self.num_trial <= 1:
                latent = self.vae_model.get_latent(fmri_t, sid_t)
            else:
                zs = []
                for _ in range(self.num_trial):
                    self.vae_model.train()
                    z, _, _ = self.vae_model.encode(fmri_t, sid_t)
                    zs.append(z)
                self.vae_model.eval()
                latent = torch.stack(zs).mean(dim=0)
        return latent.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        clip_idx, start = self._samples[idx]
        clip = self._clips[clip_idx]
        end = start + self.seq_len

        pca_full = self._load_pca(clip)
        pca_window = pca_full[start:end].copy()

        fmri_full = self._load_fmri(clip)
        fmri_window = fmri_full[start:end].copy()

        if clip["latent_path"] is not None:
            latent_full = self._load_latent(clip)
            latent_window = latent_full[start:end].copy()
        else:
            latent_window = self._encode_latent_online(
                fmri_window, clip["subject_id"],
            )

        return {
            "pca_features": torch.from_numpy(pca_window),
            "latent": torch.from_numpy(latent_window),
            "fmri": torch.from_numpy(fmri_window),
            "subject_id": torch.tensor(clip["subject_id"], dtype=torch.long),
        }


# =====================================================================
# DataLoader builder
# =====================================================================

def build_brainflow_dataloaders(
    cfg: dict,
    vae_model=None,
    num_trial: int = 1,
) -> dict[str, DataLoader | None]:
    """Build train/val dataloaders for BrainFlow."""
    bf_cfg = cfg.get("brainflow", {})
    dl_cfg = cfg["dataloader"]
    seq_len = bf_cfg.get("seq_len", 10)
    train_stride = bf_cfg.get("train_stride", 1)

    # Auto-enable workers when precomputed latents exist
    project_root = cfg["_project_root"]
    latent_dir = Path(project_root) / cfg.get("latent_dir", "Data/latents")
    has_precomputed = latent_dir.exists() and any(latent_dir.rglob("*.npy"))
    num_workers = dl_cfg.get("num_workers", 0)
    if has_precomputed and num_workers == 0:
        num_workers = min(4, dl_cfg.get("max_workers", 4))
        logger.info("Precomputed latents found -> num_workers=%d", num_workers)

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
        bs = (dl_cfg["batch_size"] if split == "train"
              else dl_cfg.get("val_batch_size", dl_cfg["batch_size"]))
        use_workers = num_workers if ds._has_precomputed else 0
        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=use_workers,
            pin_memory=dl_cfg.get("pin_memory", True),
            drop_last=(split == "train"),
            persistent_workers=(use_workers > 0),
        )
    return loaders
