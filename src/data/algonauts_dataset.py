"""AlgonautsDataset — loads from Data/algonauts_2025.features.

Each H5 file contains multiple extracted layers for a single clip.
Features are already at TR resolution (no resampling needed).

Structure:
    Data/algonauts_2025.features/{model}/{task}/{season}/{clip}.h5
    Data/algonauts_2025.competitors/fmri/sub-XX/func/{subject}_{task}_..._bold.h5

Each sample:
    features: {model: Tensor(context_trs, dim)}   # layer-aggregated
    fmri:     Tensor(n_voxels,)                    # target TR
    subject_id: int
"""

import hashlib
import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, project_root: str | Path) -> dict:
    root = Path(project_root)
    cfg = cfg.copy()
    cfg["_project_root"] = str(root)
    cfg["_fmri_dir"] = str(root / cfg["data_root"] / cfg["fmri"]["dir"])
    cfg["_features_root"] = str(root / cfg["data_root"] / cfg["features"]["root"])
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# LRU cache (byte-bounded)
# ─────────────────────────────────────────────────────────────────────────────

class _LRUCache:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._cache: OrderedDict = OrderedDict()
        self._sizes: dict = {}
        self._total = 0

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key, value: np.ndarray):
        sz = value.nbytes
        if key in self._cache:
            self._total -= self._sizes.pop(key)
            del self._cache[key]
        while self._cache and self._total + sz > self.max_bytes:
            old, _ = self._cache.popitem(last=False)
            self._total -= self._sizes.pop(old)
        self._cache[key] = value
        self._sizes[key] = sz
        self._total += sz

    def clear(self):
        self._cache.clear()
        self._sizes.clear()
        self._total = 0


# ─────────────────────────────────────────────────────────────────────────────
# fMRI loading
# ─────────────────────────────────────────────────────────────────────────────

def _fmri_path(fmri_dir: str, subject: str, task: str) -> Path:
    subj_dir = Path(fmri_dir) / subject / "func"
    atlas = "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
    stem = f"{subject}_task-{task}_{atlas}"
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
    """Load fMRI for one clip.  Returns (n_trs, n_voxels) float32."""
    path = _fmri_path(fmri_dir, subject, task)
    if not path.exists():
        raise FileNotFoundError(f"fMRI not found: {path}")

    with h5py.File(path, "r") as f:
        matched = [k for k in f.keys() if clip_key in k]
        if len(matched) != 1:
            raise ValueError(f"Expected 1 match for '{clip_key}' in {path}, got {matched}")
        data = f[matched[0]][:].astype(np.float32)  # (n_trs, n_voxels)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if standardize:
        mu = data.mean(axis=0, keepdims=True)
        sd = data.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        data = (data - mu) / sd
        data = np.nan_to_num(data, nan=0.0)

    return data   # (n_trs, 1000)


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_clip(
    features_root: str,
    model_name: str,
    mod_cfg: dict,
    task: str,
    season: str,
    clip_stem: str,
    layer_agg: str = "mean",
) -> np.ndarray:
    """Load and aggregate all layers for one model/clip.

    Returns: float32 array of shape (n_trs, dim).
    """
    model_path = Path(features_root) / mod_cfg["path"] / task / season
    h5_file = model_path / f"{clip_stem}.h5"
    if not h5_file.exists():
        raise FileNotFoundError(f"Feature file not found: {h5_file}")

    squeeze = mod_cfg.get("squeeze", False)
    selected_layers = mod_cfg.get("layers", None)  # None = all

    with h5py.File(h5_file, "r") as f:
        all_keys = list(f.keys())
        if selected_layers:
            keys = [k for k in all_keys if k in selected_layers]
        else:
            keys = all_keys

        if not keys:
            raise ValueError(f"No matching layers in {h5_file}: {all_keys}")

        arrays = []
        for k in sorted(keys):
            arr = f[k][:].astype(np.float32)    # (T, [1,] dim)
            if squeeze and arr.ndim == 3:
                arr = arr[:, 0, :]               # (T, dim)
            elif arr.ndim == 1:
                arr = arr[np.newaxis]            # edge case
            arrays.append(arr)                   # each (T, dim)

    # Stack: (n_layers, T, dim) then aggregate
    stacked = np.stack(arrays, axis=0)           # (L, T, dim)
    if layer_agg == "mean":
        feat = stacked.mean(axis=0)              # (T, dim)
    elif layer_agg == "last":
        feat = stacked[-1]                       # (T, dim)
    else:
        feat = stacked.mean(axis=0)

    return np.nan_to_num(feat, nan=0.0).astype(np.float32)  # (T, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Clip discovery
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_split(uid: str, val_ratio: float = 0.1) -> str:
    hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
    rng = random.Random(hashed)
    return "val" if rng.random() < val_ratio else "train"


def _fmri_clip_key(task: str, clip_stem: str) -> str:
    """Convert feature clip stem → fMRI h5 key fragment."""
    if task == "friends":
        # friends_s01e01a → s01e01a
        if clip_stem.startswith("friends_"):
            return clip_stem[len("friends_"):]
        return clip_stem
    else:
        # movie10_bourne01 → bourne01
        if clip_stem.startswith("movie10_"):
            return clip_stem[len("movie10_"):]
        return clip_stem

def _build_subject_fmri_keys(fmri_dir: str, subjects: list, task: str) -> dict:
    """Pre-read fMRI H5 top-level keys per subject (fast, one open per file)."""
    result = {}
    for subj in subjects:
        fp = _fmri_path(fmri_dir, subj, task)
        if not fp.exists():
            result[subj] = set()
            continue
        try:
            with h5py.File(fp, "r") as f:
                result[subj] = set(f.keys())
        except Exception:
            result[subj] = set()
    return result


def build_clip_index(cfg: dict) -> List[dict]:
    """Enumerate available (subject, clip) pairs, validated against fMRI H5 keys."""
    features_root = cfg["_features_root"]
    fmri_dir = cfg["_fmri_dir"]
    subjects = cfg["subjects"]
    val_ratio = cfg.get("val_ratio", 0.1)

    modalities = cfg["features"]["modalities"]
    first_mod_cfg = modalities[next(iter(modalities))]

    # Pre-read fMRI H5 top-level keys once per (subject, task)
    fmri_keys: dict = {}   # (subject, task) → set of h5 top-level keys

    index = []
    splits_cfg = cfg["splits"]

    for task, task_splits in splits_cfg.items():
        # One H5 open per subject×task
        for subj in subjects:
            if (subj, task) not in fmri_keys:
                fmri_keys[(subj, task)] = _build_subject_fmri_keys(
                    fmri_dir, [subj], task
                ).get(subj, set())

        for split_name, seasons in task_splits.items():
            for season in seasons:
                feat_dir = (
                    Path(features_root)
                    / first_mod_cfg["path"]
                    / task
                    / season
                )
                if not feat_dir.exists():
                    logger.warning("Feature dir missing: %s", feat_dir)
                    continue

                clip_stems = sorted(p.stem for p in feat_dir.glob("*.h5"))
                if not clip_stems:
                    continue

                for clip_stem in clip_stems:
                    fmri_key = _fmri_clip_key(task, clip_stem)
                    clip_split = (
                        "test" if split_name == "test"
                        else _deterministic_split(f"{task}/{season}/{clip_stem}", val_ratio)
                    )

                    for subject in subjects:
                        subj_h5_keys = fmri_keys.get((subject, task), set())
                        # Check that this subject's H5 actually contains this clip
                        if not any(fmri_key in k for k in subj_h5_keys):
                            continue
                        index.append({
                            "subject": subject,
                            "task": task,
                            "season": season,
                            "clip_stem": clip_stem,
                            "fmri_key": fmri_key,
                            "split": clip_split,
                        })

    logger.info("Built clip index: %d entries (%d subjects)", len(index), len(subjects))
    return index



# ─────────────────────────────────────────────────────────────────────────────
# Main Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AlgonautsDataset(Dataset):
    """Sliding-window dataset using algonauts_2025.features.

    Each sample:
        features[model]: Tensor(context_trs, dim)  — layer-aggregated
        fmri: Tensor(n_voxels,)                     — target TR
        subject_id: int
    """

    def __init__(
        self,
        cfg: dict,
        split: str = "train",
        max_cache_gb: float = 8.0,
    ):
        self.cfg = cfg
        self.split = split
        self.tr = cfg["fmri"]["tr"]
        self.n_voxels = cfg["fmri"]["n_voxels"]
        self.features_root = cfg["_features_root"]
        self.fmri_dir = cfg["_fmri_dir"]
        self.standardize = cfg.get("preprocessing", {}).get("fmri", {}).get(
            "standardize", "zscore_sample"
        ) == "zscore_sample"
        self.layer_agg = cfg["features"].get("layer_agg", "mean")

        sw = cfg["sliding_window"]
        self.context_trs = round(sw["context_duration"] / self.tr)
        self.stride_trs = max(1, round(sw.get("stride", self.tr) / self.tr))

        self.modalities = cfg["features"]["modalities"]
        self.subject_to_id = {s: i for i, s in enumerate(cfg["subjects"])}

        # Build clip & sample index
        clip_index = build_clip_index(cfg)
        self.clip_index = [c for c in clip_index if c["split"] == split]
        logger.info("AlgonautsDataset[%s]: %d clips", split, len(self.clip_index))

        self._samples = self._build_sample_index()
        logger.info(
            "AlgonautsDataset[%s]: %d samples, context=%d TRs",
            split, len(self._samples), self.context_trs,
        )

        self._cache = _LRUCache(max_bytes=int(max_cache_gb * 1e9))
        # RAM preload store
        self._ram: dict = {}

    # ── Sample index ──────────────────────────────────────────────────────────

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build sample index. Reads n_trs from feature files (fast, deduped)."""
        from tqdm import tqdm
        ntrs_cache: dict = {}  # (task, season, clip_stem) → n_trs  [deduped]
        samples = []
        for clip_idx, info in enumerate(tqdm(
            self.clip_index, desc=f"  BuildIndex [{self.split}]", leave=False,
        )):
            key = (info["task"], info["season"], info["clip_stem"])
            if key not in ntrs_cache:
                ntrs_cache[key] = self._get_n_trs_from_features(info)
            n_trs = ntrs_cache[key]
            if n_trs is None:
                continue
            for target_tr in range(self.context_trs, n_trs, self.stride_trs):
                samples.append((clip_idx, target_tr))
        return samples

    def _get_n_trs_from_features(self, info: dict) -> Optional[int]:
        """Peek at first feature H5 to get T — faster than opening fMRI H5."""
        first_mod = next(iter(self.modalities))
        mod_cfg = self.modalities[first_mod]
        feat_path = (
            Path(self.features_root)
            / mod_cfg["path"]
            / info["task"]
            / info["season"]
            / f"{info['clip_stem']}.h5"
        )
        try:
            with h5py.File(feat_path, "r") as f:
                first_key = next(iter(f.keys()))
                return f[first_key].shape[0]     # T (first dim)
        except Exception as e:
            logger.debug("Cannot get n_trs for %s: %s", info["clip_stem"], e)
            return None

    # ── Data loading with cache ────────────────────────────────────────────────

    def _load_fmri(self, clip_idx: int) -> np.ndarray:
        key = ("fmri", clip_idx)
        if key in self._ram:
            return self._ram[key]
        cached = self._cache.get(key)
        if cached is None:
            info = self.clip_index[clip_idx]
            cached = load_fmri_clip(
                self.fmri_dir, info["subject"], info["task"],
                info["fmri_key"], standardize=self.standardize,
            )
            self._cache.put(key, cached)
        return cached

    def _load_feature(self, clip_idx: int, model_name: str) -> np.ndarray:
        key = ("feat", clip_idx, model_name)
        if key in self._ram:
            return self._ram[key]
        cached = self._cache.get(key)
        if cached is None:
            info = self.clip_index[clip_idx]
            mod_cfg = self.modalities[model_name]
            cached = load_feature_clip(
                self.features_root, model_name, mod_cfg,
                info["task"], info["season"], info["clip_stem"],
                layer_agg=self.layer_agg,
            )
            self._cache.put(key, cached)
        return cached

    # ── Preload to RAM (optional) ─────────────────────────────────────────────

    def preload_to_ram(self) -> None:
        from tqdm import tqdm
        clip_ids = sorted({ci for ci, _ in self._samples})
        total = 0
        # Dedup features by (task, season, clip_stem, model) across subjects
        feat_dedup: dict = {}

        logger.info("Preloading %d clips × %d modalities + fMRI...",
                    len(clip_ids), len(self.modalities))
        for ci in tqdm(clip_ids, desc=f"Preload [{self.split}]"):
            info = self.clip_index[ci]

            # fMRI (per subject, no dedup)
            fk = ("fmri", ci)
            if fk not in self._ram:
                try:
                    arr = load_fmri_clip(
                        self.fmri_dir, info["subject"], info["task"],
                        info["fmri_key"], standardize=self.standardize,
                    )
                    self._ram[fk] = arr
                    total += arr.nbytes
                except Exception as e:
                    logger.warning("fMRI load fail clip %d: %s", ci, e)

            # Features (dedup across subjects)
            dedup_base = (info["task"], info["season"], info["clip_stem"])
            for mod in self.modalities:
                fk2 = ("feat", ci, mod)
                dedup_key = (*dedup_base, mod)
                if dedup_key in feat_dedup:
                    self._ram[fk2] = feat_dedup[dedup_key]
                else:
                    try:
                        arr = load_feature_clip(
                            self.features_root, mod,
                            self.modalities[mod],
                            info["task"], info["season"], info["clip_stem"],
                            layer_agg=self.layer_agg,
                        )
                        feat_dedup[dedup_key] = arr
                        self._ram[fk2] = arr
                        total += arr.nbytes
                    except Exception as e:
                        logger.warning("Feature load fail %s clip %d: %s", mod, ci, e)

        self._cache = _LRUCache(max_bytes=0)
        logger.info("Preloaded %.2f GB to RAM", total / 1e9)

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_idx, target_tr = self._samples[idx]
        info = self.clip_index[clip_idx]

        # fMRI target: shape (1000,)
        fmri_clip = self._load_fmri(clip_idx)           # (n_trs_fmri, 1000)
        n_trs_fmri = fmri_clip.shape[0]

        # Guard: feature files may have more TRs than fMRI — clamp to fMRI bounds
        target_tr = min(target_tr, n_trs_fmri - 1)
        ctx_start = target_tr - self.context_trs
        ctx_end = target_tr  # exclusive

        fmri_target = torch.from_numpy(fmri_clip[target_tr].copy()).float()

        # Features: {model: (context_trs, dim)}
        features = {}
        for mod_name, mod_cfg in self.modalities.items():
            expected_dim = mod_cfg["dim"]
            try:
                feat = self._load_feature(clip_idx, mod_name)  # (T, dim)
                T_feat = feat.shape[0]
                # Align context window
                s = max(0, ctx_start)
                e = min(T_feat, ctx_end)
                chunk = feat[s:e]                               # (some, dim)
                # Pad to context_trs if needed
                if chunk.shape[0] < self.context_trs:
                    pad = np.zeros(
                        (self.context_trs - chunk.shape[0], chunk.shape[1]),
                        dtype=np.float32
                    )
                    chunk = np.concatenate([pad, chunk], axis=0)
                elif chunk.shape[0] > self.context_trs:
                    chunk = chunk[-self.context_trs:]
            except Exception:
                chunk = np.zeros((self.context_trs, expected_dim), dtype=np.float32)

            features[mod_name] = torch.from_numpy(chunk.copy()).float()

        subject_id = torch.tensor(
            self.subject_to_id.get(info["subject"], 0), dtype=torch.long
        )

        return {
            "features": features,
            "fmri": fmri_target,
            "subject_id": subject_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: dict,
    project_root: str | Path,
    splits: List[str] = ("train", "val"),
) -> Dict[str, DataLoader]:
    cfg = resolve_paths(cfg, project_root)
    dl_cfg = cfg.get("dataloader", {})
    loaders = {}

    for split in splits:
        ds = AlgonautsDataset(
            cfg,
            split=split,
            max_cache_gb=dl_cfg.get("max_cache_gb", 8.0),
        )
        if not ds._samples:
            logger.warning("Empty dataset for split '%s'", split)
            continue

        if cfg.get("training", {}).get("preload_to_ram", False):
            ds.preload_to_ram()

        batch_size = (
            dl_cfg.get("val_batch_size", 128)
            if split == "val"
            else dl_cfg.get("batch_size", 256)
        )
        n_workers = dl_cfg.get("num_workers", 0)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=n_workers,
            pin_memory=dl_cfg.get("pin_memory", False) and n_workers > 0,
            drop_last=(split == "train"),
            persistent_workers=n_workers > 0,
        )
        logger.info(
            "DataLoader[%s]: %d samples → %d batches (bs=%d)",
            split, len(ds), len(loaders[split]), batch_size,
        )

    return loaders
