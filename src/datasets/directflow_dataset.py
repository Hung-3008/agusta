import logging
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

import yaml

def load_config(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _get_fmri_filepath(fmri_dir: str, subject: str, task: str) -> Path:
    func_dir = Path(fmri_dir) / subject / "func"
    matches = list(func_dir.glob(f"*{task}*.h5"))
    if not matches:
        raise FileNotFoundError(f"No h5 file found for task {task} in {func_dir}")
    return matches[0]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("datasets.directflow")

class DirectFlowDataset(Dataset):
    """Pre-extracted context latents + raw fMRI, fully preloaded into RAM.

    Seq2seq: ``(context_trs, D)`` -> ``(n_target_trs, V)`` fMRI window.
    Legacy single-TR: ``(feat_seq_len, D)`` -> ``(V,)``.
    """

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.subjects = cfg["subjects"]
        self.splits_cfg = cfg.get("splits", {})

        self.fmri_dir = Path(cfg["_fmri_dir"])

        if "context_latent_dirs" in cfg:
            self.context_dirs = [Path(PROJECT_ROOT) / d for d in cfg["context_latent_dirs"]]
        else:
            self.context_dirs = [Path(PROJECT_ROOT) / cfg["context_latent_dir"]]

        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end = cfg["fmri"].get("excluded_samples_end", 0)

        sw = cfg["sliding_window"]
        self.stride = sw.get("stride", 1)
        self.hrf_delay = cfg["fmri"].get("hrf_delay", 5)
        self.temporal_jitter = sw.get("temporal_jitter", 0) if split == "train" else 0

        # --- Seq2seq vs legacy single-TR mode ---
        if "n_target_trs" in sw:
            # Seq2seq: encoder sees context_trs TRs, decoder predicts n_target_trs fMRI
            self.n_target_trs = sw["n_target_trs"]
            self.context_trs = sw["context_trs"]
            # Split extra context evenly before/after the target feature window
            _extra = self.context_trs - self.n_target_trs
            self.context_extra_past = _extra // 2
            self.context_extra_future = _extra - self.context_extra_past
            self.feat_seq_len = self.context_trs   # encoder input length
        else:
            # Legacy: one fMRI per sample
            self.n_target_trs = 1
            self.feature_past_trs = sw.get("feature_past_trs",
                                    sw.get("feature_context_trs", 10))
            self.feature_future_trs = sw.get("feature_future_trs", 0)
            self.feat_seq_len = self.feature_past_trs + 1 + self.feature_future_trs
            self.context_trs = self.feat_seq_len
            self.context_extra_past = None
            self.context_extra_future = None

        self.subject_to_idx = {s: i for i, s in enumerate(self.subjects)}

        self.use_global_stats = cfg["fmri"].get("use_global_stats", False)
        self.fmri_stats = {}
        if self.use_global_stats:
            for subj in self.subjects:
                stats_dir = self.fmri_dir / subj / "stats"
                mean_path = stats_dir / "global_mean.npy"
                std_path = stats_dir / "global_std.npy"
                if mean_path.exists() and std_path.exists():
                    fmri_mean = np.load(mean_path).astype(np.float32)
                    fmri_std = np.load(std_path).astype(np.float32)
                    self.fmri_stats[subj] = {"mean": fmri_mean, "std": fmri_std}
                    logger.info("Loaded global fMRI stats for %s (mean=%.4f, std=%.4f)",
                                subj, fmri_mean.mean(), fmri_std.mean())
                else:
                    logger.warning("Global stats not found at %s", stats_dir)

        self._ctx_clips = {}
        self._fmri_clips = {}
        self.samples = self._build_index_and_preload()

    def _normalize_clip_name(self, clip_name: str, task: str) -> str:
        prefix = f"{task}_"
        return clip_name[len(prefix):] if clip_name.startswith(prefix) else clip_name

    def _load_fmri_clip(self, subject, task, clip_name):
        """Load fMRI from H5 (called only during init)."""
        fmri_path = _get_fmri_filepath(str(self.fmri_dir), subject, task)
        fmri_key = clip_name
        if task == "friends" and clip_name.startswith("friends_"):
            fmri_key = clip_name[len("friends_"):]
        elif task == "movie10" and clip_name.startswith("movie10_"):
            fmri_key = clip_name[len("movie10_"):]

        with h5py.File(fmri_path, "r") as f:
            matched = [k for k in f.keys() if fmri_key in k]
            if not matched:
                return None
            raw = f[matched[0]]
            end = len(raw) - self.excl_end if self.excl_end > 0 else len(raw)
            data = raw[self.excl_start:end].astype(np.float32)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if self.use_global_stats and subject in self.fmri_stats:
            stats = self.fmri_stats[subject]
            data = (data - stats["mean"][None, :]) / stats["std"][None, :]

        return data

    _modality_dim_cache = {}

    def _get_modality_dim(self, ctx_dir):
        """Detect feature dimension for a modality by loading one sample file."""
        key = str(ctx_dir)
        if key in self._modality_dim_cache:
            return self._modality_dim_cache[key]
        for f in ctx_dir.rglob("*.npy"):
            dim = np.load(f).shape[-1]
            self._modality_dim_cache[key] = dim
            return dim
        raise ValueError(f"No .npy files found in {ctx_dir}")

    def _build_index_and_preload(self):
        """Build sample index and preload all data into RAM."""
        samples = []
        ctx_bytes, fmri_bytes = 0, 0
        n_ctx_clips, n_fmri_clips = 0, 0
        logger.info("Building %s split index and preloading data...", self.split)

        for task, splits in self.splits_cfg.items():
            stim_types = splits.get(self.split, [])
            if not stim_types:
                continue

            for stim_type in stim_types:
                ref_ctx_dir = self.context_dirs[0] / task / stim_type
                if not ref_ctx_dir.exists():
                    logger.warning("Context dir missing: %s", ref_ctx_dir)
                    continue

                clip_stems = sorted(p.stem for p in ref_ctx_dir.glob("*.npy"))

                for clip_stem in clip_stems:
                    norm_name = self._normalize_clip_name(clip_stem, task)

                    ctx_key = f"{task}/{stim_type}/{norm_name}"
                    if ctx_key not in self._ctx_clips:
                        ctx_arrays = []
                        for ctx_dir in self.context_dirs:
                            c_dir = ctx_dir / task / stim_type
                            ctx_path = c_dir / f"{clip_stem}.npy"
                            if not ctx_path.exists():
                                ctx_path = c_dir / f"{norm_name}.npy"
                            if not ctx_path.exists():
                                ctx_path = c_dir / f"{task}_{norm_name}.npy"
                            if ctx_path.exists():
                                arr = np.load(ctx_path).astype(np.float32)
                                # Flatten 3-D features (T, N, D) → (T, N*D)
                                if arr.ndim == 3:
                                    arr = arr.reshape(arr.shape[0], -1)
                                ctx_arrays.append(arr)
                            else:
                                mod_dim = self._get_modality_dim(ctx_dir)
                                ref_len = ctx_arrays[0].shape[0] if ctx_arrays else 500
                                ctx_arrays.append(np.zeros((ref_len, mod_dim), dtype=np.float32))

                        min_len = min(arr.shape[0] for arr in ctx_arrays)
                        ctx_arrays = [arr[:min_len] for arr in ctx_arrays]

                        ctx_data = torch.from_numpy(
                            np.concatenate(ctx_arrays, axis=-1).astype(np.float32)
                        )
                        self._ctx_clips[ctx_key] = ctx_data
                        ctx_bytes += ctx_data.nelement() * 4
                        n_ctx_clips += 1

                    for subj in self.subjects:
                        subj_idx = self.subject_to_idx[subj]
                        fmri_key = f"{subj}/{task}/{norm_name}"

                        if fmri_key not in self._fmri_clips:
                            fmri_data = self._load_fmri_clip(subj, task, norm_name)
                            if fmri_data is None:
                                fmri_data = self._load_fmri_clip(subj, task, clip_stem)
                            if fmri_data is None:
                                self._fmri_clips[fmri_key] = None
                            else:
                                fmri_t = torch.from_numpy(fmri_data)
                                self._fmri_clips[fmri_key] = fmri_t
                                fmri_bytes += fmri_t.nelement() * 4
                                n_fmri_clips += 1

                        fmri_data = self._fmri_clips[fmri_key]
                        if fmri_data is None:
                            continue

                        n_trs = fmri_data.shape[0]
                        # Seq2seq: ensure each sample has a full n_target_trs window
                        if self.n_target_trs > 1:
                            n_valid = max(0, n_trs - self.n_target_trs + 1)
                        else:
                            n_valid = n_trs
                        for target_tr in range(0, n_valid, self.stride):
                            samples.append({
                                "ctx_key": ctx_key,
                                "fmri_key": fmri_key,
                                "task": task,
                                "stim_type": stim_type,
                                "clip_stem": clip_stem,
                                "norm_name": norm_name,
                                "subject": subj,
                                "subject_idx": subj_idx,
                                "target_tr": target_tr,
                                "n_trs": n_trs,
                            })

        total_bytes = ctx_bytes + fmri_bytes
        logger.info("Preloaded %d ctx clips (%.2f GB) + %d fMRI clips (%.2f GB) "
                     "= %.2f GB total -> %d samples (%s).",
                     n_ctx_clips, ctx_bytes / 1e9,
                     n_fmri_clips, fmri_bytes / 1e9,
                     total_bytes / 1e9, len(samples), self.split)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        ctx_data = self._ctx_clips[info["ctx_key"]]    # (T_ctx_full, D)
        fmri_data = self._fmri_clips[info["fmri_key"]]  # (N_trs, V)
        target_start = info["target_tr"]

        # Feature TR for the FIRST target fMRI (accounts for excl_start and HRF)
        actual_start_tr = target_start + self.excl_start
        feat_first = actual_start_tr - self.hrf_delay
        if self.temporal_jitter > 0:
            feat_first += random.randint(-self.temporal_jitter, self.temporal_jitter)

        if self.n_target_trs > 1:
            # ========== SEQ2SEQ MODE ==========
            # Context window: context_extra_past TRs before feat_first, then context_trs total
            ctx_start = feat_first - self.context_extra_past
            ctx_end = ctx_start + self.context_trs
            safe_start = max(0, ctx_start)
            safe_end = min(ctx_data.shape[0], ctx_end)
            if safe_start < safe_end:
                ctx = ctx_data[safe_start:safe_end]
            else:
                ctx = ctx_data.new_zeros(0, ctx_data.shape[-1])
            # Pad to exact context_trs length
            if ctx.shape[0] < self.context_trs:
                pad_before = max(0, -ctx_start)
                pad_after = max(0, self.context_trs - ctx.shape[0] - pad_before)
                ctx = F.pad(ctx, (0, 0, pad_before, pad_after))
            ctx = ctx[:self.context_trs]
            if ctx.shape[0] != self.context_trs:
                raise RuntimeError(
                    f"Seq2seq context length mismatch: got {ctx.shape[0]}, expected {self.context_trs}"
                )

            # Target fMRI: (n_target_trs, V)
            fmri_end = min(target_start + self.n_target_trs, fmri_data.shape[0])
            fmri_seq = fmri_data[target_start:fmri_end].clone()
            if fmri_seq.shape[0] < self.n_target_trs:
                pad_len = self.n_target_trs - fmri_seq.shape[0]
                fmri_seq = torch.cat([
                    fmri_seq,
                    fmri_seq.new_zeros(pad_len, fmri_seq.shape[1]),
                ], dim=0)
            if fmri_seq.shape[0] != self.n_target_trs:
                raise RuntimeError(
                    f"Seq2seq fMRI length mismatch: got {fmri_seq.shape[0]}, expected {self.n_target_trs}"
                )

            return {
                "context": ctx,                                        # (101, D)
                "fmri": fmri_seq,                                      # (50, V)
                "clip_key": f"{info['subject']}/{info['ctx_key']}",
                "subject_idx": info["subject_idx"],
                "target_tr_start": target_start,
                "n_trs": info["n_trs"],
            }
        else:
            # ========== LEGACY SINGLE-TR MODE ==========
            feat_start = feat_first - self.feature_past_trs
            feat_end = feat_first + 1 + self.feature_future_trs
            safe_start = max(0, feat_start)
            safe_end = min(ctx_data.shape[0], feat_end)
            ctx = ctx_data[safe_start:safe_end] if safe_start < safe_end else torch.zeros(0, ctx_data.shape[-1])
            if ctx.shape[0] < self.feat_seq_len:
                pad_before = max(0, -feat_start)
                pad_after = self.feat_seq_len - ctx.shape[0] - pad_before
                pad_after = max(0, pad_after)
                ctx = F.pad(ctx, (0, 0, pad_before, pad_after))

            if target_start < fmri_data.shape[0]:
                fmri_target = fmri_data[target_start].clone()
            else:
                fmri_target = torch.zeros(self.cfg["fmri"].get("n_voxels", 1000), dtype=torch.float32)

            return {
                "context": ctx,                                        # (31, D)
                "fmri": fmri_target,                                   # (V,)
                "clip_key": f"{info['subject']}/{info['ctx_key']}",
                "subject_idx": info["subject_idx"],
                "target_tr_start": target_start,
                "n_trs": info["n_trs"],
            }


class ClipGroupedBatchSampler(Sampler):
    """Groups windows from the same clip into the same batch."""

    def __init__(self, dataset, batch_size, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.clip_groups = defaultdict(list)
        for idx, info in enumerate(dataset.samples):
            clip_key = (info["task"], info["stim_type"], info["clip_stem"])
            self.clip_groups[clip_key].append(idx)

        self.clip_keys = list(self.clip_groups.keys())

    def __iter__(self):
        clip_order = list(self.clip_keys)
        random.shuffle(clip_order)

        all_indices = []
        for clip_key in clip_order:
            indices = self.clip_groups[clip_key]
            random.shuffle(indices)
            all_indices.extend(indices)

        batch = []
        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = sum(len(v) for v in self.clip_groups.values())
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def get_dataloaders(cfg):
    train_set = DirectFlowDataset(cfg, split="train")
    val_set = DirectFlowDataset(cfg, split="val")

    dl_cfg = cfg["dataloader"]
    batch_size = dl_cfg["batch_size"]
    num_workers = int(dl_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_set,
        batch_sampler=ClipGroupedBatchSampler(train_set, batch_size, drop_last=True),
        num_workers=num_workers,
        pin_memory=dl_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=dl_cfg["val_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False,
    )
    return train_loader, val_loader
