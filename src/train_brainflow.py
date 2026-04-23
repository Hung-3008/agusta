"""Train BrainFlow — Flow Matching with Pre-extracted Context.

Usage:
    python src/train_brainflow.py --config src/configs/brainflow.yaml --fast_dev_run
    python src/train_brainflow.py --config src/configs/brainflow.yaml
    python src/train_brainflow.py --config src/configs/brainflow.yaml --resume
"""

import argparse
import logging
import math as pymath
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_config, _get_fmri_filepath
from src.models.brainflow.brainflow import BrainFlow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_brainflow")


def resolve_paths(cfg: dict, project_root: Path) -> dict:
    cfg = cfg.copy()
    cfg["_project_root"] = str(project_root)
    cfg["_data_root"] = str(project_root / cfg["data_root"])
    cfg["_fmri_dir"] = str(project_root / cfg["data_root"] / cfg["fmri"]["dir"])
    return cfg


# =============================================================================
# Dataset — Pre-extracted context + raw fMRI (RAM preloaded)
# =============================================================================

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


# =============================================================================
# Batch Sampler
# =============================================================================

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


# =============================================================================
# Metrics & EMA
# =============================================================================

def pearson_corr_per_dim(pred, target):
    """Per-voxel PCC across time. Input: (1, N, D) or (N, D)."""
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    pred_cnt = pred - pred.mean(dim=0, keepdim=True)
    target_cnt = target - target.mean(dim=0, keepdim=True)
    cov = (pred_cnt * target_cnt).sum(dim=0)
    std = torch.sqrt((pred_cnt ** 2).sum(dim=0) * (target_cnt ** 2).sum(dim=0))
    return cov / (std + 1e-8)


class EMAModel:
    """EMA weights. Optional ``store_on_cpu`` frees VRAM (shadow updated on CPU)."""

    def __init__(self, model, decay=0.999, store_on_cpu: bool = False):
        self.decay = decay
        self.store_on_cpu = store_on_cpu
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                sh = param.data.clone()
                self.shadow[name] = sh.cpu() if store_on_cpu else sh

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.store_on_cpu:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data.detach().cpu(), alpha=1 - self.decay
                    )
                else:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                sh = self.shadow[name]
                if self.store_on_cpu:
                    sh = sh.to(device=param.device, dtype=param.dtype, non_blocking=True)
                param.data.copy_(sh)

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def to_cpu_shadow(self):
        """Move shadow tensors to CPU (e.g. after loading an old GPU checkpoint)."""
        if not self.store_on_cpu:
            return
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].cpu()

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = dict(state_dict["shadow"])
        self.decay = state_dict["decay"]
        if self.store_on_cpu:
            for name in self.shadow:
                self.shadow[name] = self.shadow[name].cpu()


# =============================================================================
# Dataloaders
# =============================================================================

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



# =============================================================================
# Training
# =============================================================================

def train(args):
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    if args.train_batch_size is not None:
        cfg.setdefault("dataloader", {})["batch_size"] = int(args.train_batch_size)
    if args.val_batch_size is not None:
        cfg.setdefault("dataloader", {})["val_batch_size"] = int(args.val_batch_size)

    logger.info("Loaded config: %s", cfg_path)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(cfg)
    _dl = cfg["dataloader"]
    _bs, _vbs = _dl["batch_size"], _dl["val_batch_size"]
    logger.info(
        "DataLoader: train batch_size=%d → %d micro-batches/epoch (~%d samples/epoch, drop_last); "
        "val batch_size=%d",
        _bs,
        len(train_loader),
        len(train_loader) * _bs,
        _vbs,
    )

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1
        cfg["training"]["val_every_n_epochs"] = 1

    # --- Model ---
    bf_cfg = cfg["brainflow"]
    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])
    vn_params = dict(bf_cfg.get("velocity_net", {}))
    logger.info("Context encoder mode: %s", vn_params.get("context_encoder", "multitoken"))
    modality_dims = cfg.get("modality_dims", None)

    if modality_dims:
        vn_params["modality_dims"] = modality_dims

    cont_cfg = dict(bf_cfg.get("contrastive", {}))
    cont_weight = float(cont_cfg.get("weight", bf_cfg.get("cont_weight", 0.0)))
    cont_dim = int(bf_cfg.get("cont_dim", cont_cfg.get("dim", 256)))
    csfm_mixflow_cfg = bf_cfg.get("csfm_mixflow", None)
    temporal_ot_cfg = bf_cfg.get("temporal_ot", None)

    model = BrainFlow(
        output_dim=output_dim,
        velocity_net_params=vn_params,
        n_subjects=len(cfg["subjects"]),

        tensor_fm_params=bf_cfg.get("tensor_fm", None),
        indi_flow_matching=bf_cfg.get("indi_flow_matching", False),
        indi_train_time_sqrt=bf_cfg.get("indi_train_time_sqrt", False),
        indi_min_denom=bf_cfg.get("indi_min_denom", 1e-3),
        use_csfm=bf_cfg.get("use_csfm", False),
        csfm_mixflow_params=csfm_mixflow_cfg,
        csfm_var_reg_weight=bf_cfg.get("csfm_var_reg_weight", 0.1),
        csfm_pcc_weight=bf_cfg.get("csfm_pcc_weight", 1.0),
        flow_loss_weight=bf_cfg.get("flow_loss_weight", 1.0),
        cont_weight=cont_weight,
        cont_dim=cont_dim,
        contrastive_params=cont_cfg,
        temporal_ot_params=temporal_ot_cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # --- Optimizer & Scheduler ---
    tr_cfg = cfg["training"]
    accum_steps = max(1, int(tr_cfg.get("gradient_accumulation_steps", 1)))
    opt_steps_per_epoch = max(1, (len(train_loader) + accum_steps - 1) // accum_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"],
    )

    total_steps = 2 if args.fast_dev_run else opt_steps_per_epoch * tr_cfg["n_epochs"]
    warmup_steps = int(total_steps * tr_cfg.get("warmup_ratio", 0.05))
    min_lr = tr_cfg.get("min_lr", 1e-6)
    base_lr = tr_cfg["lr"]

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    # --- Output ---
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_direct")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    ema_on_cpu = tr_cfg.get("ema_on_cpu", True)
    ema = EMAModel(
        model, decay=tr_cfg.get("ema_decay", 0.999), store_on_cpu=ema_on_cpu,
    )
    gckpt = vn_params.get("gradient_checkpointing", False)
    logger.info(
        "Memory opts: grad_accum=%d (~%d opt-steps/epoch), EMA_on_cpu=%s, "
        "velocity_grad_ckpt=%s",
        accum_steps, opt_steps_per_epoch, ema_on_cpu, gckpt,
    )
    logger.info("EMA decay=%.4f", ema.decay)

    # --- Resume / Warmstart ---
    start_epoch, best_val_corr, global_step = 1, -1.0, 0

    if args.warmstart:
        ws_path = Path(args.warmstart)
        if ws_path.exists():
            ckpt = torch.load(ws_path, map_location=device, weights_only=False)
            state = ckpt.get("model", ckpt)
            model_state = model.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_state and model_state[k].shape == v.shape}
            skipped = [k for k in state if k in model_state and model_state[k].shape != state[k].shape]
            missing, _ = model.load_state_dict(filtered, strict=False)
            if skipped:
                logger.info("Warmstart skipped %d mismatched keys", len(skipped))
            logger.info("Warmstarted from %s, loaded %d/%d keys", ws_path, len(filtered), len(state))
            del ckpt
        else:
            logger.warning("--warmstart path %s not found.", ws_path)

    if args.resume:
        resume_path = out_dir / "last.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            logger.info("Resumed from epoch %d (step=%d)", ckpt["epoch"], global_step)
            del ckpt
        else:
            logger.warning("--resume but no last.pt found.")

    history_file = out_dir / "history.csv"
    if start_epoch == 1:
        history_file.write_text("epoch,train_loss,cfm_loss,pcc_loss,geo_loss,val_fmri_pcc,lr\n")

    solver_cfg = cfg.get("solver_args", {})
    val_n_timesteps = solver_cfg.get("time_points", 50)
    val_solver_method = solver_cfg.get("method", "midpoint")
    val_cfg_scale = solver_cfg.get("cfg_scale", 0.0)
    val_temperature = solver_cfg.get("temperature", 0.0)

    context_bf16 = tr_cfg.get("context_bf16_when_amp", True)
    pin = cfg.get("dataloader", {}).get("pin_memory", False)

    # --- Training loop ---
    for epoch in range(start_epoch, tr_cfg["n_epochs"] + 1):
        model.train()
        train_losses = []
        cfm_losses = []
        pcc_losses = []
        geo_losses = []
        micro_accum = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            context = batch["context"].to(device, non_blocking=pin)
            subject_ids = batch["subject_idx"].to(device, non_blocking=pin)
            target = batch["fmri"].to(device, non_blocking=pin)
            if tr_cfg["use_amp"] and context_bf16:
                context = context.to(dtype=torch.bfloat16)
                target = target.to(dtype=torch.bfloat16)

            cfg_drop = random.random() < 0.1
            if cfg_drop:
                context = torch.zeros_like(context)

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                losses = model.compute_loss(
                    context, target,
                    subject_ids=subject_ids,
                    skip_aux=cfg_drop,
                )
                raw_loss = losses["total_loss"]
                loss = raw_loss / accum_steps

            loss.backward()
            train_losses.append(float(raw_loss.detach()))
            if "flow_loss" in losses:
                cfm_losses.append(float(losses["flow_loss"].detach()))
            if "pcc_loss" in losses:
                pcc_losses.append(float(losses["pcc_loss"].detach()))
            if "geo_loss" in losses:
                geo_losses.append(float(losses["geo_loss"].detach()))

            micro_accum += 1
            if micro_accum >= accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                micro_accum = 0
                global_step += 1

                if global_step % tr_cfg["log_every_n_steps"] == 0:
                    postfix = {
                        "loss": f"{np.mean(train_losses[-50:]):.4f}",
                        "flow": f"{losses['flow_loss'].item():.4f}",
                        "pcc": f"{losses['pcc_loss'].item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                    if model.use_temporal_ot:
                        postfix["geo"] = f"{losses['geo_loss'].item():.4f}"
                    if model.use_tensor_fm:
                        postfix["g_reg"] = f"{losses['gamma_reg'].item():.4f}"
                    pbar.set_postfix(postfix)

        if micro_accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            global_step += 1

        if tr_cfg.get("empty_cuda_cache_each_epoch", False) and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Validation ---
        mean_fmri_corr = 0.0
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()
            logger.info("Running validation...")

            fmri_pred_acc, fmri_tgt_acc = {}, {}

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break

                    context = batch["context"].to(device)
                    subject_ids = batch["subject_idx"].to(device)

                    synth_kwargs = dict(
                        n_timesteps=val_n_timesteps,
                        solver_method=val_solver_method,
                        subject_ids=subject_ids,
                        temperature=val_temperature,
                    )
                    tw = solver_cfg.get("time_grid_warp")
                    if tw:
                        synth_kwargs["time_grid_warp"] = tw
                    if val_cfg_scale > 0:
                        synth_kwargs["cfg_scale"] = val_cfg_scale

                    gen_fmri = model.synthesise(context, **synth_kwargs)

                    clip_keys = batch["clip_key"]
                    target_tr_starts = batch["target_tr_start"]
                    n_trs_batch = batch["n_trs"]
                    fmri_target = batch["fmri"].to(device)

                    B = gen_fmri.shape[0]
                    fmri_dim = gen_fmri.shape[-1]
                    is_seq = gen_fmri.dim() == 3
                    for b in range(B):
                        ck = clip_keys[b]
                        tr_start = int(target_tr_starts[b].item())
                        n_t = int(n_trs_batch[b].item())

                        if ck not in fmri_pred_acc:
                            fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                            fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}

                        if is_seq:
                            # Seq2seq: accumulate each TR offset
                            for off in range(gen_fmri.shape[1]):
                                tr_idx = tr_start + off
                                if tr_idx >= n_t:
                                    break
                                fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b, off].cpu()
                                fmri_pred_acc[ck]["count"][tr_idx] += 1
                                fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b, off].cpu()
                                fmri_tgt_acc[ck]["count"][tr_idx] += 1
                        else:
                            # Legacy single-TR
                            tr_idx = tr_start
                            if tr_idx < n_t:
                                fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b].cpu()
                                fmri_pred_acc[ck]["count"][tr_idx] += 1
                                fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b].cpu()
                                fmri_tgt_acc[ck]["count"][tr_idx] += 1

            all_gen, all_tgt = [], []
            for ck in sorted(fmri_pred_acc.keys()):
                count = fmri_pred_acc[ck]["count"]
                valid = count > 0
                if valid.sum() < 2:
                    continue
                all_gen.append(fmri_pred_acc[ck]["sum"][valid] / count[valid].unsqueeze(-1))
                all_tgt.append(fmri_tgt_acc[ck]["sum"][valid] / fmri_tgt_acc[ck]["count"][valid].unsqueeze(-1))

            if all_gen:
                all_gen = torch.cat(all_gen, dim=0)
                all_tgt = torch.cat(all_tgt, dim=0)
                pcc = pearson_corr_per_dim(all_gen.unsqueeze(0), all_tgt.unsqueeze(0))
                mean_fmri_corr = float(pcc.mean().item())

            logger.info("Epoch %d | Val fMRI PCC: %.4f", epoch, mean_fmri_corr)

            if mean_fmri_corr > best_val_corr:
                best_val_corr = mean_fmri_corr
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("Saved new best EMA model (PCC=%.4f)", best_val_corr)

            mean_cfm = np.mean(cfm_losses) if cfm_losses else 0.0
            mean_pcc = np.mean(pcc_losses) if pcc_losses else 0.0
            mean_geo = np.mean(geo_losses) if geo_losses else 0.0
            with open(history_file, "a") as f:
                f.write(f"{epoch},{np.mean(train_losses):.6f},{mean_cfm:.6f},{mean_pcc:.6f},{mean_geo:.6f},{mean_fmri_corr:.6f},"
                        f"{scheduler.get_last_lr()[0]:.2e}\n")

            ema.restore(model)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "global_step": global_step,
        }, out_dir / "last.pt")

    logger.info("Training complete. Best val PCC: %.4f", best_val_corr)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/brainflow.yaml")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override dataloader.batch_size (YAML on disk must be saved; use this if unsure).",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Override dataloader.val_batch_size.",
    )
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmstart", type=str, default=None)
    args = parser.parse_args()
    train(args)
