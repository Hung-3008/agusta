"""Train BrainFlow CFM v2 — TRIBE-style encoder + DiT/U-Net decoder.

Usage:
    python src/train_brain_flow_v2.py --config src/configs/brain_flow_v2.yaml
    python src/train_brain_flow_v2.py --config src/configs/brain_flow_v2.yaml --fast_dev_run
"""

import argparse
import logging
import copy
import math as pymath
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from functools import lru_cache

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.precompute_latents import load_vae

from src.data.dataset import (
    load_config,
    load_feature_clip_perfile,
    resample_features_to_tr,
    _get_fmri_filepath,
)
from src.models.brainflow.brain_flow_v2 import BrainFlowCFMv2
from src.data.precompute_latents import load_vae

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_cfm_v2")


def resolve_paths_v2(cfg: dict, project_root: Path) -> dict:
    """Resolve paths for v2 config (raw_features instead of features)."""
    cfg = cfg.copy()
    cfg["_project_root"] = str(project_root)
    cfg["_data_root"] = str(project_root / cfg["data_root"])
    cfg["_fmri_dir"] = str(project_root / cfg["data_root"] / cfg["fmri"]["dir"])
    return cfg


# =============================================================================
# Dataset — loads raw multimodal features (no PCA)
# =============================================================================

class BrainFlowV2Dataset(Dataset):
    """Loads raw per-modality features and VAE latents.

    Features are loaded from per-clip H5 files (algonauts_2025.features format),
    resampled to TR grid, and returned as a dict of per-modality tensors.
    """

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.subjects = cfg["subjects"]
        self.splits_cfg = cfg.get("splits", {})

        # Directories
        self.latent_dir = Path(PROJECT_ROOT) / cfg["latent_dir"]
        self.features_dir = Path(PROJECT_ROOT) / cfg["raw_features"]["dir"]
        self.layer_aggregation = cfg["raw_features"].get("layer_aggregation", "mean")
        npy_dir = cfg.get("raw_features_npy_dir")
        if npy_dir:
            if self.layer_aggregation == "cat":
                npy_dir = f"{npy_dir}_{self.layer_aggregation}"
            self.features_npy_dir = Path(PROJECT_ROOT) / npy_dir
        else:
            self.features_npy_dir = None
        self.fmri_dir = Path(cfg["_fmri_dir"])

        # fMRI params
        self.tr = cfg["fmri"]["tr"]
        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end = cfg["fmri"].get("excluded_samples_end", 0)

        # Modality config
        self.modalities = cfg["raw_features"]["modalities"]
        self.layer_aggregation = cfg["raw_features"].get("layer_aggregation", "mean")
        self.feature_freq = cfg["raw_features"].get("sample_freq", 2.0)

        # Sliding window
        self.context_dur = cfg["sliding_window"]["context_duration"]
        self.stride_dur = cfg["sliding_window"]["stride"]
        self.seq_len = max(1, int(round(self.context_dur / self.tr)))
        self.stride = max(1, int(round(self.stride_dur / self.tr)))

        # Causal feature context: extra past TRs for feature input
        self.feature_context_trs = cfg["sliding_window"].get("feature_context_trs", 0)
        self.feat_seq_len = self.seq_len + self.feature_context_trs

        # HRF delay: features at TR t-delay correspond to fMRI at TR t
        self.hrf_delay = cfg["fmri"].get("hrf_delay", 3)

        # Temporal jitter: random ±N TR offset on feature windows (train only)
        self.temporal_jitter = cfg["sliding_window"].get("temporal_jitter", 0) if split == "train" else 0

        self.samples = self._build_index()

        # Cache for opened H5 files (val only)
        self._fmri_cache = {}

        # In-memory LRU cache for NPY feature clips
        # Key: (mod_name, task, stim_type, norm_name) → np.ndarray
        self._feat_cache = {}

    def _normalize_clip_name(self, clip_name: str, task: str) -> str:
        """Strip task prefix from clip name."""
        prefix = f"{task}_"
        if clip_name.startswith(prefix):
            return clip_name[len(prefix):]
        return clip_name

    def _build_index(self):
        samples = []
        logger.info("Building %s split index (v2, raw features)...", self.split)

        for task, splits in self.splits_cfg.items():
            # Use config-defined split: "train" or "val"
            stim_types = splits.get(self.split, [])
            if not stim_types:
                continue

            for stim_type in stim_types:
                # Discover clips from first modality's feature directory
                first_mod = next(iter(self.modalities))
                mod_cfg = self.modalities[first_mod]
                clip_dir = self.features_dir / mod_cfg["subdir"] / task / stim_type

                if not clip_dir.exists():
                    logger.warning("Feature dir missing: %s", clip_dir)
                    continue

                clip_stems = sorted(p.stem for p in clip_dir.glob("*.h5"))

                for clip_stem in clip_stems:
                    # Check latent exists for first subject
                    subj = self.subjects[0]
                    norm_name = self._normalize_clip_name(clip_stem, task)
                    lat_path = self.latent_dir / subj / task / stim_type / f"{norm_name}.npy"

                    # Try with task prefix too
                    if not lat_path.exists():
                        lat_path = self.latent_dir / subj / task / stim_type / f"{clip_stem}.npy"
                    if not lat_path.exists():
                        continue

                    lat_shape = np.load(lat_path, mmap_mode="r").shape
                    n_trs = lat_shape[0]
                    if n_trs < 10:
                        continue

                    # Sliding windows (start from hrf_delay so features can look back)
                    for start_idx in range(self.hrf_delay, n_trs, self.stride):
                        actual_len = min(self.seq_len, n_trs - start_idx)
                        if actual_len < 10:
                            break
                        samples.append({
                            "task": task,
                            "stim_type": stim_type,
                            "clip_stem": clip_stem,
                            "norm_name": norm_name,
                            "start_idx": start_idx,
                            "actual_len": actual_len,
                            "n_trs": n_trs,
                        })

        logger.info("Found %d windows for %s split.", len(samples), self.split)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_fmri_clip(self, subject, task, clip_name):
        """Load ground truth fMRI for a clip (cached)."""
        cache_key = (subject, task, clip_name)
        if cache_key in self._fmri_cache:
            return self._fmri_cache[cache_key]

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
        self._fmri_cache[cache_key] = data
        return data

    def __getitem__(self, idx):
        info = self.samples[idx]
        task = info["task"]
        stim_type = info["stim_type"]
        clip_stem = info["clip_stem"]
        norm_name = info["norm_name"]
        start = info["start_idx"]
        actual_len = info["actual_len"]
        n_trs = info["n_trs"]
        end = start + actual_len

        # HRF delay: features are shifted back by hrf_delay TRs
        # Feature at TR t-delay → fMRI/latent at TR t
        feat_start = start - self.hrf_delay
        feat_end = end - self.hrf_delay

        # Temporal jitter: random ±N TR offset (train only)
        if self.temporal_jitter > 0:
            jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
            feat_start = max(0, feat_start + jitter)
            feat_end = feat_start + actual_len

        # Extended causal context: load extra past TRs of features
        context_start = max(0, feat_start - self.feature_context_trs)
        target_feat_len = actual_len + self.feature_context_trs

        # Load per-modality features
        mod_features = {}
        for mod_name, mod_cfg in self.modalities.items():
            try:
                feat = None
                # Try NPY first (fast path) with in-memory cache
                if self.features_npy_dir is not None:
                    cache_key = (mod_name, task, stim_type, norm_name)
                    if cache_key in self._feat_cache:
                        clip_data = self._feat_cache[cache_key]
                    else:
                        npy_path = self.features_npy_dir / mod_name / task / stim_type / f"{norm_name}.npy"
                        if npy_path.exists():
                            clip_data = np.load(npy_path)  # Load fully into RAM
                            self._feat_cache[cache_key] = clip_data
                        else:
                            clip_data = None
                    if clip_data is not None:
                        feat = clip_data[context_start:feat_end].astype(np.float32)

                # Fallback to H5 (slow path)
                if feat is None:
                    raw = load_feature_clip_perfile(
                        str(self.features_dir), mod_name, mod_cfg,
                        task, stim_type, clip_stem, self.layer_aggregation,
                    )
                    raw = resample_features_to_tr(raw, self.feature_freq, self.tr, n_trs)
                    feat = raw[0].T[context_start:feat_end].astype(np.float32)

                # Left-pad with zeros if not enough past context
                if feat.shape[0] < target_feat_len:
                    pad_len = target_feat_len - feat.shape[0]
                    feat = np.pad(feat, ((pad_len, 0), (0, 0)), mode="constant")

                # Right-pad if short clip at end
                if feat.shape[0] < self.feat_seq_len:
                    pad = self.feat_seq_len - feat.shape[0]
                    feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant")

                mod_features[mod_name] = torch.from_numpy(feat.copy())
            except (FileNotFoundError, KeyError, ValueError) as e:
                dim = mod_cfg.get("dim", 1024)
                mod_features[mod_name] = torch.zeros(self.feat_seq_len, dim, dtype=torch.float32)

        # Load latent (mu) and optionally logvar for posterior sampling
        subj = self.subjects[0]
        lat_path = self.latent_dir / subj / task / stim_type / f"{norm_name}.npy"
        if not lat_path.exists():
            lat_path = self.latent_dir / subj / task / stim_type / f"{clip_stem}.npy"
        lat_data = np.load(lat_path, mmap_mode="r")[start:end].astype(np.float32)

        if actual_len < self.seq_len:
            pad = self.seq_len - actual_len
            lat_data = np.pad(lat_data, ((0, pad), (0, 0)), mode="constant")

        latent = torch.from_numpy(lat_data.copy())

        # Online posterior sampling: z = μ + σε (train only)
        if self.split == "train":
            logvar_name = f"{norm_name}_logvar.npy"
            logvar_path = lat_path.parent / logvar_name
            if not logvar_path.exists():
                logvar_name = f"{clip_stem}_logvar.npy"
                logvar_path = lat_path.parent / logvar_name
            if logvar_path.exists():
                logvar_data = np.load(logvar_path, mmap_mode="r")[start:end].astype(np.float32)
                if actual_len < self.seq_len:
                    logvar_data = np.pad(logvar_data, ((0, pad), (0, 0)), mode="constant")
                logvar = torch.from_numpy(logvar_data.copy())
                # Reparameterize: z = μ + exp(0.5 * logvar) * ε
                latent = latent + torch.exp(0.5 * logvar) * torch.randn_like(latent)

        result = {
            "latent": latent,
            "valid_len": actual_len,
            "clip_key": f"{task}/{stim_type}/{norm_name}",
            "start_idx": start,
            "n_trs": n_trs,
        }
        result.update(mod_features)  # mod_name → (T, D_mod)

        # Load ground truth fMRI (for regression head + val evaluation)
        fmri_data = self._load_fmri_clip(subj, task, norm_name)
        if fmri_data is not None:
            fmri_window = fmri_data[start:end].astype(np.float32)
            if actual_len < self.seq_len:
                pad = self.seq_len - actual_len
                fmri_window = np.pad(fmri_window, ((0, pad), (0, 0)), mode="constant")
            result["fmri"] = torch.from_numpy(fmri_window.copy())

        return result


class ClipGroupedBatchSampler(Sampler):
    """Groups windows from the same clip into the same batch.

    On HDD, random access across many files is the bottleneck.
    By grouping windows from the same clip, each NPY file is
    loaded once per batch → sequential reads instead of random seeks.
    Clips are shuffled each epoch, but windows within a clip are sequential.
    """

    def __init__(self, dataset, batch_size, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group sample indices by clip
        self.clip_groups = defaultdict(list)
        for idx, info in enumerate(dataset.samples):
            clip_key = (info["task"], info["stim_type"], info["clip_stem"])
            self.clip_groups[clip_key].append(idx)

        self.clip_keys = list(self.clip_groups.keys())

    def __iter__(self):
        # Shuffle clip order each epoch
        clip_order = list(self.clip_keys)
        random.shuffle(clip_order)

        # Collect all indices in clip-grouped order
        all_indices = []
        for clip_key in clip_order:
            indices = self.clip_groups[clip_key]
            random.shuffle(indices)  # Shuffle within clip
            all_indices.extend(indices)

        # Yield batches
        batch = []
        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(sum(self.clip_groups.values(), []))
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def get_dataloaders(cfg):
    train_set = BrainFlowV2Dataset(cfg, split="train")
    val_set = BrainFlowV2Dataset(cfg, split="val")

    dl_cfg = cfg["dataloader"]
    batch_size = dl_cfg["batch_size"]

    # Use clip-grouped sampler to minimize HDD random seeks
    train_sampler = ClipGroupedBatchSampler(train_set, batch_size, drop_last=True)

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=0,  # Single process for in-memory cache efficiency
        pin_memory=dl_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=dl_cfg["val_batch_size"],
        shuffle=False,
        num_workers=0,  # Single process for in-memory cache efficiency
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False,
    )
    return train_loader, val_loader


def pearson_corr_per_dim(pred, target, valid_lens=None):
    """Compute Pearson correlation across time for each feature/voxel.

    Matches baseline: per-voxel/per-dim PCC, NOT averaged.

    Args:
        pred, target: (B, T, D)
        valid_lens: optional LongTensor (B,) — number of valid (non-padded) TRs per sample.
    Returns:
        corr: (D,) — per-dimension Pearson correlation.
    """
    if valid_lens is not None:
        rows_p, rows_t = [], []
        for b in range(pred.shape[0]):
            vl = int(valid_lens[b].item())
            rows_p.append(pred[b, :vl, :])
            rows_t.append(target[b, :vl, :])
        pred = torch.cat(rows_p, dim=0)
        target = torch.cat(rows_t, dim=0)
    else:
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])

    pred_mean = pred.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    pred_cnt = pred - pred_mean
    target_cnt = target - target_mean    # Covariance and Variance
    cov = (pred_cnt * target_cnt).sum(dim=0)
    std = torch.sqrt((pred_cnt**2).sum(dim=0) * (target_cnt**2).sum(dim=0))

    corr = cov / (std + 1e-8)
    return corr  # (D,) — per-dimension, NOT averaged


def pearson_corr(pred, target, valid_lens=None):
    """Backward-compatible: returns scalar mean PCC."""
    return pearson_corr_per_dim(pred, target, valid_lens).mean().item()


def _extract_modality_features(batch, modality_names, device):
    """Extract modality feature tensors from batch dict."""
    mod_features = {}
    for mod_name in modality_names:
        if mod_name in batch:
            mod_features[mod_name] = batch[mod_name].to(device)
    return mod_features


# =============================================================================
# EMA Model
# =============================================================================

class EMAModel:
    """Exponential Moving Average of model weights for stable inference.

    Maintains a shadow copy of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * current
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Replace model params with EMA params (for eval)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original params (after eval)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


def train(args):
    cfg = load_config(args.config)
    cfg = resolve_paths_v2(cfg, PROJECT_ROOT)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init Data
    train_loader, val_loader = get_dataloaders(cfg)

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1
        cfg["training"]["val_every_n_epochs"] = 1

    # 2. Init Model
    logger.info("Initializing BrainFlowCFMv2...")
    bf_cfg = cfg["brainflow"]

    # Build modality_dims from config
    modality_dims = {}
    for mod_name, mod_cfg in cfg["raw_features"]["modalities"].items():
        dim = mod_cfg["dim"]
        # If layer_aggregation is "cat", multiply by number of layers
        if cfg["raw_features"].get("layer_aggregation") == "cat":
            n_layers = mod_cfg.get("n_layers", 1)
            dim = dim * n_layers
        modality_dims[mod_name] = dim

    logger.info("Modality dimensions: %s", modality_dims)

    model = BrainFlowCFMv2(
        modality_dims=modality_dims,
        latent_dim=bf_cfg["latent_dim"],
        encoder_params=bf_cfg["encoder"],
        decoder_type=bf_cfg.get("decoder_type", "dit"),
        decoder_params=bf_cfg["decoder"],
        cfm_params=bf_cfg["cfm"],
        cfg_drop_prob=bf_cfg.get("cfg_drop_prob", 0.1),
        n_voxels=cfg["fmri"].get("n_voxels", 1000),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # 3. Load frozen VAE decoder for fMRI eval
    try:
        vae = load_vae(cfg, device)
        logger.info("Loaded frozen VAE decoder for fMRI evaluation")
    except Exception as e:
        logger.warning("Could not load VAE decoder: %s. fMRI PCC will be 0.", e)
        vae = None

    # 4. Optimizer & Scheduler
    tr_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
    )

    total_steps = len(train_loader) * tr_cfg["n_epochs"]
    if args.fast_dev_run:
        total_steps = 2

    warmup_steps = int(total_steps * tr_cfg.get("warmup_ratio", 0.05))
    min_lr = tr_cfg.get("min_lr", 1e-6)

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / tr_cfg["lr"] + (1 - min_lr / tr_cfg["lr"]) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    # No GradScaler needed for bf16 (same exponent range as fp32)

    # 5. Output dir
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brain_flow_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Modality names for extracting from batch
    modality_names = list(cfg["raw_features"]["modalities"].keys())

    # EMA
    ema_decay = tr_cfg.get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)
    logger.info("EMA initialized with decay=%.4f", ema_decay)

    # 6. Training Loop
    best_val_corr = -1.0
    global_step = 0

    history_file = out_dir / "history.csv"
    with open(history_file, "w") as f:
        f.write("epoch,train_loss,train_diff_loss,train_prior_loss,train_reg_loss,prior_weight,val_latent_pearson,val_fmri_pcc,lr\n")

    for epoch in range(1, tr_cfg["n_epochs"] + 1):
        model.train()
        train_losses = []
        train_diff_losses = []
        train_prior_losses = []
        train_reg_losses = []

        # Prior weight annealing: linear from start → end over anneal_epochs
        pw_start = tr_cfg.get("prior_weight_start", 1.0)
        pw_end = tr_cfg.get("prior_weight_end", 0.1)
        pw_anneal = tr_cfg.get("prior_anneal_epochs", 100)
        progress = min(epoch / max(pw_anneal, 1), 1.0)
        prior_weight = pw_start * (1 - progress) + pw_end * progress

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            mod_features = _extract_modality_features(batch, modality_names, device)
            latents = batch["latent"].to(device)
            fmri = batch["fmri"].to(device) if "fmri" in batch else None

            # Build CFM mask
            B, T, _ = latents.shape
            mask = torch.ones(B, 1, T, device=device)
            if "valid_len" in batch:
                for b, vl in enumerate(batch["valid_len"]):
                    mask[b, 0, int(vl):] = 0.0

            # --- Data Augmentation (train only) ---
            # 1. Stochastic latent noise: simulate VAE sampling variability
            latent_noise_scale = tr_cfg.get("latent_noise_scale", 0.0)
            if latent_noise_scale > 0:
                latents = latents + latent_noise_scale * torch.randn_like(latents)

            # 2. Mixup: blend random pairs for smoother latent space
            mixup_alpha = tr_cfg.get("mixup_alpha", 0.0)
            if mixup_alpha > 0 and B > 1:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(B, device=device)
                latents = lam * latents + (1 - lam) * latents[perm]
                mask = torch.min(mask, mask[perm])  # conservative: valid only where both valid
                for mod_name in mod_features:
                    mod_features[mod_name] = lam * mod_features[mod_name] + (1 - lam) * mod_features[mod_name][perm]
                if fmri is not None:
                    fmri = lam * fmri + (1 - lam) * fmri[perm]

            reg_weight = tr_cfg.get("reg_weight", 0.1)
            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                diff_loss, prior_loss, reg_loss = model(mod_features, latents, mask=mask, fmri_target=fmri)
                loss = diff_loss + prior_weight * prior_loss + reg_weight * reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            train_diff_losses.append(diff_loss.item())
            train_prior_losses.append(prior_loss.item())
            train_reg_losses.append(reg_loss.item())
            global_step += 1

            # EMA update
            ema.update(model)

            if global_step % tr_cfg["log_every_n_steps"] == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(train_losses[-50:]):.4f}",
                    "diff": f"{np.mean(train_diff_losses[-50:]):.4f}",
                    "reg": f"{np.mean(train_reg_losses[-50:]):.4f}",
                    "pw": f"{prior_weight:.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Validation
        mean_corr = 0.0
        mean_fmri_corr = 0.0
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            # Use EMA weights for validation
            ema.apply_shadow(model)
            model.eval()

            logger.info("Running validation...")

            # Per-clip accumulators: {clip_key: {"sum": (n_trs, D), "count": (n_trs,)}}
            lat_pred_acc = {}   # predicted latents
            lat_tgt_acc = {}    # target latents
            fmri_pred_acc = {}  # predicted fMRI (via VAE decode)
            fmri_tgt_acc = {}   # target fMRI

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break

                    mod_features = _extract_modality_features(batch, modality_names, device)
                    latents = batch["latent"].to(device)
                    valid_lens = batch.get("valid_len", None)
                    if valid_lens is not None:
                        valid_lens = valid_lens.to(device)

                    gen_latents = model.synthesise(
                        mod_features,
                        n_timesteps=50,
                        temperature=0.0,  # Deterministic center-path ODE
                        guidance_scale=tr_cfg.get("guidance_scale", 1.5),
                        n_output_trs=latents.shape[1],
                        n_ensembles=1,
                    )

                    # Decode to fMRI if VAE available
                    gen_fmri = None
                    if vae is not None and "fmri" in batch:
                        true_fmri = batch["fmri"].to(device)
                        dummy_subj = torch.zeros(latents.shape[0], dtype=torch.long, device=device)
                        gen_fmri = vae.decode(gen_latents, dummy_subj)

                    # Accumulate per-clip, per-TR (de-duplicate overlapping windows)
                    clip_keys = batch["clip_key"]       # list of strings
                    start_idxs = batch["start_idx"]     # (B,)
                    n_trs_batch = batch["n_trs"]         # (B,)

                    for b in range(gen_latents.shape[0]):
                        vl = int(valid_lens[b].item()) if valid_lens is not None else gen_latents.shape[1]
                        ck = clip_keys[b]
                        si = int(start_idxs[b].item())
                        n_t = int(n_trs_batch[b].item())
                        lat_dim = gen_latents.shape[-1]

                        # Initialize clip accumulator if needed
                        if ck not in lat_pred_acc:
                            lat_pred_acc[ck] = {"sum": torch.zeros(n_t, lat_dim), "count": torch.zeros(n_t)}
                            lat_tgt_acc[ck] = {"sum": torch.zeros(n_t, lat_dim), "count": torch.zeros(n_t)}

                        # Accumulate predicted and target latents (vectorized slice)
                        lat_pred_acc[ck]["sum"][si:si+vl] += gen_latents[b, :vl].cpu()
                        lat_pred_acc[ck]["count"][si:si+vl] += 1
                        lat_tgt_acc[ck]["sum"][si:si+vl] += latents[b, :vl].cpu()
                        lat_tgt_acc[ck]["count"][si:si+vl] += 1

                        # fMRI accumulation (vectorized slice)
                        if gen_fmri is not None:
                            fmri_dim = gen_fmri.shape[-1]
                            if ck not in fmri_pred_acc:
                                fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                                fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                            fmri_pred_acc[ck]["sum"][si:si+vl] += gen_fmri[b, :vl].cpu()
                            fmri_pred_acc[ck]["count"][si:si+vl] += 1
                            fmri_tgt_acc[ck]["sum"][si:si+vl] += true_fmri[b, :vl].cpu()
                            fmri_tgt_acc[ck]["count"][si:si+vl] += 1

            # Average overlapping predictions and concatenate per-clip
            all_gen_lat, all_tgt_lat = [], []
            all_gen_fmri, all_tgt_fmri = [], []

            for ck in sorted(lat_pred_acc.keys()):
                count = lat_pred_acc[ck]["count"]
                valid_mask = count > 0
                if valid_mask.sum() < 2:
                    continue
                avg_pred = lat_pred_acc[ck]["sum"][valid_mask] / count[valid_mask].unsqueeze(-1)
                avg_tgt = lat_tgt_acc[ck]["sum"][valid_mask] / lat_tgt_acc[ck]["count"][valid_mask].unsqueeze(-1)
                all_gen_lat.append(avg_pred)
                all_tgt_lat.append(avg_tgt)

            for ck in sorted(fmri_pred_acc.keys()):
                count = fmri_pred_acc[ck]["count"]
                valid_mask = count > 0
                if valid_mask.sum() < 2:
                    continue
                avg_pred = fmri_pred_acc[ck]["sum"][valid_mask] / count[valid_mask].unsqueeze(-1)
                avg_tgt = fmri_tgt_acc[ck]["sum"][valid_mask] / fmri_tgt_acc[ck]["count"][valid_mask].unsqueeze(-1)
                all_gen_fmri.append(avg_pred)
                all_tgt_fmri.append(avg_tgt)

            # Compute PCC on de-duplicated, per-clip concatenated timeseries
            if all_gen_lat:
                all_gen_lat = torch.cat(all_gen_lat, dim=0)   # (unique_TRs, 64)
                all_tgt_lat = torch.cat(all_tgt_lat, dim=0)
                lat_pcc_per_dim = pearson_corr_per_dim(
                    all_gen_lat.unsqueeze(0), all_tgt_lat.unsqueeze(0)
                )
                mean_corr = float(lat_pcc_per_dim.mean().item())
            else:
                mean_corr = 0.0

            if all_gen_fmri:
                all_gen_fmri = torch.cat(all_gen_fmri, dim=0)
                all_tgt_fmri = torch.cat(all_tgt_fmri, dim=0)
                fmri_pcc_per_voxel = pearson_corr_per_dim(
                    all_gen_fmri.unsqueeze(0), all_tgt_fmri.unsqueeze(0)
                )
                mean_fmri_corr = float(fmri_pcc_per_voxel.mean().item())
            else:
                mean_fmri_corr = 0.0

            logger.info(
                "Epoch %d | Val Latent PCC: %.4f | Val fMRI PCC: %.4f",
                epoch, mean_corr, mean_fmri_corr,
            )

            # Best model by fMRI PCC (main metric)
            metric_for_best = mean_fmri_corr if vae is not None else mean_corr
            if metric_for_best > best_val_corr:
                best_val_corr = metric_for_best
                torch.save(model.state_dict(), out_dir / "best.pt")  # EMA weights
                logger.info("Saved new best EMA model to %s", out_dir / "best.pt")

            # Save history
            mean_train_loss = float(np.mean(train_losses))
            mean_diff_loss = float(np.mean(train_diff_losses))
            mean_prior_loss = float(np.mean(train_prior_losses))
            mean_reg_loss = float(np.mean(train_reg_losses))
            current_lr = scheduler.get_last_lr()[0]
            with open(history_file, "a") as f:
                f.write(f"{epoch},{mean_train_loss:.6f},{mean_diff_loss:.6f},{mean_prior_loss:.6f},{mean_reg_loss:.6f},{prior_weight:.4f},{mean_corr:.6f},{mean_fmri_corr:.6f},{current_lr:.2e}\n")

            # Restore original weights after validation
            ema.restore(model)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
        }, out_dir / "last.pt")

    logger.info("Training complete. Best val PCC: %.4f", best_val_corr)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/brain_flow_v2.yaml")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches to test pipeline")
    args = parser.parse_args()
    train(args)
