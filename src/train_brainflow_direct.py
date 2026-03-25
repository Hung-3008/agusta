"""Train BrainFlow Direct — Flow Matching with Pre-extracted Context.

Supports three modes:
  1. fMRI-space FM (output_dim=1000): target = raw fMRI from H5
  2. Latent FM (output_dim=64): target = pre-extracted VAE latents from .npy,
     frozen VAE decoder converts latent predictions → fMRI for PCC evaluation.
  3. PCA FM (output_dim=K): target = PCA-projected fMRI (on-the-fly),
     PCA inverse_transform converts predictions → fMRI for PCC evaluation.

Usage:
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct_v3_pca.yaml --fast_dev_run
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct_v3_pca.yaml
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct_v3_pca.yaml --resume
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
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_config, _get_fmri_filepath
from src.models.brainflow.brain_flow_direct_v2 import BrainFlowDirectV2
from src.models.brainflow.brain_flow_direct_v3 import BrainFlowDirectV3

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_brainflow_direct")


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
    """Loads pre-extracted context latents + raw fMRI, fully preloaded into RAM.

    All data is loaded during __init__ for zero file I/O during training.
    Each sample: windowed context (feat_seq_len, context_dim) → 1 fMRI (V,).
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
            
        # Auto-compute context_dim from modality_dims if available
        self.modality_dims = cfg.get("modality_dims", None)
        if self.modality_dims:
            self.context_dim = sum(self.modality_dims)
        else:
            self.context_dim = cfg.get("context_dim", 5760)

        self.tr = cfg["fmri"]["tr"]
        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end = cfg["fmri"].get("excluded_samples_end", 0)

        self.feature_context_trs = cfg["sliding_window"].get("feature_context_trs", 10)
        self.feat_seq_len = self.feature_context_trs + 1  # 10 past + 1 current
        self.stride = cfg["sliding_window"].get("stride", 1)
        self.hrf_delay = cfg["fmri"].get("hrf_delay", 5)
        self.temporal_jitter = cfg["sliding_window"].get("temporal_jitter", 0) if split == "train" else 0

        # Build subject → index mapping
        self.subject_to_idx = {s: i for i, s in enumerate(self.subjects)}

        # --- PCA target mode ---
        # When pca_model_path is set, PCA-transform fMRI targets on-the-fly.
        # PCA components/mean are stored as torch tensors for fast GPU matmul.
        self.pca_mode = False
        self.pca_dim = cfg.get("pca_dim", None)
        self.pca_components = None  # (K, V) tensor
        self.pca_mean = None        # (V,) tensor
        if "pca_model_path" in cfg:
            pca_path = Path(PROJECT_ROOT) / cfg["pca_model_path"]
            if pca_path.exists():
                pca = joblib.load(pca_path)
                self.pca_components = torch.from_numpy(pca.components_.astype(np.float32))  # (K, V)
                self.pca_mean = torch.from_numpy(pca.mean_.astype(np.float32))  # (V,)
                self.pca_dim = pca.n_components_
                self.pca_mode = True
                logger.info("PCA target mode: %d components, explained_var=%.4f",
                            self.pca_dim, pca.explained_variance_ratio_.sum())
                del pca
            else:
                logger.warning("PCA model not found at %s — falling back to fMRI mode", pca_path)

        # --- Latent target mode ---
        # When latent_target_dir is set, load pre-extracted VAE latents as targets
        # instead of raw fMRI. Raw fMRI is still loaded for validation PCC.
        self.latent_target_dir = None
        self.latent_dim = cfg.get("latent_dim", None)
        if "latent_target_dir" in cfg:
            self.latent_target_dir = Path(PROJECT_ROOT) / cfg["latent_target_dir"]
            logger.info("Latent target mode: loading from %s (dim=%s)",
                        self.latent_target_dir, self.latent_dim)

        # Load global fMRI normalization stats (per-voxel mean/std from training set)
        self.use_global_stats = cfg["fmri"].get("use_global_stats", False)
        self.fmri_stats = {}  # subject → {"mean": np.array, "std": np.array}
        if self.use_global_stats:
            for subj in self.subjects:
                stats_dir = self.fmri_dir / subj / "stats"
                mean_path = stats_dir / "global_mean.npy"
                std_path = stats_dir / "global_std.npy"
                if mean_path.exists() and std_path.exists():
                    fmri_mean = np.load(mean_path).astype(np.float32)
                    fmri_std = np.load(std_path).astype(np.float32)
                    self.fmri_stats[subj] = {"mean": fmri_mean, "std": fmri_std}
                    logger.info("Loaded global fMRI stats for %s (mean_avg=%.4f, std_avg=%.4f)",
                               subj, fmri_mean.mean(), fmri_std.mean())
                else:
                    logger.warning("Global stats not found at %s", stats_dir)

        # Preloaded data: separated to avoid context duplication across subjects
        # Stored as torch float32 tensors for zero-conversion-cost in __getitem__
        self._ctx_clips = {}     # "task/stim_type/clip" → torch.Tensor (float32, shared)
        self._fmri_clips = {}    # "subject/task/clip" → torch.Tensor (float32, per-subject)
        self._latent_clips = {}  # "subject/task/clip" → torch.Tensor (float32, per-subject)
        self.samples = self._build_index_and_preload()

    def _normalize_clip_name(self, clip_name: str, task: str) -> str:
        prefix = f"{task}_"
        if clip_name.startswith(prefix):
            return clip_name[len(prefix):]
        return clip_name

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
        
        # Global normalization (per-voxel, from training set statistics)
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
        # Find any .npy file in the modality directory
        for f in ctx_dir.rglob("*.npy"):
            dim = np.load(f).shape[-1]
            self._modality_dim_cache[key] = dim
            return dim
        raise ValueError(f"No .npy files found in {ctx_dir}")

    def _build_index_and_preload(self):
        """Build sample index and preload all data into RAM.

        Memory optimizations:
          - Context features stored once per clip (shared across subjects)
          - Both context and fMRI stored as float16 (converted to float32 in __getitem__)
        """
        samples = []
        ctx_bytes = 0
        fmri_bytes = 0
        n_ctx_clips = 0
        n_fmri_clips = 0
        logger.info("Building %s split index and preloading data...", self.split)

        for task, splits in self.splits_cfg.items():
            stim_types = splits.get(self.split, [])
            if not stim_types:
                continue

            for stim_type in stim_types:
                # Use the first directory to find all available clip names
                ref_ctx_dir = self.context_dirs[0] / task / stim_type
                if not ref_ctx_dir.exists():
                    logger.warning("Context dir missing: %s", ref_ctx_dir)
                    continue

                clip_stems = sorted(p.stem for p in ref_ctx_dir.glob("*.npy"))

                for clip_stem in clip_stems:
                    norm_name = self._normalize_clip_name(clip_stem, task)

                    # --- Load context ONCE per clip (shared across subjects) ---
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
                                ctx_arrays.append(np.load(ctx_path).astype(np.float32))
                            else:
                                # Zero-pad: detect dim from any available file in this modality
                                mod_dim = self._get_modality_dim(ctx_dir)
                                # Use a default length; will be truncated below
                                ref_len = ctx_arrays[0].shape[0] if ctx_arrays else 500
                                ctx_arrays.append(np.zeros((ref_len, mod_dim), dtype=np.float32))

                        # Truncate to minimum temporal length
                        min_len = min(arr.shape[0] for arr in ctx_arrays)
                        ctx_arrays = [arr[:min_len] for arr in ctx_arrays]

                        # Concatenate all modalities and store as float32 torch tensor
                        ctx_data = torch.from_numpy(
                            np.concatenate(ctx_arrays, axis=-1).astype(np.float32)
                        )
                        self._ctx_clips[ctx_key] = ctx_data
                        ctx_bytes += ctx_data.nelement() * 4
                        n_ctx_clips += 1

                    # --- Load fMRI per subject ---
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

                        # --- Load latent targets if in latent mode ---
                        if self.latent_target_dir is not None and fmri_key not in self._latent_clips:
                            latent_path = self._find_latent_file(subj, task, norm_name, clip_stem)
                            if latent_path is not None:
                                latent_data = np.load(latent_path).astype(np.float32)
                                self._latent_clips[fmri_key] = torch.from_numpy(latent_data)
                            else:
                                self._latent_clips[fmri_key] = None
                                logger.warning("Latent file missing for %s", fmri_key)

                        n_trs = fmri_data.shape[0]

                        for target_tr in range(0, n_trs, self.stride):
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

        latent_bytes = sum(t.nelement() * 4 for t in self._latent_clips.values() if t is not None)
        total_bytes = ctx_bytes + fmri_bytes + latent_bytes
        mode_str = "LATENT" if self.latent_target_dir else "fMRI"
        logger.info("[%s mode] Preloaded %d ctx clips (%.2f GB) + %d fMRI clips (%.2f GB) "
                     "+ %.2f GB latents = %.2f GB total → %d samples for %s split.",
                     mode_str, n_ctx_clips, ctx_bytes / 1e9,
                     n_fmri_clips, fmri_bytes / 1e9,
                     latent_bytes / 1e9,
                     total_bytes / 1e9, len(samples), self.split)
        return samples

    def _find_latent_file(self, subject, task, norm_name, clip_stem):
        """Find the latent .npy file matching an fMRI clip."""
        latent_dir = self.latent_target_dir / subject / task
        # Try H5-key-matching names (same as extract_vae_latents.py output)
        for name in [norm_name, clip_stem, f"{task}_{norm_name}"]:
            p = latent_dir / f"{name}.npy"
            if p.exists():
                return p
        # Fallback: fuzzy match
        for p in latent_dir.glob("*.npy"):
            if norm_name in p.stem or clip_stem in p.stem:
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        ctx_key = info["ctx_key"]
        fmri_key = info["fmri_key"]
        target_tr = info["target_tr"]
        n_trs = info["n_trs"]

        ctx_data = self._ctx_clips[ctx_key]    # float32 torch tensor, shared
        fmri_data = self._fmri_clips[fmri_key]  # float32 torch tensor, per-subject

        # Feature window
        actual_movie_tr = target_tr + self.excl_start
        feat_current_tr = actual_movie_tr - self.hrf_delay
        if self.temporal_jitter > 0:
            jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
            feat_current_tr = feat_current_tr + jitter

        feat_start = feat_current_tr - self.feature_context_trs
        feat_end = feat_current_tr + 1

        # Window context (pure tensor slicing, no conversion)
        safe_start = max(0, feat_start)
        safe_end = min(ctx_data.shape[0], feat_end)
        if safe_start < safe_end:
            ctx = ctx_data[safe_start:safe_end]
        else:
            ctx = torch.zeros(0, ctx_data.shape[-1])
        if ctx.shape[0] < self.feat_seq_len:
            pad_len = self.feat_seq_len - ctx.shape[0]
            ctx = torch.nn.functional.pad(ctx, (0, 0, pad_len, 0))
        context = ctx

        # fMRI target (always loaded for validation PCC)
        if target_tr < fmri_data.shape[0]:
            fmri_target = fmri_data[target_tr].clone()
        else:
            n_voxels = self.cfg["fmri"].get("n_voxels", 1000)
            fmri_target = torch.zeros(n_voxels, dtype=torch.float32)

        # PCA target (on-the-fly projection, no pre-extraction needed)
        pca_target = None
        if self.pca_mode and self.pca_components is not None:
            # PCA transform: (fmri - mean) @ components.T → (K,)
            pca_target = (fmri_target - self.pca_mean) @ self.pca_components.T

        # Latent target (for training in latent mode)
        latent_target = None
        if self.latent_target_dir is not None:
            latent_data = self._latent_clips.get(fmri_key)
            if latent_data is not None and target_tr < latent_data.shape[0]:
                latent_target = latent_data[target_tr].clone()
            else:
                latent_target = torch.zeros(self.latent_dim or 64, dtype=torch.float32)

        result = {
            "context": context,
            "fmri": fmri_target,
            "clip_key": f"{info['subject']}/{ctx_key}",
            "subject_idx": info["subject_idx"],
            "target_tr": target_tr,
            "n_trs": n_trs,
        }
        if pca_target is not None:
            result["pca"] = pca_target
        if latent_target is not None:
            result["latent"] = latent_target
        return result


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
        n = len(sum(self.clip_groups.values(), []))
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# =============================================================================
# Metrics
# =============================================================================

def pearson_corr_per_dim(pred, target):
    """Per-voxel PCC across time. Input: (1, N, D) or (N, D)."""
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    pred_cnt = pred - pred.mean(dim=0, keepdim=True)
    target_cnt = target - target.mean(dim=0, keepdim=True)
    cov = (pred_cnt * target_cnt).sum(dim=0)
    std = torch.sqrt((pred_cnt**2).sum(dim=0) * (target_cnt**2).sum(dim=0))
    return cov / (std + 1e-8)


# =============================================================================
# EMA
# =============================================================================

class EMAModel:
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
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


# =============================================================================
# Dataloaders
# =============================================================================

def get_dataloaders(cfg):
    train_set = DirectFlowDataset(cfg, split="train")
    val_set = DirectFlowDataset(cfg, split="val")

    dl_cfg = cfg["dataloader"]
    batch_size = dl_cfg["batch_size"]

    train_sampler = ClipGroupedBatchSampler(train_set, batch_size, drop_last=True)

    # num_workers=0: data is pre-loaded as torch tensors in RAM,
    # so multiprocessing only adds forking overhead (14GB+ process)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=dl_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=dl_cfg["val_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False,
    )
    return train_loader, val_loader


# =============================================================================
# Training
# =============================================================================

def load_frozen_vae_decoder(cfg, device):
    """Load a frozen VAE decoder for latent → fMRI conversion during validation."""
    from src.models.brainflow.fmri_vae import build_vae
    from src.data.dataset import load_config as load_vae_config

    vae_config_path = PROJECT_ROOT / cfg["vae_config"]
    vae_ckpt_path = PROJECT_ROOT / cfg["vae_checkpoint"]

    vae_cfg = load_vae_config(vae_config_path)
    # Ensure subjects match
    vae_cfg["subjects"] = cfg["subjects"]

    vae_model = build_vae(vae_cfg).to(device)
    ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    if "ema_model" in ckpt:
        vae_model.load_state_dict(ckpt["ema_model"])
        logger.info("✅ Loaded frozen VAE decoder (EMA weights)")
    else:
        vae_model.load_state_dict(ckpt["model"])
        logger.info("✅ Loaded frozen VAE decoder")
    del ckpt

    vae_model.eval()
    for p in vae_model.parameters():
        p.requires_grad_(False)

    return vae_model


def decode_latents_to_fmri(vae_decoder, latent_pred, subject_ids):
    """Decode latent predictions to fMRI space using frozen VAE decoder.

    Args:
        vae_decoder: Frozen fMRI_VAE_v5 model.
        latent_pred: (B, Z) predicted latent vectors.
        subject_ids: (B,) subject indices.

    Returns:
        fmri_pred: (B, V) decoded fMRI predictions.
    """
    # VAE decoder expects (B, T, Z) — wrap single TR as T=1
    z = latent_pred.unsqueeze(1)  # (B, 1, Z)
    fmri = vae_decoder.decode(z, subject_ids)  # (B, 1, V)
    return fmri.squeeze(1)  # (B, V)


def train(args):
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check mode: PCA > latent > fMRI
    pca_mode = "pca_model_path" in cfg
    latent_mode = "latent_target_dir" in cfg and not pca_mode
    if pca_mode:
        logger.info("🧪 PCA FLOW MATCHING MODE (pca_dim=%d)", cfg.get("pca_dim", 100))
    elif latent_mode:
        logger.info("🧪 LATENT FLOW MATCHING MODE (target_dim=%d)", cfg.get("latent_dim", 64))

    # Load PCA model for validation inverse transform
    pca_components_gpu = None
    pca_mean_gpu = None
    if pca_mode:
        pca_path = Path(PROJECT_ROOT) / cfg["pca_model_path"]
        pca = joblib.load(pca_path)
        pca_components_gpu = torch.from_numpy(pca.components_.astype(np.float32)).to(device)  # (K, V)
        pca_mean_gpu = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)  # (V,)
        logger.info("Loaded PCA model for validation: %d components", pca.n_components_)
        del pca

    # 1. Data
    train_loader, val_loader = get_dataloaders(cfg)

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1
        cfg["training"]["val_every_n_epochs"] = 1

    # 2. Model
    bf_cfg = cfg["brainflow"]
    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])
    model_version = cfg.get("model_version", "v2")

    vn_params = dict(bf_cfg.get("velocity_net", {}))
    modality_dims = cfg.get("modality_dims", None)
    
    if "base_model" in cfg and modality_dims:
        base_cfg_path = cfg["base_model"].get("config")
        base_model_cfg = load_config(base_cfg_path)
        base_modality_count = len(base_model_cfg.get("modalities", []))
        # New model only uses the remaining modalities
        vn_params["modality_dims"] = modality_dims[base_modality_count:]
        logger.info("Residual Flow mode: target model uses %d modalities %s", 
                    len(vn_params["modality_dims"]), vn_params["modality_dims"])
    elif modality_dims:
        vn_params["modality_dims"] = modality_dims

    if model_version == "v3":
        logger.info("Initializing BrainFlowDirectV3 (NSD-inspired)...")
        reg_weight = bf_cfg.get("reg_weight", 1.0)
        cont_weight = bf_cfg.get("cont_weight", 0.1)
        cont_dim = bf_cfg.get("cont_dim", 256)
        model = BrainFlowDirectV3(
            output_dim=output_dim,
            velocity_net_params=vn_params,
            n_subjects=len(cfg["subjects"]),
            reg_weight=reg_weight,
            cont_weight=cont_weight,
            cont_dim=cont_dim,
        ).to(device)
    else:
        logger.info("Initializing BrainFlowDirectV2...")
        sp_params = dict(bf_cfg.get("source_predictor", {}))
        source_mode = bf_cfg.get("source_mode", "csfm")
        model = BrainFlowDirectV2(
            output_dim=output_dim,
            velocity_net_params=vn_params,
            n_subjects=len(cfg["subjects"]),
            source_predictor_params=sp_params,
            source_mode=source_mode,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # 3. Frozen VAE decoder (latent mode only)
    vae_decoder = None
    if latent_mode and "vae_checkpoint" in cfg:
        vae_decoder = load_frozen_vae_decoder(cfg, device)

    # 3.5 Frozen Base Model for Residual Flow Matching
    base_model = None
    if "base_model" in cfg:
        base_cfg_path = cfg["base_model"].get("config")
        base_ckpt_path = cfg["base_model"].get("checkpoint")
        if base_cfg_path and base_ckpt_path:
            logger.info("loading frozen base model for Residual Flow Matching...")
            base_model_cfg = load_config(base_cfg_path)
            
            # Reconstruct the appropriate model architecture (assuming V3 style)
            bm_output_dim = base_model_cfg["brainflow"].get("output_dim", 1000)
            bm_vn_params = dict(base_model_cfg["brainflow"]["velocity_net"])
            if "modalities" in base_model_cfg:
                bm_vn_params["modality_dims"] = [m["dim"] for m in base_model_cfg["modalities"]]
            else:
                bm_vn_params["modality_dims"] = base_model_cfg.get("modality_dims", [1408])
            bm_reg_weight = base_model_cfg["brainflow"].get("reg_weight", 0.5)

            base_model = BrainFlowDirectV3(
                output_dim=bm_output_dim,
                velocity_net_params=bm_vn_params,
                n_subjects=len(cfg["subjects"]),
                reg_weight=bm_reg_weight,
            ).to(device)

            base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
            bm_state = base_ckpt["model"] if "model" in base_ckpt else base_ckpt
            base_model.load_state_dict(bm_state)
            base_model.eval()
            for p in base_model.parameters():
                p.requires_grad_(False)
            logger.info("✅ Loaded frozen base model from %s", base_ckpt_path)
            del base_ckpt

    # 3. Optimizer & Scheduler
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
    base_lr = tr_cfg["lr"]

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    # 4. Output dir
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_direct")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    # EMA
    ema_decay = tr_cfg.get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)
    logger.info("EMA initialized with decay=%.4f", ema_decay)

    # Resume
    start_epoch = 1
    best_val_corr = -1.0
    global_step = 0

    if args.warmstart:
        ws_path = Path(args.warmstart)
        if ws_path.exists():
            ckpt = torch.load(ws_path, map_location=device, weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            # Filter out keys with shape mismatches (e.g., after adding modalities)
            model_state = model.state_dict()
            filtered_state = {}
            skipped = []
            for k, v in state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                elif k in model_state:
                    skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(model_state[k].shape)}")
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)
            if skipped:
                logger.info("Warmstart — skipped %d size-mismatched keys:", len(skipped))
                for s in skipped:
                    logger.info("  %s", s)
            if missing:
                logger.info("Warmstart — missing keys (%d): %s", len(missing), missing[:10])
            logger.info("Warmstarted from %s (epoch %s), loaded %d/%d keys",
                         ws_path, ckpt.get("epoch", "?"), len(filtered_state), len(state))
            del ckpt
        else:
            logger.warning("--warmstart path %s not found. Starting from scratch.", ws_path)

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
            logger.info("Resumed from epoch %d (global_step=%d)", ckpt["epoch"], global_step)
            del ckpt
        else:
            logger.warning("--resume specified but no last.pt found. Starting from scratch.")

    history_file = out_dir / "history.csv"
    if start_epoch == 1:
        with open(history_file, "w") as f:
            f.write("epoch,train_loss,val_fmri_pcc,lr\n")

    # Solver config
    solver_cfg = cfg.get("solver_args", {})
    val_n_timesteps = solver_cfg.get("time_points", 50)
    val_solver_method = solver_cfg.get("method", "midpoint")
    val_cfg_scale = solver_cfg.get("cfg_scale", 0.0)

    # 5. Training loop
    for epoch in range(start_epoch, tr_cfg["n_epochs"] + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            context = batch["context"].to(device)     # (B, T, C)
            subject_ids = batch["subject_idx"].to(device)  # (B,)

            # Choose target based on mode: PCA > latent > fMRI
            if pca_mode and "pca" in batch:
                target = batch["pca"].to(device)       # (B, K=100)
            elif latent_mode and "latent" in batch:
                target = batch["latent"].to(device)    # (B, Z=64)
            else:
                target = batch["fmri"].to(device)      # (B, V=1000)
            
            # Classifier-Free Guidance (CFG) / Condition Dropout
            # Drop the context 10% of the time to regularize the network
            if random.random() < 0.1:
                context = torch.zeros_like(context)

            # Residual Flow Matching: get starting distribution from base_model
            starting_distribution = None
            if base_model is not None:
                with torch.no_grad():
                    # Base model uses first N modalities (e.g. 8)
                    base_context_len = sum(base_model.fusion_block.modality_dims)
                    base_context = context[..., :base_context_len]
                    bm_encoded = base_model.velocity_net.encode_context_from_cond(base_context) 
                    bm_pooled = bm_encoded.mean(dim=1)
                    starting_distribution = base_model.reg_output(base_model.reg_head(bm_pooled))
                    
                    # Target model uses only the LAST modality (e.g. 9th)
                    # We slice off the base modalities so only the new one remains
                    context = context[..., base_context_len:]

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                losses = model.compute_loss(
                    context, 
                    target, 
                    subject_ids=subject_ids, 
                    starting_distribution=starting_distribution
                )
                loss = losses["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            global_step += 1
            ema.update(model)

            if global_step % tr_cfg["log_every_n_steps"] == 0:
                flow_l = losses["flow_loss"].item()
                align_l = losses["align_loss"].item()
                kld_l = losses["kld_loss"].item()
                cont_l = losses.get("cont_loss", torch.tensor(0.0)).item()
                pbar.set_postfix({
                    "loss": f"{np.mean(train_losses[-50:]):.4f}",
                    "flow": f"{flow_l:.4f}",
                    "reg": f"{align_l:.4f}",
                    "cont": f"{cont_l:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Validation
        mean_fmri_corr = 0.0
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()
            logger.info("Running validation...")

            fmri_pred_acc = {}
            fmri_tgt_acc = {}

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break

                    context = batch["context"].to(device)
                    fmri_target = batch["fmri"].to(device)
                    subject_ids = batch["subject_idx"].to(device)

                    synth_kwargs = dict(
                        n_timesteps=val_n_timesteps,
                        solver_method=val_solver_method,
                        subject_ids=subject_ids,
                    )
                    if model_version == "v3" and val_cfg_scale > 0:
                        synth_kwargs["cfg_scale"] = val_cfg_scale
                        
                    # Residual Flow Matching for inference
                    if base_model is not None:
                        # Extract the base model's prediction as the starting point x_0
                        base_context_len = sum(base_model.fusion_block.modality_dims)
                        base_context = context[..., :base_context_len]
                        
                        bm_encoded = base_model.velocity_net.encode_context_from_cond(base_context) 
                        bm_pooled = bm_encoded.mean(dim=1)
                        synth_kwargs["starting_distribution"] = base_model.reg_output(base_model.reg_head(bm_pooled))
                        
                        # Target model only uses the last modality
                        context = context[..., base_context_len:]

                    gen_output = model.synthesise(
                        context, **synth_kwargs,
                    )  # (B, output_dim) — PCA (100), latent (64), or fMRI (1000)

                    # Decode to fMRI space for PCC evaluation
                    if pca_mode and pca_components_gpu is not None:
                        # PCA inverse transform: pred @ components + mean → (B, 1000)
                        gen_fmri = gen_output @ pca_components_gpu + pca_mean_gpu
                    elif latent_mode and vae_decoder is not None:
                        gen_fmri = decode_latents_to_fmri(
                            vae_decoder, gen_output, subject_ids
                        )  # (B, 1000)
                    else:
                        gen_fmri = gen_output  # (B, 1000)

                    clip_keys = batch["clip_key"]
                    target_trs = batch["target_tr"]
                    n_trs_batch = batch["n_trs"]
                    fmri_target = batch["fmri"].to(device)  # Always raw fMRI for PCC

                    B = gen_fmri.shape[0]
                    fmri_dim = gen_fmri.shape[-1]
                    for b in range(B):
                        ck = clip_keys[b]
                        tr_idx = int(target_trs[b].item())
                        n_t = int(n_trs_batch[b].item())

                        if ck not in fmri_pred_acc:
                            fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                            fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}

                        fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b].cpu()
                        fmri_pred_acc[ck]["count"][tr_idx] += 1
                        fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b].cpu()
                        fmri_tgt_acc[ck]["count"][tr_idx] += 1

            # Average and compute PCC
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
            else:
                mean_fmri_corr = 0.0

            mode_tag = " (PCA→fMRI)" if pca_mode else (" (latent→fMRI)" if latent_mode else "")
            logger.info("Epoch %d | Val fMRI PCC: %.4f%s", epoch, mean_fmri_corr, mode_tag)

            if mean_fmri_corr > best_val_corr:
                best_val_corr = mean_fmri_corr
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("Saved new best EMA model (PCC=%.4f)", best_val_corr)

            mean_train_loss = float(np.mean(train_losses))
            current_lr = scheduler.get_last_lr()[0]
            with open(history_file, "a") as f:
                f.write(f"{epoch},{mean_train_loss:.6f},{mean_fmri_corr:.6f},{current_lr:.2e}\n")

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
    parser.add_argument("--config", type=str, default="src/configs/brain_flow_direct.yaml")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches to test pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    parser.add_argument("--warmstart", type=str, default=None,
                        help="Path to checkpoint for warm-starting (strict=False, ignores mismatched layers)")
    args = parser.parse_args()
    train(args)
