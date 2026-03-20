"""Train BrainFlow MOTFM-VAE — OT-CFM in VAE Latent Space.

Flow matching operates in 64-dim VAE latent space. At validation, the
generated latent is decoded by a frozen VAE decoder to 1000-dim fMRI
for PCC evaluation against ground truth.

Usage:
    python src/train_brainflow_motfm_vae.py --config src/configs/brain_flow_motfm_vae.yaml --fast_dev_run
    python src/train_brainflow_motfm_vae.py --config src/configs/brain_flow_motfm_vae.yaml
    python src/train_brainflow_motfm_vae.py --config src/configs/brain_flow_motfm_vae.yaml --resume
"""

import argparse
import copy
import logging
import math as pymath
import random
import shutil
import sys
import time
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


from src.data.dataset import (
    load_config,
    load_feature_clip_perfile,
    resample_features_to_tr,
    _get_fmri_filepath,
)
from src.models.brainflow.brain_flow_motfm_vae import BrainFlowMOTFM_VAE
from src.models.brainflow.fmri_vae import build_vae
from src.models.brainflow.moevae_temporal_encoder import build_moevae_temporal_encoder, build_moevae_direct_encoder

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_brainflow_motfm_vae")


def resolve_paths(cfg: dict, project_root: Path) -> dict:
    """Resolve paths for config."""
    cfg = cfg.copy()
    cfg["_project_root"] = str(project_root)
    cfg["_data_root"] = str(project_root / cfg["data_root"])
    cfg["_fmri_dir"] = str(project_root / cfg["data_root"] / cfg["fmri"]["dir"])
    return cfg


# =============================================================================
# Dataset — loads pre-computed VAE latents + raw fMRI for validation
# =============================================================================

class BrainFlowVAEDataset(Dataset):
    """Many-to-One dataset: N context TRs of features → 1 target VAE latent.

    Loads pre-computed VAE latent vectors as training targets instead of
    raw fMRI. Also loads raw fMRI for validation PCC computation.

    Optionally loads precomputed MoEVAE context latents (moevae_latent_dir)
    to avoid online MoEVAE inference during training.
    """

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.subjects = cfg["subjects"]
        self.splits_cfg = cfg.get("splits", {})

        self.features_dir = Path(PROJECT_ROOT) / cfg["raw_features"]["dir"]
        self.layer_aggregation = cfg["raw_features"].get("layer_aggregation", "mean")
        npy_dir = cfg.get("raw_features_npy_dir")
        if npy_dir:
            if self.layer_aggregation == "cat":
                npy_dir = f"{npy_dir}_{self.layer_aggregation}"
            self.features_npy_dir = Path(PROJECT_ROOT) / npy_dir
        else:
            self.features_npy_dir = None
        # Multilayer NPY directory (shape T, n_layers, D) — uses original H5 subdirs
        multilayer_dir = cfg.get("multilayer_npy_dir")
        self.multilayer_npy_dir = Path(PROJECT_ROOT) / multilayer_dir if multilayer_dir else None
        self.fmri_dir = Path(cfg["_fmri_dir"])

        # Latent directory (VAE 64-dim targets)
        self.latent_dir = Path(PROJECT_ROOT) / cfg["latent_dir"]

        # MoEVAE precomputed context directory (512-dim per TR)
        moevae_dir = cfg.get("moevae_latent_dir")
        self.moevae_latent_dir = Path(PROJECT_ROOT) / moevae_dir if moevae_dir else None

        self.tr = cfg["fmri"]["tr"]
        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end = cfg["fmri"].get("excluded_samples_end", 0)

        self.modalities = cfg["raw_features"]["modalities"]
        self.layer_aggregation = cfg["raw_features"].get("layer_aggregation", "mean")
        self.feature_freq = cfg["raw_features"].get("sample_freq", 2.0)

        self.feature_context_trs = cfg["sliding_window"].get("feature_context_trs", 10)
        self.feat_seq_len = self.feature_context_trs + 1
        self.stride = cfg["sliding_window"].get("stride", 1)

        self.hrf_delay = cfg["fmri"].get("hrf_delay", 5)
        self.temporal_jitter = cfg["sliding_window"].get("temporal_jitter", 0) if split == "train" else 0

        self._fmri_cache = {}
        self._feat_cache = {}
        self._latent_cache = {}
        self._moevae_cache = {}
        self.samples = self._build_index()

    def _normalize_clip_name(self, clip_name: str, task: str) -> str:
        prefix = f"{task}_"
        if clip_name.startswith(prefix):
            return clip_name[len(prefix):]
        return clip_name

    def _build_index(self):
        samples = []
        logger.info("Building %s split index...", self.split)

        for task, splits in self.splits_cfg.items():
            stim_types = splits.get(self.split, [])
            if not stim_types:
                continue

            for stim_type in stim_types:
                first_mod = next(iter(self.modalities))
                mod_cfg = self.modalities[first_mod]

                # Try to discover clips from multilayer NPY dir, then H5 dir
                clip_stems = []
                if self.multilayer_npy_dir is not None:
                    subdir = mod_cfg.get("subdir", first_mod)
                    npy_clip_dir = self.multilayer_npy_dir / subdir / task / stim_type
                    if npy_clip_dir.exists():
                        clip_stems = sorted(p.stem for p in npy_clip_dir.glob("*.npy"))

                if not clip_stems:
                    clip_dir = self.features_dir / mod_cfg["subdir"] / task / stim_type
                    if not clip_dir.exists():
                        logger.warning("Feature dir missing: %s", clip_dir)
                        continue
                    clip_stems = sorted(p.stem for p in clip_dir.glob("*.h5"))

                for clip_stem in clip_stems:
                    subj = self.subjects[0]
                    norm_name = self._normalize_clip_name(clip_stem, task)

                    fmri_data = self._load_fmri_clip(subj, task, norm_name)
                    if fmri_data is None:
                        fmri_data = self._load_fmri_clip(subj, task, clip_stem)
                    if fmri_data is None:
                        continue

                    n_trs = fmri_data.shape[0]

                    # Check latent file exists
                    latent_data = self._load_latent_clip(subj, task, stim_type, norm_name)
                    if latent_data is None:
                        latent_data = self._load_latent_clip(subj, task, stim_type, clip_stem)
                    if latent_data is None:
                        logger.warning("Latent file missing for %s/%s/%s", task, stim_type, norm_name)
                        continue

                    for target_tr in range(0, n_trs, self.stride):
                        samples.append({
                            "task": task,
                            "stim_type": stim_type,
                            "clip_stem": clip_stem,
                            "norm_name": norm_name,
                            "target_tr": target_tr,
                            "n_trs": n_trs,
                        })

        logger.info("Found %d samples for %s split.", len(samples), self.split)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_fmri_clip(self, subject, task, clip_name):
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

    def _load_latent_clip(self, subject, task, stim_type, clip_name):
        """Load pre-computed VAE latent for a clip."""
        cache_key = (subject, task, stim_type, clip_name)
        if cache_key in self._latent_cache:
            return self._latent_cache[cache_key]

        latent_path = self.latent_dir / subject / task / stim_type / f"{clip_name}.npy"
        if not latent_path.exists():
            return None

        data = np.load(latent_path).astype(np.float32)
        self._latent_cache[cache_key] = data
        return data

    def _load_moevae_clip(self, task, stim_type, clip_name):
        """Load pre-computed MoEVAE context latent for a clip."""
        if self.moevae_latent_dir is None:
            return None
        cache_key = (task, stim_type, clip_name)
        if cache_key in self._moevae_cache:
            return self._moevae_cache[cache_key]

        moevae_path = self.moevae_latent_dir / task / stim_type / f"{clip_name}.npy"
        if not moevae_path.exists():
            return None

        data = np.load(moevae_path).astype(np.float32)
        self._moevae_cache[cache_key] = data
        return data

    def __getitem__(self, idx):
        info = self.samples[idx]
        task = info["task"]
        stim_type = info["stim_type"]
        clip_stem = info["clip_stem"]
        norm_name = info["norm_name"]
        target_tr = info["target_tr"]
        n_trs = info["n_trs"]

        feat_current_tr = target_tr - self.hrf_delay

        if self.temporal_jitter > 0:
            jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
            feat_current_tr = feat_current_tr + jitter

        feat_start = feat_current_tr - self.feature_context_trs
        feat_end = feat_current_tr + 1

        mod_features = {}
        for mod_name, mod_cfg in self.modalities.items():
            try:
                feat = None

                # Try 1: Multilayer NPY (T, n_layers, D) → mean-pool → (T, D)
                if feat is None and self.multilayer_npy_dir is not None:
                    subdir = mod_cfg.get("subdir", mod_name)
                    cache_key = ("ml", mod_name, task, stim_type, norm_name)
                    if cache_key in self._feat_cache:
                        clip_data = self._feat_cache[cache_key]
                    else:
                        npy_path = self.multilayer_npy_dir / subdir / task / stim_type / f"{norm_name}.npy"
                        if not npy_path.exists():
                            npy_path = self.multilayer_npy_dir / subdir / task / stim_type / f"{clip_stem}.npy"
                        if npy_path.exists():
                            raw = np.load(npy_path)  # (T, L, D)
                            if raw.ndim == 3:
                                clip_data = raw.mean(axis=1).astype(np.float32)  # (T, D)
                            else:
                                clip_data = raw.astype(np.float32)  # already (T, D)
                            self._feat_cache[cache_key] = clip_data
                        else:
                            clip_data = None
                    if clip_data is not None:
                        safe_start = max(0, feat_start)
                        safe_end = min(clip_data.shape[0], feat_end)
                        if safe_start < safe_end:
                            feat = clip_data[safe_start:safe_end]
                        else:
                            feat = np.zeros((0, clip_data.shape[-1]), dtype=np.float32)

                # Try 2: Pre-pooled NPY (T, D)
                if feat is None and self.features_npy_dir is not None:
                    cache_key = (mod_name, task, stim_type, norm_name)
                    if cache_key in self._feat_cache:
                        clip_data = self._feat_cache[cache_key]
                    else:
                        subdir = mod_cfg.get("subdir", mod_name)
                        npy_path = self.features_npy_dir / subdir / task / stim_type / f"{norm_name}.npy"
                        if npy_path.exists():
                            clip_data = np.load(npy_path)
                            self._feat_cache[cache_key] = clip_data
                        else:
                            clip_data = None
                    if clip_data is not None:
                        safe_start = max(0, feat_start)
                        safe_end = min(clip_data.shape[0], feat_end)
                        if safe_start < safe_end:
                            feat = clip_data[safe_start:safe_end].astype(np.float32)
                        else:
                            feat = np.zeros((0, clip_data.shape[1]), dtype=np.float32)

                # Try 3: Raw H5 fallback
                if feat is None:
                    raw = load_feature_clip_perfile(
                        str(self.features_dir), mod_name, mod_cfg,
                        task, stim_type, clip_stem, self.layer_aggregation,
                    )
                    raw = resample_features_to_tr(raw, self.feature_freq, self.tr, n_trs)
                    clip_data = raw[0].T.astype(np.float32)
                    safe_start = max(0, feat_start)
                    safe_end = min(clip_data.shape[0], feat_end)
                    if safe_start < safe_end:
                        feat = clip_data[safe_start:safe_end]
                    else:
                        feat = np.zeros((0, clip_data.shape[1]), dtype=np.float32)

                if feat.shape[0] < self.feat_seq_len:
                    pad_len = self.feat_seq_len - feat.shape[0]
                    feat = np.pad(feat, ((pad_len, 0), (0, 0)), mode="constant")

                mod_features[mod_name] = torch.from_numpy(feat.copy())

            except (FileNotFoundError, KeyError, ValueError):
                dim = mod_cfg.get("dim", 1024)
                mod_features[mod_name] = torch.zeros(self.feat_seq_len, dim, dtype=torch.float32)

        # Load latent target
        subj = self.subjects[0]
        latent_data = self._load_latent_clip(subj, task, stim_type, norm_name)
        if latent_data is None:
            latent_data = self._load_latent_clip(subj, task, stim_type, clip_stem)

        latent_dim = self.cfg.get("brainflow", {}).get("latent_dim", 64)
        if latent_data is not None and target_tr < latent_data.shape[0]:
            latent_target = torch.from_numpy(latent_data[target_tr].copy())
        else:
            latent_target = torch.zeros(latent_dim, dtype=torch.float32)

        # Also load raw fMRI for validation
        fmri_data = self._load_fmri_clip(subj, task, norm_name)
        if fmri_data is None:
            fmri_data = self._load_fmri_clip(subj, task, clip_stem)
        if fmri_data is not None and target_tr < fmri_data.shape[0]:
            fmri_target = torch.from_numpy(fmri_data[target_tr].copy())
        else:
            n_voxels = self.cfg["fmri"].get("n_voxels", 1000)
            fmri_target = torch.zeros(n_voxels, dtype=torch.float32)

        result = {
            "latent": latent_target,
            "fmri": fmri_target,
            "clip_key": f"{task}/{stim_type}/{norm_name}",
            "target_tr": target_tr,
            "n_trs": n_trs,
        }
        result.update(mod_features)

        # Load precomputed MoEVAE context if available
        moevae_data = self._load_moevae_clip(task, stim_type, norm_name)
        if moevae_data is None:
            moevae_data = self._load_moevae_clip(task, stim_type, clip_stem)
        if moevae_data is not None:
            # Window the moevae context the same way as features
            safe_start = max(0, feat_start)
            safe_end = min(moevae_data.shape[0], feat_end)
            if safe_start < safe_end:
                ctx = moevae_data[safe_start:safe_end]
            else:
                ctx = np.zeros((0, moevae_data.shape[-1]), dtype=np.float32)
            if ctx.shape[0] < self.feat_seq_len:
                pad_len = self.feat_seq_len - ctx.shape[0]
                ctx = np.pad(ctx, ((pad_len, 0), (0, 0)), mode="constant")
            result["moevae_context"] = torch.from_numpy(ctx.copy())

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

def pearson_corr_per_dim(pred, target, valid_lens=None):
    """Compute Pearson correlation across time for each voxel."""
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
    target_cnt = target - target_mean
    cov = (pred_cnt * target_cnt).sum(dim=0)
    std = torch.sqrt((pred_cnt**2).sum(dim=0) * (target_cnt**2).sum(dim=0))
    corr = cov / (std + 1e-8)
    return corr


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
    """Exponential Moving Average of model weights for stable inference."""

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
    train_set = BrainFlowVAEDataset(cfg, split="train")
    val_set = BrainFlowVAEDataset(cfg, split="val")

    dl_cfg = cfg["dataloader"]
    batch_size = dl_cfg["batch_size"]

    train_sampler = ClipGroupedBatchSampler(train_set, batch_size, drop_last=True)

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
# Load frozen VAE decoder
# =============================================================================

def load_vae_decoder(cfg, device):
    """Load the pre-trained VAE and return its frozen decoder."""
    vae_cfg = cfg["vae"]
    vae_ckpt_path = Path(PROJECT_ROOT) / vae_cfg["checkpoint"]

    if not vae_ckpt_path.exists():
        logger.error("VAE checkpoint not found: %s", vae_ckpt_path)
        return None

    # Build VAE config for build_vae
    vae_build_cfg = {
        "fmri": cfg["fmri"],
        "subjects": cfg["subjects"],
        "vae": {
            "version": vae_cfg.get("version", 5),
            "latent_dim": vae_cfg.get("latent_dim", 64),
            "hidden_dim": vae_cfg.get("hidden_dim", 256),
            "num_res_blocks": vae_cfg.get("num_res_blocks", 4),
            "dropout": vae_cfg.get("dropout", 0.25),
            "use_subject": vae_cfg.get("use_subject", False),
            "subject_embed_dim": vae_cfg.get("subject_embed_dim", 64),
            "free_bits": 0.0,
            "lambda_pcc": 0.0,
        },
    }

    vae_model = build_vae(vae_build_cfg)

    # Load checkpoint
    ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    # Try EMA model first, fallback to regular model
    if "ema_model" in ckpt:
        vae_model.load_state_dict(ckpt["ema_model"])
        logger.info("Loaded VAE from EMA weights: %s", vae_ckpt_path)
    elif "model" in ckpt:
        vae_model.load_state_dict(ckpt["model"])
        logger.info("Loaded VAE from model weights: %s", vae_ckpt_path)
    else:
        vae_model.load_state_dict(ckpt)
        logger.info("Loaded VAE state dict: %s", vae_ckpt_path)

    vae_model = vae_model.to(device)
    vae_model.eval()

    # Freeze all parameters
    for p in vae_model.parameters():
        p.requires_grad = False

    logger.info("VAE decoder loaded and frozen (latent_dim=%d → n_voxels=%d)",
                vae_cfg.get("latent_dim", 64), cfg["fmri"]["n_voxels"])

    return vae_model


# =============================================================================
# Training
# =============================================================================

def train(args):
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    # Set moevae_latent_dir if using precomputed context
    use_precomputed_context = bool(args.moevae_precomputed)
    if use_precomputed_context:
        cfg["moevae_latent_dir"] = args.moevae_precomputed
        logger.info("Using PRECOMPUTED MoEVAE context from: %s", args.moevae_precomputed)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init Data
    train_loader, val_loader = get_dataloaders(cfg)

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1
        cfg["training"]["val_every_n_epochs"] = 1

    # 2. Init Model
    logger.info("Initializing BrainFlowMOTFM_VAE...")
    bf_cfg = cfg["brainflow"]

    modality_dims = {}
    for mod_name, mod_cfg in cfg["raw_features"]["modalities"].items():
        dim = mod_cfg["dim"]
        if cfg["raw_features"].get("layer_aggregation") == "cat":
            n_layers = mod_cfg.get("n_layers", 1)
            dim = dim * n_layers
        modality_dims[mod_name] = dim

    logger.info("Modality dimensions: %s", modality_dims)

    # Build encoder — MoEVAE hybrid or default TemporalFusionEncoder
    external_encoder = None
    if use_precomputed_context:
        # Precomputed mode: no encoder needed, set context_dim from MoEVAE latent
        moevae_cfg = cfg.get("moevae", {})
        moevae_latent_dim = moevae_cfg.get("model", {}).get("latent_dim", 512)
        vn_params = dict(bf_cfg.get("velocity_net", {}))
        vn_params["context_dim"] = moevae_latent_dim
        logger.info("PRECOMPUTED mode: VelocityNet context_dim=%d (no encoder)", moevae_latent_dim)
    elif args.moevae_checkpoint:
        moevae_ckpt = Path(args.moevae_checkpoint)
        if not moevae_ckpt.exists():
            logger.error("MoEVAE checkpoint not found: %s", moevae_ckpt)
            sys.exit(1)

        # Load MoEVAE modality configs from its config
        moevae_cfg = cfg.get("moevae", {})
        moevae_modality_configs = moevae_cfg.get("modalities", {})
        if not moevae_modality_configs:
            # Fallback: derive from raw_features modalities (assume n_layers=1 if missing)
            for mod_name, mod_cfg in cfg["raw_features"]["modalities"].items():
                moevae_modality_configs[mod_name] = {
                    "dim": mod_cfg["dim"],
                    "n_layers": mod_cfg.get("n_layers", 1),
                }

        if args.moevae_direct:
            # Variant A: frozen MoEVAE only, no temporal transformer
            external_encoder = build_moevae_direct_encoder(
                moevae_checkpoint=str(moevae_ckpt),
                modality_configs=moevae_modality_configs,
                moevae_params=moevae_cfg.get("model", {}),
                device=device,
            )
            logger.info("Using MoEVAE DIRECT encoder (no temporal transformer) from %s", moevae_ckpt)
        else:
            # Variant B: frozen MoEVAE + trainable temporal transformer
            temporal_cfg = bf_cfg.get("temporal_transformer", bf_cfg.get("encoder", {}))
            external_encoder = build_moevae_temporal_encoder(
                moevae_checkpoint=str(moevae_ckpt),
                modality_configs=moevae_modality_configs,
                moevae_params=moevae_cfg.get("model", {}),
                temporal_params=temporal_cfg,
                device=device,
            )
            logger.info("Using MoEVAE temporal encoder from %s", moevae_ckpt)
        vn_params = dict(bf_cfg.get("velocity_net", {}))
        vn_params["context_dim"] = external_encoder.hidden_dim
        logger.info("VelocityNet context_dim set to %d (from encoder)", external_encoder.hidden_dim)
    else:
        vn_params = dict(bf_cfg.get("velocity_net", {}))

    model = BrainFlowMOTFM_VAE(
        modality_dims=modality_dims,
        latent_dim=bf_cfg.get("latent_dim", 64),
        encoder_params=bf_cfg["encoder"],
        velocity_net_params=vn_params,
        encoder=external_encoder,
    ).to(device)

    # 3. Load frozen VAE decoder
    vae_model = load_vae_decoder(cfg, device)
    if vae_model is not None:
        # Attach the decode method to the flow model
        model.vae_decoder = vae_model.decode
        logger.info("VAE decoder attached to model for inference")

    # Load pre-trained encoder if provided
    if args.pretrained_encoder:
        enc_path = Path(args.pretrained_encoder)
        if enc_path.exists():
            enc_state = torch.load(enc_path, map_location=device, weights_only=False)
            model.encoder.load_state_dict(enc_state)
            logger.info("Loaded pre-trained encoder from %s", enc_path)
        else:
            logger.error("Pre-trained encoder not found: %s", enc_path)
            sys.exit(1)

    # Freeze encoder if requested or using precomputed context
    if use_precomputed_context:
        for p in model.encoder.parameters():
            p.requires_grad = False
        logger.info("PRECOMPUTED mode: Encoder FROZEN — only VelocityNet trainable")

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        # Keep final_norm trainable
        for p in model.encoder.final_norm.parameters():
            p.requires_grad = True
        logger.info("Encoder FROZEN — final_norm + VelocityNet trainable")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # 4. Optimizer & Scheduler
    tr_cfg = cfg["training"]

    if use_precomputed_context:
        trainable_params = list(model.velocity_net.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=tr_cfg["lr"],
            weight_decay=tr_cfg["weight_decay"],
        )
        logger.info("Optimizer: VelocityNet only, lr=%.2e", tr_cfg["lr"])
    elif args.freeze_encoder:
        trainable_params = list(model.velocity_net.parameters()) + \
                          list(model.encoder.final_norm.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=tr_cfg["lr"],
            weight_decay=tr_cfg["weight_decay"],
        )
        logger.info("Optimizer: VelocityNet + final_norm, lr=%.2e", tr_cfg["lr"])
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tr_cfg["lr"],
            weight_decay=tr_cfg["weight_decay"],
        )
        logger.info("Optimizer: all params, lr=%.2e", tr_cfg["lr"])

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

    n_param_groups = len(optimizer.param_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [cosine_with_warmup] * n_param_groups)

    # 5. Output dir
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_motfm_vae")
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, out_dir / "config.yaml")
    logger.info("Saved config to %s", out_dir / "config.yaml")

    modality_names = list(cfg["raw_features"]["modalities"].keys())

    # EMA
    ema_decay = tr_cfg.get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)
    logger.info("EMA initialized with decay=%.4f", ema_decay)

    # 6. Resume from checkpoint
    start_epoch = 1
    best_val_corr = -1.0
    global_step = 0

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

    # Solver config for validation sampling
    solver_cfg = cfg.get("solver_args", {})
    val_n_timesteps = solver_cfg.get("time_points", 50)
    val_solver_method = solver_cfg.get("method", "midpoint")

    # 7. Training loop
    for epoch in range(start_epoch, tr_cfg["n_epochs"] + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            latent_target = batch["latent"].to(device)  # (B, Z) — VAE latent

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                if use_precomputed_context and "moevae_context" in batch:
                    context = batch["moevae_context"].to(device)  # (B, T, 512)
                    loss = model.forward_with_context(context, latent_target)
                else:
                    mod_features = _extract_modality_features(batch, modality_names, device)
                    loss = model(mod_features, latent_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            global_step += 1

            ema.update(model)

            if global_step % tr_cfg["log_every_n_steps"] == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(train_losses[-50:]):.4f}",
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

                    fmri_target = batch["fmri"].to(device)  # Raw fMRI for PCC

                    # Generate fMRI via latent → VAE decode
                    if use_precomputed_context and "moevae_context" in batch:
                        context = batch["moevae_context"].to(device)
                        gen_fmri = model.synthesise_with_context(
                            context,
                            n_timesteps=val_n_timesteps,
                            solver_method=val_solver_method,
                        )
                    else:
                        mod_features = _extract_modality_features(batch, modality_names, device)
                        gen_fmri = model.synthesise(
                            mod_features,
                            n_timesteps=val_n_timesteps,
                            solver_method=val_solver_method,
                        )  # (B, V) — decoded fMRI

                    clip_keys = batch["clip_key"]
                    target_trs = batch["target_tr"]
                    n_trs_batch = batch["n_trs"]

                    B = gen_fmri.shape[0]
                    for b in range(B):
                        ck = clip_keys[b]
                        tr_idx = int(target_trs[b].item())
                        n_t = int(n_trs_batch[b].item())
                        fmri_dim = gen_fmri.shape[-1]

                        if ck not in fmri_pred_acc:
                            fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                            fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}

                        fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b].cpu()
                        fmri_pred_acc[ck]["count"][tr_idx] += 1
                        fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b].cpu()
                        fmri_tgt_acc[ck]["count"][tr_idx] += 1

            # Average overlapping predictions
            all_gen_fmri, all_tgt_fmri = [], []
            for ck in sorted(fmri_pred_acc.keys()):
                count = fmri_pred_acc[ck]["count"]
                valid_mask = count > 0
                if valid_mask.sum() < 2:
                    continue
                avg_pred = fmri_pred_acc[ck]["sum"][valid_mask] / count[valid_mask].unsqueeze(-1)
                avg_tgt = fmri_tgt_acc[ck]["sum"][valid_mask] / fmri_tgt_acc[ck]["count"][valid_mask].unsqueeze(-1)
                all_gen_fmri.append(avg_pred)
                all_tgt_fmri.append(avg_tgt)

            # Compute PCC
            if all_gen_fmri:
                all_gen_fmri = torch.cat(all_gen_fmri, dim=0)
                all_tgt_fmri = torch.cat(all_tgt_fmri, dim=0)
                fmri_pcc_per_voxel = pearson_corr_per_dim(
                    all_gen_fmri.unsqueeze(0), all_tgt_fmri.unsqueeze(0)
                )
                mean_fmri_corr = float(fmri_pcc_per_voxel.mean().item())
            else:
                mean_fmri_corr = 0.0

            logger.info("Epoch %d | Val fMRI PCC: %.4f", epoch, mean_fmri_corr)

            if mean_fmri_corr > best_val_corr:
                best_val_corr = mean_fmri_corr
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("Saved new best EMA model to %s", out_dir / "best.pt")

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
    parser.add_argument("--config", type=str, default="src/configs/brain_flow_motfm_vae.yaml")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches to test pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                        help="Path to pre-trained encoder weights (best_encoder.pt)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights (use with --pretrained_encoder)")
    parser.add_argument("--moevae_checkpoint", type=str, default=None,
                        help="Path to pretrained MoEVAE checkpoint for hybrid encoder")
    parser.add_argument("--moevae_direct", action="store_true",
                        help="Use MoEVAE direct encoder (no temporal transformer, Variant A)")
    parser.add_argument("--moevae_precomputed", type=str, default=None,
                        help="Path to precomputed MoEVAE latents dir (e.g. Data/moevae_latents)")
    args = parser.parse_args()
    train(args)
