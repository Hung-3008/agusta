"""Train BrainFlow v3 — Direct regression from multimodal features to fMRI.

Usage:
    python src/train_brain_flow_v3.py --config src/configs/brain_flow_v3.yaml
    python src/train_brain_flow_v3.py --config src/configs/brain_flow_v3.yaml --fast_dev_run
"""

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    load_config,
    load_feature_clip_perfile,
    resample_features_to_tr,
    _get_fmri_filepath,
)
from src.models.brainflow.brain_flow_v3 import BrainFlowDirect

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_v3")


SUBJECT_ID_MAP = {"sub-01": 0, "sub-02": 1, "sub-03": 2, "sub-05": 3}


def resolve_paths(cfg, project_root):
    cfg = cfg.copy()
    cfg["_project_root"] = str(project_root)
    cfg["_data_root"] = str(project_root / cfg["data_root"])
    cfg["_fmri_dir"] = str(project_root / cfg["data_root"] / cfg["fmri"]["dir"])
    return cfg


# =============================================================================
# Dataset — loads raw features + ground truth fMRI (no VAE)
# =============================================================================

class BrainFlowV3Dataset(Dataset):
    """Loads raw per-modality features and ground truth fMRI voxels."""

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.subjects = cfg["subjects"]
        self.splits_cfg = cfg.get("splits", {})

        # Directories
        self.features_dir = Path(PROJECT_ROOT) / cfg["raw_features"]["dir"]
        npy_dir = cfg.get("raw_features_npy_dir")
        self.features_npy_dir = Path(PROJECT_ROOT) / npy_dir if npy_dir else None
        self.fmri_dir = Path(cfg["_fmri_dir"])

        # fMRI params
        self.tr = cfg["fmri"]["tr"]
        self.n_voxels = cfg["fmri"]["n_voxels"]
        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end = cfg["fmri"].get("excluded_samples_end", 0)

        # Modality config
        self.modalities = cfg["raw_features"]["modalities"]
        self.layer_aggregation = cfg["raw_features"].get("layer_aggregation", "mean")
        self.feature_freq = cfg["raw_features"].get("sample_freq", 0.6711)

        # Sliding window
        self.context_dur = cfg["sliding_window"]["context_duration"]
        self.stride_dur = cfg["sliding_window"]["stride"]
        self.seq_len = max(1, int(round(self.context_dur / self.tr)))
        self.stride = max(1, int(round(self.stride_dur / self.tr)))

        # HRF delay
        self.hrf_delay = cfg["fmri"].get("hrf_delay", 3)

        # Multi-subject
        self.subject_mode = cfg.get("subject_mode", "single")

        # Cache for fMRI data (must be before _build_index)
        self._fmri_cache = {}

        self.samples = self._build_index()

    def _normalize_clip_name(self, clip_name, task):
        prefix = f"{task}_"
        if clip_name.startswith(prefix):
            return clip_name[len(prefix):]
        return clip_name

    def _build_index(self):
        samples = []
        logger.info("Building %s split index (v3, direct regression)...", self.split)

        # Pre-scan fMRI H5 to get n_trs per clip per subject
        # Only reads shapes, doesn't load data
        fmri_info = {}  # (subject, clip_id) -> n_trs
        for subj in self.subjects:
            for task in self.splits_cfg.keys():
                try:
                    fmri_path = _get_fmri_filepath(str(self.fmri_dir), subj, task)
                except FileNotFoundError:
                    continue
                with h5py.File(fmri_path, "r") as f:
                    for key in f.keys():
                        raw_len = f[key].shape[0]
                        end = raw_len - self.excl_end if self.excl_end > 0 else raw_len
                        n_trs = end - self.excl_start
                        # Extract clip_id: key is like "sub-01_task-s01e01a_..."
                        # The part after "task-" up to the next "_" is the clip_id
                        parts = key.split("task-")
                        if len(parts) > 1:
                            clip_id = parts[1].split("_")[0]
                        else:
                            clip_id = key
                        fmri_info[(subj, clip_id)] = n_trs

        logger.info("Pre-scanned fMRI: %d clips", len(fmri_info))

        for task, splits in self.splits_cfg.items():
            stim_types = splits.get(self.split, [])
            if not stim_types:
                continue

            for stim_type in stim_types:
                first_mod = next(iter(self.modalities))
                mod_cfg = self.modalities[first_mod]
                clip_dir = self.features_dir / mod_cfg["subdir"] / task / stim_type

                if not clip_dir.exists():
                    logger.warning("Feature dir missing: %s", clip_dir)
                    continue

                clip_stems = sorted(p.stem for p in clip_dir.glob("*.h5"))

                for clip_stem in clip_stems:
                    norm_name = self._normalize_clip_name(clip_stem, task)
                    # e.g. norm_name = "s01e01a" or "friends_s01e01a"
                    # clip_id in fmri_info is like "s01e01a"
                    clip_id = norm_name.replace("friends_", "").replace("movie10_", "")

                    for subj in self.subjects:
                        n_trs = fmri_info.get((subj, clip_id))
                        if n_trs is None or n_trs < 10:
                            continue

                        # Sliding windows
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
                                "subject": subj,
                            })

        logger.info("Found %d windows for %s split.", len(samples), self.split)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_fmri_clip(self, subject, task, clip_name):
        """Load ground truth fMRI for a clip (cached, lazy)."""
        cache_key = (subject, task, clip_name)
        if cache_key in self._fmri_cache:
            return self._fmri_cache[cache_key]

        try:
            fmri_path = _get_fmri_filepath(str(self.fmri_dir), subject, task)
        except FileNotFoundError:
            return None

        fmri_key = clip_name
        for prefix in ["friends_", "movie10_"]:
            if fmri_key.startswith(prefix):
                fmri_key = fmri_key[len(prefix):]
                break

        try:
            with h5py.File(fmri_path, "r") as f:
                matched = [k for k in f.keys() if fmri_key in k]
                if not matched:
                    return None
                raw = f[matched[0]]
                end = len(raw) - self.excl_end if self.excl_end > 0 else len(raw)
                data = raw[self.excl_start:end].astype(np.float32)
        except Exception:
            return None

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
        subject = info["subject"]
        end = start + actual_len

        # HRF delay: features at TR t-delay → fMRI at TR t
        feat_start = start - self.hrf_delay
        feat_end = end - self.hrf_delay

        # Load per-modality features
        mod_features = {}
        for mod_name, mod_cfg in self.modalities.items():
            try:
                feat = None
                # Try NPY first (fast path)
                if self.features_npy_dir is not None:
                    npy_path = self.features_npy_dir / mod_name / task / stim_type / f"{norm_name}.npy"
                    if npy_path.exists():
                        feat = np.load(npy_path, mmap_mode="r")[feat_start:feat_end].astype(np.float32)

                # Fallback to H5 (slow path)
                if feat is None:
                    raw = load_feature_clip_perfile(
                        str(self.features_dir), mod_name, mod_cfg,
                        task, stim_type, clip_stem, self.layer_aggregation,
                    )
                    raw = resample_features_to_tr(raw, self.feature_freq, self.tr, n_trs)
                    feat = raw[0].T[feat_start:feat_end].astype(np.float32)

                # Zero-pad if short
                if actual_len < self.seq_len:
                    pad = self.seq_len - actual_len
                    feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant")

                mod_features[mod_name] = torch.from_numpy(feat.copy())
            except (FileNotFoundError, KeyError, ValueError):
                dim = mod_cfg.get("dim", 1024)
                mod_features[mod_name] = torch.zeros(self.seq_len, dim, dtype=torch.float32)

        # Load ground truth fMRI
        fmri_data = self._load_fmri_clip(subject, task, norm_name)
        fmri_window = fmri_data[start:end].astype(np.float32)
        if actual_len < self.seq_len:
            pad = self.seq_len - actual_len
            fmri_window = np.pad(fmri_window, ((0, pad), (0, 0)), mode="constant")

        result = {
            "fmri": torch.from_numpy(fmri_window.copy()),
            "valid_len": actual_len,
            "subject_id": SUBJECT_ID_MAP.get(subject, 0),
        }
        result.update(mod_features)

        return result


def get_dataloaders(cfg):
    train_set = BrainFlowV3Dataset(cfg, split="train")
    val_set = BrainFlowV3Dataset(cfg, split="val")

    dl_cfg = cfg["dataloader"]
    train_loader = DataLoader(
        train_set,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=dl_cfg["val_batch_size"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False,
    )
    return train_loader, val_loader


def pearson_corr_eval(pred, target, valid_lens=None):
    """Compute mean per-voxel PCC for evaluation (not differentiable)."""
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
    return corr.mean().item()


def _extract_modality_features(batch, modality_names, device):
    mod_features = {}
    for mod_name in modality_names:
        if mod_name in batch:
            mod_features[mod_name] = batch[mod_name].to(device)
    return mod_features


def train(args):
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init Data
    train_loader, val_loader = get_dataloaders(cfg)

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 2
        cfg["training"]["val_every_n_epochs"] = 1

    # 2. Init Model
    logger.info("Initializing BrainFlowDirect v3...")
    bf_cfg = cfg["brainflow"]

    modality_dims = {}
    for mod_name, mod_cfg in cfg["raw_features"]["modalities"].items():
        dim = mod_cfg["dim"]
        if cfg["raw_features"].get("layer_aggregation") == "cat":
            n_layers = mod_cfg.get("n_layers", 1)
            dim = dim * n_layers
        modality_dims[mod_name] = dim

    logger.info("Modality dimensions: %s", modality_dims)

    model = BrainFlowDirect(
        modality_dims=modality_dims,
        n_voxels=bf_cfg.get("n_voxels", 1000),
        n_subjects=bf_cfg.get("n_subjects", 4),
        encoder_params=bf_cfg["encoder"],
        head_params=bf_cfg.get("head", {}),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s", model)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # 3. Optimizer & Scheduler
    tr_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
    )

    total_steps = len(train_loader) * tr_cfg["n_epochs"]
    warmup_steps = int(total_steps * tr_cfg.get("warmup_ratio", 0.05))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=tr_cfg.get("use_amp", True))

    modality_names = list(cfg["raw_features"]["modalities"].keys())

    # 4. Resume from checkpoint
    output_dir = Path(PROJECT_ROOT) / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"

    start_epoch = 0
    best_val_pcc = -1.0

    last_ckpt = output_dir / "last.pt"
    if last_ckpt.exists() and not args.fast_dev_run:
        logger.info("Resuming from %s", last_ckpt)
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_pcc = ckpt.get("best_val_pcc", -1.0)

    # 5. Training loop
    pcc_weight = tr_cfg.get("pcc_weight", 0.5)
    pcc_warmup = tr_cfg.get("pcc_warmup_epochs", 10)
    val_every = tr_cfg.get("val_every_n_epochs", 5)
    grad_clip = tr_cfg.get("grad_clip", 1.0)
    use_amp = tr_cfg.get("use_amp", True)

    for epoch in range(start_epoch, tr_cfg["n_epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_pcc_loss = 0.0
        n_batches = 0

        # PCC loss warmup
        current_pcc_weight = pcc_weight if epoch >= pcc_warmup else 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            mod_features = _extract_modality_features(batch, modality_names, device)
            fmri_target = batch["fmri"].to(device)
            valid_lens = batch["valid_len"].to(device)
            subject_ids = batch["subject_id"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(
                    mod_features, subject_ids,
                    target_fmri=fmri_target,
                    valid_lens=valid_lens,
                    pcc_weight=current_pcc_weight,
                )
                loss = result["loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_mse += result["mse_loss"].item()
            if "pcc_loss" in result:
                epoch_pcc_loss += result["pcc_loss"].item()
            n_batches += 1

            if (batch_idx + 1) % tr_cfg.get("log_every_n_steps", 20) == 0:
                avg_loss = epoch_loss / n_batches
                avg_mse = epoch_mse / n_batches
                pbar.set_postfix(loss=f"{avg_loss:.4f}", mse=f"{avg_mse:.4f}",
                               lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mse = epoch_mse / max(n_batches, 1)
        avg_pcc_loss = epoch_pcc_loss / max(n_batches, 1)

        # Validation
        val_pcc = 0.0
        if (epoch + 1) % val_every == 0 or epoch == tr_cfg["n_epochs"] - 1:
            model.eval()
            window_pccs = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Val"):
                    mod_features = _extract_modality_features(batch, modality_names, device)
                    fmri_target = batch["fmri"].to(device)
                    valid_lens = batch["valid_len"].to(device)
                    subject_ids = batch["subject_id"].to(device)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred = model.predict(mod_features, subject_ids)

                    # Compute PCC per-window (not concatenated)
                    for b in range(pred.shape[0]):
                        vl = int(valid_lens[b].item())
                        if vl < 5:
                            continue
                        p = pred[b, :vl]    # (T, V)
                        t = fmri_target[b, :vl]  # (T, V)

                        # Per-voxel PCC within this window
                        p_mean = p.mean(dim=0, keepdim=True)
                        t_mean = t.mean(dim=0, keepdim=True)
                        p_c = p - p_mean
                        t_c = t - t_mean
                        cov = (p_c * t_c).sum(dim=0)
                        std = torch.sqrt((p_c**2).sum(dim=0) * (t_c**2).sum(dim=0) + 1e-8)
                        pcc = cov / std  # (V,)
                        window_pccs.append(pcc.mean().item())

            val_pcc = sum(window_pccs) / max(len(window_pccs), 1)

            logger.info(
                "Epoch %d | loss=%.4f mse=%.4f pcc_loss=%.4f | val_fmri_pcc=%.4f | lr=%.2e",
                epoch + 1, avg_loss, avg_mse, avg_pcc_loss, val_pcc,
                scheduler.get_last_lr()[0],
            )

            # Save best
            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                torch.save(model.state_dict(), output_dir / "best.pt")
                logger.info("★ New best val_fmri_pcc: %.4f", val_pcc)
        else:
            logger.info(
                "Epoch %d | loss=%.4f mse=%.4f pcc_loss=%.4f | lr=%.2e",
                epoch + 1, avg_loss, avg_mse, avg_pcc_loss,
                scheduler.get_last_lr()[0],
            )

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_pcc": best_val_pcc,
        }, output_dir / "last.pt")

        # Log history (only when eval runs)
        if (epoch + 1) % val_every == 0 or epoch == tr_cfg["n_epochs"] - 1:
            write_header = not history_path.exists()
            with open(history_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "epoch", "loss", "mse_loss", "pcc_loss", "val_fmri_pcc", "lr",
                ])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "epoch": epoch + 1,
                    "loss": f"{avg_loss:.6f}",
                    "mse_loss": f"{avg_mse:.6f}",
                    "pcc_loss": f"{avg_pcc_loss:.6f}",
                    "val_fmri_pcc": f"{val_pcc:.6f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.8f}",
                })

    logger.info("Training complete. Best val_fmri_pcc: %.4f", best_val_pcc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    train(args)
