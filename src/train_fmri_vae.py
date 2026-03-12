"""Training script for fMRI VAE (Stage 1).

Trains the subject-aware fMRI_VAE to compress fMRI sequences from all
subjects into a shared latent space. The saved checkpoint (vae_best.pt)
will be used by Stage-2 Flow Matching training.

Usage:
    python src/train_fmri_vae.py --config src/configs/fmri_vae.yaml
    python src/train_fmri_vae.py --config src/configs/fmri_vae.yaml --fast_dev_run
    python src/train_fmri_vae.py --config src/configs/fmri_vae.yaml --resume outputs/fmri_vae/last.pt
"""

import argparse
import csv
import copy
import hashlib
import logging
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_config, resolve_paths
from src.models.brainflow.fmri_vae import build_vae, beta_schedule

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vae_train")


# =============================================================================
# fMRI path helper (mirrors src/data/dataset._get_fmri_filepath)
# =============================================================================

def _fmri_path(fmri_dir: str, subject: str, task: str) -> Path:
    subj_dir  = Path(fmri_dir) / subject / "func"
    atlas     = "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
    stem      = f"{subject}_task-{task}_{atlas}"
    if task == "friends":
        return subj_dir / f"{stem}_desc-s123456_bold.h5"
    else:
        return subj_dir / f"{stem}_bold.h5"


# =============================================================================
# fMRI-only Dataset (no features needed for VAE pretraining)
# =============================================================================

class FmriSequenceDataset(Dataset):
    """Yield non-overlapping fMRI windows of fixed length T_seq.

    Enumerates clips DIRECTLY from fMRI H5 files — does NOT depend on
    feature directories. This allows VAE pretraining even when NPY/H5
    feature files are absent.

    Each sample:
        fmri       : (T_seq, V) float32
        subject_id : int
    """

    def __init__(
        self,
        cfg: dict,
        split: str = "train",
        seq_len: int = 100,
        val_ratio: float = 0.1,
        stride: int | None = None,  # None → non-overlapping (stride=seq_len)
    ):
        self.split      = split
        self.seq_len    = seq_len
        self.n_voxels   = cfg["fmri"]["n_voxels"]
        self.standardize = cfg["preprocessing"]["fmri"]["standardize"] == "zscore_sample"
        self.fmri_dir   = cfg["_fmri_dir"]
        self.subjects   = cfg["subjects"]
        self.subject_to_id = {s: i for i, s in enumerate(self.subjects)}
        self.val_ratio  = val_ratio
        self.splits_cfg = cfg.get("splits", {})
        self.stride     = stride if stride is not None else seq_len  # default non-overlapping
        # Trim first/last N TRs per clip (same as challenge baseline; HRF not stable)
        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end   = cfg["fmri"].get("excluded_samples_end", 0)
        self.max_cache_size = cfg["dataloader"].get("max_cache_size", 50)

        # List of clip metadata dicts
        self._clips: list[dict] = []
        # List of (clip_idx, start_tr) for DataLoader
        self._samples: list[tuple[int, int]] = []
        # In-memory fMRI cache: clip_idx → np.ndarray (T, V)
        from collections import OrderedDict
        self._fmri_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        self._enumerate_from_fmri()

        logger.info(
            "FmriSequenceDataset[%s]: %d windows (seq_len=%d) "
            "from %d clips, %d subjects",
            split, len(self._samples), seq_len,
            len(self._clips), len(self.subjects),
        )

    # ------------------------------------------------------------------
    # Deterministic train/val split
    # ------------------------------------------------------------------
    def _assign_split(self, uid: str) -> str:
        h = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        s = random.Random(h).random()
        return "val" if s >= (1.0 - self.val_ratio) else "train"

    # ------------------------------------------------------------------
    # Enumerate clips from fMRI H5 files
    # ------------------------------------------------------------------
    def _enumerate_from_fmri(self):
        for subject in self.subjects:
            subj_id = self.subject_to_id[subject]

            for task in self.splits_cfg.keys():
                fmri_path = _fmri_path(self.fmri_dir, subject, task)
                if not fmri_path.exists():
                    logger.debug("Not found: %s", fmri_path)
                    continue

                try:
                    with h5py.File(fmri_path, "r") as f:
                        key_shapes = {k: f[k].shape for k in f.keys()}
                except Exception as e:
                    logger.warning("Cannot open %s: %s", fmri_path, e)
                    continue

                for h5_key, shape in key_shapes.items():
                    n_trs_raw = shape[0]   # (n_trs, n_voxels)
                    # Trim head/tail (same as baseline excluded_samples)
                    n_trs = n_trs_raw - self.excl_start - self.excl_end
                    if n_trs < self.seq_len:
                        continue

                    uid = f"{subject}_{task}_{h5_key}"
                    if self._assign_split(uid) != self.split:
                        continue

                    clip_idx = len(self._clips)
                    self._clips.append({
                        "subject":    subject,
                        "subject_id": subj_id,
                        "task":       task,
                        "h5_key":     h5_key,
                        "fmri_path":  str(fmri_path),
                        "n_trs":      n_trs,
                    })

                    # Overlapping / non-overlapping windows
                    for start in range(0, n_trs - self.seq_len + 1, self.stride):
                        self._samples.append((clip_idx, start))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_clip(self, clip_idx: int) -> np.ndarray:
        """Load & cache full fMRI clip → (T_trimmed, V) float32.

        Trims excluded_samples_start and excluded_samples_end from each clip
        (same as challenge baseline).
        """
        if clip_idx in self._fmri_cache:
            self._fmri_cache.move_to_end(clip_idx)
            return self._fmri_cache[clip_idx]

        clip = self._clips[clip_idx]
        with h5py.File(clip["fmri_path"], "r") as f:
            raw = f[clip["h5_key"]]  # lazy
            end = len(raw) - self.excl_end if self.excl_end > 0 else len(raw)
            data = raw[self.excl_start:end].astype(np.float32)  # (T_trimmed, V)

        # Z-score per clip per voxel
        if self.standardize:
            mean = data.mean(axis=0, keepdims=True)
            std  = data.std(axis=0, keepdims=True)
            std  = np.where(std < 1e-8, 1.0, std)
            data = (data - mean) / std
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        self._fmri_cache[clip_idx] = data
        if len(self._fmri_cache) > self.max_cache_size:
            self._fmri_cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        clip_idx, start = self._samples[idx]
        clip  = self._clips[clip_idx]
        data  = self._load_clip(clip_idx)                    # (T_clip, V)
        window = data[start: start + self.seq_len].copy()   # (T_seq, V)

        return {
            "fmri":       torch.from_numpy(window),
            "subject_id": torch.tensor(clip["subject_id"], dtype=torch.long),
        }


# =============================================================================
# DataLoaders
# =============================================================================

def build_vae_dataloaders(cfg: dict) -> dict[str, DataLoader | None]:
    vae_cfg   = cfg["vae"]
    dl_cfg    = cfg["dataloader"]
    seq_len   = vae_cfg.get("seq_len", 100)
    val_ratio = cfg.get("val_ratio", 0.1)
    # Train uses overlapping stride; val always non-overlapping for clean eval
    train_stride = vae_cfg.get("window_stride", seq_len)  # default non-overlapping

    loaders: dict[str, DataLoader | None] = {}
    for split in ("train", "val"):
        stride = train_stride if split == "train" else seq_len
        ds = FmriSequenceDataset(
            cfg, split=split, seq_len=seq_len, val_ratio=val_ratio, stride=stride,
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


# =============================================================================
# EMA
# =============================================================================

@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float = 0.999):
    for p, ep in zip(model.parameters(), ema_model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


# =============================================================================
# Train / Validate
# =============================================================================

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    beta: float,
    grad_clip: float,
    log_every: int,
    epoch: int,
    mask_ratio: float = 0.0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n  = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        fmri       = batch["fmri"].to(device, non_blocking=True)        # (B, T, V)
        subject_id = batch["subject_id"].to(device, non_blocking=True)  # (B,)

        target_fmri = fmri.clone()
        # TR masking augmentation: randomly zero out mask_ratio of time steps
        if mask_ratio > 0.0 and model.training:
            B, T, V = fmri.shape
            mask = torch.rand(B, T, 1, device=device) < mask_ratio  # (B, T, 1)
            fmri = fmri.masked_fill(mask, 0.0)                      # (B, T, V)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            losses = model.loss(fmri, target_fmri, subject_id, beta=beta)

        if use_amp:
            scaler.scale(losses["loss"]).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate any loss key the model returns
        for k, v in losses.items():
            val = v.item() if torch.is_tensor(v) else float(v)
            totals[k] = totals.get(k, 0.0) + val
        n += 1

        if log_every > 0 and (i + 1) % log_every == 0:
            avgs = {k: v / n for k, v in totals.items()}
            lr   = optimizer.param_groups[0]["lr"]
            spd  = n / (time.time() - t0)
            pcc_str = f" pcc_loss {avgs.get('spatial_pcc', 0):.4f}" if "spatial_pcc" in avgs else ""
            logger.info(
                "  Ep %d | Batch %d/%d | Loss %.4f (recon %.4f kl %.4f%s) "
                "| β=%.5f | LR %.2e | %.1fb/s",
                epoch, i + 1, len(loader),
                avgs["loss"], avgs.get("recon", 0), avgs.get("kl", 0), pcc_str,
                beta, lr, spd,
            )

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    beta: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    # Store per-batch recon/true for per-batch Pearson (avoid cross-clip concat)
    batch_pearsons = []

    for batch in loader:
        fmri       = batch["fmri"].to(device, non_blocking=True)
        subject_id = batch["subject_id"].to(device, non_blocking=True)

        target_fmri = fmri.clone()
        with autocast(enabled=use_amp):
            losses     = model.loss(fmri, target_fmri, subject_id, beta=beta)
            recon, *_  = model(fmri, subject_id)

        for k, v in losses.items():
            val = v.item() if torch.is_tensor(v) else float(v)
            totals[k] = totals.get(k, 0.0) + val
        n += 1

        # Per-voxel Pearson within this batch (no cross-clip contamination)
        recon_np = recon.cpu().float().numpy().reshape(-1, model.n_voxels)
        true_np  = fmri.cpu().float().numpy().reshape(-1, model.n_voxels)
        cors = []
        for v in range(model.n_voxels):
            r, _ = pearsonr(true_np[:, v], recon_np[:, v])
            if np.isfinite(r):
                cors.append(r)
        if cors:
            batch_pearsons.append(float(np.mean(cors)))

    metrics = {k: v / max(n, 1) for k, v in totals.items()}
    metrics["pearson"] = float(np.mean(batch_pearsons)) if batch_pearsons else 0.0

    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="fMRI VAE training (Stage 1)")
    parser.add_argument("--config",       type=str, required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume",       type=str, default=None)
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    cfg       = load_config(args.config)
    cfg       = resolve_paths(cfg, PROJECT_ROOT)
    vae_cfg   = cfg["vae"]
    train_cfg = cfg["training"]
    dl_cfg    = cfg["dataloader"]

    output_dir = Path(PROJECT_ROOT) / cfg.get("vae_output_dir", "outputs/fmri_vae")
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── DataLoaders ─────────────────────────────────────────────────────────
    logger.info("Building dataloaders (from fMRI H5 files directly)...")
    loaders      = build_vae_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    if train_loader is None:
        raise RuntimeError("No training data found! Check fmri.dir in config.")

    logger.info("Train: %d batches | Val: %s batches",
                len(train_loader),
                len(val_loader) if val_loader else "N/A")

    # ── Model ───────────────────────────────────────────────────────────────
    logger.info("Building VAE (version=%d)...", vae_cfg.get("version", 1))
    model     = build_vae(cfg).to(device)
    ema_model = copy.deepcopy(model)
    logger.info("%s", model)

    # ── Optimizer / Scheduler ───────────────────────────────────────────────
    n_epochs = train_cfg["n_epochs"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    total_steps = n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["lr"],
        total_steps=total_steps,
        pct_start=train_cfg.get("warmup_ratio", 0.05),
        anneal_strategy="cos",
    )
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_pearson = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_pearson = ckpt.get("val_pearson", -1.0)
        logger.info("Resumed from epoch %d", start_epoch)

    # ── History CSV ─────────────────────────────────────────────────────────
    history_path   = output_dir / "history.csv"
    history_fields = ["epoch", "train_loss", "train_recon", "train_kl",
                      "val_loss", "val_recon", "val_kl", "val_pearson",
                      "beta", "lr", "time_s"]
    if not history_path.exists() or start_epoch == 0:
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, history_fields).writeheader()

    # ── Fast dev run ─────────────────────────────────────────────────────────
    if args.fast_dev_run:
        n_epochs = 1
        logger.info("=== FAST DEV RUN ===")

    # ── Hyperparams ───────────────────────────────────────────────────────────
    mask_ratio  = vae_cfg.get("mask_ratio", 0.0)
    beta_max    = vae_cfg.get("beta_max", 0.01)
    beta_warmup = vae_cfg.get("beta_warmup_epochs", 20)
    val_every   = train_cfg.get("val_every_n_epochs", 5)
    log_every   = train_cfg.get("log_every_n_steps", 50)
    grad_clip   = train_cfg.get("grad_clip", 1.0)
    ema_decay   = train_cfg.get("ema_decay", 0.999)

    logger.info("=" * 60)
    logger.info("fMRI VAE: %d epochs | latent_dim=%d | β_max=%.4f",
                n_epochs, vae_cfg["latent_dim"], beta_max)
    logger.info("=" * 60)

    for epoch in range(start_epoch, n_epochs):
        t_epoch = time.time()
        beta = beta_schedule(epoch, beta_max=beta_max, warmup_epochs=beta_warmup)

        # Tell model which epoch we're in (for PCC warmstart logic)
        if hasattr(model, '_current_epoch'):
            model._current_epoch = epoch

        # ── Train ─────────────────────────────────────────────────────────────
        loader_iter = _limit_batches(train_loader, 2) if args.fast_dev_run else train_loader
        train_m = train_one_epoch(
            model, loader_iter, optimizer, scheduler,
            device, scaler, use_amp, beta,
            grad_clip, log_every if not args.fast_dev_run else 1, epoch + 1,
            mask_ratio=mask_ratio,
        )
        ema_update(model, ema_model, ema_decay)

        # ── Validate ───────────────────────────────────────────────────────────
        val_m = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "pearson": 0.0}
        do_val = ((epoch + 1) % val_every == 0) or args.fast_dev_run
        if do_val and val_loader:
            v_iter = _limit_batches(val_loader, 2) if args.fast_dev_run else val_loader
            val_m  = validate(ema_model, v_iter, device, use_amp, beta)

        elapsed = time.time() - t_epoch
        lr      = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d | β=%.5f | "
            "Train [loss=%.4f recon=%.4f kl=%.4f] | "
            "Val [loss=%.4f recon=%.4f kl=%.4f pearson=%.4f] | "
            "LR=%.2e | %.1fs",
            epoch + 1, n_epochs, beta,
            train_m["loss"], train_m["recon"], train_m["kl"],
            val_m["loss"],   val_m["recon"],   val_m["kl"],
            val_m["pearson"], lr, elapsed,
        )

        # ── History / Checkpoint ──────────────────────────────────────────────
        if do_val:
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, history_fields).writerow({
                    "epoch":       epoch + 1,
                    "train_loss":  f"{train_m['loss']:.6f}",
                    "train_recon": f"{train_m['recon']:.6f}",
                    "train_kl":    f"{train_m['kl']:.6f}",
                    "val_loss":    f"{val_m['loss']:.6f}",
                    "val_recon":   f"{val_m['recon']:.6f}",
                    "val_kl":      f"{val_m['kl']:.6f}",
                    "val_pearson": f"{val_m['pearson']:.6f}",
                    "beta":        f"{beta:.5f}",
                    "lr":          f"{lr:.2e}",
                    "time_s":      f"{elapsed:.1f}",
                })

        if do_val and not args.fast_dev_run:
            ckpt = {
                "epoch":       epoch,
                "model":       model.state_dict(),
                "ema_model":   ema_model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "val_loss":    val_m["loss"],
                "val_pearson": val_m["pearson"],
                "config":      cfg,
            }
            torch.save(ckpt, output_dir / "last.pt")
            if val_m["pearson"] > best_pearson:
                best_pearson = val_m["pearson"]
                torch.save(ckpt, output_dir / "best.pt")
                logger.info("  → New best Pearson: %.4f", best_pearson)

    logger.info("Done! Best Pearson: %.4f | Saved to: %s", best_pearson, output_dir)


def _limit_batches(loader, n: int):
    class _L:
        def __iter__(self_):
            for i, b in enumerate(loader):
                if i >= n: break
                yield b
        def __len__(self_): return min(n, len(loader))
    return _L()


if __name__ == "__main__":
    main()
