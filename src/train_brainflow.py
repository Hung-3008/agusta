"""BrainFlow training script.

Uses torchcfm's ExactOptimalTransportConditionalFlowMatcher for CFM training
and the existing SlidingWindowDataset for data loading.

Usage:
    python src/train_brainflow.py --config src/configs/brainflow.yaml
    python src/train_brainflow.py --config src/configs/brainflow.yaml --fast_dev_run
"""

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import pearsonr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Add torchcfm to path
TORCHCFM_PATH = PROJECT_ROOT / "Data" / "conditional-flow-matching"
sys.path.insert(0, str(TORCHCFM_PATH))

from src.data.dataset import build_dataloaders, load_config, resolve_paths, preload_datasets_to_ram
from src.models.brainflow.brain_flow import BrainFlowCFM
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brainflow")


# =============================================================================
#  EMA
# =============================================================================

@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float = 0.999):
    """Exponential moving average update."""
    for p, ep in zip(model.parameters(), ema_model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


# =============================================================================
# Build CFM
# =============================================================================

def build_cfm(method: str, sigma: float):
    """Build a Conditional Flow Matcher from config."""
    if method == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif method == "icfm":
        return ConditionalFlowMatcher(sigma=sigma)
    elif method == "fm":
        return TargetConditionalFlowMatcher(sigma=sigma)
    elif method == "si":
        return VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError(f"Unknown CFM method: {method}")


# =============================================================================
# Training step
# =============================================================================

def train_one_epoch(
    model: BrainFlowCFM,
    dataloader,
    optimizer,
    scheduler,
    fm,
    device,
    scaler,
    use_amp: bool,
    grad_clip: float,
    log_every: int,
    epoch: int,
):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        features = {k: v.to(device, non_blocking=True) for k, v in batch["features"].items()}
        fmri = batch["fmri"].to(device, non_blocking=True)        # (B, V)
        subject_id = batch["subject_id"].to(device, non_blocking=True)  # (B,)

        # CFM: sample time, noisy point, and target velocity
        x1 = fmri.float()           # target fMRI
        x0 = torch.randn_like(x1)   # Gaussian noise
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # Predict velocity
            vt = model(features, t, xt, subject_id)
            loss = nn.functional.mse_loss(vt, ut)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            avg = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            speed = n_batches / elapsed
            logger.info(
                "  Epoch %d | Batch %d/%d | Loss %.6f | LR %.2e | %.1f batch/s",
                epoch, batch_idx + 1, len(dataloader), avg, lr, speed,
            )

    return total_loss / max(n_batches, 1)


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(
    model: BrainFlowCFM,
    dataloader,
    fm,
    device,
    use_amp: bool,
    n_sample_steps: int = 50,
    sample_method: str = "euler",
):
    """Validate: compute CFM loss + Pearson correlation on generated samples."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_trues = []

    for batch in dataloader:
        features = {k: v.to(device, non_blocking=True) for k, v in batch["features"].items()}
        fmri = batch["fmri"].to(device, non_blocking=True)
        subject_id = batch["subject_id"].to(device, non_blocking=True)

        x1 = fmri.float()
        x0 = torch.randn_like(x1)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)

        with autocast(enabled=use_amp):
            vt = model(features, t, xt, subject_id)
            loss = nn.functional.mse_loss(vt, ut)

        total_loss += loss.item()
        n_batches += 1

        # Generate samples for Pearson (every 5th batch to save time)
        if n_batches % 5 == 1:
            y_pred = model.sample(
                features, subject_id,
                n_steps=n_sample_steps, method=sample_method,
            )
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(fmri.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    # Compute Pearson correlation
    pearson = 0.0
    if all_preds:
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        # Per-voxel Pearson, averaged
        n_voxels = trues.shape[1]
        cors = []
        for v in range(n_voxels):
            r, _ = pearsonr(trues[:, v], preds[:, v])
            if np.isfinite(r):
                cors.append(r)
        pearson = np.mean(cors) if cors else 0.0

    return avg_loss, pearson


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BrainFlow training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches only (smoke test)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    # Output dir
    output_dir = Path(PROJECT_ROOT) / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import yaml
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build dataloaders
    logger.info("Building dataloaders...")
    loaders = build_dataloaders(cfg, project_root=PROJECT_ROOT, splits=["train", "val"])

    if train_cfg.get("preload_to_ram", False):
        logger.info("Preloading all data to RAM...")
        preload_datasets_to_ram(loaders)

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None:
        raise RuntimeError("No training data found!")

    logger.info("Train: %d samples, Val: %d samples",
                len(train_loader.dataset),
                len(val_loader.dataset) if val_loader else 0)

    # Build model
    logger.info("Building BrainFlow model...")
    model = BrainFlowCFM(**model_cfg).to(device)
    ema_model = copy.deepcopy(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total params: %.2fM | Trainable: %.2fM",
                total_params / 1e6, trainable_params / 1e6)

    # Build CFM
    fm = build_cfm(train_cfg["cfm_method"], train_cfg["cfm_sigma"])
    logger.info("CFM method: %s (sigma=%.4f)", train_cfg["cfm_method"], train_cfg["cfm_sigma"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler: OneCycleLR
    n_epochs = train_cfg["n_epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = n_epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["lr"],
        total_steps=total_steps,
        pct_start=train_cfg["warmup_ratio"],
        anneal_strategy="cos",
    )

    # AMP scaler
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # History
    history_path = output_dir / "history.csv"
    history_fields = ["epoch", "train_loss", "val_loss", "val_pearson", "lr", "time_s"]
    if not history_path.exists() or start_epoch == 0:
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, history_fields).writeheader()

    # Fast dev run
    if args.fast_dev_run:
        n_epochs = 1
        train_cfg["log_every_n_steps"] = 1
        logger.info("=== FAST DEV RUN (2 batches) ===")

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting BrainFlow training: %d epochs", n_epochs)
    logger.info("=" * 60)

    best_val_pearson = -1.0

    for epoch in range(start_epoch, n_epochs):
        t_epoch = time.time()

        # Train
        if args.fast_dev_run:
            # Only 2 batches
            mini_loader = _limit_batches(train_loader, 2)
            train_loss = train_one_epoch(
                model, mini_loader, optimizer, scheduler, fm, device, scaler,
                use_amp, train_cfg["grad_clip"], 1, epoch,
            )
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, fm, device, scaler,
                use_amp, train_cfg["grad_clip"], train_cfg["log_every_n_steps"], epoch,
            )

        # EMA update
        ema_update(model, ema_model, train_cfg["ema_decay"])

        # Validation
        val_loss, val_pearson = 0.0, 0.0
        if val_loader and (epoch + 1) % train_cfg["val_every_n_epochs"] == 0:
            if args.fast_dev_run:
                mini_val = _limit_batches(val_loader, 2)
                val_loss, val_pearson = validate(
                    ema_model, mini_val, fm, device, use_amp,
                    train_cfg["n_sample_steps"], train_cfg["sample_method"],
                )
            else:
                val_loss, val_pearson = validate(
                    ema_model, val_loader, fm, device, use_amp,
                    train_cfg["n_sample_steps"], train_cfg["sample_method"],
                )

        elapsed = time.time() - t_epoch
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | "
            "Val Pearson: %.4f | LR: %.2e | Time: %.1fs",
            epoch + 1, n_epochs, train_loss, val_loss, val_pearson, lr, elapsed,
        )

        # Save history
        with open(history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, history_fields)
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_pearson": f"{val_pearson:.6f}",
                "lr": f"{lr:.2e}",
                "time_s": f"{elapsed:.1f}",
            })

        # Save checkpoint
        if not args.fast_dev_run:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pearson": val_pearson,
            }
            torch.save(ckpt, output_dir / "last.pt")

            if val_pearson > best_val_pearson:
                best_val_pearson = val_pearson
                torch.save(ckpt, output_dir / "best.pt")
                logger.info("  → New best val Pearson: %.4f", val_pearson)

    logger.info("=" * 60)
    logger.info("Training complete! Best Val Pearson: %.4f", best_val_pearson)
    logger.info("Outputs saved to: %s", output_dir)


def _limit_batches(loader, n):
    """Helper to yield only first n batches for fast_dev_run."""
    class _Limited:
        def __init__(self, loader, n):
            self.loader = loader
            self.n = n
        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.n:
                    break
                yield batch
        def __len__(self):
            return min(self.n, len(self.loader))
    return _Limited(loader, n)


if __name__ == "__main__":
    main()
