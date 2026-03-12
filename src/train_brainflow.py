"""BrainFlow V2 training script.

Training objective:
    x0 = encoder(features)   [deterministic prior from stimulus]
    x1 = true fMRI
    t, xt, ut = OT-CFM(x0, x1)   [ut = x1 - x0, deterministic!]

    loss = MSE(v_θ(xt, t, subject_id, x0), ut)   [CFM loss]
         + λ * MSE(x0, x1)                        [prior regression loss]

Usage:
    python src/train_brainflow.py --config src/configs/brainflow.yaml
    python src/train_brainflow.py --config src/configs/brainflow.yaml --fast_dev_run
"""

import argparse
import copy
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TORCHCFM_PATH = PROJECT_ROOT / "Data" / "conditional-flow-matching"
sys.path.insert(0, str(TORCHCFM_PATH))

from src.data.algonauts_dataset import build_dataloaders, load_config, resolve_paths
from src.models.brainflow.brain_flow import BrainFlowV2
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brainflow_v2")


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float = 0.999):
    for p, ep in zip(model.parameters(), ema_model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ─────────────────────────────────────────────────────────────────────────────
# CFM builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cfm(method: str, sigma: float):
    if method == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif method == "icfm":
        return ConditionalFlowMatcher(sigma=sigma)
    elif method == "fm":
        return TargetConditionalFlowMatcher(sigma=sigma)
    elif method == "si":
        return VariancePreservingConditionalFlowMatcher(sigma=sigma)
    raise ValueError(f"Unknown CFM method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: BrainFlowV2,
    dataloader,
    optimizer,
    scheduler,
    fm,
    device,
    scaler,
    use_amp: bool,
    grad_clip: float,
    lambda_prior: float,
    log_every: int,
    epoch: int,
):
    model.train()
    total_loss = total_cfm = total_prior = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        features = {k: v.to(device, non_blocking=True) for k, v in batch["features"].items()}
        fmri = batch["fmri"].to(device, non_blocking=True).float()       # (B, 1000)
        subject_id = batch["subject_id"].to(device, non_blocking=True)   # (B,)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # ── Compute x0 (deterministic prior) ──────────────────────────────
            x0 = model.prior_encoder(features)   # (B, 1000)
            x1 = fmri

            # ── TÁCH BACKGROUND CỦA GRADIENT CFM MỚI ─────────────────────────
            # Prevent gradient flow from Flow Matching back into the PriorEncoder
            x0_detached = x0.detach()
            
            # ── OT-CFM: interpolate x0 → x1 ──────────────────────────────────
            # ut = x1 - xt_dot ← deterministic because x0 is deterministic
            t, xt, ut = fm.sample_location_and_conditional_flow(x0_detached, x1)

            # ── Velocity prediction ───────────────────────────────────────────
            vt = model.velocity_refiner(xt, t, subject_id, x0=x0_detached)

            # ── Losses ────────────────────────────────────────────────────────
            cfm_loss   = F.mse_loss(vt, ut)
            prior_loss = F.mse_loss(x0, x1)           # encoder regression
            loss = cfm_loss + lambda_prior * prior_loss

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

        total_loss  += loss.item()
        total_cfm   += cfm_loss.item()
        total_prior += prior_loss.item()
        n_batches += 1

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            speed = n_batches / (time.time() - t0)
            logger.info(
                "  Ep%d | B%d/%d | loss=%.4f cfm=%.4f prior=%.4f | lr=%.1e | %.1f b/s",
                epoch, batch_idx + 1, len(dataloader),
                total_loss / n_batches,
                total_cfm / n_batches,
                total_prior / n_batches,
                lr, speed,
            )

    nb = max(n_batches, 1)
    return total_loss / nb, total_cfm / nb, total_prior / nb


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: BrainFlowV2,
    dataloader,
    fm,
    device,
    use_amp: bool,
    lambda_prior: float,
    n_sample_steps: int = 20,
    sample_method: str = "euler",
):
    model.eval()
    total_loss = total_cfm = total_prior = 0.0
    n_batches = 0
    all_preds, all_trues = [], []

    for batch in dataloader:
        features    = {k: v.to(device, non_blocking=True) for k, v in batch["features"].items()}
        fmri        = batch["fmri"].to(device, non_blocking=True).float()
        subject_id  = batch["subject_id"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            x0 = model.prior_encoder(features)
            x1 = fmri
            
            x0_detached = x0.detach()
            
            t, xt, ut = fm.sample_location_and_conditional_flow(x0_detached, x1)
            vt = model.velocity_refiner(xt, t, subject_id, x0=x0_detached)

            cfm_loss   = F.mse_loss(vt, ut)
            prior_loss = F.mse_loss(x0, x1)
            loss = cfm_loss + lambda_prior * prior_loss

        total_loss  += loss.item()
        total_cfm   += cfm_loss.item()
        total_prior += prior_loss.item()
        n_batches += 1

        # Sample every 5th batch for Pearson metric
        if n_batches % 5 == 1:
            y_pred = model.sample(
                features, subject_id,
                n_steps=n_sample_steps, method=sample_method,
            )
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(fmri.cpu().numpy())

    nb = max(n_batches, 1)
    avg_loss  = total_loss / nb
    avg_prior = total_prior / nb

    # Pearson correlation (per voxel, averaged)
    pearson = 0.0
    if all_preds:
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        cors = []
        for v in range(trues.shape[1]):
            r, _ = pearsonr(trues[:, v], preds[:, v])
            if np.isfinite(r):
                cors.append(r)
        pearson = float(np.mean(cors)) if cors else 0.0

    return avg_loss, avg_prior, pearson


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BrainFlow V2 training")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)
    train_cfg  = cfg["training"]
    model_cfg  = cfg["model"]

    output_dir = Path(PROJECT_ROOT) / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Building dataloaders...")
    loaders = build_dataloaders(cfg, project_root=PROJECT_ROOT, splits=["train", "val"])
    train_loader = loaders.get("train")
    val_loader   = loaders.get("val")
    if train_loader is None:
        raise RuntimeError("No training data!")
    logger.info(
        "Train: %d samples | Val: %d samples",
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader else 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building BrainFlow V2...")
    model     = BrainFlowV2(**model_cfg).to(device)
    ema_model = copy.deepcopy(model)

    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Params: %.2fM total | %.2fM trainable", total_p / 1e6, trainable / 1e6)

    # ── CFM & Optimizer ───────────────────────────────────────────────────────
    fm = build_cfm(train_cfg["cfm_method"], train_cfg["cfm_sigma"])
    logger.info("CFM: %s (σ=%.4f)", train_cfg["cfm_method"], train_cfg["cfm_sigma"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    n_epochs       = train_cfg["n_epochs"]
    steps_per_epoch = len(train_loader)
    total_steps    = n_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["lr"],
        total_steps=total_steps,
        pct_start=train_cfg["warmup_ratio"],
        anneal_strategy="cos",
    )

    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)
    lambda_prior = train_cfg.get("lambda_prior", 0.5)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # ── History CSV ───────────────────────────────────────────────────────────
    history_path = output_dir / "history.csv"
    history_fields = ["epoch", "train_loss", "train_cfm", "train_prior",
                      "val_loss", "val_prior", "val_pearson", "lr", "time_s"]
    if not history_path.exists() or start_epoch == 0:
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, history_fields).writeheader()

    if args.fast_dev_run:
        n_epochs = 1
        train_cfg["log_every_n_steps"] = 1
        logger.info("=== FAST DEV RUN ===")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BrainFlow V2 training: %d epochs", n_epochs)
    logger.info("=" * 60)
    best_pearson = -1.0

    for epoch in range(start_epoch, n_epochs):
        t_epoch = time.time()

        # Train
        loader = _limit_batches(train_loader, 2) if args.fast_dev_run else train_loader
        train_loss, train_cfm, train_prior = train_one_epoch(
            model, loader, optimizer, scheduler, fm, device, scaler,
            use_amp, train_cfg["grad_clip"], lambda_prior,
            train_cfg["log_every_n_steps"], epoch,
        )

        ema_update(model, ema_model, train_cfg["ema_decay"])

        # Validation
        val_loss = val_prior = val_pearson = 0.0
        if val_loader and (epoch + 1) % train_cfg["val_every_n_epochs"] == 0:
            vl = _limit_batches(val_loader, 2) if args.fast_dev_run else val_loader
            val_loss, val_prior, val_pearson = validate(
                ema_model, vl, fm, device, use_amp, lambda_prior,
                train_cfg["n_sample_steps"], train_cfg["sample_method"],
            )

        elapsed = time.time() - t_epoch
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Ep %d/%d | tr_loss=%.4f cfm=%.4f prior=%.4f | val_loss=%.4f prior=%.4f pearson=%.4f | lr=%.1e | %.1fs",
            epoch + 1, n_epochs,
            train_loss, train_cfm, train_prior,
            val_loss, val_prior, val_pearson,
            lr, elapsed,
        )

        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, history_fields).writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.6f}",
                "train_cfm": f"{train_cfm:.6f}",
                "train_prior": f"{train_prior:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_prior": f"{val_prior:.6f}",
                "val_pearson": f"{val_pearson:.6f}",
                "lr": f"{lr:.2e}",
                "time_s": f"{elapsed:.1f}",
            })

        if not args.fast_dev_run:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_pearson": val_pearson,
            }
            torch.save(ckpt, output_dir / "last.pt")
            if val_pearson > best_pearson:
                best_pearson = val_pearson
                torch.save(ckpt, output_dir / "best.pt")
                logger.info("  → New best Pearson: %.4f", best_pearson)

    logger.info("=" * 60)
    logger.info("Done! Best Pearson: %.4f | Output: %s", best_pearson, output_dir)


def _limit_batches(loader, n: int):
    class _L:
        def __iter__(self):
            for i, b in enumerate(loader):
                if i >= n: break
                yield b
        def __len__(self): return min(n, len(loader))
    return _L()


if __name__ == "__main__":
    main()
