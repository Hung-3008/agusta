"""BrainFlow training script.

Latent-space Conditional Flow Matching:
  - Load precomputed PCA features
  - Extract VAE latents online (with num_trial averaging)
  - Train velocity transformer to map noise → latents
  - Validate with frozen VAE decoder (latent → fMRI → Pearson)

Usage:
    python src/train_brain_flow.py --config src/configs/brain_flow.yaml
    python src/train_brain_flow.py --config src/configs/brain_flow.yaml --fast_dev_run
"""

import argparse
import copy
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# torchcfm
TORCHCFM_PATH = PROJECT_ROOT / "Data" / "conditional-flow-matching"
sys.path.insert(0, str(TORCHCFM_PATH))

from src.data.dataset import load_config, resolve_paths
from src.data.brainflow_dataset import BrainFlowDataset, build_brainflow_dataloaders
from src.models.brainflow.brain_flow import BrainFlowCFM
from src.models.brainflow.fmri_vae import build_vae
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
logger = logging.getLogger("brain_flow")


# =============================================================================
# Helpers
# =============================================================================

@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float = 0.999):
    for p, ep in zip(model.parameters(), ema_model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


def build_cfm(method: str, sigma: float):
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


def load_vae_decoder(cfg: dict, device: torch.device):
    """Load pretrained VAE and return it in eval mode (frozen)."""
    vae_ckpt_path = cfg.get("vae_checkpoint")
    if not vae_ckpt_path:
        logger.warning("No vae_checkpoint specified, VAE decoder not loaded")
        return None

    vae_ckpt_path = Path(PROJECT_ROOT) / vae_ckpt_path
    if not vae_ckpt_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")

    # Load VAE config from checkpoint
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_cfg = ckpt.get("config", cfg)

    # Build VAE model
    vae = build_vae(vae_cfg).to(device)

    # Load weights (prefer ema_model if available)
    if "ema_model" in ckpt:
        vae.load_state_dict(ckpt["ema_model"])
        logger.info("Loaded VAE EMA weights from %s", vae_ckpt_path)
    elif "model" in ckpt:
        vae.load_state_dict(ckpt["model"])
        logger.info("Loaded VAE model weights from %s", vae_ckpt_path)

    # Freeze
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    return vae


# =============================================================================
# Training step
# =============================================================================

def train_one_epoch(
    model: BrainFlowCFM,
    dataloader,
    optimizer,
    scheduler,
    sigma: float,
    device,
    scaler,
    use_amp: bool,
    grad_clip: float,
    log_every: int,
    epoch: int,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        pca_features = batch["pca_features"].to(device, non_blocking=True)  # (B, T, D)
        latent = batch["latent"].to(device, non_blocking=True)  # (B, T, Z)
        subject_id = batch["subject_id"].to(device, non_blocking=True)  # (B,)

        B, T, Z = latent.shape

        # Sample 1 flow time per sequence (consistent across all TRs)
        t_flow = torch.rand(B, device=device)  # (B,)
        t_expand = t_flow.view(B, 1, 1)  # (B, 1, 1) for broadcasting

        # Compute interpolant directly (ICFM: xt = (1-(1-σ)t)·x0 + t·x1)
        x0 = torch.randn_like(latent)  # (B, T, Z)
        x1 = latent.float()
        xt = (1 - (1 - sigma) * t_expand) * x0 + t_expand * x1  # (B, T, Z)
        ut = x1 - (1 - sigma) * x0  # (B, T, Z) — target velocity

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            vt = model(pca_features, t_flow, xt, subject_id)  # (B, T, Z)
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
    sigma: float,
    device,
    use_amp: bool,
    vae_decoder=None,
    n_sample_steps: int = 20,
    sample_method: str = "euler",
):
    """Validate: CFM loss + Pearson in latent space + optionally in voxel space."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_pred_latents = []
    all_true_latents = []
    all_pred_fmri = []
    all_true_fmri = []

    for batch in dataloader:
        pca_features = batch["pca_features"].to(device, non_blocking=True)
        latent = batch["latent"].to(device, non_blocking=True)
        fmri = batch["fmri"].to(device, non_blocking=True)
        subject_id = batch["subject_id"].to(device, non_blocking=True)

        B, T, Z = latent.shape

        # CFM loss — consistent t per sequence
        t_flow = torch.rand(B, device=device)
        t_expand = t_flow.view(B, 1, 1)
        x0 = torch.randn_like(latent)
        x1 = latent.float()
        xt = (1 - (1 - sigma) * t_expand) * x0 + t_expand * x1
        ut = x1 - (1 - sigma) * x0

        with autocast(enabled=use_amp):
            vt = model(pca_features, t_flow, xt, subject_id)
            loss = nn.functional.mse_loss(vt, ut)

        total_loss += loss.item()
        n_batches += 1

        # Generate samples for Pearson (every 5th batch)
        if n_batches % 5 == 1:
            z_pred = model.sample(
                pca_features, subject_id,
                n_steps=n_sample_steps, method=sample_method,
            )  # (B, T, Z)

            all_pred_latents.append(z_pred.cpu().numpy().reshape(-1, Z))
            all_true_latents.append(latent.cpu().numpy().reshape(-1, Z))

            # Decode to fMRI if VAE decoder available
            if vae_decoder is not None:
                # VAE decoder expects (B, T, Z) and subject_id (B,)
                # Handle case where VAE has no subject conditioning
                fmri_pred = vae_decoder.decode(z_pred, subject_id)  # (B, T, V)
                all_pred_fmri.append(fmri_pred.cpu().numpy().reshape(-1, fmri.shape[-1]))
                all_true_fmri.append(fmri.cpu().numpy().reshape(-1, fmri.shape[-1]))

    avg_loss = total_loss / max(n_batches, 1)

    # Latent-space Pearson
    latent_pearson = 0.0
    if all_pred_latents:
        preds = np.concatenate(all_pred_latents, axis=0)
        trues = np.concatenate(all_true_latents, axis=0)
        n_dims = trues.shape[1]
        cors = []
        for d in range(min(n_dims, 50)):  # Sample 50 dims for speed
            r, _ = pearsonr(trues[:, d], preds[:, d])
            if np.isfinite(r):
                cors.append(r)
        latent_pearson = np.mean(cors) if cors else 0.0

    # Voxel-space Pearson
    voxel_pearson = 0.0
    if all_pred_fmri:
        preds = np.concatenate(all_pred_fmri, axis=0)
        trues = np.concatenate(all_true_fmri, axis=0)
        n_voxels = trues.shape[1]
        cors = []
        for v in range(n_voxels):
            r, _ = pearsonr(trues[:, v], preds[:, v])
            if np.isfinite(r):
                cors.append(r)
        voxel_pearson = np.mean(cors) if cors else 0.0

    return avg_loss, latent_pearson, voxel_pearson


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BrainFlow training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    output_dir = Path(PROJECT_ROOT) / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load VAE for latent extraction + decoding ──
    logger.info("Loading pretrained VAE...")
    vae = load_vae_decoder(cfg, device)

    # ── Build dataloaders ──
    logger.info("Building dataloaders...")
    bf_cfg = cfg.get("brainflow", {})
    num_trial = bf_cfg.get("num_trial", 1)

    loaders = build_brainflow_dataloaders(
        cfg, vae_model=vae, num_trial=num_trial,
    )
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None:
        raise RuntimeError("No training data found!")

    logger.info("Train: %d samples, Val: %d samples",
                len(train_loader.dataset),
                len(val_loader.dataset) if val_loader else 0)

    # ── Build model ──
    logger.info("Building BrainFlow model...")
    model = BrainFlowCFM(**model_cfg).to(device)
    ema_model = copy.deepcopy(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s", model)
    logger.info("Total params: %.2fM | Trainable: %.2fM",
                total_params / 1e6, trainable_params / 1e6)

    # ── Build CFM ──
    fm = build_cfm(train_cfg["cfm_method"], train_cfg["cfm_sigma"])
    logger.info("CFM method: %s (sigma=%.4f)", train_cfg["cfm_method"], train_cfg["cfm_sigma"])

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

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

    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # ── Resume ──
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # ── History ──
    history_path = output_dir / "history.csv"
    history_fields = [
        "epoch", "train_loss", "val_loss",
        "val_latent_pearson", "val_voxel_pearson",
        "lr", "time_s",
    ]
    if not history_path.exists() or start_epoch == 0:
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, history_fields).writeheader()

    # ── Fast dev run ──
    if args.fast_dev_run:
        n_epochs = 1
        train_cfg["log_every_n_steps"] = 1
        logger.info("=== FAST DEV RUN (2 batches) ===")

    # ── Training loop ──
    logger.info("=" * 60)
    logger.info("Starting BrainFlow training: %d epochs", n_epochs)
    logger.info("=" * 60)

    best_val_pearson = -1.0

    for epoch in range(start_epoch, n_epochs):
        t_epoch = time.time()

        # Train
        if args.fast_dev_run:
            mini_loader = _limit_batches(train_loader, 2)
            train_loss = train_one_epoch(
                model, mini_loader, optimizer, scheduler,
                train_cfg["cfm_sigma"], device, scaler,
                use_amp, train_cfg["grad_clip"], 1, epoch + 1,
            )
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                train_cfg["cfm_sigma"], device, scaler,
                use_amp, train_cfg["grad_clip"], train_cfg["log_every_n_steps"], epoch + 1,
            )

        ema_update(model, ema_model, train_cfg["ema_decay"])

        # Validate
        did_val = False
        val_loss, val_latent_pearson, val_voxel_pearson = 0.0, 0.0, 0.0
        if val_loader and (epoch + 1) % train_cfg["val_every_n_epochs"] == 0:
            did_val = True
            if args.fast_dev_run:
                mini_val = _limit_batches(val_loader, 2)
                val_loss, val_latent_pearson, val_voxel_pearson = validate(
                    ema_model, mini_val, train_cfg["cfm_sigma"], device, use_amp,
                    vae_decoder=vae,
                    n_sample_steps=train_cfg["n_sample_steps"],
                    sample_method=train_cfg["sample_method"],
                )
            else:
                val_loss, val_latent_pearson, val_voxel_pearson = validate(
                    ema_model, val_loader, train_cfg["cfm_sigma"], device, use_amp,
                    vae_decoder=vae,
                    n_sample_steps=train_cfg["n_sample_steps"],
                    sample_method=train_cfg["sample_method"],
                )

        elapsed = time.time() - t_epoch
        lr = optimizer.param_groups[0]["lr"]

        if did_val:
            logger.info(
                "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | "
                "Latent Pearson: %.4f | Voxel Pearson: %.4f | LR: %.2e | Time: %.1fs",
                epoch + 1, n_epochs, train_loss, val_loss,
                val_latent_pearson, val_voxel_pearson, lr, elapsed,
            )
        else:
            logger.info(
                "Epoch %d/%d | Train Loss: %.6f | LR: %.2e | Time: %.1fs",
                epoch + 1, n_epochs, train_loss, lr, elapsed,
            )

        # History — only log when validation is performed
        if did_val:
            with open(history_path, "a", newline="") as f:
                writer = csv.DictWriter(f, history_fields)
                writer.writerow({
                    "epoch": epoch + 1,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_latent_pearson": f"{val_latent_pearson:.6f}",
                    "val_voxel_pearson": f"{val_voxel_pearson:.6f}",
                    "lr": f"{lr:.2e}",
                    "time_s": f"{elapsed:.1f}",
                })

        # Checkpoint
        if not args.fast_dev_run:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_latent_pearson": val_latent_pearson,
                "val_voxel_pearson": val_voxel_pearson,
            }
            torch.save(ckpt, output_dir / "last.pt")

            # Track best by latent Pearson (faster + more reliable early indicator)
            if val_latent_pearson > best_val_pearson:
                best_val_pearson = val_latent_pearson
                torch.save(ckpt, output_dir / "best.pt")
                logger.info("  → New best latent Pearson: %.4f", val_latent_pearson)

    logger.info("=" * 60)
    logger.info("Training complete! Best Latent Pearson: %.4f", best_val_pearson)
    logger.info("Outputs saved to: %s", output_dir)


def _limit_batches(loader, n):
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
