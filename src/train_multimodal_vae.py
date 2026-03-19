"""Train Multimodal MoE-VAE — Encode multi-feature → shared latent space.

Per-timestep encoding of 8 pretrained feature modalities into a shared
512-dim latent space using Weighted Layer Pooling, MoE fusion, and β-VAE
with annealing.

Usage:
    python src/train_multimodal_vae.py --config src/configs/multimodal_vae.yaml --fast_dev_run
    python src/train_multimodal_vae.py --config src/configs/multimodal_vae.yaml
    python src/train_multimodal_vae.py --config src/configs/multimodal_vae.yaml --resume
"""

import argparse
import copy
import logging
import math as pymath
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multimodal_vae_dataset import MultimodalVAEDataset
from src.models.brainflow.multimodal_vae import MultimodalMoEVAE, beta_schedule

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_multimodal_vae")


# =============================================================================
# EMA Model
# =============================================================================

class EMAModel:
    """Exponential Moving Average of model weights."""

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
# Config loading
# =============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Dataloaders
# =============================================================================

def get_dataloaders(cfg):
    features_dir = Path(PROJECT_ROOT) / cfg["features_dir"]
    modalities = cfg["modalities"]
    splits_cfg = cfg["splits"]
    dl_cfg = cfg["dataloader"]
    data_mode = cfg.get("data_mode", "npy")

    train_set = MultimodalVAEDataset(
        features_dir=features_dir,
        modalities=modalities,
        splits_cfg=splits_cfg,
        split="train",
        stride=dl_cfg.get("stride", 1),
        mode=data_mode,
    )
    val_set = MultimodalVAEDataset(
        features_dir=features_dir,
        modalities=modalities,
        splits_cfg=splits_cfg,
        split="val",
        stride=dl_cfg.get("stride", 1),
        mode=data_mode,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=dl_cfg.get("val_batch_size", dl_cfg["batch_size"]),
        shuffle=False,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        drop_last=False,
    )
    return train_loader, val_loader


# =============================================================================
# Training
# =============================================================================

def train(args):
    cfg = load_config(args.config)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data
    logger.info("Loading datasets...")
    train_loader, val_loader = get_dataloaders(cfg)
    logger.info("Train: %d samples, Val: %d samples",
                len(train_loader.dataset), len(val_loader.dataset))

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1

    # 2. Model
    model_cfg = cfg["model"]
    modality_configs = {
        name: {"dim": mc["dim"], "n_layers": mc["n_layers"]}
        for name, mc in cfg["modalities"].items()
    }

    model = MultimodalMoEVAE(
        modality_configs=modality_configs,
        latent_dim=model_cfg["latent_dim"],
        encoder_hidden=model_cfg["encoder_hidden"],
        n_experts=model_cfg["n_experts"],
        expert_hidden=model_cfg["expert_hidden"],
        decoder_hidden=model_cfg["decoder_hidden"],
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # 3. Optimizer & Scheduler
    tr_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg.get("weight_decay", 0.01),
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

    # 4. EMA
    ema_decay = tr_cfg.get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)

    # 5. Output dir
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/multimodal_vae")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    # β-VAE annealing config
    beta_cfg = cfg.get("beta", {})
    beta_start = beta_cfg.get("start", 0.0001)
    beta_max = beta_cfg.get("max", 0.01)
    beta_warmup = beta_cfg.get("warmup_epochs", 30)

    # 6. Resume
    start_epoch = 1
    best_val_loss = float("inf")
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
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            logger.info("Resumed from epoch %d (global_step=%d)", ckpt["epoch"], global_step)
            del ckpt

    history_file = out_dir / "history.csv"
    if start_epoch == 1:
        with open(history_file, "w") as f:
            f.write("epoch,train_loss,train_recon,train_kl,val_loss,val_recon,val_kl,beta,lr\n")

    # 7. Training loop
    modality_names = list(cfg["modalities"].keys())

    for epoch in range(start_epoch, tr_cfg["n_epochs"] + 1):
        model.train()
        beta = beta_schedule(epoch, beta_start, beta_max, beta_warmup)
        train_metrics = {"loss": [], "recon": [], "kl": []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=tr_cfg.get("use_amp", True), dtype=torch.bfloat16):
                loss_dict = model.loss(batch, beta=beta)

            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg.get("grad_clip", 1.0))
            optimizer.step()
            scheduler.step()

            train_metrics["loss"].append(loss_dict["loss"].item())
            train_metrics["recon"].append(loss_dict["recon"].item())
            train_metrics["kl"].append(loss_dict["kl"].item())
            global_step += 1

            ema.update(model)

            if global_step % tr_cfg.get("log_every_n_steps", 20) == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(train_metrics['loss'][-50:]):.4f}",
                    "recon": f"{np.mean(train_metrics['recon'][-50:]):.4f}",
                    "kl": f"{np.mean(train_metrics['kl'][-50:]):.4f}",
                    "β": f"{beta:.5f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Validation
        val_loss, val_recon, val_kl = 0.0, 0.0, 0.0
        if epoch % tr_cfg.get("val_every_n_epochs", 5) == 0 or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()

            val_metrics = {"loss": [], "recon": [], "kl": []}

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break

                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.amp.autocast("cuda", enabled=tr_cfg.get("use_amp", True), dtype=torch.bfloat16):
                        loss_dict = model.loss(batch, beta=beta)

                    val_metrics["loss"].append(loss_dict["loss"].item())
                    val_metrics["recon"].append(loss_dict["recon"].item())
                    val_metrics["kl"].append(loss_dict["kl"].item())

            val_loss = float(np.mean(val_metrics["loss"]))
            val_recon = float(np.mean(val_metrics["recon"]))
            val_kl = float(np.mean(val_metrics["kl"]))

            logger.info(
                "Epoch %d | β=%.5f | Train loss=%.4f recon=%.4f kl=%.4f | "
                "Val loss=%.4f recon=%.4f kl=%.4f",
                epoch, beta,
                np.mean(train_metrics["loss"]),
                np.mean(train_metrics["recon"]),
                np.mean(train_metrics["kl"]),
                val_loss, val_recon, val_kl,
            )

            # Log layer pool weights periodically
            if epoch % 20 == 0:
                weights = model.get_layer_pool_weights()
                for mod_name, w in weights.items():
                    w_str = ", ".join(f"{v:.3f}" for v in w.cpu().tolist())
                    logger.info("  %s layer weights: [%s]", mod_name, w_str)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("Saved new best EMA model (val_loss=%.4f)", val_loss)

            # Write history
            mean_train_loss = float(np.mean(train_metrics["loss"]))
            mean_train_recon = float(np.mean(train_metrics["recon"]))
            mean_train_kl = float(np.mean(train_metrics["kl"]))
            current_lr = scheduler.get_last_lr()[0]
            with open(history_file, "a") as f:
                f.write(
                    f"{epoch},{mean_train_loss:.6f},{mean_train_recon:.6f},"
                    f"{mean_train_kl:.6f},{val_loss:.6f},{val_recon:.6f},"
                    f"{val_kl:.6f},{beta:.6f},{current_lr:.2e}\n"
                )

            ema.restore(model)

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        }, out_dir / "last.pt")

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Multimodal MoE-VAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--fast_dev_run", action="store_true", help="Quick test run")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
