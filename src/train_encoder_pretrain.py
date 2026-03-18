"""Pre-train TemporalFusionEncoder with regression head.

Stage 1 of 2-stage BrainFlow training:
  1. Train encoder + regression head (MSE + PCC loss) → good context vectors
  2. Freeze encoder, train flow decoder (see train_brainflow.py --pretrained_encoder)

Usage:
    python src/train_encoder_pretrain.py --config src/configs/brain_flow.yaml
    python src/train_encoder_pretrain.py --config src/configs/brain_flow.yaml --fast_dev_run
"""

import argparse
import logging
import math as pymath
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_config
from src.models.brainflow.brain_flow import TemporalFusionEncoder
from src.train_brainflow import (
    BrainFlowDataset,
    ClipGroupedBatchSampler,
    EMAModel,
    pearson_corr_per_dim,
    resolve_paths,
    _extract_modality_features,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pretrain_encoder")


# =============================================================================
# Encoder + Regression Head
# =============================================================================

class EncoderWithRegression(nn.Module):
    """TemporalFusionEncoder + regression head for pre-training.

    Architecture:
        multimodal features → TemporalFusionEncoder → context (B, H)
        context → RegressionHead → fMRI prediction (B, output_dim)

    Head is intentionally simple (1 hidden layer) to force the encoder
    to learn good representations instead of memorizing via a large head.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 1000,
        encoder_params: dict = None,
        head_hidden_mult: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.encoder = TemporalFusionEncoder(
            modality_dims=modality_dims,
            **(encoder_params or {}),
        )
        hidden_dim = (encoder_params or {}).get("hidden_dim", 512)

        # Simple regression head — force encoder to learn good features
        head_hidden = hidden_dim * head_hidden_mult
        if head_hidden_mult <= 1:
            # Minimal head: just project to output dim
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, output_dim),
            )

    def forward(self, modality_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns fMRI prediction (B, output_dim)."""
        context = self.encoder(modality_features)  # (B, H)
        pred = self.head(context)                  # (B, output_dim)
        return pred

    def get_encoder_state_dict(self) -> dict:
        """Extract only encoder weights for loading into BrainFlowCFM."""
        return self.encoder.state_dict()


def compute_regression_loss(pred, target, pcc_weight=0.1):
    """MSE + PCC auxiliary loss.

    PCC is computed per-sample across voxels (spatial pattern similarity).
    This is a regularizer, not the main objective.

    Args:
        pred: (B, D) predicted fMRI
        target: (B, D) target fMRI
        pcc_weight: weight for PCC loss (keep low: 0.05-0.1)

    Returns:
        total_loss, mse_loss, pcc_value
    """
    mse_loss = F.mse_loss(pred, target)

    # Per-sample PCC (across voxels) — spatial pattern similarity
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    tgt_c = target - target.mean(dim=-1, keepdim=True)
    cov = (pred_c * tgt_c).sum(dim=-1)
    std = torch.sqrt((pred_c ** 2).sum(dim=-1) * (tgt_c ** 2).sum(dim=-1) + 1e-8)
    pcc = (cov / std).mean()

    pcc_loss = 1.0 - pcc
    total = mse_loss + pcc_weight * pcc_loss

    return total, mse_loss.item(), pcc.item()


# =============================================================================
# Training
# =============================================================================

def train(args):
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretrain-specific config with sensible defaults
    pretrain_cfg = cfg.get("pretrain", {})
    n_epochs = pretrain_cfg.get("n_epochs", 100)
    lr = pretrain_cfg.get("lr", 2e-4)
    pcc_weight = pretrain_cfg.get("pcc_weight", 0.1)
    early_stop_patience = pretrain_cfg.get("early_stop_patience", 15)

    # Override encoder dropout if specified
    encoder_dropout = pretrain_cfg.get("encoder_dropout", None)
    modality_dropout = pretrain_cfg.get("modality_dropout", None)

    if args.fast_dev_run:
        n_epochs = 1

    # 1. Data
    train_set = BrainFlowDataset(cfg, split="train")
    val_set = BrainFlowDataset(cfg, split="val")

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

    # 2. Model
    bf_cfg = cfg["brainflow"]
    modality_dims = {}
    for mod_name, mod_cfg in cfg["raw_features"]["modalities"].items():
        dim = mod_cfg["dim"]
        if cfg["raw_features"].get("layer_aggregation") == "cat":
            n_layers = mod_cfg.get("n_layers", 1)
            dim = dim * n_layers
        modality_dims[mod_name] = dim

    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])

    # Build encoder params — override dropout if specified in pretrain config
    encoder_params = dict(bf_cfg["encoder"])
    if encoder_dropout is not None:
        encoder_params["dropout"] = encoder_dropout
        logger.info("Overriding encoder dropout: %.2f", encoder_dropout)
    if modality_dropout is not None:
        encoder_params["modality_dropout"] = modality_dropout
        logger.info("Overriding modality dropout: %.2f", modality_dropout)

    model = EncoderWithRegression(
        modality_dims=modality_dims,
        output_dim=output_dim,
        encoder_params=encoder_params,
        head_hidden_mult=pretrain_cfg.get("head_hidden_mult", 1),
        dropout=pretrain_cfg.get("head_dropout", 0.1),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    n_head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    logger.info("Total params: %s (encoder: %s, head: %s)",
                f"{n_params:,}", f"{n_enc_params:,}", f"{n_head_params:,}")

    # 3. Optimizer & Scheduler
    tr_cfg = cfg["training"]
    head_wd = pretrain_cfg.get("head_weight_decay", 0.1)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr, "weight_decay": tr_cfg.get("weight_decay", 0.05)},
        {"params": model.head.parameters(), "lr": lr, "weight_decay": head_wd},
    ])

    total_steps = len(train_loader) * n_epochs
    if args.fast_dev_run:
        total_steps = 2
    warmup_steps = int(total_steps * tr_cfg.get("warmup_ratio", 0.05))
    min_lr = pretrain_cfg.get("min_lr", 1e-6)

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / lr + (1 - min_lr / lr) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [cosine_with_warmup, cosine_with_warmup])

    # 4. Output dir
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow") / "pretrain_encoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    modality_names = list(cfg["raw_features"]["modalities"].keys())

    # EMA
    ema_decay = tr_cfg.get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)

    # 5. Resume
    start_epoch = 1
    best_val_pcc = -1.0
    global_step = 0
    epochs_without_improvement = 0

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
            best_val_pcc = ckpt.get("best_val_pcc", -1.0)
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            logger.info("Resumed from epoch %d (best=%.4f)", ckpt["epoch"], best_val_pcc)
            del ckpt

    history_file = out_dir / "history.csv"
    if start_epoch == 1:
        with open(history_file, "w") as f:
            f.write("epoch,train_loss,train_mse,train_pcc,val_pcc,lr\n")

    val_every = pretrain_cfg.get("val_every_n_epochs", 5)
    fmri_std = cfg["fmri"].get("fmri_std", 0.6)

    logger.info("Starting encoder pre-training: %d epochs, lr=%.2e, pcc_weight=%.2f, "
                "early_stop=%d, head_wd=%.3f",
                n_epochs, lr, pcc_weight, early_stop_patience, head_wd)

    # 6. Training loop
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        losses, mses, pccs = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            mod_features = _extract_modality_features(batch, modality_names, device)
            fmri_target = batch["fmri"].to(device)  # (B, V) — normalized

            # Mixup augmentation
            B = fmri_target.shape[0]
            mixup_alpha = tr_cfg.get("mixup_alpha", 0.0)
            if mixup_alpha > 0 and B > 1:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(B, device=device)
                fmri_target = lam * fmri_target + (1 - lam) * fmri_target[perm]
                for mod_name in mod_features:
                    mod_features[mod_name] = lam * mod_features[mod_name] + (1 - lam) * mod_features[mod_name][perm]

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                pred = model(mod_features)  # (B, V)
                loss, mse_val, pcc_val = compute_regression_loss(pred, fmri_target, pcc_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg.get("grad_clip", 1.0))
            optimizer.step()
            scheduler.step()

            ema.update(model)

            losses.append(loss.item())
            mses.append(mse_val)
            pccs.append(pcc_val)
            global_step += 1

            if global_step % tr_cfg.get("log_every_n_steps", 20) == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(losses[-50:]):.4f}",
                    "pcc": f"{np.mean(pccs[-50:]):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Validation
        mean_val_pcc = 0.0
        if epoch % val_every == 0 or epoch == n_epochs or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()

            # Accumulate per-clip predictions
            fmri_pred_acc = {}
            fmri_tgt_acc = {}

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break

                    mod_features = _extract_modality_features(batch, modality_names, device)
                    fmri_target = batch["fmri"].to(device)

                    pred = model(mod_features)  # (B, V)

                    # Denormalize
                    pred = pred * fmri_std
                    fmri_target = fmri_target * fmri_std

                    clip_keys = batch["clip_key"]
                    target_trs = batch["target_tr"]
                    n_trs_batch = batch["n_trs"]

                    B = pred.shape[0]
                    for b in range(B):
                        ck = clip_keys[b]
                        tr_idx = int(target_trs[b].item())
                        n_t = int(n_trs_batch[b].item())
                        fmri_dim = pred.shape[-1]

                        if ck not in fmri_pred_acc:
                            fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                            fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}

                        fmri_pred_acc[ck]["sum"][tr_idx] += pred[b].cpu()
                        fmri_pred_acc[ck]["count"][tr_idx] += 1
                        fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b].cpu()
                        fmri_tgt_acc[ck]["count"][tr_idx] += 1

            # Average and compute PCC (per-voxel across time)
            all_pred, all_tgt = [], []
            for ck in sorted(fmri_pred_acc.keys()):
                count = fmri_pred_acc[ck]["count"]
                valid = count > 0
                if valid.sum() < 2:
                    continue
                all_pred.append(fmri_pred_acc[ck]["sum"][valid] / count[valid].unsqueeze(-1))
                all_tgt.append(fmri_tgt_acc[ck]["sum"][valid] / fmri_tgt_acc[ck]["count"][valid].unsqueeze(-1))

            if all_pred:
                all_pred = torch.cat(all_pred, dim=0)
                all_tgt = torch.cat(all_tgt, dim=0)
                pcc_per_voxel = pearson_corr_per_dim(all_pred.unsqueeze(0), all_tgt.unsqueeze(0))
                mean_val_pcc = float(pcc_per_voxel.mean().item())

            logger.info("Epoch %d | Train loss: %.4f | Train PCC(spatial): %.4f | Val PCC(temporal): %.4f",
                        epoch, np.mean(losses), np.mean(pccs), mean_val_pcc)

            # Early stopping & best model
            if mean_val_pcc > best_val_pcc:
                best_val_pcc = mean_val_pcc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), out_dir / "best.pt")
                torch.save(model.get_encoder_state_dict(), out_dir / "best_encoder.pt")
                logger.info("★ New best! Val PCC: %.4f → saved best_encoder.pt", mean_val_pcc)
            else:
                epochs_without_improvement += val_every
                logger.info("  No improvement for %d epochs (patience: %d)",
                            epochs_without_improvement, early_stop_patience)

            # History
            with open(history_file, "a") as f:
                f.write(f"{epoch},{np.mean(losses):.6f},{np.mean(mses):.6f},"
                        f"{np.mean(pccs):.4f},{mean_val_pcc:.6f},{scheduler.get_last_lr()[0]:.2e}\n")

            ema.restore(model)

            # Early stop check
            if epochs_without_improvement >= early_stop_patience and not args.fast_dev_run:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch, epochs_without_improvement)
                break

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "global_step": global_step,
            "best_val_pcc": best_val_pcc,
            "epochs_without_improvement": epochs_without_improvement,
        }, out_dir / "last.pt")

    logger.info("Pre-training complete! Best Val PCC: %.4f", best_val_pcc)
    logger.info("Encoder saved to: %s", out_dir / "best_encoder.pt")
    logger.info("Next: python src/train_brainflow.py --config %s --pretrained_encoder %s",
                args.config, out_dir / "best_encoder.pt")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Pre-train encoder with regression")
    parser.add_argument("--config", type=str, default="src/configs/brain_flow.yaml")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
