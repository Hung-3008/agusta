"""Train BrainFlow CFM v3 — Deep-Flow inspired architecture.

Usage:
    python src/train_brain_flow_v3.py --config src/configs/brain_flow_v3.yaml
    python src/train_brain_flow_v3.py --config src/configs/brain_flow_v3.yaml --fast_dev_run
"""

import argparse
import logging
import math as pymath
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.precompute_latents import load_vae
from src.data.dataset import load_config
from src.models.brainflow.brain_flow_v3 import BrainFlowCFMv3

# Reuse dataset and utilities from V2
from src.train_brain_flow_v2 import (
    BrainFlowV2Dataset,
    ClipGroupedBatchSampler,
    EMAModel,
    pearson_corr,
    pearson_corr_per_dim,
    resolve_paths_v2,
    _extract_modality_features,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_cfm_v3")


def get_dataloaders(cfg):
    train_set = BrainFlowV2Dataset(cfg, split="train")
    val_set = BrainFlowV2Dataset(cfg, split="val")

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

    # 2. Load frozen VAE (for fMRI eval only — no Latent-CFM in V3)
    try:
        vae = load_vae(cfg, device)
        logger.info("Loaded frozen VAE for fMRI evaluation")
    except Exception as e:
        logger.warning("Could not load VAE: %s. fMRI PCC will be 0.", e)
        vae = None

    # 3. Init Model
    logger.info("Initializing BrainFlowCFMv3 (Deep-Flow inspired)...")
    bf_cfg = cfg["brainflow"]

    modality_dims = {}
    for mod_name, mod_cfg in cfg["raw_features"]["modalities"].items():
        dim = mod_cfg["dim"]
        if cfg["raw_features"].get("layer_aggregation") == "cat":
            n_layers = mod_cfg.get("n_layers", 1)
            dim = dim * n_layers
        modality_dims[mod_name] = dim

    logger.info("Modality dimensions: %s", modality_dims)

    model = BrainFlowCFMv3(
        modality_dims=modality_dims,
        latent_dim=bf_cfg["latent_dim"],
        encoder_params=bf_cfg["encoder"],
        velocity_net_params=bf_cfg.get("velocity_net", {}),
        cfm_params=bf_cfg.get("cfm", {}),
        cfg_drop_prob=bf_cfg.get("cfg_drop_prob", 0.1),
        n_voxels=cfg["fmri"].get("n_voxels", 1000),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

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

    # 5. Output dir
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brain_flow_v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    shutil.copy2(args.config, out_dir / "config.yaml")
    logger.info("Saved config to %s", out_dir / "config.yaml")

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
        f.write("epoch,train_loss,val_latent_pearson,val_fmri_pcc,lr\n")

    for epoch in range(1, tr_cfg["n_epochs"] + 1):
        model.train()
        train_losses = []

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

            # --- Data Augmentation ---
            latent_noise_scale = tr_cfg.get("latent_noise_scale", 0.0)
            if latent_noise_scale > 0:
                latents = latents + latent_noise_scale * torch.randn_like(latents)

            mixup_alpha = tr_cfg.get("mixup_alpha", 0.0)
            if mixup_alpha > 0 and B > 1:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(B, device=device)
                latents = lam * latents + (1 - lam) * latents[perm]
                mask = torch.min(mask, mask[perm])
                for mod_name in mod_features:
                    mod_features[mod_name] = lam * mod_features[mod_name] + (1 - lam) * mod_features[mod_name][perm]
                if fmri is not None:
                    fmri = lam * fmri + (1 - lam) * fmri[perm]

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                loss = model(
                    mod_features, latents, mask=mask, fmri_target=fmri,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            global_step += 1

            # EMA update
            ema.update(model)

            if global_step % tr_cfg["log_every_n_steps"] == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(train_losses[-50:]):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Validation
        mean_corr = 0.0
        mean_fmri_corr = 0.0
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()

            logger.info("Running validation...")

            lat_pred_acc = {}
            lat_tgt_acc = {}
            fmri_pred_acc = {}
            fmri_tgt_acc = {}

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
                        temperature=0.0,
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

                    # Accumulate per-clip, per-TR
                    clip_keys = batch["clip_key"]
                    start_idxs = batch["start_idx"]
                    n_trs_batch = batch["n_trs"]

                    for b in range(gen_latents.shape[0]):
                        vl = int(valid_lens[b].item()) if valid_lens is not None else gen_latents.shape[1]
                        ck = clip_keys[b]
                        si = int(start_idxs[b].item())
                        n_t = int(n_trs_batch[b].item())
                        lat_dim = gen_latents.shape[-1]

                        if ck not in lat_pred_acc:
                            lat_pred_acc[ck] = {"sum": torch.zeros(n_t, lat_dim), "count": torch.zeros(n_t)}
                            lat_tgt_acc[ck] = {"sum": torch.zeros(n_t, lat_dim), "count": torch.zeros(n_t)}

                        lat_pred_acc[ck]["sum"][si:si+vl] += gen_latents[b, :vl].cpu()
                        lat_pred_acc[ck]["count"][si:si+vl] += 1
                        lat_tgt_acc[ck]["sum"][si:si+vl] += latents[b, :vl].cpu()
                        lat_tgt_acc[ck]["count"][si:si+vl] += 1

                        if gen_fmri is not None:
                            fmri_dim = gen_fmri.shape[-1]
                            if ck not in fmri_pred_acc:
                                fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                                fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                            fmri_pred_acc[ck]["sum"][si:si+vl] += gen_fmri[b, :vl].cpu()
                            fmri_pred_acc[ck]["count"][si:si+vl] += 1
                            fmri_tgt_acc[ck]["sum"][si:si+vl] += true_fmri[b, :vl].cpu()
                            fmri_tgt_acc[ck]["count"][si:si+vl] += 1

            # Average overlapping predictions
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

            # Compute PCC
            if all_gen_lat:
                all_gen_lat = torch.cat(all_gen_lat, dim=0)
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

            # Best model
            metric_for_best = mean_fmri_corr if vae is not None else mean_corr
            if metric_for_best > best_val_corr:
                best_val_corr = metric_for_best
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("Saved new best EMA model to %s", out_dir / "best.pt")

            # Save history
            mean_train_loss = float(np.mean(train_losses))
            current_lr = scheduler.get_last_lr()[0]
            with open(history_file, "a") as f:
                f.write(f"{epoch},{mean_train_loss:.6f},{mean_corr:.6f},{mean_fmri_corr:.6f},{current_lr:.2e}\n")

            # Restore original weights
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
    parser.add_argument("--config", type=str, default="src/configs/brain_flow_v3.yaml")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches to test pipeline")
    args = parser.parse_args()
    train(args)
