"""
BrainFlow V2 — FlowFM-Style Training Script.

Direct flow matching from N(0,I) to fMRI voxel space, conditioned on
raw images via a trainable ViT encoder (FlowFM architecture).

Key differences from train_brainflow.py (V1):
  - Loads raw images instead of pre-extracted DINOv2 features
  - Jointly trains ViT encoder + velocity network
  - Separate learning rates for encoder vs velocity
  - DGS operates on representation r (zero-out ViT output)

Usage:
    python -m src.train_brainflow_v2 --config src/configs/subj01/brainflow_v2.yaml
    python -m src.train_brainflow_v2 --config src/configs/subj01/brainflow_v2.yaml --debug
"""

import argparse
import copy
import csv
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from src.data.dataset import build_dataloaders, preload_datasets_to_ram

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from src.models.brainflow.brain_flow_v2 import (
    BrainFlowV2, BrainFlowV2Config, timestep_embedding, modulate,
)


# ─── Dataset ──────────────────────────────────────────────────────────────────


# FmriImageDataset removed. Using SlidingWindowDataset instead.


# ─── Utilities ────────────────────────────────────────────────────────────────


def pearson_corr_voxelwise(pred, target):
    """Per-voxel Pearson correlation averaged across voxels."""
    pred_zm = pred - pred.mean(0, keepdim=True)
    tgt_zm = target - target.mean(0, keepdim=True)
    num = (pred_zm * tgt_zm).sum(0)
    den = (pred_zm.norm(dim=0) * tgt_zm.norm(dim=0)).clamp(min=1e-8)
    return (num / den).mean().item()


def pearson_corr_samplewise(pred, target):
    """Per-sample Pearson correlation averaged across samples."""
    pred_zm = pred - pred.mean(1, keepdim=True)
    tgt_zm = target - target.mean(1, keepdim=True)
    num = (pred_zm * tgt_zm).sum(1)
    den = (pred_zm.norm(dim=1) * tgt_zm.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()


def ema_update(source, target, decay):
    with torch.no_grad():
        for s, t in zip(source.parameters(), target.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1 - decay)
        # Also copy buffers (e.g. BatchNorm running stats)
        for s, t in zip(source.buffers(), target.buffers()):
            t.data.copy_(s.data)


def cosine_lr(optimizer, epoch, total, warmup, base_lrs, min_lr=1e-6):
    """Cosine LR scheduler supporting per-group base LRs."""
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs] * len(optimizer.param_groups)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        if epoch < warmup:
            lr = base_lr * epoch / max(warmup, 1)
        else:
            p = (epoch - warmup) / max(total - warmup, 1)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * p))
        pg["lr"] = lr
    return [pg["lr"] for pg in optimizer.param_groups]


# ─── Velocity Loss Functions ─────────────────────────────────────────────────


def compute_velocity_loss(v_pred, ut, t=None, loss_type="mse",
                          huber_c=0.1,
                          timestep_weight_type="none",
                          direction_weight=1.0,
                          magnitude_weight=0.1):
    """
    Compute velocity matching loss.

    Args:
        v_pred: (B, D) predicted velocity
        ut: (B, D) target velocity
        t: (B,) timestep values for weighting
        loss_type: "mse" | "pseudo_huber" | "decoupled"
    """
    info = {}

    if loss_type == "pseudo_huber":
        diff_sq = (v_pred - ut).pow(2)
        per_element = (diff_sq + huber_c ** 2).sqrt() - huber_c
        raw_loss = per_element.mean(dim=-1)
        info["huber_raw"] = raw_loss.mean().item()

    elif loss_type == "decoupled":
        cos_sim = F.cosine_similarity(v_pred, ut, dim=-1)
        dir_loss = 1.0 - cos_sim
        pred_mag = v_pred.norm(dim=-1).clamp(min=1e-6)
        tgt_mag = ut.norm(dim=-1).clamp(min=1e-6)
        mag_loss = (pred_mag.log() - tgt_mag.log()).pow(2)
        raw_loss = direction_weight * dir_loss + magnitude_weight * mag_loss
        info["dir_loss"] = dir_loss.mean().item()
        info["mag_loss"] = mag_loss.mean().item()
        info["v_cos"] = cos_sim.mean().item()

    else:  # "mse"
        raw_loss = (v_pred - ut).pow(2).mean(dim=-1)

    # Timestep weighting
    if t is not None and timestep_weight_type != "none":
        if timestep_weight_type == "linear":
            w = 1.0 - t
        elif timestep_weight_type == "cosine":
            w = torch.cos(t * 3.14159 / 2).pow(2)
        elif timestep_weight_type == "snr":
            w = 1.0 / (1.0 + t / (1.0 - t + 1e-4))
        else:
            w = torch.ones_like(t)
        w = w / (w.mean() + 1e-8)
        loss = (w * raw_loss).mean()
        info["w_mean"] = w.mean().item()
    else:
        loss = raw_loss.mean()

    return loss, info


# ─── ODE Wrapper ─────────────────────────────────────────────────────────────


class BrainFlowV2ODEWrapper(torch.nn.Module):
    """ODE wrapper for BrainFlowV2: noise → fMRI voxels."""

    def __init__(self, model, features_dict, cfg_scale=1.0):
        super().__init__()
        self.model = model
        self.features_dict = features_dict
        self.cfg_scale = cfg_scale

    def forward(self, t, x):
        B = x.shape[0]
        t_batch = t.expand(B)
        if self.cfg_scale == 1.0:
            return self.model(t_batch, x, self.features_dict)
        else:
            return self.model.forward_with_cfg(
                t_batch, x, self.features_dict, self.cfg_scale)


@torch.no_grad()
def stochastic_euler_sample(model, x0, features_dict, T, cfg_scale=1.0):
    """FlowNP-style stochastic Euler sampling."""
    z = x0
    B = z.shape[0]
    for i in range(T):
        tt = i / T
        t_batch = torch.full((B,), tt, device=z.device)
        if cfg_scale == 1.0:
            v_pred = model(t_batch, z, features_dict)
        else:
            v_pred = model.forward_with_cfg(
                t_batch, z, features_dict, cfg_scale)
        alpha = 1 + tt * (1 - tt)
        sigma = 0.2 * (tt * (1 - tt)) ** 0.5
        z = z + (alpha * v_pred + sigma * torch.randn_like(z)) / T
    return z


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, val_loader, fm, device, ode_steps=50,
             cfg_scale=1.0, num_trials=1,
             sampler="ode"):
    """Validate BrainFlowV2 — generate fMRI directly from images."""
    if sampler == "ode":
        from torchdiffeq import odeint

    model.eval()
    total_flow_loss = 0
    n_batches = 0
    all_pred, all_true = [], []
    all_v_cos = []

    for batch in val_loader:
        fmri = batch["fmri"].to(device)
        features_dict = {k: v.to(device) for k, v in batch["features"].items()}
        # Target is fMRI

        # Flow loss
        x0 = torch.randn_like(fmri)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, fmri)
        v_pred = model(t, xt, features_dict)

        flow_loss = F.mse_loss(v_pred, ut)
        total_flow_loss += flow_loss.item()
        n_batches += 1

        cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
        all_v_cos.append(cos)

        # Generation: noise → fMRI directly
        fmri_gen = torch.zeros_like(fmri)
        for _ in range(num_trials):
            x0_trial = torch.randn_like(fmri)
            if sampler == "stochastic_euler":
                fmri_trial = stochastic_euler_sample(
                    model, x0_trial, features_dict, T=ode_steps,
                    cfg_scale=cfg_scale)
            else:
                ode_fn = BrainFlowV2ODEWrapper(model, features_dict, cfg_scale)
                t_span = torch.linspace(0, 1, ode_steps, device=device)
                traj = odeint(ode_fn, x0_trial, t_span, method="midpoint")
                fmri_trial = traj[-1]
            fmri_gen += fmri_trial
        fmri_gen = fmri_gen / num_trials

        all_pred.append(fmri_gen)
        all_true.append(fmri)

    model.train()

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)

    metrics = {
        "val_flow_loss": total_flow_loss / max(n_batches, 1),
        "val_v_cos": sum(all_v_cos) / len(all_v_cos),
    }

    metrics["val_fmri_mse"] = F.mse_loss(preds, trues).item()
    metrics["val_fmri_pcc"] = pearson_corr_voxelwise(preds, trues)
    metrics["val_fmri_spcc"] = pearson_corr_samplewise(preds, trues)

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        "BrainFlow V2 — FlowFM-Style Direct Flow Matching")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest.pt in output_dir")
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to a specific checkpoint .pt file to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    num_epochs = 2 if args.debug else train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    encoder_lr = train_cfg.get("encoder_lr", lr * 0.1)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    use_ema = train_cfg.get("use_ema", True)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    cfg_drop_prob = train_cfg.get("cfg_drop_prob", 0.5)
    cfg_scale = train_cfg.get("cfg_scale", 1.0)
    ode_steps = train_cfg.get("ode_steps", 50)
    num_trials = train_cfg.get("num_trials", 1)
    eval_interval = 1 if args.debug else train_cfg.get("eval_interval", 5)
    sampler = train_cfg.get("sampler", "ode")
    freeze_encoder_epochs = train_cfg.get("freeze_encoder_epochs", 0)
    log_interval = train_cfg.get("log_interval", 200)  # log every N batches



    # Flow options
    use_ot = train_cfg.get("use_ot", True)
    sigma = train_cfg.get("sigma", 0.0)

    timestep_sampling = train_cfg.get("timestep_sampling", "logit_normal")
    logit_normal_mu = train_cfg.get("logit_normal_mu", 0.0)
    logit_normal_sigma = train_cfg.get("logit_normal_sigma", 1.0)

    # Loss options
    velocity_loss_type = train_cfg.get("velocity_loss_type", "mse")
    huber_c = train_cfg.get("huber_c", 0.1)
    timestep_weight_type = train_cfg.get("timestep_weight_type", "none")
    direction_weight = train_cfg.get("direction_weight", 1.0)
    magnitude_weight = train_cfg.get("magnitude_weight", 0.1)


    output_dir = cfg.get("output_dir", "results/brainflow_v2")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Logger ──
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_file, mode='w')],
    )
    logger = logging.getLogger('brainflow_v2')
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    # ── Data ──
    # build_dataloaders uses a separate data.yaml in the same configs directory
    data_yaml_path = os.path.join(
        os.path.dirname(args.config), "data.yaml")
    modalities = data_cfg.get("modalities", ["video", "audio", "text", "omni"])
    loaders = build_dataloaders(data_yaml_path, modalities=modalities)
    train_loader = loaders.get("train", [])
    val_loader = loaders.get("val", [])
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Preload all data to RAM (eliminates HDD I/O during training) ──
    preload_datasets_to_ram(loaders)

    # ── Model ──
    model = BrainFlowV2(BrainFlowV2Config(**model_cfg)).to(device)
    ema_model = copy.deepcopy(model) if use_ema else None
    pc = model.param_count()
    logger.info(
        f"BrainFlowV2: encoder={pc['encoder_M']:.1f}M "
        f"cond={pc['cond_M']:.2f}M ctx={pc['ctx_proj_M']:.2f}M "
        f"blocks={pc['blocks_M']:.1f}M "
        f"embed={pc['embed_M']:.2f}M output={pc['output_M']:.3f}M "
        f"total={pc['total_M']:.1f}M | EMA={'ON' if use_ema else 'OFF'}")

    # ── Flow Matcher ──
    if use_ot:
        try:
            import ot
            fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
            logger.info(f"Using ExactOT Flow Matcher (sigma={sigma})")
        except ImportError:
            logger.warning("POT not found, reverting to standard CFM")
            fm = ConditionalFlowMatcher(sigma=sigma)
            use_ot = False
    else:
        fm = ConditionalFlowMatcher(sigma=sigma)
        logger.info(f"Using standard CFM (sigma={sigma})")

    # ── Optimizer (separate LR for encoder) ──
    optimizer = torch.optim.AdamW([
        {"params": model.get_encoder_params(),
         "lr": encoder_lr, "name": "encoder"},
        {"params": model.get_velocity_params(),
         "lr": lr, "name": "velocity"},
    ], weight_decay=train_cfg.get("weight_decay", 0.05))
    base_lrs = [encoder_lr, lr]

    # ── Resume from checkpoint ──
    start_epoch = 1
    best_pcc = -1.0
    patience_counter = 0

    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = os.path.join(output_dir, "latest.pt")

    if resume_path and os.path.exists(resume_path):
        logger.info(f"\n► Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_ema and "ema_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_pcc = ckpt.get("best_pcc", -1.0)
        logger.info(
            f"  Resumed epoch={ckpt['epoch']} | best_pcc={best_pcc:.4f} "
            f"| continuing from epoch {start_epoch}")
    elif resume_path:
        logger.warning(f"  Checkpoint not found at {resume_path}, training from scratch")
    logger.info(
        f"Optimizer: encoder_lr={encoder_lr:.2e}, velocity_lr={lr:.2e}")

    # ── History CSV ──
    history_path = os.path.join(output_dir, "history.csv")
    fields = [
        "epoch", "train_loss", "lr_enc", "lr_vel",
        "grad_avg", "grad_max",
        "val_flow_loss", "val_v_cos",
        "val_fmri_mse", "val_fmri_pcc", "val_fmri_spcc",
    ]
    # On resume: append to existing CSV. On fresh start: write header.
    is_resuming = (start_epoch > 1)
    with open(history_path, "a" if is_resuming else "w", newline="") as f:
        if not is_resuming:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    best_pcc = best_pcc  # already set above (from checkpoint or default -1.0)
    patience_counter = patience_counter  # already set above
    patience = train_cfg.get("patience", 200)

    # ── Training ──
    ts_info = f"timestep_sampling={timestep_sampling}"
    if timestep_sampling == "logit_normal":
        ts_info += f" (mu={logit_normal_mu}, sigma={logit_normal_sigma})"
    logger.info(
        f"Training {num_epochs} epochs, eval every {eval_interval} "
        f"| {ts_info}")
    logger.info(
        f"BRAINFLOW V2 (FlowFM) | OT={use_ot} "
        f"CFG_drop={cfg_drop_prob} Sampler={sampler} "
        f"freeze_enc={freeze_encoder_epochs}ep")
    logger.info(
        f"LOSS: type={velocity_loss_type} | "
        f"t_weight={timestep_weight_type}"
        + (f" | huber_c={huber_c}"
           if velocity_loss_type == "pseudo_huber" else ""))

    # Precompute ROI index tensors was here, removed.

    # ── Mixed Precision (AMP) ──
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        logger.info("Mixed Precision (AMP) enabled — FP16 training")

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()

        # Encoder freeze/unfreeze
        if freeze_encoder_epochs > 0:
            if epoch <= freeze_encoder_epochs:
                if epoch == 1:
                    model.freeze_encoder()
                    logger.info(
                        f"  Encoder FROZEN for {freeze_encoder_epochs} epochs")
            elif epoch == freeze_encoder_epochs + 1:
                model.unfreeze_encoder()
                logger.info("  Encoder UNFROZEN")

        current_lrs = cosine_lr(
            optimizer, epoch - 1, num_epochs, warmup_epochs, base_lrs)

        ep_total, n_steps = 0, 0
        grads_all, grads_max_all = [], []
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            fmri = batch["fmri"].to(device)
            features_dict = {k: v.to(device) for k, v in batch["features"].items()}
            B = fmri.shape[0]

            # Per-ROI Soft Target bypassed since feature dataset maps to continuous TR.
            x1 = fmri

            # ─── Flow Matching: x0=noise, x1=soft_target ──────────────────
            x0 = torch.randn_like(x1)

            # DGS: Dynamic Guidance Switching on representation
            drop_mask = torch.rand(B, device=device) < cfg_drop_prob

            # Timestep sampling
            if timestep_sampling == "logit_normal":
                u = torch.randn(B, device=device)
                t_sample = torch.sigmoid(
                    logit_normal_mu + logit_normal_sigma * u)
                t_expand = t_sample[:, None]
                xt = t_expand * x1 + (1 - t_expand) * x0
                ut = x1 - x0
                t = t_sample
            else:
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)

            # Forward with AMP autocast
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Forward: predict velocity
                cls_token, patch_tokens = model.encoder(features_dict)
                if drop_mask.any():
                    cls_token = cls_token.clone()
                    patch_tokens = patch_tokens.clone()
                    cls_token[drop_mask] = 0.0
                    patch_tokens[drop_mask] = 0.0

                t_emb = timestep_embedding(
                    t * 1000, model.config.hidden_dim).to(device)
                c = model.cond_mlp(cls_token, t_emb)

                context = None
                if model.use_cross_attention:
                    context = model.context_norm(
                        model.context_proj(patch_tokens))

                x_patches = model._patchify(xt)
                x_tokens = model.patch_embed(x_patches)
                x_tokens = x_tokens + model.patch_pos_embed

                for block in model.blocks:
                    x_tokens = block(x_tokens, c, context)

                mod_params = model.final_adaLN(c)
                shift, scale = mod_params.chunk(2, dim=-1)
                x_out = modulate(
                    model.final_layer_norm(x_tokens), shift, scale)
                x_out = model.output_proj(x_out)
                v_pred = model._unpatchify(x_out)

                # Loss
                loss, loss_info = compute_velocity_loss(
                    v_pred, ut, t=t,
                    loss_type=velocity_loss_type,
                    huber_c=huber_c,
                    timestep_weight_type=timestep_weight_type,
                    direction_weight=direction_weight,
                    magnitude_weight=magnitude_weight,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip if grad_clip > 0 else float('inf'))
            scaler.step(optimizer)
            scaler.update()
            if use_ema:
                ema_update(model, ema_model, ema_decay)

            ep_total += loss.item()
            grads_all.append(gn.item())
            grads_max_all.append(gn.item())
            n_steps += 1

            # ── Periodic batch logging ────────────────────────────────────
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                with torch.no_grad():
                    v_cos = F.cosine_similarity(
                        v_pred.detach(), ut.detach(), dim=-1).mean().item()
                elapsed = time.time() - t0
                total_batches = len(train_loader)
                batches_done = batch_idx + 1
                eta = elapsed / batches_done * (total_batches - batches_done)
                logger.info(
                    f"  Ep{epoch} [{batches_done:5d}/{total_batches}] "
                    f"loss={loss.item():.4f} v_cos={v_cos:.4f} "
                    f"grad={gn.item():.3f} "
                    f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m")

        avg_total = ep_total / max(n_steps, 1)
        avg_grad = sum(grads_all) / max(len(grads_all), 1)
        max_grad = max(grads_max_all) if grads_max_all else 0.0
        ep_time = time.time() - t0

        logger.info(
            f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) "
            f"[BRAINFLOW_V2] | loss={avg_total:.5f} | "
            f"lr_enc={current_lrs[0]:.2e} lr_vel={current_lrs[1]:.2e} "
            f"grad={avg_grad:.4f}")

        # ── Eval ──
        if epoch % eval_interval == 0 or epoch == 1:
            eval_model = ema_model if use_ema else model
            val = validate(eval_model, val_loader, fm, device,
                           ode_steps, cfg_scale, num_trials,
                           sampler=sampler)

            row = {
                "epoch": epoch,
                "train_loss": f"{avg_total:.6f}",
                "lr_enc": f"{current_lrs[0]:.2e}",
                "lr_vel": f"{current_lrs[1]:.2e}",
                "grad_avg": f"{avg_grad:.4f}",
                "grad_max": f"{max_grad:.4f}",
                **{k: f"{v:.6f}" if (
                    'loss' in k or 'mse' in k or 'std' in k or
                    'ratio' in k) else f"{v:.4f}"
                   for k, v in val.items()},
            }
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            spcc = val["val_fmri_spcc"]
            is_best = spcc > best_pcc
            logger.info(
                f"  VAL | v_cos={val['val_v_cos']:.4f} | "
                f"f_mse={val['val_fmri_mse']:.4f} "
                f"f_pcc={val['val_fmri_pcc']:.4f} "
                f"f_spcc={spcc:.4f}{'  ★' if is_best else ''}")

            # Per-ROI PCC logging removed.

            if is_best:
                best_pcc = spcc
                patience_counter = 0
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_pcc": best_pcc,
                    "config": cfg,
                }
                torch.save(save_dict,
                           os.path.join(output_dir, "best_model.pt"))
                logger.info(f"  ★ Saved best (PCC={best_pcc:.4f})")
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        # Save latest.pt every epoch (enables resume at any point)
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_pcc": best_pcc,
            "config": cfg,
        }
        if use_ema:
            save_dict["ema_state_dict"] = ema_model.state_dict()
        torch.save(save_dict, os.path.join(output_dir, "latest.pt"))

        # Save periodic snapshot (every save_every epochs)
        if epoch % train_cfg.get("save_every", 50) == 0:
            torch.save(save_dict,
                       os.path.join(output_dir, f"ckpt_ep{epoch:04d}.pt"))

    logger.info(f"Done! Best PCC: {best_pcc:.4f}")


if __name__ == "__main__":
    main()
