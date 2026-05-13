"""Train BrainFlow on BMD (BOLD Moments Dataset) — Event-Related Design.

Unlike Algonauts (time-series sliding windows), BMD is single-trial:
  - Context: (B, D) pooled multimodal features per video
  - Target:  (B, V) single fMRI beta vector per video per subject

Usage:
    python src/train_brainflow_bmd.py --config src/configs/brainflow_bmd.yaml --fast_dev_run
    python src/train_brainflow_bmd.py --config src/configs/brainflow_bmd.yaml
    python src/train_brainflow_bmd.py --config src/configs/brainflow_bmd.yaml --resume
"""

import argparse
import logging
import math as pymath
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.directflow_dataset import load_config
from src.datasets.bmd_dataset import BMDDataset, get_bmd_dataloaders
from src.models.brainflow.brainflow import BrainFlow
from src.train_brainflow import EMAModel, pearson_corr_per_dim

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_brainflow_bmd")


def train(args):
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    if args.train_batch_size is not None:
        cfg.setdefault("dataloader", {})["batch_size"] = int(args.train_batch_size)
    if args.val_batch_size is not None:
        cfg.setdefault("dataloader", {})["val_batch_size"] = int(args.val_batch_size)

    logger.info("Loaded config: %s", cfg_path)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    train_loader, test_loader = get_bmd_dataloaders(cfg)
    _dl = cfg["dataloader"]
    _bs, _vbs = _dl["batch_size"], _dl["val_batch_size"]
    logger.info(
        "DataLoader: train batch=%d → %d batches/epoch; val batch=%d → %d batches",
        _bs, len(train_loader), _vbs, len(test_loader),
    )

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1
        cfg["training"]["val_every_n_epochs"] = 1

    # --- Model ---
    bf_cfg = cfg["brainflow"]
    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])
    vn_params = dict(bf_cfg.get("velocity_net", {}))
    modality_dims = cfg.get("modality_dims", None)
    if modality_dims:
        vn_params["modality_dims"] = modality_dims

    logger.info("BMD mode: output_dim=%d, n_target_trs=%d, context_trs=%d",
                output_dim, vn_params.get("n_target_trs", 1), vn_params.get("context_trs", 1))

    model = BrainFlow(
        output_dim=output_dim,
        velocity_net_params=vn_params,
        n_subjects=len(cfg["subjects"]),
        indi_flow_matching=bf_cfg.get("indi_flow_matching", False),
        indi_train_time_sqrt=bf_cfg.get("indi_train_time_sqrt", False),
        indi_min_denom=bf_cfg.get("indi_min_denom", 1e-3),
        use_csfm=bf_cfg.get("use_csfm", False),
        csfm_var_reg_weight=bf_cfg.get("csfm_var_reg_weight", 0.1),
        csfm_pcc_weight=bf_cfg.get("csfm_pcc_weight", 1.0),
        flow_loss_weight=bf_cfg.get("flow_loss_weight", 1.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # --- Optimizer & Scheduler ---
    tr_cfg = cfg["training"]
    accum_steps = max(1, int(tr_cfg.get("gradient_accumulation_steps", 1)))
    opt_steps_per_epoch = max(1, (len(train_loader) + accum_steps - 1) // accum_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"],
    )

    total_steps = 2 if args.fast_dev_run else opt_steps_per_epoch * tr_cfg["n_epochs"]
    warmup_steps = int(total_steps * tr_cfg.get("warmup_ratio", 0.05))
    min_lr = tr_cfg.get("min_lr", 1e-6)
    base_lr = tr_cfg["lr"]

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    # --- Output ---
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_bmd")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    ema_on_cpu = tr_cfg.get("ema_on_cpu", True)
    ema = EMAModel(model, decay=tr_cfg.get("ema_decay", 0.999), store_on_cpu=ema_on_cpu)
    logger.info("Memory opts: grad_accum=%d (~%d opt-steps/epoch), EMA_on_cpu=%s",
                accum_steps, opt_steps_per_epoch, ema_on_cpu)

    # --- Resume / Warmstart ---
    start_epoch, best_val_corr, global_step = 1, -1.0, 0

    if args.warmstart:
        ws_path = Path(args.warmstart)
        if ws_path.exists():
            ckpt = torch.load(ws_path, map_location=device, weights_only=False)
            state = ckpt.get("model", ckpt)
            model_state = model.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_state and model_state[k].shape == v.shape}
            skipped = [k for k in state if k in model_state and model_state[k].shape != state[k].shape]
            model.load_state_dict(filtered, strict=False)
            if skipped:
                logger.info("Warmstart skipped %d mismatched keys", len(skipped))
            logger.info("Warmstarted from %s, loaded %d/%d keys", ws_path, len(filtered), len(state))
            del ckpt
        else:
            logger.warning("--warmstart path %s not found.", ws_path)

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
            logger.info("Resumed from epoch %d (step=%d)", ckpt["epoch"], global_step)
            del ckpt
        else:
            logger.warning("--resume but no last.pt found.")

    history_file = out_dir / "history.csv"

    solver_cfg = cfg.get("solver_args", {})
    val_n_timesteps = solver_cfg.get("time_points", 30)
    val_solver_method = solver_cfg.get("method", "midpoint")
    val_cfg_scale = solver_cfg.get("cfg_scale", 0.0)
    val_temperature = solver_cfg.get("temperature", 0.0)

    context_bf16 = tr_cfg.get("context_bf16_when_amp", True)
    pin = cfg.get("dataloader", {}).get("pin_memory", False)

    # --- Two-stage training ---
    warmup_hrf_epochs = tr_cfg.get("warmup_hrf_epochs", 0)
    if warmup_hrf_epochs > 0 and start_epoch <= warmup_hrf_epochs:
        logger.info(
            "Two-stage training enabled: Stage 1 (HRF warmup) epochs 1-%d, "
            "Stage 2 (flow training) from epoch %d",
            warmup_hrf_epochs, warmup_hrf_epochs + 1,
        )
        if start_epoch == 1:
            model.freeze_flow_decoder()
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("Stage 1 trainable params: %s", f"{n_train:,}")

    # --- Training loop ---
    for epoch in range(start_epoch, tr_cfg["n_epochs"] + 1):
        # --- Two-stage transition ---
        if warmup_hrf_epochs > 0 and epoch == warmup_hrf_epochs + 1:
            logger.info("=" * 60)
            logger.info("STAGE 2: Unfreezing flow decoder, freezing HRF source")
            logger.info("=" * 60)
            model.unfreeze_flow_decoder()
            model.freeze_hrf_source_only()
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("Stage 2 trainable params: %s", f"{n_train:,}")
            torch.cuda.empty_cache()

        freeze_epoch = tr_cfg.get("freeze_modules_after_epoch", -1)
        if freeze_epoch > 0 and epoch == freeze_epoch + 1:
            logger.info(f"Epoch {epoch} > {freeze_epoch}. Freezing context encoder and HRF source!")
            model.freeze_source_and_context()
            torch.cuda.empty_cache()

        is_hrf_warmup = warmup_hrf_epochs > 0 and epoch <= warmup_hrf_epochs

        model.train()
        train_losses = defaultdict(list)
        micro_accum = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            # BMD context: (B, D) → unsqueeze to (B, 1, D) for VelocityNet
            context = batch["context"].to(device, non_blocking=pin)
            context = context.unsqueeze(1)  # (B, D) → (B, 1, D)

            subject_ids = batch["subject_idx"].to(device, non_blocking=pin)

            # BMD target: (B, V) → unsqueeze to (B, 1, V) for seq2seq compatibility
            target = batch["fmri"].to(device, non_blocking=pin)
            target = target.unsqueeze(1)  # (B, V) → (B, 1, V)

            if tr_cfg["use_amp"] and context_bf16:
                context = context.to(dtype=torch.bfloat16)
                target = target.to(dtype=torch.bfloat16)

            cfg_drop_rate = tr_cfg.get("cfg_dropout", 0.0)
            cfg_drop = cfg_drop_rate > 0 and random.random() < cfg_drop_rate
            if cfg_drop:
                context = torch.zeros_like(context)

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                losses = model.compute_loss(
                    context, target,
                    subject_ids=subject_ids,
                    skip_aux=cfg_drop,
                    skip_flow=is_hrf_warmup,
                )
                raw_loss = losses["total_loss"]
                loss = raw_loss / accum_steps

            loss.backward()
            for k, v in losses.items():
                train_losses[k].append(v.item())

            micro_accum += 1
            if micro_accum >= accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                micro_accum = 0
                global_step += 1

                if global_step % tr_cfg["log_every_n_steps"] == 0:
                    postfix = {
                        "loss": f"{np.mean(train_losses['total_loss'][-50:]):.4f}",
                        "flow": f"{losses['flow_loss'].item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                    if model.use_csfm:
                        postfix["pcc"] = f"{losses['pcc_loss'].item():.4f}"
                        postfix["var"] = f"{losses['var_reg_loss'].item():.4f}"
                    pbar.set_postfix(postfix)

        if micro_accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            global_step += 1

        if tr_cfg.get("empty_cuda_cache_each_epoch", False) and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Validation ---
        # BMD: each video is independent — no overlapping window accumulation
        mean_fmri_corr = 0.0
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()
            logger.info("Running validation...")

            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break

                    # Context: (B, D) → (B, 1, D)
                    context = batch["context"].to(device)
                    context = context.unsqueeze(1)
                    subject_ids = batch["subject_idx"].to(device)

                    if is_hrf_warmup:
                        # Stage 1: DiT decoder is untrained → use HRF source directly
                        gen_fmri = model.synthesise_hrf_direct(
                            context, subject_ids=subject_ids,
                        )
                    else:
                        synth_kwargs = dict(
                            n_timesteps=val_n_timesteps,
                            solver_method=val_solver_method,
                            subject_ids=subject_ids,
                            temperature=val_temperature,
                        )
                        tw = solver_cfg.get("time_grid_warp")
                        if tw:
                            synth_kwargs["time_grid_warp"] = tw
                        if val_cfg_scale > 0:
                            synth_kwargs["cfg_scale"] = val_cfg_scale

                        gen_fmri = model.synthesise(context, **synth_kwargs)

                    fmri_target = batch["fmri"].to(device)

                    # gen_fmri may be (B, 1, V) or (B, V) — flatten to (B, V)
                    if gen_fmri.dim() == 3:
                        gen_fmri = gen_fmri.squeeze(1)
                    if fmri_target.dim() == 3:
                        fmri_target = fmri_target.squeeze(1)

                    all_preds.append(gen_fmri.cpu())
                    all_targets.append(fmri_target.cpu())

            if all_preds:
                all_preds = torch.cat(all_preds, dim=0)    # (N_total, V)
                all_targets = torch.cat(all_targets, dim=0)  # (N_total, V)

                # Per-voxel PCC across all test samples
                pcc = pearson_corr_per_dim(all_preds.unsqueeze(0), all_targets.unsqueeze(0))
                mean_fmri_corr = float(pcc.mean().item())

                # Also compute per-sample PCC (mean across voxels)
                per_sample_pcc = F.cosine_similarity(
                    all_preds - all_preds.mean(dim=-1, keepdim=True),
                    all_targets - all_targets.mean(dim=-1, keepdim=True),
                    dim=-1,
                )
                mean_sample_pcc = float(per_sample_pcc.mean().item())

                val_tag = "[HRF-direct]" if is_hrf_warmup else "[ODE-synth]"
                logger.info("Epoch %d %s | Val per-voxel PCC: %.4f | per-sample PCC: %.4f",
                            epoch, val_tag, mean_fmri_corr, mean_sample_pcc)
            else:
                logger.info("Epoch %d | Val: no predictions generated", epoch)

            if mean_fmri_corr > best_val_corr:
                best_val_corr = mean_fmri_corr
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("Saved new best EMA model (PCC=%.4f)", best_val_corr)

            if not history_file.exists() or history_file.stat().st_size == 0:
                header = "epoch," + ",".join(train_losses.keys()) + ",val_voxel_pcc,val_sample_pcc,lr\n"
                history_file.write_text(header)

            with open(history_file, "a") as f:
                loss_vals = [f"{np.mean(train_losses[k]):.6f}" for k in train_losses.keys()]
                f.write(f"{epoch},{','.join(loss_vals)},{mean_fmri_corr:.6f},"
                        f"{mean_sample_pcc:.6f},{scheduler.get_last_lr()[0]:.2e}\n")

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
    parser.add_argument("--config", type=str, default="src/configs/brainflow_bmd.yaml")
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmstart", type=str, default=None)
    args = parser.parse_args()
    train(args)
