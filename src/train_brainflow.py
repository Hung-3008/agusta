"""Train BrainFlow — Flow Matching with Pre-extracted Context.

Usage:
    python src/train_brainflow.py --config src/configs/brainflow.yaml --fast_dev_run
    python src/train_brainflow.py --config src/configs/brainflow.yaml
    python src/train_brainflow.py --config src/configs/brainflow.yaml --resume
"""

import argparse
import logging
import math as pymath
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.directflow_dataset import load_config
from src.datasets.directflow_dataset import DirectFlowDataset, get_dataloaders
from src.models.brainflow.brainflow import BrainFlow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_brainflow")


def resolve_paths(cfg: dict, project_root: Path) -> dict:
    cfg = cfg.copy()
    cfg["_project_root"] = str(project_root)
    cfg["_data_root"] = str(project_root / cfg["data_root"])
    cfg["_fmri_dir"] = str(project_root / cfg["data_root"] / cfg["fmri"]["dir"])
    return cfg


# =============================================================================
# Metrics & EMA
# =============================================================================

def pearson_corr_per_dim(pred, target):
    """Per-voxel PCC across time. Input: (1, N, D) or (N, D)."""
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    pred_cnt = pred - pred.mean(dim=0, keepdim=True)
    target_cnt = target - target.mean(dim=0, keepdim=True)
    cov = (pred_cnt * target_cnt).sum(dim=0)
    std = torch.sqrt((pred_cnt ** 2).sum(dim=0) * (target_cnt ** 2).sum(dim=0))
    return cov / (std + 1e-8)


class EMAModel:
    """EMA weights. Optional ``store_on_cpu`` frees VRAM (shadow updated on CPU)."""

    def __init__(self, model, decay=0.999, store_on_cpu: bool = False):
        self.decay = decay
        self.store_on_cpu = store_on_cpu
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                sh = param.data.clone()
                self.shadow[name] = sh.cpu() if store_on_cpu else sh

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.store_on_cpu:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data.detach().cpu(), alpha=1 - self.decay
                    )
                else:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                sh = self.shadow[name]
                if self.store_on_cpu:
                    sh = sh.to(device=param.device, dtype=param.dtype, non_blocking=True)
                param.data.copy_(sh)

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def to_cpu_shadow(self):
        """Move shadow tensors to CPU (e.g. after loading an old GPU checkpoint)."""
        if not self.store_on_cpu:
            return
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].cpu()

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = dict(state_dict["shadow"])
        self.decay = state_dict["decay"]
        if self.store_on_cpu:
            for name in self.shadow:
                self.shadow[name] = self.shadow[name].cpu()


# =============================================================================
# Training
# =============================================================================

def train(args):
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    if args.train_batch_size is not None:
        cfg.setdefault("dataloader", {})["batch_size"] = int(args.train_batch_size)
    if args.val_batch_size is not None:
        cfg.setdefault("dataloader", {})["val_batch_size"] = int(args.val_batch_size)

    logger.info("Loaded config: %s", cfg_path)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(cfg)
    _dl = cfg["dataloader"]
    _bs, _vbs = _dl["batch_size"], _dl["val_batch_size"]
    logger.info(
        "DataLoader: train batch_size=%d → %d micro-batches/epoch (~%d samples/epoch, drop_last); "
        "val batch_size=%d",
        _bs,
        len(train_loader),
        len(train_loader) * _bs,
        _vbs,
    )

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1
        cfg["training"]["val_every_n_epochs"] = 1

    # --- Model ---
    bf_cfg = cfg["brainflow"]
    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])
    vn_params = dict(bf_cfg.get("velocity_net", {}))
    logger.info("Context encoder mode: %s", vn_params.get("context_encoder", "multitoken"))
    modality_dims = cfg.get("modality_dims", None)

    if modality_dims:
        vn_params["modality_dims"] = modality_dims

    model = BrainFlow(
        output_dim=output_dim,
        velocity_net_params=vn_params,
        n_subjects=len(cfg["subjects"]),

        tensor_fm_params=bf_cfg.get("tensor_fm", None),
        indi_flow_matching=bf_cfg.get("indi_flow_matching", False),
        indi_train_time_sqrt=bf_cfg.get("indi_train_time_sqrt", False),
        indi_min_denom=bf_cfg.get("indi_min_denom", 1e-3),
        use_csfm=bf_cfg.get("use_csfm", False),
        csfm_var_reg_weight=bf_cfg.get("csfm_var_reg_weight", 0.1),
        csfm_pcc_weight=bf_cfg.get("csfm_pcc_weight", 1.0),
        csfm_align_weight=bf_cfg.get("csfm_align_weight", 1.0),
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
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_direct")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    ema_on_cpu = tr_cfg.get("ema_on_cpu", True)
    ema = EMAModel(
        model, decay=tr_cfg.get("ema_decay", 0.999), store_on_cpu=ema_on_cpu,
    )
    gckpt = vn_params.get("gradient_checkpointing", False)
    logger.info(
        "Memory opts: grad_accum=%d (~%d opt-steps/epoch), EMA_on_cpu=%s, "
        "velocity_grad_ckpt=%s",
        accum_steps, opt_steps_per_epoch, ema_on_cpu, gckpt,
    )
    logger.info("EMA decay=%.4f", ema.decay)

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
            missing, _ = model.load_state_dict(filtered, strict=False)
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
    val_n_timesteps = solver_cfg.get("time_points", 50)
    val_solver_method = solver_cfg.get("method", "midpoint")
    val_cfg_scale = solver_cfg.get("cfg_scale", 0.0)
    val_temperature = solver_cfg.get("temperature", 0.0)

    context_bf16 = tr_cfg.get("context_bf16_when_amp", True)
    pin = cfg.get("dataloader", {}).get("pin_memory", False)

    # --- Training loop ---
    for epoch in range(start_epoch, tr_cfg["n_epochs"] + 1):
        model.train()
        train_losses = defaultdict(list)
        micro_accum = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            context = batch["context"].to(device, non_blocking=pin)
            subject_ids = batch["subject_idx"].to(device, non_blocking=pin)
            target = batch["fmri"].to(device, non_blocking=pin)
            if tr_cfg["use_amp"] and context_bf16:
                context = context.to(dtype=torch.bfloat16)
                target = target.to(dtype=torch.bfloat16)

            cfg_drop = random.random() < 0.1
            if cfg_drop:
                context = torch.zeros_like(context)

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                losses = model.compute_loss(
                    context, target,
                    subject_ids=subject_ids,
                    skip_aux=cfg_drop,
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
                        "pcc": f"{losses['pcc_loss'].item():.4f}",
                        "align": f"{losses['align_loss'].item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
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
        mean_fmri_corr = 0.0
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()
            logger.info("Running validation...")

            fmri_pred_acc, fmri_tgt_acc = {}, {}

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break

                    context = batch["context"].to(device)
                    subject_ids = batch["subject_idx"].to(device)

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

                    clip_keys = batch["clip_key"]
                    target_tr_starts = batch["target_tr_start"]
                    n_trs_batch = batch["n_trs"]
                    fmri_target = batch["fmri"].to(device)

                    B = gen_fmri.shape[0]
                    fmri_dim = gen_fmri.shape[-1]
                    is_seq = gen_fmri.dim() == 3
                    for b in range(B):
                        ck = clip_keys[b]
                        tr_start = int(target_tr_starts[b].item())
                        n_t = int(n_trs_batch[b].item())

                        if ck not in fmri_pred_acc:
                            fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                            fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}

                        if is_seq:
                            # Seq2seq: accumulate each TR offset
                            for off in range(gen_fmri.shape[1]):
                                tr_idx = tr_start + off
                                if tr_idx >= n_t:
                                    break
                                fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b, off].cpu()
                                fmri_pred_acc[ck]["count"][tr_idx] += 1
                                fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b, off].cpu()
                                fmri_tgt_acc[ck]["count"][tr_idx] += 1
                        else:
                            # Legacy single-TR
                            tr_idx = tr_start
                            if tr_idx < n_t:
                                fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b].cpu()
                                fmri_pred_acc[ck]["count"][tr_idx] += 1
                                fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b].cpu()
                                fmri_tgt_acc[ck]["count"][tr_idx] += 1

            all_gen, all_tgt = [], []
            for ck in sorted(fmri_pred_acc.keys()):
                count = fmri_pred_acc[ck]["count"]
                valid = count > 0
                if valid.sum() < 2:
                    continue
                all_gen.append(fmri_pred_acc[ck]["sum"][valid] / count[valid].unsqueeze(-1))
                all_tgt.append(fmri_tgt_acc[ck]["sum"][valid] / fmri_tgt_acc[ck]["count"][valid].unsqueeze(-1))

            if all_gen:
                all_gen = torch.cat(all_gen, dim=0)
                all_tgt = torch.cat(all_tgt, dim=0)
                pcc = pearson_corr_per_dim(all_gen.unsqueeze(0), all_tgt.unsqueeze(0))
                mean_fmri_corr = float(pcc.mean().item())

            logger.info("Epoch %d | Val fMRI PCC: %.4f", epoch, mean_fmri_corr)

            if mean_fmri_corr > best_val_corr:
                best_val_corr = mean_fmri_corr
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("Saved new best EMA model (PCC=%.4f)", best_val_corr)

            if not history_file.exists() or history_file.stat().st_size == 0:
                header = "epoch," + ",".join(train_losses.keys()) + ",val_fmri_pcc,lr\n"
                history_file.write_text(header)

            with open(history_file, "a") as f:
                loss_vals = [f"{np.mean(train_losses[k]):.6f}" for k in train_losses.keys()]
                f.write(f"{epoch},{','.join(loss_vals)},{mean_fmri_corr:.6f},"
                        f"{scheduler.get_last_lr()[0]:.2e}\n")

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
    parser.add_argument("--config", type=str, default="src/configs/brainflow.yaml")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override dataloader.batch_size (YAML on disk must be saved; use this if unsure).",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Override dataloader.val_batch_size.",
    )
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmstart", type=str, default=None)
    args = parser.parse_args()
    train(args)
