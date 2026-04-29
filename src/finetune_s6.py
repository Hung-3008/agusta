"""Finetune BrainFlow on s6 — no validation, save every N epochs.

Continues training from a pre-trained checkpoint on the s6 split only.
No validation is performed; checkpoints are saved periodically.

Usage:
    python src/finetune_s6.py --config src/configs/brainflow_finetune_s6.yaml \
        --checkpoint outputs/brainflow_dit_base_new_subjlayer/best.pt
    python src/finetune_s6.py --config src/configs/brainflow_finetune_s6.yaml \
        --checkpoint outputs/brainflow_dit_base_new_subjlayer/best.pt \
        --n_epochs 100 --save_every 5 --lr 5e-5
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
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.directflow_dataset import load_config, DirectFlowDataset, ClipGroupedBatchSampler
from src.models.brainflow.brainflow import BrainFlow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("finetune_s6")


def resolve_paths(cfg: dict, project_root: Path) -> dict:
    cfg = cfg.copy()
    cfg["_project_root"] = str(project_root)
    cfg["_data_root"] = str(project_root / cfg["data_root"])
    cfg["_fmri_dir"] = str(project_root / cfg["data_root"] / cfg["fmri"]["dir"])
    return cfg


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

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = dict(state_dict["shadow"])
        self.decay = state_dict["decay"]
        if self.store_on_cpu:
            for name in self.shadow:
                self.shadow[name] = self.shadow[name].cpu()


def get_s6_train_loader(cfg):
    """Build a train-only DataLoader using splits from config."""
    train_set = DirectFlowDataset(cfg, split="train")
    dl_cfg = cfg["dataloader"]
    batch_size = dl_cfg["batch_size"]
    num_workers = int(dl_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_set,
        batch_sampler=ClipGroupedBatchSampler(train_set, batch_size, drop_last=True),
        num_workers=num_workers,
        pin_memory=dl_cfg.get("pin_memory", False),
    )
    return train_loader


def finetune(args):
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    if args.train_batch_size is not None:
        cfg.setdefault("dataloader", {})["batch_size"] = int(args.train_batch_size)

    logger.info("Loaded config: %s", cfg_path)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data: s6 only ---
    train_loader = get_s6_train_loader(cfg)
    _bs = cfg["dataloader"]["batch_size"]
    logger.info(
        "S6 DataLoader: batch_size=%d → %d micro-batches/epoch (~%d samples/epoch)",
        _bs, len(train_loader), len(train_loader) * _bs,
    )

    # --- Model ---
    bf_cfg = cfg["brainflow"]
    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])
    vn_params = dict(bf_cfg.get("velocity_net", {}))
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
        flow_loss_weight=bf_cfg.get("flow_loss_weight", 1.0),
    ).to(device)

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support both full checkpoint (with "model" key) and raw state_dict
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    logger.info("Loaded checkpoint from %s (%d keys)", ckpt_path, len(state))
    del ckpt

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # --- Training hyperparams ---
    tr_cfg = cfg["training"]
    n_epochs = args.n_epochs if args.n_epochs is not None else tr_cfg["n_epochs"]
    save_every = args.save_every if args.save_every is not None else tr_cfg.get("save_every", 5)
    lr = args.lr if args.lr is not None else tr_cfg["lr"]

    accum_steps = max(1, int(tr_cfg.get("gradient_accumulation_steps", 1)))
    opt_steps_per_epoch = max(1, (len(train_loader) + accum_steps - 1) // accum_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=tr_cfg["weight_decay"],
    )

    total_steps = opt_steps_per_epoch * n_epochs
    warmup_steps = int(total_steps * tr_cfg.get("warmup_ratio", 0.05))
    min_lr = tr_cfg.get("min_lr", 1e-6)
    base_lr = lr

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    # --- Output ---
    base_out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_direct")
    out_dir = base_out_dir / "finetune_s6"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    ema_on_cpu = tr_cfg.get("ema_on_cpu", True)
    ema = EMAModel(
        model, decay=tr_cfg.get("ema_decay", 0.999), store_on_cpu=ema_on_cpu,
    )
    logger.info("EMA decay=%.4f, on_cpu=%s", ema.decay, ema_on_cpu)

    # --- Resume finetune ---
    start_epoch, global_step = 1, 0
    if args.resume:
        resume_path = out_dir / "last.pt"
        if resume_path.exists():
            rckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(rckpt["model"])
            optimizer.load_state_dict(rckpt["optimizer"])
            scheduler.load_state_dict(rckpt["scheduler"])
            if "ema" in rckpt:
                ema.load_state_dict(rckpt["ema"])
            start_epoch = rckpt["epoch"] + 1
            global_step = rckpt.get("global_step", 0)
            logger.info("Resumed finetune from epoch %d (step=%d)", rckpt["epoch"], global_step)
            del rckpt
        else:
            logger.warning("--resume but no last.pt found in %s", out_dir)

    context_bf16 = tr_cfg.get("context_bf16_when_amp", True)
    pin = cfg.get("dataloader", {}).get("pin_memory", False)
    history_file = out_dir / "history.csv"

    logger.info(
        "Finetuning on s6: %d epochs, save_every=%d, lr=%.2e, accum=%d",
        n_epochs, save_every, lr, accum_steps,
    )

    # --- Training loop ---
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        train_losses = defaultdict(list)
        micro_accum = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"S6 Epoch {epoch}/{n_epochs}")
        for batch_idx, batch in enumerate(pbar):
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
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                    if model.use_tensor_fm:
                        postfix["g_reg"] = f"{losses['gamma_reg'].item():.4f}"
                    pbar.set_postfix(postfix)

        # Handle leftover micro-batch
        if micro_accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            global_step += 1

        if tr_cfg.get("empty_cuda_cache_each_epoch", False) and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Log history ---
        if not history_file.exists() or history_file.stat().st_size == 0:
            header = "epoch," + ",".join(train_losses.keys()) + ",lr\n"
            history_file.write_text(header)

        with open(history_file, "a") as f:
            loss_vals = [f"{np.mean(train_losses[k]):.6f}" for k in train_losses.keys()]
            f.write(f"{epoch},{','.join(loss_vals)},{scheduler.get_last_lr()[0]:.2e}\n")

        # --- Save last.pt always ---
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "global_step": global_step,
        }, out_dir / "last.pt")

        # --- Save EMA checkpoint every save_every epochs ---
        if epoch % save_every == 0:
            ema.apply_shadow(model)
            ema_state = model.state_dict()
            torch.save(ema_state, out_dir / f"epoch_{epoch}.pt")
            logger.info("Saved EMA checkpoint: epoch_%d.pt", epoch)
            ema.restore(model)

    # --- Final save ---
    ema.apply_shadow(model)
    torch.save(model.state_dict(), out_dir / "final.pt")
    logger.info("Finetuning complete. Saved final EMA model to %s/final.pt", out_dir)
    ema.restore(model)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Finetune BrainFlow on s6 (no validation)")
    parser.add_argument("--config", type=str, default="src/configs/brainflow_finetune_s6.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pre-trained checkpoint (best.pt or last.pt)")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Override n_epochs from config")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Override save_every from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--train-batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--resume", action="store_true",
                        help="Resume finetune from last.pt in output dir")
    args = parser.parse_args()
    finetune(args)
