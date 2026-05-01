"""finetune_sub05.py — Fine-tune only sub-05 subject head on S6 data.

Freezes the entire backbone. Only trains:
  - velocity_net.subject_layers (all heads, but gradient only flows through sub-05 samples)
  - velocity_net.subject_emb (all embeddings, grad only for sub-05 batches)

This is the minimal intervention to improve sub-05 without touching other subjects.

Usage:
    python src/finetune_sub05.py \\
        --checkpoint outputs/brainflow_dit_base_new_subjlayer_all_data/last.pt \\
        --output_dir outputs/brainflow_sub05_finetune \\
        --n_epochs 30 \\
        --lr 5e-5
"""

import argparse
import logging
import math as pymath
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.directflow_dataset import load_config, DirectFlowDataset
from src.models.brainflow.brainflow import BrainFlow
from src.train_brainflow import EMAModel, pearson_corr_per_dim
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("finetune_sub05")

# sub-05 is index 3 in subjects list [sub-01, sub-02, sub-03, sub-05]
SUB05_IDX = 3


def build_model(cfg: dict, device: torch.device) -> BrainFlow:
    bf = cfg["brainflow"]
    vn_params = dict(bf.get("velocity_net", {}))
    modality_dims = cfg.get("modality_dims")
    if modality_dims:
        vn_params["modality_dims"] = modality_dims

    model = BrainFlow(
        output_dim=bf.get("output_dim", cfg["fmri"]["n_voxels"]),
        velocity_net_params=vn_params,
        n_subjects=len(cfg["subjects"]),
        tensor_fm_params=bf.get("tensor_fm", None),
        indi_flow_matching=bf.get("indi_flow_matching", False),
        indi_train_time_sqrt=bf.get("indi_train_time_sqrt", False),
        indi_min_denom=bf.get("indi_min_denom", 1e-3),
        use_csfm=bf.get("use_csfm", False),
        csfm_var_reg_weight=bf.get("csfm_var_reg_weight", 0.1),
        csfm_pcc_weight=bf.get("csfm_pcc_weight", 1.0),
        flow_loss_weight=bf.get("flow_loss_weight", 1.0),
    ).to(device)
    return model


def freeze_backbone(model: BrainFlow):
    """Freeze everything except subject_layers and subject_emb."""
    total, frozen, trainable = 0, 0, 0
    trainable_names = []

    for name, param in model.named_parameters():
        total += param.numel()
        is_subject_param = (
            "subject_layers" in name or
            "subject_emb" in name
        )
        if is_subject_param:
            param.requires_grad = True
            trainable += param.numel()
            trainable_names.append(name)
        else:
            param.requires_grad = False
            frozen += param.numel()

    log.info("Frozen: %s params | Trainable: %s params",
             f"{frozen:,}", f"{trainable:,}")
    log.info("Trainable param groups: %s", trainable_names[:10])
    return trainable_names


def load_checkpoint(model: BrainFlow, checkpoint_path: Path, device: torch.device):
    """Load EMA weights from last.pt."""
    log.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "ema" in ckpt:
        # Apply EMA shadow weights
        ema = EMAModel(model, decay=0.999, store_on_cpu=False)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow(model)
        log.info("  Loaded EMA shadow weights from checkpoint")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        log.info("  Loaded model weights from checkpoint")
    else:
        model.load_state_dict(ckpt)
        log.info("  Loaded plain state_dict")
    del ckpt


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sub-05 subject head")
    parser.add_argument("--config", default="src/configs/brainflow_all_data.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to last.pt checkpoint (EMA weights will be used)")
    parser.add_argument("--output_dir", default="outputs/brainflow_sub05_finetune")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Config: use S6 only for fine-tuning sub-05 ──────────────────
    cfg = load_config(args.config)
    cfg["_fmri_dir"] = str(PROJECT_ROOT / cfg["data_root"] / cfg["fmri"]["dir"])

    # Override splits: only train on S6 (has ground truth for all subjects)
    cfg["splits"] = {
        "friends": {"train": ["s6"], "val": []},
        "movie10": {"train": [], "val": []},
    }
    cfg["dataloader"]["batch_size"] = args.batch_size
    cfg["dataloader"]["val_batch_size"] = args.batch_size
    cfg["sliding_window"]["stride"] = 5   # denser for fine-tuning
    cfg["sliding_window"]["temporal_jitter"] = 0

    log.info("Building dataset (S6 only, all subjects)...")
    train_set = DirectFlowDataset(cfg, split="train")

    # Filter to sub-05 samples only
    sub05_samples = [s for s in train_set.samples if s["subject"] == "sub-05"]
    log.info("sub-05 training samples: %d / %d total", len(sub05_samples), len(train_set.samples))
    train_set.samples = sub05_samples

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    load_checkpoint(model, Path(args.checkpoint), device)
    freeze_backbone(model)

    # ── Optimizer ────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.n_epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    base_lr = args.lr
    min_lr = 1e-6

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    # ── Output ───────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ema = EMAModel(model, decay=0.999, store_on_cpu=True)
    global_step = 0
    best_loss = float("inf")

    log.info("Starting fine-tuning: %d epochs, lr=%.1e, %d batches/epoch",
             args.n_epochs, args.lr, len(train_loader))

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        losses_epoch = defaultdict(list)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}")
        for batch in pbar:
            context = batch["context"].to(device)
            subject_ids = batch["subject_idx"].to(device)
            target = batch["fmri"].to(device)

            # Verify all samples are sub-05
            assert (subject_ids == SUB05_IDX).all(), \
                f"Expected all sub-05 (idx={SUB05_IDX}), got {subject_ids.unique()}"

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                losses = model.compute_loss(
                    context, target,
                    subject_ids=subject_ids,
                    skip_aux=False,
                )
                loss = losses["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            global_step += 1

            for k, v in losses.items():
                losses_epoch[k].append(v.item())

            if global_step % 20 == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(losses_epoch['total_loss'][-50:]):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        mean_loss = np.mean(losses_epoch["total_loss"])
        log.info("Epoch %d | loss=%.4f | flow=%.4f",
                 epoch, mean_loss,
                 np.mean(losses_epoch.get("flow_loss", [0])))

        # Save checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, out_dir / "last.pt")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            log.info("  Saved best (loss=%.4f)", best_loss)

    log.info("Fine-tuning complete. Best loss: %.4f", best_loss)
    log.info("Checkpoint: %s", out_dir / "last.pt")
    log.info("")
    log.info("Now evaluate with:")
    log.info("  python src/evaluate_brainflow.py \\")
    log.info("    --config src/configs/brainflow_all_data.yaml \\")
    log.info("    --eval_session s7 \\")
    log.info("    --checkpoint %s/last.pt \\", out_dir)
    log.info("    --temperature 0.0 --stride 5")


if __name__ == "__main__":
    main()
