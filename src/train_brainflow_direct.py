"""Train BrainFlow Direct — Flow Matching in fMRI Space with Pre-extracted Context.

No encoder needed. Context latents are loaded from pre-extracted .npy files.

Usage:
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct.yaml --fast_dev_run
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct.yaml
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct.yaml --resume
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

from src.data.dataset import load_config, _get_fmri_filepath
from src.models.brainflow.brain_flow_direct import BrainFlowDirect

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_brainflow_direct")


def resolve_paths(cfg: dict, project_root: Path) -> dict:
    cfg = cfg.copy()
    cfg["_project_root"] = str(project_root)
    cfg["_data_root"] = str(project_root / cfg["data_root"])
    cfg["_fmri_dir"] = str(project_root / cfg["data_root"] / cfg["fmri"]["dir"])
    return cfg


# =============================================================================
# Dataset — Pre-extracted context + raw fMRI (RAM preloaded)
# =============================================================================

class DirectFlowDataset(Dataset):
    """Loads pre-extracted context latents + raw fMRI, fully preloaded into RAM.

    All data is loaded during __init__ for zero file I/O during training.
    Each sample: windowed context (feat_seq_len, context_dim) → 1 fMRI (V,).
    """

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.subjects = cfg["subjects"]
        self.splits_cfg = cfg.get("splits", {})

        self.fmri_dir = Path(cfg["_fmri_dir"])
        self.context_dir = Path(PROJECT_ROOT) / cfg["context_latent_dir"]
        self.context_dim = cfg.get("context_dim", 512)

        self.tr = cfg["fmri"]["tr"]
        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end = cfg["fmri"].get("excluded_samples_end", 0)

        self.feature_context_trs = cfg["sliding_window"].get("feature_context_trs", 10)
        self.feat_seq_len = self.feature_context_trs + 1  # 10 past + 1 current
        self.stride = cfg["sliding_window"].get("stride", 1)
        self.hrf_delay = cfg["fmri"].get("hrf_delay", 5)
        self.temporal_jitter = cfg["sliding_window"].get("temporal_jitter", 0) if split == "train" else 0

        # Load global fMRI normalization stats (per-voxel mean/std from training set)
        self.use_global_stats = cfg["fmri"].get("use_global_stats", False)
        self.fmri_mean = None
        self.fmri_std = None
        if self.use_global_stats:
            subj = self.subjects[0]
            stats_dir = self.fmri_dir / subj / "stats"
            mean_path = stats_dir / "global_mean.npy"
            std_path = stats_dir / "global_std.npy"
            if mean_path.exists() and std_path.exists():
                self.fmri_mean = np.load(mean_path).astype(np.float32)
                self.fmri_std = np.load(std_path).astype(np.float32)
                logger.info("Loaded global fMRI stats for %s (mean_avg=%.4f, std_avg=%.4f)",
                           subj, self.fmri_mean.mean(), self.fmri_std.mean())
            else:
                logger.warning("Global stats not found at %s, falling back to no normalization", stats_dir)
                self.use_global_stats = False

        # Preloaded data: keyed by clip identifier
        self._clips = {}  # clip_id → {"ctx": np.array, "fmri": np.array}
        self.samples = self._build_index_and_preload()

    def _normalize_clip_name(self, clip_name: str, task: str) -> str:
        prefix = f"{task}_"
        if clip_name.startswith(prefix):
            return clip_name[len(prefix):]
        return clip_name

    def _load_fmri_clip(self, subject, task, clip_name):
        """Load fMRI from H5 (called only during init)."""
        fmri_path = _get_fmri_filepath(str(self.fmri_dir), subject, task)
        fmri_key = clip_name
        if task == "friends" and clip_name.startswith("friends_"):
            fmri_key = clip_name[len("friends_"):]
        elif task == "movie10" and clip_name.startswith("movie10_"):
            fmri_key = clip_name[len("movie10_"):]

        with h5py.File(fmri_path, "r") as f:
            matched = [k for k in f.keys() if fmri_key in k]
            if not matched:
                return None
            raw = f[matched[0]]
            end = len(raw) - self.excl_end if self.excl_end > 0 else len(raw)
            data = raw[self.excl_start:end].astype(np.float32)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Global normalization (per-voxel, from training set statistics)
        if self.use_global_stats and self.fmri_mean is not None:
            data = (data - self.fmri_mean[None, :]) / self.fmri_std[None, :]
        
        return data

    def _build_index_and_preload(self):
        """Build sample index and preload all data into RAM."""
        samples = []
        total_bytes = 0
        n_clips = 0
        fmri_cache = {}  # (subj, task, name) → np.array — dedup H5 reads
        logger.info("Building %s split index and preloading data...", self.split)

        for task, splits in self.splits_cfg.items():
            stim_types = splits.get(self.split, [])
            if not stim_types:
                continue

            for stim_type in stim_types:
                ctx_clip_dir = self.context_dir / task / stim_type
                if not ctx_clip_dir.exists():
                    logger.warning("Context dir missing: %s", ctx_clip_dir)
                    continue

                clip_stems = sorted(p.stem for p in ctx_clip_dir.glob("*.npy"))

                for clip_stem in clip_stems:
                    subj = self.subjects[0]
                    norm_name = self._normalize_clip_name(clip_stem, task)

                    # Load fMRI (with dedup)
                    fmri_key = (subj, task, norm_name)
                    if fmri_key not in fmri_cache:
                        fmri_data = self._load_fmri_clip(subj, task, norm_name)
                        if fmri_data is None:
                            fmri_data = self._load_fmri_clip(subj, task, clip_stem)
                        fmri_cache[fmri_key] = fmri_data
                    fmri_data = fmri_cache[fmri_key]
                    if fmri_data is None:
                        continue

                    n_trs = fmri_data.shape[0]

                    # Load context
                    ctx_path = ctx_clip_dir / f"{clip_stem}.npy"
                    if not ctx_path.exists():
                        ctx_path = ctx_clip_dir / f"{norm_name}.npy"
                    if not ctx_path.exists():
                        logger.warning("Context missing for %s/%s/%s", task, stim_type, clip_stem)
                        continue
                    ctx_data = np.load(ctx_path).astype(np.float32)

                    # Store in preloaded dict
                    clip_id = f"{task}/{stim_type}/{norm_name}"
                    self._clips[clip_id] = {
                        "ctx": ctx_data,   # (n_trs_ctx, context_dim)
                        "fmri": fmri_data, # (n_trs, n_voxels)
                    }
                    total_bytes += ctx_data.nbytes + fmri_data.nbytes
                    n_clips += 1

                    for target_tr in range(0, n_trs, self.stride):
                        samples.append({
                            "clip_id": clip_id,
                            "task": task,
                            "stim_type": stim_type,
                            "clip_stem": clip_stem,
                            "norm_name": norm_name,
                            "target_tr": target_tr,
                            "n_trs": n_trs,
                        })

        logger.info("Preloaded %d clips (%.1f MB) → %d samples for %s split.",
                     n_clips, total_bytes / 1e6, len(samples), self.split)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        clip_id = info["clip_id"]
        target_tr = info["target_tr"]
        n_trs = info["n_trs"]

        clip = self._clips[clip_id]
        ctx_data = clip["ctx"]
        fmri_data = clip["fmri"]

        # Feature window
        actual_movie_tr = target_tr + self.excl_start
        feat_current_tr = actual_movie_tr - self.hrf_delay
        if self.temporal_jitter > 0:
            jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
            feat_current_tr = feat_current_tr + jitter

        feat_start = feat_current_tr - self.feature_context_trs
        feat_end = feat_current_tr + 1

        # Window context (pure numpy slicing, no file I/O)
        safe_start = max(0, feat_start)
        safe_end = min(ctx_data.shape[0], feat_end)
        if safe_start < safe_end:
            ctx = ctx_data[safe_start:safe_end]
        else:
            ctx = np.zeros((0, ctx_data.shape[-1]), dtype=np.float32)
        if ctx.shape[0] < self.feat_seq_len:
            pad_len = self.feat_seq_len - ctx.shape[0]
            ctx = np.pad(ctx, ((pad_len, 0), (0, 0)), mode="constant")
        context = torch.from_numpy(ctx.copy())

        # fMRI target (direct array access)
        if target_tr < fmri_data.shape[0]:
            fmri_target = torch.from_numpy(fmri_data[target_tr].copy())
        else:
            n_voxels = self.cfg["fmri"].get("n_voxels", 1000)
            fmri_target = torch.zeros(n_voxels, dtype=torch.float32)

        return {
            "context": context,           # (feat_seq_len, context_dim)
            "fmri": fmri_target,           # (V,)
            "clip_key": clip_id,
            "target_tr": target_tr,
            "n_trs": n_trs,
        }


# =============================================================================
# Batch Sampler
# =============================================================================

class ClipGroupedBatchSampler(Sampler):
    """Groups windows from the same clip into the same batch."""

    def __init__(self, dataset, batch_size, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.clip_groups = defaultdict(list)
        for idx, info in enumerate(dataset.samples):
            clip_key = (info["task"], info["stim_type"], info["clip_stem"])
            self.clip_groups[clip_key].append(idx)

        self.clip_keys = list(self.clip_groups.keys())

    def __iter__(self):
        clip_order = list(self.clip_keys)
        random.shuffle(clip_order)

        all_indices = []
        for clip_key in clip_order:
            indices = self.clip_groups[clip_key]
            random.shuffle(indices)
            all_indices.extend(indices)

        batch = []
        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(sum(self.clip_groups.values(), []))
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# =============================================================================
# Metrics
# =============================================================================

def pearson_corr_per_dim(pred, target):
    """Per-voxel PCC across time. Input: (1, N, D) or (N, D)."""
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    pred_cnt = pred - pred.mean(dim=0, keepdim=True)
    target_cnt = target - target.mean(dim=0, keepdim=True)
    cov = (pred_cnt * target_cnt).sum(dim=0)
    std = torch.sqrt((pred_cnt**2).sum(dim=0) * (target_cnt**2).sum(dim=0))
    return cov / (std + 1e-8)


# =============================================================================
# EMA
# =============================================================================

class EMAModel:
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
# Dataloaders
# =============================================================================

def get_dataloaders(cfg):
    train_set = DirectFlowDataset(cfg, split="train")
    val_set = DirectFlowDataset(cfg, split="val")

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


# =============================================================================
# Training
# =============================================================================

def train(args):
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data
    train_loader, val_loader = get_dataloaders(cfg)

    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1
        cfg["training"]["val_every_n_epochs"] = 1

    # 2. Model
    logger.info("Initializing BrainFlowDirect...")
    bf_cfg = cfg["brainflow"]
    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])

    model = BrainFlowDirect(
        output_dim=output_dim,
        velocity_net_params=bf_cfg.get("velocity_net", {}),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # 3. Optimizer & Scheduler
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
    base_lr = tr_cfg["lr"]

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    # 4. Output dir
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_direct")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    # EMA
    ema_decay = tr_cfg.get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)
    logger.info("EMA initialized with decay=%.4f", ema_decay)

    # Resume
    start_epoch = 1
    best_val_corr = -1.0
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
            logger.info("Resumed from epoch %d (global_step=%d)", ckpt["epoch"], global_step)
            del ckpt
        else:
            logger.warning("--resume specified but no last.pt found. Starting from scratch.")

    history_file = out_dir / "history.csv"
    if start_epoch == 1:
        with open(history_file, "w") as f:
            f.write("epoch,train_loss,val_fmri_pcc,lr\n")

    # Solver config
    solver_cfg = cfg.get("solver_args", {})
    val_n_timesteps = solver_cfg.get("time_points", 50)
    val_solver_method = solver_cfg.get("method", "midpoint")

    # 5. Training loop
    for epoch in range(start_epoch, tr_cfg["n_epochs"] + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            context = batch["context"].to(device)     # (B, T, C)
            fmri_target = batch["fmri"].to(device)     # (B, V)

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                loss = model(context, fmri_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            global_step += 1
            ema.update(model)

            if global_step % tr_cfg["log_every_n_steps"] == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(train_losses[-50:]):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Validation
        mean_fmri_corr = 0.0
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()
            logger.info("Running validation...")

            fmri_pred_acc = {}
            fmri_tgt_acc = {}

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break

                    context = batch["context"].to(device)
                    fmri_target = batch["fmri"].to(device)

                    gen_fmri = model.synthesise(
                        context,
                        n_timesteps=val_n_timesteps,
                        solver_method=val_solver_method,
                    )  # (B, V)

                    clip_keys = batch["clip_key"]
                    target_trs = batch["target_tr"]
                    n_trs_batch = batch["n_trs"]

                    B = gen_fmri.shape[0]
                    for b in range(B):
                        ck = clip_keys[b]
                        tr_idx = int(target_trs[b].item())
                        n_t = int(n_trs_batch[b].item())
                        fmri_dim = gen_fmri.shape[-1]

                        if ck not in fmri_pred_acc:
                            fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                            fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}

                        fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b].cpu()
                        fmri_pred_acc[ck]["count"][tr_idx] += 1
                        fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b].cpu()
                        fmri_tgt_acc[ck]["count"][tr_idx] += 1

            # Average and compute PCC
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
            else:
                mean_fmri_corr = 0.0

            logger.info("Epoch %d | Val fMRI PCC: %.4f", epoch, mean_fmri_corr)

            if mean_fmri_corr > best_val_corr:
                best_val_corr = mean_fmri_corr
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("Saved new best EMA model (PCC=%.4f)", best_val_corr)

            mean_train_loss = float(np.mean(train_losses))
            current_lr = scheduler.get_last_lr()[0]
            with open(history_file, "a") as f:
                f.write(f"{epoch},{mean_train_loss:.6f},{mean_fmri_corr:.6f},{current_lr:.2e}\n")

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
    parser.add_argument("--config", type=str, default="src/configs/brain_flow_direct.yaml")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches to test pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    args = parser.parse_args()
    train(args)
