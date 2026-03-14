import argparse
import logging
import random
import sys
import time
from pathlib import Path
from functools import lru_cache

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_config, resolve_paths
from src.models.brainflow.brain_flow import BrainFlowCFM
from src.data.precompute_latents import load_vae

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_cfm")


def _get_fmri_filepath(fmri_dir: str, subject: str, task: str) -> Path:
    """Resolve fMRI H5 file path (same logic as precompute_latents)."""
    subj_dir = Path(fmri_dir) / subject / "func"
    atlas = "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
    stem = f"{subject}_task-{task}_{atlas}"
    if task == "friends":
        return subj_dir / f"{stem}_desc-s123456_bold.h5"
    else:
        return subj_dir / f"{stem}_bold.h5"


class BrainFlowDataset(Dataset):
    """
    Loads pre-computed PCA features and VAE latents.
    Yields sliding windows of (features, latents).
    For val split, also loads ground truth fMRI for metric computation.
    """
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.subjects = cfg["subjects"]
        self.splits_cfg = cfg.get("splits", {})
        
        self.latent_dir = Path(PROJECT_ROOT) / cfg["latent_dir"]
        self.pca_dir = Path(PROJECT_ROOT) / cfg["pca_features"]["output_dir"]
        
        # fMRI ground truth access (for val)
        self.fmri_dir = Path(cfg["_fmri_dir"])
        self.excl_start = cfg["fmri"].get("excluded_samples_start", 0)
        self.excl_end = cfg["fmri"].get("excluded_samples_end", 0)
        
        # Sliding window params
        self.tr = cfg["fmri"]["tr"]
        self.context_dur = cfg["sliding_window"]["context_duration"]
        self.stride_dur = cfg["sliding_window"]["stride"]
        self.seq_len = max(1, int(round(self.context_dur / self.tr)))
        self.stride = max(1, int(round(self.stride_dur / self.tr)))
        
        self.val_ratio = cfg.get("val_ratio", 0.1)
        self.samples = self._build_index()
        
        # Cache for opened H5 files (val only)
        self._fmri_cache = {}

    def _build_index(self):
        samples = []
        logger.info("Building %s split index...", self.split)
        
        for task, splits in self.splits_cfg.items():
            for split_name, stim_types in splits.items():
                for stim_type in stim_types:
                    pca_stim_dir = self.pca_dir / task / stim_type
                    if not pca_stim_dir.exists():
                        continue
                        
                    clip_names = sorted([p.stem for p in pca_stim_dir.glob("*.npy")])
                    
                    # Deterministic val split based on clip names
                    rng = random.Random(42)
                    shuffled_clips = clip_names.copy()
                    rng.shuffle(shuffled_clips)
                    
                    n_val = int(len(shuffled_clips) * self.val_ratio)
                    val_clips = set(shuffled_clips[:n_val])
                    
                    for clip_name in clip_names:
                        is_val = clip_name in val_clips
                        if (self.split == "train" and is_val) or (self.split == "val" and not is_val):
                            continue
                            
                        # Load shapes to determine sliding windows
                        try:
                            # Use subject 0's latent shape as reference
                            subj = self.subjects[0]
                            lat_path = self.latent_dir / subj / task / stim_type / f"{clip_name}.npy"
                            feat_path = pca_stim_dir / f"{clip_name}.npy"
                            
                            if not lat_path.exists() or not feat_path.exists():
                                continue
                                
                            # We just need the temporal length. We assume feat and latent have same length.
                            # In precompute scripts, features and latents should be aligned to the same TRs.
                            # We lazy-load using np.load(mmap_mode="r") to check shape quickly.
                            lat_shape = np.load(lat_path, mmap_mode="r").shape
                            n_trs = lat_shape[0]
                            
                            if n_trs < 10:  # need at least 10 TRs to be meaningful
                                continue
                                
                            # TRIBE-style: one window per clip (or multiple non-overlapping)
                            # If clip is shorter than seq_len, we record the actual length
                            # and pad in __getitem__
                            for start_idx in range(0, n_trs, self.stride):
                                actual_len = min(self.seq_len, n_trs - start_idx)
                                if actual_len < 10:  # skip tiny tail fragments
                                    break
                                samples.append({
                                    "task": task,
                                    "stim_type": stim_type,
                                    "clip_name": clip_name,
                                    "start_idx": start_idx,
                                    "actual_len": actual_len,
                                })
                        except Exception as e:
                            logger.warning(f"Error reading shape for {clip_name}: {e}")
                            
        logger.info("Found %d windows for %s split.", len(samples), self.split)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_fmri_clip(self, subject, task, clip_name):
        """Load ground truth fMRI for a clip from H5 file (cached)."""
        cache_key = (subject, task, clip_name)
        if cache_key in self._fmri_cache:
            return self._fmri_cache[cache_key]
        
        fmri_path = _get_fmri_filepath(str(self.fmri_dir), subject, task)
        # Determine fMRI key inside H5
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
        self._fmri_cache[cache_key] = data
        return data

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        task = sample_info["task"]
        stim_type = sample_info["stim_type"]
        clip_name = sample_info["clip_name"]
        start = sample_info["start_idx"]
        actual_len = sample_info.get("actual_len", self.seq_len)
        end = start + actual_len
        
        # Load features (shared across subjects)
        feat_path = self.pca_dir / task / stim_type / f"{clip_name}.npy"
        feat_data = np.load(feat_path, mmap_mode="r")[start:end].astype(np.float32)
        
        # Load latent (for now, single subject mode uses subjects[0])
        subj = self.subjects[0]
        lat_path = self.latent_dir / subj / task / stim_type / f"{clip_name}.npy"
        lat_data = np.load(lat_path, mmap_mode="r")[start:end].astype(np.float32)
        
        # Zero-pad to seq_len if this is a short tail fragment
        if actual_len < self.seq_len:
            pad = self.seq_len - actual_len
            feat_data = np.pad(feat_data, ((0, pad), (0, 0)), mode="constant")
            lat_data  = np.pad(lat_data,  ((0, pad), (0, 0)), mode="constant")
        
        result = {
            "features": torch.from_numpy(feat_data.copy()),
            "latent": torch.from_numpy(lat_data.copy()),
            "valid_len": actual_len,  # for CFM mask in training and PCC mask in eval
        }
        
        # For val split, also load ground truth fMRI
        if self.split == "val":
            fmri_data = self._load_fmri_clip(subj, task, clip_name)
            if fmri_data is not None:
                fmri_window = fmri_data[start:end].astype(np.float32)
                if actual_len < self.seq_len:
                    pad = self.seq_len - actual_len
                    fmri_window = np.pad(fmri_window, ((0, pad), (0, 0)), mode="constant")
                result["fmri"] = torch.from_numpy(fmri_window.copy())
        
        return result


def get_dataloaders(cfg):
    train_set = BrainFlowDataset(cfg, split="train")
    val_set = BrainFlowDataset(cfg, split="val")
    
    dl_cfg = cfg["dataloader"]
    
    train_loader = DataLoader(
        train_set,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=dl_cfg["val_batch_size"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False
    )
    
    return train_loader, val_loader


def pearson_corr(pred, target, valid_lens=None):
    """Compute Pearson correlation across time for each feature/voxel, then mean over features.
    
    Args:
        pred, target: (B, T, D)
        valid_lens: optional LongTensor (B,) — number of valid (non-padded) TRs per sample.
                    If provided, padded TRs are excluded from the computation.
    """
    if valid_lens is not None:
        # Collect only valid TRs for each sample in the batch
        rows_p, rows_t = [], []
        for b in range(pred.shape[0]):
            vl = int(valid_lens[b].item())
            rows_p.append(pred[b, :vl, :])
            rows_t.append(target[b, :vl, :])
        pred   = torch.cat(rows_p, dim=0)   # (total_valid_T, D)
        target = torch.cat(rows_t, dim=0)
    else:
        pred   = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])
    
    # Mean across time for each voxel
    pred_mean   = pred.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    
    pred_cnt   = pred   - pred_mean
    target_cnt = target - target_mean
    
    # Covariance and Variance
    cov = (pred_cnt * target_cnt).sum(dim=0)
    std = torch.sqrt((pred_cnt**2).sum(dim=0) * (target_cnt**2).sum(dim=0))
    
    corr = cov / (std + 1e-8)
    return corr.mean().item()


def train(args):
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Init Data
    train_loader, val_loader = get_dataloaders(cfg)
    
    if args.fast_dev_run:
        logger.info("Fast dev run mode")
        cfg["training"]["n_epochs"] = 1
        cfg["training"]["val_every_n_epochs"] = 1
    
    # 2. Init Model
    logger.info("Initializing BrainFlowCFM...")
    bf_cfg = cfg["brainflow"]
    model = BrainFlowCFM(
        feat_dim=bf_cfg["feat_dim"],
        latent_dim=bf_cfg["latent_dim"],
        encoder_params=bf_cfg["encoder"],
        decoder_params=bf_cfg["decoder"],
        cfm_params=bf_cfg["cfm"],
        cfg_drop_prob=bf_cfg.get("cfg_drop_prob", 0.1),
    ).to(device)
    
    # 3. Load frozen VAE for validation (optional but good for tracking fMRI Pearson)
    vae = None
    try:
        vae = load_vae(cfg, device)
        logger.info("Loaded VAE for computing fMRI-level Pearson correlation.")
    except Exception as e:
        logger.warning(f"Failed to load VAE: {e}. Will only compute Latent Pearson.")
    
    # 4. Optimizer & Scheduler
    tr_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"]
    )
    
    total_steps = len(train_loader) * tr_cfg["n_epochs"]
    if args.fast_dev_run:
        total_steps = 2
        
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=tr_cfg["lr"],
        total_steps=total_steps,
        pct_start=tr_cfg["warmup_ratio"],
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=tr_cfg["use_amp"])
    
    # 5. Output dir & WandB
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brain_flow_cfm")
    out_dir.mkdir(parents=True, exist_ok=True)
    

        
    # 6. Training Loop
    best_val_corr = -1.0
    global_step = 0
    mean_corr = 0.0
    mean_fmri_corr = 0.0
    
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
                
            features = batch["features"].to(device)
            latents = batch["latent"].to(device)
            
            # Build CFM mask: 0 for zero-padded TRs, 1 for valid TRs
            B, T, _ = latents.shape
            mask = torch.ones(B, 1, T, device=device)
            if "valid_len" in batch:
                for b, vl in enumerate(batch["valid_len"]):
                    mask[b, 0, int(vl):] = 0.0
            
            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"]):
                loss = model(features, latents, mask=mask)
                
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_losses.append(loss.item())
            global_step += 1
            
            if global_step % tr_cfg["log_every_n_steps"] == 0:
                pbar.set_postfix({"loss": f"{np.mean(train_losses[-50:]):.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                pass
        # Validation
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            model.eval()
            val_lat_corr = []
            val_fmri_corr = []
            
            logger.info("Running validation...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break
                        
                    features = batch["features"].to(device)
                    latents = batch["latent"].to(device)
                    valid_lens = batch.get("valid_len", None)  # (B,) or None for full-length
                    if valid_lens is not None:
                        valid_lens = valid_lens.to(device)
                    
                    # Generate latents using ODE solver with CFG
                    gen_latents = model.synthesise(
                        features,
                        n_timesteps=20,
                        temperature=1.0,
                        guidance_scale=bf_cfg.get("cfg_guidance_scale", 2.0),
                    )
                    
                    batch_corr = pearson_corr(gen_latents, latents, valid_lens=valid_lens)
                    val_lat_corr.append(batch_corr)
                    
                    if vae is not None and "fmri" in batch:
                        true_fmri = batch["fmri"].to(device)  # ground truth from scanner
                        dummy_subj = torch.zeros(features.shape[0], dtype=torch.long, device=device)
                        gen_fmri = vae.decode(gen_latents, dummy_subj)
                        val_fmri_corr.append(pearson_corr(gen_fmri, true_fmri, valid_lens=valid_lens))
                        
            mean_corr = float(np.mean(val_lat_corr))
            mean_fmri_corr = float(np.mean(val_fmri_corr)) if val_fmri_corr else 0.0
            logger.info(f"Epoch {epoch} | Val Latent Pearson Corr: {mean_corr:.4f} | Val fMRI Pearson Corr: {mean_fmri_corr:.4f}")
            
            # Save best
            if mean_corr > best_val_corr:
                best_val_corr = mean_corr
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info(f"Saved new best model to {out_dir / 'best.pt'}")
                
            # Save history
            mean_train_loss = float(np.mean(train_losses))
            current_lr = scheduler.get_last_lr()[0]
            with open(history_file, "a") as f:
                f.write(f"{epoch},{mean_train_loss:.6f},{mean_corr:.6f},{mean_fmri_corr:.6f},{current_lr:.2e}\n")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, out_dir / "last.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to brain_flow.yaml")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches to test pipeline")
    args = parser.parse_args()
    train(args)
