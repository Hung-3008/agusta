"""Precompute VAE latents for all fMRI clips.

Batch-encodes all fMRI data through a frozen VAE encoder and saves the
latent representations as NPY files. This eliminates the need for
online VAE encoding during training (the main training bottleneck).

Supports num_trial > 1 for multi-sample averaging (stochastic data augmentation).

Usage:
    python src/data/precompute_latents.py --config src/configs/brain_flow.yaml
    python src/data/precompute_latents.py --config src/configs/brain_flow.yaml --num_trial 3
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_config, resolve_paths
from src.models.brainflow.fmri_vae import build_vae

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("precompute_latents")


def _get_fmri_filepath(fmri_dir: str, subject: str, task: str) -> Path:
    subj_dir = Path(fmri_dir) / subject / "func"
    atlas = "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
    stem = f"{subject}_task-{task}_{atlas}"
    if task == "friends":
        return subj_dir / f"{stem}_desc-s123456_bold.h5"
    else:
        return subj_dir / f"{stem}_bold.h5"


def load_vae(cfg: dict, device: torch.device):
    """Load pretrained VAE in eval mode."""
    vae_ckpt_path = cfg.get("vae_checkpoint")
    if not vae_ckpt_path:
        if "vae_output_dir" in cfg:
            vae_ckpt_path = Path(cfg["vae_output_dir"]) / "best.pt"
        else:
            raise ValueError("Neither vae_checkpoint nor vae_output_dir is specified in config")

    vae_ckpt_path = Path(PROJECT_ROOT) / vae_ckpt_path
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_cfg = ckpt.get("config", cfg)

    vae = build_vae(vae_cfg).to(device)
    if "ema_model" in ckpt:
        vae.load_state_dict(ckpt["ema_model"])
        logger.info("Loaded VAE EMA weights from %s", vae_ckpt_path)
    elif "model" in ckpt:
        vae.load_state_dict(ckpt["model"])
        logger.info("Loaded VAE model weights from %s", vae_ckpt_path)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


@torch.no_grad()
def encode_clip(
    vae, fmri_data: np.ndarray, subject_id: int,
    device: torch.device, num_trial: int = 1,
    chunk_size: int = 200,
) -> np.ndarray:
    """Encode a full clip's fMRI to latent space.

    Parameters
    ----------
    vae : nn.Module
    fmri_data : (n_trs, n_voxels) float32
    subject_id : int
    device : torch.device
    num_trial : int
        1 = deterministic μ, >1 = sample and average.
    chunk_size : int
        Max TRs to encode at once (avoids OOM for long clips).

    Returns
    -------
    latent : (n_trs, latent_dim) float32
    """
    n_trs = fmri_data.shape[0]
    all_latents = []

    for start in range(0, n_trs, chunk_size):
        end = min(start + chunk_size, n_trs)
        chunk = torch.from_numpy(fmri_data[start:end]).unsqueeze(0).to(device)  # (1, T_chunk, V)
        sid = torch.tensor([subject_id], dtype=torch.long, device=device)

        if num_trial <= 1:
            # Deterministic: just use μ
            z = vae.get_latent(chunk, sid)  # (1, T_chunk, Z)
        else:
            # Multi-trial averaging
            zs = []
            for _ in range(num_trial):
                vae.train()  # enable reparameterize sampling
                z_sample, _, _ = vae.encode(chunk, sid)  # (1, T_chunk, Z)
                zs.append(z_sample)
            vae.eval()
            z = torch.stack(zs).mean(dim=0)  # (1, T_chunk, Z)

        all_latents.append(z.squeeze(0).cpu().numpy())  # (T_chunk, Z)

    return np.concatenate(all_latents, axis=0)  # (n_trs, Z)


def main():
    parser = argparse.ArgumentParser(description="Precompute VAE latents")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_trial", type=int, default=None,
                        help="Override num_trial from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load VAE
    vae = load_vae(cfg, device)

    # Config
    fmri_dir = cfg["_fmri_dir"]
    subjects = cfg["subjects"]
    splits_cfg = cfg["splits"]
    excl_start = cfg["fmri"].get("excluded_samples_start", 0)
    excl_end = cfg["fmri"].get("excluded_samples_end", 0)
    standardize = cfg["preprocessing"]["fmri"]["standardize"] == "zscore_sample"

    bf_cfg = cfg.get("brainflow", {})
    num_trial = args.num_trial or bf_cfg.get("num_trial", 1)

    # Output directory
    latent_dir = Path(PROJECT_ROOT) / cfg.get("latent_dir", "Data/latents")
    latent_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s (num_trial=%d)", latent_dir, num_trial)

    # Subject → ID mapping (for VAE)
    # Note: VAE was trained with use_subject=False, so subject_id=0 works for all
    vae_ckpt_path = cfg.get("vae_checkpoint")
    if not vae_ckpt_path:
        vae_ckpt_path = Path(cfg["vae_output_dir"]) / "best.pt"
    vae_cfg = torch.load(
        Path(PROJECT_ROOT) / vae_ckpt_path, map_location="cpu"
    ).get("config", {})
    vae_use_subject = vae_cfg.get("vae", {}).get("use_subject", False)

    total_clips = 0
    total_trs = 0

    # Discover clips from PCA features dir (same clips that training uses)
    pca_dir = Path(PROJECT_ROOT) / cfg.get("pca_features", {}).get("output_dir", "Data/pca_features")

    for task, task_splits in splits_cfg.items():
        for split_name, stim_types in task_splits.items():
            for stim_type in stim_types:
                # Check PCA features exist for this stim
                pca_stim_dir = pca_dir / task / stim_type
                if not pca_stim_dir.exists():
                    logger.debug("No PCA dir: %s", pca_stim_dir)
                    continue

                clip_names = sorted(p.stem for p in pca_stim_dir.glob("*.npy"))

                for clip_name in tqdm(clip_names, desc=f"{task}/{stim_type}"):
                    # Determine fMRI key
                    fmri_key = clip_name
                    if task == "friends" and clip_name.startswith("friends_"):
                        fmri_key = clip_name[len("friends_"):]
                    elif task == "movie10" and clip_name.startswith("movie10_"):
                        fmri_key = clip_name[len("movie10_"):]

                    for subject in subjects:
                        subject_id = 0 if not vae_use_subject else subjects.index(subject)

                        # Output path
                        out_path = latent_dir / subject / task / stim_type / f"{clip_name}.npy"
                        if out_path.exists():
                            total_clips += 1
                            continue

                        # Load fMRI
                        fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
                        if not fmri_path.exists():
                            continue

                        try:
                            with h5py.File(fmri_path, "r") as f:
                                matched = [k for k in f.keys() if fmri_key in k]
                                if not matched:
                                    continue
                                raw = f[matched[0]]
                                end = len(raw) - excl_end if excl_end > 0 else len(raw)
                                data = raw[excl_start:end].astype(np.float32)
                        except Exception as e:
                            logger.warning("Skip %s/%s: %s", subject, clip_name, e)
                            continue

                        if standardize:
                            mean = data.mean(axis=0, keepdims=True)
                            std = data.std(axis=0, keepdims=True)
                            std = np.where(std < 1e-8, 1.0, std)
                            data = (data - mean) / std

                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                        # Encode
                        latent = encode_clip(
                            vae, data, subject_id, device,
                            num_trial=num_trial,
                        )  # (n_trs, Z)

                        # Save
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(out_path, latent)

                        total_clips += 1
                        total_trs += latent.shape[0]

    logger.info("Done! %d clips, %d total TRs encoded to %s",
                total_clips, total_trs, latent_dir)


if __name__ == "__main__":
    main()
