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
    return_logvar: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
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
    return_logvar : bool
        If True, also return log_var for posterior sampling.

    Returns
    -------
    latent : (n_trs, latent_dim) float32
    logvar : (n_trs, latent_dim) float32 or None
    """
    n_trs = fmri_data.shape[0]
    all_latents = []
    all_logvars = [] if return_logvar else None

    for start in range(0, n_trs, chunk_size):
        end = min(start + chunk_size, n_trs)
        chunk = torch.from_numpy(fmri_data[start:end]).unsqueeze(0).to(device)  # (1, T_chunk, V)
        sid = torch.tensor([subject_id], dtype=torch.long, device=device)

        if num_trial <= 1:
            # Deterministic: just use μ
            z = vae.get_latent(chunk, sid)  # (1, T_chunk, Z)
            if return_logvar:
                _, mu_out, logvar = vae.encode(chunk, sid)
                all_logvars.append(logvar.squeeze(0).cpu().numpy())
        else:
            # Multi-trial averaging
            zs = []
            for _ in range(num_trial):
                vae.train()  # enable reparameterize sampling
                z_sample, _, _ = vae.encode(chunk, sid)  # (1, T_chunk, Z)
                zs.append(z_sample)
            vae.eval()
            z = torch.stack(zs).mean(dim=0)  # (1, T_chunk, Z)
            if return_logvar:
                _, mu_out, logvar = vae.encode(chunk, sid)
                all_logvars.append(logvar.squeeze(0).cpu().numpy())

        all_latents.append(z.squeeze(0).cpu().numpy())  # (T_chunk, Z)

    latent = np.concatenate(all_latents, axis=0)  # (n_trs, Z)
    logvar_out = np.concatenate(all_logvars, axis=0) if return_logvar else None
    return latent, logvar_out


def main():
    parser = argparse.ArgumentParser(description="Precompute VAE latents")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_trial", type=int, default=None,
                        help="Override num_trial from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Handle both v1 and v2 config formats
    if "features" in cfg:
        cfg = resolve_paths(cfg, PROJECT_ROOT)
    else:
        cfg["_project_root"] = str(PROJECT_ROOT)
        cfg["_data_root"] = str(PROJECT_ROOT / cfg["data_root"])
        cfg["_fmri_dir"] = str(PROJECT_ROOT / cfg["data_root"] / cfg["fmri"]["dir"])

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
    standardize = cfg.get("preprocessing", {}).get("fmri", {}).get("standardize", "none") == "zscore_sample"

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

    for subject in subjects:
        subject_id = 0 if not vae_use_subject else subjects.index(subject)

        for task in splits_cfg.keys():
            # Load fMRI H5 file for this subject/task
            fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
            if not fmri_path.exists():
                logger.warning("fMRI file not found: %s", fmri_path)
                continue

            # Get all stim_types from splits (train + test)
            all_stim_types = []
            for split_name, stim_types in splits_cfg[task].items():
                all_stim_types.extend(stim_types)

            with h5py.File(fmri_path, "r") as f:
                all_keys = sorted(f.keys())
                logger.info("Task %s: %d clips in fMRI H5", task, len(all_keys))

                for clip_key in tqdm(all_keys, desc=f"{subject}/{task}"):
                    # Determine stim_type for this clip
                    stim_type = None

                    if task == "friends":
                        # Keys like "ses-001_task-s01e02a"
                        # Extract episode name after "task-": "s01e02a"
                        # Season = "s" + first 2 digits = "s01" → map to "s1"
                        import re
                        m = re.search(r'task-s(\d+)e', clip_key)
                        if m:
                            season_num = int(m.group(1))
                            season_key = f"s{season_num}"
                            if season_key in all_stim_types:
                                stim_type = season_key
                    else:
                        # movie10: keys like "bourne_01", stim_types like "bourne"
                        for st in all_stim_types:
                            if st in clip_key:
                                stim_type = st
                                break

                    if stim_type is None:
                        continue

                    # Normalize clip key to match feature NPY convention
                    # "ses-003_task-s01e01a" → "s01e01a"
                    # "ses-001_task-bourne01" → "bourne01"
                    # "ses-001_task-figures01_run-1" → "figures01" (strip _run-X)
                    import re
                    clip_name = re.sub(r'^ses-\d+_task-', '', clip_key)
                    clip_name = re.sub(r'_run-\d+$', '', clip_name)

                    # Output path
                    out_path = latent_dir / subject / task / stim_type / f"{clip_name}.npy"
                    logvar_path = out_path.parent / f"{clip_name}_logvar.npy"
                    if out_path.exists() and logvar_path.exists():
                        total_clips += 1
                        continue

                    # Load fMRI data
                    try:
                        raw = f[clip_key]
                        end = len(raw) - excl_end if excl_end > 0 else len(raw)
                        data = raw[excl_start:end].astype(np.float32)
                    except Exception as e:
                        logger.warning("Skip %s/%s: %s", subject, clip_key, e)
                        continue

                    if standardize:
                        mean = data.mean(axis=0, keepdims=True)
                        std = data.std(axis=0, keepdims=True)
                        std = np.where(std < 1e-8, 1.0, std)
                        data = (data - mean) / std

                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                    # Encode through VAE
                    latent, logvar = encode_clip(
                        vae, data, subject_id, device,
                        num_trial=num_trial,
                        return_logvar=True,
                    )  # (n_trs, Z)

                    # Save mu (latent)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, latent)

                    # Save logvar for posterior sampling
                    if logvar is not None:
                        logvar_path = out_path.parent / f"{clip_name}_logvar.npy"
                        np.save(logvar_path, logvar)

                    total_clips += 1
                    total_trs += latent.shape[0]

    logger.info("Done! %d clips, %d total TRs encoded to %s",
                total_clips, total_trs, latent_dir)


if __name__ == "__main__":
    main()
