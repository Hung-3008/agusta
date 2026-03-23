"""Extract fMRI VAE latent vectors offline for all subjects and clips.

Reads raw fMRI from H5 files, encodes through frozen VAE encoder,
and saves per-subject per-clip latent .npy files for downstream
Latent Flow Matching.

Output structure (mirrors fMRI H5 structure):
  outputs/fmri_vae_64/latents/{subject}/{task}/{clip_key}.npy
  Each file: shape (T_trimmed, latent_dim), float32

Usage:
  python src/extract_vae_latents.py --config src/configs/fmri_vae.yaml
  python src/extract_vae_latents.py --config src/configs/fmri_vae.yaml --checkpoint outputs/fmri_vae_64/best.pt
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import load_config, resolve_paths, _get_fmri_filepath
from models.brainflow.fmri_vae import build_vae

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("extract_latents")


def extract_latents_for_clip(
    model: torch.nn.Module,
    fmri_data: np.ndarray,
    subject_id: int,
    device: torch.device,
    seq_len: int = 100,
) -> np.ndarray:
    """Encode a full fMRI clip through the VAE encoder.

    Handles clips longer than seq_len by chunking with overlap,
    then stitching the middle portions together.

    Parameters
    ----------
    model : fMRI_VAE_v5
        Frozen VAE model.
    fmri_data : np.ndarray
        Shape (T, V), raw fMRI after trimming excluded samples.
    subject_id : int
        Integer subject index.
    device : torch.device
    seq_len : int
        Window size for encoding (should match training seq_len).

    Returns
    -------
    np.ndarray
        Shape (T, Z), latent vectors (mu, deterministic).
    """
    T, V = fmri_data.shape
    Z = model.latent_dim

    if T <= seq_len:
        # Short clip: encode in one go
        fmri_t = torch.from_numpy(fmri_data).unsqueeze(0).to(device)  # (1, T, V)
        sid_t = torch.tensor([subject_id], device=device)
        mu = model.get_latent(fmri_t, sid_t)  # (1, T, Z)
        return mu.squeeze(0).cpu().numpy()

    # Long clip: sliding window with overlap for smooth stitching
    stride = seq_len // 2  # 50% overlap
    latents = np.zeros((T, Z), dtype=np.float32)
    counts = np.zeros((T, 1), dtype=np.float32)

    for start in range(0, T, stride):
        end = min(start + seq_len, T)
        if end - start < 10:  # Skip tiny tail
            break

        chunk = fmri_data[start:end]
        fmri_t = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, chunk_len, V)
        sid_t = torch.tensor([subject_id], device=device)

        mu = model.get_latent(fmri_t, sid_t)  # (1, chunk_len, Z)
        mu_np = mu.squeeze(0).cpu().numpy()

        latents[start:end] += mu_np
        counts[start:end] += 1.0

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    latents /= counts

    return latents


def main():
    parser = argparse.ArgumentParser(description="Extract VAE latent vectors offline")
    parser.add_argument("--config", type=str, default="src/configs/fmri_vae.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to VAE checkpoint (default: {vae_output_dir}/best.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {vae_output_dir}/latents)")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use EMA model weights (default: True)")
    args = parser.parse_args()

    # ── Config ──
    cfg = load_config(PROJECT_ROOT / args.config)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    vae_output_dir = Path(PROJECT_ROOT) / cfg.get("vae_output_dir", "outputs/fmri_vae")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else vae_output_dir / "best.pt"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = vae_output_dir / "latents"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    logger.info("Loading VAE from %s", ckpt_path)
    model = build_vae(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if args.use_ema and "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
        logger.info("✅ Loaded EMA model weights")
    else:
        model.load_state_dict(ckpt["model"])
        logger.info("✅ Loaded model weights")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    seq_len = cfg["vae"].get("seq_len", 100)
    latent_dim = cfg["vae"].get("latent_dim", 64)

    # ── Build subject mapping (must match training order!) ──
    subjects = cfg["subjects"]
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    logger.info("Subjects: %s", subjects)

    # ── Iterate through all subjects and clips ──
    fmri_dir = cfg["_fmri_dir"]
    excl_start = cfg["fmri"].get("excluded_samples_start", 0)
    excl_end = cfg["fmri"].get("excluded_samples_end", 0)
    splits_cfg = cfg.get("splits", {})

    total_clips = 0
    total_trs = 0
    stats = defaultdict(lambda: {"clips": 0, "trs": 0})

    for subject in subjects:
        subj_idx = subject_to_idx[subject]
        logger.info("Processing %s (idx=%d)...", subject, subj_idx)

        for task in splits_cfg.keys():
            fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
            if not fmri_path.exists():
                logger.warning("  fMRI file not found: %s", fmri_path)
                continue

            with h5py.File(fmri_path, "r") as f:
                clip_keys = sorted(f.keys())

            for clip_key in clip_keys:
                # Load raw fMRI
                with h5py.File(fmri_path, "r") as f:
                    raw = f[clip_key][:]
                    end = len(raw) - excl_end if excl_end > 0 else len(raw)
                    fmri_data = raw[excl_start:end].astype(np.float32)

                fmri_data = np.nan_to_num(fmri_data, nan=0.0, posinf=0.0, neginf=0.0)
                T_trimmed = fmri_data.shape[0]

                # Extract latents
                with torch.no_grad():
                    latents = extract_latents_for_clip(
                        model, fmri_data, subj_idx, device, seq_len
                    )

                # Save: outputs/fmri_vae_64/latents/{subject}/{task}/{clip_key}.npy
                save_dir = output_dir / subject / task
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{clip_key}.npy"
                np.save(save_path, latents.astype(np.float32))

                total_clips += 1
                total_trs += T_trimmed
                stats[subject]["clips"] += 1
                stats[subject]["trs"] += T_trimmed

            logger.info("  %s/%s: %d clips extracted", subject, task, stats[subject]["clips"])

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("Extraction complete!")
    logger.info("  Output dir:  %s", output_dir)
    logger.info("  Total clips: %d", total_clips)
    logger.info("  Total TRs:   %d", total_trs)
    logger.info("  Latent dim:  %d", latent_dim)
    logger.info("  Per-subject stats:")
    for subj in subjects:
        s = stats[subj]
        logger.info("    %s: %d clips, %d TRs", subj, s["clips"], s["trs"])

    # ── Verify a random file ──
    verify_dir = output_dir / subjects[0]
    npy_files = list(verify_dir.rglob("*.npy"))
    if npy_files:
        sample = np.load(npy_files[0])
        logger.info("  Sample file: %s → shape %s, dtype %s",
                     npy_files[0].name, sample.shape, sample.dtype)
        logger.info("  Sample stats: mean=%.4f, std=%.4f, min=%.3f, max=%.3f",
                     sample.mean(), sample.std(), sample.min(), sample.max())

    logger.info("=" * 60)
    logger.info("Done! Latents ready for Latent Flow Matching.")


if __name__ == "__main__":
    main()
