"""Precompute MoEVAE encoder features for all clips.

Batch-encodes all multi-modality features through a frozen MoEVAE encoder
and saves the μ (mean) latent vectors as NPY files. This eliminates the
need for online MoEVAE inference during BrainFlow training (~69M frozen
params forward pass per step).

For each clip and each TR:
  8 modalities → [Frozen MoEVAE encoder] → μ (512-dim)
  Stack all TRs → (n_trs, 512) → save as NPY

Usage:
    python src/data/precompute_moevae_features.py \
        --config src/configs/brain_flow_motfm_vae.yaml \
        --moevae_checkpoint outputs/multimodal_vae/best.pt

    # Dry run (just print what would be done):
    python src/data/precompute_moevae_features.py \
        --config src/configs/brain_flow_motfm_vae.yaml \
        --moevae_checkpoint outputs/multimodal_vae/best.pt \
        --dry_run
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_config
from src.models.brainflow.multimodal_vae import MultimodalMoEVAE

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("precompute_moevae")


def load_moevae(cfg: dict, checkpoint_path: str, device: torch.device) -> MultimodalMoEVAE:
    """Load pretrained MoEVAE in eval mode."""
    moevae_cfg = cfg["moevae"]
    mod_cfgs = moevae_cfg["modalities"]
    model_p = moevae_cfg["model"]

    moevae = MultimodalMoEVAE(
        modality_configs=mod_cfgs,
        latent_dim=model_p.get("latent_dim", 512),
        encoder_hidden=model_p.get("encoder_hidden", 1024),
        n_experts=model_p.get("n_experts", 4),
        expert_hidden=model_p.get("expert_hidden", 1024),
        decoder_hidden=model_p.get("decoder_hidden", 1024),
        dropout=model_p.get("dropout", 0.1),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "ema_model" in ckpt:
        moevae.load_state_dict(ckpt["ema_model"])
        logger.info("Loaded MoEVAE from EMA weights")
    elif "model" in ckpt:
        moevae.load_state_dict(ckpt["model"])
        logger.info("Loaded MoEVAE from model weights")
    else:
        moevae.load_state_dict(ckpt)
        logger.info("Loaded MoEVAE state dict")

    moevae = moevae.to(device)
    moevae.eval()
    for p in moevae.parameters():
        p.requires_grad = False

    return moevae


@torch.no_grad()
def encode_clip_moevae(
    moevae: MultimodalMoEVAE,
    clip_features: dict[str, np.ndarray],  # {mod_name: (T, D)}
    device: torch.device,
    chunk_size: int = 256,
) -> np.ndarray:
    """Encode a full clip's multimodal features through MoEVAE → μ.

    Parameters
    ----------
    moevae : MultimodalMoEVAE
    clip_features : dict mapping modality name → (T, D) array
    device : torch.device
    chunk_size : int
        Max TRs to encode at once to avoid OOM.

    Returns
    -------
    latent : (T, Z) float32 — MoEVAE μ for each TR
    """
    # Determine T from the first available modality
    first_feat = next(iter(clip_features.values()))
    T = first_feat.shape[0]

    all_mus = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_len = end - start

        # Gather per-modality chunks
        modality_mus = []
        for mod_name in moevae.modality_names:
            if mod_name in clip_features:
                feat = clip_features[mod_name][start:end]  # (chunk, D)
                x = torch.from_numpy(feat).to(device)

                # Pool layers if needed (input is (chunk, D) = already pooled)
                x_pooled = x
                mu_mod, _ = moevae.encoders[mod_name](x_pooled)  # (chunk, Z)
                modality_mus.append(mu_mod)
            else:
                # Missing modality → zero
                z_dim = moevae.latent_dim
                modality_mus.append(
                    torch.zeros(chunk_len, z_dim, device=device)
                )

        # MoE fusion
        mu_fused, _ = moevae.moe_fusion(modality_mus)  # (chunk, Z)
        all_mus.append(mu_fused.cpu().numpy())

    return np.concatenate(all_mus, axis=0).astype(np.float32)  # (T, Z)


def discover_clips(features_dir: Path, modalities: dict, splits_cfg: dict) -> list[dict]:
    """Discover all clips from the feature NPY directory.

    Returns a list of clip info dicts: {task, stim_type, clip_stem}.
    """
    clips = []
    seen = set()

    # Use the first modality to discover clips
    first_mod_name = next(iter(modalities))
    first_mod_cfg = modalities[first_mod_name]
    subdir = first_mod_cfg.get("subdir", first_mod_name)

    for task, task_splits in splits_cfg.items():
        all_stim_types = []
        for split_stims in task_splits.values():
            all_stim_types.extend(split_stims)

        for stim_type in all_stim_types:
            clip_dir = features_dir / subdir / task / stim_type
            if not clip_dir.exists():
                logger.warning("Directory not found: %s", clip_dir)
                continue

            for npy_file in sorted(clip_dir.glob("*.npy")):
                clip_stem = npy_file.stem
                key = (task, stim_type, clip_stem)
                if key not in seen:
                    seen.add(key)
                    clips.append({
                        "task": task,
                        "stim_type": stim_type,
                        "clip_stem": clip_stem,
                    })

    return clips


def main():
    parser = argparse.ArgumentParser(description="Precompute MoEVAE encoder features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--moevae_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: Data/moevae_latents)")
    parser.add_argument("--chunk_size", type=int, default=256,
                        help="Max TRs to encode at once")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print what would be done, don't encode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Feature directory (pre-pooled NPY: {subdir}/{task}/{stim_type}/{clip}.npy)
    features_dir = Path(PROJECT_ROOT) / cfg.get("raw_features_npy_dir", "Data/features_npy_pooled")
    logger.info("Features dir: %s", features_dir)

    # Modalities config
    modalities = cfg["moevae"]["modalities"]
    # Build subdir mapping: mod_name → subdir (for feature loading)
    raw_modalities = cfg["raw_features"]["modalities"]
    mod_subdir = {}
    for mod_name in modalities:
        if mod_name in raw_modalities:
            mod_subdir[mod_name] = raw_modalities[mod_name].get("subdir", mod_name)
        else:
            mod_subdir[mod_name] = mod_name

    logger.info("Modalities: %s", list(modalities.keys()))
    logger.info("Subdir mapping: %s", mod_subdir)

    # Output directory
    output_dir = Path(PROJECT_ROOT) / (args.output_dir or "Data/moevae_latents")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # Discover clips
    splits_cfg = cfg["splits"]
    clips = discover_clips(features_dir, raw_modalities, splits_cfg)
    logger.info("Found %d clips to process", len(clips))

    if args.dry_run:
        for clip in clips[:10]:
            logger.info("  Would encode: %s/%s/%s",
                        clip["task"], clip["stim_type"], clip["clip_stem"])
        if len(clips) > 10:
            logger.info("  ... and %d more", len(clips) - 10)
        return

    # Load MoEVAE
    moevae = load_moevae(cfg, str(Path(PROJECT_ROOT) / args.moevae_checkpoint), device)
    latent_dim = moevae.latent_dim
    logger.info("MoEVAE latent_dim: %d", latent_dim)

    total_clips = 0
    total_trs = 0
    skipped = 0

    for clip_info in tqdm(clips, desc="Encoding clips"):
        task = clip_info["task"]
        stim_type = clip_info["stim_type"]
        clip_stem = clip_info["clip_stem"]

        # Check if already done
        out_path = output_dir / task / stim_type / f"{clip_stem}.npy"
        if out_path.exists():
            skipped += 1
            continue

        # Load features for all modalities
        clip_features = {}
        min_T = None

        for mod_name in modalities:
            subdir = mod_subdir[mod_name]
            npy_path = features_dir / subdir / task / stim_type / f"{clip_stem}.npy"

            if npy_path.exists():
                feat = np.load(npy_path).astype(np.float32)  # (T, D)
                clip_features[mod_name] = feat
                if min_T is None:
                    min_T = feat.shape[0]
                else:
                    min_T = min(min_T, feat.shape[0])
            else:
                # Modality missing for this clip — will use zeros in encoder
                pass

        if not clip_features:
            logger.warning("No features found for %s/%s/%s", task, stim_type, clip_stem)
            continue

        # Align all modalities to the same T
        for mod_name in clip_features:
            if clip_features[mod_name].shape[0] > min_T:
                clip_features[mod_name] = clip_features[mod_name][:min_T]

        # Encode
        latent = encode_clip_moevae(
            moevae, clip_features, device,
            chunk_size=args.chunk_size,
        )  # (T, Z)

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, latent)

        total_clips += 1
        total_trs += latent.shape[0]

    logger.info("Done! %d clips encoded (%d TRs), %d skipped (already exist)",
                total_clips, total_trs, skipped)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
