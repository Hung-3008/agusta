"""Precompute DAE encoder features for all clips.

Encodes multi-modality features through a frozen DAE encoder
and saves the deterministic latent vectors as NPY files.

Usage:
    python src/data/precompute_dae_features.py \
        --config src/configs/multimodal_dae.yaml \
        --checkpoint outputs/multimodal_dae/best.pt

    python src/data/precompute_dae_features.py \
        --config src/configs/multimodal_dae.yaml \
        --checkpoint outputs/multimodal_dae/best.pt \
        --output_dir Data/dae_latents
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
from src.models.brainflow.multimodal_dae import MultimodalDAE

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("precompute_dae")


def load_dae(cfg: dict, checkpoint_path: str, device: torch.device) -> MultimodalDAE:
    """Load pretrained DAE in eval mode."""
    model_cfg = cfg["model"]
    modality_configs = {
        name: {"dim": mc["dim"]}
        for name, mc in cfg["modalities"].items()
    }

    model = MultimodalDAE(
        modality_configs=modality_configs,
        latent_dim=model_cfg["latent_dim"],
        encoder_hidden=model_cfg["encoder_hidden"],
        n_experts=model_cfg["n_experts"],
        expert_hidden=model_cfg["expert_hidden"],
        decoder_hidden=model_cfg["decoder_hidden"],
        dropout=model_cfg.get("dropout", 0.1),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    logger.info("Loaded DAE state dict")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def encode_clip(
    model: MultimodalDAE,
    clip_features: dict[str, np.ndarray],
    device: torch.device,
    chunk_size: int = 256,
) -> np.ndarray:
    """Encode clip features → (T, Z) deterministic latent."""
    first_feat = next(iter(clip_features.values()))
    T = first_feat.shape[0]
    all_z = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        batch = {}
        for mod_name, feat in clip_features.items():
            batch[mod_name] = torch.from_numpy(feat[start:end]).to(device)

        z = model.get_latent(batch)  # (chunk, Z)
        all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def discover_clips(features_dir: Path, modalities: dict, splits_cfg: dict) -> list[dict]:
    """Discover all clips from the first modality's directory."""
    clips = []
    seen = set()

    first_mod = next(iter(modalities))
    first_subdir = modalities[first_mod].get("subdir", first_mod)

    for task, task_splits in splits_cfg.items():
        all_stim_types = []
        for split_stims in task_splits.values():
            all_stim_types.extend(split_stims)

        for stim_type in all_stim_types:
            clip_dir = features_dir / first_subdir / task / stim_type
            if not clip_dir.exists():
                continue
            for npy_file in sorted(clip_dir.glob("*.npy")):
                key = (task, stim_type, npy_file.stem)
                if key not in seen:
                    seen.add(key)
                    clips.append({"task": task, "stim_type": stim_type, "clip_stem": npy_file.stem})

    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    features_dir = Path(PROJECT_ROOT) / cfg.get("features_dir", "Data/features_npy_pooled")
    modalities = cfg["modalities"]
    mod_subdirs = {name: mc.get("subdir", name) for name, mc in modalities.items()}

    output_dir = Path(PROJECT_ROOT) / (args.output_dir or "Data/dae_latents")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    clips = discover_clips(features_dir, modalities, cfg["splits"])
    logger.info("Found %d clips to process", len(clips))

    # Load model
    model = load_dae(cfg, str(Path(PROJECT_ROOT) / args.checkpoint), device)
    logger.info("DAE latent_dim: %d", model.latent_dim)

    total_clips, total_trs, skipped = 0, 0, 0

    for clip_info in tqdm(clips, desc="Encoding clips"):
        task = clip_info["task"]
        stim_type = clip_info["stim_type"]
        clip_stem = clip_info["clip_stem"]

        out_path = output_dir / task / stim_type / f"{clip_stem}.npy"
        if out_path.exists():
            skipped += 1
            continue

        # Load features
        clip_features = {}
        min_T = None
        for mod_name in modalities:
            subdir = mod_subdirs[mod_name]
            npy_path = features_dir / subdir / task / stim_type / f"{clip_stem}.npy"
            if npy_path.exists():
                feat = np.load(npy_path).astype(np.float32)
                clip_features[mod_name] = feat
                min_T = feat.shape[0] if min_T is None else min(min_T, feat.shape[0])

        if not clip_features:
            continue

        for mod_name in clip_features:
            if clip_features[mod_name].shape[0] > min_T:
                clip_features[mod_name] = clip_features[mod_name][:min_T]

        latent = encode_clip(model, clip_features, device, chunk_size=args.chunk_size)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, latent)
        total_clips += 1
        total_trs += latent.shape[0]

    logger.info("Done! %d clips encoded (%d TRs), %d skipped", total_clips, total_trs, skipped)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
