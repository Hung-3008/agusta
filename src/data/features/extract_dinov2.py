"""Extract per-TR visual embeddings using DINOv2 (frozen ViT backbone).

DINOv2 captures low-level visual features (edges, textures, spatial layout)
that CLIP/SigLIP miss due to language bias. Excellent for early visual cortex
(V1–V4) prediction in brain encoding.

Produces .npy files with shape (n_trs, embed_dim) float32 in
Data/features_npy_pooled/dinov2_{variant}/.

Usage:
    # Giant model (dim=1536, ~5GB VRAM with bf16)
    python src/data/features/extract_dinov2.py \\
        --model_variant giant --movie_type friends --stimulus_type s1 --dry_run

    # Large model (dim=1024, ~2GB VRAM)
    python src/data/features/extract_dinov2.py \\
        --model_variant large --movie_type friends --stimulus_type s1

    # Full extraction (all clips)
    python src/data/features/extract_dinov2.py \\
        --movie_type friends --stimulus_type s1
"""

import argparse
import gc
import glob
import logging
import os

# Suppress OpenCV ffmpeg warnings (e.g., interlaced to progressive)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TR = 1.49
REF_MODALITY = "whisper"

MODEL_CONFIGS = {
    "giant": {
        "hub_name": "facebook/dinov2-giant",
        "embed_dim": 1536,
        "output_name": "dinov2_giant",
    },
    "large": {
        "hub_name": "facebook/dinov2-large",
        "embed_dim": 1024,
        "output_name": "dinov2_large",
    },
    "base": {
        "hub_name": "facebook/dinov2-base",
        "embed_dim": 768,
        "output_name": "dinov2_base",
    },
}


# ---------------------------------------------------------------------------
# ffmpeg / video helpers
# ---------------------------------------------------------------------------
def _get_ffmpeg_exe() -> str:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError("ffmpeg not found!")


def get_video_duration(video_path: str) -> float:
    """Get video duration using cv2 (no ffprobe needed)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0 and frame_count > 0:
        return frame_count / fps
    raise RuntimeError(f"Cannot determine duration of {video_path}")


def extract_frames_from_segment(
    video_path: str, start: float, duration: float, n_frames: int = 4,
) -> list[np.ndarray]:
    """Extract n_frames from a video segment using cv2.

    Returns list of BGR frames (H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []

    start_frame = int(start * fps)
    end_frame = int((start + duration) * fps)
    total_seg_frames = max(end_frame - start_frame, 1)

    # Sample n_frames uniformly from the segment
    indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# DINOv2 Extractor
# ---------------------------------------------------------------------------
class DINOv2Extractor:
    """Load DINOv2 model for per-frame visual feature extraction."""

    def __init__(self, model_variant: str = "giant", device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel

        cfg = MODEL_CONFIGS[model_variant]
        self.embed_dim = cfg["embed_dim"]
        self.output_name = cfg["output_name"]

        logger.info("Loading DINOv2 model: %s", cfg["hub_name"])
        self.processor = AutoImageProcessor.from_pretrained(cfg["hub_name"])
        self.model = AutoModel.from_pretrained(
            cfg["hub_name"],
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(
            "DINOv2 loaded (variant=%s, dim=%d, %.0fM params, device=%s)",
            model_variant, self.embed_dim, n_params, device,
        )

    @torch.inference_mode()
    def extract_embeddings_batch(
        self, frames: list[np.ndarray], frames_per_tr: int = 4,
        num_temporal_pools: int = 4
    ) -> np.ndarray:
        """Extract embeddings from a batched list of BGR frames for multiple TRs.
        Splits frames within each TR into `num_temporal_pools` continuous segments
        and pools each segment independently.

        Returns:
            np.ndarray of shape (batch_trs, num_temporal_pools, embed_dim)
        """
        if not frames:
            return np.zeros((0, num_temporal_pools, self.embed_dim), dtype=np.float32)

        batch_trs = len(frames) // frames_per_tr

        # Convert BGR → RGB
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        # Process all frames in a single batch
        inputs = self.processor(images=rgb_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # CLS token: (N, dim) where N = batch_trs * frames_per_tr
        cls_tokens = outputs.last_hidden_state[:, 0, :]

        # Group by TR: (batch_trs, frames_per_tr, dim)
        cls_tokens = cls_tokens.view(batch_trs, frames_per_tr, self.embed_dim)

        # Split frames of each TR into `num_temporal_pools` chunks
        pooled_embeddings = []
        for b in range(batch_trs):
            chunks = torch.tensor_split(cls_tokens[b], num_temporal_pools, dim=0)
            pooled_chunks = [chunk.mean(dim=0) for chunk in chunks]
            pooled_embeddings.append(torch.stack(pooled_chunks))

        embeddings = torch.stack(pooled_embeddings).float().cpu().numpy()

        del outputs, inputs
        return embeddings


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def get_reference_n_trs(ref_dir, movie_type, stim_type, clip_stem):
    ref_path = ref_dir / movie_type / stim_type / f"{clip_stem}.npy"
    if ref_path.exists():
        return np.load(ref_path).shape[0]
    return None


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 24.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 24.0


def extract_features_for_clip(
    extractor: DINOv2Extractor,
    video_path: str,
    n_trs: int,
    num_temporal_pools: int = 4,
    **kwargs
) -> np.ndarray:
    """Extract DINOv2 embeddings sequentially to eliminate CPU bottleneck."""
    duration = get_video_duration(video_path)
    fps = get_video_fps(video_path)
    
    segment_duration = duration / n_trs
    frames_per_tr = max(1, int(round(segment_duration * fps)))
    
    # We use a batch size that keeps GPU fed constantly without OOMing (e.g. 500 frames)
    batch_trs = max(1, 500 // frames_per_tr)
    
    logger.info("  Video: %.1fs, FPS: %.2f, %d TRs (%d frames/TR -> %d pools, batch=%d)",
                duration, fps, n_trs, frames_per_tr, num_temporal_pools, batch_trs)

    output = np.zeros((n_trs, num_temporal_pools, extractor.embed_dim), dtype=np.float32)

    # Open video for sequential reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video %s", video_path)
        return output

    current_tr = 0
    frames_buffer = []
    
    pbar = tqdm(total=n_trs, desc="  Encoding TRs", leave=False)
    
    while current_tr < n_trs:
        valid_trs_in_batch = min(batch_trs, n_trs - current_tr)
        target_frames = valid_trs_in_batch * frames_per_tr
        
        batch_frames = []
        # Sequential read
        while len(batch_frames) < target_frames:
            ret, frame = cap.read()
            if not ret:
                # Video ended prematurely, pad with zeros
                empty = np.zeros((224, 224, 3), dtype=np.uint8)
                batch_frames.extend([empty] * (target_frames - len(batch_frames)))
                break
            
            # Fast resize and center crop
            H, W = frame.shape[:2]
            target_size = 224
            scale = target_size / min(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
            frame_resized = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            
            y_start = (new_H - target_size) // 2
            x_start = (new_W - target_size) // 2
            frame_cropped = frame_resized[y_start:y_start+target_size, x_start:x_start+target_size]
            
            batch_frames.append(frame_cropped)
            
        # Push batch to GPU
        try:
            if batch_frames:
                embeddings = extractor.extract_embeddings_batch(
                    batch_frames, frames_per_tr=frames_per_tr,
                    num_temporal_pools=num_temporal_pools
                )
                output[current_tr : current_tr + valid_trs_in_batch] = embeddings
        except Exception as e:
            logger.warning("  Error processing batch at TR %d: %s", current_tr, e)
            
        current_tr += valid_trs_in_batch
        pbar.update(valid_trs_in_batch)
        
        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()
    cap.release()
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# All movie/stimulus combinations
# ---------------------------------------------------------------------------
ALL_COMBOS = {
    "friends": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
    "movie10": ["bourne", "figures", "life", "wolf"],
    "ood": ["chaplin", "mononoke", "passepartout", "planetearth",
            "pulpfiction", "wot"],
}


def list_movie_splits(stimuli_dir, movie_type, stimulus_type):
    movie_dir = os.path.join(stimuli_dir, "movies", movie_type, stimulus_type)
    mkv_files = sorted(glob.glob(os.path.join(movie_dir, "*.mkv")))
    if not mkv_files:
        logger.warning("No .mkv files found in %s (skipping)", movie_dir)
        return []
    logger.info("Found %d movie splits in %s", len(mkv_files), movie_dir)
    return mkv_files


def process_combo(
    extractor, stimuli_dir, ref_features_dir, out_dir,
    movie_type, stim_type, frames_per_tr, batch_trs, num_temporal_pools, dry_run,
):
    """Process all clips for one movie_type/stimulus_type combination."""
    mkv_files = list_movie_splits(str(stimuli_dir), movie_type, stim_type)
    if not mkv_files:
        return 0
    if dry_run:
        mkv_files = mkv_files[:1]

    processed = 0
    for mkv_path in tqdm(mkv_files, desc=f"  {movie_type}/{stim_type}", leave=False):
        clip_stem = Path(mkv_path).stem
        out_path = out_dir / movie_type / stim_type / f"{clip_stem}.npy"

        if out_path.exists():
            logger.info("    Skipping %s (exists)", clip_stem)
            continue

        n_trs = get_reference_n_trs(
            ref_features_dir, movie_type, stim_type, clip_stem,
        )
        if n_trs is None:
            duration = get_video_duration(mkv_path)
            n_trs = max(1, int(duration / TR) - 1)

        logger.info("    %s → %d TRs", clip_stem, n_trs)

        features = extract_features_for_clip(
            extractor, mkv_path, n_trs, frames_per_tr=frames_per_tr,
            batch_trs=batch_trs, num_temporal_pools=num_temporal_pools
        )
        logger.info("    Output: %s", features.shape)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, features)

        del features
        gc.collect()
        torch.cuda.empty_cache()
        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 visual embeddings per fMRI TR"
    )
    parser.add_argument(
        "--model_variant", type=str, default="giant",
        choices=list(MODEL_CONFIGS.keys()),
        help="giant (1536-D), large (1024-D), or base (768-D)",
    )
    parser.add_argument(
        "--movie_type", type=str, default="all",
        choices=["all", "friends", "movie10", "ood"],
        help="'all' to process everything in one run",
    )
    parser.add_argument("--stimulus_type", type=str, default="all",
                        help="e.g. 's1' or 'all'")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--ref_modality", type=str, default=REF_MODALITY)
    parser.add_argument("--frames_per_tr", type=int, default=4)
    parser.add_argument("--num_temporal_pools", type=int, default=4,
                        help="Number of temporal features per TR to pool into")
    parser.add_argument("--batch_trs", type=int, default=8,
                        help="Number of TRs to process at once (batch size)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only 1 clip per combo")
    args = parser.parse_args()

    # Resolve paths
    if args.data_dir is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        data_dir = project_root / "Data"
    else:
        data_dir = Path(args.data_dir)

    stimuli_dir = data_dir / "algonauts_2025.competitors" / "stimuli"
    ref_features_dir = data_dir / "features_npy_pooled" / args.ref_modality

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Suppress OpenCV swscaler warnings for interlaced video
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

    # Load model ONCE
    extractor = DINOv2Extractor(model_variant=args.model_variant, device=device)
    out_dir = data_dir / "features_npy_pooled" / extractor.output_name

    # Build list of (movie_type, stim_type) combos
    if args.movie_type == "all":
        combos = [(mt, st) for mt, sts in ALL_COMBOS.items() for st in sts]
    elif args.stimulus_type == "all":
        combos = [(args.movie_type, st) for st in ALL_COMBOS[args.movie_type]]
    else:
        combos = [(args.movie_type, args.stimulus_type)]

    logger.info("Model: %s (dim=%d)", args.model_variant, extractor.embed_dim)
    logger.info("Output dir: %s", out_dir)
    logger.info("Processing %d combos: %s", len(combos), combos)
    if args.dry_run:
        logger.info("DRY RUN: 1 clip per combo only")

    total = 0
    for movie_type, stim_type in combos:
        logger.info("=== %s / %s ===", movie_type, stim_type)
        n = process_combo(
            extractor, stimuli_dir, ref_features_dir, out_dir,
            movie_type, stim_type, args.frames_per_tr, args.batch_trs, 
            args.num_temporal_pools, args.dry_run,
        )
        total += n

    logger.info("Done! Processed %d clips → %s", total, out_dir)


if __name__ == "__main__":
    main()
