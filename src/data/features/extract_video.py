"""Extract video features using V-JEPA 2 ViT-G.

Adapted from TRIBE (Meta/Facebook Research) for the Algonauts 2025 Challenge.
Uses facebook/vjepa2-vitg-fpc64-256 to extract multi-layer hidden state features
from movie stimulus clips.

Optimizations:
  - Model loaded in float16 (~10GB VRAM instead of ~20GB)
  - Batched inference: multiple timepoints processed per GPU forward pass
  - Pre-reads all frames once, then uses nearest-frame lookup (no random I/O)
  - Spatial mean-pooling done on GPU before CPU transfer

Usage:
    python extract_video.py --movie_type friends --stimulus_type s1
    python extract_video.py --movie_type friends --stimulus_type s1 --batch_timepoints 3
    python extract_video.py --movie_type friends --stimulus_type s1 --dry_run
"""

import argparse
import gc
import glob
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
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
MODEL_NAME = "facebook/vjepa2-vitg-fpc64-256"
SAMPLE_FREQ = 2.0  # Hz — temporal resolution of output features
NUM_FRAMES = 64     # frames per V-JEPA 2 input clip


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
class VJepa2Extractor:
    """Wrapper around V-JEPA 2 for hidden-state extraction."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel, AutoVideoProcessor

        logger.info("Loading V-JEPA 2 model: %s (float16 for speed)", MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True,
            torch_dtype=torch.float16,  # ~10GB VRAM instead of ~20GB
        )
        self.model.eval()
        self.model.to(device)

        self.processor = AutoVideoProcessor.from_pretrained(
            MODEL_NAME, do_rescale=True
        )
        self.device = device
        self.num_frames = NUM_FRAMES
        logger.info("V-JEPA 2 loaded on %s (float16)", device)

    @torch.inference_mode()
    def extract_hidden_states(self, batch_frames: list) -> torch.Tensor:
        """Feed a batch of frame sets and return spatially-pooled hidden states.

        Parameters
        ----------
        batch_frames : list of np.ndarray
            List of frame arrays, each with shape (num_frames, H, W, 3).

        Returns
        -------
        torch.Tensor
            (B, num_layers, dim) — spatially pooled, float32, CPU
        """
        videos = [list(frames) for frames in batch_frames]
        inputs = self.processor(videos=videos, return_tensors="pt")
        if "pixel_values" in inputs:
            nans = inputs["pixel_values"].isnan()
            if nans.any():
                inputs["pixel_values"][nans] = 0
            inputs["pixel_values"] = inputs["pixel_values"].half()
        inputs = inputs.to(self.device)

        pred = self.model(**inputs)
        states = pred.hidden_states
        # Mean-pool spatial tokens ON GPU, then move to CPU as float32
        pooled = [x.mean(dim=-2).float().cpu() for x in states]  # list of (B, D)
        out = torch.stack(pooled, dim=1)  # (B, num_layers, D)
        return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features_for_clip(
    extractor: VJepa2Extractor,
    video_path: str,
    batch_timepoints: int = 2,
) -> np.ndarray:
    """Extract V-JEPA 2 features for a single video clip.

    Uses lazy per-batch frame reading to keep RAM usage low (~2-3GB instead
    of loading entire video which can exceed 80GB+ for long clips).

    For each batch of timepoints:
      1. Compute which raw frame indices are needed (64 per timepoint)
      2. Deduplicate (adjacent timepoints share ~87% of frames)
      3. Read only unique frames from disk via decord seek
      4. Assemble per-timepoint frame arrays and run GPU inference

    Parameters
    ----------
    batch_timepoints : int
        Number of timepoints per GPU forward pass.

    Returns
    -------
    np.ndarray
        Features with shape (num_layers, dim, num_timepoints)
    """
    from decord import VideoReader, cpu

    logger.info("Processing: %s (batch=%d)", video_path, batch_timepoints)

    # Open video with decord, extracting at scale needed by model to save RAM
    vr = VideoReader(video_path, ctx=cpu(0), width=256, height=256)
    total_frames = len(vr)
    native_fps = vr.get_avg_fps()
    duration = total_frames / native_fps
    logger.info("  Video: %.1fs, %.1f fps, %d frames", duration, native_fps, total_frames)

    # Number of output timepoints at SAMPLE_FREQ Hz
    num_timepoints = max(1, int(duration * SAMPLE_FREQ))
    times = np.linspace(0, duration, num_timepoints + 1)[1:]

    # Sub-time offsets for NUM_FRAMES preceding frames (4s window)
    subtimes = np.array([k / extractor.num_frames * 4.0
                         for k in reversed(range(extractor.num_frames))])

    # Precompute ALL frame indices at once via numpy broadcasting
    # wanted_times[t, f] = the timestamp of frame f for timepoint t
    wanted_times = times[:, None] - subtimes[None, :]  # (T, 64)
    # Map wanted times → raw frame indices (clipped to valid range)
    raw_frame_idx = np.clip(
        np.round(wanted_times * native_fps).astype(np.int64),
        0, total_frames - 1,
    )  # (num_timepoints, NUM_FRAMES)

    logger.info("  Precomputed %d × %d frame index matrix", *raw_frame_idx.shape)
    logger.info("  Using LAZY per-batch frame reading (low RAM)")

    output = None
    num_batches = (num_timepoints + batch_timepoints - 1) // batch_timepoints

    for batch_start in tqdm(
        range(0, num_timepoints, batch_timepoints),
        total=num_batches,
        desc="  Encoding video",
        leave=False,
    ):
        batch_end = min(batch_start + batch_timepoints, num_timepoints)
        batch_idx = raw_frame_idx[batch_start:batch_end]  # (B, 64)

        # --- Deduplicate: read each unique frame only once ---
        unique_indices = np.unique(batch_idx)
        frames_batch = vr.get_batch(unique_indices.tolist()).asnumpy()  # (U, H, W, 3)
        # Build lookup: raw_frame_index → position in frames_batch
        idx_to_pos = {idx: pos for pos, idx in enumerate(unique_indices)}

        # Assemble per-timepoint frame arrays using the lookup
        batch_frames_list = []
        for i in range(batch_end - batch_start):
            positions = [idx_to_pos[fi] for fi in batch_idx[i]]
            batch_frames_list.append(frames_batch[positions])  # (64, H, W, 3)

        # Batched forward pass
        hidden = extractor.extract_hidden_states(batch_frames_list)  # (B, L, D)
        embds = hidden.numpy()  # (B, L, D)

        if output is None:
            output = np.zeros((num_timepoints, embds.shape[1], embds.shape[2]), dtype=np.float32)
            logger.info("  Feature tensor shape: %s", output.shape)
        output[batch_start:batch_end] = embds

        # Cleanup
        del batch_frames_list, frames_batch, hidden, embds
        if batch_start % (batch_timepoints * 10) == 0:
            torch.cuda.empty_cache()

    del vr
    gc.collect()

    # Transpose: (T, L, D) → (L, D, T)
    output = output.transpose(1, 2, 0)
    return output


def list_movie_splits(stimuli_dir: str, movie_type: str, stimulus_type: str) -> list:
    """List available .mkv movie split files."""
    movie_dir = os.path.join(stimuli_dir, "movies", movie_type, stimulus_type)
    mkv_files = sorted(glob.glob(os.path.join(movie_dir, "*.mkv")))
    if not mkv_files:
        logger.error("No .mkv files found in %s", movie_dir)
        sys.exit(1)
    logger.info("Found %d movie splits in %s", len(mkv_files), movie_dir)
    return mkv_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA 2 video features")
    parser.add_argument("--movie_type", type=str, default="friends",
                        choices=["friends", "movie10", "ood"],
                        help="Movie dataset type")
    parser.add_argument("--stimulus_type", type=str, default="s1",
                        help="Season/movie identifier (e.g. s1, s2, ..., bourne, wolf)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to Data/ directory (default: auto-detect)")
    parser.add_argument("--batch_timepoints", type=int, default=12,
                        help="Timepoints per GPU batch (default=12 for RTX 3090 24GB)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only the first movie split for testing")
    args = parser.parse_args()

    # Resolve paths
    if args.data_dir is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        data_dir = project_root / "Data"
    else:
        data_dir = Path(args.data_dir)

    stimuli_dir = data_dir / "stimuli"
    features_dir = data_dir / "features" / "video"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load model
    extractor = VJepa2Extractor(device=device)

    # List movie splits
    mkv_files = list_movie_splits(str(stimuli_dir), args.movie_type, args.stimulus_type)
    if args.dry_run:
        mkv_files = mkv_files[:1]
        logger.info("DRY RUN: processing only 1 file")

    # Output file
    out_file = features_dir / f"{args.movie_type}_{args.stimulus_type}_features_vjepa2.h5"
    logger.info("Output file: %s", out_file)

    # Extract features
    for mkv_path in tqdm(mkv_files, desc="Movie splits"):
        split_name = Path(mkv_path).stem  # e.g. "friends_s01e01a"

        # Skip if already extracted
        if out_file.exists():
            with h5py.File(out_file, "r") as f:
                if split_name in f:
                    logger.info("  Skipping %s (already exists)", split_name)
                    continue

        features = extract_features_for_clip(
            extractor, mkv_path, batch_timepoints=args.batch_timepoints
        )
        logger.info("  %s → shape %s", split_name, features.shape)

        # Save
        flag = "a" if out_file.exists() else "w"
        with h5py.File(out_file, flag) as f:
            group = f.create_group(split_name)
            group.create_dataset("video", data=features, dtype=np.float32)

        # Per-clip cleanup
        del features
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done! Features saved to %s", out_file)


if __name__ == "__main__":
    main()
