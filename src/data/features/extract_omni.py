"""Extract unified multimodal features using Qwen2.5-Omni-7B.

Replaces 3 separate extractors (V-JEPA 2, Wav2Vec-BERT 2.0, LLaMA 3.2)
with a single model that processes video + audio + text simultaneously,
producing cross-modal fused features aligned to fMRI TRs.

Architecture:
    Video (.mkv) + Audio + Text → Qwen2.5-Omni Thinker → hidden_states
    Each TR (1.49s) of stimulus → 1 feature vector (hidden_dim=3584)

For each movie clip:
  1. Read transcript TSV (1 row = 1 TR)
  2. For each TR chunk (1.49s):
     a. Extract video segment
     b. Extract audio segment
     c. Get transcript text for this TR
     d. Forward through Qwen2.5-Omni Thinker with output_hidden_states=True
     e. Mean-pool hidden states over sequence → (n_layers, hidden_dim)
  3. Stack all TRs → (n_TRs, n_layers, hidden_dim)
  4. Save as HDF5

Usage:
    python extract_omni.py --movie_type friends --stimulus_type s1
    python extract_omni.py --movie_type friends --stimulus_type s1 --dry_run
    python extract_omni.py --movie_type friends --stimulus_type s1 --layers -1 -4 -8

Requirements:
    pip install git+https://github.com/huggingface/transformers
    pip install qwen-omni-utils accelerate flash-attn
    System: ffmpeg
"""

import argparse
import gc
import glob
import logging
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
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
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
TR = 1.49  # fMRI repetition time (seconds)
DEFAULT_LAYERS = [-1, -4, -8, -12, -16]  # Which layers to extract (negative = from end)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
class QwenOmniExtractor:
    """Wrapper around Qwen2.5-Omni-7B Thinker for hidden-state extraction."""

    def __init__(self, device: str = "cuda", use_flash_attn: bool = True):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        logger.info("Loading Qwen2.5-Omni-7B model: %s", MODEL_NAME)

        kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_NAME, **kwargs
        )
        self.model.eval()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
        self.device = device
        logger.info("Qwen2.5-Omni-7B loaded (device_map=auto)")

    @torch.inference_mode()
    def extract_hidden_states(
        self,
        video_path: str | None = None,
        audio_path: str | None = None,
        text: str | None = None,
        layers: list[int] | None = None,
    ) -> np.ndarray:
        """Extract hidden states from the Thinker for multimodal input.

        Parameters
        ----------
        video_path : str or None
            Path to video file (with or without audio track).
        audio_path : str or None
            Path to separate audio file (if not using audio from video).
        text : str or None
            Text transcript for this segment.
        layers : list[int]
            Which transformer layers to extract (negative indexing).

        Returns
        -------
        np.ndarray
            Mean-pooled hidden states, shape (n_layers, hidden_dim).
        """
        from qwen_omni_utils import process_mm_info

        if layers is None:
            layers = DEFAULT_LAYERS

        # Build conversation with available modalities
        content = []
        if video_path is not None:
            content.append({"type": "video", "video": video_path})
        if audio_path is not None:
            content.append({"type": "audio", "audio": audio_path})
        if text is not None and text.strip():
            content.append({"type": "text", "text": text})

        if not content:
            return None

        # If only text, add a minimal prompt
        if all(c["type"] == "text" for c in content):
            pass  # text only is fine

        conversation = [{"role": "user", "content": content}]

        # Process multimodal info
        use_audio_in_video = (audio_path is None and video_path is not None)
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video
        )

        # Tokenize
        text_input = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text_input,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Forward pass through thinker only
        outputs = self.model.thinker(
            **inputs,
            output_hidden_states=True,
        )

        # outputs.hidden_states: tuple of (n_total_layers+1) tensors
        # Each tensor: (1, seq_len, hidden_dim)
        all_hidden = outputs.hidden_states  # tuple of tensors

        # Select requested layers + mean pool over sequence
        selected = []
        for layer_idx in layers:
            hs = all_hidden[layer_idx]  # (1, seq_len, hidden_dim)
            pooled = hs.mean(dim=1).squeeze(0)  # (hidden_dim,)
            selected.append(pooled.float().cpu().numpy())

        result = np.stack(selected, axis=0)  # (n_layers, hidden_dim)

        # Cleanup
        del outputs, inputs, all_hidden
        torch.cuda.empty_cache()

        return result


# ---------------------------------------------------------------------------
# ffmpeg helper
# ---------------------------------------------------------------------------
def _get_ffmpeg_exe() -> str:
    """Find ffmpeg binary — prefer system, fallback to imageio-ffmpeg."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "ffmpeg not found! Install via: conda install -y ffmpeg -c conda-forge "
        "or pip install imageio-ffmpeg"
    )


# ---------------------------------------------------------------------------
# TR-aligned feature extraction
# ---------------------------------------------------------------------------
def extract_video_segment(
    video_path: str,
    start: float,
    duration: float,
    temp_dir: str,
) -> str:
    """Extract a video segment using ffmpeg.

    Returns path to temporary segment file.
    """
    import subprocess

    ffmpeg_exe = _get_ffmpeg_exe()
    segment_path = os.path.join(temp_dir, "segment.mp4")
    cmd = [
        ffmpeg_exe, "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "ultrafast",
        "-loglevel", "error",
        segment_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return segment_path


def read_transcript(tsv_path: str) -> pd.DataFrame:
    """Read TR-aligned transcript TSV.

    Each row corresponds to 1 fMRI TR.
    Columns: text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr
    """
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def get_tr_text(df: pd.DataFrame, tr_idx: int) -> str:
    """Get transcript text for a specific TR index."""
    if tr_idx >= len(df):
        return ""
    text = df.iloc[tr_idx].get("text_per_tr", "")
    if pd.isna(text) or str(text).strip() == "":
        return ""
    return str(text).strip()


def extract_features_for_clip(
    extractor: QwenOmniExtractor,
    video_path: str,
    tsv_path: str | None,
    layers: list[int],
    tr: float = TR,
) -> np.ndarray:
    """Extract Qwen2.5-Omni features for a single movie clip.

    For each TR (1.49s chunk):
      - Cut video segment [start, start+TR]
      - Get transcript text for this TR
      - Feed video+audio+text to Qwen2.5-Omni Thinker
      - Mean-pool hidden states → (n_layers, hidden_dim)

    Parameters
    ----------
    extractor : QwenOmniExtractor
    video_path : str
        Path to .mkv movie file.
    tsv_path : str or None
        Path to .tsv transcript file (1 row per TR).
    layers : list[int]
        Transformer layers to extract.
    tr : float
        fMRI repetition time.

    Returns
    -------
    np.ndarray
        Shape (n_TRs, n_layers, hidden_dim), float32.
    """
    from moviepy import VideoFileClip

    logger.info("Processing: %s", video_path)

    # Get video duration
    clip = VideoFileClip(video_path)
    duration = clip.duration
    clip.close()

    # Read transcript
    df = None
    if tsv_path is not None and os.path.exists(tsv_path):
        df = read_transcript(tsv_path)
        logger.info("  Transcript: %d TRs from %s", len(df), tsv_path)

    # Compute TR start times (same as baseline)
    start_times = np.arange(0, duration, tr)[:-1]
    n_trs = len(start_times)
    logger.info("  Video: %.1fs, %d TRs", duration, n_trs)

    # Temporary directory for video segments
    with tempfile.TemporaryDirectory() as temp_dir:
        features = []
        dummy_shape = None  # will be set from first successful extraction
        for tr_idx, start in enumerate(
            tqdm(start_times, desc="  Extracting TRs", leave=False)
        ):
            # Extract video segment
            segment_path = extract_video_segment(
                video_path, start, tr, temp_dir
            )

            # Get text for this TR
            text = get_tr_text(df, tr_idx) if df is not None else ""

            # Extract features
            try:
                feat = extractor.extract_hidden_states(
                    video_path=segment_path,
                    text=text if text else None,
                    layers=layers,
                )
            except Exception as e:
                logger.warning(
                    "  Error at TR %d (t=%.1fs): %s. Using zeros.", tr_idx, start, e
                )
                feat = None

            if feat is not None:
                features.append(feat)
                if dummy_shape is None:
                    dummy_shape = feat.shape  # (n_layers, hidden_dim)
            else:
                # Fallback: create zero features matching expected shape
                if dummy_shape is not None:
                    features.append(np.zeros(dummy_shape, dtype=np.float32))
                else:
                    features.append(None)  # will be replaced later

            # Periodic cleanup
            if tr_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # Replace any remaining None entries
    if dummy_shape is None:
        logger.error("  No features extracted at all!")
        # Use a sensible default shape: (n_layers, hidden_dim=3584)
        dummy_shape = (len(layers), 3584)

    for i, f in enumerate(features):
        if f is None:
            features[i] = np.zeros(dummy_shape, dtype=np.float32)

    output = np.stack(features, axis=0)  # (n_TRs, n_layers, hidden_dim)
    logger.info("  Output shape: %s", output.shape)
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_movie_splits(stimuli_dir: str, movie_type: str, stimulus_type: str) -> list:
    """List available movie splits (.mkv files)."""
    movie_dir = os.path.join(stimuli_dir, "movies", movie_type, stimulus_type)
    mkv_files = sorted(glob.glob(os.path.join(movie_dir, "*.mkv")))
    if not mkv_files:
        logger.error("No .mkv files found in %s", movie_dir)
        sys.exit(1)
    logger.info("Found %d movie splits in %s", len(mkv_files), movie_dir)
    return mkv_files


def find_transcript(
    stimuli_dir: str, movie_type: str, stimulus_type: str, split_name: str,
) -> str | None:
    """Find matching transcript TSV for a movie split."""
    tsv_dir = os.path.join(stimuli_dir, "transcripts", movie_type, stimulus_type)
    tsv_path = os.path.join(tsv_dir, f"{split_name}.tsv")
    if os.path.exists(tsv_path):
        return tsv_path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract unified multimodal features using Qwen2.5-Omni-7B"
    )
    parser.add_argument(
        "--movie_type", type=str, default="friends",
        choices=["friends", "movie10", "ood"],
        help="Movie dataset type",
    )
    parser.add_argument(
        "--stimulus_type", type=str, default="s1",
        help="Season/movie identifier (e.g. s1, s2, ..., bourne, wolf)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to Data/ directory (default: auto-detect)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
        help=f"Transformer layers to extract (negative index, default: {DEFAULT_LAYERS})",
    )
    parser.add_argument(
        "--no_flash_attn", action="store_true",
        help="Disable FlashAttention 2 (slower but more compatible)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Process only the first movie split for testing",
    )
    args = parser.parse_args()

    # Resolve paths
    if args.data_dir is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        data_dir = project_root / "Data"
    else:
        data_dir = Path(args.data_dir)

    stimuli_dir = data_dir / "stimuli"
    features_dir = data_dir / "features" / "omni"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load model
    extractor = QwenOmniExtractor(
        device=device, use_flash_attn=not args.no_flash_attn,
    )

    # List movie splits
    mkv_files = list_movie_splits(
        str(stimuli_dir), args.movie_type, args.stimulus_type,
    )
    if args.dry_run:
        mkv_files = mkv_files[:1]
        logger.info("DRY RUN: processing only 1 file")

    # Output file
    out_file = features_dir / (
        f"{args.movie_type}_{args.stimulus_type}_features_qwen_omni.h5"
    )
    logger.info("Output file: %s", out_file)
    logger.info("Layers: %s", args.layers)

    # Extract features
    for mkv_path in tqdm(mkv_files, desc="Movie splits"):
        split_name = Path(mkv_path).stem  # e.g. "friends_s01e01a"

        # Skip if already extracted
        if out_file.exists():
            with h5py.File(out_file, "r") as f:
                if split_name in f:
                    logger.info("  Skipping %s (already exists)", split_name)
                    continue

        # Find matching transcript
        tsv_path = find_transcript(
            str(stimuli_dir), args.movie_type, args.stimulus_type, split_name,
        )

        # Extract features
        features = extract_features_for_clip(
            extractor, mkv_path, tsv_path, layers=args.layers,
        )
        logger.info("  %s → shape %s", split_name, features.shape)

        # Save
        flag = "a" if out_file.exists() else "w"
        with h5py.File(out_file, flag) as f:
            group = f.create_group(split_name)
            group.create_dataset("omni", data=features, dtype=np.float32)
            # Store metadata
            group.attrs["layers"] = args.layers
            group.attrs["model"] = MODEL_NAME
            group.attrs["tr"] = TR

        # Per-clip cleanup
        del features
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done! Features saved to %s", out_file)


if __name__ == "__main__":
    main()
