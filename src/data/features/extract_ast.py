"""Extract per-TR audio embeddings using AST (Audio Spectrogram Transformer).

AST is trained on AudioSet and captures environmental sounds, music, and
ambient audio that speech-focused models like Whisper miss. This complements
Whisper's speech features for auditory cortex prediction in brain encoding.

Produces .npy files with shape (n_trs, 768) float32 in
Data/features_npy_pooled/ast/.

Usage:
    # Dry run (1 clip, test)
    python src/data/features/extract_ast.py \\
        --movie_type friends --stimulus_type s1 --dry_run

    # Full extraction
    python src/data/features/extract_ast.py \\
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
import tempfile
import wave
import struct
from pathlib import Path

import cv2
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
TR = 1.49
REF_MODALITY = "whisper"
SAMPLE_RATE = 16000  # AST expects 16kHz mono audio
EMBED_DIM = 768  # AST hidden size


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


def extract_audio_from_video(video_path: str, output_wav: str) -> None:
    """Extract full audio track as 16kHz mono WAV."""
    ffmpeg_exe = _get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",               # mono
        "-loglevel", "error",
        output_wav,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def load_wav_as_numpy(wav_path: str) -> np.ndarray:
    """Load WAV file as float32 numpy array (n_samples,)."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = struct.unpack(f"<{n_frames * n_channels}h", raw)
    elif sampwidth == 4:
        samples = struct.unpack(f"<{n_frames * n_channels}i", raw)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio = np.array(samples, dtype=np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Normalize to [-1, 1]
    max_val = 2 ** (8 * sampwidth - 1)
    audio = audio / max_val
    return audio


# ---------------------------------------------------------------------------
# AST Extractor
# ---------------------------------------------------------------------------
class ASTExtractor:
    """Audio Spectrogram Transformer for environmental audio features."""

    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, device: str = "cuda"):
        from transformers import ASTModel, AutoFeatureExtractor

        self.embed_dim = EMBED_DIM
        self.output_name = "ast"

        logger.info("Loading AST model: %s", self.MODEL_NAME)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.MODEL_NAME
        )
        self.model = ASTModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        self.model.to(device)
        self.device = device

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(
            "AST loaded (dim=%d, %.0fM params, device=%s)",
            self.embed_dim, n_params, device,
        )

    @torch.inference_mode()
    def extract_embeddings_batch(self, audio_segments: list[np.ndarray]) -> np.ndarray:
        """Extract embeddings from a batch of audio segments.

        Args:
            audio_segments: list of 1-D float32 numpy arrays at 16kHz.

        Returns:
            np.ndarray of shape (batch_size, embed_dim)
        """
        if not audio_segments:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        # Handle tiny/empty segments
        valid_segments = [
            seg if seg.size >= 100 else np.zeros(16000, dtype=np.float32)
            for seg in audio_segments
        ]

        inputs = self.feature_extractor(
            valid_segments,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # CLS token: (batch_size, dim)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        embeddings = cls_tokens.float().cpu().numpy()

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


def extract_features_for_clip(
    extractor: ASTExtractor,
    video_path: str,
    n_trs: int,
    batch_trs: int = 32,
) -> np.ndarray:
    """Extract AST embeddings for each TR of a video clip using batching.

    1. Extract full audio track from video.
    2. Slice audio into per-TR segments.
    3. Run AST on batches of segments.

    Returns shape (n_trs, embed_dim), float32.
    """
    duration = get_video_duration(video_path)
    logger.info("  Video: %.1fs, extracting %d TRs (batch_size=%d)",
                duration, n_trs, batch_trs)

    # Extract full audio to temporary WAV
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = os.path.join(temp_dir, "audio.wav")
        extract_audio_from_video(video_path, wav_path)
        full_audio = load_wav_as_numpy(wav_path)

    total_samples = full_audio.shape[0]
    samples_per_tr = total_samples // n_trs

    output = np.zeros((n_trs, extractor.embed_dim), dtype=np.float32)

    for i in tqdm(range(0, n_trs, batch_trs), desc="  Encoding TRs", leave=False):
        batch_segments = []
        valid_trs = min(batch_trs, n_trs - i)

        for j in range(valid_trs):
            tr_idx = i + j
            start_sample = tr_idx * samples_per_tr
            end_sample = min((tr_idx + 1) * samples_per_tr, total_samples)
            audio_segment = full_audio[start_sample:end_sample]
            batch_segments.append(audio_segment)

        try:
            embeddings = extractor.extract_embeddings_batch(batch_segments)
            output[i : i + valid_trs] = embeddings
        except Exception as e:
            logger.warning("  Error processing batch at TR %d: %s", i, e)

        if i % (batch_trs * 10) == 0 and i > 0:
            gc.collect()
            torch.cuda.empty_cache()

    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    return output



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
    movie_type, stim_type, batch_trs, dry_run,
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

        features = extract_features_for_clip(extractor, mkv_path, n_trs, batch_trs=batch_trs)
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
        description="Extract AST audio embeddings per fMRI TR"
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
    parser.add_argument("--batch_trs", type=int, default=32,
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

    # Load model ONCE
    extractor = ASTExtractor(device=device)
    out_dir = data_dir / "features_npy_pooled" / extractor.output_name

    # Build list of (movie_type, stim_type) combos
    if args.movie_type == "all":
        combos = [(mt, st) for mt, sts in ALL_COMBOS.items() for st in sts]
    elif args.stimulus_type == "all":
        combos = [(args.movie_type, st) for st in ALL_COMBOS[args.movie_type]]
    else:
        combos = [(args.movie_type, args.stimulus_type)]

    logger.info("Model: AST (dim=%d)", extractor.embed_dim)
    logger.info("Output dir: %s", out_dir)
    logger.info("Processing %d combos: %s", len(combos), combos)
    if args.dry_run:
        logger.info("DRY RUN: 1 clip per combo only")

    total = 0
    for movie_type, stim_type in combos:
        logger.info("=== %s / %s ===", movie_type, stim_type)
        n = process_combo(
            extractor, stimuli_dir, ref_features_dir, out_dir,
            movie_type, stim_type, args.batch_trs, args.dry_run,
        )
        total += n

    logger.info("Done! Processed %d clips → %s", total, out_dir)


if __name__ == "__main__":
    main()

