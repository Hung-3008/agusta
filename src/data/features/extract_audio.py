"""Extract audio features using Wav2Vec-BERT 2.0.

Adapted from TRIBE (Meta/Facebook Research) for the Algonauts 2025 Challenge.
Uses facebook/w2v-bert-2.0 to extract multi-layer hidden state features
from the audio tracks of movie stimulus clips.

For each movie split (.mkv):
  1. Extract audio track with moviepy, convert to mono, resample to 16kHz
  2. Normalize waveform (zero-mean, unit std)
  3. Feed through Wav2Vec-BERT 2.0 → get hidden_states from all transformer layers
  4. Stack hidden states → (num_layers, dim, time_steps)
  5. Interpolate to 2Hz temporal resolution → save as .h5 file

Usage:
    python extract_audio.py --movie_type friends --stimulus_type s1
    python extract_audio.py --movie_type friends --stimulus_type s1 --dry_run
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
MODEL_NAME = "facebook/w2v-bert-2.0"
SAMPLE_FREQ = 2.0      # Hz — temporal resolution of output features
INPUT_SR = 16_000       # Wav2Vec-BERT 2.0 expects 16kHz input


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
class Wav2VecBertExtractor:
    """Wrapper around Wav2Vec-BERT 2.0 for hidden-state extraction."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

        logger.info("Loading Wav2Vec-BERT 2.0 model: %s", MODEL_NAME)
        self.model = Wav2Vec2BertModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.model.to(device)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        self.device = device
        logger.info("Wav2Vec-BERT 2.0 loaded on %s", device)

    @staticmethod
    def resample_wav(wav: torch.Tensor, old_sr: int, new_sr: int) -> torch.Tensor:
        """Resample waveform from old_sr to new_sr using torchaudio."""
        import torchaudio.functional as AF
        # wav shape: (num_samples, num_channels) → need (num_channels, num_samples)
        wav_t = wav.T
        wav_resampled = AF.resample(wav_t, int(old_sr), int(new_sr))
        return wav_resampled.T  # back to (num_samples, num_channels)

    @staticmethod
    def preprocess_wav(wav: torch.Tensor) -> torch.Tensor:
        """Convert to mono and normalize."""
        # Average channels → mono
        wav = torch.mean(wav, dim=1)
        # Normalize
        wav = (wav - wav.mean()) / (1e-8 + wav.std())
        return wav

    @torch.inference_mode()
    def extract_hidden_states(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from preprocessed waveform.

        Processes audio in chunks of CHUNK_SEC seconds to avoid CUDA OOM
        on long clips (e.g., 684s at 16kHz → 11M samples).

        Parameters
        ----------
        wav : torch.Tensor
            1-D waveform tensor (mono, 16kHz, normalized).

        Returns
        -------
        torch.Tensor
            Stacked hidden states: (num_layers, dim, time_steps)
        """
        CHUNK_SEC = 30  # process 30s at a time
        chunk_samples = CHUNK_SEC * INPUT_SR  # 480,000 samples

        total_samples = len(wav)
        all_chunks = []

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            wav_chunk = wav[start:end]

            features = self.feature_extractor(
                wav_chunk.numpy(),
                return_tensors="pt",
                sampling_rate=INPUT_SR,
                do_normalize=True,
            )
            try:
                input_tensor = features["input_features"]
            except KeyError:
                input_tensor = features["input_values"]

            outputs = self.model(
                input_tensor.to(self.device),
                output_hidden_states=True,
            )
            hidden = outputs.get("hidden_states")
            if isinstance(hidden, tuple):
                hidden = torch.stack(hidden)

            # (num_layers, batch=1, time, dim) → (num_layers, dim, time)
            chunk_out = hidden.squeeze(1).detach().cpu().transpose(-1, -2)
            all_chunks.append(chunk_out)

            # Free GPU memory between chunks
            del outputs, hidden, input_tensor, features
            torch.cuda.empty_cache()

        # Concatenate all chunks along the time dimension
        out = torch.cat(all_chunks, dim=-1)
        return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features_for_clip(
    extractor: Wav2VecBertExtractor,
    video_path: str,
) -> np.ndarray:
    """Extract Wav2Vec-BERT 2.0 features for a single video's audio track.

    Uses moviepy iter_chunks() instead of to_soundarray() to avoid numpy
    vstack bug in newer numpy versions.

    Returns
    -------
    np.ndarray
        Features with shape (num_layers, dim, num_timepoints_at_2Hz)
    """
    from moviepy.editor import VideoFileClip

    logger.info("Processing: %s", video_path)

    clip = VideoFileClip(video_path)
    duration = clip.duration
    audio = clip.audio
    if audio is None:
        logger.warning("  No audio track found in %s, returning zeros", video_path)
        clip.close()
        return np.zeros((1, 1, max(1, int(duration * SAMPLE_FREQ))), dtype=np.float32)

    audio_fps = audio.fps

    # Collect audio via iter_chunks (avoids to_soundarray numpy bug)
    chunks = []
    for chunk in audio.iter_chunks(chunksize=INPUT_SR, fps=audio_fps):
        chunks.append(chunk)
    audio_array = np.concatenate(chunks, axis=0)  # (num_samples, num_channels)
    clip.close()
    del chunks

    wav = torch.tensor(audio_array, dtype=torch.float32)

    # Resample to 16kHz
    wav = extractor.resample_wav(wav, old_sr=audio_fps, new_sr=INPUT_SR)

    # Convert to mono + normalize
    wav = extractor.preprocess_wav(wav)

    logger.info("  Audio: %.1fs, resampled to 16kHz, %d samples", duration, len(wav))

    # Extract hidden states
    latents = extractor.extract_hidden_states(wav)  # (L, D, T_model)

    # Free raw audio data immediately
    del wav, audio_array
    gc.collect()

    # Interpolate to target temporal resolution (2Hz)
    target_timepoints = max(1, int(duration * SAMPLE_FREQ))
    if abs(target_timepoints - latents.shape[-1]) > 0:
        L, D, T = latents.shape
        # Reshape to (L*D, 1, T) for 1D interpolation, then back to (L, D, T')
        latents_flat = latents.reshape(L * D, 1, T)
        latents_flat = F.interpolate(
            latents_flat,
            size=target_timepoints,
            mode="linear",
            align_corners=False,
        )
        latents = latents_flat.reshape(L, D, target_timepoints)

    logger.info("  Output shape: %s", tuple(latents.shape))
    return latents.numpy().astype(np.float32)


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
    parser = argparse.ArgumentParser(description="Extract Wav2Vec-BERT 2.0 audio features")
    parser.add_argument("--movie_type", type=str, default="friends",
                        choices=["friends", "movie10", "ood"],
                        help="Movie dataset type")
    parser.add_argument("--stimulus_type", type=str, default="s1",
                        help="Season/movie identifier (e.g. s1, s2, ..., bourne, wolf)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to Data/ directory (default: auto-detect)")
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
    features_dir = data_dir / "features" / "audio"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load model
    extractor = Wav2VecBertExtractor(device=device)

    # List movie splits
    mkv_files = list_movie_splits(str(stimuli_dir), args.movie_type, args.stimulus_type)
    if args.dry_run:
        mkv_files = mkv_files[:1]
        logger.info("DRY RUN: processing only 1 file")

    # Output file
    out_file = features_dir / f"{args.movie_type}_{args.stimulus_type}_features_w2vbert.h5"
    logger.info("Output file: %s", out_file)

    # Extract features
    for mkv_path in tqdm(mkv_files, desc="Movie splits"):
        split_name = Path(mkv_path).stem

        # Skip if already extracted
        if out_file.exists():
            with h5py.File(out_file, "r") as f:
                if split_name in f:
                    logger.info("  Skipping %s (already exists)", split_name)
                    continue

        features = extract_features_for_clip(extractor, mkv_path)
        logger.info("  %s → shape %s", split_name, features.shape)

        # Save
        flag = "a" if out_file.exists() else "w"
        with h5py.File(out_file, flag) as f:
            group = f.create_group(split_name)
            group.create_dataset("audio", data=features, dtype=np.float32)

        # --- Per-clip cleanup to free RAM/VRAM ---
        del features
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done! Features saved to %s", out_file)


if __name__ == "__main__":
    main()
