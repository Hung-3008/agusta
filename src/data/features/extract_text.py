"""Extract text features using LLaMA 3.2 3B.

Adapted from TRIBE (Meta/Facebook Research) for the Algonauts 2025 Challenge.
Uses meta-llama/Llama-3.2-3B to extract multi-layer hidden state features
from movie transcript word annotations.

For each movie split (.tsv transcript):
  1. Read word-level annotations with timestamps
  2. For each word: tokenize with context (preceding text), feed to LLaMA
  3. Get hidden_states from all transformer layers
  4. Mean-pool target word tokens per layer → (num_layers, dim) per word
  5. Align word embeddings to TR timestamps (1.49s) → save as .h5 file

Usage:
    python extract_text.py --movie_type friends --stimulus_type s1
    python extract_text.py --movie_type friends --stimulus_type s1 --dry_run

Note: Requires HuggingFace access to meta-llama/Llama-3.2-3B.
      Run `huggingface-cli login` first.
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
MODEL_NAME = "meta-llama/Llama-3.2-3B"
TR = 1.49             # fMRI repetition time in seconds
SAMPLE_FREQ = 2.0     # Hz — output temporal resolution (to match video/audio)
BATCH_SIZE = 8         # batch size for tokenized text


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
class LlamaExtractor:
    """Wrapper around LLaMA 3.2 3B for hidden-state extraction."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading LLaMA 3.2 3B model: %s", MODEL_NAME)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, truncation_side="left"
        )
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.model.to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.eos_token_id

        self.device = device
        logger.info("LLaMA 3.2 3B loaded on %s", device)

    @torch.inference_mode()
    def extract_word_features(
        self,
        words: list[str],
        contexts: list[str],
    ) -> list[np.ndarray]:
        """Extract hidden states for a batch of words with their contexts.

        Parameters
        ----------
        words : list[str]
            Target words to extract features for.
        contexts : list[str]
            Context strings (preceding text including the target word).

        Returns
        -------
        list[np.ndarray]
            List of arrays, each (num_layers, dim), one per word.
        """
        results = []

        # Process in batches
        for i in range(0, len(words), BATCH_SIZE):
            batch_words = words[i : i + BATCH_SIZE]
            batch_contexts = contexts[i : i + BATCH_SIZE]

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            inputs = self.tokenizer(
                batch_contexts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            outputs = self.model(**inputs, output_hidden_states=True)

            if "hidden_states" in outputs:
                states = outputs.hidden_states
            else:
                states = (
                    outputs.encoder_hidden_states + outputs.decoder_hidden_states
                )

            # (num_layers, batch, tokens, dim)
            hidden_states = torch.stack([layer.cpu().float() for layer in states])
            n_layers, n_batch, n_tokens, n_dims = hidden_states.shape

            for j, target_word in enumerate(batch_words):
                hidden_state = hidden_states[:, j]  # (num_layers, tokens, dim)

                # Remove padding
                n_pads = (inputs["input_ids"][j].cpu() == self.pad_id).sum().item()
                if n_pads:
                    hidden_state = hidden_state[:, :-n_pads]

                # Select tokens corresponding to the target word
                # (last N tokens where N = number of tokens in the target word)
                word_tokens = self.tokenizer.tokenize(target_word)
                n_word_tokens = len(word_tokens)
                word_state = hidden_state[:, -n_word_tokens:]  # (L, n_word_tokens, D)

                # Mean-pool over word tokens
                word_embd = word_state.mean(dim=1).numpy()  # (L, D)
                results.append(word_embd)

        return results


# ---------------------------------------------------------------------------
# Transcript processing helpers
# ---------------------------------------------------------------------------
def read_transcript(tsv_path: str) -> pd.DataFrame:
    """Read a word-level transcript TSV file.

    Expected columns: onset, duration, text (word-level annotations).
    The exact format depends on the Algonauts dataset.
    """
    df = pd.read_csv(tsv_path, sep="\t")

    # Handle different column naming conventions
    if "onset" not in df.columns and "start" in df.columns:
        df = df.rename(columns={"start": "onset"})
    if "word" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"word": "text"})
    if "text_per_tr" in df.columns and "text" not in df.columns:
        # Baseline format: one row per TR with text_per_tr
        df = df.rename(columns={"text_per_tr": "text"})

    return df


def build_word_contexts(df: pd.DataFrame, max_context_words: int = 200) -> tuple:
    """Build word list and context strings from transcript.

    Returns
    -------
    words : list[str]
        Individual words.
    contexts : list[str]
        Context strings (preceding words + current word).
    word_onsets : list[float]
        Onset times for each word.
    word_durations : list[float]
        Duration of each word.
    """
    words = []
    contexts = []
    word_onsets = []
    word_durations = []
    all_words_so_far = []

    for _, row in df.iterrows():
        text = row.get("text", "")
        if pd.isna(text) or str(text).strip() == "":
            continue

        onset = row.get("onset", row.get("start", 0.0))
        dur = row.get("duration", 0.0)
        if pd.isna(onset):
            continue

        # Split text into words if it contains multiple words
        text_words = str(text).strip().split()
        for w in text_words:
            w = w.strip()
            if not w:
                continue
            all_words_so_far.append(w)
            context = " ".join(all_words_so_far[-max_context_words:])
            words.append(w)
            contexts.append(context)
            word_onsets.append(float(onset))
            word_durations.append(float(dur) if not pd.isna(dur) else 0.0)

    return words, contexts, word_onsets, word_durations


def align_to_timepoints(
    word_features: list[np.ndarray],
    word_onsets: list[float],
    duration: float,
    sample_freq: float = SAMPLE_FREQ,
) -> np.ndarray:
    """Align word-level features to uniform temporal grid.

    Averages word features within each time bin.

    Parameters
    ----------
    word_features : list of (num_layers, dim) arrays
    word_onsets : list of float onset times
    duration : float total duration
    sample_freq : float output Hz

    Returns
    -------
    np.ndarray
        Shape (num_layers, dim, num_timepoints)
    """
    num_timepoints = max(1, int(duration * sample_freq))
    bin_edges = np.linspace(0, duration, num_timepoints + 1)

    if len(word_features) == 0:
        # No words — return NaN placeholder
        return np.full((1, 1, num_timepoints), np.nan, dtype=np.float32)

    num_layers, dim = word_features[0].shape
    output = np.full((num_layers, dim, num_timepoints), np.nan, dtype=np.float32)

    for t in range(num_timepoints):
        t_start = bin_edges[t]
        t_end = bin_edges[t + 1]

        # Find words whose onset falls in this time bin
        bin_features = []
        for feat, onset in zip(word_features, word_onsets):
            if t_start <= onset < t_end:
                bin_features.append(feat)

        if bin_features:
            output[:, :, t] = np.mean(bin_features, axis=0)

    # Forward-fill NaN values (carry last word embedding until new words appear)
    for l in range(num_layers):
        for d in range(dim):
            series = output[l, d, :]
            # Find first non-NaN and fill before it with zeros
            non_nan_idx = np.where(~np.isnan(series))[0]
            if len(non_nan_idx) > 0:
                first_valid = non_nan_idx[0]
                series[:first_valid] = 0.0
                # Forward fill
                for t in range(1, num_timepoints):
                    if np.isnan(series[t]):
                        series[t] = series[t - 1]
            else:
                series[:] = 0.0
            output[l, d, :] = series

    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features_for_transcript(
    extractor: LlamaExtractor,
    tsv_path: str,
    video_duration: float | None = None,
) -> np.ndarray:
    """Extract LLaMA features for a single transcript file.

    Parameters
    ----------
    extractor : LlamaExtractor
    tsv_path : str
    video_duration : float or None
        Duration of the corresponding video in seconds.

    Returns
    -------
    np.ndarray
        Features with shape (num_layers, dim, num_timepoints)
    """
    logger.info("Processing: %s", tsv_path)

    df = read_transcript(tsv_path)
    words, contexts, word_onsets, word_durations = build_word_contexts(df)

    if len(words) == 0:
        logger.warning("  No words found in %s", tsv_path)
        num_tp = max(1, int((video_duration or 60) * SAMPLE_FREQ))
        return np.zeros((1, 1, num_tp), dtype=np.float32)

    logger.info("  Found %d words", len(words))

    # Extract word-level features
    word_features = extractor.extract_word_features(words, contexts)
    logger.info("  Word features: %d words, each shape %s", len(word_features), word_features[0].shape)

    # Determine duration
    if video_duration is None:
        # Estimate from last word onset
        video_duration = max(word_onsets) + 5.0  # add buffer

    # Align to temporal grid
    output = align_to_timepoints(word_features, word_onsets, video_duration)
    logger.info("  Output shape: %s", output.shape)
    return output


def list_transcript_splits(stimuli_dir: str, movie_type: str, stimulus_type: str) -> list:
    """List available .tsv transcript files."""
    tsv_dir = os.path.join(stimuli_dir, "transcripts", movie_type, stimulus_type)
    tsv_files = sorted(glob.glob(os.path.join(tsv_dir, "*.tsv")))
    if not tsv_files:
        logger.error("No .tsv files found in %s", tsv_dir)
        sys.exit(1)
    logger.info("Found %d transcript splits in %s", len(tsv_files), tsv_dir)
    return tsv_files


def get_video_duration(stimuli_dir: str, movie_type: str, stimulus_type: str, split_name: str) -> float | None:
    """Try to get the video duration for a matching movie split."""
    from moviepy import VideoFileClip

    movie_dir = os.path.join(stimuli_dir, "movies", movie_type, stimulus_type)
    mkv_path = os.path.join(movie_dir, f"{split_name}.mkv")
    if os.path.exists(mkv_path):
        try:
            clip = VideoFileClip(mkv_path)
            dur = clip.duration
            clip.close()
            return dur
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract LLaMA 3.2 3B text features")
    parser.add_argument("--movie_type", type=str, default="friends",
                        choices=["friends", "movie10", "ood"],
                        help="Movie dataset type")
    parser.add_argument("--stimulus_type", type=str, default="s1",
                        help="Season/movie identifier (e.g. s1, s2)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to Data/ directory (default: auto-detect)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only the first transcript for testing")
    args = parser.parse_args()

    # Resolve paths
    if args.data_dir is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        data_dir = project_root / "Data"
    else:
        data_dir = Path(args.data_dir)

    stimuli_dir = data_dir / "stimuli"
    features_dir = data_dir / "features" / "text"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load model
    extractor = LlamaExtractor(device=device)

    # List transcript splits
    tsv_files = list_transcript_splits(str(stimuli_dir), args.movie_type, args.stimulus_type)
    if args.dry_run:
        tsv_files = tsv_files[:1]
        logger.info("DRY RUN: processing only 1 file")

    # Output file
    out_file = features_dir / f"{args.movie_type}_{args.stimulus_type}_features_llama3.h5"
    logger.info("Output file: %s", out_file)

    # Extract features
    for tsv_path in tqdm(tsv_files, desc="Transcript splits"):
        split_name = Path(tsv_path).stem

        # Skip if already extracted
        if out_file.exists():
            with h5py.File(out_file, "r") as f:
                if split_name in f:
                    logger.info("  Skipping %s (already exists)", split_name)
                    continue

        # Get corresponding video duration for proper alignment
        video_duration = get_video_duration(
            str(stimuli_dir), args.movie_type, args.stimulus_type, split_name
        )

        features = extract_features_for_transcript(
            extractor, tsv_path, video_duration=video_duration
        )
        logger.info("  %s → shape %s", split_name, features.shape)

        # Save
        flag = "a" if out_file.exists() else "w"
        with h5py.File(out_file, flag) as f:
            group = f.create_group(split_name)
            group.create_dataset("text", data=features, dtype=np.float32)

        # --- Per-transcript cleanup to free RAM/VRAM ---
        del features
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done! Features saved to %s", out_file)


if __name__ == "__main__":
    main()
