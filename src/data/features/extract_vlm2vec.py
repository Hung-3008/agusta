"""Extract per-TR video embeddings using VLM2Vec (Qwen2-VL + LoRA).

Loads Qwen2-VL base model directly via transformers, then applies the
VLM2Vec LoRA adapter via peft — no need to clone the VLM2Vec repo.
No flash-attn required (uses SDPA attention).

Produces .npy files with shape (n_trs, embed_dim) float32 in
Data/features_npy_pooled/vlm2vec_{variant}/.

Usage:
    # 7B model (dim=3584, ~16GB VRAM)
    python src/data/features/extract_vlm2vec.py \\
        --model_variant v1_7b \\
        --movie_type friends --stimulus_type s1 --dry_run

    # 2B model (dim=1536, ~6GB VRAM)
    python src/data/features/extract_vlm2vec.py \\
        --model_variant v1_2b \\
        --movie_type friends --stimulus_type s1 --dry_run

    # Full extraction
    python src/data/features/extract_vlm2vec.py \\
        --movie_type friends --stimulus_type s1
"""

import argparse
import gc
import glob
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

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

MODEL_CONFIGS = {
    "v1_7b": {
        "base_model": "Qwen/Qwen2-VL-7B-Instruct",
        "lora_adapter": "TIGER-Lab/VLM2Vec-Qwen2VL-7B",
        "embed_dim": 3584,
        "output_name": "vlm2vec_7b",
    },
    "v1_2b": {
        "base_model": "Qwen/Qwen2-VL-2B-Instruct",
        "lora_adapter": "TIGER-Lab/VLM2Vec-Qwen2VL-2B",
        "embed_dim": 1536,
        "output_name": "vlm2vec_2b",
    },
}


# ---------------------------------------------------------------------------
# ffmpeg helpers
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


def extract_video_segment(
    video_path: str, start: float, duration: float, temp_dir: str,
) -> str:
    ffmpeg_exe = _get_ffmpeg_exe()
    segment_path = os.path.join(temp_dir, "segment.mp4")
    cmd = [
        ffmpeg_exe, "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-an",
        "-preset", "ultrafast",
        "-loglevel", "error",
        segment_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return segment_path


def get_video_duration(video_path: str) -> float:
    ffmpeg_exe = _get_ffmpeg_exe()
    ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")
    cmd = [
        ffprobe_exe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# VLM2Vec Extractor — uses transformers + peft directly
# ---------------------------------------------------------------------------
class VLM2VecExtractor:
    """Load Qwen2-VL + VLM2Vec LoRA adapter for video embedding extraction.

    No VLM2Vec repo or flash-attn needed. Uses SDPA attention.
    """

    def __init__(self, model_variant: str = "v1_7b", device: str = "cuda"):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        cfg = MODEL_CONFIGS[model_variant]
        self.embed_dim = cfg["embed_dim"]
        self.output_name = cfg["output_name"]

        logger.info("Loading base model: %s", cfg["base_model"])
        # Load base Qwen2-VL (no flash-attn, uses sdpa)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg["base_model"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        logger.info("Applying LoRA adapter: %s", cfg["lora_adapter"])
        self.model = PeftModel.from_pretrained(
            self.model,
            cfg["lora_adapter"],
        )
        self.model = self.model.merge_and_unload()  # Merge LoRA for speed
        self.model.eval()
        self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(cfg["base_model"])
        self.device = device

        logger.info(
            "VLM2Vec loaded (variant=%s, dim=%d, device=%s, bfloat16, sdpa)",
            model_variant, self.embed_dim, device,
        )

    @torch.inference_mode()
    def extract_embedding(self, video_path: str) -> np.ndarray:
        """Extract a single embedding from a short video segment.

        Feeds the video through Qwen2-VL, takes the last token's
        hidden state as the fixed-dimensional embedding (VLM2Vec style).

        Returns shape (embed_dim,) float32.
        """
        from qwen_vl_utils import process_vision_info

        # Build conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 4.0,
                    },
                    {"type": "text", "text": "Represent this video."},
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Forward — get hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # Last token of last hidden layer = VLM2Vec embedding
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, dim)
        embedding = last_hidden[0, -1, :].float().cpu().numpy()  # (dim,)

        del outputs, inputs
        torch.cuda.empty_cache()
        return embedding


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def get_reference_n_trs(ref_dir, movie_type, stim_type, clip_stem):
    ref_path = ref_dir / movie_type / stim_type / f"{clip_stem}.npy"
    if ref_path.exists():
        return np.load(ref_path).shape[0]
    return None


def extract_features_for_clip(
    extractor: VLM2VecExtractor,
    video_path: str,
    n_trs: int,
) -> np.ndarray:
    """Extract embeddings for each TR of a video clip.

    Returns shape (n_trs, embed_dim), float32.
    """
    duration = get_video_duration(video_path)
    logger.info("  Video: %.1fs, extracting %d TRs", duration, n_trs)

    segment_duration = duration / n_trs
    start_times = np.arange(n_trs) * segment_duration

    output = np.zeros((n_trs, extractor.embed_dim), dtype=np.float32)

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, start in enumerate(tqdm(start_times, desc="  Encoding TRs", leave=False)):
            seg_dur = min(segment_duration, duration - start)
            if seg_dur <= 0.05:
                continue

            try:
                segment_path = extract_video_segment(
                    video_path, start, seg_dur, temp_dir,
                )
                embedding = extractor.extract_embedding(segment_path)
                output[i] = embedding
            except Exception as e:
                logger.warning("  Error at TR %d (t=%.1fs): %s", i, start, e)

            if i % 50 == 0 and i > 0:
                gc.collect()
                torch.cuda.empty_cache()

    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def list_movie_splits(stimuli_dir, movie_type, stimulus_type):
    movie_dir = os.path.join(stimuli_dir, "movies", movie_type, stimulus_type)
    mkv_files = sorted(glob.glob(os.path.join(movie_dir, "*.mkv")))
    if not mkv_files:
        logger.error("No .mkv files found in %s", movie_dir)
        sys.exit(1)
    logger.info("Found %d movie splits in %s", len(mkv_files), movie_dir)
    return mkv_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract VLM2Vec video embeddings per fMRI TR"
    )
    parser.add_argument(
        "--model_variant", type=str, default="v1_7b",
        choices=list(MODEL_CONFIGS.keys()),
        help="v1_7b (3584-D, ~16GB) or v1_2b (1536-D, ~6GB)",
    )
    parser.add_argument(
        "--movie_type", type=str, default="friends",
        choices=["friends", "movie10", "ood"],
    )
    parser.add_argument("--stimulus_type", type=str, default="s1")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--ref_modality", type=str, default=REF_MODALITY)
    parser.add_argument("--dry_run", action="store_true")
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

    # Load model
    extractor = VLM2VecExtractor(model_variant=args.model_variant, device=device)
    out_dir = data_dir / "features_npy_pooled" / extractor.output_name

    # List movie splits
    mkv_files = list_movie_splits(
        str(stimuli_dir), args.movie_type, args.stimulus_type,
    )
    if args.dry_run:
        mkv_files = mkv_files[:1]
        logger.info("DRY RUN: processing only 1 file")

    logger.info("Output dir: %s", out_dir)
    logger.info("Model: %s (dim=%d)", args.model_variant, extractor.embed_dim)

    for mkv_path in tqdm(mkv_files, desc="Movie splits"):
        clip_stem = Path(mkv_path).stem

        out_path = out_dir / args.movie_type / args.stimulus_type / f"{clip_stem}.npy"
        if out_path.exists():
            logger.info("  Skipping %s (already exists)", clip_stem)
            continue

        n_trs = get_reference_n_trs(
            ref_features_dir, args.movie_type, args.stimulus_type, clip_stem,
        )
        if n_trs is None:
            duration = get_video_duration(mkv_path)
            n_trs = max(1, int(duration / TR) - 1)

        logger.info("  %s → %d TRs", clip_stem, n_trs)

        features = extract_features_for_clip(extractor, mkv_path, n_trs)
        logger.info("  Output shape: %s", features.shape)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, features)
        logger.info("  Saved to %s", out_path)

        del features
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done! Features saved to %s", out_dir)


if __name__ == "__main__":
    main()
