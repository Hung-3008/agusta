"""Utility helpers for BMD feature extraction.

Functions
---------
get_bmd_paths       — canonical path struct for the project
load_annotations    — load annotations.json
load_llm_annotations — load llm_frame_annotations.json
build_text_input    — original text builder (spoken+desc0)
build_rich_text     — v2 rich text from all annotation sources
load_evenly_spaced_frames — N frames uniformly sampled from video
load_exact_frames_with_loop — exactly N frames (loop if video too short)
load_middle_frame_pil — middle frame as PIL Image
load_multi_frames_pil — multiple evenly-spaced frames as PIL Images
iter_vid_ids        — iterator over video IDs
save_feature        — save feature vector with dim validation
vid_id_to_split     — map video ID int to train/test
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("bmd_utils")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Path management
# ---------------------------------------------------------------------------

@dataclass
class BMDPaths:
    project_root: Path
    video_dir: Path        # mp4_h264/
    frames_dir: Path       # frames/
    frames_middle_dir: Path
    annotations_path: Path
    llm_annotations_path: Path
    feature_root: Path     # features_npy_pooled/
    feature_root_v2: Path  # features_npy_pooled_v2/


def get_bmd_paths(project_root: Path | None = None) -> BMDPaths:
    root = project_root or PROJECT_ROOT
    bmd = root / "Data" / "BOLDMomentsDataset"
    return BMDPaths(
        project_root=root,
        video_dir=bmd / "stimulus_set" / "mp4_h264",
        frames_dir=bmd / "stimulus_set" / "frames",
        frames_middle_dir=bmd / "stimulus_set" / "frames_middle",
        annotations_path=bmd / "derivatives" / "stimuli_metadata" / "annotations.json",
        llm_annotations_path=bmd / "derivatives" / "stimuli_metadata" / "llm_frame_annotations.json",
        feature_root=root / "Data" / "features_npy_pooled",
        feature_root_v2=root / "Data" / "features_npy_pooled_v2",
    )


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def load_annotations(path: Path) -> dict:
    """Load annotations.json → dict keyed by video ID string ('0001', ...)."""
    with open(path, "r") as f:
        return json.load(f)


def load_llm_annotations(path: Path) -> dict:
    """Load llm_frame_annotations.json."""
    if not path.exists():
        logger.warning("LLM annotations not found: %s", path)
        return {}
    with open(path, "r") as f:
        return json.load(f)


def build_text_input(
    vid_id: str,
    annotations: dict,
    mode: str = "spoken+desc0",
) -> str:
    """Original text builder — minimal text from annotations.

    Modes:
        spoken+desc0: spoken transcription + first text description
        spoken: spoken transcription only
        desc0: first text description only
    """
    ann = annotations.get(vid_id, {})
    parts = []

    if mode in ("spoken+desc0", "spoken"):
        spoken = ann.get("spoken_transcription", "")
        if spoken:
            parts.append(spoken.strip())

    if mode in ("spoken+desc0", "desc0"):
        descs = ann.get("text_descriptions", [])
        if descs:
            parts.append(descs[0].strip())

    return " ".join(parts) if parts else f"Video clip {vid_id}"


def build_rich_text(
    vid_id: str,
    annotations: dict,
    llm_annotations: dict | None = None,
) -> str:
    """V2 rich text builder — uses ALL annotation sources for maximum context.

    Combines: all 5 text descriptions, spoken transcription, actions,
    objects, scenes, and optionally one LLM-generated caption.
    """
    ann = annotations.get(vid_id, {})
    parts = []

    # All 5 text descriptions
    descs = ann.get("text_descriptions", [])
    if descs:
        parts.append("Descriptions: " + " | ".join(d.strip() for d in descs if d.strip()))

    # Spoken transcription
    spoken = ann.get("spoken_transcription", "")
    if spoken and spoken.strip():
        parts.append("Narration: " + spoken.strip())

    # Actions (list of strings from 5 annotators)
    actions = ann.get("actions", [])
    if actions:
        unique_actions = sorted(set(
            a.strip() for a in actions
            if isinstance(a, str) and a.strip() and a.strip() != "--"
        ))
        if unique_actions:
            parts.append("Actions: " + ", ".join(unique_actions))

    # Objects (list of [obj1, obj2, obj3] per annotator)
    objects = ann.get("objects", [])
    if objects:
        flat = set()
        for obj_list in objects:
            if isinstance(obj_list, list):
                flat.update(o.strip() for o in obj_list if isinstance(o, str) and o.strip() != "--")
            elif isinstance(obj_list, str) and obj_list.strip() != "--":
                flat.add(obj_list.strip())
        if flat:
            parts.append("Objects: " + ", ".join(sorted(flat)))

    # Scenes (list of scene labels)
    scenes = ann.get("scenes", [])
    if scenes:
        unique_scenes = sorted(set(
            s.strip() for s in scenes
            if isinstance(s, str) and s.strip() and s.strip() != "--"
        ))
        if unique_scenes:
            parts.append("Scenes: " + ", ".join(unique_scenes))

    # LLM caption (optional, just one to add diversity)
    if llm_annotations and vid_id in llm_annotations:
        llm_entry = llm_annotations[vid_id]
        if isinstance(llm_entry, dict):
            for model_name, captions in llm_entry.items():
                if captions and isinstance(captions, list) and captions[0]:
                    parts.append("Visual caption: " + captions[0].strip())
                    break

    return ". ".join(parts) if parts else f"Video clip {vid_id}"


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def _read_video_frames_cv2(video_path: Path | str) -> np.ndarray:
    """Read all frames from a video file using OpenCV. Returns (N, H, W, 3) uint8 RGB."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    return np.stack(frames, axis=0)


def load_evenly_spaced_frames(
    video_path: Path | str,
    n_frames: int = 8,
) -> np.ndarray:
    """Load n_frames uniformly spaced from a video. Returns (n_frames, H, W, 3) uint8."""
    all_frames = _read_video_frames_cv2(video_path)
    total = len(all_frames)

    if total <= n_frames:
        return all_frames

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    return all_frames[indices]


def load_exact_frames_with_loop(
    video_path: Path | str,
    num_frames: int = 64,
) -> np.ndarray:
    """Load exactly num_frames from video, looping if video is shorter. Returns (num_frames, H, W, 3) uint8."""
    all_frames = _read_video_frames_cv2(video_path)
    total = len(all_frames)

    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        return all_frames[indices]

    # Loop
    reps = (num_frames // total) + 1
    looped = np.tile(all_frames, (reps, 1, 1, 1))
    indices = np.linspace(0, len(looped) - 1, num_frames, dtype=int)
    return looped[indices]


def load_middle_frame_pil(
    vid_id: str,
    frames_dir: Path,
    video_path: Path,
):
    """Load the middle frame as a PIL Image.

    First tries pre-extracted frames_middle/, then frames/{vid_id}/,
    then falls back to extracting from video.
    """
    from PIL import Image

    # Try frames_middle/ first
    middle_dir = frames_dir.parent / "frames_middle"
    if middle_dir.exists():
        candidates = sorted(middle_dir.glob(f"{vid_id}*"))
        if candidates:
            return Image.open(candidates[0]).convert("RGB")

    # Try frames/{vid_id}/ — pick middle frame
    frame_dir = frames_dir / vid_id
    if frame_dir.exists():
        frame_files = sorted(frame_dir.glob("*.jpg"))
        if frame_files:
            mid_idx = len(frame_files) // 2
            return Image.open(frame_files[mid_idx]).convert("RGB")

    # Fallback: extract from video
    frames = _read_video_frames_cv2(video_path)
    mid_frame = frames[len(frames) // 2]
    return Image.fromarray(mid_frame)


def load_multi_frames_pil(
    vid_id: str,
    frames_dir: Path,
    video_path: Path,
    n_frames: int = 4,
):
    """Load n evenly-spaced frames as PIL Images for multi-frame VLMs.

    Returns a list of PIL Images.
    """
    from PIL import Image

    # Try pre-extracted frames first
    frame_dir = frames_dir / vid_id
    if frame_dir.exists():
        frame_files = sorted(frame_dir.glob("*.jpg"))
        if len(frame_files) >= n_frames:
            indices = np.linspace(0, len(frame_files) - 1, n_frames, dtype=int)
            return [Image.open(frame_files[i]).convert("RGB") for i in indices]

    # Fallback: extract from video
    frames = load_evenly_spaced_frames(video_path, n_frames=n_frames)
    return [Image.fromarray(f) for f in frames]


# ---------------------------------------------------------------------------
# Video ID helpers
# ---------------------------------------------------------------------------

def vid_id_to_split(vid_id: str | int) -> str:
    """Map video ID to 'train' or 'test'."""
    n = int(vid_id)
    return "train" if n <= 1000 else "test"


def iter_vid_ids(
    split: str = "all",
    start_id: int | None = None,
    end_id: int | None = None,
):
    """Yield video ID strings ('0001', ..., '1102').

    Args:
        split: 'train' (1-1000), 'test' (1001-1102), or 'all'
        start_id: override start (inclusive)
        end_id: override end (inclusive)
    """
    if split == "train":
        s, e = 1, 1000
    elif split == "test":
        s, e = 1001, 1102
    else:
        s, e = 1, 1102

    if start_id is not None:
        s = max(s, start_id)
    if end_id is not None:
        e = min(e, end_id)

    for i in range(s, e + 1):
        yield f"{i:04d}"


# ---------------------------------------------------------------------------
# Feature I/O
# ---------------------------------------------------------------------------

def save_feature(
    out_path: Path,
    feat: np.ndarray,
    expected_dim: int,
) -> None:
    """Save a feature vector as (1, D) float32 .npy, with dim validation."""
    feat = np.asarray(feat, dtype=np.float32).squeeze()

    if feat.ndim == 0:
        raise ValueError(f"Feature is scalar, expected dim={expected_dim}")

    actual_dim = feat.shape[-1]
    if actual_dim != expected_dim:
        logger.warning(
            "Dim mismatch: got %d, expected %d → truncating/padding. File: %s",
            actual_dim, expected_dim, out_path,
        )
        if actual_dim > expected_dim:
            feat = feat[:expected_dim]
        else:
            padded = np.zeros(expected_dim, dtype=np.float32)
            padded[:actual_dim] = feat
            feat = padded

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat.reshape(1, -1))
