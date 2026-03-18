"""Evaluate BrainFlow — generate fMRI predictions for Friends S7 submission.

Aligned with train_brainflow.py:
  - Same sliding window construction with HRF delay
  - Full sequence predictions (not just last TR)
  - Proper feature resampling matching training dataset

Usage:
    python src/evaluate_brainflow.py --config src/configs/brain_flow.yaml
    python src/evaluate_brainflow.py --config src/configs/brain_flow.yaml --guidance_scale 3.0
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gc
import torch
import numpy as np
import zipfile
from tqdm import tqdm

from src.models.brainflow.brain_flow import BrainFlowCFM
from src.data.precompute_latents import load_vae
from src.data.dataset import load_config, load_feature_clip_perfile, resample_features_to_tr


def load_fmri_samples(project_root: Path, subject: str) -> dict:
    """Load target fMRI sample counts for Friends season 7.
    Returns dict: {episode_key: n_trs}, e.g. {'s07e01a': 460, ...}
    """
    samples_file = (
        project_root / "Data" / "algonauts_2025.competitors" / "fmri"
        / subject / "target_sample_number"
        / f"{subject}_friends-s7_fmri_samples.npy"
    )
    return np.load(samples_file, allow_pickle=True).item()


def load_clip_features_as_2d(
    features_dir: str,
    cfg: dict,
    task: str,
    stim_type: str,
    clip_name: str,
    n_trs_target: int,
) -> dict:
    """Load and resample features for one clip, returning {mod: (T, D)} numpy.

    Matches the training dataset's feature format:
      - load_feature_clip_perfile → (1, D, T_feat)
      - resample_features_to_tr  → (1, D, n_trs)
      - Transpose to (n_trs, D)
    """
    fmri_tr = cfg["fmri"]["tr"]
    layer_aggregation = cfg["raw_features"].get("layer_aggregation", "mean")
    feature_freq = cfg["raw_features"].get("sample_freq", 2.0)

    mod_features = {}
    for mod, mod_cfg in cfg["raw_features"]["modalities"].items():
        try:
            raw = load_feature_clip_perfile(
                features_dir, mod, mod_cfg,
                task, stim_type, clip_name,
                layer_aggregation=layer_aggregation,
            )
            # raw: (1, D, T_feat)
            if raw.ndim == 2:
                raw = raw[np.newaxis, :, :]  # (1, D, T_feat)

            # Resample to match n_trs_target directly
            # Features are at ~1/TR rate (not 2Hz), so use n_trs_target
            resampled = resample_features_to_tr(raw, feature_freq, fmri_tr, n_trs_target)

            # Convert to (T, D) — same as training NPY format
            feat_2d = resampled[0].T.astype(np.float32)  # (n_trs_target, D)
            mod_features[mod] = feat_2d

            del raw, resampled
        except Exception as e:
            print(f"  Missing {mod} for {clip_name}: {e}")

    return mod_features


def predict_episode(
    model,
    vae,
    mod_features: dict,
    n_trs: int,
    cfg: dict,
    device: torch.device,
    batch_size: int = 8,
    n_timesteps: int = 20,
    guidance_scale: float = 2.0,
) -> np.ndarray:
    """Generate fMRI predictions for one episode using sliding windows.

    Matches training pipeline exactly:
      - Windows start at hrf_delay, stride by stride_trs
      - Features shifted back by hrf_delay TRs
      - Full sequence predictions (all TRs in window, not just last)
      - Overlap-average where windows overlap

    Returns:
        fmri_pred: (n_trs, 1000) float32
    """
    hrf_delay = cfg["fmri"].get("hrf_delay", 3)
    fmri_tr = cfg["fmri"]["tr"]
    seq_len = max(1, int(round(cfg["sliding_window"]["context_duration"] / fmri_tr)))
    stride = max(1, int(round(cfg["sliding_window"]["stride"] / fmri_tr)))
    latent_dim = cfg["brainflow"]["latent_dim"]

    # Accumulation arrays for overlap-averaging
    latent_sum = np.zeros((n_trs, latent_dim), dtype=np.float64)
    latent_count = np.zeros(n_trs, dtype=np.float64)

    # --- Build windows matching training exactly ---
    # Training: for start_idx in range(self.hrf_delay, n_trs, self.stride)
    # But we also need predictions for TRs [0, hrf_delay). 
    # Handle by starting from 0 with adjusted feature window.
    windows = []
    for start in range(0, n_trs, stride):
        actual_len = min(seq_len, n_trs - start)
        if actual_len < 1:
            break
        windows.append((start, actual_len))

    # --- Batch inference ---
    for wi in range(0, len(windows), batch_size):
        batch_windows = windows[wi: wi + batch_size]
        B = len(batch_windows)

        # Build per-modality feature batches
        batch_features = {}
        for mod, feat_2d in mod_features.items():
            T_feat = feat_2d.shape[0]
            D = feat_2d.shape[1]
            batch_tensors = np.zeros((B, seq_len, D), dtype=np.float32)

            for j, (start, actual_len) in enumerate(batch_windows):
                # HRF delay: features at [start - hrf_delay, start - hrf_delay + actual_len)
                feat_start = start - hrf_delay
                feat_end = feat_start + actual_len

                # Handle boundaries: clamp to [0, T_feat) and zero-pad
                src_start = max(0, feat_start)
                src_end = min(T_feat, feat_end)
                dst_start = max(0, -feat_start)  # offset if feat_start < 0
                copy_len = src_end - src_start

                if copy_len > 0:
                    batch_tensors[j, dst_start:dst_start + copy_len] = feat_2d[src_start:src_end]

            batch_features[mod] = torch.from_numpy(batch_tensors).to(device)

        # Model inference
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_latents = model.synthesise(
                batch_features,
                n_timesteps=n_timesteps,
                guidance_scale=guidance_scale,
            )  # (B, seq_len, latent_dim)

        pred_np = pred_latents.float().cpu().numpy()

        # Accumulate predictions (overlap-average)
        for j, (start, actual_len) in enumerate(batch_windows):
            latent_sum[start:start + actual_len] += pred_np[j, :actual_len]
            latent_count[start:start + actual_len] += 1.0

        del batch_features, pred_latents, pred_np
        torch.cuda.empty_cache()

    # Average overlapping predictions
    latent_count = np.maximum(latent_count, 1.0)
    pred_latents_avg = (latent_sum / latent_count[:, None]).astype(np.float32)

    # --- VAE decode latents → fMRI ---
    # Process in chunks to avoid OOM
    chunk_size = 200
    fmri_chunks = []
    for cs in range(0, n_trs, chunk_size):
        ce = min(cs + chunk_size, n_trs)
        lat_chunk = torch.from_numpy(pred_latents_avg[cs:ce]).unsqueeze(0).to(device)
        dummy_subj = torch.zeros(1, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            fmri_chunk = vae.decode(lat_chunk, dummy_subj)  # (1, chunk_len, 1000)

        fmri_chunks.append(fmri_chunk.squeeze(0).float().cpu().numpy())
        del lat_chunk, fmri_chunk
        torch.cuda.empty_cache()

    return np.concatenate(fmri_chunks, axis=0)  # (n_trs, 1000)


@torch.inference_mode()
def evaluate_brainflow():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/brain_flow.yaml")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_timesteps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip subjects whose per-subject npy file already exists.")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="Reprocess all subjects from scratch.")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)

    # Resolve paths
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_data_root"] = str(PROJECT_ROOT / cfg["data_root"])
    cfg["_fmri_dir"] = str(PROJECT_ROOT / cfg["data_root"] / cfg["fmri"]["dir"])
    features_dir = str(PROJECT_ROOT / cfg["raw_features"]["dir"])

    out_dir = Path(cfg.get("output_dir", "outputs/brainflow"))
    output_dir = PROJECT_ROOT / "results" / "brain_flow_submission"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print key config for verification
    hrf_delay = cfg["fmri"].get("hrf_delay", 3)
    fmri_tr = cfg["fmri"]["tr"]
    seq_len = max(1, int(round(cfg["sliding_window"]["context_duration"] / fmri_tr)))
    stride = max(1, int(round(cfg["sliding_window"]["stride"] / fmri_tr)))
    print(f"Config: hrf_delay={hrf_delay}, seq_len={seq_len}, stride={stride}, "
          f"n_timesteps={args.n_timesteps}, guidance_scale={args.guidance_scale}")

    # ── 1. Load VAE ──────────────────────────────────────────────────────────
    print("Loading VAE...")
    vae = load_vae(cfg, device)

    # ── 2. Build model ───────────────────────────────────────────────────────
    bf_cfg = cfg["brainflow"]
    modality_dims = {}
    for mod_name, mod_cfg in cfg["raw_features"]["modalities"].items():
        dim = mod_cfg["dim"]
        if cfg["raw_features"].get("layer_aggregation") == "cat":
            n_layers = mod_cfg.get("n_layers", 1)
            dim = dim * n_layers
        modality_dims[mod_name] = dim

    print(f"Modality dims: {modality_dims}")

    model = BrainFlowCFM(
        modality_dims=modality_dims,
        latent_dim=bf_cfg["latent_dim"],
        encoder_params=bf_cfg["encoder"],
        velocity_net_params=bf_cfg.get("velocity_net", {}),
        cfm_params=bf_cfg.get("cfm", {}),
        cfg_drop_prob=bf_cfg.get("cfg_drop_prob", 0.1),
        n_voxels=cfg["fmri"].get("n_voxels", 1000),
    ).to(device)

    ckpt_path = out_dir / "best.pt"
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    subjects = ["sub-01", "sub-02", "sub-03", "sub-05"]

    # ── 3. Per-subject inference ─────────────────────────────────────────────
    subject_result_paths = {}

    for subject in subjects:
        subj_path = output_dir / f"{subject}_predictions.npy"

        # Resume: skip if already computed
        if args.resume and subj_path.exists():
            print(f"\n[RESUME] Skipping {subject} — found {subj_path}")
            subject_result_paths[subject] = subj_path
            continue

        print(f"\n{'='*60}")
        print(f"Processing {subject}...")
        print(f"{'='*60}")

        subject_dict = {}
        target_samples = load_fmri_samples(PROJECT_ROOT, subject)

        for epi, n_trs in tqdm(target_samples.items(), desc=subject):
            clip_name = f"friends_{epi}"

            # ── Load & resample features (same as training) ──────────────
            mod_features = load_clip_features_as_2d(
                features_dir, cfg, "friends", "s7", clip_name, n_trs,
            )

            if not mod_features:
                print(f"  No features found for {clip_name}. Emitting zeros.")
                subject_dict[epi] = np.zeros((n_trs, 1000), dtype=np.float32)
                continue

            try:
                fmri_pred = predict_episode(
                    model, vae, mod_features, n_trs, cfg, device,
                    batch_size=args.batch_size,
                    n_timesteps=args.n_timesteps,
                    guidance_scale=args.guidance_scale,
                )
                subject_dict[epi] = fmri_pred
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  Inference failed for {clip_name}: {e}. Emitting zeros.")
                subject_dict[epi] = np.zeros((n_trs, 1000), dtype=np.float32)

            del mod_features
            gc.collect()

        # ── Save subject result ──────────────────────────────────────────
        np.save(subj_path, subject_dict)
        subject_result_paths[subject] = subj_path
        print(f"  Saved {subject} → {subj_path} ({subj_path.stat().st_size / 1024 / 1024:.1f} MB)")

        del subject_dict
        gc.collect()
        torch.cuda.empty_cache()

    # ── 4. Assemble final submission ─────────────────────────────────────────
    print("\nAssembling submission dict from per-subject files...")
    submission_dict = {}
    for subject, path in subject_result_paths.items():
        submission_dict[subject] = np.load(path, allow_pickle=True).item()

    submission_path = output_dir / "submission.npy"
    print(f"Saving submission.npy to {submission_path}...")
    np.save(submission_path, submission_dict)

    del submission_dict
    gc.collect()

    zip_path = submission_path.with_suffix(".zip")
    print(f"Zipping to {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(submission_path, arcname="submission.npy")

    print(f"Done! Zip size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Clean up per-subject temp files
    for path in subject_result_paths.values():
        path.unlink(missing_ok=True)
    print("Cleaned up per-subject temp files.")


if __name__ == "__main__":
    evaluate_brainflow()
