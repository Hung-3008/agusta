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

from src.models.brainflow.brain_flow_v2 import BrainFlowCFMv2
from src.data.precompute_latents import load_vae
from src.data.dataset import load_config, load_feature_clip_perfile, resample_features_to_tr
import importlib.util

spec = importlib.util.spec_from_file_location(
    "test_encoding_utils",
    str(PROJECT_ROOT / "src/models/challenge_baseline_model/03_encoding_model_testing/test_encoding_utils.py")
)
test_encoding_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_encoding_utils)
load_fmri_samples = test_encoding_utils.load_fmri_samples


def make_window_batch(raw_feats: dict, s_start: int, s_end: int, context_trs: int) -> dict:
    """
    Build a batch of sliding-window feature tensors for TRs [s_start, s_end).
    Returns dict: mod -> Tensor of shape (batch, context_trs, D)
    Avoids storing all windows at once by operating on a chunk.
    """
    batch_dict = {}
    batch_size = s_end - s_start

    for mod, resampled in raw_feats.items():
        L, D, T = resampled.shape
        LD = L * D
        windows = np.empty((batch_size, context_trs, LD), dtype=np.float32)

        for i, s in enumerate(range(s_start, s_end)):
            ctx_start = s - context_trs
            ctx_end = s  # exclusive

            if ctx_start < 0:
                pad_left_len = -ctx_start
                valid_start = 0
                valid_end = min(ctx_end, T)
                feat_slice = resampled[:, :, valid_start:valid_end]  # (L, D, valid_len)
                valid_len = feat_slice.shape[2]
                w = np.zeros((L, D, context_trs), dtype=np.float32)
                # Place valid slice after left padding
                w[:, :, pad_left_len:pad_left_len + valid_len] = feat_slice
            else:
                valid_start = ctx_start
                valid_end = min(ctx_end, T)
                feat_slice = resampled[:, :, valid_start:valid_end]
                valid_len = feat_slice.shape[2]
                w = np.zeros((L, D, context_trs), dtype=np.float32)
                w[:, :, :valid_len] = feat_slice

            # w: (L, D, context_trs) -> reshape -> (context_trs, L*D)
            windows[i] = w.reshape(LD, context_trs).T

        batch_dict[mod] = torch.from_numpy(windows)  # (B, context_trs, LD)

    return batch_dict


@torch.inference_mode()
def run_episode(
    model,
    vae,
    raw_feats: dict,
    n_trs: int,
    context_trs: int,
    batch_size: int,
    n_timesteps: int,
    guidance_scale: float,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference for one episode. Returns (n_trs, 1000) float32 array.
    Streams through TRs in chunks to avoid accumulating all windows in RAM.
    """
    predictions = []

    for i in range(0, n_trs, batch_size):
        s_end = min(i + batch_size, n_trs)
        # Build only this chunk's windows on-the-fly
        batch_features = make_window_batch(raw_feats, i, s_end, context_trs)
        batch_features = {mod: t.to(device) for mod, t in batch_features.items()}

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_latents = model.synthesise(
                batch_features,
                n_timesteps=n_timesteps,
                guidance_scale=guidance_scale,
            )  # (B, latent_dim)

            dummy_subj = torch.zeros(pred_latents.shape[0], dtype=torch.long, device=device)
            fmri_pred = vae.decode(pred_latents, dummy_subj)  # (B, 1000)

        predictions.append(fmri_pred.float().cpu().numpy())

        # Free GPU tensors immediately
        del batch_features, pred_latents, dummy_subj, fmri_pred
        torch.cuda.empty_cache()

    return np.concatenate(predictions, axis=0)  # (n_trs, 1000)


@torch.inference_mode()
def evaluate_brainflow():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/brain_flow_v2.yaml")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_timesteps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)

    # Resolve paths
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_data_root"] = str(PROJECT_ROOT / cfg["data_root"])
    cfg["_fmri_dir"] = str(PROJECT_ROOT / cfg["data_root"] / cfg["fmri"]["dir"])
    cfg["_features_dir"] = str(PROJECT_ROOT / cfg["raw_features"]["dir"])

    out_dir = Path(cfg.get("output_dir", "outputs/brain_flow_v2_dit"))
    output_dir = PROJECT_ROOT / "results" / "brain_flow_submission"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    model = BrainFlowCFMv2(
        modality_dims=modality_dims,
        latent_dim=bf_cfg["latent_dim"],
        encoder_params=bf_cfg["encoder"],
        decoder_type=bf_cfg.get("decoder_type", "dit"),
        decoder_params=bf_cfg["decoder"],
        cfm_params=bf_cfg["cfm"],
    ).to(device)

    ckpt_path = out_dir / "best.pt"
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # ── 3. Inference settings ─────────────────────────────────────────────────
    context_trs = round(cfg["sliding_window"]["context_duration"] / cfg["fmri"]["tr"])
    features_dir = cfg["_features_dir"]
    layer_aggregation = cfg["raw_features"]["layer_aggregation"]
    feature_freq = cfg["raw_features"]["sample_freq"]
    fmri_tr = cfg["fmri"]["tr"]

    print(f"context_trs={context_trs}, n_timesteps={args.n_timesteps}, "
          f"guidance_scale={args.guidance_scale}, batch_size={args.batch_size}")

    subjects = ["sub-01", "sub-02", "sub-03", "sub-05"]

    # ── 4. Per-subject inference — save incrementally ─────────────────────────
    # We save each subject's dict separately to avoid one giant dict in RAM.
    subject_result_paths = {}

    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Processing {subject}...")
        print(f"{'='*60}")

        subject_dict = {}

        class Args:
            project_dir = str(PROJECT_ROOT)

        target_samples = load_fmri_samples(Args(), int(subject[-2:]))

        for epi, n_trs in tqdm(target_samples.items(), desc=subject):
            clip_name = f"friends_{epi}"

            # ── Load & resample features ──────────────────────────────────
            raw_feats = {}
            for mod, mod_cfg in cfg["raw_features"]["modalities"].items():
                try:
                    raw = load_feature_clip_perfile(
                        features_dir, mod, mod_cfg,
                        "friends", "s7", clip_name,
                        layer_aggregation=layer_aggregation,
                    )
                    T_orig = raw.shape[2]
                    estimated_trs = int(round((T_orig / feature_freq) / fmri_tr))
                    resampled = resample_features_to_tr(raw, feature_freq, fmri_tr, estimated_trs)
                    raw_feats[mod] = resampled
                    del raw  # free original immediately
                except Exception as e:
                    print(f"  Missing {mod} for {clip_name}: {e}")

            if not raw_feats:
                print(f"  No features found for {clip_name}. Emitting zeros.")
                subject_dict[epi] = np.zeros((n_trs, 1000), dtype=np.float32)
                continue

            # ── Streaming batch inference ─────────────────────────────────
            try:
                episode_preds = run_episode(
                    model=model,
                    vae=vae,
                    raw_feats=raw_feats,
                    n_trs=n_trs,
                    context_trs=context_trs,
                    batch_size=args.batch_size,
                    n_timesteps=args.n_timesteps,
                    guidance_scale=args.guidance_scale,
                    device=device,
                )
                subject_dict[epi] = episode_preds
            except Exception as e:
                print(f"  Inference failed for {clip_name}: {e}. Emitting zeros.")
                subject_dict[epi] = np.zeros((n_trs, 1000), dtype=np.float32)

            # Free feature arrays after each episode
            del raw_feats
            gc.collect()

        # ── Save subject result to disk immediately ───────────────────────
        subj_path = output_dir / f"{subject}_predictions.npy"
        np.save(subj_path, subject_dict)
        subject_result_paths[subject] = subj_path
        print(f"  Saved {subject} → {subj_path} ({subj_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Free subject dict from RAM
        del subject_dict
        gc.collect()
        torch.cuda.empty_cache()

    # ── 5. Assemble final submission dict ─────────────────────────────────────
    print("\nAssembling submission dict from per-subject files...")
    submission_dict = {}
    for subject, path in subject_result_paths.items():
        submission_dict[subject] = np.load(path, allow_pickle=True).item()

    submission_path = output_dir / "submission.npy"
    print(f"Saving submission.npy to {submission_path}...")
    np.save(submission_path, submission_dict)

    # Free assembled dict
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
