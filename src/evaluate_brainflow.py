"""Evaluate BrainFlowDirect — generate fMRI predictions for Friends S7 submission.

Uses the same data pipeline as train_brainflow_direct.py:
  - Pre-extracted context latents (multi-modal ensemble)
  - Per-TR sliding window with HRF delay
  - Global stats denormalization for raw fMRI output
  - Subject embedding conditioning

Usage:
    python src/evaluate_brainflow_direct.py --config src/configs/brainflow_gaussian.yaml
    python src/evaluate_brainflow_direct.py --config src/configs/brainflow_v2.yaml --n_timesteps 50
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gc
import joblib
import torch
import numpy as np
import zipfile
from tqdm import tqdm

from src.models.brainflow.brainflow import BrainFlow
from src.data.dataset import load_config


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


def load_context_for_clip(
    context_dirs: list,
    task: str,
    stim_type: str,
    clip_name: str,
) -> np.ndarray:
    """Load and concatenate multi-modal context features for one clip.
    
    Handles missing modalities by zero-padding (e.g. silent films).
    Returns: (n_trs, context_dim) float32 array.
    """
    ctx_arrays = []
    modality_dim_cache = {}
    
    for ctx_dir in context_dirs:
        c_dir = ctx_dir / task / stim_type
        
        # Try multiple naming conventions
        ctx_path = c_dir / f"{clip_name}.npy"
        if not ctx_path.exists():
            ctx_path = c_dir / f"{task}_{clip_name}.npy"
        
        if ctx_path.exists():
            ctx_arrays.append(np.load(ctx_path).astype(np.float32))
        else:
            # Zero-pad missing modality
            key = str(ctx_dir)
            if key not in modality_dim_cache:
                for f in ctx_dir.rglob("*.npy"):
                    modality_dim_cache[key] = np.load(f).shape[-1]
                    break
            mod_dim = modality_dim_cache.get(key, 512)
            ref_len = ctx_arrays[0].shape[0] if ctx_arrays else 100
            ctx_arrays.append(np.zeros((ref_len, mod_dim), dtype=np.float32))
    
    # Truncate to minimum temporal length
    min_len = min(arr.shape[0] for arr in ctx_arrays)
    ctx_arrays = [arr[:min_len] for arr in ctx_arrays]
    
    return np.concatenate(ctx_arrays, axis=-1)


@torch.inference_mode()
def evaluate_brainflow_direct():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/brainflow.yaml")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_timesteps", type=int, default=20)
    parser.add_argument("--solver_method", type=str, default="euler")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Starting noise std (overrides config). 0=deterministic.")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip subjects whose per-subject npy file already exists.")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)

    # Resolve paths
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_data_root"] = str(PROJECT_ROOT / cfg["data_root"])
    cfg["_fmri_dir"] = str(PROJECT_ROOT / cfg["data_root"] / cfg["fmri"]["dir"])

    fmri_dir = Path(cfg["_fmri_dir"])

    # Context dirs
    if "context_latent_dirs" in cfg:
        context_dirs = [PROJECT_ROOT / d for d in cfg["context_latent_dirs"]]
    else:
        context_dirs = [PROJECT_ROOT / cfg["context_latent_dir"]]
    
    subjects = cfg["subjects"]
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    # Sliding window config
    feature_context_trs = cfg["sliding_window"].get("feature_context_trs", 10)
    feat_seq_len = feature_context_trs + 1  # 10 past + 1 current

    # NOTE: S7 test features are pre-aligned 1:1 with target TRs
    # (e.g. 460 feature frames for 460 target TRs).
    # Unlike training features (full movie length needing excl_start + hrf_delay),
    # test features already account for these offsets → set both to 0.
    hrf_delay = 0
    excl_start = 0

    # Global stats for denormalization
    use_global_stats = cfg["fmri"].get("use_global_stats", False)
    fmri_stats = {}
    if use_global_stats:
        for subj in subjects:
            stats_dir = fmri_dir / subj / "stats"
            mean_path = stats_dir / "global_mean.npy"
            std_path = stats_dir / "global_std.npy"
            if mean_path.exists() and std_path.exists():
                fmri_stats[subj] = {
                    "mean": np.load(mean_path).astype(np.float32),
                    "std": np.load(std_path).astype(np.float32),
                }
                print(f"Loaded global stats for {subj}")

    # Output dirs
    out_dir = Path(cfg.get("output_dir", "outputs/brainflow_direct"))
    output_dir = PROJECT_ROOT / "results" / "brainflow_direct_submission"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 0.5. Load PCA model if PCA mode ────────────────────────────────────────
    pca_mode = "pca_model_path" in cfg
    pca_components = None  # (K, V)
    pca_mean = None        # (V,)
    if pca_mode:
        pca_path = PROJECT_ROOT / cfg["pca_model_path"]
        pca = joblib.load(pca_path)
        pca_components = torch.from_numpy(pca.components_.astype(np.float32)).to(device)
        pca_mean = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)
        print(f"PCA mode: {pca.n_components_} components, explained_var={pca.explained_variance_ratio_.sum():.4f}")
        del pca

    # ── 1. Build model ───────────────────────────────────────────────────────────
    bf_cfg = cfg["brainflow"]
    vn_cfg = bf_cfg.get("velocity_net", {})

    vn_params = dict(vn_cfg)
    modality_dims = cfg.get("modality_dims", None)
    if modality_dims:
        vn_params["modality_dims"] = modality_dims
    model_version = cfg.get("model_version", "v2")
    source_mode = bf_cfg.get("source_mode", "csfm")
    if model_version == "v3":
        reg_weight = bf_cfg.get("reg_weight", 0.5)
        model = BrainFlow(
            output_dim=bf_cfg["output_dim"],
            velocity_net_params=vn_params,
            n_subjects=len(subjects),
            reg_weight=reg_weight,
            cont_weight=bf_cfg.get("cont_weight", 0.1),
            cont_dim=bf_cfg.get("cont_dim", 256),
            tensor_fm_params=bf_cfg.get("tensor_fm", None),
            indi_flow_matching=bf_cfg.get("indi_flow_matching", False),
            indi_train_time_sqrt=bf_cfg.get("indi_train_time_sqrt", False),
            indi_min_denom=bf_cfg.get("indi_min_denom", 1e-3),
        ).to(device)
    else:
        sp_params = dict(bf_cfg.get("source_predictor", {}))
        model = BrainFlow_V2(
            output_dim=bf_cfg["output_dim"],
            velocity_net_params=vn_params,
            n_subjects=len(subjects),
            source_predictor_params=sp_params,
            source_mode=source_mode,
        ).to(device)

    ckpt_path = PROJECT_ROOT / out_dir / "best.pt"
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params (source_mode={source_mode})")
    print(f"Config: hrf_delay={hrf_delay}, context_trs={feature_context_trs}, "
          f"n_timesteps={args.n_timesteps}, solver={args.solver_method}")

    # ── 2. Per-subject inference ─────────────────────────────────────────────
    subject_result_paths = {}

    for subject in subjects:
        subj_path = output_dir / f"{subject}_predictions.npy"

        if args.resume and subj_path.exists():
            print(f"\n[RESUME] Skipping {subject} — found {subj_path}")
            subject_result_paths[subject] = subj_path
            continue

        print(f"\n{'='*60}")
        print(f"Processing {subject}...")
        print(f"{'='*60}")

        subject_idx = subject_to_idx[subject]
        subject_dict = {}
        target_samples = load_fmri_samples(PROJECT_ROOT, subject)

        for epi, n_trs in tqdm(target_samples.items(), desc=subject):
            clip_name = f"friends_{epi}"

            # Load multi-modal context
            ctx_data = load_context_for_clip(
                context_dirs, "friends", "s7", clip_name,
            )

            if ctx_data is None or ctx_data.shape[0] == 0:
                print(f"  No context for {clip_name}. Emitting zeros.")
                subject_dict[epi] = np.zeros((n_trs, 1000), dtype=np.float32)
                continue

            # ── Build per-TR batches ──
            # The scoring script maps submission index `target_tr` directly to
            # the fMRI target index. But the features corresponding to this target
            # are offset by excl_start in the actual movie TR space!
            # See training: actual_movie_tr = target_tr + excl_start
            all_contexts = []
            for target_tr in range(n_trs):
                actual_movie_tr = target_tr + excl_start
                feat_current_tr = actual_movie_tr - hrf_delay

                feat_start = feat_current_tr - feature_context_trs
                feat_end = feat_current_tr + 1

                safe_start = max(0, feat_start)
                safe_end = min(ctx_data.shape[0], feat_end)

                if safe_start < safe_end:
                    ctx = ctx_data[safe_start:safe_end]
                else:
                    ctx = np.zeros((0, ctx_data.shape[-1]), dtype=np.float32)

                if ctx.shape[0] < feat_seq_len:
                    pad_len = feat_seq_len - ctx.shape[0]
                    ctx = np.pad(ctx, ((pad_len, 0), (0, 0)), mode="constant")

                all_contexts.append(ctx)

            all_contexts = np.stack(all_contexts)  # (n_trs, feat_seq_len, ctx_dim)

            # ── Batch inference ──
            fmri_preds = []
            for bi in range(0, n_trs, args.batch_size):
                be = min(bi + args.batch_size, n_trs)
                B = be - bi

                ctx_batch = torch.from_numpy(all_contexts[bi:be]).to(device)
                subj_batch = torch.full((B,), subject_idx, dtype=torch.long, device=device)

                solver_args = cfg.get("solver_args", {})
                temperature = args.temperature if args.temperature is not None else solver_args.get("temperature", 0.0)
                synth_kwargs = dict(
                    n_timesteps=args.n_timesteps,
                    solver_method=args.solver_method,
                    subject_ids=subj_batch,
                    temperature=temperature,
                )
                if model_version == "v3":
                    cfg_scale = solver_args.get("cfg_scale", 0.0)
                    if cfg_scale > 0:
                        synth_kwargs["cfg_scale"] = cfg_scale
                    tw = solver_args.get("time_grid_warp")
                    if tw:
                        synth_kwargs["time_grid_warp"] = tw

                # Use float32 for ODE solver — bfloat16 causes overflow/NaN
                pred = model.synthesise(
                    ctx_batch, **synth_kwargs
                )  # (B, output_dim) — PCA (100) or fMRI (1000)

                # PCA inverse transform: pred @ components + mean → (B, 1000)
                if pca_mode and pca_components is not None:
                    pred = pred @ pca_components + pca_mean

                fmri_preds.append(pred.float().cpu().numpy())
                del ctx_batch, subj_batch, pred
                torch.cuda.empty_cache()

            fmri_pred = np.concatenate(fmri_preds, axis=0)  # (n_trs, 1000)

            # ── Denormalize (undo z-scoring) ──
            if use_global_stats and subject in fmri_stats:
                stats = fmri_stats[subject]
                fmri_pred = fmri_pred * stats["std"][None, :] + stats["mean"][None, :]

            # Safety: replace any residual NaN/Inf
            fmri_pred = np.nan_to_num(fmri_pred, nan=0.0, posinf=0.0, neginf=0.0)
            subject_dict[epi] = fmri_pred.astype(np.float32)
            del all_contexts, fmri_preds

        # Save subject result
        np.save(subj_path, subject_dict)
        subject_result_paths[subject] = subj_path
        print(f"  Saved {subject} → {subj_path} ({subj_path.stat().st_size / 1024 / 1024:.1f} MB)")

        del subject_dict
        gc.collect()
        torch.cuda.empty_cache()

    # ── 3. Assemble final submission ─────────────────────────────────────────
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
    evaluate_brainflow_direct()
