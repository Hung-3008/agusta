"""Evaluate BrainFlow — per-subject PCC (S6 validation) or S7 submission generation.

Supports both Seq2Seq v5 (1D-DiT, 101→50 TRs) and legacy single-TR modes,
auto-detected from the config file.

Modes:
  --eval_session s6   → evaluate on Friends S6 with ground-truth fMRI, save per-subject PCC
  --eval_session s7   → generate submission predictions for Friends S7 (no ground truth)

For Seq2Seq mode (S6):
  - Generates overlapping 50-TR prediction windows across each clip
  - Averages overlapping predictions for the same TR
  - Compares against ground-truth fMRI loaded from H5

Usage:
    # S6 evaluation (per-subject PCC)
    python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s6
    python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s6 --stride 5

    # S7 submission
    python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s7



python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s6

python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s7

python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s6 --stride 5 --n_timesteps 100


"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gc
import logging
import zipfile

import h5py
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.brainflow.brainflow import BrainFlow
from src.data.dataset import load_config, _get_fmri_filepath

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate_brainflow")


# =============================================================================
# PCC Metrics
# =============================================================================

def pearson_corr_per_dim(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-voxel Pearson correlation across time.

    Args:
        pred:   (N, D) where N=time, D=voxels
        target: (N, D)

    Returns:
        (D,) per-voxel PCC values
    """
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    tgt_c = target - target.mean(axis=0, keepdims=True)
    cov = (pred_c * tgt_c).sum(axis=0)
    std = np.sqrt((pred_c ** 2).sum(axis=0) * (tgt_c ** 2).sum(axis=0))
    return cov / (std + 1e-8)


# =============================================================================
# Data Loading
# =============================================================================

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


def load_fmri_clip_gt(fmri_dir: str, subject: str, task: str, clip_key: str,
                      excl_start: int = 5, excl_end: int = 5,
                      use_global_stats: bool = False, fmri_stats: dict = None
                      ) -> np.ndarray | None:
    """Load ground-truth fMRI data for a single clip from H5.

    Returns:
        (n_trs, n_voxels) float32 array, or None if not found.
    """
    fmri_path = _get_fmri_filepath(fmri_dir, subject, task)
    if not fmri_path.exists():
        return None

    # Normalise key — strip prefix if present
    fmri_key = clip_key
    if task == "friends" and clip_key.startswith("friends_"):
        fmri_key = clip_key[len("friends_"):]
    elif task == "movie10" and clip_key.startswith("movie10_"):
        fmri_key = clip_key[len("movie10_"):]

    with h5py.File(fmri_path, "r") as f:
        matched = [k for k in f.keys() if fmri_key in k]
        if not matched:
            return None
        raw = f[matched[0]]
        end = len(raw) - excl_end if excl_end > 0 else len(raw)
        data = raw[excl_start:end].astype(np.float32)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if use_global_stats and fmri_stats is not None and subject in fmri_stats:
        stats = fmri_stats[subject]
        data = (data - stats["mean"][None, :]) / stats["std"][None, :]

    return data


def load_context_for_clip(context_dirs: list, task: str, stim_type: str,
                          clip_name: str) -> np.ndarray | None:
    """Load and concatenate multi-modal context features for one clip.

    Returns: (n_trs, context_dim) float32 array, or None.
    """
    ctx_arrays = []
    modality_dim_cache = {}

    for ctx_dir in context_dirs:
        c_dir = ctx_dir / task / stim_type

        # Try multiple naming conventions
        ctx_path = c_dir / f"{clip_name}.npy"
        if not ctx_path.exists():
            # Strip task prefix: friends_s06e01a → s06e01a
            norm_name = clip_name
            if clip_name.startswith(f"{task}_"):
                norm_name = clip_name[len(f"{task}_"):]
            ctx_path = c_dir / f"{norm_name}.npy"
        if not ctx_path.exists():
            ctx_path = c_dir / f"{task}_{clip_name}.npy"

        if ctx_path.exists():
            arr = np.load(ctx_path).astype(np.float32)
            if arr.ndim == 3:
                arr = arr.reshape(arr.shape[0], -1)
            ctx_arrays.append(arr)
        else:
            # Zero-pad missing modality
            key = str(ctx_dir)
            if key not in modality_dim_cache:
                for f_path in ctx_dir.rglob("*.npy"):
                    modality_dim_cache[key] = np.load(f_path).shape[-1]
                    break
            mod_dim = modality_dim_cache.get(key, 512)
            ref_len = ctx_arrays[0].shape[0] if ctx_arrays else 100
            ctx_arrays.append(np.zeros((ref_len, mod_dim), dtype=np.float32))

    if not ctx_arrays:
        return None

    min_len = min(arr.shape[0] for arr in ctx_arrays)
    ctx_arrays = [arr[:min_len] for arr in ctx_arrays]

    return np.concatenate(ctx_arrays, axis=-1)


# =============================================================================
# Window Builders
# =============================================================================

def build_seq2seq_windows(ctx_data: np.ndarray, n_trs_fmri: int,
                          context_trs: int, n_target_trs: int,
                          hrf_delay: int, excl_start: int,
                          stride: int = 10) -> list[dict]:
    """Build overlapping seq2seq context windows for one clip.

    Each window:
      - target_start: first target fMRI TR index (0-based, in trimmed space)
      - context: (context_trs, D) context features

    The context window is centred on the target window's corresponding feature range.
    """
    windows = []
    extra = context_trs - n_target_trs
    extra_past = extra // 2

    n_valid = max(0, n_trs_fmri - n_target_trs + 1)

    for target_start in range(0, n_valid, stride):
        # Feature TR for the first target fMRI
        actual_start_tr = target_start + excl_start
        feat_first = actual_start_tr - hrf_delay

        # Context window: extra_past before feat_first, then context_trs total
        ctx_start = feat_first - extra_past
        ctx_end = ctx_start + context_trs
        safe_start = max(0, ctx_start)
        safe_end = min(ctx_data.shape[0], ctx_end)

        if safe_start < safe_end:
            ctx = ctx_data[safe_start:safe_end]
        else:
            ctx = np.zeros((0, ctx_data.shape[-1]), dtype=np.float32)

        # Pad to exact context_trs length
        if ctx.shape[0] < context_trs:
            pad_before = max(0, -ctx_start)
            pad_after = max(0, context_trs - ctx.shape[0] - pad_before)
            ctx = np.pad(ctx, ((pad_before, pad_after), (0, 0)), mode="constant")
        ctx = ctx[:context_trs]

        windows.append({
            "target_start": target_start,
            "context": ctx,
        })

    return windows


def build_legacy_windows(ctx_data: np.ndarray, n_trs_fmri: int,
                         feature_past_trs: int, feature_future_trs: int,
                         hrf_delay: int, excl_start: int,
                         stride: int = 1) -> list[dict]:
    """Build per-TR context windows (legacy single-TR mode)."""
    feat_seq_len = feature_past_trs + 1 + feature_future_trs
    windows = []

    for target_tr in range(0, n_trs_fmri, stride):
        actual_start_tr = target_tr + excl_start
        feat_current = actual_start_tr - hrf_delay

        feat_start = feat_current - feature_past_trs
        feat_end = feat_current + 1 + feature_future_trs
        safe_start = max(0, feat_start)
        safe_end = min(ctx_data.shape[0], feat_end)

        if safe_start < safe_end:
            ctx = ctx_data[safe_start:safe_end]
        else:
            ctx = np.zeros((0, ctx_data.shape[-1]), dtype=np.float32)

        if ctx.shape[0] < feat_seq_len:
            pad_before = max(0, -feat_start)
            pad_after = max(0, feat_seq_len - ctx.shape[0] - pad_before)
            ctx = np.pad(ctx, ((pad_before, pad_after), (0, 0)), mode="constant")

        windows.append({
            "target_start": target_tr,
            "context": ctx,
        })

    return windows


def build_s7_windows_seq2seq(ctx_data: np.ndarray, n_trs: int,
                             context_trs: int, n_target_trs: int,
                             stride: int = 10) -> list[dict]:
    """Build seq2seq windows for S7 test data.

    S7 test features are pre-aligned 1:1 with target TRs (no hrf_delay/excl_start).
    """
    windows = []
    extra = context_trs - n_target_trs
    extra_past = extra // 2

    n_valid = max(0, n_trs - n_target_trs + 1)

    for target_start in range(0, n_valid, stride):
        feat_first = target_start
        ctx_start = feat_first - extra_past
        ctx_end = ctx_start + context_trs
        safe_start = max(0, ctx_start)
        safe_end = min(ctx_data.shape[0], ctx_end)

        if safe_start < safe_end:
            ctx = ctx_data[safe_start:safe_end]
        else:
            ctx = np.zeros((0, ctx_data.shape[-1]), dtype=np.float32)

        if ctx.shape[0] < context_trs:
            pad_before = max(0, -ctx_start)
            pad_after = max(0, context_trs - ctx.shape[0] - pad_before)
            ctx = np.pad(ctx, ((pad_before, pad_after), (0, 0)), mode="constant")
        ctx = ctx[:context_trs]

        windows.append({
            "target_start": target_start,
            "context": ctx,
        })

    return windows


def build_s7_windows_legacy(ctx_data: np.ndarray, n_trs: int,
                            feature_past_trs: int, feature_future_trs: int) -> list[dict]:
    """Build per-TR windows for S7 test data (pre-aligned features)."""
    feat_seq_len = feature_past_trs + 1 + feature_future_trs
    windows = []

    for target_tr in range(n_trs):
        feat_start = target_tr - feature_past_trs
        feat_end = target_tr + 1 + feature_future_trs
        safe_start = max(0, feat_start)
        safe_end = min(ctx_data.shape[0], feat_end)

        if safe_start < safe_end:
            ctx = ctx_data[safe_start:safe_end]
        else:
            ctx = np.zeros((0, ctx_data.shape[-1]), dtype=np.float32)

        if ctx.shape[0] < feat_seq_len:
            pad_before = max(0, -feat_start)
            pad_after = max(0, feat_seq_len - ctx.shape[0] - pad_before)
            ctx = np.pad(ctx, ((pad_before, pad_after), (0, 0)), mode="constant")

        windows.append({
            "target_start": target_tr,
            "context": ctx,
        })

    return windows


# =============================================================================
# Model Builder (shared between S6 and S7)
# =============================================================================

def build_model(cfg, device):
    """Build and load BrainFlow model from config + checkpoint."""
    bf_cfg = cfg["brainflow"]
    vn_params = dict(bf_cfg.get("velocity_net", {}))
    modality_dims = cfg.get("modality_dims", None)
    if modality_dims:
        vn_params["modality_dims"] = modality_dims

    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])
    subjects = cfg["subjects"]

    model = BrainFlow(
        output_dim=output_dim,
        velocity_net_params=vn_params,
        n_subjects=len(subjects),
        reg_weight=bf_cfg.get("reg_weight", 1.0),
        cont_weight=bf_cfg.get("cont_weight", 0.1),
        cont_dim=bf_cfg.get("cont_dim", 256),
        tensor_fm_params=bf_cfg.get("tensor_fm", None),
        indi_flow_matching=bf_cfg.get("indi_flow_matching", False),
        indi_train_time_sqrt=bf_cfg.get("indi_train_time_sqrt", False),
        indi_min_denom=bf_cfg.get("indi_min_denom", 1e-3),
    ).to(device)

    return model, output_dim


def load_checkpoint(model, out_dir, device, checkpoint_override=None):
    """Load model checkpoint (best.pt or last.pt with EMA)."""
    ckpt_path = Path(checkpoint_override) if checkpoint_override else out_dir / "best.pt"

    if not ckpt_path.exists():
        last_path = out_dir / "last.pt"
        if last_path.exists():
            logger.info("best.pt not found, loading EMA from last.pt")
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            if "ema" in ckpt:
                from src.train_brainflow import EMAModel
                ema = EMAModel(model, decay=0.999, store_on_cpu=False)
                ema.load_state_dict(ckpt["ema"])
                ema.apply_shadow(model)
                logger.info("Applied EMA shadow weights from last.pt")
            del ckpt
        else:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path} or {last_path}")
    else:
        logger.info("Loading checkpoint from %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        del checkpoint

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s params", f"{n_params:,}")
    return model


# =============================================================================
# Batch Inference Helper
# =============================================================================

def run_batch_inference(model, windows, subject_idx, output_dim, n_trs,
                        is_seq2seq, n_target_trs, batch_size, device,
                        synth_kwargs, pca_components=None, pca_mean=None):
    """Run ODE inference on windows and return overlap-averaged predictions.

    Returns:
        (n_trs, output_dim) float32 array of averaged predictions
    """
    pred_sum = np.zeros((n_trs, output_dim), dtype=np.float64)
    pred_count = np.zeros(n_trs, dtype=np.float64)

    all_contexts = np.stack([w["context"] for w in windows])
    target_starts = [w["target_start"] for w in windows]

    for bi in range(0, len(windows), batch_size):
        be = min(bi + batch_size, len(windows))
        B = be - bi

        ctx_batch = torch.from_numpy(all_contexts[bi:be]).to(device)
        subj_batch = torch.full((B,), subject_idx, dtype=torch.long, device=device)

        kw = dict(synth_kwargs)
        kw["subject_ids"] = subj_batch

        pred = model.synthesise(ctx_batch, **kw)

        # PCA inverse transform if needed
        if pca_components is not None:
            pred = pred @ pca_components + pca_mean

        pred_np = pred.float().cpu().numpy()

        for j in range(B):
            ts = target_starts[bi + j]
            if is_seq2seq:
                for off in range(n_target_trs):
                    tr_idx = ts + off
                    if tr_idx >= n_trs:
                        break
                    pred_sum[tr_idx] += pred_np[j, off]
                    pred_count[tr_idx] += 1
            else:
                if ts < n_trs:
                    pred_sum[ts] += pred_np[j]
                    pred_count[ts] += 1

        del ctx_batch, subj_batch, pred
        torch.cuda.empty_cache()

    valid_mask = pred_count > 0
    pred_avg = np.zeros((n_trs, output_dim), dtype=np.float32)
    pred_avg[valid_mask] = (pred_sum[valid_mask] / pred_count[valid_mask, None]).astype(np.float32)

    return pred_avg, valid_mask


# =============================================================================
# S7 Submission Mode
# =============================================================================

@torch.inference_mode()
def evaluate_s7(args, cfg, model, output_dim, context_dirs, device):
    """Generate fMRI predictions for Friends S7 submission."""
    subjects = cfg["subjects"]
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    fmri_dir = Path(cfg["_fmri_dir"])

    sw = cfg["sliding_window"]
    is_seq2seq = "n_target_trs" in sw

    if is_seq2seq:
        context_trs = sw["context_trs"]
        n_target_trs = sw["n_target_trs"]
        stride = args.stride if args.stride is not None else sw.get("stride", 10)
    else:
        feature_past_trs = sw.get("feature_past_trs",
                                   sw.get("feature_context_trs", 10))
        feature_future_trs = sw.get("feature_future_trs", 0)
        n_target_trs = 1

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
                logger.info("Loaded global stats for %s", subj)

    # PCA model (optional)
    pca_components, pca_mean = None, None
    if "pca_model_path" in cfg:
        pca_path = PROJECT_ROOT / cfg["pca_model_path"]
        pca = joblib.load(pca_path)
        pca_components = torch.from_numpy(pca.components_.astype(np.float32)).to(device)
        pca_mean = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)
        logger.info("PCA mode: %d components, explained_var=%.4f",
                     pca.n_components_, pca.explained_variance_ratio_.sum())
        del pca

    # Output directory
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_direct")
    output_dir = PROJECT_ROOT / "results" / "brainflow_direct_submission"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Solver config
    solver_cfg = cfg.get("solver_args", {})
    n_timesteps = args.n_timesteps or solver_cfg.get("time_points", 50)
    solver_method = args.solver_method or solver_cfg.get("method", "midpoint")
    temperature = args.temperature if args.temperature is not None else solver_cfg.get("temperature", 0.0)
    cfg_scale = solver_cfg.get("cfg_scale", 0.0)
    time_grid_warp = solver_cfg.get("time_grid_warp", None)

    synth_kwargs = dict(
        n_timesteps=n_timesteps,
        solver_method=solver_method,
        temperature=temperature,
    )
    if cfg_scale > 0:
        synth_kwargs["cfg_scale"] = cfg_scale
    if time_grid_warp:
        synth_kwargs["time_grid_warp"] = time_grid_warp

    logger.info("S7 submission mode — n_timesteps=%d, solver=%s, temperature=%.3f",
                n_timesteps, solver_method, temperature)

    subject_result_paths = {}

    for subject in subjects:
        subj_path = output_dir / f"{subject}_predictions.npy"

        if args.resume and subj_path.exists():
            logger.info("[RESUME] Skipping %s — found %s", subject, subj_path)
            subject_result_paths[subject] = subj_path
            continue

        logger.info("\n%s Processing %s %s", "=" * 20, subject, "=" * 20)

        subject_idx = subject_to_idx[subject]
        subject_dict = {}
        target_samples = load_fmri_samples(PROJECT_ROOT, subject)

        for epi, n_trs in tqdm(target_samples.items(), desc=subject):
            clip_name = f"friends_{epi}"

            ctx_data = load_context_for_clip(context_dirs, "friends", "s7", clip_name)
            if ctx_data is None or ctx_data.shape[0] == 0:
                logger.warning("  No context for %s. Emitting zeros.", clip_name)
                subject_dict[epi] = np.zeros((n_trs, 1000), dtype=np.float32)
                continue

            # Build windows (S7: pre-aligned features, hrf=0, excl=0)
            if is_seq2seq:
                windows = build_s7_windows_seq2seq(
                    ctx_data, n_trs, context_trs, n_target_trs, stride,
                )
            else:
                windows = build_s7_windows_legacy(
                    ctx_data, n_trs, feature_past_trs, feature_future_trs,
                )

            if not windows:
                subject_dict[epi] = np.zeros((n_trs, 1000), dtype=np.float32)
                continue

            # The output_dim for inference may be PCA dim; final output always 1000
            infer_dim = output_dim if pca_components is None else output_dim
            final_dim = 1000

            pred_avg, valid_mask = run_batch_inference(
                model, windows, subject_idx, infer_dim, n_trs,
                is_seq2seq, n_target_trs, args.batch_size, device,
                synth_kwargs, pca_components, pca_mean,
            )

            # Denormalize (undo z-scoring)
            if use_global_stats and subject in fmri_stats:
                stats = fmri_stats[subject]
                pred_avg = pred_avg * stats["std"][None, :] + stats["mean"][None, :]

            pred_avg = np.nan_to_num(pred_avg, nan=0.0, posinf=0.0, neginf=0.0)
            subject_dict[epi] = pred_avg.astype(np.float32)
            del ctx_data, windows

        np.save(subj_path, subject_dict)
        subject_result_paths[subject] = subj_path
        logger.info("  Saved %s → %s (%.1f MB)",
                     subject, subj_path, subj_path.stat().st_size / 1024 / 1024)

        del subject_dict
        gc.collect()
        torch.cuda.empty_cache()

    # Assemble final submission
    logger.info("\nAssembling submission dict from per-subject files...")
    submission_dict = {}
    for subject, path in subject_result_paths.items():
        submission_dict[subject] = np.load(path, allow_pickle=True).item()

    submission_path = output_dir / "submission.npy"
    logger.info("Saving submission.npy to %s...", submission_path)
    np.save(submission_path, submission_dict)

    del submission_dict
    gc.collect()

    zip_path = submission_path.with_suffix(".zip")
    logger.info("Zipping to %s...", zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(submission_path, arcname="submission.npy")

    logger.info("Done! Zip size: %.1f MB", zip_path.stat().st_size / 1024 / 1024)

    # Clean up per-subject temp files
    for path in subject_result_paths.values():
        path.unlink(missing_ok=True)
    logger.info("Cleaned up per-subject temp files.")


# =============================================================================
# S6 (or other val sessions) Evaluation Mode
# =============================================================================

@torch.inference_mode()
def evaluate_with_gt(args, cfg, model, output_dim, context_dirs, device):
    """Evaluate on a session with ground-truth fMRI (e.g., S6).

    Saves per-subject results with PCC metrics.
    """
    subjects = cfg["subjects"]
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    fmri_dir = cfg["_fmri_dir"]

    sw = cfg["sliding_window"]
    is_seq2seq = "n_target_trs" in sw
    hrf_delay = cfg["fmri"].get("hrf_delay", 5)
    excl_start = cfg["fmri"].get("excluded_samples_start", 0)
    excl_end = cfg["fmri"].get("excluded_samples_end", 0)

    if is_seq2seq:
        context_trs = sw["context_trs"]
        n_target_trs = sw["n_target_trs"]
        stride = args.stride if args.stride is not None else sw.get("stride", 10)
        logger.info("Seq2Seq mode: context_trs=%d, n_target_trs=%d, stride=%d",
                     context_trs, n_target_trs, stride)
    else:
        feature_past_trs = sw.get("feature_past_trs",
                                   sw.get("feature_context_trs", 10))
        feature_future_trs = sw.get("feature_future_trs", 0)
        feat_seq_len = feature_past_trs + 1 + feature_future_trs
        n_target_trs = 1
        stride = args.stride if args.stride is not None else sw.get("stride", 1)
        logger.info("Legacy single-TR mode: past=%d, future=%d, seq_len=%d, stride=%d",
                     feature_past_trs, feature_future_trs, feat_seq_len, stride)

    # Global stats
    use_global_stats = cfg["fmri"].get("use_global_stats", False)
    fmri_stats = {}
    if use_global_stats:
        for subj in subjects:
            stats_dir = Path(fmri_dir) / subj / "stats"
            mean_path = stats_dir / "global_mean.npy"
            std_path = stats_dir / "global_std.npy"
            if mean_path.exists() and std_path.exists():
                fmri_stats[subj] = {
                    "mean": np.load(mean_path).astype(np.float32),
                    "std": np.load(std_path).astype(np.float32),
                }
                logger.info("Loaded global stats for %s", subj)

    # Discover clips
    eval_session = args.eval_session
    ref_ctx_dir = context_dirs[0] / "friends" / eval_session
    if not ref_ctx_dir.exists():
        logger.error("Feature dir for friends/%s not found: %s", eval_session, ref_ctx_dir)
        return
    clip_stems = sorted(p.stem for p in ref_ctx_dir.glob("*.npy"))
    logger.info("Found %d clips in friends/%s", len(clip_stems), eval_session)

    # Output directory
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_direct")
    eval_output_dir = out_dir / f"eval_{eval_session}"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Solver config
    solver_cfg = cfg.get("solver_args", {})
    n_timesteps = args.n_timesteps or solver_cfg.get("time_points", 50)
    solver_method = args.solver_method or solver_cfg.get("method", "midpoint")
    temperature = args.temperature if args.temperature is not None else solver_cfg.get("temperature", 0.0)
    cfg_scale = solver_cfg.get("cfg_scale", 0.0)
    time_grid_warp = solver_cfg.get("time_grid_warp", None)

    synth_kwargs = dict(
        n_timesteps=n_timesteps,
        solver_method=solver_method,
        temperature=temperature,
    )
    if cfg_scale > 0:
        synth_kwargs["cfg_scale"] = cfg_scale
    if time_grid_warp:
        synth_kwargs["time_grid_warp"] = time_grid_warp

    logger.info("Solver: method=%s, n_timesteps=%d, temperature=%.3f, cfg_scale=%.2f, warp=%s",
                solver_method, n_timesteps, temperature, cfg_scale, time_grid_warp)

    # ── Per-subject evaluation ────────────────────────────────────────────────
    all_subject_results = {}

    for subject in subjects:
        subject_idx = subject_to_idx[subject]
        logger.info("\n%s Processing %s %s", "=" * 20, subject, "=" * 20)

        clip_results = {}
        all_pred_list = []
        all_tgt_list = []

        for clip_stem in tqdm(clip_stems, desc=subject):
            norm_name = clip_stem
            if clip_stem.startswith("friends_"):
                norm_name = clip_stem[len("friends_"):]

            # Load ground-truth fMRI
            fmri_gt = load_fmri_clip_gt(
                fmri_dir, subject, "friends", norm_name,
                excl_start=excl_start, excl_end=excl_end,
                use_global_stats=use_global_stats, fmri_stats=fmri_stats,
            )
            if fmri_gt is None:
                logger.warning("  No fMRI for %s/%s, skipping", subject, clip_stem)
                continue

            n_trs_fmri = fmri_gt.shape[0]

            # Load context
            ctx_data = load_context_for_clip(
                context_dirs, "friends", eval_session, clip_stem,
            )
            if ctx_data is None or ctx_data.shape[0] == 0:
                logger.warning("  No context for %s, skipping", clip_stem)
                continue

            # Build windows
            if is_seq2seq:
                windows = build_seq2seq_windows(
                    ctx_data, n_trs_fmri,
                    context_trs, n_target_trs,
                    hrf_delay, excl_start, stride,
                )
            else:
                windows = build_legacy_windows(
                    ctx_data, n_trs_fmri,
                    feature_past_trs, feature_future_trs,
                    hrf_delay, excl_start, stride,
                )

            if not windows:
                logger.warning("  No valid windows for %s (n_trs=%d)", clip_stem, n_trs_fmri)
                continue

            pred_avg, valid_mask = run_batch_inference(
                model, windows, subject_idx, output_dim, n_trs_fmri,
                is_seq2seq, n_target_trs, args.batch_size, device,
                synth_kwargs,
            )

            if valid_mask.sum() < 2:
                logger.warning("  Too few valid TRs for %s (%d), skipping",
                               clip_stem, valid_mask.sum())
                continue

            # Per-clip PCC
            clip_pcc = pearson_corr_per_dim(
                pred_avg[valid_mask],
                fmri_gt[valid_mask],
            )
            clip_mean_pcc = float(np.median(clip_pcc))
            clip_results[clip_stem] = {
                "mean_pcc": clip_mean_pcc,
                "n_valid_trs": int(valid_mask.sum()),
                "n_total_trs": n_trs_fmri,
            }
            logger.info("  %s: PCC=%.4f (%d/%d TRs)",
                         clip_stem, clip_mean_pcc, valid_mask.sum(), n_trs_fmri)

            all_pred_list.append(pred_avg[valid_mask])
            all_tgt_list.append(fmri_gt[valid_mask])

            del ctx_data, fmri_gt, pred_avg

        # Global subject PCC
        if all_pred_list:
            all_pred = np.concatenate(all_pred_list, axis=0)
            all_tgt = np.concatenate(all_tgt_list, axis=0)
            global_pcc = pearson_corr_per_dim(all_pred, all_tgt)
            subject_median_pcc = float(np.median(global_pcc))
            subject_mean_pcc = float(np.mean(global_pcc))
        else:
            subject_median_pcc = 0.0
            subject_mean_pcc = 0.0
            global_pcc = np.array([])
            all_pred = np.array([])

        logger.info("\n  >>> %s OVERALL: median_PCC=%.4f, mean_PCC=%.4f (%d clips, %d TRs)",
                     subject, subject_median_pcc, subject_mean_pcc,
                     len(clip_results), all_pred.shape[0] if len(all_pred_list) else 0)

        subject_result = {
            "subject": subject,
            "eval_session": eval_session,
            "median_pcc": subject_median_pcc,
            "mean_pcc": subject_mean_pcc,
            "per_voxel_pcc": global_pcc,
            "clip_results": clip_results,
            "config": {
                "n_timesteps": n_timesteps,
                "solver_method": solver_method,
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "stride": stride,
                "is_seq2seq": is_seq2seq,
            },
        }

        result_path = eval_output_dir / f"{subject}_eval_{eval_session}.npy"
        np.save(result_path, subject_result)
        logger.info("  Saved → %s", result_path)

        all_subject_results[subject] = subject_result

        del all_pred_list, all_tgt_list
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    logger.info("\n%s EVALUATION SUMMARY %s", "=" * 25, "=" * 25)
    logger.info("Session: friends/%s", eval_session)
    logger.info("%-8s  %10s  %10s  %6s", "Subject", "Median PCC", "Mean PCC", "Clips")
    logger.info("-" * 40)

    pccs = []
    for subj, res in all_subject_results.items():
        logger.info("%-8s  %10.4f  %10.4f  %6d",
                     subj, res["median_pcc"], res["mean_pcc"], len(res["clip_results"]))
        pccs.append(res["median_pcc"])

    if pccs:
        logger.info("-" * 40)
        logger.info("%-8s  %10.4f  %10.4f", "AVG",
                     np.mean(pccs), np.mean([r["mean_pcc"] for r in all_subject_results.values()]))

    summary_path = eval_output_dir / f"summary_{eval_session}.npy"
    np.save(summary_path, all_subject_results)
    logger.info("\nSummary saved → %s", summary_path)
    logger.info("Individual results saved in %s/", eval_output_dir)


# =============================================================================
# Main Entry Point
# =============================================================================

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BrainFlow — S6 PCC evaluation or S7 submission"
    )
    parser.add_argument("--config", type=str, default="src/configs/brainflow.yaml")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_timesteps", type=int, default=None,
                        help="ODE solver steps (overrides config).")
    parser.add_argument("--solver_method", type=str, default=None,
                        help="ODE solver method (overrides config).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Starting noise std (overrides config). 0=deterministic.")
    parser.add_argument("--eval_session", type=str, default="s6",
                        help="Session: s6 = PCC eval with GT, s7 = submission generation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (default: {output_dir}/best.pt)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override stride for window stepping (default from config)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="[S7] Skip subjects whose per-subject npy already exists.")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)

    # Resolve paths
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_data_root"] = str(PROJECT_ROOT / cfg["data_root"])
    cfg["_fmri_dir"] = str(PROJECT_ROOT / cfg["data_root"] / cfg["fmri"]["dir"])

    # Context dirs
    if "context_latent_dirs" in cfg:
        context_dirs = [PROJECT_ROOT / d for d in cfg["context_latent_dirs"]]
    else:
        context_dirs = [PROJECT_ROOT / cfg["context_latent_dir"]]

    # Build and load model
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow_direct")
    model, output_dim = build_model(cfg, device)
    model = load_checkpoint(model, out_dir, device, args.checkpoint)

    # Dispatch
    if args.eval_session == "s7":
        evaluate_s7(args, cfg, model, output_dim, context_dirs, device)
    else:
        evaluate_with_gt(args, cfg, model, output_dim, context_dirs, device)


if __name__ == "__main__":
    main()
