"""OOD — blind submission (out-of-distribution movies)."""

import gc
import logging
from pathlib import Path

import numpy as np
import torch

from src.evaluate.config import SolverConfig
from src.evaluate.model_runner import ModelRunner
from src.evaluate.data_helpers import (
    load_context_clip,
    load_fmri_stats,
    build_s7_windows,
)
from src.evaluate.submission import (
    load_sample_counts,
    denormalize_predictions,
    save_submission,
)
from src.utils.utils import (
    calibration_path,
    load_parcel_calibration,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
log = logging.getLogger("evaluate")


def run_ood(runner: ModelRunner, cfg: dict, context_dirs: list[Path],
            args, solver: SolverConfig):
    """Generate OOD blind submission with cross-clip batching.

    Feature path:  {ctx_dir}/ood/{movie}/{clip_key}.npy
    Sample file:   sub-XX_ood_fmri_samples.npy → {'chaplin1': 432, ...}
    """
    sw = cfg["sliding_window"]
    context_trs  = sw["context_trs"]
    n_target_trs = sw["n_target_trs"]
    stride = args.stride or sw.get("stride", 10)
    fmri_dir = cfg["_fmri_dir"]
    n_voxels = runner.n_voxels
    hrf_delay = cfg["fmri"].get("hrf_delay", 5)

    use_stats = cfg["fmri"].get("use_global_stats", False)
    stats = load_fmri_stats(fmri_dir, runner.subjects) if use_stats else {}

    run_name = Path(cfg.get("output_dir", "outputs/brainflow")).name
    out_dir = PROJECT_ROOT / "outputs" / "submissions" / run_name / "ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("OOD submission — output: %s", out_dir)
    log.info(
        "Solver: %s, steps=%d, temp=%.3f, stride=%d, mode=%s, n_seeds=%d",
        solver.method,
        solver.n_timesteps,
        solver.temperature,
        stride,
        solver.ensemble_mode,
        solver.n_seeds,
    )

    run_root = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow")
    calib = None
    if solver.ensemble_mode == "parcel_stitch" and solver.n_seeds > 1:
        calib = load_parcel_calibration(calibration_path(run_root))
        if calib is None:
            log.warning("No S6 parcel calibration found. Falling back to mean ensemble.")

    subj_paths = {}

    # --- Pre-cache: load OOD context features once into RAM ---
    _ref_subject = None
    for s in runner.subjects:
        out_path = out_dir / f"{s}_predictions.npy"
        if not (args.resume and out_path.exists()):
            _ref_subject = s
            break
    if _ref_subject is None:
        for s in runner.subjects:
            subj_paths[s] = out_dir / f"{s}_predictions.npy"
        save_submission(subj_paths, out_dir, tag="ood")
        return

    ref_samples = load_sample_counts(_ref_subject, "ood")
    context_cache = {}
    log.info("Pre-caching OOD context features for %d clips...", len(ref_samples))
    for clip_key in ref_samples:
        ctx = load_context_clip(context_dirs, "ood", None, clip_key, expected_dims=cfg.get("modality_dims"))
        context_cache[clip_key] = ctx
        if ctx is not None:
            log.info("  Cached %s: shape=%s", clip_key, ctx.shape)
    log.info("Context cache ready (%d clips in RAM)", len(context_cache))

    for subject in runner.subjects:
        out_path = out_dir / f"{subject}_predictions.npy"
        if args.resume and out_path.exists():
            log.info("[RESUME] %s already done.", subject)
            subj_paths[subject] = out_path
            continue

        sid = runner.subject_to_idx[subject]
        subject_seed_map = None
        if calib is not None:
            sub_info = calib.get("subjects", {}).get(subject)
            if sub_info is not None:
                subject_seed_map = sub_info.get("best_seed_map")
            else:
                log.warning("No parcel seed map for %s. Falling back to mean ensemble.", subject)
        log.info("\n%s %s %s", "=" * 20, subject, "=" * 20)
        samples = load_sample_counts(subject, "ood")  # {'chaplin1': 432, 'chaplin2': 405, ...}

        # --- Phase 1: Build windows from cached context ---
        clip_windows = {}
        clip_n_trs = {}
        zero_clips = {}

        for clip_key, n_trs in samples.items():
            ctx = context_cache.get(clip_key)

            if ctx is None or (hasattr(ctx, 'shape') and ctx.shape[0] == 0):
                log.warning("  No context: %s — emitting zeros", clip_key)
                zero_clips[clip_key] = n_trs
                continue

            windows = build_s7_windows(ctx, n_trs, context_trs, n_target_trs, hrf_delay, stride)
            if not windows:
                zero_clips[clip_key] = n_trs
                continue

            clip_windows[clip_key] = windows
            clip_n_trs[clip_key] = n_trs

        total_windows = sum(len(w) for w in clip_windows.values())
        log.info("  Prepared %d clips (%d zero-fill), %d total windows → %d GPU calls",
                 len(clip_windows), len(zero_clips), total_windows,
                 (total_windows + args.batch_size - 1) // args.batch_size)

        # --- Phase 2: Cross-clip batched inference ---
        subj_dict = {}

        for ck, nt in zero_clips.items():
            subj_dict[ck] = np.zeros((nt, n_voxels), dtype=np.float32)

        if clip_windows:
            batch_results = runner.run_all_clips(
                clip_windows=clip_windows,
                clip_n_trs=clip_n_trs,
                subject_id=sid,
                n_target_trs=n_target_trs,
                batch_size=args.batch_size,
                solver=solver,
                parcel_seed_map=subject_seed_map,
                return_seed_preds=False,
                desc=subject,
            )

            # --- Phase 3: Collect results ---
            for clip_key in clip_windows:
                pred_avg, valid = batch_results[clip_key]

                if use_stats:
                    pred_avg = denormalize_predictions(pred_avg, stats, subject)

                pred_avg = np.nan_to_num(pred_avg, nan=0.0)
                subj_dict[clip_key] = pred_avg.astype(np.float32)
                log.info("  %s: %d/%d TRs predicted", clip_key, int(valid.sum()), clip_n_trs[clip_key])

            del batch_results

        np.save(out_path, subj_dict)
        subj_paths[subject] = out_path
        log.info("  Saved %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
        del subj_dict
        gc.collect(); torch.cuda.empty_cache()

    del context_cache
    save_submission(subj_paths, out_dir, tag="ood")
