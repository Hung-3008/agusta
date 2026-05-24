"""S7 — Friends blind submission (sequential and parallel modes)."""

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


# =============================================================================
# S7 — Sequential (one subject at a time)
# =============================================================================

def run_s7(runner: ModelRunner, cfg: dict, context_dirs: list[Path],
           args, solver: SolverConfig):
    """Generate Friends S7 blind submission with cross-clip batching."""
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
    out_dir = PROJECT_ROOT / "outputs" / "submissions" / run_name / "s7"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("S7 submission — output: %s", out_dir)
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

    # --- Pre-cache: load context features once into RAM (shared across subjects) ---
    # Discover episodes from first non-resumed subject's sample counts
    _ref_subject = None
    for s in runner.subjects:
        out_path = out_dir / f"{s}_predictions.npy"
        if not (args.resume and out_path.exists()):
            _ref_subject = s
            break
    if _ref_subject is None:
        # All subjects resumed
        for s in runner.subjects:
            subj_paths[s] = out_dir / f"{s}_predictions.npy"
        save_submission(subj_paths, out_dir, tag="s7")
        return

    ref_samples = load_sample_counts(_ref_subject, "friends-s7")
    context_cache = {}  # {epi: np.ndarray (T, D_total) or None}
    log.info("Pre-caching context features for %d episodes...", len(ref_samples))
    for epi in ref_samples:
        clip = f"friends_{epi}"
        ctx = load_context_clip(context_dirs, "friends", "s7", clip, expected_dims=cfg.get("modality_dims"))
        context_cache[epi] = ctx
        if ctx is not None:
            log.info("  Cached %s: shape=%s", epi, ctx.shape)
    log.info("Context cache ready (%d episodes in RAM)", len(context_cache))

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
        samples = load_sample_counts(subject, "friends-s7")

        # --- Phase 1: Build windows from cached context ---
        clip_windows = {}       # {epi: [windows]}
        clip_n_trs = {}         # {epi: n_trs}
        zero_episodes = {}      # {epi: n_trs} for episodes with no context

        for epi, n_trs in samples.items():
            ctx = context_cache.get(epi)

            if ctx is None or ctx.shape[0] == 0:
                clip = f"friends_{epi}"
                log.warning("  No context: %s — emitting zeros", clip)
                zero_episodes[epi] = n_trs
                continue

            windows = build_s7_windows(ctx, n_trs, context_trs, n_target_trs, hrf_delay, stride)
            if not windows:
                zero_episodes[epi] = n_trs
                continue

            clip_windows[epi] = windows
            clip_n_trs[epi] = n_trs

        total_windows = sum(len(w) for w in clip_windows.values())
        log.info("  Prepared %d episodes (%d zero-fill), %d total windows → %d GPU calls",
                 len(clip_windows), len(zero_episodes), total_windows,
                 (total_windows + args.batch_size - 1) // args.batch_size)

        # --- Phase 2: Cross-clip batched inference ---
        subj_dict = {}

        # Fill zero episodes
        for epi, n_trs in zero_episodes.items():
            subj_dict[epi] = np.zeros((n_trs, n_voxels), dtype=np.float32)

        if clip_windows:
            batch_results = runner.run_all_clips(
                clip_contexts=context_cache,
                clip_windows=clip_windows,
                clip_n_trs=clip_n_trs,
                subject_id=sid,
                n_target_trs=n_target_trs,
                batch_size=args.batch_size,
                solver=solver,
                parcel_seed_map=subject_seed_map,
                return_seed_preds=False,
                desc=subject,
                hrf_delay=hrf_delay,
                excl_start=0,
            )

            # --- Phase 3: Collect results ---
            for epi in clip_windows:
                pred_avg, valid = batch_results[epi]

                # Denormalize if trained with z-score
                if use_stats:
                    pred_avg = denormalize_predictions(pred_avg, stats, subject)

                pred_avg = np.nan_to_num(pred_avg, nan=0.0)
                subj_dict[epi] = pred_avg.astype(np.float32)
                log.info("  %s: %d/%d TRs predicted", epi, int(valid.sum()), clip_n_trs[epi])

            del batch_results

        np.save(out_path, subj_dict)
        subj_paths[subject] = out_path
        log.info("  Saved %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
        del subj_dict
        gc.collect(); torch.cuda.empty_cache()

    del context_cache
    save_submission(subj_paths, out_dir, tag="s7")


# =============================================================================
# S7 Parallel — All subjects decoded simultaneously
# =============================================================================

def run_s7_parallel(runner: ModelRunner, cfg: dict, context_dirs: list[Path],
                    args, solver: SolverConfig):
    """Generate Friends S7 blind submission with multi-subject parallel inference.

    Encodes context ONCE (subject-independent), then decodes for all subjects
    simultaneously using mixed subject_ids batches. ~2.5-3× faster than sequential.
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
    out_dir = PROJECT_ROOT / "outputs" / "submissions" / run_name / "s7"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("S7 PARALLEL submission — output: %s", out_dir)
    log.info(
        "Solver: %s, steps=%d, temp=%.3f, stride=%d, n_subjects=%d",
        solver.method, solver.n_timesteps, solver.temperature, stride,
        len(runner.subjects),
    )

    # Check which subjects still need processing
    subj_paths = {}
    subjects_to_run = []
    for s in runner.subjects:
        out_path = out_dir / f"{s}_predictions.npy"
        if args.resume and out_path.exists():
            log.info("[RESUME] %s already done.", s)
            subj_paths[s] = out_path
        else:
            subjects_to_run.append(s)

    if not subjects_to_run:
        save_submission(subj_paths, out_dir, tag="s7")
        return

    # Load S7 sample counts for all subjects to run
    all_samples = {s: load_sample_counts(s, "friends-s7") for s in subjects_to_run}

    # Get union of all episodes across subjects
    all_episodes = set()
    for samples in all_samples.values():
        all_episodes.update(samples.keys())
    all_episodes = sorted(all_episodes)

    # Pre-cache context features (shared across subjects)
    log.info("Pre-caching context features for %d episodes...", len(all_episodes))
    context_cache = {}
    for epi in all_episodes:
        clip = f"friends_{epi}"
        ctx = load_context_clip(context_dirs, "friends", "s7", clip,
                                expected_dims=cfg.get("modality_dims"))
        context_cache[epi] = ctx
        if ctx is not None:
            log.info("  Cached %s: shape=%s", epi, ctx.shape)
    log.info("Context cache ready (%d episodes in RAM)", len(context_cache))

    # Build windows (shared across subjects — same context, same starts)
    clip_windows = {}
    clip_n_trs_per_subject = {s: {} for s in subjects_to_run}
    zero_episodes_per_subject = {s: {} for s in subjects_to_run}

    # Use max n_trs across subjects for window building (windows are shared)
    max_n_trs = {}
    for epi in all_episodes:
        max_n_trs[epi] = max(
            all_samples[s].get(epi, 0) for s in subjects_to_run
        )

    for epi in all_episodes:
        n_trs_max = max_n_trs[epi]
        ctx = context_cache.get(epi)

        # Track per-subject n_trs
        for s in subjects_to_run:
            n_trs_s = all_samples[s].get(epi, 0)
            if n_trs_s > 0:
                clip_n_trs_per_subject[s][epi] = n_trs_s

        if ctx is None or ctx.shape[0] == 0:
            for s in subjects_to_run:
                n_trs_s = all_samples[s].get(epi, 0)
                if n_trs_s > 0:
                    zero_episodes_per_subject[s][epi] = n_trs_s
            continue

        # Build windows using max n_trs (superset)
        windows = build_s7_windows(ctx, n_trs_max, context_trs, n_target_trs,
                                   hrf_delay, stride)
        if not windows:
            for s in subjects_to_run:
                n_trs_s = all_samples[s].get(epi, 0)
                if n_trs_s > 0:
                    zero_episodes_per_subject[s][epi] = n_trs_s
            continue

        clip_windows[epi] = windows

    total_windows = sum(len(w) for w in clip_windows.values())
    n_subjects = len(subjects_to_run)
    subject_ids_list = [runner.subject_to_idx[s] for s in subjects_to_run]

    log.info("Prepared %d episodes, %d total windows × %d subjects",
             len(clip_windows), total_windows, n_subjects)

    # Run multi-subject parallel inference
    if clip_windows:
        batch_results = runner.run_all_clips_multisubject(
            clip_contexts=context_cache,
            clip_windows=clip_windows,
            clip_n_trs_per_subject=clip_n_trs_per_subject,
            subject_ids_list=subject_ids_list,
            n_target_trs=n_target_trs,
            batch_size=args.batch_size,
            solver=solver,
            desc="s7-parallel",
            hrf_delay=hrf_delay,
            excl_start=0,
        )
    else:
        batch_results = {s: {} for s in subjects_to_run}

    del context_cache

    # Collect and save per-subject results
    for subject in subjects_to_run:
        out_path = out_dir / f"{subject}_predictions.npy"
        subj_dict = {}

        # Fill zero episodes
        for epi, n_trs in zero_episodes_per_subject[subject].items():
            subj_dict[epi] = np.zeros((n_trs, n_voxels), dtype=np.float32)

        # Fill predicted episodes
        for epi in clip_windows:
            if epi not in batch_results[subject]:
                continue
            pred_avg, valid = batch_results[subject][epi]

            # Denormalize if trained with z-score
            if use_stats:
                pred_avg = denormalize_predictions(pred_avg, stats, subject)

            pred_avg = np.nan_to_num(pred_avg, nan=0.0)
            subj_dict[epi] = pred_avg.astype(np.float32)
            log.info("  %s/%s: %d/%d TRs predicted", subject, epi,
                     int(valid.sum()), clip_n_trs_per_subject[subject].get(epi, 0))

        np.save(out_path, subj_dict)
        subj_paths[subject] = out_path
        log.info("  Saved %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
        del subj_dict

    del batch_results
    gc.collect(); torch.cuda.empty_cache()

    save_submission(subj_paths, out_dir, tag="s7")
