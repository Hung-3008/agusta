"""S6 — validation with ground-truth PCC + extended metrics."""

import gc
import logging
from pathlib import Path

import numpy as np
import torch

from src.evaluate.config import SolverConfig
from src.evaluate.model_runner import ModelRunner
from src.evaluate.data_helpers import (
    load_context_clip,
    load_fmri_clip,
    load_fmri_stats,
    build_seq2seq_windows,
    pcc,
)
from src.utils.utils import (
    best_seed_map_from_s6,
    calibration_path,
    save_parcel_calibration,
)
from src.utils.metrics import (
    compute_noise_ceiling,
    compute_extended_metrics,
    compute_cross_subject_summary,
    per_network_pcc,
    NETWORK_NAMES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
log = logging.getLogger("evaluate")


def run_s6(runner: ModelRunner, cfg: dict, context_dirs: list[Path],
           args, solver: SolverConfig):
    """Per-subject S6 PCC evaluation with cross-clip batching.

    Instead of processing clips one-by-one (~40 windows each × 49 clips =
    49 tiny GPU calls), this pre-builds ALL windows and processes them in
    large batches (e.g. 4 batches of 512 = ~2000 windows total).
    """
    sw = cfg["sliding_window"]
    context_trs  = sw["context_trs"]
    n_target_trs = sw["n_target_trs"]
    stride    = args.stride or sw.get("stride", 10)
    hrf_delay = cfg["fmri"].get("hrf_delay", 2)
    excl_s    = cfg["fmri"].get("excluded_samples_start", 5)
    excl_e    = cfg["fmri"].get("excluded_samples_end", 5)
    fmri_dir  = cfg["_fmri_dir"]

    use_stats = cfg["fmri"].get("use_global_stats", False)
    stats = load_fmri_stats(fmri_dir, runner.subjects) if use_stats else {}

    # Discover S6 clips from first context dir
    clip_dir = context_dirs[0] / "friends" / "s6"
    if not clip_dir.exists():
        log.error("S6 feature dir not found: %s", clip_dir)
        return
    clips = sorted(p.stem for p in clip_dir.glob("*.npy"))
    log.info("S6: %d clips | context_trs=%d, n_target=%d, stride=%d, hrf=%d",
             len(clips), context_trs, n_target_trs, stride, hrf_delay)
    log.info(
        "Solver: %s, steps=%d, temp=%.3f, cfg=%.2f, t_max=%.3f, final_jump=%s",
        solver.method,
        solver.n_timesteps,
        solver.temperature,
        solver.cfg_scale,
        solver.time_grid_max,
        solver.final_jump,
    )
    log.info(
        "Strategies: mode=%s, n_seeds=%d, pruned=%s, prune_k=%d",
        solver.ensemble_mode,
        solver.n_seeds,
        solver.use_pruned_sampling,
        solver.prune_k,
    )

    out_dir = (Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow")) / "eval_s6"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Results → %s", out_dir)

    all_results = {}
    # Collect concatenated fMRI GT per subject for noise ceiling computation
    subject_fmri_concat: dict[str, np.ndarray] = {}
    calibration = {
        "metadata": {
            "session": "s6",
            "n_seeds": int(solver.n_seeds),
            "ensemble_mode": solver.ensemble_mode,
            "base_seed": int(solver.base_seed),
            "use_pruned_sampling": bool(solver.use_pruned_sampling),
            "prune_k": int(solver.prune_k),
            "stride": int(stride),
        },
        "subjects": {},
    }
    need_calibration = solver.ensemble_mode == "parcel_stitch" and solver.n_seeds > 1

    for subject in runner.subjects:
        sid = runner.subject_to_idx[subject]
        log.info("\n%s %s %s", "=" * 20, subject, "=" * 20)

        # --- Phase 1: Pre-load all clips & build all windows ---
        clip_fmri_gt = {}       # {clip: fmri_gt array}
        clip_windows = {}       # {clip: [windows]}
        clip_contexts = {}      # {clip: ctx array}
        clip_n_trs = {}         # {clip: n_trs}
        skipped = 0

        for clip in clips:
            norm = clip.removeprefix("friends_")
            fmri_gt = load_fmri_clip(fmri_dir, subject, "friends", norm,
                                     excl_s, excl_e, stats if use_stats else None)
            if fmri_gt is None:
                log.warning("  No fMRI: %s", clip)
                skipped += 1
                continue

            ctx = load_context_clip(context_dirs, "friends", "s6", clip, expected_dims=cfg.get("modality_dims"))
            if ctx is None:
                log.warning("  No context: %s", clip)
                skipped += 1
                continue

            windows = build_seq2seq_windows(ctx, fmri_gt.shape[0], context_trs,
                                            n_target_trs, hrf_delay, excl_s, stride)
            if not windows:
                skipped += 1
                continue

            clip_fmri_gt[clip] = fmri_gt
            clip_windows[clip] = windows
            clip_contexts[clip] = ctx
            clip_n_trs[clip] = fmri_gt.shape[0]

        total_windows = sum(len(w) for w in clip_windows.values())
        log.info("  Prepared %d clips (%d skipped), %d total windows → batch_size=%d → %d GPU calls",
                 len(clip_windows), skipped, total_windows, args.batch_size,
                 (total_windows + args.batch_size - 1) // args.batch_size)

        if not clip_windows:
            log.warning("  No valid clips for %s", subject)
            continue

        # --- Phase 2: Cross-clip batched inference ---
        batch_results = runner.run_all_clips(
            clip_contexts=clip_contexts,
            clip_windows=clip_windows,
            clip_n_trs=clip_n_trs,
            subject_id=sid,
            n_target_trs=n_target_trs,
            batch_size=args.batch_size,
            solver=solver,
            parcel_seed_map=None,
            return_seed_preds=need_calibration,
            desc=subject,
            hrf_delay=hrf_delay,
            excl_start=excl_s,
        )

        # --- Phase 3: Compute per-clip PCC from batched results ---
        all_pred, all_tgt = [], []
        all_tgt_for_calib = []
        seed_preds_for_calib = [[] for _ in range(max(1, int(solver.n_seeds)))]
        clip_pccs = {}

        for clip in clips:
            if clip not in batch_results:
                continue

            fmri_gt = clip_fmri_gt[clip]
            if need_calibration:
                pred_avg, valid, seed_preds = batch_results[clip]
            else:
                pred_avg, valid = batch_results[clip]

            if valid.sum() < 2:
                continue

            c_pcc = pcc(pred_avg[valid], fmri_gt[valid])
            clip_pccs[clip] = float(np.median(c_pcc))
            log.info("  %s — PCC=%.4f (%d TRs)", clip, clip_pccs[clip], valid.sum())

            all_pred.append(pred_avg[valid])
            all_tgt.append(fmri_gt[valid])
            if need_calibration:
                all_tgt_for_calib.append(fmri_gt[valid])
                for si in range(len(seed_preds)):
                    seed_preds_for_calib[si].append(seed_preds[si][valid])

        if all_pred:
            g_pcc = pcc(np.concatenate(all_pred), np.concatenate(all_tgt))
            med_pcc = float(np.median(g_pcc))
            mean_pcc = float(np.mean(g_pcc))
        else:
            med_pcc = mean_pcc = 0.0
            g_pcc = np.array([])

        log.info(">>> %s — median PCC=%.4f, mean PCC=%.4f", subject, med_pcc, mean_pcc)

        result = {
            "subject": subject, "eval_session": "s6",
            "median_pcc": med_pcc, "mean_pcc": mean_pcc,
            "per_voxel_pcc": g_pcc, "clip_pccs": clip_pccs,
            "solver": vars(solver), "stride": stride,
        }
        np.save(out_dir / f"{subject}_eval_s6.npy", result)
        all_results[subject] = result

        if need_calibration and all_tgt_for_calib:
            tgt_cat = np.concatenate(all_tgt_for_calib, axis=0)
            seed_cat = []
            for si in range(len(seed_preds_for_calib)):
                if seed_preds_for_calib[si]:
                    seed_cat.append(np.concatenate(seed_preds_for_calib[si], axis=0))
            if len(seed_cat) == max(1, int(solver.n_seeds)):
                best_seed_map, pcc_table = best_seed_map_from_s6(seed_cat, tgt_cat)
                calibration["subjects"][subject] = {
                    "best_seed_map": best_seed_map,
                    "pcc_by_seed": pcc_table,
                }
                log.info("Calibration saved in-memory for %s (%d parcels)", subject, best_seed_map.shape[0])

        # Store concatenated GT for noise ceiling (before cleanup)
        if all_tgt:
            subject_fmri_concat[subject] = np.concatenate(all_tgt)

        del all_pred, all_tgt, clip_fmri_gt, clip_windows, batch_results
        gc.collect(); torch.cuda.empty_cache()

    # =====================================================================
    # Extended metrics: noise ceiling, normalized scores, per-network PCC, SEM
    # =====================================================================
    extended_metrics_list = []
    if len(subject_fmri_concat) >= 2:
        log.info("\n%s NOISE CEILING %s", "=" * 20, "=" * 20)
        log.info("Computing inter-subject noise ceiling from %d subjects...",
                 len(subject_fmri_concat))

        # Align all subjects to the same number of TRs (min across subjects)
        min_trs = min(v.shape[0] for v in subject_fmri_concat.values())
        aligned_fmri = {s: v[:min_trs] for s, v in subject_fmri_concat.items()}
        log.info("  Aligned to %d common TRs across %d subjects",
                 min_trs, len(aligned_fmri))

        for subject, r in all_results.items():
            g_pcc = r.get("per_voxel_pcc", np.array([]))
            if len(g_pcc) == 0 or subject not in aligned_fmri:
                continue

            ceiling = compute_noise_ceiling(aligned_fmri, subject)
            ext = compute_extended_metrics(subject, g_pcc, noise_ceiling=ceiling)
            extended_metrics_list.append(ext)

            # Augment stored result with extended metrics
            r["noise_ceiling"] = ceiling
            r["median_noise_ceiling"] = ext.median_noise_ceiling
            r["mean_noise_ceiling"] = ext.mean_noise_ceiling
            r["normalized_score"] = ext.normalized_score
            r["median_normalized"] = ext.median_normalized
            r["mean_normalized"] = ext.mean_normalized
            r["network_pcc"] = ext.network_pcc
            r["network_normalized"] = ext.network_normalized

            log.info(
                "  %s — NC=%.4f | Norm(med/mean)=%.4f/%.4f | Net PCC: %s",
                subject,
                ext.mean_noise_ceiling,
                ext.median_normalized,
                ext.mean_normalized,
                ", ".join(f"{k}={v:.3f}" for k, v in ext.network_pcc.items()),
            )

            # Re-save individual result with extended metrics
            np.save(out_dir / f"{subject}_eval_s6.npy", r)
    else:
        log.info("Skipping noise ceiling: need ≥2 subjects, got %d",
                 len(subject_fmri_concat))
        # Still compute per-network PCC without noise ceiling
        for subject, r in all_results.items():
            g_pcc = r.get("per_voxel_pcc", np.array([]))
            if len(g_pcc) == 0:
                continue
            ext = compute_extended_metrics(subject, g_pcc, noise_ceiling=None)
            extended_metrics_list.append(ext)
            r["network_pcc"] = ext.network_pcc
            np.save(out_dir / f"{subject}_eval_s6.npy", r)

    del subject_fmri_concat  # free memory

    # Cross-subject summary with SEM
    cross_summary = compute_cross_subject_summary(extended_metrics_list)

    # Summary table — basic PCC
    log.info("\n%s SUMMARY %s", "=" * 20, "=" * 20)
    has_nc = any("noise_ceiling" in r for r in all_results.values())
    if has_nc:
        log.info("%-8s  %10s  %10s  %10s  %10s  %10s",
                 "Subject", "Med PCC", "Mean PCC", "Mean NC", "Med Norm", "Mean Norm")
        log.info("-" * 68)
        for subj, r in all_results.items():
            log.info(
                "%-8s  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f",
                subj,
                r["median_pcc"],
                r["mean_pcc"],
                r.get("mean_noise_ceiling", 0.0),
                r.get("median_normalized", 0.0),
                r.get("mean_normalized", 0.0),
            )
        if cross_summary:
            log.info("-" * 68)
            log.info(
                "%-8s  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f",
                "AVG",
                cross_summary.get("avg_median_pcc", 0.0),
                cross_summary.get("avg_mean_pcc", 0.0),
                cross_summary.get("avg_mean_noise_ceiling", 0.0),
                cross_summary.get("avg_median_normalized", 0.0),
                cross_summary.get("avg_mean_normalized", 0.0),
            )
            log.info(
                "%-8s  %10.4f  %10.4f  %10s  %10.4f  %10.4f",
                "SEM",
                cross_summary.get("sem_median_pcc", 0.0),
                cross_summary.get("sem_mean_pcc", 0.0),
                "-",
                cross_summary.get("sem_median_normalized", 0.0),
                cross_summary.get("sem_mean_normalized", 0.0),
            )
    else:
        log.info("%-8s  %10s  %10s", "Subject", "Median PCC", "Mean PCC")
        log.info("-" * 32)
        for subj, r in all_results.items():
            log.info("%-8s  %10.4f  %10.4f", subj, r["median_pcc"], r["mean_pcc"])
        if all_results:
            log.info("-" * 32)
            log.info("%-8s  %10.4f  %10.4f", "AVG",
                     np.mean([r["median_pcc"] for r in all_results.values()]),
                     np.mean([r["mean_pcc"] for r in all_results.values()]))

    # Per-network PCC summary table
    if extended_metrics_list and extended_metrics_list[0].network_pcc:
        log.info("\n%s PER-NETWORK PCC %s", "=" * 15, "=" * 15)
        header_names = [n[:7] for n in NETWORK_NAMES]  # truncate for alignment
        log.info("%-8s  %s", "Subject", "  ".join(f"{n:>8s}" for n in header_names))
        log.info("-" * (10 + 10 * len(NETWORK_NAMES)))
        for ext in extended_metrics_list:
            vals = [ext.network_pcc.get(n, 0.0) for n in NETWORK_NAMES]
            log.info("%-8s  %s", ext.subject,
                     "  ".join(f"{v:8.4f}" for v in vals))
        if cross_summary:
            log.info("-" * (10 + 10 * len(NETWORK_NAMES)))
            avg_vals = [cross_summary.get(f"net_{n}_mean_pcc", 0.0) for n in NETWORK_NAMES]
            sem_vals = [cross_summary.get(f"net_{n}_sem_pcc", 0.0) for n in NETWORK_NAMES]
            log.info("%-8s  %s", "AVG",
                     "  ".join(f"{v:8.4f}" for v in avg_vals))
            log.info("%-8s  %s", "SEM",
                     "  ".join(f"{v:8.4f}" for v in sem_vals))

    # Save everything
    all_results["_cross_subject_summary"] = cross_summary
    np.save(out_dir / "summary_s6.npy", all_results)
    log.info("\nSummary saved → %s", out_dir / "summary_s6.npy")

    if need_calibration and calibration["subjects"]:
        run_root = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow")
        cpath = calibration_path(run_root)
        save_parcel_calibration(cpath, calibration)
        log.info("Parcel calibration saved → %s", cpath)
