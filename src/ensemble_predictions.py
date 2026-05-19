"""Prediction-Level Ensemble: average predictions from multiple eval runs.

Usage:
    python src/ensemble_predictions.py \
        --pred-dirs outputs/submissions/pred_seed1001/s7 outputs/submissions/pred_seed1234/s7 ... \
        --output-dir outputs/submissions/pred_ensemble_4seed/s7
"""

import argparse
import gc
import logging
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ensemble_pred")


def main():
    parser = argparse.ArgumentParser(description="Prediction-level ensemble")
    parser.add_argument(
        "--pred-dirs", type=str, nargs="+", required=True,
        help="Directories containing per-subject submission.npy files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for ensembled predictions",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help="Optional weights for each prediction set (will be normalized)",
    )
    args = parser.parse_args()

    pred_dirs = [Path(d) for d in args.pred_dirs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate
    for d in pred_dirs:
        if not (d / "submission.npy").exists():
            log.error("submission.npy not found in %s", d)
            return
    log.info("Ensembling predictions from %d runs:", len(pred_dirs))
    for d in pred_dirs:
        log.info("  • %s", d)

    # Weights
    n = len(pred_dirs)
    if args.weights:
        assert len(args.weights) == n, f"Expected {n} weights, got {len(args.weights)}"
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
    else:
        weights = [1.0 / n] * n
    log.info("Weights: %s", [f"{w:.3f}" for w in weights])

    # Load all submissions
    submissions = []
    for d in pred_dirs:
        sub = np.load(d / "submission.npy", allow_pickle=True).item()
        submissions.append(sub)
        log.info("  Loaded %s: %d subjects", d.name, len(sub))

    # Get subjects and episodes
    subjects = sorted(submissions[0].keys())
    log.info("Subjects: %s", subjects)

    # Ensemble
    ensembled = {}
    for subj in subjects:
        log.info("\n%s %s %s", "=" * 15, subj, "=" * 15)
        ensembled[subj] = {}

        episodes = sorted(submissions[0][subj].keys())
        for epi in episodes:
            preds = []
            for i, sub in enumerate(submissions):
                if subj not in sub or epi not in sub[subj]:
                    log.warning("  Missing %s/%s in run %d", subj, epi, i)
                    continue
                preds.append(sub[subj][epi].astype(np.float64))

            if not preds:
                log.warning("  No predictions for %s/%s", subj, epi)
                continue

            # Weighted average
            avg = np.zeros_like(preds[0])
            for p, w in zip(preds, weights[:len(preds)]):
                avg += w * p

            # Re-normalize weights if some runs were missing
            if len(preds) < n:
                w_sum = sum(weights[:len(preds)])
                avg /= w_sum

            ensembled[subj][epi] = avg.astype(np.float32)

        log.info("  %d episodes ensembled", len(ensembled[subj]))

        # Stats comparison
        sample_epi = episodes[0]
        for i, sub in enumerate(submissions):
            v = sub[subj][sample_epi]
            log.info("  Run %d [%s]: std=%.4f, range=[%.3f, %.3f]",
                     i, sample_epi, v.std(), v.min(), v.max())
        ev = ensembled[subj][sample_epi]
        log.info("  Ensembled [%s]: std=%.4f, range=[%.3f, %.3f]",
                 sample_epi, ev.std(), ev.min(), ev.max())

    del submissions
    gc.collect()

    # Save
    sub_path = out_dir / "submission.npy"
    np.save(sub_path, ensembled)
    log.info("\nSaved ensembled submission → %s (%.1f MB)",
             sub_path, sub_path.stat().st_size / 1e6)

    # Create zip
    zip_path = sub_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.npy")
    log.info("Created ZIP → %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    del ensembled
    gc.collect()
    log.info("\nDone!")


if __name__ == "__main__":
    main()
