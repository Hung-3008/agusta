"""Ensemble (weight-average) the 4 medium-seed checkpoints.

Averages the 4 best.pt checkpoints from outputs/agusta_ensemble_50_medium/
(seeds 1001, 1234, 3137, 5347) into a single checkpoint.

Usage:
    python src/ensemble_medium.py
    python src/ensemble_medium.py --output outputs/agusta_ensemble_50_medium/ensemble.pt
    python src/ensemble_medium.py --weights 1.0 1.0 1.0 1.0
"""

import argparse
import logging
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ensemble_medium")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_state_dict(path: str) -> OrderedDict:
    """Load a state_dict from a checkpoint file.

    Handles both plain state_dicts and wrapped checkpoints (with 'model' key).
    For wrapped checkpoints with EMA, extracts the EMA shadow weights.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    log.info("Loading %s ...", p.name)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        # Wrapped checkpoint — check for EMA
        if "ema" in ckpt:
            shadow = ckpt["ema"]["shadow"]
            model_sd = ckpt["model"]
            full_sd = OrderedDict()
            for k, v in model_sd.items():
                full_sd[k] = shadow[k].clone() if k in shadow else v.clone()
            epoch = ckpt.get("epoch", "?")
            log.info("  → Extracted EMA shadow (epoch=%s), %d keys", epoch, len(full_sd))
            del ckpt
            return full_sd
        else:
            sd = OrderedDict((k, v.clone()) for k, v in ckpt["model"].items())
            epoch = ckpt.get("epoch", "?")
            log.info("  → Loaded model weights (epoch=%s), %d keys", epoch, len(sd))
            del ckpt
            return sd
    else:
        # Plain state_dict
        if isinstance(ckpt, OrderedDict) or isinstance(ckpt, dict):
            sd = OrderedDict((k, v.clone()) for k, v in ckpt.items())
        else:
            sd = ckpt
        log.info("  → Loaded plain state_dict, %d keys", len(sd))
        del ckpt
        return sd


def average_state_dicts(state_dicts: list, weights: list = None) -> OrderedDict:
    """Weighted average of multiple state_dicts. Equal weights if not specified."""
    n = len(state_dicts)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    log.info("Averaging %d checkpoints with weights: %s", n, [f"{w:.3f}" for w in weights])

    # Verify all state_dicts have the same keys
    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        cur_keys = set(sd.keys())
        if cur_keys != ref_keys:
            missing = ref_keys - cur_keys
            extra = cur_keys - ref_keys
            if missing:
                log.warning("  Checkpoint %d missing %d keys: %s", i, len(missing), list(missing)[:5])
            if extra:
                log.warning("  Checkpoint %d has %d extra keys: %s", i, len(extra), list(extra)[:5])

    # Weighted average
    avg_sd = OrderedDict()
    for key in state_dicts[0].keys():
        tensors = []
        for sd in state_dicts:
            if key in sd:
                tensors.append(sd[key].float())  # upcast to float32 for precision

        if len(tensors) == n:
            avg = sum(w * t for w, t in zip(weights, tensors))
            avg_sd[key] = avg.to(dtype=state_dicts[0][key].dtype)
        else:
            log.warning("  Key %s missing in some checkpoints, using first checkpoint value", key)
            avg_sd[key] = state_dicts[0][key].clone()

    return avg_sd


def main():
    parser = argparse.ArgumentParser(description="Ensemble 4 medium-seed BrainFlow checkpoints")
    parser.add_argument(
        "--output", type=str, default="outputs/agusta_ensemble_50_medium/ensemble.pt",
        help="Output path for the ensembled checkpoint",
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default="outputs/agusta_ensemble_50_medium",
        help="Directory containing the seed checkpoints",
    )
    parser.add_argument(
        "--config-source", type=str, default=None,
        help="Config YAML to copy alongside the ensemble checkpoint. "
             "Defaults to <ckpt-dir>/config.yaml if it exists, else brainflow_dit_medium.yaml",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help="Custom weights for each checkpoint (will be normalized). Default: equal weights.",
    )
    args = parser.parse_args()

    ckpt_dir = PROJECT_ROOT / args.ckpt_dir

    # ---- Discover checkpoints ----
    # Look for best_*.pt files in the checkpoint directory
    ckpt_files = sorted(ckpt_dir.glob("best_*.pt"))
    if not ckpt_files:
        # Fallback: look for any .pt files that are not ensemble.pt
        ckpt_files = sorted(
            p for p in ckpt_dir.glob("*.pt")
            if "ensemble" not in p.name
        )

    if not ckpt_files:
        log.error("No checkpoint files found in %s", ckpt_dir)
        sys.exit(1)

    log.info("Found %d checkpoints in %s:", len(ckpt_files), ckpt_dir)
    for f in ckpt_files:
        log.info("  • %s (%.1f MB)", f.name, f.stat().st_size / 1e6)

    # Validate weights count
    if args.weights is not None and len(args.weights) != len(ckpt_files):
        log.error("Number of weights (%d) doesn't match number of checkpoints (%d)",
                  len(args.weights), len(ckpt_files))
        sys.exit(1)

    # ---- Load all state_dicts ----
    state_dicts = []
    for ckpt_path in ckpt_files:
        sd = load_state_dict(str(ckpt_path))
        state_dicts.append(sd)

    # ---- Verify shape compatibility ----
    ref_sd = state_dicts[0]
    for i, sd in enumerate(state_dicts[1:], 1):
        mismatched = []
        for k in ref_sd:
            if k in sd and ref_sd[k].shape != sd[k].shape:
                mismatched.append((k, ref_sd[k].shape, sd[k].shape))
        if mismatched:
            log.error("Shape mismatches with checkpoint %d (%s):", i, ckpt_files[i].name)
            for k, s1, s2 in mismatched:
                log.error("  %s: %s vs %s", k, s1, s2)
            sys.exit(1)
    log.info("✓ All %d checkpoints have compatible shapes (%d keys)", len(state_dicts), len(ref_sd))

    # ---- Average ----
    avg_sd = average_state_dicts(state_dicts, weights=args.weights)

    # Free memory
    del state_dicts

    # ---- Save ----
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(avg_sd, out_path)
    log.info("Saved ensemble checkpoint → %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Copy config alongside
    config_src = None
    if args.config_source:
        config_src = PROJECT_ROOT / args.config_source
    else:
        # Try ckpt_dir/config.yaml first, then fall back to brainflow_dit_medium.yaml
        candidate = ckpt_dir / "config.yaml"
        if candidate.exists():
            config_src = candidate
        else:
            candidate = PROJECT_ROOT / "src/configs/brainflow_dit_medium.yaml"
            if candidate.exists():
                config_src = candidate

    if config_src and config_src.exists():
        cfg_dst = out_path.parent / "config.yaml"
        if config_src.resolve() != cfg_dst.resolve():
            shutil.copy2(config_src, cfg_dst)
            log.info("Copied config → %s", cfg_dst)

    # Also save as best.pt for easy use with evaluate_brainflow.py
    best_path = out_path.parent / "best.pt"
    if best_path.resolve() != out_path.resolve():
        shutil.copy2(out_path, best_path)
        log.info("Copied as best.pt → %s", best_path)

    log.info("\nDone! To evaluate:")
    cfg_eval = out_path.parent / "config.yaml"
    log.info("  python src/evaluate_brainflow.py --config %s --eval_session s6 --checkpoint %s", cfg_eval, out_path)


if __name__ == "__main__":
    main()
