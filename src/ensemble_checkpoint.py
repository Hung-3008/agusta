"""Ensemble (weight-average) multiple BrainFlow checkpoints into one.

Supports loading EMA weights from last.pt or plain state_dict from best.pt.
All checkpoints must share the same architecture (same keys & shapes).

Usage:
    python src/ensemble_checkpoint.py
    python src/ensemble_checkpoint.py --output outputs/brainflow_ensemble/ensemble.pt
"""

import argparse
import copy
import logging
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ensemble_ckpt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_state_dict_from_checkpoint(path: str, use_ema: bool = True) -> OrderedDict:
    """Load a state_dict, optionally extracting EMA shadow weights from last.pt."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    log.info("Loading %s (use_ema=%s) ...", p, use_ema)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        if use_ema and "ema" in ckpt:
            # Extract EMA shadow weights and map to model keys
            shadow = ckpt["ema"]["shadow"]
            model_sd = ckpt["model"]
            # EMA shadow only contains requires_grad params; fill in the rest
            full_sd = OrderedDict()
            for k, v in model_sd.items():
                if k in shadow:
                    full_sd[k] = shadow[k].clone()
                else:
                    full_sd[k] = v.clone()
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
        sd = OrderedDict((k, v.clone()) for k, v in ckpt.items())
        log.info("  → Loaded plain state_dict, %d keys", len(sd))
        del ckpt
        return sd


def average_state_dicts(state_dicts: list[OrderedDict], weights: list[float] = None) -> OrderedDict:
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
            else:
                log.warning("  Key %s missing in a checkpoint, skipping from average", key)

        if len(tensors) == n:
            avg = sum(w * t for w, t in zip(weights, tensors))
            # Preserve original dtype
            avg_sd[key] = avg.to(dtype=state_dicts[0][key].dtype)
        else:
            avg_sd[key] = state_dicts[0][key].clone()

    return avg_sd


def main():
    parser = argparse.ArgumentParser(description="Ensemble BrainFlow checkpoints via weight averaging")
    parser.add_argument(
        "--output", type=str, default="outputs/brainflow_ensemble/ensemble.pt",
        help="Output path for the ensembled checkpoint",
    )
    parser.add_argument(
        "--config-source", type=str, default="outputs/brainflow_dit_large/config.yaml",
        help="Config to copy alongside the ensemble checkpoint",
    )
    args = parser.parse_args()

    # ---- Define checkpoints to ensemble ----
    checkpoints = [
        {
            "name": "dit_large_last",
            "path": "outputs/brainflow_dit_large/last.pt",
            "use_ema": True,
        },
        {
            "name": "dit_large_best",
            "path": "outputs/brainflow_dit_large/best.pt",
            "use_ema": False,
        },
        {
            "name": "dit_large_80_last",
            "path": "outputs/brainflow_dit_large_80/last.pt",
            "use_ema": True,
        },
        {
            "name": "dit_large_80_best",
            "path": "outputs/brainflow_dit_large_80/best.pt",
            "use_ema": False,
        },
        {
            "name": "dit_large_finetune_best",
            "path": "outputs/brainflow_dit_large_finetune/best.pt",
            "use_ema": False,
        },
    ]

    # ---- Load all state_dicts ----
    state_dicts = []
    for ckpt_info in checkpoints:
        p = str(PROJECT_ROOT / ckpt_info["path"])
        sd = load_state_dict_from_checkpoint(p, use_ema=ckpt_info["use_ema"])
        state_dicts.append(sd)
        log.info("  %s: %d keys, sample param shape: %s",
                 ckpt_info["name"], len(sd),
                 list(sd.values())[0].shape)

    # ---- Verify shape compatibility ----
    ref_sd = state_dicts[0]
    for i, sd in enumerate(state_dicts[1:], 1):
        mismatched = []
        for k in ref_sd:
            if k in sd and ref_sd[k].shape != sd[k].shape:
                mismatched.append((k, ref_sd[k].shape, sd[k].shape))
        if mismatched:
            log.error("Shape mismatches with checkpoint %d:", i)
            for k, s1, s2 in mismatched:
                log.error("  %s: %s vs %s", k, s1, s2)
            sys.exit(1)
    log.info("✓ All %d checkpoints have compatible shapes", len(state_dicts))

    # ---- Average ----
    avg_sd = average_state_dicts(state_dicts, weights=None)  # equal weights

    # ---- Save ----
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as plain state_dict (compatible with best.pt loading path)
    torch.save(avg_sd, out_path)
    log.info("Saved ensemble checkpoint → %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Copy config alongside
    cfg_src = PROJECT_ROOT / args.config_source
    cfg_dst = out_path.parent / "config.yaml"
    if cfg_src.exists():
        shutil.copy2(cfg_src, cfg_dst)
        log.info("Copied config → %s", cfg_dst)

    # Also save as best.pt for easy use with evaluate_brainflow.py
    best_path = out_path.parent / "best.pt"
    if not best_path.exists() or best_path == out_path:
        shutil.copy2(out_path, best_path)
        log.info("Copied as best.pt → %s", best_path)

    log.info("\nDone! To evaluate:")
    log.info("  python src/evaluate_brainflow.py --config %s --eval_session s6 --checkpoint %s", cfg_dst, out_path)
    log.info("  python src/evaluate_brainflow.py --config %s --eval_session s7 --checkpoint %s", cfg_dst, out_path)


if __name__ == "__main__":
    main()
