"""evaluate_brainflow.py — Clean seq2seq evaluation & submission for BrainFlow.

Framework design:
  - ModelRunner wraps model init + inference; swap it for a new architecture
    without touching window/metric/IO logic.
  - Only seq2seq mode is supported (n_target_trs > 1).

Modes:
  s6  — per-subject PCC validation (Friends S6, has ground-truth fMRI)
  s7  — blind submission for Friends S7 in-distribution test
  ood — blind submission for OOD movies (chaplin, mononoke, passepartout,
         planetearth, pulpfiction, wot; each split into part 1 and 2)

Feature directory layout:
  friends/s6/{clip}.npy   (S6 validation)
  friends/s7/{clip}.npy   (S7 in-distribution)
  ood/{movie}/{clip}.npy  (OOD out-of-distribution)

Sample count files (in fmri/sub-XX/target_sample_number/):
  sub-XX_friends-s7_fmri_samples.npy  → {episode: n_trs}
  sub-XX_ood_fmri_samples.npy         → {movie_split: n_trs} e.g. {'chaplin1': 432, ...}

Usage:
    python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s6
    python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s7
    python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session ood
    python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s6 --ema_only --stride 5
    python src/evaluate_brainflow.py --config src/configs/brainflow.yaml --eval_session s7 --checkpoint outputs/.../last.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.directflow_dataset import load_config
from src.evaluate.config import SolverConfig
from src.evaluate.model_runner import ModelRunner
from src.evaluate.run_s6 import run_s6
from src.evaluate.run_s7 import run_s7, run_s7_parallel
from src.evaluate.run_ood import run_ood

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")


def main():
    p = argparse.ArgumentParser(description="BrainFlow Seq2Seq Evaluator")
    p.add_argument("--config",         default="src/configs/brainflow.yaml")
    p.add_argument("--eval_session",   default="s6", choices=["s6", "s7", "ood"],
                   help="s6=PCC validation | s7=Friends S7 submission | ood=OOD submission")
    p.add_argument("--checkpoint",     default=None,  help="Explicit checkpoint path")
    p.add_argument("--ema_only",       action="store_true", help="Load EMA from last.pt (skip best.pt)")
    p.add_argument("--batch_size",     type=int,   default=512)
    p.add_argument("--stride",         type=int,   default=None, help="Override window stride")
    p.add_argument("--n_timesteps",    type=int,   default=None, help="ODE steps")
    p.add_argument("--solver_method",  default=None, help="midpoint | euler")
    p.add_argument("--temperature",    type=float, default=None, help="Initial noise std")
    p.add_argument("--cfg_scale",      type=float, default=None, help="Classifier-free guidance scale")
    p.add_argument("--time_grid_warp", default=None, help="none|linear|sqrt")
    p.add_argument("--time_grid_max",  type=float, default=None,
                   help="Upper ODE integration bound in [0,1] for singularity avoidance")
    p.add_argument("--final_jump", action=argparse.BooleanOptionalAction, default=None,
                   help="Apply residual jump at final time if time_grid_max < 1")

    p.add_argument("--use_pruned_sampling", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable strategy-2 pruned sampling around regression anchor")
    p.add_argument("--prune_k", type=int, default=None, help="Number of x0 candidates for pruning")
    p.add_argument("--n_seeds", type=int, default=None, help="Number of seeds for strategy-4 ensemble")
    p.add_argument("--ensemble_mode", default=None, choices=["none", "mean", "max", "parcel_stitch"],
                   help="Ensemble aggregation mode")
    p.add_argument("--base_seed", type=int, default=None, help="Base random seed for multi-seed inference")

    p.add_argument("--device",         default="cuda")
    p.add_argument("--resume",         action="store_true", default=True,
                   help="[S7] Skip already-completed subjects")
    p.add_argument("--no_resume",      dest="resume", action="store_false")
    p.add_argument("--output_dir",     default=None, help="Override output directory")
    p.add_argument("--parallel_subjects", action="store_true", default=True,
                   help="[S7/OOD] Decode all subjects simultaneously (default: enabled)")
    p.add_argument("--sequential_subjects", dest="parallel_subjects", action="store_false",
                   help="[S7/OOD] Decode subjects sequentially (legacy mode)")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg["_fmri_dir"] = str(PROJECT_ROOT / cfg["data_root"] / cfg["fmri"]["dir"])

    context_dirs = [PROJECT_ROOT / d for d in cfg.get("context_latent_dirs", [cfg.get("context_latent_dir")])]
    out_dir = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow")
    solver = SolverConfig.from_cfg(cfg, args)

    log.info("Config: %s | Session: %s | Model version: %s",
             args.config, args.eval_session, cfg.get("model_version", "?"))
    log.info(
        "Inference config: mode=%s seeds=%d prune=%s(k=%d) t_max=%.3f final_jump=%s",
        solver.ensemble_mode,
        solver.n_seeds,
        solver.use_pruned_sampling,
        solver.prune_k,
        solver.time_grid_max,
        solver.final_jump,
    )

    # Build model + load weights
    runner = ModelRunner(cfg, device)
    runner.load_checkpoint(out_dir, override=args.checkpoint, ema_only=args.ema_only)

    if args.eval_session == "s6":
        run_s6(runner, cfg, context_dirs, args, solver)
    elif args.eval_session == "s7":
        if args.parallel_subjects:
            run_s7_parallel(runner, cfg, context_dirs, args, solver)
        else:
            run_s7(runner, cfg, context_dirs, args, solver)
    else:  # ood
        run_ood(runner, cfg, context_dirs, args, solver)


if __name__ == "__main__":
    main()
