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
import gc
import logging
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.directflow_dataset import load_config, _get_fmri_filepath
from src.models.brainflow.brainflow import BrainFlow
from src.utils.utils import (
    InferenceStrategyConfig,
    best_seed_map_from_s6,
    calibration_path,
    load_parcel_calibration,
    run_multiseed_synthesis,
    save_parcel_calibration,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")


# =============================================================================
# ModelRunner — isolates all model-specific logic
# Adapt ONLY this class when the model architecture changes.
# =============================================================================

@dataclass
class SolverConfig:
    """ODE solver settings, loaded from config and optionally overridden by CLI."""
    n_timesteps: int = 30
    method: str = "midpoint"
    temperature: float = 0.05
    cfg_scale: float = 0.0
    time_grid_warp: str | None = None
    time_grid_max: float = 1.0
    final_jump: bool = False
    use_pruned_sampling: bool = False
    prune_k: int = 5
    n_seeds: int = 1
    ensemble_mode: str = "none"
    base_seed: int = 1234

    @classmethod
    def from_cfg(cls, cfg: dict, args) -> "SolverConfig":
        s = cfg.get("solver_args", {})
        return cls(
            n_timesteps=args.n_timesteps or s.get("time_points", 30),
            method=args.solver_method or s.get("method", "midpoint"),
            temperature=args.temperature if args.temperature is not None else s.get("temperature", 0.05),
            cfg_scale=args.cfg_scale if args.cfg_scale is not None else s.get("cfg_scale", 0.0),
            time_grid_warp=args.time_grid_warp if args.time_grid_warp is not None else s.get("time_grid_warp", None),
            time_grid_max=args.time_grid_max if args.time_grid_max is not None else s.get("time_grid_max", 1.0),
            final_jump=args.final_jump if args.final_jump is not None else s.get("final_jump", False),
            use_pruned_sampling=args.use_pruned_sampling if args.use_pruned_sampling is not None else s.get("use_pruned_sampling", False),
            prune_k=args.prune_k if args.prune_k is not None else s.get("prune_k", 5),
            n_seeds=args.n_seeds if args.n_seeds is not None else s.get("n_seeds", 1),
            ensemble_mode=args.ensemble_mode if args.ensemble_mode is not None else s.get("ensemble_mode", "none"),
            base_seed=args.base_seed if args.base_seed is not None else s.get("base_seed", 1234),
        )

    def as_synth_kwargs(self) -> dict:
        kw = dict(
            n_timesteps=self.n_timesteps,
            solver_method=self.method,
            temperature=self.temperature,
            time_grid_max=self.time_grid_max,
            final_jump=self.final_jump,
        )
        if self.cfg_scale > 0:
            kw["cfg_scale"] = self.cfg_scale
        if self.time_grid_warp:
            kw["time_grid_warp"] = self.time_grid_warp
        return kw

    def as_strategy_config(self) -> InferenceStrategyConfig:
        return InferenceStrategyConfig(
            use_pruned_sampling=bool(self.use_pruned_sampling),
            prune_k=max(1, int(self.prune_k)),
            n_seeds=max(1, int(self.n_seeds)),
            ensemble_mode=(self.ensemble_mode or "none").lower(),
            base_seed=int(self.base_seed),
        )


class ModelRunner:
    """Wraps BrainFlow: build → load checkpoint → run inference.

    To support a new model architecture, subclass this and override:
      - build()        → instantiate the new model
      - synthesise()   → call the new synthesis API

    Everything else (windowing, PCC, IO) stays the same.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.subjects = cfg["subjects"]
        self.subject_to_idx = {s: i for i, s in enumerate(self.subjects)}
        self.n_voxels = cfg["fmri"]["n_voxels"]
        self.model = self._build()

    def _build(self) -> BrainFlow:
        """Build BrainFlow from YAML config — no hardcoded defaults."""
        bf = self.cfg["brainflow"]
        vn_params = dict(bf.get("velocity_net", {}))
        modality_dims = self.cfg.get("modality_dims")
        if modality_dims:
            vn_params["modality_dims"] = modality_dims

        model = BrainFlow(
            output_dim=bf.get("output_dim", self.n_voxels),
            velocity_net_params=vn_params,
            n_subjects=len(self.subjects),
            tensor_fm_params=bf.get("tensor_fm", None),
            indi_flow_matching=bf.get("indi_flow_matching", False),
            indi_train_time_sqrt=bf.get("indi_train_time_sqrt", False),
            indi_min_denom=bf.get("indi_min_denom", 1e-3),
            use_csfm=bf.get("use_csfm", False),
            csfm_var_reg_weight=bf.get("csfm_var_reg_weight", 0.1),
            csfm_pcc_weight=bf.get("csfm_pcc_weight", 1.0),
            flow_loss_weight=bf.get("flow_loss_weight", 1.0),
        ).to(self.device)

        log.info("Built BrainFlow: %s params | decoder=%s",
                 f"{sum(p.numel() for p in model.parameters()):,}",
                 bf.get("velocity_net", {}).get("decoder_type", "?"))
        return model

    def load_checkpoint(self, out_dir: Path, override: str | None = None, ema_only: bool = False):
        """Load weights with EMA support.

        Priority: override → best.pt (state_dict) → last.pt (EMA).
        """
        def _load_full(path: Path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and ("ema" in ckpt or "model" in ckpt):
                if "ema" in ckpt:
                    from src.train_brainflow import EMAModel
                    ema = EMAModel(self.model, decay=0.999, store_on_cpu=False)
                    ema.load_state_dict(ckpt["ema"])
                    ema.apply_shadow(self.model)
                    log.info("  Applied EMA shadow weights from %s", path.name)
                else:
                    self.model.load_state_dict(ckpt["model"])
                    log.info("  Loaded model weights from %s", path.name)
            else:
                self.model.load_state_dict(ckpt)
                log.info("  Loaded plain state_dict from %s", path.name)
            del ckpt

        if override:
            p = Path(override)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            log.info("Loading checkpoint (override): %s", p)
            _load_full(p)
        elif not ema_only and (out_dir / "best.pt").exists():
            p = out_dir / "best.pt"
            log.info("Loading best.pt: %s", p)
            try:
                ckpt = torch.load(p, map_location=self.device, weights_only=True)
                self.model.load_state_dict(ckpt)
                log.info("  Loaded plain state_dict from best.pt")
                del ckpt
            except Exception:
                _load_full(p)
        elif (out_dir / "last.pt").exists():
            log.info("Loading EMA from last.pt: %s", out_dir / 'last.pt')
            _load_full(out_dir / "last.pt")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {out_dir}. Use --checkpoint to specify a path."
            )

        self.model.eval()

    @torch.inference_mode()
    def run_windows(
        self,
        windows: list[dict],
        subject_id: int,
        n_trs: int,
        n_target_trs: int,
        batch_size: int,
        solver: SolverConfig,
        parcel_seed_map: np.ndarray | None = None,
        return_seed_preds: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Batched ODE inference over seq2seq windows with overlap-average.

        Returns:
            pred_avg:   (n_trs, n_voxels) float32
            valid_mask: (n_trs,) bool — TRs with at least one window prediction
        """
        pred_sum = np.zeros((n_trs, self.n_voxels), dtype=np.float64)
        pred_count = np.zeros(n_trs, dtype=np.float64)

        all_ctx = np.stack([w["context"] for w in windows])
        starts = [w["target_start"] for w in windows]
        synth_kw = solver.as_synth_kwargs()
        strategy = solver.as_strategy_config()
        effective_seeds = strategy.n_seeds if strategy.ensemble_mode not in ("none", "single") else 1
        seed_sums = [np.zeros((n_trs, self.n_voxels), dtype=np.float64) for _ in range(effective_seeds)]
        seed_counts = [np.zeros(n_trs, dtype=np.float64) for _ in range(effective_seeds)]

        for bi in range(0, len(windows), batch_size):
            be = min(bi + batch_size, len(windows))
            B = be - bi

            ctx = torch.from_numpy(all_ctx[bi:be]).to(self.device)
            subj = torch.full((B,), subject_id, dtype=torch.long, device=self.device)

            pred_agg, preds_by_seed = run_multiseed_synthesis(
                model=self.model,
                context=ctx,
                subject_ids=subj,
                synth_kwargs=synth_kw,
                strategy=strategy,
                parcel_seed_map=parcel_seed_map,
            )
            pred_np = pred_agg.float().cpu().numpy()
            pred_np_by_seed = [p.float().cpu().numpy() for p in preds_by_seed]

            for j in range(B):
                ts = starts[bi + j]
                for off in range(n_target_trs):
                    tr = ts + off
                    if tr >= n_trs:
                        break
                    pred_sum[tr] += pred_np[j, off]
                    pred_count[tr] += 1
                    for si in range(len(pred_np_by_seed)):
                        seed_sums[si][tr] += pred_np_by_seed[si][j, off]
                        seed_counts[si][tr] += 1

            del ctx, subj, pred_agg, preds_by_seed
            torch.cuda.empty_cache()

        valid = pred_count > 0
        pred_avg = np.zeros((n_trs, self.n_voxels), dtype=np.float32)
        pred_avg[valid] = (pred_sum[valid] / pred_count[valid, None]).astype(np.float32)

        if not return_seed_preds:
            return pred_avg, valid

        preds_seed_avg: list[np.ndarray] = []
        for si in range(effective_seeds):
            s_valid = seed_counts[si] > 0
            s_avg = np.zeros((n_trs, self.n_voxels), dtype=np.float32)
            s_avg[s_valid] = (seed_sums[si][s_valid] / seed_counts[si][s_valid, None]).astype(np.float32)
            preds_seed_avg.append(s_avg)
        return pred_avg, valid, preds_seed_avg


# =============================================================================
# Data helpers
# =============================================================================

def _movie_from_clip(clip_key: str) -> str:
    """Extract movie name from OOD clip key: 'chaplin1' → 'chaplin'."""
    return clip_key.rstrip("0123456789")


def load_context_clip(context_dirs: list[Path], task: str, session: str,
                      clip_name: str, expected_dims: list[int] | None = None) -> np.ndarray | None:
    """Load and concatenate multi-modal context for one clip → (T, D_total).

    For task='friends':  looks in {ctx_dir}/friends/{session}/{clip_name}.npy
    For task='ood':      looks in {ctx_dir}/ood/{movie}/{clip_name}.npy
                         where movie = clip_name with trailing digits stripped.
    """
    arrays = []
    for ctx_dir in context_dirs:
        if task == "ood":
            movie = _movie_from_clip(clip_name)
            base = ctx_dir / "ood" / movie
            candidates = [
                base / f"{clip_name}.npy",
                base / f"{clip_name.lstrip(movie)}.npy",  # just the number part
                base / f"task-{clip_name}_video.npy",     # e.g. task-chaplin1_video.npy
                base / f"ood_{clip_name}.npy",            # e.g. ood_chaplin1.npy
                base / f"task-{clip_name}.npy"
            ]
        else:
            base = ctx_dir / task / session
            candidates = [
                base / f"{clip_name}.npy",
                base / f"{clip_name.removeprefix(f'{task}_')}.npy",
                base / f"{task}_{clip_name}.npy",
            ]

        for candidate in candidates:
            if candidate.exists():
                arr = np.load(candidate).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0], -1)
                arrays.append(arr)
                break
        else:
            # Zero-pad missing modality with correct or best-guess dim
            dim = 512
            if expected_dims and len(arrays) < len(expected_dims):
                dim = expected_dims[len(arrays)]
            else:
                for f in (ctx_dir / "ood" / _movie_from_clip(clip_name) if task == "ood" else ctx_dir / task / session).rglob("*.npy"):
                    dim = np.load(f).shape[-1]
                    break
            ref = arrays[0].shape[0] if arrays else 100
            arrays.append(np.zeros((ref, dim), dtype=np.float32))

    if not arrays:
        return None
    t = max(a.shape[0] for a in arrays)
    padded_arrays = []
    for a in arrays:
        if a.shape[0] < t:
            pad_len = t - a.shape[0]
            padded_arrays.append(np.pad(a, ((0, pad_len), (0, 0))))
        else:
            padded_arrays.append(a)
    return np.concatenate(padded_arrays, axis=-1)


def load_fmri_clip(fmri_dir: str, subject: str, task: str, clip_key: str,
                   excl_start: int, excl_end: int,
                   fmri_stats: dict | None) -> np.ndarray | None:
    """Load and (optionally) z-score fMRI for one clip → (T, V)."""
    path = _get_fmri_filepath(fmri_dir, subject, task)
    if not path.exists():
        return None

    key = clip_key
    if clip_key.startswith(f"{task}_"):
        key = clip_key[len(f"{task}_"):]

    with h5py.File(path, "r") as f:
        matches = [k for k in f.keys() if key in k]
        if not matches:
            return None
        raw = f[matches[0]]
        end = len(raw) - excl_end if excl_end > 0 else len(raw)
        data = raw[excl_start:end].astype(np.float32)

    data = np.nan_to_num(data, nan=0.0)
    if fmri_stats and subject in fmri_stats:
        s = fmri_stats[subject]
        data = (data - s["mean"]) / s["std"]
    return data


def load_fmri_stats(fmri_dir: str, subjects: list[str]) -> dict:
    """Load per-subject global mean/std for z-score normalization."""
    stats = {}
    for subj in subjects:
        d = Path(fmri_dir) / subj / "stats"
        mp, sp = d / "global_mean.npy", d / "global_std.npy"
        if mp.exists() and sp.exists():
            stats[subj] = {
                "mean": np.load(mp).astype(np.float32)[None, :],  # (1, V)
                "std":  np.load(sp).astype(np.float32)[None, :],
            }
            log.info("Loaded fMRI stats: %s", subj)
    return stats


def build_seq2seq_windows(ctx: np.ndarray, n_trs: int, context_trs: int,
                          n_target_trs: int, hrf_delay: int, excl_start: int,
                          stride: int) -> list[dict]:
    """Sliding-window builder for S6 evaluation (with HRF shift)."""
    extra_past = (context_trs - n_target_trs) // 2
    windows = []
    for ts in range(0, max(0, n_trs - n_target_trs + 1), stride):
        feat0 = (ts + excl_start) - hrf_delay
        c0 = feat0 - extra_past
        c1 = c0 + context_trs
        chunk = ctx[max(0, c0):min(ctx.shape[0], c1)]
        if chunk.shape[0] < context_trs:
            pb = max(0, -c0)
            pa = context_trs - chunk.shape[0] - pb
            chunk = np.pad(chunk, ((pb, pa), (0, 0)))
        windows.append({"target_start": ts, "context": chunk[:context_trs]})
    return windows


def build_s7_windows(ctx: np.ndarray, n_trs: int, context_trs: int,
                     n_target_trs: int, hrf_delay: int, stride: int) -> list[dict]:
    """Sliding-window builder for S7/OOD submission (with HRF shift)."""
    extra_past = (context_trs - n_target_trs) // 2
    windows = []
    for ts in range(0, max(0, n_trs - n_target_trs + 1), stride):
        feat0 = ts - hrf_delay
        c0 = feat0 - extra_past
        c1 = c0 + context_trs
        chunk = ctx[max(0, c0):min(ctx.shape[0], c1)]
        if chunk.shape[0] < context_trs:
            pb = max(0, -c0)
            pa = context_trs - chunk.shape[0] - pb
            chunk = np.pad(chunk, ((pb, pa), (0, 0)))
        windows.append({"target_start": ts, "context": chunk[:context_trs]})
    return windows


def pcc(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-voxel Pearson correlation → (V,)."""
    p = pred - pred.mean(0, keepdims=True)
    t = target - target.mean(0, keepdims=True)
    cov = (p * t).sum(0)
    std = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    return cov / (std + 1e-8)


# =============================================================================
# S6 — validation with ground-truth
# =============================================================================

def run_s6(runner: ModelRunner, cfg: dict, context_dirs: list[Path],
           args, solver: SolverConfig):
    """Per-subject S6 PCC evaluation. Saves results under {output_dir}/eval_s6/."""
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
        all_pred, all_tgt = [], []
        all_tgt_for_calib = []
        seed_preds_for_calib = [[] for _ in range(max(1, int(solver.n_seeds)))]
        clip_pccs = {}

        for clip in tqdm(clips, desc=subject):
            norm = clip.removeprefix("friends_")

            fmri_gt = load_fmri_clip(fmri_dir, subject, "friends", norm,
                                     excl_s, excl_e, stats if use_stats else None)
            if fmri_gt is None:
                log.warning("  No fMRI: %s", clip)
                continue

            ctx = load_context_clip(context_dirs, "friends", "s6", clip, expected_dims=cfg.get("modality_dims"))
            if ctx is None:
                log.warning("  No context: %s", clip)
                continue

            windows = build_seq2seq_windows(ctx, fmri_gt.shape[0], context_trs,
                                            n_target_trs, hrf_delay, excl_s, stride)
            if not windows:
                continue

            run_out = runner.run_windows(
                windows,
                sid,
                fmri_gt.shape[0],
                n_target_trs,
                args.batch_size,
                solver,
                parcel_seed_map=None,
                return_seed_preds=need_calibration,
            )
            if need_calibration:
                pred_avg, valid, seed_preds = run_out
            else:
                pred_avg, valid = run_out

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
            del ctx, fmri_gt, pred_avg

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

        del all_pred, all_tgt
        gc.collect(); torch.cuda.empty_cache()

    # Summary table
    log.info("\n%s SUMMARY %s", "=" * 20, "=" * 20)
    log.info("%-8s  %10s  %10s", "Subject", "Median PCC", "Mean PCC")
    log.info("-" * 32)
    for subj, r in all_results.items():
        log.info("%-8s  %10.4f  %10.4f", subj, r["median_pcc"], r["mean_pcc"])
    if all_results:
        log.info("-" * 32)
        log.info("%-8s  %10.4f  %10.4f", "AVG",
                 np.mean([r["median_pcc"] for r in all_results.values()]),
                 np.mean([r["mean_pcc"] for r in all_results.values()]))

    np.save(out_dir / "summary_s6.npy", all_results)
    log.info("Summary saved → %s", out_dir / "summary_s6.npy")

    if need_calibration and calibration["subjects"]:
        run_root = Path(PROJECT_ROOT) / cfg.get("output_dir", "outputs/brainflow")
        cpath = calibration_path(run_root)
        save_parcel_calibration(cpath, calibration)
        log.info("Parcel calibration saved → %s", cpath)


# =============================================================================
# S7 — Friends blind submission (in-distribution)
# =============================================================================

def run_s7(runner: ModelRunner, cfg: dict, context_dirs: list[Path],
           args, solver: SolverConfig):
    """Generate Friends S7 blind submission (in-distribution test)."""
    sw = cfg["sliding_window"]
    context_trs  = sw["context_trs"]
    n_target_trs = sw["n_target_trs"]
    stride = args.stride or sw.get("stride", 10)
    fmri_dir = cfg["_fmri_dir"]
    n_voxels = runner.n_voxels
    hrf_delay = cfg["fmri"].get("hrf_delay", 5)

    use_stats = cfg["fmri"].get("use_global_stats", False)
    stats = load_fmri_stats(fmri_dir, runner.subjects) if use_stats else {}

    run_name = args.sub_name if args.sub_name else Path(cfg.get("output_dir", "outputs/brainflow")).name
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

    # Load S7 sample counts: {episode: n_trs}
    def _load_s7_samples(subject):
        p = (Path(PROJECT_ROOT) / "Data" / "algonauts_2025.competitors" / "fmri"
             / subject / "target_sample_number"
             / f"{subject}_friends-s7_fmri_samples.npy")
        return np.load(p, allow_pickle=True).item()

    subj_paths = {}

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
        samples = _load_s7_samples(subject)
        subj_dict = {}

        for epi, n_trs in tqdm(samples.items(), desc=subject):
            clip = f"friends_{epi}"
            ctx = load_context_clip(context_dirs, "friends", "s7", clip, expected_dims=cfg.get("modality_dims"))

            if ctx is None or ctx.shape[0] == 0:
                log.warning("  No context: %s — emitting zeros", clip)
                subj_dict[epi] = np.zeros((n_trs, n_voxels), dtype=np.float32)
                continue

            windows = build_s7_windows(ctx, n_trs, context_trs, n_target_trs, hrf_delay, stride)
            if not windows:
                subj_dict[epi] = np.zeros((n_trs, n_voxels), dtype=np.float32)
                continue

            pred_avg, valid = runner.run_windows(
                windows,
                sid,
                n_trs,
                n_target_trs,
                args.batch_size,
                solver,
                parcel_seed_map=subject_seed_map,
                return_seed_preds=False,
            )

            # Denormalize if trained with z-score
            if use_stats and subject in stats:
                s = stats[subject]
                pred_avg = pred_avg * s["std"] + s["mean"]

            pred_avg = np.nan_to_num(pred_avg, nan=0.0)
            subj_dict[epi] = pred_avg.astype(np.float32)
            log.info("  %s: %d/%d TRs predicted", epi, int(valid.sum()), n_trs)
            del ctx, windows, pred_avg

        np.save(out_path, subj_dict)
        subj_paths[subject] = out_path
        log.info("  Saved %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
        del subj_dict
        gc.collect(); torch.cuda.empty_cache()

    _save_submission(subj_paths, out_dir, tag="s7")


# =============================================================================
# OOD — blind submission (out-of-distribution)
# =============================================================================

def run_ood(runner: ModelRunner, cfg: dict, context_dirs: list[Path],
            args, solver: SolverConfig):
    """Generate OOD blind submission (movies: chaplin, mononoke, passepartout,
    planetearth, pulpfiction, wot — each split into part 1 and 2).

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

    run_name = args.sub_name if args.sub_name else Path(cfg.get("output_dir", "outputs/brainflow")).name
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

    def _load_ood_samples(subject):
        p = (Path(PROJECT_ROOT) / "Data" / "algonauts_2025.competitors" / "fmri"
             / subject / "target_sample_number"
             / f"{subject}_ood_fmri_samples.npy")
        return np.load(p, allow_pickle=True).item()

    subj_paths = {}

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
        samples = _load_ood_samples(subject)  # {'chaplin1': 432, 'chaplin2': 405, ...}
        subj_dict = {}

        for clip_key, n_trs in tqdm(samples.items(), desc=subject):
            # clip_key = 'chaplin1' → movie = 'chaplin', feature at ood/chaplin/chaplin1.npy
            ctx = load_context_clip(context_dirs, "ood", None, clip_key, expected_dims=cfg.get("modality_dims"))

            if ctx is None or ctx.shape[0] == 0:
                log.warning("  No context: %s — emitting zeros", clip_key)
                subj_dict[clip_key] = np.zeros((n_trs, n_voxels), dtype=np.float32)
                continue

            windows = build_s7_windows(ctx, n_trs, context_trs, n_target_trs, hrf_delay, stride)
            if not windows:
                subj_dict[clip_key] = np.zeros((n_trs, n_voxels), dtype=np.float32)
                continue

            pred_avg, valid = runner.run_windows(
                windows,
                sid,
                n_trs,
                n_target_trs,
                args.batch_size,
                solver,
                parcel_seed_map=subject_seed_map,
                return_seed_preds=False,
            )

            if use_stats and subject in stats:
                s = stats[subject]
                pred_avg = pred_avg * s["std"] + s["mean"]

            pred_avg = np.nan_to_num(pred_avg, nan=0.0)
            subj_dict[clip_key] = pred_avg.astype(np.float32)
            log.info("  %s: %d/%d TRs predicted", clip_key, int(valid.sum()), n_trs)
            del ctx, windows, pred_avg

        np.save(out_path, subj_dict)
        subj_paths[subject] = out_path
        log.info("  Saved %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
        del subj_dict
        gc.collect(); torch.cuda.empty_cache()

    _save_submission(subj_paths, out_dir, tag="ood")


# =============================================================================
# Shared: assemble + zip submission
# =============================================================================

def _save_submission(subj_paths: dict, out_dir: Path, tag: str):
    """Merge per-subject npy files → submission.npy + submission.zip."""
    log.info("\nAssembling %s submission...", tag)
    submission = {s: np.load(p, allow_pickle=True).item() for s, p in subj_paths.items()}
    sub_path = out_dir / "submission.npy"
    np.save(sub_path, submission)
    del submission; gc.collect()

    zip_path = sub_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_path, arcname="submission.npy")

    log.info("submission.zip → %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)
    for p in subj_paths.values():
        p.unlink(missing_ok=True)
    log.info("Temp files cleaned.")


# =============================================================================
# Entry point
# =============================================================================

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
    p.add_argument("--sub_name",       default=None, help="Custom memorable name for the submission directory")
    p.add_argument("--output_dir",     default=None, help="Override output directory")
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
        run_s7(runner, cfg, context_dirs, args, solver)
    else:  # ood
        run_ood(runner, cfg, context_dirs, args, solver)


if __name__ == "__main__":
    main()
