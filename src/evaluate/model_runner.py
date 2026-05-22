"""ModelRunner — isolates all model-specific logic.

Adapt ONLY this class when the model architecture changes.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.evaluate.config import SolverConfig
from src.models.brainflow.brainflow import BrainFlow
from src.utils.utils import run_multiseed_synthesis

log = logging.getLogger("evaluate")


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
                    missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
                    log.info("  Loaded model weights from %s (strict=False)", path.name)
                    if missing:
                        log.warning("  Missing keys in state_dict: %s", missing)
                    if unexpected:
                        log.warning("  Unexpected keys in state_dict: %s", unexpected)
            else:
                missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
                log.info("  Loaded plain state_dict from %s (strict=False)", path.name)
                if missing:
                    log.warning("  Missing keys in state_dict: %s", missing)
                if unexpected:
                    log.warning("  Unexpected keys in state_dict: %s", unexpected)
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
                missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
                log.info("  Loaded plain state_dict from best.pt (strict=False)")
                if missing:
                    log.warning("  Missing keys in state_dict: %s", missing)
                if unexpected:
                    log.warning("  Unexpected keys in state_dict: %s", unexpected)
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

    @torch.inference_mode()
    def run_all_clips(
        self,
        clip_windows: dict[str, list[dict]],
        clip_n_trs: dict[str, int],
        subject_id: int,
        n_target_trs: int,
        batch_size: int,
        solver: SolverConfig,
        parcel_seed_map: np.ndarray | None = None,
        return_seed_preds: bool = False,
        desc: str = "",
    ) -> dict[str, tuple]:
        """Cross-clip batched ODE inference — aggregate windows from ALL clips
        into large GPU batches for maximum throughput.

        Instead of processing ~40 windows/clip × 49 clips sequentially,
        this pools ~2000 windows and processes them in ceil(2000/batch_size)
        large GPU calls.

        Args:
            clip_windows:  {clip_name: [windows]} from build_seq2seq_windows.
            clip_n_trs:    {clip_name: n_trs} total TRs per clip.
            subject_id:    Integer subject index.
            n_target_trs:  Number of target TRs per window.
            batch_size:    GPU batch size (now actually used at scale).
            solver:        ODE solver config.
            parcel_seed_map: Optional parcel seed map for stitching.
            return_seed_preds: If True, also return per-seed predictions.
            desc:          tqdm description string.

        Returns:
            {clip_name: (pred_avg, valid) or (pred_avg, valid, seed_preds)}
        """
        synth_kw = solver.as_synth_kwargs()
        strategy = solver.as_strategy_config()
        effective_seeds = strategy.n_seeds if strategy.ensemble_mode not in ("none", "single") else 1

        # --- Phase 1: Flatten all windows into a single array ---
        clip_names = list(clip_windows.keys())
        all_contexts = []       # list of (T_ctx, D) arrays
        all_starts = []         # target_start per window
        all_clip_idx = []       # which clip each window belongs to
        clip_name_to_idx = {c: i for i, c in enumerate(clip_names)}

        for clip_name in clip_names:
            for w in clip_windows[clip_name]:
                all_contexts.append(w["context"])
                all_starts.append(w["target_start"])
                all_clip_idx.append(clip_name_to_idx[clip_name])

        total_windows = len(all_contexts)
        if total_windows == 0:
            return {}

        all_starts_np = np.array(all_starts, dtype=np.int64)
        all_clip_idx_np = np.array(all_clip_idx, dtype=np.int64)
        del all_starts, all_clip_idx

        # --- Phase 2: Allocate per-clip accumulators ---
        accums = {}
        for clip_name in clip_names:
            n_trs = clip_n_trs[clip_name]
            accums[clip_name] = {
                "pred_sum": np.zeros((n_trs, self.n_voxels), dtype=np.float64),
                "pred_count": np.zeros(n_trs, dtype=np.float64),
            }
            if return_seed_preds:
                accums[clip_name]["seed_sums"] = [
                    np.zeros((n_trs, self.n_voxels), dtype=np.float64)
                    for _ in range(effective_seeds)
                ]
                accums[clip_name]["seed_counts"] = [
                    np.zeros(n_trs, dtype=np.float64)
                    for _ in range(effective_seeds)
                ]

        # --- Phase 3: Process in large cross-clip batches ---
        n_batches = (total_windows + batch_size - 1) // batch_size
        pbar = tqdm(
            range(0, total_windows, batch_size),
            desc=desc or "batches",
            total=n_batches,
        )

        for bi in pbar:
            be = min(bi + batch_size, total_windows)
            B = be - bi

            ctx_batch = np.stack(all_contexts[bi:be])
            ctx = torch.from_numpy(ctx_batch).to(self.device)
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
            pred_np_by_seed = [p.float().cpu().numpy() for p in preds_by_seed] if return_seed_preds else []

            # Scatter results back to per-clip accumulators
            batch_clip_idx = all_clip_idx_np[bi:be]
            batch_starts = all_starts_np[bi:be]

            for j in range(B):
                clip_name = clip_names[batch_clip_idx[j]]
                acc = accums[clip_name]
                n_trs = clip_n_trs[clip_name]
                ts = batch_starts[j]
                for off in range(n_target_trs):
                    tr = ts + off
                    if tr >= n_trs:
                        break
                    acc["pred_sum"][tr] += pred_np[j, off]
                    acc["pred_count"][tr] += 1
                    if return_seed_preds:
                        for si in range(len(pred_np_by_seed)):
                            acc["seed_sums"][si][tr] += pred_np_by_seed[si][j, off]
                            acc["seed_counts"][si][tr] += 1

            del ctx_batch, ctx, subj, pred_agg, preds_by_seed
            torch.cuda.empty_cache()

        del all_contexts

        # --- Phase 4: Reduce accumulators to final predictions ---
        results = {}
        for clip_name in clip_names:
            acc = accums[clip_name]
            n_trs = clip_n_trs[clip_name]
            valid = acc["pred_count"] > 0
            pred_avg = np.zeros((n_trs, self.n_voxels), dtype=np.float32)
            pred_avg[valid] = (acc["pred_sum"][valid] / acc["pred_count"][valid, None]).astype(np.float32)

            if not return_seed_preds:
                results[clip_name] = (pred_avg, valid)
            else:
                preds_seed_avg = []
                for si in range(effective_seeds):
                    s_valid = acc["seed_counts"][si] > 0
                    s_avg = np.zeros((n_trs, self.n_voxels), dtype=np.float32)
                    s_avg[s_valid] = (acc["seed_sums"][si][s_valid] / acc["seed_counts"][si][s_valid, None]).astype(np.float32)
                    preds_seed_avg.append(s_avg)
                results[clip_name] = (pred_avg, valid, preds_seed_avg)

        return results

    @torch.inference_mode()
    def run_all_clips_multisubject(
        self,
        clip_windows: dict[str, list[dict]],
        clip_n_trs_per_subject: dict[str, dict[str, int]],
        subject_ids_list: list[int],
        n_target_trs: int,
        batch_size: int,
        solver: SolverConfig,
        encode_batch_size: int | None = None,
        desc: str = "",
    ) -> dict[str, dict[str, tuple]]:
        """Multi-subject parallel inference: encode context ONCE, decode for ALL subjects.

        Phase 1: Pre-encode all context windows (subject-independent).
        Phase 2: For each decode batch, replicate encoded context × N_subjects,
                 run ODE with mixed subject_ids, scatter results to per-subject accumulators.

        Args:
            clip_windows:           {clip_name: [windows]} — shared across subjects.
            clip_n_trs_per_subject: {subject_name: {clip_name: n_trs}} per subject.
            subject_ids_list:       list of integer subject indices to decode simultaneously.
            n_target_trs:           Number of target TRs per window.
            batch_size:             GPU batch size for decode (per-subject, total = batch_size × n_subjects).
            solver:                 ODE solver config.
            encode_batch_size:      GPU batch size for context encoding (default: batch_size).
            desc:                   tqdm description string.

        Returns:
            {subject_name: {clip_name: (pred_avg, valid)}}
        """
        synth_kw = solver.as_synth_kwargs()
        strategy = solver.as_strategy_config()
        n_subjects = len(subject_ids_list)
        subjects = list(clip_n_trs_per_subject.keys())

        if encode_batch_size is None:
            encode_batch_size = batch_size

        # --- Phase 1: Flatten all windows ---
        clip_names = list(clip_windows.keys())
        all_contexts = []
        all_starts = []
        all_clip_idx = []
        clip_name_to_idx = {c: i for i, c in enumerate(clip_names)}

        for clip_name in clip_names:
            for w in clip_windows[clip_name]:
                all_contexts.append(w["context"])
                all_starts.append(w["target_start"])
                all_clip_idx.append(clip_name_to_idx[clip_name])

        total_windows = len(all_contexts)
        if total_windows == 0:
            return {s: {} for s in subjects}

        all_starts_np = np.array(all_starts, dtype=np.int64)
        all_clip_idx_np = np.array(all_clip_idx, dtype=np.int64)
        del all_starts, all_clip_idx

        # --- Phase 2: Pre-encode all context (subject-independent) ---
        log.info("  Phase 1: Pre-encoding %d windows (batch_size=%d)...",
                 total_windows, encode_batch_size)
        # Encode in batches, store on CPU to save VRAM
        all_encoded = []
        for bi in tqdm(range(0, total_windows, encode_batch_size),
                       desc="encoding", total=(total_windows + encode_batch_size - 1) // encode_batch_size):
            be = min(bi + encode_batch_size, total_windows)
            ctx_batch = np.stack(all_contexts[bi:be])
            ctx = torch.from_numpy(ctx_batch).to(self.device)
            encoded = self.model.velocity_net.encode_context_from_cond(ctx)
            all_encoded.append(encoded.cpu())
            del ctx, encoded
            torch.cuda.empty_cache()

        all_encoded_cat = torch.cat(all_encoded, dim=0)  # (total_windows, n_target_trs, hidden_dim)
        del all_encoded, all_contexts
        log.info("  Encoded context: shape=%s (%.1f MB on CPU)",
                 list(all_encoded_cat.shape),
                 all_encoded_cat.element_size() * all_encoded_cat.numel() / 1e6)

        # --- Phase 3: Allocate per-subject, per-clip accumulators ---
        accums = {}  # {subject_name: {clip_name: {pred_sum, pred_count}}}
        for subj_name in subjects:
            accums[subj_name] = {}
            for clip_name in clip_names:
                n_trs = clip_n_trs_per_subject[subj_name].get(clip_name, 0)
                if n_trs > 0:
                    accums[subj_name][clip_name] = {
                        "pred_sum": np.zeros((n_trs, self.n_voxels), dtype=np.float64),
                        "pred_count": np.zeros(n_trs, dtype=np.float64),
                    }

        # --- Phase 4: Decode in batches for ALL subjects simultaneously ---
        # Per decode batch: take `batch_size` windows, replicate × n_subjects
        decode_batch = max(1, batch_size // n_subjects)  # per-subject windows per GPU call
        n_batches = (total_windows + decode_batch - 1) // decode_batch
        log.info("  Phase 2: Decoding %d windows × %d subjects (decode_batch=%d, %d GPU calls)...",
                 total_windows, n_subjects, decode_batch, n_batches)

        pbar = tqdm(range(0, total_windows, decode_batch), desc=desc or "multi-subj",
                    total=n_batches)

        for bi in pbar:
            be = min(bi + decode_batch, total_windows)
            W = be - bi  # windows in this batch

            # Get pre-encoded context for this batch and replicate for each subject
            enc_batch = all_encoded_cat[bi:be].to(self.device)  # (W, T_enc, H)

            # Replicate: (W*n_subjects, T_enc, H)
            enc_rep = enc_batch.repeat(n_subjects, 1, 1)

            # Build subject_ids: [s0,s0,...,s1,s1,...,s2,s2,...,s3,s3,...]
            subj_ids = torch.cat([
                torch.full((W,), sid, dtype=torch.long, device=self.device)
                for sid in subject_ids_list
            ])

            # Run multi-seed synthesis with pre-encoded context
            pred_agg, _ = run_multiseed_synthesis(
                model=self.model,
                context=None,
                subject_ids=subj_ids,
                synth_kwargs=synth_kw,
                strategy=strategy,
                parcel_seed_map=None,
                pre_encoded_context=enc_rep,
            )
            pred_np = pred_agg.float().cpu().numpy()  # (W*n_subjects, T_target, V)

            # Scatter results to per-subject accumulators
            batch_clip_idx = all_clip_idx_np[bi:be]
            batch_starts = all_starts_np[bi:be]

            for si, subj_name in enumerate(subjects):
                pred_subj = pred_np[si * W : (si + 1) * W]  # (W, T_target, V)
                for j in range(W):
                    clip_name = clip_names[batch_clip_idx[j]]
                    if clip_name not in accums[subj_name]:
                        continue
                    acc = accums[subj_name][clip_name]
                    n_trs = clip_n_trs_per_subject[subj_name].get(clip_name, 0)
                    ts = batch_starts[j]
                    for off in range(n_target_trs):
                        tr = ts + off
                        if tr >= n_trs:
                            break
                        acc["pred_sum"][tr] += pred_subj[j, off]
                        acc["pred_count"][tr] += 1

            del enc_batch, enc_rep, subj_ids, pred_agg
            torch.cuda.empty_cache()

        del all_encoded_cat

        # --- Phase 5: Reduce to final predictions ---
        results = {}
        for subj_name in subjects:
            results[subj_name] = {}
            for clip_name in clip_names:
                if clip_name not in accums[subj_name]:
                    continue
                acc = accums[subj_name][clip_name]
                n_trs = clip_n_trs_per_subject[subj_name][clip_name]
                valid = acc["pred_count"] > 0
                pred_avg = np.zeros((n_trs, self.n_voxels), dtype=np.float32)
                pred_avg[valid] = (acc["pred_sum"][valid] / acc["pred_count"][valid, None]).astype(np.float32)
                results[subj_name][clip_name] = (pred_avg, valid)

        return results
