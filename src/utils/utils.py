"""Utility helpers for BrainFlow inference-time strategies.

This module provides reusable pieces for:
1) Strategy 2: pruned sampling from multiple start candidates.
2) Strategy 4: multi-seed ensembling and parcel-wise stitching.
3) Calibration artifacts used to transfer parcel seed choices from S6 to S7/OOD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InferenceStrategyConfig:
	"""Runtime controls for inference-time strategy extensions."""

	use_pruned_sampling: bool = False
	prune_k: int = 5
	n_seeds: int = 1
	ensemble_mode: str = "none"  # none | mean | max | parcel_stitch
	base_seed: int = 1234


def _to_device_generator(device: torch.device, seed: int) -> torch.Generator:
	if device.type == "cuda":
		g = torch.Generator(device="cuda")
	else:
		g = torch.Generator()
	g.manual_seed(int(seed))
	return g


@torch.inference_mode()
def build_regression_anchor(
	model,
	context: torch.Tensor,
	subject_ids: torch.Tensor,
) -> torch.Tensor:
	"""Build sequence-shaped anchor from BrainFlow regression branch.

	Returns:
		anchor: (B, n_target_trs, output_dim) for seq2seq models.
	"""
	context_encoded = model.velocity_net.encode_context_from_cond(context)
	pooled = context_encoded.mean(dim=1)
	hidden = model.reg_head(pooled)
	latent = model.reg_output(hidden)

	if model.velocity_net.use_subject_head:
		anchor_2d = model.velocity_net.subject_layers(latent, subject_ids)
	else:
		anchor_2d = latent

	n_target = int(getattr(model.velocity_net, "n_target_trs", 1))
	if n_target > 1:
		return anchor_2d.unsqueeze(1).expand(-1, n_target, -1)
	return anchor_2d


@torch.inference_mode()
def build_pruned_starting_distribution(
	model,
	context: torch.Tensor,
	subject_ids: torch.Tensor,
	temperature: float,
	prune_k: int,
	seed: int,
) -> torch.Tensor:
	"""Create x0 via candidate pruning by cosine similarity to regression anchor.

	Candidate selection is performed per sample in the batch.
	"""
	anchor = build_regression_anchor(model, context, subject_ids)

	if temperature <= 0:
		return anchor.clone()

	device = anchor.device
	dtype = anchor.dtype
	g = _to_device_generator(device, seed)
	k = max(1, int(prune_k))

	if k == 1:
		noise = torch.randn(anchor.shape, generator=g, device=device, dtype=dtype)
		return anchor + temperature * noise

	candidates = []
	anchor_flat = anchor.reshape(anchor.shape[0], -1)
	for _ in range(k):
		noise = torch.randn(anchor.shape, generator=g, device=device, dtype=dtype)
		candidates.append(anchor + temperature * noise)

	cands = torch.stack(candidates, dim=0)  # (K, B, ...)
	sims = []
	for i in range(k):
		sim_i = F.cosine_similarity(
			cands[i].reshape(anchor.shape[0], -1),
			anchor_flat,
			dim=1,
			eps=1e-8,
		)
		sims.append(sim_i)
	sim_mat = torch.stack(sims, dim=0)  # (K, B)
	best_idx = sim_mat.argmax(dim=0)  # (B,)

	selected = cands[best_idx, torch.arange(anchor.shape[0], device=device)]
	return selected


@torch.inference_mode()
def run_multiseed_synthesis(
	model,
	context: torch.Tensor,
	subject_ids: torch.Tensor,
	synth_kwargs: dict,
	strategy: InferenceStrategyConfig,
	parcel_seed_map: np.ndarray | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
	"""Run multi-seed synthesis and aggregate prediction per strategy mode.

	Key optimizations:
	  1. Context is encoded ONCE and reused across all seeds (avoids N×encoder cost).
	  2. When pruned_sampling is disabled, all seeds are batched into a single
	     forward pass (S×B batch) so the GPU processes everything in parallel.

	Returns:
		pred_agg: aggregated prediction tensor (B, T, V) or (B, V).
		preds_by_seed: list of per-seed tensors, each (B, T, V) or (B, V).
	"""
	mode = (strategy.ensemble_mode or "none").lower()
	seeds = max(1, int(strategy.n_seeds))
	if mode in ("none", "single"):
		seeds = 1

	# ── Optimization 1: encode context once, reuse for all seeds ──────────────
	with torch.inference_mode():
		context_encoded = model.velocity_net.encode_context_from_cond(context)

	preds_by_seed: list[torch.Tensor] = []

	# ── Optimization 2: batch all seeds into one forward pass ─────────────────
	# Only valid when pruned_sampling is disabled (no anchor dependency per seed)
	if seeds > 1 and not strategy.use_pruned_sampling and mode not in ("none", "single"):
		B = context.shape[0]
		# Replicate context_encoded and subject_ids S times
		ctx_enc_rep = context_encoded.repeat(seeds, 1, 1)   # (S*B, T, D)
		subj_rep    = subject_ids.repeat(seeds)              # (S*B,)

		# Build S different noise starts with different seeds
		noise_list = []
		for seed_offset in range(seeds):
			seed = int(strategy.base_seed + seed_offset)
			g = _to_device_generator(context.device, seed)
			if model.use_csfm:
				# CSFM: use mu_phi + sigma * noise — compute per seed
				# Fall back to sequential for CSFM (state-dependent start)
				noise_list = None
				break
			n_target = getattr(model.velocity_net, 'n_target_trs', 1)
			temp = float(synth_kwargs.get("temperature", 0.05))
			shape = (B, n_target, model.output_dim) if n_target > 1 else (B, model.output_dim)
			noise_list.append(temp * torch.randn(*shape, generator=g,
			                                    device=context.device, dtype=context.dtype))

		if noise_list is not None:
			# Stack noise: (S*B, ...) and run ONE batched ODE solve
			x0_batch = torch.cat(noise_list, dim=0)  # (S*B, T, V)
			kw = dict(synth_kwargs)
			kw["starting_distribution"] = x0_batch
			kw["temperature"] = 0.0  # noise already baked in
			pred_batch = model.synthesise(
				context.repeat(seeds, 1, 1),
				subject_ids=subj_rep,
				pre_encoded_context=ctx_enc_rep,
				**kw,
			)
			# Split back into per-seed predictions
			for s in range(seeds):
				preds_by_seed.append(pred_batch[s * B:(s + 1) * B])

			if mode == "mean":
				return torch.stack(preds_by_seed, dim=0).mean(dim=0), preds_by_seed
			if mode == "max":
				return torch.stack(preds_by_seed, dim=0).max(dim=0)[0], preds_by_seed
			if mode == "parcel_stitch":
				if parcel_seed_map is None:
					return torch.stack(preds_by_seed, dim=0).mean(dim=0), preds_by_seed
				return stitch_predictions_by_seed_map(preds_by_seed, parcel_seed_map), preds_by_seed

	# ── Fallback: sequential seeds (CSFM mode or pruned sampling) ────────────
	# Still benefits from Optimization 1 (context_encoded cached above)
	for seed_offset in range(seeds):
		seed = int(strategy.base_seed + seed_offset)
		if strategy.use_pruned_sampling:
			start = build_pruned_starting_distribution(
				model=model,
				context=context,
				subject_ids=subject_ids,
				temperature=float(synth_kwargs.get("temperature", 0.0)),
				prune_k=int(strategy.prune_k),
				seed=seed,
			)
			kw = dict(synth_kwargs)
			kw["starting_distribution"] = start
			kw["temperature"] = 0.0
			pred = model.synthesise(
				context, subject_ids=subject_ids,
				pre_encoded_context=context_encoded,
				**kw,
			)
		else:
			torch.manual_seed(seed)
			if context.device.type == "cuda":
				torch.cuda.manual_seed_all(seed)
			pred = model.synthesise(
				context, subject_ids=subject_ids,
				pre_encoded_context=context_encoded,  # ← reuse cached encoding
				**synth_kwargs,
			)
		preds_by_seed.append(pred)

	if seeds == 1 or mode in ("none", "single"):
		return preds_by_seed[0], preds_by_seed

	if mode == "mean":
		return torch.stack(preds_by_seed, dim=0).mean(dim=0), preds_by_seed

	if mode == "max":
		return torch.stack(preds_by_seed, dim=0).max(dim=0)[0], preds_by_seed

	if mode == "parcel_stitch":
		if parcel_seed_map is None:
			return torch.stack(preds_by_seed, dim=0).mean(dim=0), preds_by_seed
		stitched = stitch_predictions_by_seed_map(preds_by_seed, parcel_seed_map)
		return stitched, preds_by_seed

	raise ValueError(f"Unsupported ensemble_mode: {strategy.ensemble_mode!r}")


def stitch_predictions_by_seed_map(
	preds_by_seed: list[torch.Tensor],
	parcel_seed_map: np.ndarray,
) -> torch.Tensor:
	"""Stitch prediction by voxel/parcel index using best-seed map.

	Args:
		preds_by_seed: list length S with each tensor (B, T, V) or (B, V).
		parcel_seed_map: (V,) integer seed index for each output parcel.
	"""
	if not preds_by_seed:
		raise ValueError("preds_by_seed must not be empty")

	arr = np.asarray(parcel_seed_map, dtype=np.int64)
	if arr.ndim != 1:
		raise ValueError("parcel_seed_map must have shape (V,)")

	base = preds_by_seed[0]
	if base.dim() == 2:
		# Normalize to (B, T=1, V) then squeeze back.
		pred_stack = torch.stack([p.unsqueeze(1) for p in preds_by_seed], dim=0)
		squeeze_t = True
	elif base.dim() == 3:
		pred_stack = torch.stack(preds_by_seed, dim=0)
		squeeze_t = False
	else:
		raise ValueError("Only 2D/3D prediction tensors are supported")

	# pred_stack: (S, B, T, V)
	s, b, t, v = pred_stack.shape
	if arr.shape[0] != v:
		raise ValueError(f"parcel_seed_map length {arr.shape[0]} != output dim {v}")

	seed_idx = torch.from_numpy(arr).to(pred_stack.device)
	seed_idx = seed_idx.clamp(min=0, max=s - 1)

	# Gather per-voxel from chosen seed: index on seed axis for each voxel.
	out = torch.empty((b, t, v), device=pred_stack.device, dtype=pred_stack.dtype)
	for voxel in range(v):
		out[..., voxel] = pred_stack[seed_idx[voxel], :, :, voxel]

	if squeeze_t:
		return out[:, 0, :]
	return out


def per_voxel_pcc(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
	"""Compute per-voxel Pearson correlation (V,)."""
	p = pred - pred.mean(axis=0, keepdims=True)
	t = target - target.mean(axis=0, keepdims=True)
	cov = (p * t).sum(axis=0)
	std = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
	return cov / (std + 1e-8)


def best_seed_map_from_s6(
	seed_preds: list[np.ndarray],
	target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Derive per-parcel best seed index from S6 targets.

	Args:
		seed_preds: list of arrays [(N, V), ...] where N is concatenated valid TRs.
		target:     array (N, V) concatenated valid TRs.

	Returns:
		best_seed_map: (V,) best seed index per parcel.
		pcc_by_seed:   (S, V) PCC table.
	"""
	if not seed_preds:
		raise ValueError("seed_preds must not be empty")

	pcc_table = np.stack([per_voxel_pcc(p, target) for p in seed_preds], axis=0)
	best = np.argmax(pcc_table, axis=0).astype(np.int64)
	return best, pcc_table


def calibration_path(output_dir: Path) -> Path:
	"""Standard path for S6 parcel calibration artifact."""
	return output_dir / "eval_s6" / "parcel_seed_calibration.npy"


def save_parcel_calibration(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, payload)


def load_parcel_calibration(path: Path) -> dict | None:
	if not path.exists():
		return None
	return np.load(path, allow_pickle=True).item()

