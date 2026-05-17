"""Sweep CFG scale + ODE steps + TTA temperatures on a trained BrainFlow checkpoint.

Usage:
    python scripts/eval_with_cfg.py \\
        --config outputs/brainflow_dit_large_80/config.yaml \\
        --ckpt   outputs/brainflow_dit_large_80/best.pt \\
        --cfg-scales 0.0 1.0 1.5 2.0 \\
        --time-points 30 50 \\
        --tta '0.0,0.05,0.1,0.15'

Writes a CSV table of PCC for each (cfg_scale, time_points, tta) combo to
``--out`` (default: alongside the checkpoint).
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.directflow_dataset import load_config, get_dataloaders
from src.models.brainflow.brainflow import BrainFlow
from src.train_brainflow import pearson_corr_per_dim, resolve_paths

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_with_cfg")


def parse_tta(spec: str | None):
    if not spec:
        return None
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    return [float(p) for p in parts] if parts else None


def build_model_and_loader(cfg_path: Path, ckpt_path: Path, device):
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, PROJECT_ROOT)

    _, val_loader = get_dataloaders(cfg)

    bf_cfg = cfg["brainflow"]
    output_dim = bf_cfg.get("output_dim", cfg["fmri"]["n_voxels"])
    vn_params = dict(bf_cfg.get("velocity_net", {}))
    modality_dims = cfg.get("modality_dims", None)
    if modality_dims:
        vn_params["modality_dims"] = modality_dims

    model = BrainFlow(
        output_dim=output_dim,
        velocity_net_params=vn_params,
        n_subjects=len(cfg["subjects"]),
        tensor_fm_params=bf_cfg.get("tensor_fm", None),
        indi_flow_matching=bf_cfg.get("indi_flow_matching", False),
        indi_train_time_sqrt=bf_cfg.get("indi_train_time_sqrt", False),
        indi_min_denom=bf_cfg.get("indi_min_denom", 1e-3),
        use_csfm=bf_cfg.get("use_csfm", False),
        csfm_var_reg_weight=bf_cfg.get("csfm_var_reg_weight", 0.1),
        csfm_pcc_weight=bf_cfg.get("csfm_pcc_weight", 1.0),
        flow_loss_weight=bf_cfg.get("flow_loss_weight", 1.0),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys: %d (first: %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys: %d (first: %s)", len(unexpected), unexpected[:3])
    model.eval()
    return cfg, model, val_loader


@torch.no_grad()
def evaluate(model, val_loader, device, *, cfg_scale, time_points, tta_temps,
             solver_method, time_grid_warp, base_temperature):
    fmri_pred_acc, fmri_tgt_acc = {}, {}

    for batch in tqdm(val_loader, desc=f"cfg={cfg_scale} T={time_points} tta={tta_temps}"):
        context = batch["context"].to(device)
        subject_ids = batch["subject_idx"].to(device)
        fmri_target = batch["fmri"].to(device)

        synth_kwargs = dict(
            n_timesteps=time_points,
            solver_method=solver_method,
            subject_ids=subject_ids,
            temperature=base_temperature,
        )
        if time_grid_warp:
            synth_kwargs["time_grid_warp"] = time_grid_warp
        if cfg_scale > 0:
            synth_kwargs["cfg_scale"] = cfg_scale

        if tta_temps:
            gen_acc = None
            for tta_t in tta_temps:
                synth_kwargs["temperature"] = float(tta_t)
                g = model.synthesise(context, **synth_kwargs)
                gen_acc = g if gen_acc is None else gen_acc + g
            gen_fmri = gen_acc / len(tta_temps)
        else:
            gen_fmri = model.synthesise(context, **synth_kwargs)

        clip_keys = batch["clip_key"]
        target_tr_starts = batch["target_tr_start"]
        n_trs_batch = batch["n_trs"]

        B = gen_fmri.shape[0]
        fmri_dim = gen_fmri.shape[-1]
        is_seq = gen_fmri.dim() == 3
        for b in range(B):
            ck = clip_keys[b]
            tr_start = int(target_tr_starts[b].item())
            n_t = int(n_trs_batch[b].item())

            if ck not in fmri_pred_acc:
                fmri_pred_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}
                fmri_tgt_acc[ck] = {"sum": torch.zeros(n_t, fmri_dim), "count": torch.zeros(n_t)}

            if is_seq:
                for off in range(gen_fmri.shape[1]):
                    tr_idx = tr_start + off
                    if tr_idx >= n_t:
                        break
                    fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b, off].cpu()
                    fmri_pred_acc[ck]["count"][tr_idx] += 1
                    fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b, off].cpu()
                    fmri_tgt_acc[ck]["count"][tr_idx] += 1
            else:
                tr_idx = tr_start
                if tr_idx < n_t:
                    fmri_pred_acc[ck]["sum"][tr_idx] += gen_fmri[b].cpu()
                    fmri_pred_acc[ck]["count"][tr_idx] += 1
                    fmri_tgt_acc[ck]["sum"][tr_idx] += fmri_target[b].cpu()
                    fmri_tgt_acc[ck]["count"][tr_idx] += 1

    all_gen, all_tgt = [], []
    for ck in sorted(fmri_pred_acc.keys()):
        count = fmri_pred_acc[ck]["count"]
        valid = count > 0
        if valid.sum() < 2:
            continue
        all_gen.append(fmri_pred_acc[ck]["sum"][valid] / count[valid].unsqueeze(-1))
        all_tgt.append(fmri_tgt_acc[ck]["sum"][valid] / fmri_tgt_acc[ck]["count"][valid].unsqueeze(-1))

    if not all_gen:
        return float("nan")
    all_gen = torch.cat(all_gen, dim=0)
    all_tgt = torch.cat(all_tgt, dim=0)
    pcc = pearson_corr_per_dim(all_gen.unsqueeze(0), all_tgt.unsqueeze(0))
    return float(pcc.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--cfg-scales", nargs="+", type=float, default=[0.0, 1.0, 1.5, 2.0])
    p.add_argument("--time-points", nargs="+", type=int, default=[30, 50])
    p.add_argument(
        "--tta", type=str, default="none",
        help="Comma-separated temperatures, or 'none' to disable TTA. "
             "Pass multiple --tta-variants (repeat) to sweep multiple TTA sets.",
    )
    p.add_argument("--tta-variants", action="append", default=None,
                   help="Repeat to sweep multiple TTA temperature sets, e.g. --tta-variants none --tta-variants 0.0,0.05,0.1,0.15")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = Path(args.config).resolve()
    ckpt_path = Path(args.ckpt).resolve()

    cfg, model, val_loader = build_model_and_loader(cfg_path, ckpt_path, device)

    solver_cfg = cfg.get("solver_args", {})
    solver_method = solver_cfg.get("method", "midpoint")
    time_grid_warp = solver_cfg.get("time_grid_warp")
    base_temperature = solver_cfg.get("temperature", 0.05)

    if args.tta_variants:
        tta_specs = [None if v.lower() == "none" else parse_tta(v) for v in args.tta_variants]
    else:
        tta_specs = [None if args.tta.lower() == "none" else parse_tta(args.tta)]

    out_path = Path(args.out) if args.out else ckpt_path.with_name(
        ckpt_path.stem + "_cfg_sweep.csv"
    )

    rows = []
    for tta in tta_specs:
        for tp in args.time_points:
            for cs in args.cfg_scales:
                pcc = evaluate(
                    model, val_loader, device,
                    cfg_scale=cs, time_points=tp, tta_temps=tta,
                    solver_method=solver_method,
                    time_grid_warp=time_grid_warp,
                    base_temperature=base_temperature,
                )
                tta_str = "none" if tta is None else ",".join(f"{t:g}" for t in tta)
                logger.info(
                    "cfg_scale=%.2f  time_points=%d  tta=%s  →  PCC=%.4f",
                    cs, tp, tta_str, pcc,
                )
                rows.append({
                    "cfg_scale": cs,
                    "time_points": tp,
                    "tta": tta_str,
                    "pcc": f"{pcc:.6f}",
                })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cfg_scale", "time_points", "tta", "pcc"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote sweep results to %s", out_path)

    best = max(rows, key=lambda r: float(r["pcc"]))
    logger.info("Best combo: cfg=%s tp=%s tta=%s  →  PCC=%s",
                best["cfg_scale"], best["time_points"], best["tta"], best["pcc"])


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
