#!/usr/bin/env python3
"""
Comprehensive fMRI VAE Latent Space Analysis
=============================================
Evaluates the quality of the learned latent representation from fMRI_VAE_v5.

Analyses:
  1. Reconstruction Quality — Per-voxel & per-TR Pearson, MSE distribution
  2. Latent Distribution   — Gaussianity, KL vs N(0,1), mu/logvar stats
  3. Feature Diversity     — Effective rank, correlation matrix, dead units
  4. Temporal Structure    — Autocorrelation, smoothness in latent space
  5. Visualization         — PCA/t-SNE of latent codes colored by clip

Usage:
  python src/analyze_vae_latent.py --checkpoint outputs/fmri_vae
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, normaltest, kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Setup ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="outputs/fmri_vae",
                   help="Path to VAE output directory (containing best.pt & config.yaml)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_clips", type=int, default=50,
                   help="Max clips to analyze (for speed)")
    return p.parse_args()


def load_model_and_config(ckpt_dir: Path, device: str):
    """Load model from checkpoint."""
    config_path = ckpt_dir / "config.yaml"
    ckpt_path = ckpt_dir / "best.pt"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Add required internal paths
    project_root = Path(cfg.get("_project_root", "."))
    sys.path.insert(0, str(project_root))

    from src.models.brainflow.fmri_vae import build_vae

    model = build_vae(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Use EMA weights if available
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model = model.to(device).eval()
    print(f"Loaded {state_key} from {ckpt_path}")
    print(f"  Model: {model}")
    return model, cfg


def load_all_clips(cfg: dict, split: str = "val", max_clips: int = 50):
    """Load fMRI clips directly from H5 files."""
    import hashlib, random
    sys.path.insert(0, str(Path(cfg.get("_project_root", "."))))
    from src.train_fmri_vae import _fmri_path

    fmri_dir = Path(cfg["_fmri_dir"])
    subjects = cfg["subjects"]
    splits_cfg = cfg.get("splits", {})
    excl_start = cfg["fmri"].get("excluded_samples_start", 0)
    excl_end = cfg["fmri"].get("excluded_samples_end", 0)
    val_ratio = cfg.get("val_ratio", 0.1)

    clips = []
    for subject in subjects:
        for task in splits_cfg.keys():
            fmri_path = _fmri_path(str(fmri_dir), subject, task)
            if not fmri_path.exists():
                print(f"  [skip] {fmri_path}")
                continue
            try:
                with h5py.File(fmri_path, "r") as f:
                    for h5_key in f.keys():
                        uid = f"{subject}_{task}_{h5_key}"
                        h = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
                        s = random.Random(h).random()
                        clip_split = "val" if s >= (1.0 - val_ratio) else "train"
                        if clip_split != split:
                            continue

                        raw = f[h5_key][:]
                        end = len(raw) - excl_end if excl_end > 0 else len(raw)
                        data = raw[excl_start:end].astype(np.float32)
                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                        if data.shape[0] < 10:
                            continue
                        clips.append({
                            "uid": uid,
                            "data": data,    # (T, V)
                            "subject": subject,
                        })
                        if len(clips) >= max_clips:
                            return clips
            except Exception as e:
                print(f"Warning: cannot open {fmri_path}: {e}")
    return clips


# ── Analysis Functions ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(model, clips, device, seq_len=100):
    """Extract mu, logvar, and reconstruction for all clips."""
    results = []
    for clip in clips:
        data = clip["data"]  # (T_clip, V)
        T_clip = data.shape[0]

        all_mu, all_logvar, all_recon, all_orig = [], [], [], []

        # Process in non-overlapping windows
        for start in range(0, T_clip - seq_len + 1, seq_len):
            window = data[start:start + seq_len]  # (T, V)
            x = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, T, V)
            sid = torch.zeros(1, dtype=torch.long, device=device)

            z, mu, logvar = model.encode(x, sid)
            recon = model.decode(mu, sid)  # Use mu (deterministic)

            all_mu.append(mu.cpu().numpy()[0])          # (T, Z)
            all_logvar.append(logvar.cpu().numpy()[0])   # (T, Z)
            all_recon.append(recon.cpu().numpy()[0])      # (T, V)
            all_orig.append(window)

        if not all_mu:
            continue

        results.append({
            "uid": clip["uid"],
            "mu": np.concatenate(all_mu, axis=0),
            "logvar": np.concatenate(all_logvar, axis=0),
            "recon": np.concatenate(all_recon, axis=0),
            "orig": np.concatenate(all_orig, axis=0),
        })
    return results


def analyze_reconstruction(results, save_dir):
    """Analysis 1: Reconstruction quality."""
    print("\n" + "="*70)
    print("1. RECONSTRUCTION QUALITY")
    print("="*70)

    all_pcc_per_voxel = []
    all_pcc_per_tr = []
    all_mse = []

    for r in results:
        orig, recon = r["orig"], r["recon"]  # (T, V)
        T, V = orig.shape

        # Per-voxel Pearson (temporal correlation per voxel)
        for v in range(V):
            rr, _ = pearsonr(orig[:, v], recon[:, v])
            if np.isfinite(rr):
                all_pcc_per_voxel.append(rr)

        # Per-TR Pearson (spatial correlation per timepoint)
        for t in range(T):
            rr, _ = pearsonr(orig[t, :], recon[t, :])
            if np.isfinite(rr):
                all_pcc_per_tr.append(rr)

        # MSE per TR
        mse = np.mean((orig - recon) ** 2, axis=1)  # (T,)
        all_mse.extend(mse.tolist())

    pcc_voxel = np.array(all_pcc_per_voxel)
    pcc_tr = np.array(all_pcc_per_tr)
    mse_arr = np.array(all_mse)

    print(f"  Per-Voxel Pearson:  mean={pcc_voxel.mean():.4f}  std={pcc_voxel.std():.4f}  "
          f"median={np.median(pcc_voxel):.4f}  min={pcc_voxel.min():.4f}")
    print(f"  Per-TR Pearson:     mean={pcc_tr.mean():.4f}  std={pcc_tr.std():.4f}  "
          f"median={np.median(pcc_tr):.4f}  min={pcc_tr.min():.4f}")
    print(f"  MSE per TR:         mean={mse_arr.mean():.4f}  std={mse_arr.std():.4f}")

    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(pcc_voxel, bins=50, alpha=0.7, edgecolor="black", color="#2196F3")
    axes[0].axvline(pcc_voxel.mean(), color="red", linestyle="--", label=f"mean={pcc_voxel.mean():.3f}")
    axes[0].set_title("Per-Voxel Pearson (Temporal)")
    axes[0].set_xlabel("Pearson r")
    axes[0].legend()

    axes[1].hist(pcc_tr, bins=50, alpha=0.7, edgecolor="black", color="#4CAF50")
    axes[1].axvline(pcc_tr.mean(), color="red", linestyle="--", label=f"mean={pcc_tr.mean():.3f}")
    axes[1].set_title("Per-TR Pearson (Spatial)")
    axes[1].set_xlabel("Pearson r")
    axes[1].legend()

    axes[2].hist(mse_arr, bins=50, alpha=0.7, edgecolor="black", color="#FF9800")
    axes[2].axvline(mse_arr.mean(), color="red", linestyle="--", label=f"mean={mse_arr.mean():.4f}")
    axes[2].set_title("MSE Distribution per TR")
    axes[2].set_xlabel("MSE")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "01_reconstruction_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_dir / '01_reconstruction_quality.png'}")


def analyze_latent_distribution(results, save_dir):
    """Analysis 2: Latent space distribution."""
    print("\n" + "="*70)
    print("2. LATENT DISTRIBUTION")
    print("="*70)

    all_mu = np.concatenate([r["mu"] for r in results], axis=0)     # (N, Z)
    all_logvar = np.concatenate([r["logvar"] for r in results], axis=0)  # (N, Z)
    all_std = np.exp(0.5 * all_logvar)

    N, Z = all_mu.shape
    print(f"  Total latent samples: {N} × {Z} dims")

    # Per-dimension statistics
    mu_mean = all_mu.mean(axis=0)      # (Z,)
    mu_std = all_mu.std(axis=0)        # (Z,)
    std_mean = all_std.mean(axis=0)    # (Z,)

    print(f"\n  μ statistics:")
    print(f"    Global mean:   {all_mu.mean():.4f}")
    print(f"    Global std:    {all_mu.std():.4f}")
    print(f"    Per-dim mean range: [{mu_mean.min():.4f}, {mu_mean.max():.4f}]")
    print(f"    Per-dim std  range: [{mu_std.min():.4f}, {mu_std.max():.4f}]")

    print(f"\n  σ = exp(logvar/2) statistics:")
    print(f"    Global mean:   {all_std.mean():.4f}")
    print(f"    Per-dim range: [{std_mean.min():.4f}, {std_mean.max():.4f}]")

    # KL per dimension: KL(q||p) = -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + all_logvar - all_mu**2 - np.exp(all_logvar))
    kl_mean_per_dim = kl_per_dim.mean(axis=0)  # (Z,)
    print(f"\n  KL divergence per dim:")
    print(f"    Total KL:       {kl_mean_per_dim.sum():.4f}")
    print(f"    Active dims (KL > 0.1): {(kl_mean_per_dim > 0.1).sum()}/{Z}")
    print(f"    Dead dims   (KL < 0.01): {(kl_mean_per_dim < 0.01).sum()}/{Z}")

    # Gaussianity test on a few dimensions
    print(f"\n  Gaussianity (D'Agostino-Pearson test, first 10 dims):")
    for d in range(min(10, Z)):
        stat, p = normaltest(all_mu[:, d])
        sk = skew(all_mu[:, d])
        ku = kurtosis(all_mu[:, d])
        gaussian_str = "✓ Gaussian" if p > 0.05 else "✗ Non-Gaussian"
        print(f"    dim {d:3d}: skew={sk:+.3f} kurt={ku:+.3f} p={p:.2e} → {gaussian_str}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # μ histogram (aggregated)
    axes[0, 0].hist(all_mu.flatten(), bins=100, alpha=0.7, density=True,
                     edgecolor="black", color="#2196F3")
    axes[0, 0].set_title(f"μ Distribution (all dims, N={N})")
    axes[0, 0].set_xlabel("μ")

    # σ histogram
    axes[0, 1].hist(all_std.flatten(), bins=100, alpha=0.7, density=True,
                     edgecolor="black", color="#4CAF50")
    axes[0, 1].set_title("σ = exp(logvar/2) Distribution")
    axes[0, 1].set_xlabel("σ")

    # KL per dimension (sorted)
    sorted_kl = np.sort(kl_mean_per_dim)[::-1]
    axes[0, 2].bar(range(Z), sorted_kl, color="#FF5722", alpha=0.7)
    axes[0, 2].axhline(0.1, color="gray", linestyle="--", alpha=0.5, label="KL=0.1 threshold")
    axes[0, 2].set_title("KL Divergence per Latent Dim (sorted)")
    axes[0, 2].set_xlabel("Latent dim (sorted)")
    axes[0, 2].set_ylabel("KL")
    axes[0, 2].legend()

    # Per-dim mean of μ
    axes[1, 0].bar(range(Z), mu_mean, color="#673AB7", alpha=0.7)
    axes[1, 0].set_title("Per-dim Mean of μ")
    axes[1, 0].set_xlabel("Latent dim")

    # Per-dim std of μ (activity level)
    axes[1, 1].bar(range(Z), mu_std, color="#009688", alpha=0.7)
    axes[1, 1].axhline(0.01, color="red", linestyle="--", alpha=0.5, label="Dead threshold")
    axes[1, 1].set_title("Per-dim Std of μ (Activity Level)")
    axes[1, 1].set_xlabel("Latent dim")
    axes[1, 1].legend()

    # Per-dim mean of σ
    axes[1, 2].bar(range(Z), std_mean, color="#FF9800", alpha=0.7)
    axes[1, 2].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="σ=1 (prior)")
    axes[1, 2].set_title("Per-dim Mean of σ (Posterior Width)")
    axes[1, 2].set_xlabel("Latent dim")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "02_latent_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_dir / '02_latent_distribution.png'}")


def analyze_feature_diversity(results, save_dir):
    """Analysis 3: Feature diversity and redundancy."""
    print("\n" + "="*70)
    print("3. FEATURE DIVERSITY")
    print("="*70)

    all_mu = np.concatenate([r["mu"] for r in results], axis=0)  # (N, Z)
    N, Z = all_mu.shape

    # Correlation matrix between latent dims
    corr_matrix = np.corrcoef(all_mu.T)  # (Z, Z)

    # Effective rank via singular values
    U, S, Vt = np.linalg.svd(all_mu - all_mu.mean(axis=0), full_matrices=False)
    S_norm = S / S.sum()
    effective_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))

    # Variance explained by top-k PCs
    var_explained = (S ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)

    # Dead unit analysis
    dim_std = all_mu.std(axis=0)
    dead_units = (dim_std < 0.01).sum()
    low_activity = (dim_std < 0.1).sum()

    print(f"  Effective Rank:       {effective_rank:.1f} / {Z}")
    print(f"  Top-10 PCs explain:   {cumvar[9]*100:.1f}%")
    print(f"  Top-50 PCs explain:   {cumvar[min(49,Z-1)]*100:.1f}%")
    print(f"  Dead units (std<0.01): {dead_units}/{Z}")
    print(f"  Low activity (std<0.1): {low_activity}/{Z}")

    # Off-diagonal correlation stats
    mask = ~np.eye(Z, dtype=bool)
    off_diag = np.abs(corr_matrix[mask])
    print(f"  Correlation (off-diag): mean={off_diag.mean():.4f}  max={off_diag.max():.4f}")
    print(f"    Highly correlated pairs (|r|>0.5): {(off_diag > 0.5).sum() // 2}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Correlation matrix
    im = axes[0].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title("Latent Correlation Matrix")
    axes[0].set_xlabel("Latent dim")
    axes[0].set_ylabel("Latent dim")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Cumulative variance
    axes[1].plot(range(1, Z + 1), cumvar * 100, color="#2196F3", linewidth=2)
    axes[1].axhline(90, color="gray", linestyle="--", alpha=0.5)
    axes[1].axhline(95, color="gray", linestyle="--", alpha=0.5)
    n_90 = np.searchsorted(cumvar, 0.9) + 1
    n_95 = np.searchsorted(cumvar, 0.95) + 1
    axes[1].axvline(n_90, color="red", linestyle=":", label=f"90% at PC {n_90}")
    axes[1].axvline(n_95, color="orange", linestyle=":", label=f"95% at PC {n_95}")
    axes[1].set_title(f"Cumulative Variance (Eff. Rank={effective_rank:.0f})")
    axes[1].set_xlabel("Number of PCs")
    axes[1].set_ylabel("% Variance Explained")
    axes[1].legend()

    # Singular value spectrum
    axes[2].semilogy(range(1, Z + 1), S, color="#FF5722", linewidth=2)
    axes[2].set_title("Singular Value Spectrum")
    axes[2].set_xlabel("Component index")
    axes[2].set_ylabel("Singular value (log scale)")

    plt.tight_layout()
    plt.savefig(save_dir / "03_feature_diversity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_dir / '03_feature_diversity.png'}")


def analyze_temporal_structure(results, save_dir):
    """Analysis 4: Temporal smoothness and autocorrelation."""
    print("\n" + "="*70)
    print("4. TEMPORAL STRUCTURE")
    print("="*70)

    # Collect temporal differences
    all_diffs = []
    all_autocorr = []

    for r in results:
        mu = r["mu"]   # (T, Z)
        T = mu.shape[0]
        if T < 10:
            continue

        # Frame-to-frame L2 distance
        diffs = np.linalg.norm(mu[1:] - mu[:-1], axis=1)
        all_diffs.extend(diffs.tolist())

        # Autocorrelation (lag 1-5) averaged over dims
        for lag in range(1, 6):
            if T <= lag:
                continue
            corrs = []
            for d in range(mu.shape[1]):
                r_val, _ = pearsonr(mu[:T-lag, d], mu[lag:, d])
                if np.isfinite(r_val):
                    corrs.append(r_val)
            if corrs:
                all_autocorr.append((lag, np.mean(corrs)))

    diffs_arr = np.array(all_diffs)
    print(f"  Frame-to-frame L2:  mean={diffs_arr.mean():.4f}  std={diffs_arr.std():.4f}")

    # Autocorrelation summary
    for lag in range(1, 6):
        vals = [ac for l, ac in all_autocorr if l == lag]
        if vals:
            print(f"  Autocorrelation lag-{lag}: mean={np.mean(vals):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Frame-to-frame distances
    axes[0].hist(diffs_arr, bins=50, alpha=0.7, edgecolor="black", color="#9C27B0")
    axes[0].axvline(diffs_arr.mean(), color="red", linestyle="--")
    axes[0].set_title("Frame-to-Frame L2 Distance in Latent Space")
    axes[0].set_xlabel("L2 distance")

    # Autocorrelation by lag
    lag_means = []
    for lag in range(1, 6):
        vals = [ac for l, ac in all_autocorr if l == lag]
        lag_means.append(np.mean(vals) if vals else 0)
    axes[1].bar(range(1, 6), lag_means, color="#00BCD4", alpha=0.7, edgecolor="black")
    axes[1].set_title("Mean Autocorrelation by Lag")
    axes[1].set_xlabel("Lag (TRs)")
    axes[1].set_ylabel("Pearson r")
    axes[1].set_ylim(0, 1)

    # Example latent trajectory (first clip, first 3 dims)
    if results:
        mu0 = results[0]["mu"]
        for d in range(min(3, mu0.shape[1])):
            axes[2].plot(mu0[:, d], label=f"dim {d}", alpha=0.8)
        axes[2].set_title(f"Latent Trajectory (Clip: {results[0]['uid'][:30]})")
        axes[2].set_xlabel("TR index")
        axes[2].set_ylabel("μ value")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "04_temporal_structure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_dir / '04_temporal_structure.png'}")


def analyze_visualization(results, save_dir):
    """Analysis 5: PCA and t-SNE visualization."""
    print("\n" + "="*70)
    print("5. VISUALIZATION (PCA & t-SNE)")
    print("="*70)

    # Collect latent codes with clip labels
    all_mu = []
    all_labels = []
    for i, r in enumerate(results):
        all_mu.append(r["mu"])
        all_labels.extend([i] * len(r["mu"]))

    all_mu = np.concatenate(all_mu, axis=0)
    all_labels = np.array(all_labels)

    # Subsample for t-SNE (max 5000 points)
    N = len(all_mu)
    if N > 5000:
        idx = np.random.choice(N, 5000, replace=False)
        mu_sub = all_mu[idx]
        labels_sub = all_labels[idx]
    else:
        mu_sub = all_mu
        labels_sub = all_labels

    # PCA
    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(mu_sub)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # t-SNE
    print(f"  Running t-SNE on {len(mu_sub)} points...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    mu_tsne = tsne.fit_transform(mu_sub)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    n_clips = len(results)
    cmap = plt.cm.get_cmap("tab20", min(n_clips, 20))

    scatter1 = axes[0].scatter(mu_pca[:, 0], mu_pca[:, 1],
                                c=labels_sub, cmap=cmap, s=3, alpha=0.5)
    axes[0].set_title(f"PCA ({pca.explained_variance_ratio_.sum()*100:.1f}% var)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    scatter2 = axes[1].scatter(mu_tsne[:, 0], mu_tsne[:, 1],
                                c=labels_sub, cmap=cmap, s=3, alpha=0.5)
    axes[1].set_title("t-SNE")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    # Add colorbar with clip names
    if n_clips <= 20:
        cbar = plt.colorbar(scatter2, ax=axes[1], shrink=0.8)
        tick_locs = np.linspace(0, n_clips - 1, min(n_clips, 10)).astype(int)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([results[i]["uid"][:20] for i in tick_locs])

    plt.tight_layout()
    plt.savefig(save_dir / "05_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_dir / '05_visualization.png'}")


def write_summary(results, save_dir):
    """Write a text summary of all analyses."""
    all_mu = np.concatenate([r["mu"] for r in results], axis=0)
    all_logvar = np.concatenate([r["logvar"] for r in results], axis=0)
    N, Z = all_mu.shape

    kl_per_dim = -0.5 * (1 + all_logvar - all_mu**2 - np.exp(all_logvar))
    kl_mean = kl_per_dim.mean(axis=0)

    dim_std = all_mu.std(axis=0)
    U, S, Vt = np.linalg.svd(all_mu - all_mu.mean(axis=0), full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))

    summary = f"""fMRI VAE Latent Space Analysis Summary
========================================
Total latent samples:     {N}
Latent dimensions:        {Z}
Active dims (KL > 0.1):   {(kl_mean > 0.1).sum()}/{Z}
Dead dims (std < 0.01):   {(dim_std < 0.01).sum()}/{Z}
Effective rank:           {eff_rank:.1f}/{Z}
Total KL divergence:      {kl_mean.sum():.4f}
μ global mean:            {all_mu.mean():.4f}
μ global std:             {all_mu.std():.4f}
σ global mean:            {np.exp(0.5 * all_logvar).mean():.4f}
"""
    summary_path = save_dir / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\n  → Summary saved: {summary_path}")
    print(summary)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint)
    save_dir = ckpt_dir / "analysis"
    save_dir.mkdir(exist_ok=True)

    print("="*70)
    print("fMRI VAE Latent Space Analysis")
    print("="*70)

    # Load model
    model, cfg = load_model_and_config(ckpt_dir, args.device)

    # Load validation clips
    print("\nLoading validation clips...")
    clips = load_all_clips(cfg, split="val", max_clips=args.max_clips)
    print(f"  Loaded {len(clips)} val clips")

    if not clips:
        print("No validation clips found! Trying train split...")
        clips = load_all_clips(cfg, split="train", max_clips=args.max_clips)
        print(f"  Loaded {len(clips)} train clips")

    if not clips:
        print("ERROR: No clips found!")
        return

    # Extract latents
    print("\nExtracting latent representations...")
    seq_len = cfg["vae"].get("seq_len", 100)
    results = extract_latents(model, clips, args.device, seq_len=seq_len)
    print(f"  Extracted latents for {len(results)} clips")

    # Run analyses
    analyze_reconstruction(results, save_dir)
    analyze_latent_distribution(results, save_dir)
    analyze_feature_diversity(results, save_dir)
    analyze_temporal_structure(results, save_dir)
    analyze_visualization(results, save_dir)
    write_summary(results, save_dir)

    print("\n" + "="*70)
    print(f"All plots saved to: {save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
