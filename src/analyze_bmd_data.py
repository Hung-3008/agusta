"""Deep analysis of BMD feature quality and feature-target alignment."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import torch.nn.functional as F

from src.datasets.bmd_dataset import BMDDataset
import yaml

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

# Modality names and expected dims (from BMDDataset.DEFAULT_MODALITIES)
MODALITY_INFO = [
    ("vjepa2_avg_feat", 1408),
    ("Llama-3.2-1B", 2048),
    ("Llama-3.2-3B", 3072),
    ("qwen2-5_3B", 2048),
    ("internvl3_8b_8bit", 3584),
    ("InternVL3_14B", 5120),
    ("dinov2_giant", 1536),
    ("vlm2vec_7b", 3072),
]

def analyze():
    cfg = load_config(PROJECT_ROOT / "src/configs/brainflow_bmd.yaml")
    cfg["data_root"] = "data"
    cfg["output_dir"] = "outputs/tmp_analysis"

    # Load TRAIN set for statistics
    print("Loading TRAIN dataset...")
    ds_train = BMDDataset(cfg, split="train")
    print("Loading TEST (val) dataset...")
    ds_test = BMDDataset(cfg, split="test")

    # Get unique video features (not repeated per subject)
    train_feats = ds_train.features  # (N_videos, D_total)
    test_feats = ds_test.features    # (N_videos_test, D_total)
    print(f"\nTrain features: {train_feats.shape}, Test features: {test_feats.shape}")

    # =============================================
    # PART 1: Per-modality cosine similarity
    # =============================================
    print("\n" + "="*70)
    print("PART 1: Per-Modality Cosine Similarity (among unique test videos)")
    print("="*70)

    offset = 0
    for name, dim in MODALITY_INFO:
        mod_feat = test_feats[:, offset:offset+dim]
        mod_normed = F.normalize(mod_feat, p=2, dim=1)
        sim = torch.mm(mod_normed, mod_normed.t())
        n = len(mod_feat)
        mask = ~torch.eye(n, dtype=torch.bool)
        sim_off = sim[mask]
        print(f"  {name:25s} (dim={dim:5d}): cos_sim={sim_off.mean():.4f} ± {sim_off.std():.4f}  "
              f"[min={sim_off.min():.4f}, max={sim_off.max():.4f}]  "
              f"L2={torch.norm(mod_feat, dim=1).mean():.1f}")
        offset += dim

    # Full concat
    full_normed = F.normalize(test_feats, p=2, dim=1)
    full_sim = torch.mm(full_normed, full_normed.t())
    n = len(test_feats)
    mask = ~torch.eye(n, dtype=torch.bool)
    full_off = full_sim[mask]
    print(f"  {'FULL CONCAT':25s} (dim={test_feats.shape[1]:5d}): cos_sim={full_off.mean():.4f} ± {full_off.std():.4f}  "
          f"[min={full_off.min():.4f}, max={full_off.max():.4f}]")

    # =============================================
    # PART 2: After mean-subtraction (centering)
    # =============================================
    print("\n" + "="*70)
    print("PART 2: After Mean-Subtraction (centering on train mean)")
    print("="*70)

    train_mean = train_feats.mean(dim=0, keepdim=True)  # (1, D)
    test_centered = test_feats - train_mean

    offset = 0
    for name, dim in MODALITY_INFO:
        mod_feat = test_centered[:, offset:offset+dim]
        mod_normed = F.normalize(mod_feat, p=2, dim=1)
        sim = torch.mm(mod_normed, mod_normed.t())
        n = len(mod_feat)
        mask = ~torch.eye(n, dtype=torch.bool)
        sim_off = sim[mask]
        print(f"  {name:25s} (dim={dim:5d}): cos_sim={sim_off.mean():.4f} ± {sim_off.std():.4f}  "
              f"[min={sim_off.min():.4f}, max={sim_off.max():.4f}]")
        offset += dim

    # Full centered
    fc_normed = F.normalize(test_centered, p=2, dim=1)
    fc_sim = torch.mm(fc_normed, fc_normed.t())
    fc_off = fc_sim[mask]
    print(f"  {'FULL CENTERED':25s} (dim={test_centered.shape[1]:5d}): cos_sim={fc_off.mean():.4f} ± {fc_off.std():.4f}  "
          f"[min={fc_off.min():.4f}, max={fc_off.max():.4f}]")

    # =============================================
    # PART 3: After L2-norm per modality
    # =============================================
    print("\n" + "="*70)
    print("PART 3: Per-Modality L2-Norm then Concat")
    print("="*70)

    offset = 0
    normed_parts = []
    for name, dim in MODALITY_INFO:
        mod_feat = test_feats[:, offset:offset+dim]
        mod_normed = F.normalize(mod_feat, p=2, dim=1)
        normed_parts.append(mod_normed)
        offset += dim

    normed_concat = torch.cat(normed_parts, dim=1)
    nc_normed = F.normalize(normed_concat, p=2, dim=1)
    nc_sim = torch.mm(nc_normed, nc_normed.t())
    nc_off = nc_sim[mask]
    print(f"  Per-mod L2-norm concat: cos_sim={nc_off.mean():.4f} ± {nc_off.std():.4f}  "
          f"[min={nc_off.min():.4f}, max={nc_off.max():.4f}]")

    # =============================================
    # PART 4: Feature-fMRI correlation (is there signal?)
    # =============================================
    print("\n" + "="*70)
    print("PART 4: Feature-fMRI Correlation (Subject 0, test set)")
    print("="*70)

    # Get subject 0 test fMRI — align with unique video count
    subj = ds_test.subjects[0]
    subj_fmri = ds_test.targets[subj]  # may be (N_reps*N_videos, V)
    n_videos = test_centered.shape[0]  # 102 unique videos
    if subj_fmri.shape[0] > n_videos:
        # Average over reps: reshape (N_videos, N_reps, V) → (N_videos, V)
        n_reps = subj_fmri.shape[0] // n_videos
        subj_fmri = subj_fmri[:n_videos * n_reps].reshape(n_videos, n_reps, -1).mean(dim=1)
        print(f"  Averaged {n_reps} reps → {subj_fmri.shape}")
    
    # Use centered features
    test_c = test_centered  # (N_videos, D)

    # Top-k voxels by variance
    fmri_var = subj_fmri.var(dim=0)
    topk_idx = fmri_var.argsort(descending=True)[:100]
    fmri_topk = subj_fmri[:, topk_idx]  # (N, 100)

    # Compute PCC between each top voxel and PCA of features
    # Use SVD for quick PCA
    U, S, V = torch.svd(test_c)
    feat_pcs = U[:, :50] * S[:50]  # (N, 50) top 50 PCs

    # Correlate each PC with each top voxel
    feat_pcs_z = feat_pcs - feat_pcs.mean(0, keepdim=True)
    feat_pcs_z = feat_pcs_z / (feat_pcs_z.std(0, keepdim=True) + 1e-8)
    fmri_z = fmri_topk - fmri_topk.mean(0, keepdim=True)
    fmri_z = fmri_z / (fmri_z.std(0, keepdim=True) + 1e-8)

    # (50, 100) correlation matrix
    corr = (feat_pcs_z.t() @ fmri_z) / len(test_c)
    print(f"  Feature PC ↔ Top-100 voxel correlation matrix: {corr.shape}")
    print(f"  Max absolute correlation: {corr.abs().max():.4f}")
    print(f"  Mean absolute correlation: {corr.abs().mean():.4f}")
    print(f"  Top-5 absolute correlations: {corr.abs().flatten().topk(5).values.tolist()}")

    # Per-modality correlation with fMRI
    print(f"\n  Per-modality max|corr| with top-100 voxels (Subject 0):")
    offset = 0
    for name, dim in MODALITY_INFO:
        mod_feat = test_centered[:, offset:offset+dim]
        # Quick: use mean of modality as single feature
        mod_mean = mod_feat.mean(dim=1, keepdim=True)  # (N, 1)
        mod_z = mod_mean - mod_mean.mean(0)
        mod_z = mod_z / (mod_z.std(0) + 1e-8)
        mod_corr = (mod_z.t() @ fmri_z) / len(test_c)  # (1, 100)
        # Also try L2 norm as feature
        mod_norm = torch.norm(mod_feat, dim=1, keepdim=True)
        mod_norm_z = mod_norm - mod_norm.mean(0)
        mod_norm_z = mod_norm_z / (mod_norm_z.std(0) + 1e-8)
        norm_corr = (mod_norm_z.t() @ fmri_z) / len(test_c)
        
        all_corr = torch.cat([mod_corr, norm_corr], dim=0)
        print(f"    {name:25s}: max|r|={all_corr.abs().max():.4f}, mean|r|={all_corr.abs().mean():.4f}")
        offset += dim

    # =============================================
    # PART 5: fMRI target stats per subject
    # =============================================
    print("\n" + "="*70)
    print("PART 5: fMRI Target Stats per Subject")
    print("="*70)
    for subj in ds_test.subjects:
        fmri = ds_test.targets[subj]
        print(f"  {subj}: shape={fmri.shape}, mean={fmri.mean():.4f}, std={fmri.std():.4f}, "
              f"min={fmri.min():.4f}, max={fmri.max():.4f}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    analyze()
