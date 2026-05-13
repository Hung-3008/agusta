"""
BOLD Moments Dataset (BMD) — DataLoader for fMRI synthesis.

Unlike Algonauts (time-series sliding window), BMD is event-related:
  - Each stimulus is a 3s video clip → one pooled feature vector per modality
  - fMRI betas are trial-averaged or per-rep: (N, N_voxels) per subject
  - No temporal context / sliding window needed

Target modes (atlas):
  - "roi_subset": 8,335-dim (46 specific ROIs)
  - "bmd_general": 14,672-dim (whole brain mask)

Repetition modes:
  - "avg": trial-averaged (1000 train / 102 test per subject)
  - "per_rep": each repetition as separate sample (3000 train / 1020 test)

Feature format:
  Data/features_npy_pooled/{modality}/bmd/{train,test}/{XXXX}.npy
  Each file: shape (1, D) float32

Target format (pre-extracted by create_bmd_targets.py):
  avg:     targets/{roi_subset,bmd_general}/{sub-XX}_{split}.npy  — (N_videos, V)
  per_rep: targets/{roi_subset,bmd_general}_per_rep/{sub-XX}_{split}.npy — (N_videos*N_reps, V)
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("datasets.bmd")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Number of repetitions per split (defined by BMD protocol)
N_REPS = {"train": 3, "test": 10}


class BMDDataset(Dataset):
    """BOLD Moments Dataset for fMRI synthesis from multimodal features.

    Each sample returns:
        context: (D_total,) — concatenated features from all modalities
        fmri:    (V,)       — target fMRI beta vector
        subject_idx: int    — subject index for conditioning
        video_id: str       — video identifier (e.g. '0001')
    """

    # Modality dims excluding whisper (no audio in BMD)
    DEFAULT_MODALITIES = [
        "vjepa2_avg_feat",      # 1408
        "Llama-3.2-1B",         # 2048
        "Llama-3.2-3B",         # 3072
        "qwen2-5_3B",           # 2048
        "qwen-2-5-omni-7b",    # 3584
        "internvl3_8b_8bit",    # 3584
        "InternVL3_14B",        # 5120
        "dinov2_giant",         # 6144
        "vlm2vec_7b",           # 3584
    ]

    TARGET_DIMS = {
        "roi_subset": 8335,
        "roi_subset_grouped": 8335,  # same voxels, ordered by 9 categories
        "bmd_general": 14672,
    }

    def __init__(self, cfg, split="train"):
        """
        Args:
            cfg: Config dict with keys:
                - bmd.target_mode: "roi_subset" or "bmd_general"
                - bmd.rep_mode: "avg" (default) or "per_rep"
                - bmd.modalities: list of modality dir names (optional)
                - bmd.feature_root: path to features
                - bmd.target_root: path to targets
                - subjects: list of subject IDs
            split: "train" or "test"
        """
        self.cfg = cfg
        self.split = split

        bmd_cfg = cfg.get("bmd", {})
        self.target_mode = bmd_cfg.get("target_mode", "roi_subset")
        self.rep_mode = bmd_cfg.get("rep_mode", "avg")

        assert self.target_mode in self.TARGET_DIMS, \
            f"Unknown target_mode: {self.target_mode}. Choose from {list(self.TARGET_DIMS.keys())}"
        assert self.rep_mode in ("avg", "per_rep"), \
            f"Unknown rep_mode: {self.rep_mode}. Choose from ['avg', 'per_rep']"

        self.subjects = cfg["subjects"]
        self.subject_to_idx = {s: i for i, s in enumerate(self.subjects)}

        # Feature directories
        feature_root = Path(PROJECT_ROOT) / bmd_cfg.get(
            "feature_root", "Data/features_npy_pooled"
        )
        self.modalities = bmd_cfg.get("modalities", self.DEFAULT_MODALITIES)
        self.modality_dirs = [feature_root / m / "bmd" for m in self.modalities]

        # Target directory — per_rep uses {mode}_per_rep subdirectory
        target_root = Path(PROJECT_ROOT) / bmd_cfg.get(
            "target_root", "Data/BOLDMomentsDataset/targets"
        )
        if self.rep_mode == "per_rep":
            self.target_dir = target_root / f"{self.target_mode}_per_rep"
        else:
            self.target_dir = target_root / self.target_mode

        self.n_reps = N_REPS[split] if self.rep_mode == "per_rep" else 1

        # Preload everything into RAM
        self.features, self.targets, self.samples = self._preload()

    def _preload(self):
        """Preload all features and targets into RAM."""
        split_dir = "train" if self.split == "train" else "test"

        # --- 1. Load features ---
        ref_dir = self.modality_dirs[0] / split_dir
        video_ids = sorted([f.stem for f in ref_dir.glob("*.npy")])
        n_videos = len(video_ids)
        logger.info("[BMD %s/%s] Found %d videos from %s",
                    self.split, self.rep_mode, n_videos, ref_dir)

        # Load and concatenate all modalities
        mod_arrays = []
        total_dim = 0
        for mod_dir in self.modality_dirs:
            feat_dir = mod_dir / split_dir
            mod_feats = []
            for vid_id in video_ids:
                fpath = feat_dir / f"{vid_id}.npy"
                if fpath.exists():
                    feat = np.load(fpath).astype(np.float32).squeeze()  # (D,)
                else:
                    logger.warning("Missing feature: %s", fpath)
                    feat = np.zeros(mod_feats[0].shape[-1], dtype=np.float32) if mod_feats else None
                    if feat is None:
                        raise FileNotFoundError(f"No features found at {fpath}")
                mod_feats.append(feat)
            mod_array = np.stack(mod_feats, axis=0)  # (N_videos, D_mod)
            mod_arrays.append(mod_array)
            total_dim += mod_array.shape[-1]
            logger.info("  Modality %s: %s → dim=%d",
                        mod_dir.parent.name, mod_array.shape, mod_array.shape[-1])

        # Concatenate: (N_videos, D_total)
        features = np.concatenate(mod_arrays, axis=-1).astype(np.float32)
        features_tensor = torch.from_numpy(features)
        logger.info("[BMD %s] Features: %s (%.1f MB), total_dim=%d",
                    self.split, features.shape,
                    features.nbytes / 1024 / 1024, total_dim)

        # --- 2. Load targets per subject ---
        targets = {}
        for subj in self.subjects:
            target_path = self.target_dir / f"{subj}_{split_dir}.npy"
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Target not found: {target_path}. "
                    f"Run `python scripts/create_bmd_targets.py` first."
                )
            target_data = np.load(target_path).astype(np.float32)

            if self.rep_mode == "per_rep":
                # per_rep: (N_videos * N_reps, V)
                expected_rows = n_videos * self.n_reps
                assert target_data.shape[0] == expected_rows, \
                    f"Per-rep target shape mismatch for {subj}: " \
                    f"{target_data.shape[0]} != {expected_rows} ({n_videos}×{self.n_reps})"
            else:
                # avg: (N_videos, V)
                assert target_data.shape[0] == n_videos, \
                    f"Target shape mismatch for {subj}: {target_data.shape[0]} != {n_videos}"

            targets[subj] = torch.from_numpy(target_data)
            logger.info("  Target %s [%s/%s]: %s (%.1f MB)",
                        subj, self.target_mode, self.rep_mode,
                        target_data.shape, target_data.nbytes / 1024 / 1024)

        # --- 3. Build sample index ---
        samples = []
        if self.rep_mode == "per_rep":
            # Each (video, rep, subject) is a sample
            # Target layout: rep-major within each video: [vid0_rep0, vid0_rep1, vid0_rep2, vid1_rep0, ...]
            for vid_idx, vid_id in enumerate(video_ids):
                for rep_idx in range(self.n_reps):
                    for subj in self.subjects:
                        samples.append({
                            "video_idx": vid_idx,           # feature index
                            "target_idx": vid_idx * self.n_reps + rep_idx,  # target index
                            "video_id": vid_id,
                            "rep_idx": rep_idx,
                            "subject": subj,
                            "subject_idx": self.subject_to_idx[subj],
                        })
        else:
            # avg: each (video, subject) is a sample
            for vid_idx, vid_id in enumerate(video_ids):
                for subj in self.subjects:
                    samples.append({
                        "video_idx": vid_idx,
                        "target_idx": vid_idx,
                        "video_id": vid_id,
                        "rep_idx": -1,
                        "subject": subj,
                        "subject_idx": self.subject_to_idx[subj],
                    })

        total_mb = (features.nbytes + sum(t.nelement() * 4 for t in targets.values())) / 1024 / 1024
        n_effective = n_videos * self.n_reps if self.rep_mode == "per_rep" else n_videos
        logger.info("[BMD %s/%s] Total: %d samples (%d videos × %s × %d subjects), "
                    "%.0f MB in RAM",
                    self.split, self.rep_mode, len(samples), n_videos,
                    f"{self.n_reps} reps" if self.rep_mode == "per_rep" else "avg",
                    len(self.subjects), total_mb)

        return features_tensor, targets, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]

        context = self.features[info["video_idx"]]              # (D_total,)
        fmri = self.targets[info["subject"]][info["target_idx"]]  # (V,)

        return {
            "context": context,
            "fmri": fmri,
            "subject_idx": info["subject_idx"],
            "video_id": info["video_id"],
            "clip_key": f"{info['subject']}/bmd/{info['video_id']}",
        }

    @property
    def n_voxels(self):
        return self.TARGET_DIMS[self.target_mode]

    @property
    def context_dim(self):
        return self.features.shape[-1]

    @property
    def n_subjects(self):
        return len(self.subjects)


def get_bmd_dataloaders(cfg):
    """Create train and test dataloaders for BMD."""
    train_set = BMDDataset(cfg, split="train")
    test_set = BMDDataset(cfg, split="test")

    dl_cfg = cfg.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 64)
    val_batch_size = dl_cfg.get("val_batch_size", batch_size)
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = dl_cfg.get("pin_memory", True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info("BMD DataLoaders created — train: %d batches, test: %d batches",
                len(train_loader), len(test_loader))

    return train_loader, test_loader


# ─── Quick smoke test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # Usage: python -m src.datasets.bmd_dataset [target_mode] [rep_mode]
    target_mode = sys.argv[1] if len(sys.argv) > 1 else "roi_subset"
    rep_mode = sys.argv[2] if len(sys.argv) > 2 else "avg"

    cfg = {
        "subjects": ["sub-01", "sub-02"],
        "bmd": {
            "target_mode": target_mode,
            "rep_mode": rep_mode,
        },
        "dataloader": {
            "batch_size": 32,
            "val_batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
        },
    }

    print(f"\n{'='*60}")
    print(f"  BMD Dataset Smoke Test")
    print(f"  target_mode: {target_mode}, rep_mode: {rep_mode}")
    print(f"{'='*60}\n")

    train_loader, test_loader = get_bmd_dataloaders(cfg)

    print(f"\nTrain: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    # Sample one batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"  context:     {batch['context'].shape} {batch['context'].dtype}")
    print(f"  fmri:        {batch['fmri'].shape} {batch['fmri'].dtype}")
    print(f"  subject_idx: {batch['subject_idx'].shape}")
    print(f"  video_id:    {batch['video_id'][:3]}")

    # Verify dimensions
    ds = train_loader.dataset
    print(f"\nDataset properties:")
    print(f"  n_voxels:    {ds.n_voxels}")
    print(f"  context_dim: {ds.context_dim}")
    print(f"  n_subjects:  {ds.n_subjects}")
    print(f"  rep_mode:    {ds.rep_mode}")
    print(f"  n_reps:      {ds.n_reps}")

    assert batch["fmri"].shape[-1] == ds.n_voxels
    print(f"\n✅ All checks passed!")
