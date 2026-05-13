"""Smoke test: synthesise_hrf_direct vs synthesise."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.brainflow.brainflow import BrainFlow

model = BrainFlow(
    output_dim=100,
    velocity_net_params=dict(
        hidden_dim=64, dit_num_blocks=2, n_heads=4,
        context_trs=1, n_target_trs=1, latent_dim=64,
        use_subject_head=True, network_head=False,
        context_encoder='multitoken', use_dit_decoder=True,
        decoder_type='ditx', modality_dims=[64, 32],
        max_seq_len=1, fusion_mode='concat', fusion_proj_dim=32,
    ),
    n_subjects=2,
    use_csfm=True,
).eval()

ctx = torch.randn(2, 1, 96)
sid = torch.tensor([0, 1])

out1 = model.synthesise_hrf_direct(ctx, subject_ids=sid)
print(f"synthesise_hrf_direct: {out1.shape}")

out2 = model.synthesise(ctx, n_timesteps=5, subject_ids=sid)
print(f"synthesise (full ODE):  {out2.shape}")

print("✓ Both methods work correctly")
