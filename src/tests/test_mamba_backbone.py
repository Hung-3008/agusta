"""Smoke test: verify MambaFlowBackbone shape & gradient flow."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from models.brainflow.components import RotaryEmbedding
from models.brainflow.mamba_backbone import MambaFlowBackbone

def test_shape_and_grad():
    B, T, D = 4, 50, 1024
    T_ctx = 50
    n_heads = 16

    rotary = RotaryEmbedding(D // n_heads, max_seq_len=64)
    backbone = MambaFlowBackbone(
        d_model=D, nhead=n_heads, dim_feedforward=D * 4,
        dropout=0.0, time_dim=D, rotary_emb=rotary,
        dit_depth=12, d_state=16, d_conv=4, expand_factor=2,
    )

    # Count params
    n_params = sum(p.numel() for p in backbone.parameters())
    n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total params: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Trainable:    {n_trainable:,}")

    h = torch.randn(B, T, D, requires_grad=True)
    t_emb = torch.randn(B, D)
    ctx = torch.randn(B, T_ctx, D)

    # Forward
    out = backbone(h, t_emb, ctx)
    print(f"Input shape:  {h.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, T, D), f"Shape mismatch: {out.shape} != {(B, T, D)}"
    print("✓ Shape test passed")

    # Gradient flow
    loss = out.sum()
    loss.backward()
    assert h.grad is not None, "No gradient on input"
    assert not torch.isnan(h.grad).any(), "NaN in input gradient"
    print(f"✓ Gradient test passed (grad norm: {h.grad.norm():.4f})")

    # Check all params have gradients
    no_grad = [n for n, p in backbone.named_parameters() if p.grad is None]
    if no_grad:
        print(f"⚠ Params without gradients: {no_grad}")
    else:
        print("✓ All parameters have gradients")

    # Check for NaN gradients
    nan_params = [n for n, p in backbone.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    if nan_params:
        print(f"✗ NaN gradients in: {nan_params}")
    else:
        print("✓ No NaN gradients")

    # Gradient checkpointing test
    backbone.gradient_checkpointing = True
    backbone.train()
    h2 = torch.randn(B, T, D, requires_grad=True)
    out2 = backbone(h2, t_emb, ctx)
    loss2 = out2.sum()
    loss2.backward()
    assert h2.grad is not None, "No gradient with checkpointing"
    print("✓ Gradient checkpointing test passed")

    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    test_shape_and_grad()
