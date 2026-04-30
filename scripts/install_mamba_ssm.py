#!/usr/bin/env python3
"""
Install and verify mamba-ssm for BrainFlow DiMamba backbone.

Run this script from inside your active training environment:
    python scripts/install_mamba_ssm.py

It will:
1. Detect PyTorch + CUDA version
2. Check if mamba-ssm is already installed
3. If not, attempt to install the correct wheel
4. Verify the import works
5. Run a quick forward-pass sanity check on DiMamba1DBackbone
"""

import sys
import subprocess

def run(cmd, capture=True):
    r = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    return r.stdout.strip(), r.returncode

# ── Step 1: Detect environment ───────────────────────────────────────────────
print("=" * 60)
print("Step 1: Detecting PyTorch / CUDA version")
print("=" * 60)

try:
    import torch
    pt_ver = torch.__version__
    cuda_ver = torch.version.cuda or "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"  PyTorch : {pt_ver}")
    print(f"  CUDA    : {cuda_ver}")
    print(f"  Device  : {device} ({gpu_name})")
except ImportError:
    print("  ERROR: PyTorch not found. Are you in the correct conda/venv?")
    sys.exit(1)

# ── Step 2: Check mamba_ssm ──────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 2: Checking mamba-ssm")
print("=" * 60)

try:
    from mamba_ssm import Mamba
    print("  mamba-ssm is already installed!")
    mamba_installed = True
except ImportError:
    out, _ = run("pip show mamba-ssm 2>&1")
    print(f"  mamba-ssm not found. pip show: {out or 'not installed'}")
    mamba_installed = False

# ── Step 3: Install if missing ───────────────────────────────────────────────
if not mamba_installed:
    print()
    print("=" * 60)
    print("Step 3: Installing mamba-ssm")
    print("=" * 60)

    # Determine CUDA tag for wheel
    # e.g. CUDA 12.1 → cu121, CUDA 11.8 → cu118
    if cuda_ver == "cpu":
        print("  WARNING: No CUDA detected. mamba-ssm requires CUDA.")
        print("  Installing anyway (will use fallback in dim_backbone.py)...")
        cuda_tag = ""
    else:
        cuda_tag = "cu" + cuda_ver.replace(".", "")[:3]  # e.g. "cu121"

    # PyTorch version tag: e.g. 2.1.0 → torch2.1
    pt_short = "torch" + ".".join(pt_ver.split(".")[:2])
    
    # Try installing from PyPI (builds from source if no pre-built wheel)
    print(f"  Attempting: pip install mamba-ssm (CUDA={cuda_tag}, {pt_short})")
    print("  This may take a few minutes (CUDA extension compilation)...")
    
    out, code = run(
        "pip install mamba-ssm --no-build-isolation 2>&1",
        capture=False,
    )
    
    if code != 0:
        print()
        print("  pip install failed. Try the causal-conv1d dependency first:")
        print("    pip install causal-conv1d && pip install mamba-ssm")
        print()
        print("  Or install with extra CUDA deps:")
        print("    pip install mamba-ssm[causal-conv1d]")
        print()
        print("  The BrainFlow fallback SSM in dim_backbone.py will be used")
        print("  if mamba-ssm is not available (slower but functional).")
        sys.exit(0)

# ── Step 4: Verify import ────────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 4: Verifying mamba-ssm import")
print("=" * 60)

try:
    from mamba_ssm import Mamba
    print("  ✓ from mamba_ssm import Mamba — OK")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("  The pure-PyTorch fallback in dim_backbone.py will be used.")
    sys.exit(0)

# ── Step 5: Quick forward-pass test ─────────────────────────────────────────
print()
print("=" * 60)
print("Step 5: Quick DiMamba1DBackbone forward-pass sanity check")
print("=" * 60)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from src.models.brainflow.dim_backbone import DiMamba1DBackbone
    from src.models.brainflow.components import RotaryEmbedding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, T, B = 128, 50, 4   # small dims for quick test

    rotary_emb = RotaryEmbedding(D // 8, max_seq_len=64).to(device)
    backbone = DiMamba1DBackbone(
        d_model=D, nhead=8, dropout=0.0, time_dim=D,
        rotary_emb=rotary_emb, dit_depth=4,   # 2 enc + 2 dec
        d_state=16, d_conv=4, expand=2, use_unet_skip=True,
    ).to(device)

    with torch.no_grad():
        h = torch.randn(B, T, D, device=device)
        t_emb = torch.randn(B, D, device=device)
        ctx = torch.randn(B, T, D, device=device)
        out = backbone(h, t_emb, ctx)

    assert out.shape == (B, T, D), f"Shape mismatch: {out.shape}"
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"  ✓ Forward pass OK: ({B}, {T}, {D}) → {tuple(out.shape)}")
    print(f"  ✓ Params: {n_params:,}")
    print()
    print("  All checks passed! DiMamba1DBackbone is ready.")
    print()
    print("  To train:")
    print("    python src/train_brainflow.py \\")
    print("           --config src/configs/brainflow_dimamba.yaml \\")
    print("           --fast_dev_run   # sanity check first")
    print()
    print("    python src/train_brainflow.py \\")
    print("           --config src/configs/brainflow_dimamba.yaml")

except Exception as e:
    import traceback
    print(f"  ✗ Forward-pass test failed: {e}")
    traceback.print_exc()
    sys.exit(1)
