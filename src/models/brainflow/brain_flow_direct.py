"""BrainFlow Direct — Flow Matching in fMRI Space with Pre-extracted Context.

No encoder — context is pre-extracted and loaded from .npy files.
Flow matching operates directly in 1000-dim fMRI space from Gaussian noise.

Architecture:
  - Input conditioning: pre-extracted context (B, T, context_dim)
  - Source: x_0 ~ N(0, I) in fMRI space
  - Target: x_1 = raw fMRI (B, output_dim)
  - VelocityNet: MLP blocks with FiLM (time) + CrossAttn (context)
  - OT-CFM path + ODESolver from flow_matching library

Usage:
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct.yaml
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver


# =============================================================================
# Building Blocks
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Scale continuous time [0, 1] to [0, 1000] for standard diffusion frequencies
        t = t * 1000.0
        
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# =============================================================================
# CrossAttentionResBlock — FiLM (time) + FFN + CrossAttn (context)
# =============================================================================

class CrossAttentionResBlock(nn.Module):
    """Residual block with FiLM (time) + cross-attention (context).

    Architecture per block:
        h = FiLM(LayerNorm(x), t_emb)     — time-conditioned modulation
        h = x + FFN(h)                     — residual MLP
        h = h + CrossAttn(h, context)      — attend to full context sequence
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        context_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # FiLM from timestep
        self.film = nn.Linear(time_dim, dim * 2)

        # FFN with pre-norm
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Cross-attention to full context sequence
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, D) hidden state
            t_emb:   (B, D_t) timestep embedding
            context: (B, T, C) full context sequence
        """
        # 1. FiLM modulated FFN
        scale_shift = self.film(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.ffn(h)

        # 2. Cross-attention to context
        q = self.norm_q(x).unsqueeze(1)       # (B, 1, D)
        kv = self.norm_kv(context)             # (B, T, C)
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, D)
        x = x + attn_out.squeeze(1)           # (B, D)

        return x


# =============================================================================
# DirectVelocityNet — FiLM + Per-Block Cross-Attention
# =============================================================================

class DirectVelocityNet(nn.Module):
    """Velocity network for direct fMRI generation.

    Architecture:
        1. Input: x_t (B, output_dim) → Linear → (B, hidden_dim)
        2. Time: t → sinusoidal → MLP → (B, hidden_dim)
        3. N blocks, each with:
           - FiLM(time_emb) → FFN residual
           - CrossAttn(hidden, context)
        4. Output: LayerNorm → Linear → (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 512,
        context_dim: int = 512,
        n_blocks: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 11,
        n_subjects: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Global Flattened Context MLP (Algonauts 2025 Trick)
        # Maps the entire rigid window into a single global context vector
        self.context_mlp = nn.Sequential(
            nn.Linear(max_seq_len * context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Context Positional Embedding
        self.context_pos_emb = nn.Parameter(torch.randn(1, max_seq_len, context_dim) * 0.02)

        # Subject Embedding (injected into t_emb for per-subject FiLM modulation)
        self.subject_emb = nn.Embedding(n_subjects, hidden_dim)

        # Residual blocks with FiLM + cross-attention
        self.blocks = nn.ModuleList([
            CrossAttentionResBlock(hidden_dim, hidden_dim, context_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # Output projection (initialized to zero for zero-velocity predict at start)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        subject_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x:           (B, output_dim) noisy fMRI at time t.
            t:           (B,) or scalar, timestep in [0, 1].
            cond:        (B, T, context_dim) pre-extracted context sequence.
            subject_ids: (B,) long tensor of subject indices.

        Returns:
            v_pred: (B, output_dim) predicted velocity.
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # Embeddings
        t_emb = self.time_mlp(SinusoidalPosEmb(self.hidden_dim)(t))  # (B, hidden_dim)

        # Subject embedding (added to t_emb for per-subject FiLM modulation)
        if subject_ids is not None:
            t_emb = t_emb + self.subject_emb(subject_ids)

        # Prepare context
        if cond is not None and cond.dim() == 2:
            cond = cond.unsqueeze(1)
        if cond is None:
            cond = torch.zeros(x.shape[0], 1, self.hidden_dim, device=x.device, dtype=x.dtype)
            
        # Add Positional Embedding to context so cross-attention knows temporal order!
        seq_len = cond.shape[1]
        
        # Pad or truncate to max_seq_len for global flatten
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            # Pad along the sequence dimension (dim=1)
            cond = F.pad(cond, (0, 0, 0, pad_len))
        elif seq_len > self.max_seq_len:
            cond = cond[:, :self.max_seq_len, :]
            
        cond = cond + self.context_pos_emb[:, :cond.shape[1], :] # Ensure positional embedding matches current sequence length

        # Compute global flattened context and inject into time embedding!
        # This gives a massive boost by providing rigid spatio-temporal structure
        global_cond = self.context_mlp(cond.reshape(-1, self.max_seq_len * cond.shape[-1]))
        t_emb = t_emb + global_cond

        # Forward
        h = self.input_proj(x)  # (B, hidden_dim)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = self.final_norm(h)
        return self.output_proj(h)  # (B, output_dim)


# =============================================================================
# BrainFlowDirect — Top-level model
# =============================================================================

class BrainFlowDirect(nn.Module):
    """Direct fMRI flow matching with pre-extracted context conditioning.

    No encoder — context (B, T, context_dim) is loaded externally.
    Flow matching in fMRI 1000-dim space from Gaussian noise.
    """

    def __init__(
        self,
        output_dim: int = 1000,
        velocity_net_params: dict = None,
        n_subjects: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Velocity network
        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg.setdefault("n_subjects", n_subjects)
        self.velocity_net = DirectVelocityNet(**vn_cfg)

        # OT-CFM path
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log parameters
        n_params = sum(p.numel() for p in self.velocity_net.parameters())
        print(f"[BrainFlowDirect] VelocityNet: {n_params:,} params")

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute OT-CFM loss in fMRI space.

        Args:
            context: (B, T, context_dim) pre-extracted encoder latents.
            target: (B, output_dim) ground truth fMRI.
            subject_ids: (B,) long tensor of subject indices.

        Returns:
            loss: scalar MSE(v_pred, dx_t).
        """
        # OT-CFM: sample path
        x_1 = target
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # Predict velocity
        v_pred = self.velocity_net(
            x=sample_info.x_t,
            t=sample_info.t,
            cond=context,
            subject_ids=subject_ids,
        )

        # MSE loss
        return F.mse_loss(v_pred, sample_info.dx_t)

    @torch.inference_mode()
    def synthesise(
        self,
        context: torch.Tensor,
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
        subject_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE with context conditioning.

        Args:
            context: (B, T, context_dim) pre-extracted encoder latents.
            n_timesteps: Number of ODE solver steps.
            solver_method: ODE solver method.
            subject_ids: (B,) long tensor of subject indices.

        Returns:
            fmri_pred: (B, output_dim) predicted fMRI.
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        x_init = torch.zeros(B, self.output_dim, device=device, dtype=dtype)

        # ODE solve
        solver = ODESolver(velocity_model=self.velocity_net)
        T = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)

        fmri_pred = solver.sample(
            time_grid=T,
            x_init=x_init,
            method=solver_method,
            step_size=1.0 / n_timesteps,
            return_intermediates=False,
            cond=context,
            subject_ids=subject_ids,
        )

        return fmri_pred
