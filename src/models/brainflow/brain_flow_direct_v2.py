"""BrainFlow Direct V2 — End-to-End Flow Matching with Raw Multimodal Features.

No DAE bottleneck — raw features are projected per-modality and fused via
self-attention in the velocity network.

Architecture:
  - ModalityProjector: per-modality MLP → proj_dim
  - TransformerVelocityNet:
      1. Project x_t to 1 token
      2. Concat [x_t_token, ctx_1..ctx_T] → full self-attention
      3. FiLM (time) + StochasticDepth + ModalityDropout
  - OT-CFM path from flow_matching library

Usage:
    python src/train_brainflow_direct_v2.py --config src/configs/brain_flow_direct_v2.yaml
"""

import math
import random

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
        t = t * 1000.0  # Scale [0,1] → [0,1000]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class StochasticDepth(nn.Module):
    """Drop entire residual branch with probability p during training."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = torch.rand(1, device=x.device) > self.p
        return x * keep / max(1.0 - self.p, 1e-8)


class SubjectLayers(nn.Module):
    """Subject-specific linear projection."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels))
        self.weights.data.normal_()
        self.bias.data.normal_()
        self.weights.data *= 1 / in_channels**0.5
        self.bias.data *= 1 / in_channels**0.5

    def forward(self, x: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        B, C = x.shape
        w = self.weights.index_select(0, subject_idx)  # (B, C, D)
        b = self.bias.index_select(0, subject_idx)     # (B, D)
        out = torch.einsum("bc,bcd->bd", x, w) + b
        return out


class LayerMixer(nn.Module):
    """Learnable scalar mixing for multilayer features."""
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., L, D)
        weights = F.softmax(self.weights, dim=0)
        w_shape = [1] * x.ndim
        w_shape[-2] = self.weights.shape[0]
        w = weights.view(*w_shape)
        return (x * w).sum(dim=-2)


# =============================================================================
# Per-Modality Projector
# =============================================================================

class ModalityProjector(nn.Module):
    """Projects raw modality features (optionally multi-layer) to a common dimension."""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 1:
            self.mixer = LayerMixer(num_layers)
        else:
            self.mixer = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle (B, T, L, D) -> (B, T, D)
        if self.num_layers > 1 and x.ndim == 4:
            x = self.mixer(x)
        elif x.ndim == 4:
            # Fallback if config said 1 but data has layers: mean pool
            x = x.mean(dim=-2)
            
        return self.net(x)  # (B, T, output_dim)


# =============================================================================
# Transformer Block with adaLN (FiLM-style time conditioning)
# =============================================================================

class AdaLNTransformerBlock(nn.Module):
    """Transformer block with adaptive LayerNorm (FiLM from time embedding).

    Architecture:
        1. adaLN(x, t_emb) → Self-Attention → residual + StochasticDepth
        2. adaLN(x, t_emb) → FFN → residual + StochasticDepth
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        n_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.15,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # adaLN modulation for attention (scale, shift, gate) × 2 branches
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 6),  # 6 params: γ1,β1,α1, γ2,β2,α2
        )

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )

        # FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

        # Stochastic depth
        self.drop_path1 = StochasticDepth(drop_path)
        self.drop_path2 = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, S, D) sequence of tokens
            t_emb: (B, D_t) timestep embedding (will broadcast to all tokens)
        """
        # Get modulation parameters
        mod = self.adaLN_modulation(t_emb)  # (B, 6D)
        γ1, β1, α1, γ2, β2, α2 = mod.chunk(6, dim=-1)  # each (B, D)

        # 1. Self-attention with adaLN
        h = self.norm1(x)
        h = h * (1 + γ1.unsqueeze(1)) + β1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + self.drop_path1(α1.unsqueeze(1) * h)

        # 2. FFN with adaLN
        h = self.norm2(x)
        h = h * (1 + γ2.unsqueeze(1)) + β2.unsqueeze(1)
        h = self.ffn(h)
        x = x + self.drop_path2(α2.unsqueeze(1) * h)

        return x


# =============================================================================
# TransformerVelocityNet — Full Self-Attention with FiLM
# =============================================================================

class TransformerVelocityNet(nn.Module):
    """Velocity network with full self-attention over [x_t, context] tokens.

    Architecture:
        1. Per-modality projection + fusion → context (B, T, D)
        2. x_t → Linear → 1 token (B, 1, D)
        3. Concat [x_t_token, ctx] → (B, 1+T, D)
        4. Time embedding → adaLN modulation
        5. N Transformer blocks with self-attention
        6. Extract x_t_token → Linear → output (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 1024,
        modality_dims: dict = None,  # {"video": {"dim": 1408, "layers": 13}, ...}
        n_blocks: int = 8,
        n_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.15,
        drop_path: float = 0.1,
        modality_dropout: float = 0.2,
        max_seq_len: int = 20,
        n_subjects: int = 10,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout

        if modality_dims is None:
            modality_dims = {"context": {"dim": 512, "layers": 1}}
        self.modality_names = sorted(modality_dims.keys())
        self.n_modalities = len(self.modality_names)

        # Per-modality projectors
        self.modality_projectors = nn.ModuleDict()
        for name, meta in modality_dims.items():
            if isinstance(meta, dict):
                dim = meta.get("dim", 512)
                layers = meta.get("layers", 1)
            else:
                dim = meta
                layers = 1
            self.modality_projectors[name] = ModalityProjector(dim, hidden_dim, layers, dropout)

        # Input projection for x_t (noisy fMRI)
        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Positional embeddings
        # +1 for the x_t token at position 0
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len + 1, hidden_dim) * 0.02)

        # Transformer blocks with stochastic depth (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(
                dim=hidden_dim,
                time_dim=hidden_dim,
                n_heads=n_heads,
                ff_mult=ff_mult,
                dropout=dropout,
                drop_path=drop_path * i / max(n_blocks - 1, 1),
            )
            for i in range(n_blocks)
        ])

        # Final norm + output projection with SubjectLayers
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = SubjectLayers(hidden_dim, output_dim, n_subjects)
        
        # Zero-init
        nn.init.constant_(self.output_proj.weights, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def _apply_modality_dropout(
        self, modality_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """During training, randomly zero out entire modalities (keep ≥ 1)."""
        if not self.training or self.modality_dropout <= 0:
            return modality_features

        active_mods = list(modality_features.keys())
        if len(active_mods) <= 1:
            return modality_features

        # Decide which to drop
        to_drop = []
        for mod in active_mods:
            if random.random() < self.modality_dropout:
                to_drop.append(mod)

        # Keep at least 1
        if len(to_drop) >= len(active_mods):
            keep = random.choice(active_mods)
            to_drop = [m for m in to_drop if m != keep]

        result = {}
        for mod, feat in modality_features.items():
            if mod in to_drop:
                result[mod] = torch.zeros_like(feat)
            else:
                result[mod] = feat

        return result

    def _fuse_modalities(
        self, modality_features: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Project and sum multimodal features → (B, T, hidden_dim).

        Each modality is projected to hidden_dim, then element-wise summed.
        Using sum (not concat) keeps sequence length manageable and allows
        varying numbers of active modalities.
        """
        projected = []
        for mod_name in self.modality_names:
            if mod_name in modality_features:
                feat = modality_features[mod_name]  # (B, T, dim_mod)
                proj = self.modality_projectors[mod_name](feat)  # (B, T, D)
                projected.append(proj)

        if not projected:
            # Fallback: zero context
            first = next(iter(modality_features.values()))
            B, T = first.shape[0], first.shape[1]
            return torch.zeros(B, T, self.hidden_dim, device=first.device, dtype=first.dtype)

        # Element-wise sum of all projected modalities
        context = projected[0]
        for p in projected[1:]:
            context = context + p
        return context

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict = None,
        subject_idx: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, output_dim) noisy fMRI at time t.
            t:    (B,) or scalar, timestep in [0, 1].
            cond: dict of {mod_name: (B, T, L, dim_mod)} multimodal features.
            subject_idx: (B,) index of subjects.
            
        Returns:
            v_pred: (B, output_dim) predicted velocity.
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        if subject_idx is None:
            # Fallback for fast dev run without subject_idx
            subject_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        B = x.shape[0]

        # Time embedding
        t_emb = self.time_embed(t)  # (B, D)

        # Modality dropout + fusion
        if cond is None or not isinstance(cond, dict):
            # Fallback for ODE solver which passes cond as-is
            if cond is not None and isinstance(cond, torch.Tensor):
                # Legacy compat: single tensor context
                context = cond
                if context.dim() == 2:
                    context = context.unsqueeze(1)
                # Project if dim != hidden_dim
                if context.shape[-1] != self.hidden_dim:
                    first_proj = next(iter(self.modality_projectors.values()))
                    context = first_proj(context)
            else:
                context = torch.zeros(B, 1, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            cond_dropped = self._apply_modality_dropout(cond)
            context = self._fuse_modalities(cond_dropped)  # (B, T, D)

        # x_t → 1 token
        x_token = self.input_proj(x).unsqueeze(1)  # (B, 1, D)

        # Concat: [x_t_token, context_tokens]
        seq = torch.cat([x_token, context], dim=1)  # (B, 1+T, D)

        # Add positional embedding
        S = seq.shape[1]
        seq = seq + self.pos_emb[:, :S, :]

        # Transformer blocks
        for block in self.blocks:
            seq = block(seq, t_emb)

        # Extract x_t token (position 0)
        x_out = seq[:, 0, :]  # (B, D)

        # Output projection
        x_out = self.final_norm(x_out)
        return self.output_proj(x_out, subject_idx)  # (B, output_dim)


# =============================================================================
# BrainFlowDirectV2 — Top-level model
# =============================================================================

class BrainFlowDirectV2(nn.Module):
    """Direct fMRI flow matching with end-to-end multimodal features.

    No DAE bottleneck — raw features are projected and fused inside the
    velocity network via self-attention.
    """

    def __init__(
        self,
        output_dim: int = 1000,
        modality_dims: dict = None,
        velocity_net_params: dict = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.modality_dims = modality_dims or {"context": 512}

        # Velocity network
        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg["modality_dims"] = self.modality_dims
        self.velocity_net = TransformerVelocityNet(**vn_cfg)

        # OT-CFM path
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log parameters
        n_params = sum(p.numel() for p in self.velocity_net.parameters())
        print(f"[BrainFlowDirectV2] VelocityNet: {n_params:,} params")

    def forward(
        self,
        modality_features: dict,
        target: torch.Tensor,
        subject_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute OT-CFM loss in fMRI space.

        Args:
            modality_features: dict of {mod_name: (B, T, dim_mod)}.
            target: (B, output_dim) ground truth fMRI.
            subject_idx: (B,) index of subjects.

        Returns:
            loss: scalar MSE(v_pred, dx_t).
        """
        x_1 = target
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        v_pred = self.velocity_net(
            x=sample_info.x_t,
            t=sample_info.t,
            cond=modality_features,
            subject_idx=subject_idx,
        )

        return F.mse_loss(v_pred, sample_info.dx_t)

    @torch.inference_mode()
    def synthesise(
        self,
        modality_features: dict,
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
        subject_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE with multimodal conditioning.

        Args:
            modality_features: dict of {mod_name: (B, T, dim_mod)}.
            n_timesteps: Number of ODE solver steps.
            solver_method: ODE solver method.

        Returns:
            fmri_pred: (B, output_dim) predicted fMRI.
        """
        # Get B and device from first modality
        first_feat = next(iter(modality_features.values()))
        B = first_feat.shape[0]
        device = first_feat.device
        dtype = first_feat.dtype

        # Start from mode of prior (zeros for best PCC)
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
            cond=modality_features,
            subject_idx=subject_idx,
        )

        return fmri_pred
