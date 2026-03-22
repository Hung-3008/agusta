"""BrainFlow Direct v3 — Enhanced Flow Matching with 5 Architectural Improvements.

Key improvements over v2:
  1. Minibatch OT coupling (straighter flows for gaussian/zero modes)
  2. Spatial-Aware Decoder (self-attention on output parcel groups)
  3. MoE Gating for dynamic modality routing (handles missing modalities)
  4. Time-Adaptive Cross-Attention (time-modulated query)
  5. Divergence Regularization (smooth velocity field)

Usage:
    python src/train_brainflow_direct.py --config src/configs/brain_flow_direct_v3.yaml
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Building Blocks (shared with v2)
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t * 1000.0
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SubjectLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels)) if bias else None
        self.weights.data.normal_(0, 1.0 / in_channels ** 0.5)
        if self.bias is not None:
            self.bias.data.normal_(0, 1.0 / in_channels ** 0.5)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        w = self.weights[subject_ids]
        out = torch.einsum("bd,bdo->bo", x, w)
        if self.bias is not None:
            out = out + self.bias[subject_ids]
        return out


# =============================================================================
# [IMPROVEMENT 3] MoE Gated Cross-Modal Fusion
# =============================================================================

class MoEGatedCrossModalFusion(nn.Module):
    """Factored Cross-Modal Fusion with learned Mixture-of-Experts gating.

    The gate network learns to weight each modality based on its content.
    When a modality is missing (zero-padded), the gate learns to zero it out
    and redistribute capacity to remaining modalities.
    """

    def __init__(
        self,
        modality_dims: list[int],
        hidden_dim: int = 1024,
        max_seq_len: int = 11,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.modality_dropout = modality_dropout

        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for dim in modality_dims
        ])

        self.modality_emb = nn.Parameter(
            torch.randn(1, 1, self.n_modalities, hidden_dim) * 0.02
        )

        # MoE Gate: learns per-modality importance from projected features
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # Zero-init → uniform softmax at start
        nn.init.zeros_(self.gate_proj[-1].weight)
        nn.init.zeros_(self.gate_proj[-1].bias)

        self.cross_modal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        B, T = modality_features[0].shape[:2]
        T = min(T, self.max_seq_len)

        projected = []
        for feat, proj in zip(modality_features, self.projectors):
            projected.append(proj(feat[:, :T]))

        stacked = torch.stack(projected, dim=2)  # (B, T, M, D)
        stacked = stacked + self.modality_emb

        # MoE Gating: compute per-modality weights
        gate_input = stacked.mean(dim=1)  # (B, M, D) temporal average
        gate_logits = self.gate_proj(gate_input).squeeze(-1)  # (B, M)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, M)
        stacked = stacked * gate_weights[:, None, :, None]

        # Modality dropout (training only, after gating)
        if self.training and self.modality_dropout > 0:
            keep_mask = (
                torch.rand(B, 1, self.n_modalities, 1, device=stacked.device)
                > self.modality_dropout
            ).float()
            all_dropped = (keep_mask.sum(dim=2, keepdim=True) == 0).float()
            keep_mask[:, :, 0:1, :] = torch.max(
                keep_mask[:, :, 0:1, :], all_dropped
            )
            stacked = stacked * keep_mask

        x = stacked.reshape(B * T, self.n_modalities, self.hidden_dim)
        for layer in self.cross_modal_layers:
            x = layer(x)
        x = x.mean(dim=1)
        context = x.reshape(B, T, self.hidden_dim)
        return self.final_norm(context)


# =============================================================================
# [IMPROVEMENT 4] Time-Adaptive Cross-Attention Residual Block
# =============================================================================

class TimeAdaptiveCrossAttentionResBlock(nn.Module):
    """CrossAttentionResBlock with time-modulated query.

    At t≈0 (near source), the model needs coarse semantic features.
    At t≈1 (near target), it needs fine-grained detail.
    The time_q_film adapts what the cross-attention looks for at each timestep.
    """

    def __init__(self, dim: int, time_dim: int, context_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.film = nn.Linear(time_dim, dim * 2)

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
            kdim=context_dim, vdim=context_dim,
        )

        # Time-adaptive query modulation (FiLM on Q)
        self.time_q_film = nn.Linear(time_dim, dim * 2)
        # Zero-init for identity at start
        nn.init.zeros_(self.time_q_film.weight)
        nn.init.zeros_(self.time_q_film.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # FiLM + FFN
        scale_shift = self.film(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.ffn(h)

        # Time-adaptive cross-attention
        q = self.norm_q(x).unsqueeze(1)
        kv = self.norm_kv(context)

        # Modulate query with time: controls "what to look for" at each timestep
        q_scale, q_shift = self.time_q_film(t_emb).chunk(2, dim=-1)
        q = q * (1 + q_scale.unsqueeze(1)) + q_shift.unsqueeze(1)

        attn_out, _ = self.cross_attn(q, kv, kv)
        x = x + attn_out.squeeze(1)
        return x


# =============================================================================
# [IMPROVEMENT 2] Spatial Refinement Block
# =============================================================================

class SpatialRefinementBlock(nn.Module):
    """Self-attention on output parcel groups for spatial coherence.

    Reshapes the 1000-d output into n_groups spatial tokens, applies
    self-attention to capture inter-region dependencies, then reshapes back.
    Uses residual connection for stability.
    """

    def __init__(
        self,
        output_dim: int = 1000,
        n_groups: int = 10,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert output_dim % n_groups == 0
        self.n_groups = n_groups
        self.group_dim = output_dim // n_groups  # e.g., 100

        self.group_pos_emb = nn.Parameter(
            torch.randn(1, n_groups, self.group_dim) * 0.02
        )

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.group_dim,
                nhead=n_heads,
                dim_feedforward=self.group_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.group_dim)

        # Gate for residual (learnable blend, starts at 0 = pure residual)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = x.reshape(B, self.n_groups, self.group_dim)
        h = h + self.group_pos_emb
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        refined = h.reshape(B, -1)
        # Learnable residual gate (sigmoid starts at 0.5, but gate param starts at 0)
        alpha = torch.sigmoid(self.gate)
        return (1 - alpha) * x + alpha * refined


# =============================================================================
# SourcePredictor (reused from v2)
# =============================================================================

class SourcePredictor(nn.Module):
    """Predicts condition-dependent source x₀ for flow matching (CSFM)."""

    def __init__(self, hidden_dim: int, output_dim: int, n_subjects: int = 4, use_variational: bool = True):
        super().__init__()
        self.use_variational = use_variational
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mean_head = SubjectLayers(hidden_dim, output_dim, n_subjects, bias=True)
        if use_variational:
            self.log_var_head = nn.Linear(hidden_dim, output_dim)
            nn.init.normal_(self.log_var_head.weight, std=1e-4)
            nn.init.constant_(self.log_var_head.bias, -2.0)

    def forward(self, context: torch.Tensor, subject_ids: torch.Tensor = None):
        h = context.mean(dim=1)
        h = self.pool_proj(h)
        if subject_ids is None:
            subject_ids = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        mu = self.mean_head(h, subject_ids)
        if self.use_variational:
            log_var = self.log_var_head(h)
            log_var = torch.clamp(log_var, min=-10.0, max=2.0)
            if self.training:
                std = torch.exp(0.5 * log_var)
                x0 = mu + torch.randn_like(mu) * std
            else:
                x0 = mu
        else:
            log_var = None
            x0 = mu
        return x0, mu, log_var


# =============================================================================
# DirectVelocityNetV3 — All architectural improvements
# =============================================================================

class DirectVelocityNetV3(nn.Module):
    """Velocity network v3 with MoE Fusion, Time-Adaptive Attention, Spatial Decoder."""

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 1024,
        modality_dims: list[int] = None,
        n_blocks: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        cross_modal_n_layers: int = 2,
        max_seq_len: int = 11,
        n_subjects: int = 4,
        # V3 specific
        spatial_n_groups: int = 10,
        spatial_n_layers: int = 1,
        spatial_n_heads: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dims = modality_dims or [1408]

        # [IMPROVEMENT 3] MoE Gated Fusion
        self.fusion_block = MoEGatedCrossModalFusion(
            modality_dims=self.modality_dims,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            n_layers=cross_modal_n_layers,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

        self.context_pos_emb = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.subject_emb = nn.Embedding(n_subjects, hidden_dim)

        # [IMPROVEMENT 4] Time-Adaptive Cross-Attention blocks
        self.blocks = nn.ModuleList([
            TimeAdaptiveCrossAttentionResBlock(hidden_dim, hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.subject_output = SubjectLayers(hidden_dim, output_dim, n_subjects, bias=True)

        # Zero-init output
        nn.init.constant_(self.subject_output.weights, 0)
        if self.subject_output.bias is not None:
            nn.init.constant_(self.subject_output.bias, 0)

        # [IMPROVEMENT 2] Spatial Refinement
        self.spatial_refine = SpatialRefinementBlock(
            output_dim=output_dim,
            n_groups=spatial_n_groups,
            n_heads=spatial_n_heads,
            n_layers=spatial_n_layers,
            dropout=dropout,
        )

    def encode_context_from_cond(self, cond: torch.Tensor) -> torch.Tensor:
        splits = []
        offset = 0
        for dim in self.modality_dims:
            splits.append(cond[:, :, offset:offset + dim])
            offset += dim
        context = self.fusion_block(splits)
        T = context.shape[1]
        context = context + self.context_pos_emb[:, :T, :]
        return context

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        pre_encoded_context: torch.Tensor = None,
        subject_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        if pre_encoded_context is not None:
            context_encoded = pre_encoded_context
        elif cond is not None:
            context_encoded = self.encode_context_from_cond(cond)
        else:
            context_encoded = torch.zeros(
                x.shape[0], 1, self.hidden_dim, device=x.device, dtype=x.dtype
            )

        t_emb = self.time_mlp(SinusoidalPosEmb(self.hidden_dim)(t))
        if subject_ids is not None:
            t_emb = t_emb + self.subject_emb(subject_ids)

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb, context_encoded)

        h = self.final_norm(h)
        if subject_ids is not None:
            raw_pred = self.subject_output(h, subject_ids)
        else:
            avg_w = self.subject_output.weights.mean(dim=0)
            raw_pred = h @ avg_w
            if self.subject_output.bias is not None:
                raw_pred = raw_pred + self.subject_output.bias.mean(dim=0)

        # [IMPROVEMENT 2] Spatial refinement
        return self.spatial_refine(raw_pred)


# =============================================================================
# BrainFlowDirectV3 — Top-level model
# =============================================================================

class BrainFlowDirectV3(nn.Module):
    """BrainFlow v3 with all 5 architectural improvements."""

    def __init__(
        self,
        output_dim: int = 1000,
        velocity_net_params: dict = None,
        n_subjects: int = 4,
        source_predictor_params: dict = None,
        source_mode: str = "csfm",
        use_minibatch_ot: bool = False,
        div_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.source_mode = source_mode
        self.use_minibatch_ot = use_minibatch_ot and HAS_SCIPY
        self.div_reg_weight = div_reg_weight

        # Velocity network (V3)
        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg.setdefault("n_subjects", n_subjects)
        self.velocity_net = DirectVelocityNetV3(**vn_cfg)

        # Source predictor (CSFM mode)
        self.source_predictor = None
        self.align_weight = 0.0
        self.kld_weight = 0.0
        if source_mode == "csfm":
            sp_cfg = dict(source_predictor_params or {})
            self.align_weight = sp_cfg.pop("align_weight", 1.0)
            self.kld_weight = sp_cfg.pop("kld_weight", 0.1)
            self.align_loss_type = sp_cfg.pop("align_loss_type", "normalized_l2")
            self.source_noise_std = sp_cfg.pop("source_noise_std", 0.0)
            sp_cfg.setdefault("hidden_dim", vn_cfg.get("hidden_dim", 1024))
            sp_cfg.setdefault("output_dim", output_dim)
            sp_cfg.setdefault("n_subjects", n_subjects)
            self.source_predictor = SourcePredictor(**sp_cfg)
        else:
            self.source_noise_std = 0.0

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log parameters
        vn_p = sum(p.numel() for p in self.velocity_net.parameters())
        sp_p = sum(p.numel() for p in self.source_predictor.parameters()) if self.source_predictor else 0
        print(f"[BrainFlowDirectV3] source_mode={source_mode}, minibatch_ot={self.use_minibatch_ot}, div_reg={div_reg_weight}")
        print(f"[BrainFlowDirectV3] VelocityNet: {vn_p:,} params")
        if self.source_predictor:
            print(f"[BrainFlowDirectV3] SourcePredictor: {sp_p:,} params")
        print(f"[BrainFlowDirectV3] Total: {vn_p + sp_p:,} params")

    # [IMPROVEMENT 1] Minibatch OT
    def _apply_minibatch_ot(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Reorder x_0 via optimal transport to minimize total transport cost."""
        with torch.no_grad():
            cost = torch.cdist(x_1, x_0, p=2)  # (B, B)
            _, col_ind = linear_sum_assignment(cost.cpu().numpy())
        return x_0[col_ind]

    # [IMPROVEMENT 5] Divergence regularization
    def _compute_divergence_loss(self, v_pred: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Hutchinson's trace estimator for velocity field divergence."""
        eps = torch.randn_like(x_t)
        vjp = torch.autograd.grad(
            outputs=v_pred, inputs=x_t,
            grad_outputs=eps, create_graph=True, retain_graph=True,
        )[0]
        div_estimate = (vjp * eps).sum(dim=-1)
        return div_estimate.pow(2).mean()

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        context_encoded = self.velocity_net.encode_context_from_cond(context)

        x_1 = target
        if self.source_mode == "csfm":
            x_0, mu, log_var = self.source_predictor(context_encoded, subject_ids)
            if self.training and self.source_noise_std > 0:
                x_0 = x_0 + self.source_noise_std * torch.randn_like(x_0)
        elif self.source_mode == "gaussian":
            x_0 = torch.randn_like(x_1)
            mu, log_var = None, None
        else:
            x_0 = torch.zeros_like(x_1)
            mu, log_var = None, None

        # [IMPROVEMENT 1] Minibatch OT (only for non-CSFM)
        if self.use_minibatch_ot and self.source_mode != "csfm":
            x_0 = self._apply_minibatch_ot(x_0, x_1)

        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # [IMPROVEMENT 5] Divergence reg needs grad w.r.t. x_t
        x_t = sample_info.x_t
        if self.div_reg_weight > 0:
            x_t = x_t.detach().requires_grad_(True)

        v_pred = self.velocity_net(
            x=x_t,
            t=sample_info.t,
            pre_encoded_context=context_encoded,
            subject_ids=subject_ids,
        )

        flow_loss = F.mse_loss(v_pred, sample_info.dx_t)

        # Align loss
        if mu is not None:
            if self.align_loss_type == "normalized_l2":
                align_loss = F.mse_loss(F.normalize(mu, dim=-1), F.normalize(target, dim=-1))
            else:
                align_loss = F.mse_loss(mu, target)
        else:
            align_loss = torch.tensor(0.0, device=target.device)

        # Var-KLD loss
        if log_var is not None:
            kld_loss = 0.5 * (torch.exp(log_var) - log_var - 1.0).mean()
        else:
            kld_loss = torch.tensor(0.0, device=target.device)

        # [IMPROVEMENT 5] Divergence loss
        if self.div_reg_weight > 0:
            div_loss = self._compute_divergence_loss(v_pred, x_t)
        else:
            div_loss = torch.tensor(0.0, device=target.device)

        total_loss = (
            flow_loss
            + self.align_weight * align_loss
            + self.kld_weight * kld_loss
            + self.div_reg_weight * div_loss
        )

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "align_loss": align_loss,
            "kld_loss": kld_loss,
            "div_loss": div_loss,
        }

    @torch.inference_mode()
    def synthesise(
        self,
        context: torch.Tensor,
        n_timesteps: int = 50,
        solver_method: str = "midpoint",
        subject_ids: torch.Tensor = None,
        n_trials: int = 1,
    ) -> torch.Tensor:
        """Generate fMRI via ODE. Supports multi-trial averaging for CSFM."""
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        context_encoded = self.velocity_net.encode_context_from_cond(context)
        solver = ODESolver(velocity_model=self.velocity_net)
        T = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)

        # Multi-trial for CSFM
        if n_trials > 1 and self.source_mode == "csfm" and self.source_predictor is not None:
            h = context_encoded.mean(dim=1)
            h = self.source_predictor.pool_proj(h)
            sid = subject_ids if subject_ids is not None else torch.zeros(B, dtype=torch.long, device=device)
            mu = self.source_predictor.mean_head(h, sid)
            log_var = None
            if self.source_predictor.use_variational:
                log_var = self.source_predictor.log_var_head(h)
                log_var = torch.clamp(log_var, min=-10.0, max=2.0)
                std = torch.exp(0.5 * log_var)

            fmri_acc = torch.zeros(B, self.output_dim, device=device, dtype=dtype)
            for _ in range(n_trials):
                x_init = mu + torch.randn_like(mu) * std if log_var is not None else mu
                pred = solver.sample(
                    time_grid=T, x_init=x_init, method=solver_method,
                    step_size=1.0 / n_timesteps, return_intermediates=False,
                    pre_encoded_context=context_encoded, subject_ids=subject_ids,
                )
                fmri_acc += pred
            return fmri_acc / n_trials

        # Single-trial
        if self.source_mode == "csfm":
            x_init, _, _ = self.source_predictor(context_encoded, subject_ids)
        elif self.source_mode == "gaussian":
            x_init = torch.randn(B, self.output_dim, device=device, dtype=dtype)
        else:
            x_init = torch.zeros(B, self.output_dim, device=device, dtype=dtype)

        return solver.sample(
            time_grid=T, x_init=x_init, method=solver_method,
            step_size=1.0 / n_timesteps, return_intermediates=False,
            pre_encoded_context=context_encoded, subject_ids=subject_ids,
        )
