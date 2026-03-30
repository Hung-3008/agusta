"""BrainFlow CSFM — Condition-dependent Source Flow Matching for fMRI.

Adapted from CSFM (https://arxiv.org/abs/2602.05951).

Key idea: Instead of random Gaussian noise as source x₀, learn a
condition-dependent source distribution via a variational encoder (SourceVE).

Architecture:
  Context (B, T, [mod_dims...])
    → MultiTokenFusion: per-mod proj + modality_emb → (B, T, hidden)
    → Temporal Self-Attention → context_encoded

  SourceVE(context_encoded)
    → Cross-attention with learnable queries → μ, log_var → x₀

  Training:
    t ~ shifted_uniform,  xt = (1-t)*x₁ + t*x₀,  ut = x₀ - x₁
    v_pred = VelocityNet(xt, t, context_encoded)
    loss = MSE(v_pred, ut) + kld_weight * KLD + align_weight * align

  Inference:
    x₀ = SourceVE(context) → ODE solve → x₁ (fMRI)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transport import create_transport, Sampler


# =============================================================================
# Building Blocks
# =============================================================================


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

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
    """Per-subject linear output head (from TRIBE/Brain-Diffuser)."""

    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels)) if bias else None
        self.weights.data.normal_(0, 1.0 / in_channels**0.5)
        if self.bias is not None:
            self.bias.data.normal_(0, 1.0 / in_channels**0.5)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        w = self.weights[subject_ids]
        out = torch.einsum("bd,bdo->bo", x, w)
        if self.bias is not None:
            out = out + self.bias[subject_ids]
        return out


# =============================================================================
# MultiTokenFusion — per-modality projection + modality embeddings
# =============================================================================


class MultiTokenFusion(nn.Module):
    """Per-modality projection + modality embeddings, then average across mods.

    Output: (B, T, hidden_dim)
    """

    def __init__(
        self,
        modality_dims: list[int],
        hidden_dim: int = 1024,
        max_seq_len: int = 11,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.modality_dims = modality_dims
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
            torch.randn(self.n_modalities, hidden_dim) * 0.02
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        B, T = modality_features[0].shape[:2]
        T = min(T, self.max_seq_len)

        projected = []
        for i, (feat, proj) in enumerate(zip(modality_features, self.projectors)):
            h = proj(feat[:, :T])
            h = h + self.modality_emb[i]
            projected.append(h)

        if self.training and self.modality_dropout > 0:
            keep_mask = (
                torch.rand(B, 1, self.n_modalities, device=projected[0].device)
                > self.modality_dropout
            )
            all_dropped = keep_mask.sum(dim=2, keepdim=True) == 0
            keep_mask[:, :, 0:1] = torch.max(keep_mask[:, :, 0:1], all_dropped)
            
            # Scale to preserve expected magnitude between train and eval!
            scale = 1.0 / (1.0 - self.modality_dropout)
            for i in range(self.n_modalities):
                projected[i] = projected[i] * keep_mask[:, :, i : i + 1] * scale

        x = torch.stack(projected, dim=0).mean(dim=0)
        return self.output_proj(x)


# =============================================================================
# SimpleFiLMBlock — FiLM (time) + FFN + CrossAttn
# =============================================================================


class SimpleFiLMBlock(nn.Module):
    """Residual block: FiLM conditioning + FFN + cross-attention."""

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

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale_shift = self.film(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.ffn(h)

        q = self.norm_q(x).unsqueeze(1)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = x + attn_out.squeeze(1)
        return x


# =============================================================================
# VelocityNet — context encoder + velocity estimation
# =============================================================================


class VelocityNet(nn.Module):
    """Velocity network: MultiTokenFusion → Temporal Self-Attention → FiLM blocks.

    This serves as the "DiT" backbone in the CSFM framework.
    """

    def __init__(
        self,
        output_dim: int = 1000,
        hidden_dim: int = 1024,
        modality_dims: list[int] = None,
        n_blocks: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        max_seq_len: int = 31,
        n_subjects: int = 4,
        temporal_attn_layers: int = 2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dims = modality_dims or [1408]

        # --- Fusion ---
        self.fusion_block = MultiTokenFusion(
            modality_dims=self.modality_dims,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

        # --- Temporal position + self-attention ---
        self.context_pos_emb = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=temporal_attn_layers,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # --- Input projection (xt → hidden) ---
        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        # --- Time embedding ---
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Subject embedding ---
        self.subject_emb = nn.Embedding(n_subjects, hidden_dim)

        # --- Velocity blocks ---
        self.blocks = nn.ModuleList([
            SimpleFiLMBlock(hidden_dim, hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # --- Output (zero-init for stable start) ---
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.constant_(self.output_layer.weight, 0)
        nn.init.constant_(self.output_layer.bias, 0)

    def encode_context(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode concatenated context → fused temporal representation.

        Args:
            cond: (B, T, total_context_dim) pre-concatenated modalities.

        Returns:
            (B, T, hidden_dim) refined context.
        """
        splits = []
        offset = 0
        for dim in self.modality_dims:
            splits.append(cond[:, :, offset : offset + dim])
            offset += dim

        context = self.fusion_block(splits)
        T = context.shape[1]
        context = context + self.context_pos_emb[:, :T, :]
        context = self.temporal_attn(context)
        context = self.temporal_norm(context)
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
        """Predict velocity v(xt, t, context).

        Args:
            x:                   (B, output_dim) noisy state xt.
            t:                   (B,) timestep in [0, 1].
            cond:                (B, T, total_dim) raw context (optional).
            pre_encoded_context: (B, T, hidden_dim) pre-encoded context.
            subject_ids:         (B,) long tensor of subject indices.

        Returns:
            v_pred: (B, output_dim) predicted velocity.
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # Context
        if pre_encoded_context is not None:
            context_encoded = pre_encoded_context
        elif cond is not None:
            context_encoded = self.encode_context(cond)
        else:
            context_encoded = torch.zeros(
                x.shape[0], 1, self.hidden_dim, device=x.device, dtype=x.dtype
            )

        # Embeddings
        t_emb = self.time_mlp(SinusoidalPosEmb(self.hidden_dim)(t))
        if subject_ids is not None:
            t_emb = t_emb + self.subject_emb(subject_ids)

        # Forward
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb, context_encoded)

        h = self.final_norm(h)
        return self.output_layer(h)


# =============================================================================
# SourceVE — Variational Encoder for learned source distribution
# =============================================================================


class SourceVE(nn.Module):
    """Variational encoder that maps encoded context → learned source x₀.

    Architecture: learnable queries → cross-attention with context → self-attn → mean/logvar heads.
    Uses 16 query tokens → flatten → linear → output_dim.

    This replaces the random Gaussian source in standard flow matching.
    """

    def __init__(
        self,
        context_dim: int = 1024,
        output_dim: int = 1000,
        hidden_dim: int = 1024,
        depth: int = 4,
        num_heads: int = 8,
        num_queries: int = 16,
        dropout: float = 0.1,
        use_variational: bool = True,
        init_logvar: float = 1.0,
        fixed_std: float = None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.use_variational = use_variational
        self.fixed_std = fixed_std

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.query_pos_emb = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # Input projection (context_dim → hidden_dim if different)
        self.input_proj = nn.Linear(context_dim, hidden_dim) if context_dim != hidden_dim else nn.Identity()

        # Perceiver layers: cross-attn + self-attn + FFN
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(hidden_dim),
                "norm_kv": nn.LayerNorm(hidden_dim),
                "cross_attn": nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                "norm_sa": nn.LayerNorm(hidden_dim),
                "self_attn": nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                "norm_ffn": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

        # Output heads: queries → mean pool → mean/logvar → output_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, output_dim)

        if use_variational and fixed_std is None:
            self.log_var_head = nn.Linear(hidden_dim, output_dim)
        else:
            self.log_var_head = None

        self._init_weights(init_logvar)

    def _init_weights(self, init_logvar):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.log_var_head is not None:
            nn.init.normal_(self.log_var_head.weight, std=1e-4)
            nn.init.constant_(self.log_var_head.bias, init_logvar)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Map encoded context → source distribution parameters.

        Args:
            context: (B, T, context_dim) encoded context from VelocityNet.

        Returns:
            x0:      (B, output_dim) sampled source (reparameterized).
            mu:      (B, output_dim) mean.
            log_var: (B, output_dim) or None if not variational.
        """
        B = context.shape[0]

        # Project context to hidden_dim
        kv = self.input_proj(context)

        # Initialize queries
        queries = self.query_tokens.expand(B, -1, -1) + self.query_pos_emb

        # Perceiver layers
        for layer in self.layers:
            # Cross-attention
            q_norm = layer["norm_q"](queries)
            kv_norm = layer["norm_kv"](kv)
            attn_out, _ = layer["cross_attn"](q_norm, kv_norm, kv_norm)
            queries = queries + attn_out

            # Self-attention
            sa_out, _ = layer["self_attn"](layer["norm_sa"](queries), layer["norm_sa"](queries), layer["norm_sa"](queries))
            queries = queries + sa_out

            # FFN
            queries = queries + layer["ffn"](layer["norm_ffn"](queries))

        # Output: mean pool all query tokens → mean/logvar
        queries = self.norm(queries)
        pooled = queries.mean(dim=1)  # (B, hidden_dim)

        mu = self.mean_head(pooled)  # (B, output_dim)

        if self.use_variational:
            if self.fixed_std is not None:
                log_var = torch.full_like(mu, math.log(self.fixed_std**2))
            else:
                log_var = self.log_var_head(pooled)
        else:
            log_var = None

        # Reparameterize
        if log_var is not None and self.training:
            std = torch.exp(0.5 * log_var)
            x0 = mu + torch.randn_like(mu) * std
        else:
            x0 = mu

        return x0, mu, log_var


# =============================================================================
# KLD and Align losses (from CSFM)
# =============================================================================


def var_kld_loss(mu: torch.Tensor, log_var: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """Variance-only KLD loss (CSFM default): regularize var without penalizing mu.

    KLD = -0.5 * mean(1 + log_var - var), scaled to target_std.
    """
    var = log_var.exp()
    if target_std != 1.0:
        sigma2_star = target_std**2
        var = var / sigma2_star
        log_var = log_var - math.log(sigma2_star)
    return -0.5 * torch.mean(1 + log_var - var)





# =============================================================================
# BrainFlowCSFM — Top-level model
# =============================================================================


class BrainFlowCSFM(nn.Module):
    """CSFM-style Flow Matching for fMRI generation.

    Components:
      1. VelocityNet: context encoder + velocity estimator (our "DiT")
      2. SourceVE: variational encoder producing learned source x₀
      3. Transport: CSFM path sampling + timestep scheduling

    Loss = diffusion_loss + kld_weight * kld_loss + align_weight * align_loss
    """

    def __init__(
        self,
        output_dim: int = 1000,
        velocity_net_params: dict = None,
        n_subjects: int = 4,
        source_ve_params: dict = None,
        kld_weight: float = 3.0,
        kld_target_std: float = 1.0,
        detach_ut: bool = False,
        transport_params: dict = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.kld_weight = kld_weight
        self.kld_target_std = kld_target_std
        self.detach_ut = detach_ut

        # --- Velocity Network ---
        vn_cfg = dict(velocity_net_params or {})
        vn_cfg.setdefault("output_dim", output_dim)
        vn_cfg.setdefault("n_subjects", n_subjects)
        self.velocity_net = VelocityNet(**vn_cfg)

        # --- Source Variational Encoder ---
        sve_cfg = dict(source_ve_params or {})
        sve_cfg.setdefault("context_dim", vn_cfg.get("hidden_dim", 1024))
        sve_cfg.setdefault("output_dim", output_dim)
        sve_cfg.setdefault("hidden_dim", vn_cfg.get("hidden_dim", 1024))
        self.source_ve = SourceVE(**sve_cfg)

        # --- CSFM Transport ---
        tp_cfg = dict(transport_params or {})
        tp_cfg.setdefault("path_type", "Linear")
        tp_cfg.setdefault("prediction", "velocity")
        tp_cfg.setdefault("time_dist_type", "uniform")
        tp_cfg.setdefault("time_dist_shift", 1.0)
        self.transport = create_transport(**tp_cfg)

        # Log parameter counts
        vn_params = sum(p.numel() for p in self.velocity_net.parameters())
        sve_params = sum(p.numel() for p in self.source_ve.parameters())
        total = vn_params + sve_params
        print(f"[BrainFlowCSFM] VelocityNet: {vn_params:,} params")
        print(f"[BrainFlowCSFM] SourceVE: {sve_params:,} params")
        print(f"[BrainFlowCSFM] Total: {total:,} params")
        print(f"[BrainFlowCSFM] Loss weights: kld={kld_weight}, detach_ut={detach_ut}")

    def compute_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Compute CSFM training loss.

        Args:
            context: (B, T, total_dim) concatenated multimodal features.
            target:  (B, output_dim) ground-truth fMRI (x₁).
            subject_ids: (B,) subject indices.

        Returns:
            dict with: total_loss, flow_loss, kld_loss, align_loss, reg_loss.
        """
        x1 = target

        # 1. Encode context (shared between velocity net and source VE)
        context_encoded = self.velocity_net.encode_context(context)

        # 2. Generate learned source x₀ via SourceVE
        x0, mu, log_var = self.source_ve(context_encoded)

        # 3. Sample timestep and interpolate
        t = self.transport.sample_timestep(x1)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

        # 4. Predict velocity
        v_pred = self.velocity_net(
            x=xt,
            t=t,
            pre_encoded_context=context_encoded,
            subject_ids=subject_ids,
        )

        # 5. Diffusion loss (optionally detach ut from source encoder)
        ut_target = ut.detach() if self.detach_ut else ut
        flow_loss = F.mse_loss(v_pred, ut_target)

        # 6. KLD loss (regularize source variance)
        if log_var is not None:
            kld_loss = var_kld_loss(mu, log_var, target_std=self.kld_target_std)
        else:
            kld_loss = torch.tensor(0.0, device=target.device)

        # 7. Total
        total_loss = flow_loss + self.kld_weight * kld_loss

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "kld_loss": kld_loss,
        }

    @torch.inference_mode()
    def synthesise(
        self,
        context: torch.Tensor,
        n_timesteps: int = 50,
        solver_method: str = "euler",
        subject_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate fMRI by solving ODE from learned source.

        Args:
            context:       (B, T, total_dim) concatenated context.
            n_timesteps:   ODE solver steps.
            solver_method: "euler", "midpoint", or "dopri5".
            subject_ids:   (B,) subject indices.

        Returns:
            fmri_pred: (B, output_dim) predicted fMRI.
        """
        # 1. Encode context
        context_encoded = self.velocity_net.encode_context(context)

        # 2. Generate source x₀ from SourceVE (use mean, no sampling)
        x0, _, _ = self.source_ve(context_encoded)

        # 3. Create ODE sampler
        sampler = Sampler(self.transport)
        sample_fn = sampler.sample_ode(
            sampling_method=solver_method,
            num_steps=n_timesteps,
        )

        # 4. Wrap velocity net for ODE solver interface
        def model_fn(x, t, **kwargs):
            return self.velocity_net(
                x=x, t=t,
                pre_encoded_context=context_encoded,
                subject_ids=subject_ids,
            )

        # 5. Solve ODE: x₀ → x₁
        trajectory = sample_fn(x0, model_fn)

        # Return final state (last timestep)
        return trajectory[-1]
