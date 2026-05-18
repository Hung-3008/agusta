import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .components import (
    RotaryEmbedding, apply_rotary_emb, modulate,
    CrossAttention, AttnResOperator,
)


# =============================================================================
# SwiGLU Feed-Forward Network
# =============================================================================

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020).

    FFN(x) = (SiLU(x @ W_gate) ⊙ x @ W_up) @ W_down

    Uses d_ff = ⌊8d/3⌋ (aligned to 8) for equivalent FLOPs to GELU 4d.
    All projections are bias-free following LLaMA convention.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        d_ff = ((d_model * 8 // 3 + 7) // 8) * 8  # 8/3 ratio, GPU-aligned
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# =============================================================================
# DiT-1D Block (legacy, with SwiGLU upgrade)
# =============================================================================

class DiT1DBlock(nn.Module):
    """1D DiT block: AdaLN-Zero + RoPE self-attention + SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.ffn = SwiGLUFFN(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(
            6, dim=-1
        )

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv_proj(x_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary_emb(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_norm2)
        return x


# =============================================================================
# DiT-Original Block (Peebles & Xie, 2022 — adapted for 1D)
# =============================================================================

class DiTOriginalBlock(nn.Module):
    """Original DiT block adapted for 1D sequences (Peebles & Xie, 2022).

    Faithful to the paper:
    - GELU MLP with 4× expansion (not SwiGLU)
    - No RoPE — positional info comes from input embedding only
    - Standard additive residuals (no Attention Residuals)
    - qkv_bias=True
    - LayerNorm eps=1e-6
    - adaLN-Zero with 6 params, zero-initialized
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,  # accepted for API compat, NOT used
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        # rotary_emb intentionally unused — original DiT has no in-attention pos encoding
        self.attn_dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # adaLN-Zero: 6 modulation params (shift, scale, gate) × 2 sub-layers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # Self-attention with qkv_bias=True (paper default)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # GELU MLP with 4× expansion (paper standard)
        mlp_hidden = int(d_model * 4)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(
            6, dim=-1
        )

        # Self-Attention (no RoPE)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv_proj(x_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = x + gate_msa.unsqueeze(1) * attn_out

        # FFN (GELU 4× expansion)
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        return x


class DiTOriginalBackbone(nn.Module):
    """Original DiT backbone adapted for 1D — sequential blocks, no AttnRes."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, time_dim, rotary_emb, dit_depth):
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList([
            DiTOriginalBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                time_dim=time_dim,
                rotary_emb=rotary_emb,
            )
            for _ in range(dit_depth)
        ])

    def forward(self, h, t_emb, context_encoded):
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, t_emb, use_reentrant=False)
            else:
                h = block(h, t_emb)
        return h


# =============================================================================
# DiT-X Block with Paper-Faithful Attention Residuals + SwiGLU
# =============================================================================

class DiTXBlock(nn.Module):
    """DiT-X block with sub-layer Attention Residuals and SwiGLU FFN.

    Key differences from the previous version:
    - 3 AttnResOperators (one per sub-layer) replace the single shared projection
    - Each sub-layer independently attends over depth history via learned pseudo-query
    - SwiGLU FFN replaces GELU FFN for stronger feature gating
    - Block AttnRes: block summaries are managed by DiTXBackbone

    Adapted from ManiFlow DiT-X + Attention Residuals (Kimi Team, 2026).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        # Per-sub-layer Attention Residual operators (paper-faithful)
        # Each has its own zero-initialized pseudo-query for learned depth routing
        self.attn_res_sa = AttnResOperator(d_model)
        self.attn_res_ca = AttnResOperator(d_model)
        self.attn_res_ffn = AttnResOperator(d_model)

        # Layer norms (no affine — modulated by AdaLN)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)  # self-attn
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)  # cross-attn
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)  # FFN

        # AdaLN-Zero: 9 modulation params (shift, scale, gate) × 3 sub-layers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 9 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # Self-attention
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # Cross-attention (Q from target, K/V from context)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            rotary_emb=rotary_emb,
        )

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(d_model, dropout=dropout)

    def forward(
        self,
        block_sources: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DiT-X forward with sub-layer Block Attention Residuals.

        Args:
            block_sources: (N_src, B, T, D) stack of completed block summaries
                           (embedding + prior blocks). No partial — managed internally.
            t_emb:         (B, D) time embedding.
            context:       (B, T_ctx, D) encoded context tokens.

        Returns:
            h_out:         (B, T, D) output hidden state for downstream use.
            block_summary: (B, T, D) sum of all 3 sub-layer outputs (block rep).
        """
        # Generate 9 modulation parameters from time embedding
        modulation = self.adaLN_modulation(t_emb)  # (B, 9*D)
        chunks = modulation.chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # ── Sub-layer 1: Self-Attention ──────────────────────────────────
        h_sa = self.attn_res_sa(block_sources)  # (B, T, D)
        B, T, D = h_sa.shape

        x_norm = modulate(self.norm1(h_sa), shift_msa, scale_msa)
        qkv = self.qkv_proj(x_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary_emb(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        sa_out = gate_msa.unsqueeze(1) * attn_out

        # Add partial (self-attn output) to sources for next sub-layer
        partial = sa_out
        sources_2 = torch.cat([block_sources, partial.unsqueeze(0)], dim=0)

        # ── Sub-layer 2: Cross-Attention ─────────────────────────────────
        h_ca = self.attn_res_ca(sources_2)
        x_norm_cross = modulate(self.norm2(h_ca), shift_cross, scale_cross)
        cross_out = self.cross_attn(x_norm_cross, context)
        ca_out = gate_cross.unsqueeze(1) * cross_out

        # Update partial: replace with cumulative sum
        partial = partial + ca_out
        sources_3 = torch.cat([block_sources, partial.unsqueeze(0)], dim=0)

        # ── Sub-layer 3: FFN ─────────────────────────────────────────────
        h_ffn = self.attn_res_ffn(sources_3)
        x_norm_mlp = modulate(self.norm3(h_ffn), shift_mlp, scale_mlp)
        ffn_out = gate_mlp.unsqueeze(1) * self.ffn(x_norm_mlp)

        # Block summary = sum of all 3 sub-layer gated outputs
        block_summary = sa_out + ca_out + ffn_out

        # Output hidden state = last AttnRes input + last sub-layer output
        h_out = h_ffn + ffn_out

        return h_out, block_summary


# =============================================================================
# High-level Backbone Wrappers
# =============================================================================

class DiTXBackbone(nn.Module):
    """DiT-X backbone with Block Attention Residuals.

    Manages block-level summaries across DiTXBlocks. Each block internally
    handles sub-layer AttnRes with 3 operators. The backbone maintains
    completed block summaries for inter-block depth routing.

    Final output = sum of all block sources (embedding + all block summaries),
    following the paper convention for aggregating depth information.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, time_dim, rotary_emb,
                 dit_depth, stochastic_depth_rate: float = 0.0):
        super().__init__()
        self.gradient_checkpointing = False
        self.stochastic_depth_rate = stochastic_depth_rate
        self.blocks = nn.ModuleList([
            DiTXBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                time_dim=time_dim,
                rotary_emb=rotary_emb,
            )
            for _ in range(dit_depth)
        ])

    def forward(self, h, t_emb, context_encoded):
        # Block sources: list of completed block summaries
        # b_0 = initial token embedding (always included)
        block_list = [h]  # list of (B, T, D) tensors
        depth = len(self.blocks)

        for i, block in enumerate(self.blocks):
            # Stochastic depth: linearly increasing drop probability across depth.
            # When dropped, the block's contribution to the final sum is zero;
            # the running tensor h is left unchanged so subsequent blocks see
            # an intact source list (with the dropped slot occupied by zeros).
            if (
                self.training
                and self.stochastic_depth_rate > 0
                and depth > 1
                and torch.rand(1).item() < i / (depth - 1) * self.stochastic_depth_rate
            ):
                block_list.append(torch.zeros_like(h))
                continue

            sources = torch.stack(block_list, dim=0)  # (N_src, B, T, D)
            if self.gradient_checkpointing and self.training:
                h, block_summary = checkpoint(
                    block, sources, t_emb, context_encoded, use_reentrant=False
                )
            else:
                h, block_summary = block(sources, t_emb, context_encoded)
            block_list.append(block_summary)

        # Final output: sum of all block sources (paper convention)
        return torch.stack(block_list, dim=0).sum(dim=0)


class DiT1DBackbone(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, time_dim, rotary_emb, dit_depth):
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList([
            DiT1DBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                time_dim=time_dim,
                rotary_emb=rotary_emb,
            )
            for _ in range(dit_depth)
        ])

    def forward(self, h, t_emb, context_encoded):
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, t_emb, use_reentrant=False)
            else:
                h = block(h, t_emb)
        return h


# =============================================================================
# DiT-Hybrid Block: Sparse Cross-Attention + SwiGLU + RoPE
# =============================================================================

class DiTHybridBlock(nn.Module):
    """Hybrid DiT block: Self-Attention + optional Cross-Attention + SwiGLU FFN.

    A middle ground between DiT-Original (no cross-attn) and DiT-X (full
    cross-attn + AttnRes at every block).

    Design:
    - SwiGLU FFN + RoPE (proven upgrades over GELU / fixed pos)
    - Standard additive residuals (no AttnRes → saves O(N²) memory)
    - Cross-Attention only at selected blocks (controlled by has_cross_attn)
    - Blocks without cross-attn: 6 adaLN params (Self-Attn + FFN)
    - Blocks with cross-attn: 9 adaLN params (Self-Attn + Cross-Attn + FFN)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,
        has_cross_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout
        self.has_cross_attn = has_cross_attn

        # Layer norms (no affine — modulated by AdaLN)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)  # self-attn
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False)  # FFN

        # AdaLN-Zero: 6 or 9 params depending on cross-attn presence
        n_mod = 9 if has_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, n_mod * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # Self-attention with RoPE
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # Cross-attention (only at selected blocks)
        if has_cross_attn:
            self.norm_cross = nn.LayerNorm(d_model, elementwise_affine=False)
            self.cross_attn = CrossAttention(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                rotary_emb=rotary_emb,
            )
        else:
            self.norm_cross = None
            self.cross_attn = None

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Parse adaLN modulation parameters
        modulation = self.adaLN_modulation(t_emb)
        if self.has_cross_attn:
            chunks = modulation.chunk(9, dim=-1)
            shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
            shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
            shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(
                6, dim=-1
            )

        # ── Sub-layer 1: Self-Attention with RoPE ────────────────────────
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv_proj(x_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary_emb(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = x + gate_msa.unsqueeze(1) * attn_out

        # ── Sub-layer 2 (optional): Cross-Attention ──────────────────────
        if self.has_cross_attn and context is not None:
            x_norm_cross = modulate(self.norm_cross(x), shift_cross, scale_cross)
            cross_out = self.cross_attn(x_norm_cross, context)
            x = x + gate_cross.unsqueeze(1) * cross_out

        # ── Sub-layer 3: SwiGLU FFN ─────────────────────────────────────
        x_norm_mlp = modulate(self.norm_ffn(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_norm_mlp)
        return x


class DiTHybridBackbone(nn.Module):
    """Hybrid backbone with sparse cross-attention every N blocks.

    Context signal is refreshed periodically instead of at every block,
    preventing signal vanishing in deep networks while keeping memory
    much lower than DiT-X (no AttnRes, fewer cross-attn layers).

    With dit_depth=24, cross_attn_every_n=4:
    - 6 blocks have cross-attention (blocks 3, 7, 11, 15, 19, 23)
    - 18 blocks are self-attention + FFN only
    - No Attention Residuals → O(1) memory per block instead of O(N)

    Stochastic depth (LayerDrop):
    - Drop probability increases linearly from 0 to ``stochastic_depth_rate``
    - Blocks with cross-attention are NEVER dropped (preserve context signal)
    - Dropped blocks pass input through unchanged (identity)
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        time_dim,
        rotary_emb,
        dit_depth,
        cross_attn_every_n: int = 4,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.stochastic_depth_rate = stochastic_depth_rate
        self._has_cross_attn = []  # track which blocks have cross-attn
        self.blocks = nn.ModuleList()
        for i in range(dit_depth):
            has_ca = ((i + 1) % cross_attn_every_n == 0)
            self._has_cross_attn.append(has_ca)
            self.blocks.append(
                DiTHybridBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    time_dim=time_dim,
                    rotary_emb=rotary_emb,
                    has_cross_attn=has_ca,
                )
            )

    def forward(self, h, t_emb, context_encoded):
        depth = len(self.blocks)
        for i, block in enumerate(self.blocks):
            # Stochastic depth: linearly increasing drop probability
            # Never drop cross-attention blocks to preserve context signal
            if (self.training and self.stochastic_depth_rate > 0
                    and not self._has_cross_attn[i]):
                drop_prob = i / max(depth - 1, 1) * self.stochastic_depth_rate
                if torch.rand(1).item() < drop_prob:
                    continue  # skip this block (identity)

            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, t_emb, context_encoded, use_reentrant=False)
            else:
                h = block(h, t_emb, context_encoded)
        return h


# =============================================================================
# DiT-Joint Block: MMDiT-style Joint Attention (SD3-inspired)
# =============================================================================

class DiTJointBlock(nn.Module):
    """MMDiT-style block with joint attention for bidirectional context evolution.

    Inspired by Stable Diffusion 3 (Esser et al., 2024). Target and context
    tokens have SEPARATE QKV projections (different modality spaces) but share
    a SINGLE attention operation. Both streams evolve at every layer.

    Design:
    - Separate QKV projections per stream (target vs context)
    - Joint self-attention on concatenated [target; context] sequence
    - RoPE with shared temporal positions (both streams represent same TR axis)
    - adaLN-Zero on target only (context not timestep-conditioned)
    - Separate SwiGLU FFN per stream (different feature distributions)
    - Standard additive residuals
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        # ── Layer norms (no affine — target modulated by adaLN) ──────────
        self.norm1_target = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm1_context = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2_target = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2_context = nn.LayerNorm(d_model)  # affine=True for context (no adaLN)

        # ── adaLN-Zero for target stream only (6 params) ────────────────
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # ── Separate QKV projections per modality ────────────────────────
        self.qkv_target = nn.Linear(d_model, 3 * d_model)
        self.qkv_context = nn.Linear(d_model, 3 * d_model)

        # ── Separate output projections ──────────────────────────────────
        self.out_proj_target = nn.Linear(d_model, d_model)
        self.out_proj_context = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # ── Separate SwiGLU FFN per stream ───────────────────────────────
        self.ffn_target = SwiGLUFFN(d_model, dropout=dropout)
        self.ffn_context = SwiGLUFFN(d_model, dropout=dropout)

    def forward(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MMDiT-style forward with joint attention.

        Args:
            target:  (B, T, D) target/fMRI tokens (being denoised).
            context: (B, C, D) context tokens (evolving conditioning).
            t_emb:   (B, D)   timestep embedding.

        Returns:
            target:  (B, T, D) updated target tokens.
            context: (B, C, D) updated context tokens.
        """
        B, T, D = target.shape
        C = context.shape[1]

        # Parse adaLN modulation (target only — context is NOT timestep-conditioned)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        )

        # ── Joint Attention ──────────────────────────────────────────────
        # Normalize: target with adaLN modulation, context with standard LN
        t_norm = modulate(self.norm1_target(target), shift_msa, scale_msa)
        c_norm = self.norm1_context(context)

        # Separate QKV projections (different modality spaces)
        qkv_t = self.qkv_target(t_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv_c = self.qkv_context(c_norm).reshape(B, C, 3, self.nhead, self.head_dim)
        qkv_t = qkv_t.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
        qkv_c = qkv_c.permute(2, 0, 3, 1, 4)  # (3, B, H, C, d)
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]
        q_c, k_c, v_c = qkv_c[0], qkv_c[1], qkv_c[2]

        # Apply RoPE to each stream with SHARED temporal positions
        # Both target and context represent the same TR-aligned axis
        cos_t, sin_t = self.rotary_emb(T)
        q_t = apply_rotary_emb(q_t, cos_t, sin_t)
        k_t = apply_rotary_emb(k_t, cos_t, sin_t)
        cos_c, sin_c = self.rotary_emb(C)
        q_c = apply_rotary_emb(q_c, cos_c, sin_c)
        k_c = apply_rotary_emb(k_c, cos_c, sin_c)

        # Concatenate along sequence dim for joint attention
        Q = torch.cat([q_t, q_c], dim=2)  # (B, H, T+C, d)
        K = torch.cat([k_t, k_c], dim=2)
        V = torch.cat([v_t, v_c], dim=2)

        # Single attention over combined sequence (bidirectional interaction)
        joint_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        joint_out = joint_out.transpose(1, 2)  # (B, T+C, H, d)

        # Split back into target and context streams
        target_attn = joint_out[:, :T].reshape(B, T, D)
        context_attn = joint_out[:, T:].reshape(B, C, D)

        # Separate output projections + gating
        target = target + gate_msa.unsqueeze(1) * self.out_proj_target(
            self.attn_drop(target_attn)
        )
        context = context + self.out_proj_context(
            self.attn_drop(context_attn)
        )

        # ── Separate FFN ─────────────────────────────────────────────────
        t_norm2 = modulate(self.norm2_target(target), shift_mlp, scale_mlp)
        target = target + gate_mlp.unsqueeze(1) * self.ffn_target(t_norm2)
        context = context + self.ffn_context(self.norm2_context(context))

        return target, context


class DiTJointBackbone(nn.Module):
    """MMDiT-style backbone with joint attention at every block.

    Target and context tokens co-evolve through depth via bidirectional
    joint attention. Context is NOT static — it adapts to the target's
    evolving state at every layer, enabling much stronger conditioning.

    Only target tokens are returned (context is internal state).
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, time_dim, rotary_emb, dit_depth):
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList([
            DiTJointBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                time_dim=time_dim,
                rotary_emb=rotary_emb,
            )
            for _ in range(dit_depth)
        ])

    def forward(self, h, t_emb, context_encoded):
        target = h
        context = context_encoded
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                target, context = checkpoint(
                    block, target, context, t_emb, use_reentrant=False
                )
            else:
                target, context = block(target, context, t_emb)
        return target  # only return evolved target tokens


