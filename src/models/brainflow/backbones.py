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

    def __init__(self, d_model, nhead, dim_feedforward, dropout, time_dim, rotary_emb, dit_depth):
        super().__init__()
        self.gradient_checkpointing = False
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

        for block in self.blocks:
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
