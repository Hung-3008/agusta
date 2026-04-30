"""Mamba-based backbone for BrainFlow flow matching.

MambaFlowBlock: Bidirectional Mamba SSM + Cross-Attention + FFN,
    with AdaLN-Zero conditioning (same 9-param scheme as DiTXBlock).

MambaFlowBackbone: Stack of MambaFlowBlock with plug-compatible interface
    matching DiTXBackbone.forward(h, t_emb, context_encoded).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .mamba_block import MambaBlock, MambaConfig
from .components import CrossAttention, RotaryEmbedding, modulate


class MambaFlowBlock(nn.Module):
    """Bidirectional Mamba + Cross-Attention + FFN with AdaLN-Zero.

    Architecture per block:
        1. Bidirectional SSM (forward + backward Mamba, merged via linear proj)
        2. Cross-Attention (Q from target, K/V from context, with RoPE)
        3. FFN (2-layer MLP with GELU)

    All three sub-layers use AdaLN-Zero modulation from t_emb (time + subject).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        # --- AdaLN-Zero: 9 modulation params (3 sub-layers × shift/scale/gate) ---
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 9 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # --- Layer norms (no affine — modulated by AdaLN) ---
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)  # SSM
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)  # cross-attn
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)  # FFN

        # --- Bidirectional Mamba SSM ---
        mamba_cfg = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
        )
        self.mamba_fwd = MambaBlock(mamba_cfg)
        self.mamba_bwd = MambaBlock(mamba_cfg)
        # Merge forward + backward outputs
        self.merge_proj = nn.Linear(2 * d_model, d_model, bias=False)
        nn.init.zeros_(self.merge_proj.weight)  # zero-init for residual-safe startup

        # --- Cross-Attention (reuse existing component with RoPE) ---
        self.cross_attn = CrossAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            rotary_emb=rotary_emb,
        )

        # --- FFN ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """MambaFlow block forward.

        Args:
            x:       (B, T, D) target tokens (noisy fMRI at timestep t).
            t_emb:   (B, D) time + subject embedding.
            context: (B, T_ctx, D) encoded context tokens.

        Returns:
            x: (B, T, D) updated hidden state.
        """
        # Generate 9 modulation params
        mod = self.adaLN_modulation(t_emb)  # (B, 9*D)
        chunks = mod.chunk(9, dim=-1)
        shift_ssm, scale_ssm, gate_ssm = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_ffn, scale_ffn, gate_ffn = chunks[6], chunks[7], chunks[8]

        # 1. Bidirectional Mamba SSM
        x_norm = modulate(self.norm1(x), shift_ssm, scale_ssm)
        out_fwd = self.mamba_fwd(x_norm)                            # (B, T, D)
        out_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])  # (B, T, D)
        merged = self.merge_proj(torch.cat([out_fwd, out_bwd], dim=-1))  # (B, T, D)
        x = x + gate_ssm.unsqueeze(1) * merged

        # 2. Cross-Attention
        x_norm_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_out = self.cross_attn(x_norm_cross, context)
        x = x + gate_cross.unsqueeze(1) * cross_out

        # 3. FFN
        x_norm_ffn = modulate(self.norm3(x), shift_ffn, scale_ffn)
        x = x + gate_ffn.unsqueeze(1) * self.ffn(x_norm_ffn)

        return x


class MambaFlowBackbone(nn.Module):
    """Stack of MambaFlowBlock — plug-compatible with DiTXBackbone.

    Interface: forward(h, t_emb, context_encoded) → h
    Same signature as DiTXBackbone and DiT1DBackbone.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        time_dim: int,
        rotary_emb: RotaryEmbedding,
        dit_depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList([
            MambaFlowBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                time_dim=time_dim,
                rotary_emb=rotary_emb,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
            )
            for _ in range(dit_depth)
        ])

    def forward(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor,
        context_encoded: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through all blocks.

        Args:
            h:               (B, T, D) input tokens.
            t_emb:           (B, D) time + subject embedding.
            context_encoded: (B, T_ctx, D) encoded context.

        Returns:
            h: (B, T, D) output tokens.
        """
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, t_emb, context_encoded, use_reentrant=False)
            else:
                h = block(h, t_emb, context_encoded)
        return h
