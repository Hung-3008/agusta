# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Adapted from facebookresearch/DiT for 1D sequence flow matching.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .components import RotaryEmbedding, apply_rotary_emb, CrossAttention


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift.

    Copied from facebookresearch/DiT.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                             Core DiT Blocks                                   #
#################################################################################


class DiTBlock(nn.Module):
    """Standard DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Faithfully adapted from facebookresearch/DiT. Uses inline attention
    (no timm dependency) with qkv_bias=True and GELU(approximate='tanh').
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Self-attention (inline, equivalent to timm.Attention with qkv_bias=True)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # FFN (inline, equivalent to timm.Mlp with approx GELU)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # adaLN-Zero: 6 modulation parameters (shift, scale, gate) × 2 sub-layers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input tokens.
            c: (B, D) conditioning vector (timestep + optional label).
        Returns:
            (B, T, D) output tokens.
        """
        B, T, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.attn_out_proj(attn_out)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # FFN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTCrossBlock(nn.Module):
    """DiT block with cross-attention for temporal context conditioning.

    Extends the standard DiTBlock with a cross-attention sub-layer between
    self-attention and FFN. adaLN-Zero uses 8 parameters:
    - 6 from standard DiT (shift/scale/gate × SA + shift/scale/gate × FFN)
    - 2 for cross-attention (shift/scale, no gate — cross-attention is always active)

    The cross-attention uses RoPE on Q/K for temporal alignment between
    target tokens and context tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        rotary_emb: RotaryEmbedding = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Cross-attention (Q from target, K/V from context)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=0.0,
            rotary_emb=rotary_emb,
        )

        # FFN
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # adaLN-Zero: 8 params = (shift, scale, gate) × SA + (shift, scale) × XA + (shift, scale, gate) × FFN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 8 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, T, D) target tokens.
            c:       (B, D) conditioning vector (timestep).
            context: (B, T_ctx, D) encoded context tokens.
        Returns:
            (B, T, D) output tokens.
        """
        B, T, D = x.shape

        # Generate 8 modulation parameters
        mods = self.adaLN_modulation(c).chunk(8, dim=1)
        shift_msa, scale_msa, gate_msa = mods[0], mods[1], mods[2]
        shift_cross, scale_cross = mods[3], mods[4]
        shift_mlp, scale_mlp, gate_mlp = mods[5], mods[6], mods[7]

        # 1. Self-Attention with adaLN-Zero
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.attn_out_proj(attn_out)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. Cross-Attention with adaLN (no gate — always active)
        x_norm_cross = modulate(self.norm_cross(x), shift_cross, scale_cross)
        cross_out = self.cross_attn(x_norm_cross, context)
        x = x + cross_out

        # 3. FFN with adaLN-Zero
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT.

    Copied from facebookresearch/DiT. Adapted for 1D output (no unpatchify).
    Uses adaLN (shift + scale, no gate) followed by a zero-initialized linear.
    """

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                          High-level Backbone Wrapper                          #
#################################################################################


class DiTBackbone(nn.Module):
    """Stack of DiTCrossBlock layers with gradient checkpointing support.

    This is the decoder backbone for VelocityNet. Each block receives:
    - x: target tokens (noisy fMRI)
    - c: conditioning vector (timestep embedding)
    - context: encoded multimodal context tokens
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        rotary_emb: RotaryEmbedding = None,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList([
            DiTCrossBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                rotary_emb=rotary_emb,
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, c, context, use_reentrant=False)
            else:
                x = block(x, c, context)
        return x


#################################################################################
#                        TimeDiT Blocks (AdaLN, no gate)                        #
#################################################################################


class TimeDiTBlock(nn.Module):
    """TimeDiT block: Self-Attention + FFN with AdaLN conditioning (no gate).

    Unlike DiTBlock/DiTCrossBlock which use adaLN-Zero (shift, scale, gate),
    TimeDiT uses plain AdaLN (shift, scale only) — the condition always
    modulates the hidden state without a learnable gate.

    No cross-attention: context is injected entirely through the AdaLN
    condition vector ``c``, which must already include pooled context info.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # FFN with SiLU (following TimeDiT paper)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # AdaLN modulation: 4 params = (shift, scale) × SA + (shift, scale) × FFN
        # No gate — condition always active (unlike adaLN-Zero's 6 params)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input tokens.
            c: (B, D) conditioning vector (timestep + subject + pooled context).
        Returns:
            (B, T, D) output tokens.
        """
        B, T, D = x.shape
        shift_msa, scale_msa, shift_mlp, scale_mlp = (
            self.adaLN_modulation(c).chunk(4, dim=1)
        )

        # Self-Attention with AdaLN (no gate)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.attn_out_proj(attn_out)
        x = x + attn_out  # no gate

        # FFN with AdaLN (no gate)
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))  # no gate
        return x


class TimeDiTBackbone(nn.Module):
    """Stack of TimeDiTBlock layers with context pooling for AdaLN injection.

    This is a drop-in replacement for DiTBackbone. The key difference:
    - DiTBackbone uses per-token cross-attention at every block
    - TimeDiTBackbone pools context into a single vector and injects via AdaLN

    The forward signature matches DiTBackbone exactly:
        forward(x, c, context) -> x

    Internally, context (B, T_ctx, D) is mean-pooled → MLP → added to c
    before being passed as the AdaLN condition to each TimeDiTBlock.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        rotary_emb=None,  # accepted for API compat, unused
    ):
        super().__init__()
        self.gradient_checkpointing = False

        # Context pooling: mean-pool temporal tokens → MLP → condition vector
        self.context_pool_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList([
            TimeDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, T, D) target tokens (already includes additive context).
            c:       (B, D) conditioning vector (timestep + subject embedding).
            context: (B, T_ctx, D) encoded context tokens.
        Returns:
            (B, T, D) output tokens.
        """
        # Pool context → combine with timestep/subject conditioning
        c_ctx = self.context_pool_proj(context.mean(dim=1))  # (B, D)
        c_combined = c + c_ctx  # (B, D)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, c_combined, use_reentrant=False)
            else:
                x = block(x, c_combined)
        return x
