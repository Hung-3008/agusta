import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .components import RotaryEmbedding, apply_rotary_emb, modulate, RMSNormLastDim, CrossAttention

class DiT1DBlock(nn.Module):
    """1D DiT block: AdaLN-Zero + RoPE self-attention + FFN (temporal consistency on T_target)."""

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

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

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
            q,
            k,
            v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_norm2)
        return x


class DiTXBlock(nn.Module):
    """DiT-X block: AdaLN-Zero on Self-Attention + Cross-Attention + FFN.

    Adapted from ManiFlow. Key difference from DiT1DBlock:
    - 9 modulation parameters (3×3) instead of 6 (3×2)
    - Cross-attention at every block with AdaLN-Zero gating
    - Context is queried every layer (not added once at start)
    - RoPE on both Self-Attention and Cross-Attention Q/K
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

        # Full Attention Residuals over decoder depth.
        # Zero-init ensures near-uniform depth attention at initialization.
        self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.attn_res_proj.weight)
        self.attn_res_norm = RMSNormLastDim(d_model)

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
            rotary_emb=rotary_emb,  # RoPE on Q/K for temporal alignment
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x_history: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DiT-X forward with Full Attention Residuals over depth.

        Args:
            x_history: (L, B, T, D) history of values over decoder depth.
                       Contains initial token state and prior block values.
            t_emb:   (B, D) time embedding.
            context: (B, T_ctx, D) encoded context tokens.

        Returns:
            h_out:      (B, T, D) updated hidden state after this block.
            block_value:(B, T, D) gated block contribution to append to history.
        """
        if x_history.dim() == 3:
            x_history = x_history.unsqueeze(0)

        if x_history.dim() != 4:
            raise ValueError(f"x_history must be (L,B,T,D), got shape {tuple(x_history.shape)}")

        # 1) Full Attention Residuals across depth.
        # Keys are RMS-normalized to prevent magnitude bias to late layers.
        k_res = self.attn_res_norm(x_history)                            # (L,B,T,D)
        logits = self.attn_res_proj(k_res).squeeze(-1)                   # (L,B,T)
        alpha = torch.softmax(logits, dim=0)                             # depth-softmax
        x = torch.einsum("lbt,lbtd->btd", alpha, x_history)             # (B,T,D)

        B, T, D = x.shape

        # Generate 9 modulation parameters from time embedding
        modulation = self.adaLN_modulation(t_emb)  # (B, 9*D)
        chunks = modulation.chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # 1. Self-Attention with AdaLN-Zero
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
        msa_value = gate_msa.unsqueeze(1) * attn_out
        x = x + msa_value

        # 2. Cross-Attention with AdaLN-Zero
        x_norm_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_out = self.cross_attn(x_norm_cross, context)
        cross_value = gate_cross.unsqueeze(1) * cross_out
        x = x + cross_value

        # 3. FFN with AdaLN-Zero
        x_norm_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_value = gate_mlp.unsqueeze(1) * self.ffn(x_norm_mlp)
        x = x + mlp_value

        block_value = msa_value + cross_value + mlp_value

        return x, block_value


# =============================================================================
# High-level Backbone Wrappers
# =============================================================================

class DiTXBackbone(nn.Module):
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
        history = h.unsqueeze(0)  # (1, B, T, D)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h, block_value = checkpoint(
                    block, history, t_emb, context_encoded, use_reentrant=False
                )
            else:
                h, block_value = block(history, t_emb, context_encoded)
            history = torch.cat([history, block_value.unsqueeze(0)], dim=0)
        return h


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
