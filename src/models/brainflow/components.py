import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Copied from facebookresearch/DiT.
    Uses a fixed sinusoidal encoding of size ``frequency_embedding_size``
    followed by a two-layer MLP with SiLU activation.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class RotaryEmbedding(nn.Module):
    """Precomputes sin/cos rotation frequencies for RoPE."""

    def __init__(self, dim: int, max_seq_len: int = 128, theta: float = 10000.0):
        super().__init__()
        # dim is head_dim (hidden_dim // n_heads)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute for max_seq_len
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # (T, dim//2)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)  # (T, dim//2)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)  # (T, dim//2)

    def forward(self, seq_len: int):
        """Return (cos, sin) each of shape (seq_len, dim//2)."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to x of shape (B, H, T, D).

    cos, sin: (T, D//2) — broadcast over B, H.
    Splits x into even/odd pairs and applies 2D rotation.
    """
    # x: (B, H, T, D), cos/sin: (T, D//2) -> (1, 1, T, D//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-style modulation: x * (1 + scale) + shift. x is (B,T,D); shift/scale are (B,D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNormLastDim(nn.Module):
    """RMSNorm over the last dimension for tensors with arbitrary leading dims."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CrossAttention(nn.Module):
    """Cross-attention with RoPE on Q/K for temporally-aligned context.

    Adapted from ManiFlow DiT-X. Q comes from target/action tokens,
    K/V from context tokens. RoPE is applied when both sequences share
    the same temporal axis (e.g. BrainFlow's 50-token aligned windows).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        rotary_emb: RotaryEmbedding = None,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend from x (query) to context (key/value).

        Args:
            x:       (B, T_q, D) target/action tokens.
            context: (B, T_kv, D) context tokens.

        Returns:
            (B, T_q, D) attended output.
        """
        B, T_q, D = x.shape
        _, T_kv, _ = context.shape

        q = self.q_proj(x).reshape(B, T_q, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(B, T_kv, 2, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Apply RoPE to Q and K when context is temporally aligned with target
        if self.rotary_emb is not None:
            cos_q, sin_q = self.rotary_emb(T_q)
            q = apply_rotary_emb(q, cos_q, sin_q)
            cos_k, sin_k = self.rotary_emb(T_kv)
            k = apply_rotary_emb(k, cos_k, sin_k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T_q, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        return attn_out


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE on Q/K (pre-norm architecture)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer with RoPE attention.

        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape

        # --- Self-attention with RoPE ---
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv_proj(x_norm).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q, K (not V)
        cos, sin = self.rotary_emb(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention (uses FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        attn_out = self.out_proj(self.attn_drop(attn_out))
        x = residual + attn_out

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        return x
