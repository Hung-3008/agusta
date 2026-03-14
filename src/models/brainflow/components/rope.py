import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        RoPE (Rotary Positional Embedding) implementation for 1D sequences.
        Args:
            dim: Dimension of each attention head (e.g., hidden_dim // n_heads)
            base: Base for frequency computation Default: 10000.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        """
        Args:
            x: Input tensor (B, ..., T, ..., D)
            seq_dim: The dimension representing the sequence length
        Returns:
            cos, sin: Cached trigonometric frequencies matching the sequence length.
        """
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            # Duplicate for complex rotation: (T, dim/2) -> (T, dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]  # (1, 1, T, D)
            self.sin_cached = emb.sin()[None, None, :, :]  # (1, 1, T, D)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies Rotary Position Embedding to Query and Key tensors.
    Assumes q, k are of shape (B, n_heads, T, head_dim)
    and cos, sin are broadcastable, e.g., (1, 1, T, head_dim)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

