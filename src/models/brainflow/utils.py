import torch
import torch.nn as nn
import torch.nn.functional as F

def recover_velocity_indi(
    u: torch.Tensor,
    t: torch.Tensor,
    min_denom: float = 1e-3,
) -> torch.Tensor:
    """Map InDI-trained output u ≈ (1-t)(x_1-x_0) to OT velocity v = x_1 - x_0.

    ``t`` may be scalar (ODE solvers) or (B,) matching batch of ``u`` (B, D).
    """
    if t.dim() == 0:
        denom = (1.0 - t).clamp(min=min_denom)
        return u / denom
    denom = (1.0 - t).clamp(min=min_denom)
    while denom.dim() < u.dim():
        denom = denom.unsqueeze(-1)
    return u / denom


class IndiVelocityCallable:
    """Wraps ``VelocityNet`` for ODE integration without duplicating parameters."""

    def __init__(self, net: nn.Module, min_denom: float = 1e-3):
        self.net = net
        self.min_denom = min_denom

    def __call__(self, *, x, t, **kwargs):
        u = self.net(x=x, t=t, **kwargs)
        return recover_velocity_indi(u, t, self.min_denom)


def flow_train_time_sample(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sqrt_bias_end: bool,
) -> torch.Tensor:
    """Sample t in [0, 1]. If ``sqrt_bias_end``, use t = sqrt(U) for more mass near t→1."""
    u = torch.rand(batch_size, device=device, dtype=dtype)
    if sqrt_bias_end:
        return torch.sqrt(u)
    return u


def info_nce_loss(z_pred, z_target, temperature=0.07):
    """Bidirectional InfoNCE loss in projected space.

    Args:
        z_pred:   (B, D) L2-normalized prediction embeddings.
        z_target: (B, D) L2-normalized target embeddings.
        temperature: softmax temperature.

    Returns:
        loss: scalar, average of pred→target and target→pred NCE.
    """
    logits = z_pred @ z_target.T / temperature  # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.T, labels)
    return (loss_p2t + loss_t2p) / 2
