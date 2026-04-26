import torch
import torch.nn as nn

def tensor_warp_schedule(
    gamma: torch.Tensor, t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension exponential time-warping schedule.

    lambda_t = (exp(gamma * t) - 1) / (exp(gamma) - 1)
    d_lambda  = gamma * exp(gamma * t) / (exp(gamma) - 1)

    Boundary conditions: lambda(0)=0, lambda(1)=1 for any gamma.
    When |gamma| -> 0 the limit is the standard OT schedule: lambda_t=t, d_lambda=1.

    Args:
        gamma: (B, D) per-dimension speed parameters (clamped internally).
        t:     (B,)   timestep in [0, 1].

    Returns:
        lambda_t:   (B, D) interpolation coefficients.
        d_lambda_t: (B, D) time-derivatives of lambda_t.
    """
    gamma = gamma.clamp(-5.0, 5.0)
    t = t.unsqueeze(-1)                       # (B, 1)

    exp_gt = torch.exp(gamma * t)             # (B, D)
    exp_g = torch.exp(gamma)                  # (B, D)
    denom = exp_g - 1.0                       # (B, D)

    lambda_t = (exp_gt - 1.0) / denom
    d_lambda_t = gamma * exp_gt / denom

    small = gamma.abs() < 1e-4
    lambda_t = torch.where(small, t.expand_as(gamma), lambda_t)
    d_lambda_t = torch.where(small, torch.ones_like(gamma), d_lambda_t)

    return lambda_t, d_lambda_t


class TimeWarpNet(nn.Module):
    """Predicts per-dimension (or per-group) speed parameters gamma from context.

    Zero-initialized output layer so gamma=0 at init, recovering standard OT.
    """

    def __init__(
        self,
        context_dim: int,
        output_dim: int,
        warp_hidden_dim: int = 512,
        n_groups: int | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_groups = n_groups

        out_features = n_groups if n_groups else output_dim
        if n_groups is not None:
            assert output_dim % n_groups == 0, (
                f"output_dim ({output_dim}) must be divisible by n_groups ({n_groups})"
            )
            self.group_size = output_dim // n_groups
        else:
            self.group_size = 1

        self.net = nn.Sequential(
            nn.Linear(context_dim, warp_hidden_dim),
            nn.GELU(),
            nn.Linear(warp_hidden_dim, warp_hidden_dim),
            nn.GELU(),
            nn.Linear(warp_hidden_dim, out_features),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_pooled: (B, context_dim) mean-pooled context.
        Returns:
            gamma: (B, output_dim) per-dimension speed parameters.
        """
        gamma = self.net(context_pooled)          # (B, out_features)
        if self.n_groups is not None:
            gamma = gamma.repeat_interleave(self.group_size, dim=-1)
        return gamma
