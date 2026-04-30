"""Mamba SSM block — pure PyTorch implementation.

Vendored and simplified from https://github.com/alxndrTL/mamba.py (MIT License).
Stripped: language model wrapper, CUDA backend, step/chunk inference.
Kept: MambaBlock with parallel scan selective SSM.

Input/Output: (B, L, D) → (B, L, D).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pscan import pscan


@dataclass
class MambaConfig:
    """Minimal Mamba configuration for BrainFlow backbone."""

    d_model: int
    d_state: int = 16       # N — SSM state expansion factor
    expand_factor: int = 2   # E — inner dim = E * d_model
    d_conv: int = 4          # local convolution kernel size
    dt_rank: int | str = "auto"

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    """Selective State Space Model block with parallel scan.

    Architecture (Figure 3 from Mamba paper):
        x → in_proj → split(x, z)
        x → Conv1d → SiLU → SSM → y
        z → SiLU → z
        output = out_proj(y * z)
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # Projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # Projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"dt_init={config.dt_init}")

        # dt bias (softplus-inverse of log-uniform samples)
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization: A = diag(1, 2, ..., N) per channel
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # Projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        _, L, _ = x.shape

        xz = self.in_proj(x)           # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)     # each (B, L, ED)

        # x branch: conv1d → SiLU → SSM
        x = x.transpose(1, 2)                      # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]               # depthwise conv over time
        x = x.transpose(1, 2)                      # (B, L, ED)
        x = F.silu(x)
        y = self._ssm(x)

        # z branch: gating
        z = F.silu(z)

        output = self.out_proj(y * z)   # (B, L, D)
        return output

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan with parallel scan.

        Args:
            x: (B, L, ED) — post-conv, post-activation
        Returns:
            y: (B, L, ED)
        """
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )

        # Δ projection: (ED, dt_rank) @ (B, L, dt_rank)^T → (B, ED, L)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)
        delta = delta.transpose(1, 2)                      # (B, L, ED)
        delta = F.softplus(delta + self.dt_proj.bias)       # (B, L, ED)

        # Discretize: ΔA, ΔB
        deltaA = torch.exp(delta.unsqueeze(-1) * A)         # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)       # (B, L, ED, N)
        BX = deltaB * x.unsqueeze(-1)                       # (B, L, ED, N)

        # Parallel scan: H[t] = ΔA[t] * H[t-1] + ΔB[t] * x[t]
        hs = pscan(deltaA, BX)                              # (B, L, ED, N)

        # Readout: y[t] = C[t]^T * H[t] + D * x[t]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)               # (B, L, ED)
        y = y + D * x

        return y
