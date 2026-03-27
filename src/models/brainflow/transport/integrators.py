"""ODE integrators for CSFM flow matching — adapted from CSFM."""

import torch as th
from torchdiffeq import odeint


class ode:
    """ODE solver with fixed-step (Euler/Heun) or adaptive (dopri5) methods.

    The time grid accounts for `time_dist_shift` used during training,
    so the sampling schedule matches the training distribution.
    """

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # Build shifted time grid matching training distribution
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        """Solve ODE from x0 (source) to x1 (target).

        Args:
            x: (B, ...) initial state (source distribution)
            model: velocity model fn(x, t, **kwargs) → velocity
            **model_kwargs: extra kwargs for model

        Returns:
            samples: trajectory tensor (T, B, ...)
        """
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t_batch = (
                th.ones(x[0].size(0)).to(device) * t
                if isinstance(x, tuple)
                else th.ones(x.size(0)).to(device) * t
            )
            return self.drift(x, t_batch, model, **model_kwargs)

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]

        samples = odeint(
            _fn, x, t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol,
        )
        return samples
