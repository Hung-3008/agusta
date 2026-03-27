"""CSFM Transport — core flow matching logic adapted from CSFM.

Handles:
  - Timestep sampling (uniform, with time_dist_shift)
  - ODE drift computation for velocity models
  - Sampler creation for inference
"""

import enum
import math

import torch as th

from . import path
from .integrators import ode


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    """Flow matching transport with configurable path and timestep distribution."""

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.time_dist_type = time_dist_type
        self.time_dist_shift = max(time_dist_shift, 1.0)
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(self, train_eps, sample_eps, *, sde=False, eval=False, reverse=False, last_step_size=0.0):
        t0 = 0
        t1 = 1 - 1 / 1000
        eps = train_eps if not eval else sample_eps

        if isinstance(self.path_sampler, path.ICPlan) and (
            self.model_type != ModelType.VELOCITY or sde
        ):
            t0 = eps if sde else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample_timestep(self, x1):
        """Sample timesteps for training with time_dist_shift.

        Args:
            x1: (B, ...) target tensor (used for shape/device)

        Returns:
            t: (B,) shifted timesteps
        """
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if self.time_dist_type == "uniform":
            t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        else:
            raise NotImplementedError(f"Unknown time distribution: {self.time_dist_type}")

        t = t.to(x1)

        # Apply time distribution shift (CSFM key feature)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t

    def get_drift(self):
        """Get ODE drift function for velocity model."""
        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        def body_fn(x, t, model, **model_kwargs):
            model_output = velocity_ode(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, (
                f"Output shape {model_output.shape} != input shape {x.shape}"
            )
            return model_output

        return body_fn


class Sampler:
    """ODE sampler for inference."""

    def __init__(self, transport):
        self.transport = transport
        self.drift = transport.get_drift()

    def sample_ode(
        self,
        *,
        sampling_method="euler",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """Create ODE sampling function.

        Args:
            sampling_method: "euler", "heun", "midpoint", or "dopri5"
            num_steps: integration steps
            atol/rtol: tolerances for adaptive solvers
            reverse: solve data → noise if True

        Returns:
            sample_fn(x_init, model, **kwargs) → trajectory
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(
                x, th.ones_like(t) * (1 - t), model, **kwargs
            )
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
        )

        return _ode.sample
