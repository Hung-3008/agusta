"""Flow matching path plans — adapted from CSFM.

Defines the interpolation between source x0 and target x1:
  xt = alpha_t * x1 + sigma_t * x0
  ut = d_alpha_t * x1 + d_sigma_t * x0
"""

import torch as th
import numpy as np


def expand_t_like_x(t, x):
    """Reshape time t to broadcastable dimension of x.

    Args:
        t: (B,) time vector
        x: (B, ...) data point
    """
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


class ICPlan:
    """Linear Interpolation Coupling Plan (Optimal Transport path).

    Path: xt = (1 - t) * x1 + t * x0
    Velocity: ut = -x1 + x0 = x0 - x1
    """

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Coefficient of x1 along the path."""
        return 1 - t, -1

    def compute_sigma_t(self, t):
        """Coefficient of x0 along the path."""
        return t, 1

    def compute_d_alpha_alpha_ratio_t(self, t):
        return -1 / (1 - t)

    def compute_drift(self, x, t):
        """SDE drift (for score parameterization)."""
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Transform velocity prediction to score."""
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - x) / var
        return score

    def compute_mu_t(self, t, x0, x1):
        """Mean of time-dependent density p_t."""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from p_t (deterministic for linear path)."""
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t, x0, x1, xt):
        """Compute velocity field ut corresponding to p_t."""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        """Compute (t, xt, ut) for training.

        Args:
            t: (B,) timesteps
            x0: (B, ...) source (learned or noise)
            x1: (B, ...) target (data)

        Returns:
            t, xt, ut
        """
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class GVPCPlan(ICPlan):
    """Geodesic VP path (cosine schedule)."""

    def __init__(self, sigma=0.0):
        super().__init__(sigma)

    def compute_alpha_t(self, t):
        alpha_t = th.cos(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t = th.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return -np.pi / 2 * th.tan(t * np.pi / 2)
