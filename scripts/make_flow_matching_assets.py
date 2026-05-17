"""Generate flow-matching illustration "raw material" distribution images.

Produces transparent-background PNGs of:
  - source distribution p  (isotropic Gaussian, concentric rings)
  - target distribution q  (banana / peanut-shaped bimodal)
  - intermediate p_t snapshots (thin gray contours)

Palette: blue-dominant (Matplotlib `Blues`).

Output: Latex/BMVCTemplate2026-master/images/flow_matching_assets/
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = Path(__file__).resolve().parents[1] / (
    "Latex/BMVCTemplate2026-master/images/flow_matching_assets"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- shared styling ----------
BLUE_FILLED = LinearSegmentedColormap.from_list(
    "blue_filled",
    [
        (0.0, "#ffffff00"),   # transparent at outer edge
        (0.15, "#cfe1f5"),
        (0.45, "#7cb1de"),
        (0.75, "#2f6db0"),
        (1.0, "#0b2f5a"),
    ],
)

GRAY_LINE_COLOR = "#7a8693"
LINE_BLUE = "#1f4e89"

FIG_KW = dict(figsize=(4, 4), dpi=200)


def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  -> {path.relative_to(OUT_DIR.parent.parent)}")


def _ax(fig):
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ---------- distributions ----------
def gaussian_pdf(xx, yy, mu, cov):
    """2-D Gaussian pdf on a meshgrid."""
    inv = np.linalg.inv(cov)
    d = np.stack([xx - mu[0], yy - mu[1]], axis=-1)
    quad = np.einsum("...i,ij,...j", d, inv, d)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))
    return norm * np.exp(-0.5 * quad)


def banana_pdf(xx, yy, mu=(0.0, 0.0), scale=1.0, bend=0.55):
    """Banana-shaped density: warp a Gaussian along y = bend * x^2."""
    x = (xx - mu[0]) / scale
    y = (yy - mu[1]) / scale
    y_warp = y - bend * (x**2 - 1.0)
    z = np.exp(-0.5 * (x**2 / 1.6 + y_warp**2 / 0.35))
    return z


def mixture_pdf(xx, yy):
    """Two-mode 'peanut' target like the reference image."""
    g1 = gaussian_pdf(xx, yy, mu=(0.55, 1.05), cov=[[0.14, 0.07], [0.07, 0.32]])
    g2 = gaussian_pdf(xx, yy, mu=(0.20, -0.35), cov=[[0.28, 0.10], [0.10, 0.20]])
    return 0.55 * g1 + 0.45 * g2


# ---------- panels ----------
GRID = np.linspace(-2.6, 2.6, 400)
XX, YY = np.meshgrid(GRID, GRID)


def make_source_p():
    """Concentric-ring Gaussian source `p`."""
    z = gaussian_pdf(XX, YY, mu=(0.0, 0.0), cov=[[0.35, 0.0], [0.0, 0.35]])
    fig = plt.figure(**FIG_KW)
    ax = _ax(fig)
    levels = np.linspace(z.max() * 0.05, z.max(), 7)
    ax.contourf(XX, YY, z, levels=levels, cmap=BLUE_FILLED)
    ax.contour(XX, YY, z, levels=levels, colors=LINE_BLUE, linewidths=0.6, alpha=0.6)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    _save(fig, "p_source.png")


def make_target_q():
    """Two-mode peanut target `q`."""
    z = mixture_pdf(XX, YY)
    fig = plt.figure(**FIG_KW)
    ax = _ax(fig)
    levels = np.linspace(z.max() * 0.06, z.max(), 7)
    ax.contourf(XX, YY, z, levels=levels, cmap=BLUE_FILLED)
    ax.contour(XX, YY, z, levels=levels, colors=LINE_BLUE, linewidths=0.6, alpha=0.6)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    _save(fig, "q_target.png")


def make_path_pt(n_steps: int = 5):
    """Intermediate distributions p_t as thin contours along the morph."""
    z_p = gaussian_pdf(XX, YY, mu=(0.0, 0.0), cov=[[0.35, 0.0], [0.0, 0.35]])
    z_q = mixture_pdf(XX, YY)

    ts = np.linspace(0.15, 0.85, n_steps)
    for i, t in enumerate(ts):
        # geodesic-ish linear interpolation of densities (visual mix)
        z = (1 - t) * z_p + t * z_q
        fig = plt.figure(**FIG_KW)
        ax = _ax(fig)
        levels = np.linspace(z.max() * 0.08, z.max(), 6)
        ax.contour(XX, YY, z, levels=levels, colors=GRAY_LINE_COLOR, linewidths=0.7)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        _save(fig, f"pt_step_{i:02d}.png")


def make_path_pt_filled(n_steps: int = 5):
    """Same intermediate distributions p_t, but filled with the blue palette."""
    z_p = gaussian_pdf(XX, YY, mu=(0.0, 0.0), cov=[[0.35, 0.0], [0.0, 0.35]])
    z_q = mixture_pdf(XX, YY)

    ts = np.linspace(0.15, 0.85, n_steps)
    for i, t in enumerate(ts):
        z = (1 - t) * z_p + t * z_q
        fig = plt.figure(**FIG_KW)
        ax = _ax(fig)
        levels = np.linspace(z.max() * 0.06, z.max(), 7)
        ax.contourf(XX, YY, z, levels=levels, cmap=BLUE_FILLED)
        ax.contour(XX, YY, z, levels=levels, colors=LINE_BLUE, linewidths=0.6, alpha=0.6)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        _save(fig, f"pt_step_{i:02d}_filled.png")


def make_combined_overview():
    """One PNG combining p (filled blue) | thin p_t contours | q (filled blue)."""
    z_p = gaussian_pdf(XX, YY, mu=(0.0, 0.0), cov=[[0.35, 0.0], [0.0, 0.35]])
    z_q = mixture_pdf(XX, YY)

    fig = plt.figure(figsize=(12, 4), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    # Place p on the left, q on the right
    shift_p, shift_q = -4.0, 4.0
    XXp, YYp = XX + shift_p, YY
    XXq, YYq = XX + shift_q, YY

    levels_p = np.linspace(z_p.max() * 0.05, z_p.max(), 7)
    levels_q = np.linspace(z_q.max() * 0.06, z_q.max(), 7)
    ax.contourf(XXp, YYp, z_p, levels=levels_p, cmap=BLUE_FILLED)
    ax.contour(XXp, YYp, z_p, levels=levels_p, colors=LINE_BLUE, linewidths=0.6, alpha=0.7)
    ax.contourf(XXq, YYq, z_q, levels=levels_q, cmap=BLUE_FILLED)
    ax.contour(XXq, YYq, z_q, levels=levels_q, colors=LINE_BLUE, linewidths=0.6, alpha=0.7)

    # Intermediate p_t contours between the two
    for t in np.linspace(0.15, 0.85, 5):
        z = (1 - t) * z_p + t * z_q
        shift = shift_p + t * (shift_q - shift_p)
        levels = np.linspace(z.max() * 0.08, z.max(), 5)
        ax.contour(XX + shift, YY, z, levels=levels, colors=GRAY_LINE_COLOR, linewidths=0.55)

    ax.set_xlim(shift_p - 2.5, shift_q + 2.5)
    ax.set_ylim(-2.6, 2.6)
    _save(fig, "overview_p_to_q.png")


if __name__ == "__main__":
    print(f"Writing assets to: {OUT_DIR}")
    make_source_p()
    make_target_q()
    make_path_pt(n_steps=5)
    make_path_pt_filled(n_steps=5)
    make_combined_overview()
    print("Done.")
