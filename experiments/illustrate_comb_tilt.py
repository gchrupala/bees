"""Schematic of the comb-tilt geometry used by the two dance codes.

Renders three panels at increasing tilt (horizontal, half, vertical), drawing
the comb surface, its unit normal, the gravity direction, and the two in-plane
reference directions (the projected food direction for the direct code and the
projected gravity direction for the gravity code). The geometry is taken
directly from the model so the figure stays faithful to ``bees.model``.
"""

from __future__ import annotations

import argparse
import sys
from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
    }
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import (  # noqa: E402
    _comb_basis,
    _comb_normal,
    _project_onto_plane,
    _world_direction_vector,
)

DEFAULT_OUTPUT = ROOT / "report" / "figures" / "comb_tilt_geometry.png"

# Fixed scene parameters chosen only for legibility of the schematic.
COMB_ORIENTATION = pi / 6.0  # phi: compass heading the comb faces
FOOD_AZIMUTH = 1.9  # d: example food direction (radians)
PLANE_HALF_WIDTH = 1.1

DIRECT_COLOR = "#1f77b4"
GRAVITY_COLOR = "#d62728"
NORMAL_COLOR = "#2c2c2c"
GRAVITY_REF_COLOR = "#7f7f7f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def comb_plane_corners(basis) -> np.ndarray:
    first = np.asarray(basis.first_axis)
    second = np.asarray(basis.second_axis)
    signs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    return np.array(
        [PLANE_HALF_WIDTH * (a * first + b * second) for a, b in signs]
    )


def draw_arrow(ax, vector, color, label, *, dashed=False, width=2.5, label_offset=(0, 0, 0)):
    vector = np.asarray(vector)
    ax.quiver(
        0,
        0,
        0,
        vector[0],
        vector[1],
        vector[2],
        color=color,
        linewidth=width,
        arrow_length_ratio=0.15,
        linestyle="--" if dashed else "-",
    )
    if label:
        tip = vector * 1.08 + np.asarray(label_offset)
        ax.text(tip[0], tip[1], tip[2], label, color=color, fontsize=12)


def draw_panel(ax, gamma: float, title: str) -> None:
    theta = gamma * pi / 2.0
    basis = _comb_basis(gamma, COMB_ORIENTATION)
    normal = np.asarray(_comb_normal(gamma, COMB_ORIENTATION))

    # Comb surface.
    corners = comb_plane_corners(basis)
    surface = Poly3DCollection(
        [corners], alpha=0.18, facecolor="#cccccc", edgecolor="#888888"
    )
    ax.add_collection3d(surface)

    # Faint horizontal ground plane for reference.
    ground = PLANE_HALF_WIDTH * np.array(
        [[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]], dtype=float
    )
    ax.add_collection3d(
        Poly3DCollection([ground], alpha=0.05, facecolor="#000000", edgecolor="none")
    )

    # Comb normal and gravity. Nudge the normal label aside so it stays legible
    # on the horizontal comb, where the normal coincides with gravity.
    draw_arrow(ax, normal, NORMAL_COLOR, r"$\mathbf{n}$", label_offset=(0.12, 0.0, 0.06))
    draw_arrow(ax, (0, 0, 1), GRAVITY_REF_COLOR, r"$\mathbf{g}$", dashed=True, width=1.8)

    # World food direction (lies in the ground plane), drawn short, and its
    # in-plane projection (the direct-code reference, with length s_dir).
    food = 0.65 * np.asarray(_world_direction_vector(FOOD_AZIMUTH))
    draw_arrow(ax, food, DIRECT_COLOR, r"$\mathbf{f}_d$", dashed=True, width=1.6)
    direct_ref = np.asarray(_project_onto_plane(_world_direction_vector(FOOD_AZIMUTH), tuple(normal)))
    draw_arrow(
        ax, direct_ref, DIRECT_COLOR, r"$s_{\mathrm{dir}}$", label_offset=(0.0, 0.18, 0.0)
    )

    # Gravity reference: projection of the vertical onto the comb surface
    # (the gravity-code reference, with length s_grav).
    gravity_ref = np.asarray(_project_onto_plane((0.0, 0.0, 1.0), tuple(normal)))
    draw_arrow(
        ax, gravity_ref, GRAVITY_COLOR, r"$s_{\mathrm{grav}}$", label_offset=(-0.1, 0.0, 0.08)
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0, 1.2)
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=22, azim=-60)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("east", fontsize=9, labelpad=-12)
    ax.set_ylabel("north", fontsize=9, labelpad=-12)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (0.0, r"Horizontal ($\theta=0$)"),
        (0.5, r"Tilted ($\theta=\pi/4$)"),
        (1.0, r"Vertical ($\theta=\pi/2$)"),
    ]
    fig = plt.figure(figsize=(13, 4.6))
    for index, (gamma, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        draw_panel(ax, gamma, title)

    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
