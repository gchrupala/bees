"""Schematic of the comb-tilt geometry used by the two dance codes.

Renders a single tilted comb, drawn as a honeycomb tile, together with its unit
normal, the gravity direction, an example world food direction, and the two
in-plane reference directions obtained by projecting onto the comb (the
direct-code reference, length ``s_dir``, and the gravity-code reference, length
``s_grav``). The geometry is taken directly from the model so the figure stays
faithful to ``bees.model``.
"""

from __future__ import annotations

import argparse
import sys
from math import cos, hypot, pi, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
        "font.size": 12,
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
COMB_TILT = 0.5  # gamma: a half-tilted comb (theta = pi/4)
COMB_ORIENTATION = pi / 6.0  # phi: compass heading the comb faces
FOOD_AZIMUTH = 1.0  # d: example food direction (radians), near the facing heading
COMB_RADIUS = 0.78
HEX_RADIUS = 0.13
SURFACE_LIFT = 0.04  # lift in-plane arrows just off the comb so they stay visible

DIRECT_COLOR = "#1f77b4"
GRAVITY_COLOR = "#d62728"
NORMAL_COLOR = "#2c2c2c"
GRAVITY_REF_COLOR = "#7f7f7f"
COMB_FACE = (0.953, 0.851, 0.627, 0.5)  # translucent wax so in-plane arrows show through
COMB_EDGE = "#c79a3f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def hex_centers(radius: float, hex_radius: float) -> list[tuple[float, float]]:
    """Pointy-top hexagon centres tiling a disc of the given radius."""
    dx = sqrt(3.0) * hex_radius
    dy = 1.5 * hex_radius
    rows = int(np.ceil(radius / dy)) + 1
    cols = int(np.ceil(radius / dx)) + 1
    centers = []
    for j in range(-rows, rows + 1):
        y = j * dy
        x_offset = dx / 2.0 if j % 2 else 0.0
        for i in range(-cols, cols + 1):
            x = i * dx + x_offset
            if hypot(x, y) <= radius:
                centers.append((x, y))
    return centers


def hex_vertices(cx: float, cy: float, hex_radius: float) -> list[tuple[float, float]]:
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return [(cx + hex_radius * cos(a), cy + hex_radius * sin(a)) for a in angles]


def draw_comb(ax, basis) -> None:
    first = np.asarray(basis.first_axis)
    second = np.asarray(basis.second_axis)
    polygons = [
        [u * first + v * second for u, v in hex_vertices(cx, cy, HEX_RADIUS)]
        for cx, cy in hex_centers(COMB_RADIUS, HEX_RADIUS)
    ]
    comb = Poly3DCollection(
        polygons, facecolor=COMB_FACE, edgecolor=COMB_EDGE, linewidth=0.7
    )
    comb.set_zorder(0)
    ax.add_collection3d(comb)


def draw_arrow(
    ax, vector, color, label, *, base=(0, 0, 0), dashed=False, width=2.6, label_offset=(0, 0, 0)
):
    base = np.asarray(base, dtype=float)
    vector = np.asarray(vector, dtype=float)
    tip = base + vector
    ax.quiver(
        base[0],
        base[1],
        base[2],
        vector[0],
        vector[1],
        vector[2],
        color=color,
        linewidth=width,
        arrow_length_ratio=0.16,
        linestyle="--" if dashed else "-",
        zorder=5,
    )
    if label:
        pos = base + vector * 1.1 + np.asarray(label_offset)
        ax.text(pos[0], pos[1], pos[2], label, color=color, fontsize=13, zorder=6)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    basis = _comb_basis(COMB_TILT, COMB_ORIENTATION)
    normal = np.asarray(_comb_normal(COMB_TILT, COMB_ORIENTATION))
    lift = SURFACE_LIFT * normal

    fig = plt.figure(figsize=(7.5, 6.2))
    # Disable depth-based z-ordering so our explicit zorders are honoured and the
    # line segments paint on top of (mask) the translucent comb.
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    draw_comb(ax, basis)

    # Horizontal reference ring drawn so the comb sits inside it: the near arc
    # (on the camera side of the comb plane) masks the comb, while the far arc
    # passes behind it. A ring point p (at z = 0) is on the camera side when
    # p . n > 0, i.e. cos(angle - phi) > 0, since the camera looks from the +n
    # side. Dots (not radial lines) mark east and north.
    ring_angle = np.linspace(0.0, 2 * pi, 400)
    ring_x = 0.95 * np.cos(ring_angle)
    ring_y = 0.95 * np.sin(ring_angle)
    ring_z = np.zeros_like(ring_angle)
    near_side = np.cos(ring_angle - COMB_ORIENTATION) >= 0.0
    near = np.where(near_side, 1.0, np.nan)
    far = np.where(near_side, np.nan, 1.0)
    ax.plot(ring_x * far, ring_y * far, ring_z, color="#bbbbbb", linewidth=0.9, zorder=-1)
    ax.plot(ring_x * near, ring_y * near, ring_z, color="#aaaaaa", linewidth=0.9, zorder=3)
    for axis_dir, name in (((1.0, 0.0, 0.0), "east"), ((0.0, 1.0, 0.0), "north")):
        end = 0.95 * np.asarray(axis_dir)
        ax.scatter([end[0]], [end[1]], [0.0], color="#888888", s=20, zorder=5)
        ax.text(end[0] * 1.08, end[1] * 1.08, 0.0, name, color="#888888", fontsize=10)

    # Comb normal and gravity (the vertical world direction).
    draw_arrow(ax, normal, NORMAL_COLOR, r"$\mathbf{n}$", label_offset=(0.05, 0.0, 0.04))
    draw_arrow(ax, (0, 0, 1), GRAVITY_REF_COLOR, r"$\mathbf{g}$", dashed=True, width=1.8)

    # World food direction (in the ground plane) and its in-plane projection,
    # the direct-code reference (lifted just off the comb so it stays visible).
    food = 0.75 * np.asarray(_world_direction_vector(FOOD_AZIMUTH))
    draw_arrow(ax, food, DIRECT_COLOR, r"$\mathbf{f}_d$", dashed=True, width=1.8,
               label_offset=(0.05, -0.05, -0.04))
    direct_ref = np.asarray(
        _project_onto_plane(_world_direction_vector(FOOD_AZIMUTH), tuple(normal))
    )
    draw_arrow(ax, direct_ref, DIRECT_COLOR, r"$s_{\mathrm{dir}}$", base=lift,
               label_offset=(0.05, 0.0, 0.1))

    # Gravity reference: projection of the vertical onto the comb surface.
    gravity_ref = np.asarray(_project_onto_plane((0.0, 0.0, 1.0), tuple(normal)))
    draw_arrow(ax, gravity_ref, GRAVITY_COLOR, r"$s_{\mathrm{grav}}$", base=lift,
               label_offset=(-0.08, 0.0, 0.08))

    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(-0.45, 1.15)
    ax.set_box_aspect((1, 1, 0.85))
    ax.view_init(elev=20, azim=-55)
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(args.output, dpi=190)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
