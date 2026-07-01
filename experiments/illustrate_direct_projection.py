"""Minimal sketch of the direct code's projection step.

Two plain squares sharing the hive's east-north position: a flat ground
square (the world-horizontal plane) and a tilted comb square, drawn directly
above it along the vertical axis. A food-direction vector lies in the ground
square; its orthogonal projection onto the comb -- computed by carrying both
the vector's tail and tip along the comb's own normal until they land on the
comb's plane -- is drawn on the comb square, with dashed rays (parallel to
the normal, by construction) connecting the two. The part of the food vector
that the comb occludes from this viewpoint is drawn underneath it, so it
shows through pale rather than being fully hidden or fully overdrawn.

Geometry is taken directly from ``bees.model`` so the figure stays faithful
to the simulation.
"""

from __future__ import annotations

import argparse
import sys
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

from bees.model import _comb_basis, _world_direction_vector  # noqa: E402

DEFAULT_OUTPUT = ROOT / "report" / "figures" / "direct_projection_sketch.png"

TILT = 0.3  # gamma: theta = 27 degrees -- shallow, so the tilt reads clearly
ORIENTATION = 0.0  # phi: comb tilts due east
FOOD_AZIMUTH = 0.7  # d: an arbitrary food direction, in radians
FOOD_LENGTH = 0.6  # shorter than SQUARE_SIZE so its projection stays in-bounds
SQUARE_SIZE = 1.5

GROUND_OFFSET = np.array([0.0, 0.0, 0.0])
COMB_OFFSET = np.array([0.0, 0.0, 1.3])
ELEV, AZIM = 22.0, -60.0

GROUND_COLOR = "#dce6f0"
GROUND_EDGE = "#8fa3b8"
COMB_COLOR = "#f0dca0"
COMB_EDGE = "#b8963f"
FOOD_COLOR = "#1f77b4"
PROJECTED_COLOR = "#d62728"
NORMAL_COLOR = "#2c2c2c"
RAY_COLOR = "#999999"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def square_corners(u_axis, v_axis, size):
    return np.array(
        [
            u_axis * size + v_axis * size,
            -u_axis * size + v_axis * size,
            -u_axis * size - v_axis * size,
            u_axis * size - v_axis * size,
        ]
    )


def camera_direction(elev_deg, azim_deg):
    """Unit vector from the scene toward the camera, for the given view_init
    angles (matches matplotlib's own convention for an orthographic view).
    """
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    return np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )


def occluded_by_comb(point, cam_dir, plane_point, plane_normal, u_axis, v_axis, half_size):
    """True if the comb square sits between `point` and the camera, i.e. the
    ray from `point` toward the camera crosses the (finite) comb square.
    """
    denom = np.dot(cam_dir, plane_normal)
    if abs(denom) < 1e-9:
        return False
    s = np.dot(plane_point - point, plane_normal) / denom
    if s <= 0:
        return False
    hit = point + s * cam_dir
    rel = hit - plane_point
    u, v = np.dot(rel, u_axis), np.dot(rel, v_axis)
    return abs(u) <= half_size and abs(v) <= half_size


def shadow_on_plane(point, plane_point, plane_normal):
    """Where the ray through `point`, travelling along `plane_normal`, hits
    the plane through `plane_point` with that normal. By construction, the
    segment from `point` to this result is exactly parallel to plane_normal.
    """
    t = np.dot(plane_point - point, plane_normal)
    return point + t * plane_normal


def draw_food_vector(ax, food_base, food_vec, normal, first, second):
    """Draw the food vector split at the comb's occlusion boundary: the
    hidden run underneath the comb (so it shows through pale), the visible
    run (with the arrowhead) on top of it.
    """
    cam_dir = camera_direction(ELEV, AZIM)

    n_samples = 400
    sample_t = np.linspace(0.0, 1.0, n_samples)
    sample_points = food_base + np.outer(sample_t, food_vec)
    occluded = np.array([
        occluded_by_comb(p, cam_dir, COMB_OFFSET, normal, first, second, SQUARE_SIZE)
        for p in sample_points
    ])

    arrowhead_length = 0.12 * np.linalg.norm(food_vec)
    run_start = 0
    for i in range(1, n_samples + 1):
        if i == n_samples or occluded[i] != occluded[run_start]:
            t_start, t_end = sample_t[run_start], sample_t[i - 1]
            p_start = food_base + t_start * food_vec
            p_end = food_base + t_end * food_vec
            is_last_run = i == n_samples
            if occluded[run_start]:
                ax.plot(*zip(p_start, p_end), color=FOOD_COLOR, linewidth=2.5, zorder=0.5)
            elif is_last_run:
                # This run reaches the tip, so it gets the arrowhead; keep the
                # head size fixed regardless of how long this visible run is.
                seg_vec = p_end - p_start
                seg_len = np.linalg.norm(seg_vec)
                ratio = min(1.0, arrowhead_length / seg_len) if seg_len > 0 else 1.0
                ax.quiver(
                    *p_start, *seg_vec, color=FOOD_COLOR, linewidth=2.5,
                    arrow_length_ratio=ratio, zorder=3,
                )
            else:
                ax.plot(*zip(p_start, p_end), color=FOOD_COLOR, linewidth=2.5, zorder=3)
            run_start = i


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    basis = _comb_basis(TILT, ORIENTATION)
    normal = np.asarray(basis.normal)
    first = np.asarray(basis.first_axis)
    second = np.asarray(basis.second_axis)

    ground_corners = square_corners(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), SQUARE_SIZE
    ) + GROUND_OFFSET
    # Plain symmetric square, centered on its own offset -- no hinging.
    comb_corners = square_corners(first, second, SQUARE_SIZE) + COMB_OFFSET

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)

    ground_poly = Poly3DCollection(
        [ground_corners], facecolor=GROUND_COLOR, edgecolor=GROUND_EDGE,
        linewidth=1.0, alpha=0.6,
    )
    ground_poly.set_zorder(0)
    ax.add_collection3d(ground_poly)

    comb_poly = Poly3DCollection(
        [comb_corners], facecolor=COMB_COLOR, edgecolor=COMB_EDGE,
        linewidth=1.0, alpha=0.8,
    )
    comb_poly.set_zorder(1)
    ax.add_collection3d(comb_poly)

    food_vec = FOOD_LENGTH * np.asarray(_world_direction_vector(FOOD_AZIMUTH))
    food_base = GROUND_OFFSET
    draw_food_vector(ax, food_base, food_vec, normal, first, second)

    # Both the vector's tail and tip are carried onto the comb plane by the
    # same light direction (the comb's own normal) -- not the arbitrary
    # vertical shift used to place the comb square for a clear picture. This
    # keeps both rays exactly parallel to the normal, and reproduces the
    # plain orthogonal projection at the tail (which starts on the plane
    # through the origin).
    comb_base = shadow_on_plane(food_base, COMB_OFFSET, normal)
    comb_tip = shadow_on_plane(food_base + food_vec, COMB_OFFSET, normal)
    projected_vec = comb_tip - comb_base
    ax.quiver(
        *comb_base, *projected_vec, color=PROJECTED_COLOR, linewidth=2.5,
        arrow_length_ratio=0.12, zorder=4,
    )

    # Projection rays: dashed segments connecting the two vectors' start
    # points and end points -- both parallel to the comb's normal by
    # construction.
    ax.plot(*zip(food_base, comb_base), color=RAY_COLOR, linewidth=1.2,
            linestyle="--", zorder=2)
    ax.plot(*zip(food_base + food_vec, comb_tip), color=RAY_COLOR, linewidth=1.2,
            linestyle="--", zorder=2)

    # The comb's own normal, drawn from the plane's center, on top of
    # everything so it isn't masked by the (semi-transparent) comb square.
    normal_vec = 0.8 * normal
    ax.quiver(
        *COMB_OFFSET, *normal_vec, color=NORMAL_COLOR, linewidth=2.0,
        arrow_length_ratio=0.15, zorder=5,
    )

    xlim = (-2.0, 2.0)
    ylim = (-2.0, 2.0)
    zlim = (0.0, 2.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # Match the box aspect to the actual data spans (not a plain cube), so a
    # unit is the same physical length on every axis and the tilt renders
    # undistorted.
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.set_proj_type("ortho")  # perspective projection doesn't preserve parallel lines
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("x (east)")
    ax.set_ylabel("y (north)")
    ax.set_zlabel("z (up)")

    fig.tight_layout()
    fig.savefig(args.output, dpi=190)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
