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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont

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
FOOD_LENGTH = 0.9  # long, but its projection (u~1.2, v~0.6) still clears SQUARE_SIZE
SQUARE_SIZE = 1.275  # ~15% smaller than the original 1.5

GROUND_OFFSET = np.array([0.0, 0.0, 0.0])
COMB_OFFSET = np.array([0.0, 0.0, 1.3])
ELEV, AZIM = 22.0, -60.0
HEX_RADIUS = 0.16
# hex_centers_square keeps any hex whose *center* is within half_size +
# hex_radius, and each hex's own vertices then extend up to hex_radius
# beyond its center -- so the tiling's true visual edge reaches roughly
# SQUARE_SIZE + 2 * HEX_RADIUS, not SQUARE_SIZE. The occlusion test must use
# that same true extent, or points just past SQUARE_SIZE get judged
# "unoccluded" and drawn solid on top of comb tiles that are actually there.
COMB_OCCLUSION_HALF_SIZE = SQUARE_SIZE + 2 * HEX_RADIUS

# Food-site marker: a purple flower, rendered as a NotoColorEmoji bitmap since
# matplotlib text cannot draw colour emoji.
FLOWER_MARKER = "\U0001FABB"  # hyacinth
FLOWER_MARKER_SIZE = 30
FLOWER_EDGE_MARGIN = 0.92  # how close to the ground square's edge to place it
FONT_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMOJI_STRIKE_SIZE = 109  # NotoColorEmoji ships a single bitmap strike at this size

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


def hex_centers_square(half_size, hex_radius):
    """Pointy-top hex centres tiling the square [-half_size, half_size]^2."""
    dx = np.sqrt(3.0) * hex_radius
    dy = 1.5 * hex_radius
    rows = int(np.ceil(half_size / dy)) + 1
    cols = int(np.ceil(half_size / dx)) + 1
    centers = []
    for j in range(-rows, rows + 1):
        y = j * dy
        if y < -half_size - hex_radius or y > half_size + hex_radius:
            continue
        x_offset = dx / 2.0 if j % 2 else 0.0
        for i in range(-cols, cols + 1):
            x = i * dx + x_offset
            if -half_size - hex_radius <= x <= half_size + hex_radius:
                centers.append((x, y))
    return centers


def hex_vertices(cx, cy, hex_radius):
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return [(cx + hex_radius * np.cos(a), cy + hex_radius * np.sin(a)) for a in angles]


def edge_point(base, unit_dir, half_size, margin):
    """The point base + s * unit_dir, with s the largest value that keeps the
    point within [-half_size, half_size] in x and y, scaled by `margin` (< 1)
    so it lands just inside the boundary rather than exactly on it.
    """
    bounds = []
    for axis in (0, 1):
        d = unit_dir[axis]
        if abs(d) < 1e-9:
            continue
        for bound in (half_size, -half_size):
            s = (bound - base[axis]) / d
            if s > 0:
                bounds.append(s)
    return base + margin * min(bounds) * unit_dir


def render_symbol_image(text, size):
    """Rasterise a colour-emoji glyph to an RGBA array of the requested size."""
    font = ImageFont.truetype(FONT_PATH, EMOJI_STRIKE_SIZE)
    canvas = EMOJI_STRIKE_SIZE * 2
    image = Image.new("RGBA", (canvas, canvas), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), text, font=font, embedded_color=True)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (canvas - width) / 2 - bbox[0]
    y = (canvas - height) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255), embedded_color=True)

    image = image.crop(image.getbbox())
    image = image.resize((size, size), Image.LANCZOS)
    return np.asarray(image)


def place_flower(ax, point3d):
    """Place the flower marker at a 3D point, projected to the current 2D
    view (matplotlib text can't draw colour emoji directly in 3D).
    """
    x2, y2, _ = proj3d.proj_transform(*point3d, ax.get_proj())
    image = render_symbol_image(FLOWER_MARKER, FLOWER_MARKER_SIZE)
    annotation = AnnotationBbox(
        OffsetImage(image, zoom=1.0, interpolation="nearest"),
        (x2, y2), xycoords=ax.transData, frameon=False, pad=0,
        annotation_clip=False, zorder=6,
    )
    ax.add_artist(annotation)


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


def ground_point_shadowing_to(target, plane_normal):
    """The point on the ground (z = 0) whose shadow along `plane_normal`
    lands exactly on `target` -- i.e. target, the point, and plane_normal
    are colinear.
    """
    k = target[2] / plane_normal[2]
    return target - k * plane_normal


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
        occluded_by_comb(p, cam_dir, COMB_OFFSET, normal, first, second, COMB_OCCLUSION_HALF_SIZE)
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
            zorder = 0.5 if occluded[run_start] else 3
            if is_last_run:
                # This run reaches the tip, so it gets the arrowhead -- even
                # if it's the occluded run, so the arrow always has one, just
                # pale when hidden under the comb. Keep the head size fixed
                # regardless of how long this run is.
                seg_vec = p_end - p_start
                seg_len = np.linalg.norm(seg_vec)
                ratio = min(1.0, arrowhead_length / seg_len) if seg_len > 0 else 1.0
                ax.quiver(
                    *p_start, *seg_vec, color=FOOD_COLOR, linewidth=2.5,
                    arrow_length_ratio=ratio, zorder=zorder,
                )
            else:
                ax.plot(*zip(p_start, p_end), color=FOOD_COLOR, linewidth=2.5, zorder=zorder)
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

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)

    ground_poly = Poly3DCollection(
        [ground_corners], facecolor=GROUND_COLOR, edgecolor=GROUND_EDGE,
        linewidth=1.0, alpha=0.6,
    )
    ground_poly.set_zorder(0)
    ax.add_collection3d(ground_poly)

    # Comb: a hexagonal-cell tile, filling the same square footprint (still
    # centered on its own offset, no hinging) that the occlusion test below
    # uses as its boundary.
    comb_hexes = [
        [u * first + v * second + COMB_OFFSET for u, v in hex_vertices(cx, cy, HEX_RADIUS)]
        for cx, cy in hex_centers_square(SQUARE_SIZE, HEX_RADIUS)
    ]
    comb_poly = Poly3DCollection(
        comb_hexes, facecolor=COMB_COLOR, edgecolor=COMB_EDGE,
        linewidth=0.7, alpha=0.8,
    )
    comb_poly.set_zorder(1)
    ax.add_collection3d(comb_poly)

    food_vec = FOOD_LENGTH * np.asarray(_world_direction_vector(FOOD_AZIMUTH))
    # Start the food vector at the ground point that projects to the comb's
    # own center, rather than at the origin.
    food_base = ground_point_shadowing_to(COMB_OFFSET, normal)
    draw_food_vector(ax, food_base, food_vec, normal, first, second)

    # Both the vector's tail and tip are carried onto the comb plane by the
    # same light direction (the comb's own normal) -- not the arbitrary
    # vertical shift used to place the comb square for a clear picture. This
    # keeps both rays exactly parallel to the normal; the tail was chosen
    # (via ground_point_shadowing_to) to land exactly on the comb's center.
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

    xlim = (-1.6, 1.6)
    ylim = (-1.6, 1.6)
    zlim = (-0.05, 2.05)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # Match the box aspect to the actual data spans (not a plain cube), so a
    # unit is the same physical length on every axis and the tilt renders
    # undistorted.
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.set_proj_type("ortho")  # perspective projection doesn't preserve parallel lines
    ax.view_init(elev=ELEV, azim=AZIM)

    # Declutter: no axis lines, ticks, or labels, but keep the pane grid for
    # spatial reference.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        axis.set_label_text("")
        axis.line.set_visible(False)
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
    ax.tick_params(axis="both", which="both", length=0)

    # Food-site marker, along the food vector's direction, near the edge of
    # the ground square.
    unit_dir = food_vec / np.linalg.norm(food_vec)
    flower_point = edge_point(food_base, unit_dir, SQUARE_SIZE, FLOWER_EDGE_MARGIN)
    place_flower(ax, flower_point)

    fig.savefig(args.output, dpi=190, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
