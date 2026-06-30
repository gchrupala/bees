"""Combined schematic: the two dance codes and how they blend on a tilted comb.

Three panels share one tilted comb (fixed tilt ``gamma`` and orientation
``phi``):

* (a) the comb seen in 3D, with its unit normal, gravity, an example world food
  direction, and the two in-plane reference directions obtained by projecting
  onto the comb -- the direct-code reference (length ``s_dir``) and the
  gravity-code reference (length ``s_grav``).
* (b, c) the *same* comb seen face-on (looking along the normal, so angles are
  undistorted), showing the weighted circular mean that blends the two codes at
  two transposition settings ``t_s``. The strengths ``s_dir`` and ``s_grav`` are
  fixed by the tilt, so only the weights ``w_dir = (1 - t_s) s_dir`` and
  ``w_grav = t_s s_grav`` -- and hence the resultant dance angle ``delta`` --
  change between the two panels.

Geometry is taken directly from ``bees.model`` so the figure stays faithful to
the simulation. Angles are computed without strength-based degradation (the
schematic shows the clean projected directions, not a noisy realisation).
"""

from __future__ import annotations

import argparse
import sys
from math import atan2, cos, hypot, pi, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Polygon
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

from bees.model import (  # noqa: E402
    _comb_basis,
    _comb_normal,
    _dot,
    _length,
    _project_onto_plane,
    _world_direction_vector,
)

DEFAULT_OUTPUT = ROOT / "report" / "figures" / "comb_codes.png"

# Fixed scene parameters chosen only for legibility of the schematic.
COMB_TILT = 0.5  # gamma: a half-tilted comb (theta = pi/4)
COMB_ORIENTATION = pi / 6.0  # phi: compass heading the comb faces
FOOD_AZIMUTH = 1.0  # d: example food direction (radians)
SUN_AZIMUTH = 2.16  # chosen so delta_grav is well separated from delta_dir
T_S_VALUES = (0.25, 0.75)  # direct-dominated and gravity-dominated blends

COMB_RADIUS = 0.78
FACE_COMB_RADIUS = 0.60  # face-on comb sits inside the ring with padding
HEX_RADIUS = 0.13
SURFACE_LIFT = 0.04  # lift 3D in-plane arrows just off the comb so they stay visible

DIRECT_COLOR = "#1f77b4"
GRAVITY_COLOR = "#d62728"
NORMAL_COLOR = "#2c2c2c"
RESULT_COLOR = "#2c2c2c"
GRAVITY_REF_COLOR = "#7f7f7f"
COMB_FACE = (0.953, 0.851, 0.627, 1.0)  # opaque wax
COMB_EDGE = "#c79a3f"

# Food-site marker: the same pink blossom emoji used in the environment figure
# (Fig 3), rendered as a NotoColorEmoji bitmap since matplotlib text cannot draw
# colour emoji.
FOOD_MARKER = "🌸"
FOOD_MARKER_SIZE = 30
FONT_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMOJI_STRIKE_SIZE = 109  # NotoColorEmoji ships a single bitmap strike at this size


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


def render_symbol_image(text: str, size: int) -> np.ndarray:
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


def place_flower(ax, xy, *, xycoords="data") -> None:
    image = render_symbol_image(FOOD_MARKER, FOOD_MARKER_SIZE)
    annotation = AnnotationBbox(
        OffsetImage(image, zoom=1.0, interpolation="nearest"),
        xy, xycoords=xycoords, frameon=False, pad=0,
        annotation_clip=False, zorder=6,
    )
    ax.add_artist(annotation)


def projected_angle_and_strength(vector, basis) -> tuple[float, float]:
    """Clean (non-degraded) in-plane angle and strength of a world vector."""
    projected = _project_onto_plane(vector, basis.normal)
    strength = _length(projected)
    if strength <= 1e-9:
        return 0.0, 0.0
    angle = atan2(_dot(projected, basis.second_axis), _dot(projected, basis.first_axis))
    return angle % (2 * pi), strength


# ---------------------------------------------------------------------------
# Panel (a): the comb in 3D.
# ---------------------------------------------------------------------------
def draw_3d_panel(ax, basis) -> None:
    first = np.asarray(basis.first_axis)
    second = np.asarray(basis.second_axis)
    normal = np.asarray(_comb_normal(COMB_TILT, COMB_ORIENTATION))
    lift = SURFACE_LIFT * normal

    polygons = [
        [u * first + v * second for u, v in hex_vertices(cx, cy, HEX_RADIUS)]
        for cx, cy in hex_centers(COMB_RADIUS, HEX_RADIUS)
    ]
    comb = Poly3DCollection(
        polygons, facecolor=COMB_FACE, edgecolor=COMB_EDGE, linewidth=0.7
    )
    comb.set_zorder(0)
    ax.add_collection3d(comb)

    # Horizontal reference ring around the comb: the near arc (camera side of the
    # comb plane, p . n > 0) masks the comb, the far arc passes behind it. Dots
    # mark east and north.
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

    def arrow(vector, color, label, *, base=(0, 0, 0), dashed=False, width=2.6,
              label_offset=(0, 0, 0)):
        base = np.asarray(base, dtype=float)
        vector = np.asarray(vector, dtype=float)
        ax.quiver(
            base[0], base[1], base[2], vector[0], vector[1], vector[2],
            color=color, linewidth=width, arrow_length_ratio=0.16,
            linestyle="--" if dashed else "-", zorder=5,
        )
        if label:
            pos = base + vector * 1.1 + np.asarray(label_offset)
            ax.text(pos[0], pos[1], pos[2], label, color=color, fontsize=13, zorder=6)

    arrow(normal, NORMAL_COLOR, r"$\mathbf{n}$", label_offset=(0.05, 0.0, 0.04))
    arrow((0, 0, 1), GRAVITY_REF_COLOR, r"$\mathbf{g}$", dashed=True, width=1.8)

    food = 0.75 * np.asarray(_world_direction_vector(FOOD_AZIMUTH))
    arrow(food, DIRECT_COLOR, r"$\mathbf{f}_d$", dashed=True, width=1.8,
          label_offset=(0.05, -0.05, -0.04))
    direct_ref = np.asarray(
        _project_onto_plane(_world_direction_vector(FOOD_AZIMUTH), tuple(normal))
    )
    arrow(direct_ref, DIRECT_COLOR, r"$s_{\mathrm{dir}}$", base=lift,
          label_offset=(0.05, 0.0, 0.1))

    gravity_ref = np.asarray(_project_onto_plane((0.0, 0.0, 1.0), tuple(normal)))
    arrow(gravity_ref, GRAVITY_COLOR, r"$s_{\mathrm{grav}}$", base=lift,
          label_offset=(-0.08, 0.0, 0.08))

    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(-0.45, 1.15)
    ax.set_box_aspect((1, 1, 0.85), zoom=1.35)
    ax.view_init(elev=20, azim=-55)
    ax.set_axis_off()
    ax.set_title("(a) comb geometry", fontsize=13, y=0.97)

    # Flower marking the food-site direction, on the ring at the food bearing.
    # Projected to the axes' 2D frame (after the view is fixed) so the bitmap
    # sits at the right bearing; ax.transData keeps it placed if the layout moves.
    food_dir = np.asarray(_world_direction_vector(FOOD_AZIMUTH))
    fpos = 1.04 * food_dir
    x2, y2, _ = proj3d.proj_transform(fpos[0], fpos[1], 0.0, ax.get_proj())
    place_flower(ax, (x2, y2), xycoords=ax.transData)


# ---------------------------------------------------------------------------
# Panels (b, c): the same comb face-on, blending the two codes.
# ---------------------------------------------------------------------------
def draw_face_panel(ax, t_s, delta_dir, s_dir, delta_grav, s_grav, compass, *,
                    scale) -> None:
    # Honeycomb disc seen face-on (the comb's own 2D frame), drawn inside the ring
    # with padding. Reuse the exact pointy-top vertices from the 3D panel so the
    # tiling matches and tessellates.
    for cx, cy in hex_centers(FACE_COMB_RADIUS, HEX_RADIUS):
        ax.add_patch(
            Polygon(
                hex_vertices(cx, cy, HEX_RADIUS), closed=True,
                facecolor=COMB_FACE, edgecolor=COMB_EDGE, linewidth=0.7, zorder=0,
            )
        )
    ax.add_patch(Circle((0, 0), COMB_RADIUS, fill=False, edgecolor="#bbbbbb",
                         linewidth=0.8, zorder=1))

    # East and north marked on the ring at their in-plane bearings, matching the
    # 3D panel's compass.
    for bearing, name in compass:
        dx, dy = COMB_RADIUS * cos(bearing), COMB_RADIUS * sin(bearing)
        ax.scatter([dx], [dy], color="#888888", s=18, zorder=5)
        ax.text(dx * 1.12, dy * 1.12, name, color="#888888", fontsize=9,
                ha="center", va="center")

    w_dir = (1.0 - t_s) * s_dir
    w_grav = t_s * s_grav

    def ray(angle, length, color, *, dashed=False, width=2.6, zorder=4):
        x, y = length * scale * cos(angle), length * scale * sin(angle)
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0), zorder=zorder,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=width,
                            linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0),
        )

    def weight_label(angle, weight, color, label, perp_sign):
        # Place beside the bold arrow, offset perpendicular to it so the text
        # never lands on the (collinear) dashed ray or the arrow shaft.
        bx = 0.6 * weight * scale * cos(angle)
        by = 0.6 * weight * scale * sin(angle)
        px, py = -sin(angle), cos(angle)
        ax.text(bx + perp_sign * 0.18 * px, by + perp_sign * 0.18 * py, label,
                color=color, fontsize=12, ha="center", va="center", zorder=6)

    # Full-strength reference directions (faint), then the weighted contributions
    # (bold) along the same directions: bold length is a fraction of the faint ray.
    ray(delta_dir, s_dir, DIRECT_COLOR, dashed=True, width=1.3, zorder=3)
    ray(delta_grav, s_grav, GRAVITY_COLOR, dashed=True, width=1.3, zorder=3)
    ray(delta_dir, w_dir, DIRECT_COLOR, width=3.0, zorder=4)
    ray(delta_grav, w_grav, GRAVITY_COLOR, width=3.0, zorder=4)
    weight_label(delta_dir, w_dir, DIRECT_COLOR, r"$w_{\mathrm{dir}}$", -1)
    weight_label(delta_grav, w_grav, GRAVITY_COLOR, r"$w_{\mathrm{grav}}$", +1)

    # Flower marking the food-site direction (its in-plane projection delta_dir),
    # in the padding between the comb and the ring.
    fr = FACE_COMB_RADIUS + 0.11
    place_flower(ax, (fr * cos(delta_dir), fr * sin(delta_dir)))

    # Parallelogram closing the two weighted contributions onto the resultant.
    dx, dy = w_dir * scale * cos(delta_dir), w_dir * scale * sin(delta_dir)
    gx, gy = w_grav * scale * cos(delta_grav), w_grav * scale * sin(delta_grav)
    rx, ry = dx + gx, dy + gy
    for (sx, sy) in ((dx, dy), (gx, gy)):
        ax.plot([sx, rx], [sy, ry], color="#999999", linewidth=0.9, linestyle=":",
                zorder=2)
    ax.annotate(
        "", xy=(rx, ry), xytext=(0, 0), zorder=5,
        arrowprops=dict(arrowstyle="-|>", color=RESULT_COLOR, lw=3.4, shrinkA=0,
                        shrinkB=0),
    )
    ax.text(rx * 1.14, ry * 1.14, r"$\delta$", color=RESULT_COLOR, fontsize=13,
            ha="center", va="center", zorder=6)

    ax.set_xlim(-COMB_RADIUS * 1.35, COMB_RADIUS * 1.35)
    ax.set_ylim(-COMB_RADIUS * 1.35, COMB_RADIUS * 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    panel = "(b)" if t_s == T_S_VALUES[0] else "(c)"
    ax.set_title(rf"{panel} $t_s = {t_s:g}$", fontsize=13)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    basis = _comb_basis(COMB_TILT, COMB_ORIENTATION)
    delta_dir, s_dir = projected_angle_and_strength(
        _world_direction_vector(FOOD_AZIMUTH), basis
    )
    ref_angle, s_grav = projected_angle_and_strength((0.0, 0.0, 1.0), basis)
    delta_grav = (ref_angle + FOOD_AZIMUTH - SUN_AZIMUTH) % (2 * pi)

    # In-plane bearings of world east and north, to mark the ring face-on.
    compass = [
        (projected_angle_and_strength((1.0, 0.0, 0.0), basis)[0], "east"),
        (projected_angle_and_strength((0.0, 1.0, 0.0), basis)[0], "north"),
    ]

    # Scale so the stronger reference reaches near the face-on comb's edge,
    # keeping the arrows on the wax and inside the ring.
    scale = (FACE_COMB_RADIUS * 0.95) / max(s_dir, s_grav)

    fig = plt.figure(figsize=(14.0, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d", computed_zorder=False)
    draw_3d_panel(ax3d, basis)
    for col, t_s in enumerate(T_S_VALUES, start=1):
        ax = fig.add_subplot(gs[0, col])
        draw_face_panel(ax, t_s, delta_dir, s_dir, delta_grav, s_grav, compass,
                        scale=scale)

    fig.tight_layout()
    fig.savefig(args.output, dpi=190)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
