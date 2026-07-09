"""Schematic illustration of the honeybee waggle dance.

Two versions are drawn side by side:

  (1) A bee dancing on a *horizontal* comb, where the straight waggle run points
      directly at the food patch (the celestial map is transferred unchanged).
  (2) A bee dancing on a *vertical* comb, where gravity replaces the sun: the
      angle of the waggle run relative to straight up equals the food's azimuth
      relative to the sun.

The figure is intentionally schematic rather than to scale.
"""
from __future__ import annotations

import argparse
from math import cos, radians, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyBboxPatch,
    Rectangle,
    RegularPolygon,
)
from matplotlib.transforms import Bbox, TransformedBbox
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "report" / "figures" / "waggle_dance_schematic.png"

# NotoColorEmoji is a bitmap (CBDT) font with a single strike, so it must be
# loaded at this pixel size; rendered glyphs are resized afterwards.
FONT_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMOJI_STRIKE_SIZE = 109

# Match plotnine's default text styling (sans-serif -> DejaVu Sans).
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
        "font.size": 13,
    }
)

COMB_FILL = "#fdf3d3"
COMB_EDGE = "#caa53d"
HEX_COLOR = "#e9cf86"
RUN_COLOR = "#c0392b"
SUN_COLOR = "#e8a200"
REF_COLOR = "#34495e"

# Panel header colours (blue for the horizontal comb, green for the vertical one)
# and a common style for the large arrow annotations.
HEAD_H = "#1f5fa8"
HEAD_V = "#2e7d32"
LABEL_SIZE = 15
# Light grey for the azimuth angle alpha, so it does not clash with the orange sun
# reference or the red waggle run.
AZIMUTH_COLOR = "#8a8a8a"

# Shared field geometry, measured counter-clockwise from east (the +x axis).
# The food patch sits 40 degrees clockwise of (i.e. to the right of) the sun.
SUN_AZIMUTH_DEG = 75.0
FOOD_AZIMUTH_DEG = 35.0
AZIMUTH_DEG = SUN_AZIMUTH_DEG - FOOD_AZIMUTH_DEG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the schematic.",
    )
    return parser.parse_args()


def render_emoji(text: str, size: int) -> np.ndarray:
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

    image = image.crop(image.getbbox()).resize((size, size), Image.LANCZOS)
    return np.asarray(image)


def place_emoji(ax, x: float, y: float, text: str, size: int, zorder: int = 6) -> None:
    offset_image = OffsetImage(render_emoji(text, size), zoom=1.0)
    annotation = AnnotationBbox(
        offset_image,
        (x, y),
        frameon=False,
        pad=0,
        annotation_clip=False,
        zorder=zorder,
    )
    ax.add_artist(annotation)


def draw_comb_texture(ax, clip_patch, bbox: tuple[float, float, float, float], radius: float) -> None:
    """Fill the clipped comb shape with a faint honeycomb hex pattern."""
    x_min, x_max, y_min, y_max = bbox
    dx = sqrt(3.0) * radius
    dy = 1.5 * radius

    # Clip to the comb outline, plus its bounding box as a robust fallback
    # (path clipping alone can leak hexagons past a rectangular comb).
    clip_box = TransformedBbox(
        Bbox.from_extents(x_min, y_min, x_max, y_max), ax.transData
    )

    row = 0
    y = y_min - dy
    while y <= y_max + dy:
        offset = dx / 2.0 if row % 2 else 0.0
        x = x_min - dx + offset
        while x <= x_max + dx:
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=radius,
                orientation=0.0,
                fill=False,
                edgecolor=HEX_COLOR,
                linewidth=0.6,
                zorder=1,
            )
            ax.add_patch(hexagon)
            hexagon.set_clip_path(clip_patch)
            hexagon.set_clip_box(clip_box)
            x += dx
        y += dy
        row += 1


def draw_dance_loops(ax, center: tuple[float, float], angle_deg: float, length: float) -> None:
    """Draw the two figure-eight loops of the dance, centred on the bee.

    The loops bulge to either side of the run (perpendicular to it) and meet at a
    single waist at the centre, which the bee then covers.
    """
    angle = radians(angle_deg)
    px, py = -sin(angle), cos(angle)
    cx, cy = center

    loop_offset = 0.36 * length
    for sign in (-1.0, 1.0):
        loop = Ellipse(
            (cx + sign * loop_offset * px, cy + sign * loop_offset * py),
            width=0.66 * length,
            height=0.72 * length,
            angle=angle_deg,
            fill=False,
            edgecolor=RUN_COLOR,
            linewidth=1.4,
            alpha=0.55,
            zorder=4,
        )
        ax.add_patch(loop)


def draw_run_arrow(ax, origin, angle_deg: float, base_r: float, tip_r: float) -> None:
    """Draw the straight waggle run as a bold arrow out from the reference vertex."""
    ox, oy = origin
    angle = radians(angle_deg)
    ax.annotate(
        "",
        xy=(ox + tip_r * cos(angle), oy + tip_r * sin(angle)),
        xytext=(ox + base_r * cos(angle), oy + base_r * sin(angle)),
        arrowprops=dict(arrowstyle="-|>", color=RUN_COLOR, lw=3.0, mutation_scale=24),
        zorder=5,
    )


def draw_ray(ax, origin, angle_deg: float, length: float, color: str, style: str, lw: float = 1.6) -> None:
    ox, oy = origin
    angle = radians(angle_deg)
    ax.plot(
        [ox, ox + length * cos(angle)],
        [oy, oy + length * sin(angle)],
        color=color,
        linestyle=style,
        linewidth=lw,
        zorder=3,
    )


def draw_panel_frame(ax, color: str, title: str) -> None:
    """Draw a rounded border box with a coloured header for one panel."""
    frame = FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0,rounding_size=0.035",
        transform=ax.transAxes,
        facecolor="none",
        edgecolor=color,
        linewidth=2.4,
        zorder=20,
        clip_on=False,
    )
    ax.add_patch(frame)
    ax.text(
        0.5, 0.895, title,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=19, fontweight="bold", color=color, zorder=21,
    )


def draw_horizontal_panel(ax) -> None:
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.5, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_panel_frame(ax, HEAD_H, "Horizontal comb")

    R = 1.08
    comb = Circle(
        (0, 0), R, facecolor=COMB_FILL, edgecolor=COMB_EDGE, linewidth=1.6, zorder=0
    )
    ax.add_patch(comb)
    draw_comb_texture(ax, comb, (-R, R, -R, R), radius=0.16)

    # The reference vertex sits in the lower-left of the comb; the sun reference
    # and the run toward the food start there, with the azimuth angle alpha marked at the vertex,
    # while the bee dances near the comb centre -- so the angle and the
    # figure-eight are spatially separated.
    u = radians(FOOD_AZIMUTH_DEG)
    vertex = (-0.72 * cos(u), -0.72 * sin(u))

    draw_ray(ax, vertex, SUN_AZIMUTH_DEG, 1.78, SUN_COLOR, (0, (4, 3)), lw=1.4)
    draw_run_arrow(ax, vertex, FOOD_AZIMUTH_DEG, 0.0, 1.5)

    ax.add_patch(
        Arc(
            vertex, 0.8, 0.8, angle=0.0,
            theta1=FOOD_AZIMUTH_DEG, theta2=SUN_AZIMUTH_DEG,
            color=AZIMUTH_COLOR, linewidth=2.0,
        )
    )
    mid = radians((SUN_AZIMUTH_DEG + FOOD_AZIMUTH_DEG) / 2.0)
    ax.text(
        vertex[0] + 0.52 * cos(mid), vertex[1] + 0.52 * sin(mid), "α",
        fontsize=22, color=AZIMUTH_COLOR, ha="center", va="center", fontweight="bold",
    )

    bee = (vertex[0] + 0.88 * cos(u), vertex[1] + 0.88 * sin(u))
    draw_dance_loops(ax, bee, FOOD_AZIMUTH_DEG, length=0.72)
    place_emoji(ax, bee[0], bee[1], "🐝", 46)

    place_emoji(
        ax,
        vertex[0] + 1.95 * cos(u),
        vertex[1] + 1.95 * sin(u),
        "🌼",
        46,
    )
    place_emoji(
        ax,
        vertex[0] + 1.78 * cos(radians(SUN_AZIMUTH_DEG)),
        vertex[1] + 1.78 * sin(radians(SUN_AZIMUTH_DEG)),
        "☀️",
        46,
    )


def draw_vertical_panel(ax) -> None:
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.5, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_panel_frame(ax, HEAD_V, "Vertical comb")

    H = 1.08
    comb = Rectangle(
        (-H, -H),
        2 * H,
        2 * H,
        facecolor=COMB_FILL,
        edgecolor=COMB_EDGE,
        linewidth=1.6,
        zorder=0,
    )
    ax.add_patch(comb)
    draw_comb_texture(ax, comb, (-H, H, -H, H), radius=0.16)

    # The reference vertex sits in the lower-left of the comb; the "up" (sun)
    # reference and the run both start there, with the azimuth angle alpha marked at the vertex,
    # while the bee dances near the comb centre.
    run_angle = 90.0 - AZIMUTH_DEG
    u = radians(run_angle)
    vertex = (-0.72 * cos(u), -0.72 * sin(u))

    ax.annotate(
        "",
        xy=(vertex[0], vertex[1] + 2.05),
        xytext=vertex,
        arrowprops=dict(arrowstyle="-|>", color=SUN_COLOR, lw=2.2, mutation_scale=22),
        zorder=3,
    )
    ax.text(
        vertex[0] + 0.14, vertex[1] + 1.72, "up = toward sun",
        color=SUN_COLOR, fontsize=LABEL_SIZE, va="center", ha="left",
        fontweight="bold",
    )

    draw_run_arrow(ax, vertex, run_angle, 0.0, 1.5)

    # Gravity reference (a world direction), to the left of the comb.
    ax.annotate(
        "",
        xy=(-1.34, -0.6),
        xytext=(-1.34, 0.6),
        arrowprops=dict(arrowstyle="-|>", color=REF_COLOR, lw=2.4, mutation_scale=20),
        zorder=3,
    )
    ax.text(
        -1.5, 0.0, "gravity",
        color=REF_COLOR, fontsize=LABEL_SIZE, rotation=90, va="center", ha="center",
        fontweight="bold",
    )

    ax.add_patch(
        Arc(
            vertex, 0.8, 0.8, angle=0.0,
            theta1=run_angle, theta2=90.0,
            color=AZIMUTH_COLOR, linewidth=2.0,
        )
    )
    mid = radians((90.0 + run_angle) / 2.0)
    ax.text(
        vertex[0] + 0.52 * cos(mid), vertex[1] + 0.52 * sin(mid), "α",
        fontsize=22, color=AZIMUTH_COLOR, ha="center", va="center", fontweight="bold",
    )

    bee = (vertex[0] + 0.88 * cos(u), vertex[1] + 0.88 * sin(u))
    draw_dance_loops(ax, bee, run_angle, length=0.72)
    place_emoji(ax, bee[0], bee[1], "🐝", 46)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 8))
    draw_horizontal_panel(left)
    draw_vertical_panel(right)

    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
