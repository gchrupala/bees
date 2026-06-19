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
from matplotlib.patches import Arc, Circle, Ellipse, Rectangle, RegularPolygon
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
        "font.size": 12,
    }
)

COMB_FILL = "#fdf3d3"
COMB_EDGE = "#caa53d"
HEX_COLOR = "#e9cf86"
RUN_COLOR = "#c0392b"
SUN_COLOR = "#e8a200"
REF_COLOR = "#34495e"

# Shared field geometry, measured counter-clockwise from east (the +x axis).
# The food patch sits 40 degrees clockwise of (i.e. to the right of) the sun.
SUN_AZIMUTH_DEG = 75.0
FOOD_AZIMUTH_DEG = 35.0
THETA_DEG = SUN_AZIMUTH_DEG - FOOD_AZIMUTH_DEG


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
            hexagon.set_clip_path(clip_patch)
            ax.add_patch(hexagon)
            x += dx
        y += dy
        row += 1


def draw_waggle(ax, center: tuple[float, float], angle_deg: float, length: float) -> None:
    """Draw a schematic figure-eight dance with a straight waggle run."""
    angle = radians(angle_deg)
    ux, uy = cos(angle), sin(angle)
    px, py = -sin(angle), cos(angle)
    cx, cy = center

    # The two return loops, offset perpendicular to the run and overlapping
    # along it so the straight run threads through the middle of the "8".
    loop_offset = 0.30 * length
    for sign in (-1.0, 1.0):
        loop = Ellipse(
            (cx + sign * loop_offset * px, cy + sign * loop_offset * py),
            width=1.18 * length,
            height=0.74 * length,
            angle=angle_deg,
            fill=False,
            edgecolor=RUN_COLOR,
            linewidth=1.4,
            alpha=0.55,
            zorder=4,
        )
        ax.add_patch(loop)

    # The straight waggle run itself, as a bold arrow.
    base = (cx - 0.5 * length * ux, cy - 0.5 * length * uy)
    tip = (cx + 0.62 * length * ux, cy + 0.62 * length * uy)
    ax.annotate(
        "",
        xy=tip,
        xytext=base,
        arrowprops=dict(arrowstyle="-|>", color=RUN_COLOR, lw=3.0, mutation_scale=24),
        zorder=5,
    )


def draw_ray(ax, angle_deg: float, length: float, color: str, style: str, lw: float = 1.6) -> None:
    angle = radians(angle_deg)
    ax.plot(
        [0.0, length * cos(angle)],
        [0.0, length * sin(angle)],
        color=color,
        linestyle=style,
        linewidth=lw,
        zorder=3,
    )


def draw_horizontal_panel(ax) -> None:
    ax.set_title("(1) Horizontal comb — run points straight at the food", pad=14)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")

    comb = Circle(
        (0, 0), 1.0, facecolor=COMB_FILL, edgecolor=COMB_EDGE, linewidth=1.6, zorder=0
    )
    ax.add_patch(comb)
    draw_comb_texture(ax, comb, (-1.0, 1.0, -1.0, 1.0), radius=0.16)

    # Guide rays toward the sun and the food, in the bee's own (world) frame.
    draw_ray(ax, SUN_AZIMUTH_DEG, 1.5, SUN_COLOR, (0, (4, 3)), lw=1.4)
    draw_ray(ax, FOOD_AZIMUTH_DEG, 1.3, RUN_COLOR, (0, (1, 2)), lw=1.2)

    draw_waggle(ax, (0.0, 0.0), FOOD_AZIMUTH_DEG, length=1.0)
    place_emoji(ax, 0, 0, "🐝", 46)

    place_emoji(
        ax,
        1.5 * cos(radians(FOOD_AZIMUTH_DEG)),
        1.5 * sin(radians(FOOD_AZIMUTH_DEG)),
        "🌼",
        46,
    )
    place_emoji(
        ax,
        1.62 * cos(radians(SUN_AZIMUTH_DEG)),
        1.62 * sin(radians(SUN_AZIMUTH_DEG)),
        "☀️",
        44,
    )
    ax.text(
        0.0,
        -1.45,
        "On a level comb the bee sees the sky, so the waggle run is\n"
        "aimed directly along the true food direction.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
    )


def draw_field_inset(ax) -> None:
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.5, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("in the field (from above)", fontsize=9.5, pad=2)

    draw_ray(ax, SUN_AZIMUTH_DEG, 1.0, SUN_COLOR, (0, (4, 3)), lw=1.3)
    draw_ray(ax, FOOD_AZIMUTH_DEG, 0.95, RUN_COLOR, "-", lw=1.6)
    ax.add_patch(
        Arc(
            (0, 0),
            1.1,
            1.1,
            angle=0.0,
            theta1=FOOD_AZIMUTH_DEG,
            theta2=SUN_AZIMUTH_DEG,
            color=REF_COLOR,
            linewidth=1.3,
        )
    )
    mid = radians((SUN_AZIMUTH_DEG + FOOD_AZIMUTH_DEG) / 2.0)
    ax.text(0.72 * cos(mid), 0.72 * sin(mid), "θ", fontsize=12, color=REF_COLOR)

    place_emoji(ax, 0, 0, "🐝", 24)
    place_emoji(ax, 1.05 * cos(radians(SUN_AZIMUTH_DEG)), 1.05 * sin(radians(SUN_AZIMUTH_DEG)), "☀️", 24)
    place_emoji(ax, 1.05 * cos(radians(FOOD_AZIMUTH_DEG)), 1.05 * sin(radians(FOOD_AZIMUTH_DEG)), "🌼", 24)


def draw_vertical_panel(ax) -> None:
    ax.set_title("(2) Vertical comb — angle from vertical = food's angle from the sun", pad=14)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")

    comb = Rectangle(
        (-1.0, -1.0),
        2.0,
        2.0,
        facecolor=COMB_FILL,
        edgecolor=COMB_EDGE,
        linewidth=1.6,
        zorder=0,
    )
    ax.add_patch(comb)
    draw_comb_texture(ax, comb, (-1.0, 1.0, -1.0, 1.0), radius=0.16)

    # "Up" on the comb represents the direction of the sun.
    ax.annotate(
        "",
        xy=(0.0, 1.45),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=SUN_COLOR, lw=1.6, mutation_scale=18),
        zorder=3,
    )
    ax.text(0.06, 1.5, "up ≡ toward the sun", color=SUN_COLOR, fontsize=10, va="center")

    # Gravity reference.
    ax.annotate(
        "",
        xy=(-1.45, -0.55),
        xytext=(-1.45, 0.55),
        arrowprops=dict(arrowstyle="-|>", color=REF_COLOR, lw=2.0, mutation_scale=18),
        zorder=3,
    )
    ax.text(-1.6, 0.0, "gravity", color=REF_COLOR, fontsize=10, rotation=90, va="center", ha="center")

    # Waggle run is THETA_DEG clockwise of straight up (90 deg from +x axis).
    run_angle = 90.0 - THETA_DEG
    draw_waggle(ax, (0.0, 0.0), run_angle, length=1.0)
    place_emoji(ax, 0, 0, "🐝", 46)

    # Angle arc between "up" and the waggle run.
    ax.add_patch(
        Arc(
            (0, 0),
            0.95,
            0.95,
            angle=0.0,
            theta1=run_angle,
            theta2=90.0,
            color=REF_COLOR,
            linewidth=1.6,
        )
    )
    mid = radians((90.0 + run_angle) / 2.0)
    ax.text(0.66 * cos(mid), 0.66 * sin(mid), "θ", fontsize=14, color=REF_COLOR)

    ax.text(
        0.0,
        -1.45,
        "On a vertical comb the bee cannot see the sun, so it uses gravity:\n"
        "it rotates the run away from vertical by the food–sun angle θ.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
    )

    inset = ax.inset_axes([0.66, 0.7, 0.34, 0.32])
    draw_field_inset(inset)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 8))
    draw_horizontal_panel(left)
    draw_vertical_panel(right)

    fig.suptitle("Schematic of the honeybee waggle dance", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
