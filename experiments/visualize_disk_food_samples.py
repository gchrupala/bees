from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw, ImageFont

# Match plotnine's default text styling (theme_gray uses the sans-serif family,
# which resolves to DejaVu Sans, with base size 11 and a 13.2 title).
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 11,
        "figure.titlesize": 13.2,
    }
)

ROOT = Path(__file__).resolve().parents[1]
# NotoColorEmoji is a bitmap (CBDT) font that only ships a single strike, so it
# must be loaded at this exact pixel size; we resize the rendered glyph after.
FONT_PATH = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMOJI_STRIKE_SIZE = 109
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, generate_food_sites, sample_sun_azimuth

DEFAULT_CONFIG = ROOT / "configs" / "food_distribution_disk.json"
DEFAULT_OUTPUT = ROOT / "report" / "figures" / "disk_food_samples.png"

DISK_FACE = "#4C72B0"
DISK_EDGE = "#2F4A75"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample and visualize several food-site distributions under the "
            "ecological disk geometry, drawing each patch as a spatially "
            "to-scale disk so the marker size shows the sampled patch radius."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Disk-geometry config that defines the representative settings.",
    )
    parser.add_argument(
        "--food-site-count",
        type=int,
        default=6,
        help=(
            "Override the site count for a fuller display; the config baseline "
            "of 2 leaves most panels near-empty. Other settings are untouched."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sampled food distributions to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Base random seed for the sample realizations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the visualization.",
    )
    return parser.parse_args()


def load_settings(config_path: Path, food_site_count: int) -> DirectionSettings:
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    config["food_site_count"] = food_site_count
    settings = DirectionSettings(**config)
    if settings.food_geometry != "disk":
        raise ValueError(
            f"config {config_path} is not disk geometry (food_geometry="
            f"{settings.food_geometry!r})"
        )
    return settings


def render_symbol_image(text: str, size: int) -> np.ndarray:
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

    # Crop to the rendered glyph and scale the fixed-size strike to the
    # requested marker size.
    image = image.crop(image.getbbox())
    image = image.resize((size, size), Image.LANCZOS)
    return np.asarray(image)


def add_image_marker(ax, theta: float, radius: float, text: str, size: int, zorder: int) -> None:
    image = render_symbol_image(text, size)
    offset_image = OffsetImage(image, zoom=1.0, interpolation="nearest")
    annotation = AnnotationBbox(
        offset_image,
        (theta, radius),
        xycoords="data",
        boxcoords="data",
        frameon=False,
        pad=0,
        annotation_clip=False,
        zorder=zorder,
    )
    ax.add_artist(annotation)


def draw_disk(ax, center_distance: float, center_bearing: float, radius: float) -> None:
    """Draw a patch as a spatially to-scale filled disk on the polar axis.

    The disk boundary is a true circle in Cartesian space centered at the polar
    point ``(center_distance, center_bearing)``; we sample it and convert back
    to polar coordinates so the drawn extent equals the sampled patch radius.
    """
    x0 = center_distance * math.cos(center_bearing)
    y0 = center_distance * math.sin(center_bearing)
    t = np.linspace(0.0, 2 * math.pi, 120)
    xs = x0 + radius * np.cos(t)
    ys = y0 + radius * np.sin(t)
    thetas = np.arctan2(ys, xs)
    radii = np.hypot(xs, ys)
    ax.fill(
        thetas,
        radii,
        facecolor=DISK_FACE,
        edgecolor=DISK_EDGE,
        linewidth=1.0,
        alpha=0.6,
        zorder=3,
    )


def style_polar_axis(ax, max_radius: float) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max_radius)
    ax.set_yticks(np.linspace(0, max_radius, 5))
    # Drop both the radial distance labels and the angular (degree) tick labels;
    # the rings still convey relative distance without numeric clutter.
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_rlabel_position(90)
    ax.grid(True, alpha=0.3)


def draw_sample(ax, sites, settings, sun_azimuth: float) -> None:
    max_radius = settings.food_site_max_distance
    style_polar_axis(ax, max_radius)

    # Put the bee at the center to represent the observer.
    add_image_marker(ax, 0, 0, "🐝", 40, zorder=5)

    # Mark the sun's azimuth just outside the outer ring; it is the external
    # reference for the gravity-based code and is drawn fresh each episode.
    add_image_marker(ax, sun_azimuth, max_radius * 1.12, "☀️", 46, zorder=6)

    # Each patch is drawn as a to-scale disk (size == sampled radius); a small
    # fixed flower marks the center so the site reads as food. Unlike the legacy
    # angular model, the patch radius here is a physical spatial extent, so the
    # disk -- not the flower -- carries the radius information.
    for site in sites:
        draw_disk(ax, site.distance, site.direction, site.radius)
        add_image_marker(ax, site.direction, site.distance, "🌸", 11, zorder=4)

    # faint dashed ring marking the maximum food distance (outer boundary)
    ax.plot(
        np.linspace(0, 2 * math.pi, 360),
        [max_radius] * 360,
        color="#999999",
        linewidth=0.8,
        linestyle=(0, (1, 2)),
        alpha=0.5,
        zorder=1,
    )


def draw_size_key(ax, settings: DirectionSettings) -> None:
    """A plain-axis legend translating disk size into patch radius, drawn to the
    same data scale as the sample panels so the disks are directly comparable."""
    max_radius = settings.food_site_max_distance
    ax.set_aspect("equal")
    ax.set_xlim(-0.05 * max_radius, max_radius)
    ax.set_ylim(0, max_radius)
    ax.axis("off")
    ax.set_title("Patch radius key", fontsize=11)

    key_radii = [0.05, 0.20, 0.40, 0.80]
    x_center = max_radius * 0.42
    spacing = max_radius / (len(key_radii) + 1)
    for index, radius in enumerate(key_radii):
        y = max_radius - (index + 1) * spacing
        circle = plt.Circle(
            (x_center, y),
            radius,
            facecolor=DISK_FACE,
            edgecolor=DISK_EDGE,
            linewidth=0.8,
            alpha=0.55,
        )
        ax.add_patch(circle)
        ax.text(
            x_center + max_radius * 0.16,
            y,
            f"r = {radius:.2f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.text(
        x_center,
        max_radius * 0.06,
        f"median r = {settings.food_site_radius:.2f}\n"
        f"log-sd = {settings.food_site_radius_log_sd:.2f}",
        va="bottom",
        ha="center",
        fontsize=9,
        color="#444444",
    )


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config, args.food_site_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    columns = math.ceil((args.samples + 1) / 2)
    fig = plt.figure(figsize=(16, 9))

    for sample_index in range(args.samples):
        ax = fig.add_subplot(2, columns, sample_index + 1, projection="polar")
        rng = random.Random(args.seed + sample_index)
        sites = generate_food_sites(settings, rng)
        sun_azimuth = sample_sun_azimuth(settings, rng)
        draw_sample(ax, sites, settings, sun_azimuth)
        ax.set_title(f"Sample {sample_index + 1} ({len(sites)} sites)")

    key_ax = fig.add_subplot(2, columns, args.samples + 1)
    draw_size_key(key_ax, settings)

    for cell in range(args.samples + 2, 2 * columns + 1):
        blank = fig.add_subplot(2, columns, cell)
        blank.axis("off")

    fig.suptitle(
        "Disk-geometry food-site samples "
        f"(n={settings.food_site_count}, median radius="
        f"{settings.food_site_radius:.2f}, max distance="
        f"{settings.food_site_max_distance:.1f}, capacity="
        f"{settings.food_site_capacity}); disk size is the to-scale patch radius"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
