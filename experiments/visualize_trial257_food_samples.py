from __future__ import annotations

import argparse
import csv
import json
import math
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


DEFAULT_CONFIG = ROOT / "configs" / "long_vertical_transition.json"
DEFAULT_POINTS = ROOT / "results" / "food_transition_v2_validation_points.csv"
DEFAULT_OUTPUT = ROOT / "report" / "figures" / "trial_257_food_samples.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample and visualize several food-site distributions for the "
            "baseline Optuna candidate trial_257."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base config used to define the simulation settings.",
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=DEFAULT_POINTS,
        help="CSV containing candidate parameter rows.",
    )
    parser.add_argument(
        "--candidate",
        default="trial_257",
        help="Candidate name to load from the points file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
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


def load_candidate_row(path: Path, candidate: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["candidate"] == candidate:
                return row
    raise ValueError(f"candidate {candidate!r} not found in {path}")


def load_settings(config_path: Path, row: dict[str, str]) -> DirectionSettings:
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    config.update(
        {
            "food_site_count": int(row["food_site_count"]),
            "food_site_width": float(row["food_site_width"]),
            "food_site_capacity": int(row["food_site_capacity"]),
            "food_value": float(row["food_value"]),
            "food_site_max_distance": float(row["food_site_max_distance"]),
            "vertical_comb_benefit": float(row["vertical_comb_benefit"]),
            "travel_cost_per_distance": float(row["travel_cost_per_distance"]),
            "mutation_sd": float(row["mutation_sd"]),
            "transposition_mutation_correlation": float(
                row["transposition_mutation_correlation"]
            ),
        }
    )
    return DirectionSettings(**config)


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


def draw_sample(ax, sites, settings, sun_azimuth: float) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    max_radius = settings.food_site_max_distance
    ax.set_ylim(0, max_radius)
    ax.set_yticks(np.linspace(0, max_radius, 5))
    # Drop both the radial distance labels and the angular (degree) tick labels;
    # the rings still convey relative distance without numeric clutter.
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Put the bee at the center to represent the observer.
    add_image_marker(ax, 0, 0, "🐝", 40, zorder=5)

    # Mark the sun's azimuth just outside the outer ring; it is the external
    # reference for the gravity-based code and is drawn fresh each episode.
    add_image_marker(ax, sun_azimuth, max_radius * 1.12, "☀️", 46, zorder=6)

    # Fixed, small marker: site size in the model is a directional tolerance, not
    # a spatial extent, so we do not map it to icon size here.
    for site in sites:
        add_image_marker(
            ax,
            site.direction,
            site.distance,
            "🌸",
            26,
            zorder=3,
        )

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

    ax.set_rlabel_position(90)
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = parse_args()
    row = load_candidate_row(args.points, args.candidate)
    settings = load_settings(args.config, row)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        math.ceil(args.samples / 2),
        subplot_kw={"projection": "polar"},
        figsize=(16, 9),
    )
    axes = np.asarray(axes).ravel()

    for sample_index in range(args.samples):
        rng = __import__("random").Random(args.seed + sample_index)
        drawn = generate_food_sites(settings, rng)
        sun_azimuth = sample_sun_azimuth(settings, rng)
        draw_sample(axes[sample_index], drawn, settings, sun_azimuth)

    for ax in axes[args.samples:]:
        ax.axis("off")

    # Parameter values (site count, width, max distance, capacity) are reported
    # in the figure caption rather than overlaid on the image, to reduce clutter.
    plt.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
