from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, generate_food_sites


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


def draw_sample(ax, sites, settings, sample_index: int) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    max_radius = settings.food_site_max_distance
    ax.set_ylim(0, max_radius)
    ax.set_yticks(np.linspace(0, max_radius, 5))
    ax.set_yticklabels([f"{value:.0f}" for value in np.linspace(0, max_radius, 5)])
    ax.set_title(f"Sample {sample_index} ({settings.food_site_count} sites)")

    for site in sites:
        theta = site.direction
        radius = site.distance
        width = site.width
        capacity = site.capacity

        # The previous red spans showed the angular extent of each patch.
        # Here we use the circle diameter as a direct visual proxy for that
        # same extent, with a scale that keeps the markers readable.
        marker_size = max(600, 9000 * width)
        ax.scatter(
            [theta],
            [radius],
            s=marker_size,
            c="#4e79a7",
            alpha=0.75,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )

    # show the site-width span as a faint angular band around the full circle
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
        draw_sample(axes[sample_index], drawn, settings, sample_index + 1)

    for ax in axes[args.samples:]:
        ax.axis("off")

    fig.suptitle(
        f"Food-site samples for {args.candidate} "
        f"(n={settings.food_site_count}, width={settings.food_site_width:.3f}, "
        f"max distance={settings.food_site_max_distance:.1f}, capacity={settings.food_site_capacity})",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
