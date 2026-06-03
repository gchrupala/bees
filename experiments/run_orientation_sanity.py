from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from math import atan2, cos, hypot, pi, sin, tau
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate

FIELDNAMES = [
    "initial_comb_tilt",
    "vertical_comb_benefit",
    "seeds",
    "mean_final_tilt",
    "mean_final_sender_transposition",
    "mean_final_receiver_transposition",
    "mean_final_success",
    "mean_final_payoff",
    "mean_within_orientation_alignment",
    "across_seed_orientation_alignment",
    "weighted_across_seed_orientation_alignment",
    "weighted_mean_orientation_degrees",
    "elapsed_seconds",
]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    seeds = parse_ints(args.seeds)
    initial_comb_tilts = parse_floats(args.initial_comb_tilts)
    vertical_comb_benefits = parse_floats(args.vertical_comb_benefits)

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    sys.stdout.flush()

    for initial_comb_tilt in initial_comb_tilts:
        for vertical_comb_benefit in vertical_comb_benefits:
            started = perf_counter()
            settings = replace(
                base_settings,
                initial_comb_tilt=initial_comb_tilt,
                vertical_comb_benefit=vertical_comb_benefit,
            )
            row = summarize_condition(
                initial_comb_tilt=initial_comb_tilt,
                vertical_comb_benefit=vertical_comb_benefit,
                settings=settings,
                seeds=seeds,
            )
            row["elapsed_seconds"] = format_float(perf_counter() - started)
            writer.writerow(row)
            sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether comb-tilt orientation converges within runs and "
            "across seeds under selected tilt-geometry conditions."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "horizontal_direction.json",
        help="Path to the base model config.",
    )
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105,106,107,108,109,110",
        help="Comma-separated random seeds to average over.",
    )
    parser.add_argument(
        "--initial-comb-tilts",
        default="0.0,0.5,1.0",
        help="Comma-separated initial comb tilt values.",
    )
    parser.add_argument(
        "--vertical-comb-benefits",
        default="0.15,0.20,0.25",
        help="Comma-separated vertical-comb benefit values.",
    )
    return parser.parse_args()


def load_settings(config_path: Path) -> DirectionSettings:
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    return DirectionSettings(**config)


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def summarize_condition(
    initial_comb_tilt: float,
    vertical_comb_benefit: float,
    settings: DirectionSettings,
    seeds: Iterable[int],
) -> dict[str, str]:
    finals = [simulate(settings, seed=seed)[-1] for seed in seeds]
    orientation_summary = summarize_orientations(finals)

    return {
        "initial_comb_tilt": format_float(initial_comb_tilt),
        "vertical_comb_benefit": format_float(vertical_comb_benefit),
        "seeds": str(len(finals)),
        "mean_final_tilt": mean_summary(finals, "average_comb_tilt"),
        "mean_final_sender_transposition": mean_summary(
            finals,
            "average_sender_transposition",
        ),
        "mean_final_receiver_transposition": mean_summary(
            finals,
            "average_receiver_transposition",
        ),
        "mean_final_success": mean_summary(finals, "average_success_rate"),
        "mean_final_payoff": mean_summary(finals, "average_payoff"),
        "mean_within_orientation_alignment": format_float(
            orientation_summary.mean_within_alignment,
        ),
        "across_seed_orientation_alignment": format_float(
            orientation_summary.across_seed_alignment,
        ),
        "weighted_across_seed_orientation_alignment": format_float(
            orientation_summary.weighted_across_seed_alignment,
        ),
        "weighted_mean_orientation_degrees": format_float(
            orientation_summary.weighted_mean_orientation * 180.0 / pi,
        ),
        "elapsed_seconds": "",
    }


@dataclass(frozen=True)
class OrientationSummary:
    mean_within_alignment: float
    across_seed_alignment: float
    weighted_across_seed_alignment: float
    weighted_mean_orientation: float


def summarize_orientations(states: list[GenerationSummary]) -> OrientationSummary:
    unweighted_x = sum(cos(state.average_comb_orientation) for state in states)
    unweighted_y = sum(sin(state.average_comb_orientation) for state in states)
    across_seed_alignment = hypot(unweighted_x, unweighted_y) / len(states)

    weighted_x = sum(
        state.comb_orientation_alignment * cos(state.average_comb_orientation)
        for state in states
    )
    weighted_y = sum(
        state.comb_orientation_alignment * sin(state.average_comb_orientation)
        for state in states
    )
    total_weight = sum(state.comb_orientation_alignment for state in states)

    if total_weight <= 0.0:
        weighted_across_seed_alignment = 0.0
        weighted_mean_orientation = 0.0
    else:
        weighted_across_seed_alignment = hypot(weighted_x, weighted_y) / total_weight
        weighted_mean_orientation = atan2(weighted_y, weighted_x) % tau

    return OrientationSummary(
        mean_within_alignment=mean(
            state.comb_orientation_alignment for state in states
        ),
        across_seed_alignment=across_seed_alignment,
        weighted_across_seed_alignment=weighted_across_seed_alignment,
        weighted_mean_orientation=weighted_mean_orientation,
    )


def mean_summary(states: list[GenerationSummary], attribute: str) -> str:
    return format_float(mean(getattr(state, attribute) for state in states))


def format_float(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
