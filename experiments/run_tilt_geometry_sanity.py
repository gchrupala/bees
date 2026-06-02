from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate

FIELDNAMES = [
    "initial_comb_tilt",
    "vertical_comb_benefit",
    "seeds",
    "reached_fraction",
    "mean_reach_generation",
    "reach_generation_sd",
    "mean_final_bias",
    "mean_final_attention",
    "mean_final_sender_transposition",
    "mean_final_receiver_transposition",
    "mean_final_comb_tilt",
    "mean_final_comb_orientation_alignment",
    "mean_final_search_limit",
    "mean_final_success",
    "mean_final_payoff",
    "mean_comb_tilt_delta",
    "mean_sender_transposition_delta",
    "mean_receiver_transposition_delta",
    "mean_success_delta",
    "elapsed_seconds",
]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    seeds = parse_ints(args.seeds)
    initial_comb_tilts = parse_floats(args.initial_comb_tilts)
    vertical_comb_benefits = parse_floats(args.vertical_comb_benefits)

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
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
            row = summarize_setting(
                initial_comb_tilt=initial_comb_tilt,
                vertical_comb_benefit=vertical_comb_benefit,
                settings=settings,
                seeds=seeds,
                threshold=args.threshold,
            )
            row["elapsed_seconds"] = format_float(perf_counter() - started)
            writer.writerow(row)
            sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a grid sanity check over initial comb tilt and vertical-comb "
            "benefit for the tilt-geometry model."
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
        default="101,102,103,104,105",
        help="Comma-separated random seeds to average over.",
    )
    parser.add_argument(
        "--initial-comb-tilts",
        default="0.0,0.5,1.0",
        help="Comma-separated initial comb tilt values.",
    )
    parser.add_argument(
        "--vertical-comb-benefits",
        default="0.0,0.02,0.08",
        help="Comma-separated vertical-comb benefit values.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Directional-bias threshold used for the reach-generation metric.",
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


def summarize_setting(
    initial_comb_tilt: float,
    vertical_comb_benefit: float,
    settings: DirectionSettings,
    seeds: Iterable[int],
    threshold: float,
) -> dict[str, str]:
    histories = [simulate(settings, seed=seed) for seed in seeds]
    initial_states = [history[0] for history in histories]
    final_states = [history[-1] for history in histories]
    reach_generations = [
        first_reach_generation(history, threshold) for history in histories
    ]
    reached = [generation is not None for generation in reach_generations]
    reached_generations = [
        generation for generation in reach_generations if generation is not None
    ]

    return {
        "initial_comb_tilt": format_float(initial_comb_tilt),
        "vertical_comb_benefit": format_float(vertical_comb_benefit),
        "seeds": str(len(final_states)),
        "reached_fraction": format_float(mean(reached)),
        "mean_reach_generation": (
            format_float(mean(reached_generations))
            if reached_generations
            else "not_reached"
        ),
        "reach_generation_sd": (
            format_float(pstdev(reached_generations))
            if len(reached_generations) > 1
            else "0.000"
        ),
        "mean_final_bias": mean_summary(final_states, "average_directional_bias"),
        "mean_final_attention": mean_summary(final_states, "average_receiver_attention"),
        "mean_final_sender_transposition": mean_summary(
            final_states,
            "average_sender_transposition",
        ),
        "mean_final_receiver_transposition": mean_summary(
            final_states,
            "average_receiver_transposition",
        ),
        "mean_final_comb_tilt": mean_summary(final_states, "average_comb_tilt"),
        "mean_final_comb_orientation_alignment": mean_summary(
            final_states,
            "comb_orientation_alignment",
        ),
        "mean_final_search_limit": mean_summary(final_states, "average_search_limit"),
        "mean_final_success": mean_summary(final_states, "average_success_rate"),
        "mean_final_payoff": mean_summary(final_states, "average_payoff"),
        "mean_comb_tilt_delta": format_float(
            mean(
                final.average_comb_tilt - initial.average_comb_tilt
                for initial, final in zip(initial_states, final_states)
            )
        ),
        "mean_sender_transposition_delta": format_float(
            mean(
                final.average_sender_transposition
                - initial.average_sender_transposition
                for initial, final in zip(initial_states, final_states)
            )
        ),
        "mean_receiver_transposition_delta": format_float(
            mean(
                final.average_receiver_transposition
                - initial.average_receiver_transposition
                for initial, final in zip(initial_states, final_states)
            )
        ),
        "mean_success_delta": format_float(
            mean(
                final.average_success_rate - initial.average_success_rate
                for initial, final in zip(initial_states, final_states)
            )
        ),
        "elapsed_seconds": "",
    }


def first_reach_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if state.average_directional_bias >= threshold:
            return state.generation
    return None


def mean_summary(states: list[GenerationSummary], attribute: str) -> str:
    return format_float(mean(getattr(state, attribute) for state in states))


def format_float(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
