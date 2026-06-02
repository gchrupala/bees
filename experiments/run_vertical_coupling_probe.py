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
    "transposition_mutation_correlation",
    "seeds",
    "generations",
    "initial_comb_tilt",
    "comb_tilt_mutation_sd",
    "comb_orientation_axial",
    "reached_gravity_fraction",
    "mean_gravity_reach_generation",
    "gravity_reach_generation_sd",
    "reached_directional_fraction",
    "mean_directional_reach_generation",
    "directional_reach_generation_sd",
    "mean_final_bias",
    "mean_final_attention",
    "mean_final_sender_transposition",
    "mean_final_receiver_transposition",
    "mean_final_min_transposition",
    "mean_final_transposition_gap",
    "mean_final_comb_tilt",
    "mean_final_comb_orientation_alignment",
    "mean_final_search_limit",
    "mean_final_success",
    "mean_final_payoff",
    "mean_success_delta",
    "elapsed_seconds",
]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    seeds = parse_ints(args.seeds)
    correlations = parse_floats(args.correlations)

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    sys.stdout.flush()

    for correlation in correlations:
        started = perf_counter()
        settings = replace(
            base_settings,
            transposition_mutation_correlation=correlation,
        )
        row = summarize_setting(
            settings=settings,
            seeds=seeds,
            gravity_threshold=args.gravity_threshold,
            directional_threshold=args.directional_threshold,
        )
        row["elapsed_seconds"] = format_float(perf_counter() - started)
        writer.writerow(row)
        sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a constrained near-vertical, axial-orientation probe over "
            "sender-receiver transposition mutation correlation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "vertical_coupling_probe.json",
        help="Path to the constrained model config.",
    )
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105",
        help="Comma-separated random seeds to average over.",
    )
    parser.add_argument(
        "--correlations",
        default="0.0,0.3,0.6,0.9,1.0",
        help="Comma-separated sender-receiver mutation correlations.",
    )
    parser.add_argument(
        "--gravity-threshold",
        type=float,
        default=0.50,
        help=(
            "Sender and receiver transposition threshold for gravity-dominant "
            "communication."
        ),
    )
    parser.add_argument(
        "--directional-threshold",
        type=float,
        default=0.30,
        help="Directional-bias threshold used for the reach-generation metric.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override the generation count from the config.",
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
    settings: DirectionSettings,
    seeds: Iterable[int],
    gravity_threshold: float,
    directional_threshold: float,
) -> dict[str, str]:
    histories = [simulate(settings, seed=seed) for seed in seeds]
    initial_states = [history[0] for history in histories]
    final_states = [history[-1] for history in histories]
    gravity_reach_generations = [
        first_gravity_reach_generation(history, gravity_threshold)
        for history in histories
    ]
    directional_reach_generations = [
        first_directional_reach_generation(history, directional_threshold)
        for history in histories
    ]

    row = {
        "transposition_mutation_correlation": format_float(
            settings.transposition_mutation_correlation,
        ),
        "seeds": str(len(final_states)),
        "generations": str(settings.generations),
        "initial_comb_tilt": format_float(settings.initial_comb_tilt),
        "comb_tilt_mutation_sd": format_float(comb_tilt_mutation_sd(settings)),
        "comb_orientation_axial": str(settings.comb_orientation_axial).lower(),
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
        "mean_final_min_transposition": format_float(
            mean(
                min(
                    state.average_sender_transposition,
                    state.average_receiver_transposition,
                )
                for state in final_states
            )
        ),
        "mean_final_transposition_gap": format_float(
            mean(
                abs(
                    state.average_sender_transposition
                    - state.average_receiver_transposition
                )
                for state in final_states
            )
        ),
        "mean_final_comb_tilt": mean_summary(final_states, "average_comb_tilt"),
        "mean_final_comb_orientation_alignment": mean_summary(
            final_states,
            "comb_orientation_alignment",
        ),
        "mean_final_search_limit": mean_summary(final_states, "average_search_limit"),
        "mean_final_success": mean_summary(final_states, "average_success_rate"),
        "mean_final_payoff": mean_summary(final_states, "average_payoff"),
        "mean_success_delta": format_float(
            mean(
                final.average_success_rate - initial.average_success_rate
                for initial, final in zip(initial_states, final_states)
            )
        ),
        "elapsed_seconds": "",
    }
    row.update(reach_summary("gravity", gravity_reach_generations))
    row.update(reach_summary("directional", directional_reach_generations))
    return row


def first_gravity_reach_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if (
            state.average_sender_transposition >= threshold
            and state.average_receiver_transposition >= threshold
        ):
            return state.generation

    return None


def first_directional_reach_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if state.average_directional_bias >= threshold:
            return state.generation

    return None


def reach_summary(prefix: str, generations: list[int | None]) -> dict[str, str]:
    reached_generations = [
        generation for generation in generations if generation is not None
    ]

    return {
        f"reached_{prefix}_fraction": format_float(
            len(reached_generations) / len(generations),
        ),
        f"mean_{prefix}_reach_generation": (
            format_float(mean(reached_generations))
            if reached_generations
            else "not_reached"
        ),
        f"{prefix}_reach_generation_sd": (
            format_float(pstdev(reached_generations))
            if len(reached_generations) > 1
            else "0.000"
        ),
    }


def comb_tilt_mutation_sd(settings: DirectionSettings) -> float:
    if settings.comb_tilt_mutation_sd is None:
        return settings.mutation_sd

    return settings.comb_tilt_mutation_sd


def mean_summary(states: list[GenerationSummary], attribute: str) -> str:
    return format_float(mean(getattr(state, attribute) for state in states))


def format_float(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
