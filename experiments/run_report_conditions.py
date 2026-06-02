from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate

SEEDS = tuple(range(101, 111))
BIAS_THRESHOLD = 0.30

FOOD_DISTRIBUTIONS = {
    "hard": {"food_site_count": 1, "food_site_width": 0.08},
    "baseline": {"food_site_count": 2, "food_site_width": 0.20},
    "easy": {"food_site_count": 8, "food_site_width": 0.50},
}

COMB_TILTS = {
    "horizontal": {"comb_tilt": 0.0},
    "tilted": {"comb_tilt": 0.5},
    "vertical": {"comb_tilt": 1.0},
}

FIELDNAMES = [
    "experiment",
    "condition",
    "seeds",
    "reached_fraction",
    "mean_reach_generation",
    "final_bias",
    "final_attention",
    "final_sender_transposition",
    "final_receiver_transposition",
    "final_search_limit",
    "final_success",
    "final_payoff",
]


def main() -> None:
    base_settings = load_settings()
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()

    for condition, overrides in FOOD_DISTRIBUTIONS.items():
        settings = replace(base_settings, comb_tilt=0.0, **overrides)
        writer.writerow(summarize("food_distribution", condition, settings))
        sys.stdout.flush()

    for condition, overrides in COMB_TILTS.items():
        settings = replace(base_settings, **overrides)
        writer.writerow(summarize("comb_tilt", condition, settings))
        sys.stdout.flush()


def load_settings() -> DirectionSettings:
    config_path = ROOT / "configs" / "horizontal_direction.json"
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    return DirectionSettings(**config)


def summarize(
    experiment: str,
    condition: str,
    settings: DirectionSettings,
) -> dict[str, str]:
    histories = [simulate(settings, seed=seed) for seed in SEEDS]
    finals = [history[-1] for history in histories]
    reach_generations = [first_reach_generation(history) for history in histories]
    reached = [generation is not None for generation in reach_generations]
    reached_generations = [
        generation for generation in reach_generations if generation is not None
    ]

    return {
        "experiment": experiment,
        "condition": condition,
        "seeds": str(len(SEEDS)),
        "reached_fraction": format_float(mean(reached)),
        "mean_reach_generation": (
            format_float(mean(reached_generations))
            if reached_generations
            else "not_reached"
        ),
        "final_bias": mean_summary(finals, "average_directional_bias"),
        "final_attention": mean_summary(finals, "average_receiver_attention"),
        "final_sender_transposition": mean_summary(
            finals,
            "average_sender_transposition",
        ),
        "final_receiver_transposition": mean_summary(
            finals,
            "average_receiver_transposition",
        ),
        "final_search_limit": mean_summary(finals, "average_search_limit"),
        "final_success": mean_summary(finals, "average_success_rate"),
        "final_payoff": mean_summary(finals, "average_payoff"),
    }


def first_reach_generation(history: Iterable[GenerationSummary]) -> int | None:
    for state in history:
        if state.average_directional_bias >= BIAS_THRESHOLD:
            return state.generation
    return None


def mean_summary(states: list[GenerationSummary], attribute: str) -> str:
    return format_float(mean(getattr(state, attribute) for state in states))


def format_float(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
