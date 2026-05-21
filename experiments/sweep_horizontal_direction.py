from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate

SWEEPS = {
    "colony_count": [20, 60, 120],
    "workers_per_colony": [20, 80, 160],
    "episodes_per_colony": [15, 50, 120],
    "recruits_per_episode": [2, 6, 12],
    "mutation_sd": [0.01, 0.04, 0.10],
    "stable_worker_sd": [0.0, 0.08, 0.20],
    "max_signal_concentration": [4.0, 14.0, 30.0],
    "dance_noise_sd": [0.0, 0.18, 0.50],
    "interpretation_noise_sd": [0.0, 0.12, 0.50],
    "success_angle": [0.15, 0.35, 0.75],
    "food_value": [0.5, 1.0, 2.0],
    "cue_cost": [0.0, 0.02, 0.08],
    "attention_cost": [0.0, 0.01, 0.05],
}

FIELDNAMES = [
    "parameter",
    "value",
    "seeds",
    "reached_fraction",
    "mean_reach_generation",
    "reach_generation_sd",
    "mean_bias_delta",
    "mean_attention_delta",
    "mean_success_delta",
    "mean_final_bias",
    "mean_final_attention",
    "mean_final_success",
    "elapsed_seconds",
]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    seeds = parse_ints(args.seeds)
    parameters = parse_parameters(args.parameters)

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    sys.stdout.flush()

    for parameter in parameters:
        for value in SWEEPS[parameter]:
            started = perf_counter()
            settings = replace(base_settings, **{parameter: value})
            row = summarize_setting(
                parameter=parameter,
                value=value,
                settings=settings,
                seeds=seeds,
                threshold=args.threshold,
            )
            row["elapsed_seconds"] = format_float(perf_counter() - started)
            writer.writerow(row)
            sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a one-parameter-at-a-time sweep for the horizontal direction model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "horizontal_direction.json",
        help="Path to the base model config.",
    )
    parser.add_argument(
        "--seeds",
        default="101,102,103",
        help="Comma-separated random seeds to average over.",
    )
    parser.add_argument(
        "--parameters",
        default=",".join(SWEEPS),
        help="Comma-separated parameter names to sweep.",
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


def parse_parameters(raw: str) -> list[str]:
    parameters = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(parameters) - set(SWEEPS))
    if unknown:
        known = ", ".join(SWEEPS)
        raise SystemExit(f"Unknown sweep parameter(s): {', '.join(unknown)}. Known: {known}")

    return parameters


def summarize_setting(
    parameter: str,
    value: Any,
    settings: DirectionSettings,
    seeds: Iterable[int],
    threshold: float,
) -> dict[str, str]:
    metrics = [run_metrics(settings, seed, threshold) for seed in seeds]
    reached = [metric["reached"] for metric in metrics]

    return {
        "parameter": parameter,
        "value": str(value),
        "seeds": str(len(metrics)),
        "reached_fraction": format_float(mean(reached)),
        "mean_reach_generation": format_float(
            mean(metric["reach_generation"] for metric in metrics)
        ),
        "reach_generation_sd": format_float(
            pstdev(metric["reach_generation"] for metric in metrics)
        ),
        "mean_bias_delta": format_float(
            mean(metric["bias_delta"] for metric in metrics)
        ),
        "mean_attention_delta": format_float(
            mean(metric["attention_delta"] for metric in metrics)
        ),
        "mean_success_delta": format_float(
            mean(metric["success_delta"] for metric in metrics)
        ),
        "mean_final_bias": format_float(
            mean(metric["final_bias"] for metric in metrics)
        ),
        "mean_final_attention": format_float(
            mean(metric["final_attention"] for metric in metrics)
        ),
        "mean_final_success": format_float(
            mean(metric["final_success"] for metric in metrics)
        ),
        "elapsed_seconds": "",
    }


def run_metrics(
    settings: DirectionSettings,
    seed: int,
    threshold: float,
) -> dict[str, float]:
    history = simulate(settings, seed=seed)
    initial = history[0]
    final = history[-1]
    reach_generation = first_reach_generation(history, threshold)

    if reach_generation is None:
        reach_generation = settings.generations + 1
        reached = 0.0
    else:
        reached = 1.0

    return {
        "reached": reached,
        "reach_generation": float(reach_generation),
        "bias_delta": final.average_directional_bias
        - initial.average_directional_bias,
        "attention_delta": final.average_receiver_attention
        - initial.average_receiver_attention,
        "success_delta": final.average_success_rate - initial.average_success_rate,
        "final_bias": final.average_directional_bias,
        "final_attention": final.average_receiver_attention,
        "final_success": final.average_success_rate,
    }


def first_reach_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if state.average_directional_bias >= threshold:
            return state.generation

    return None


def format_float(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
