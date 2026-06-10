from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, simulate
from run_long_vertical_transition import (
    SeedMetrics,
    format_float,
    generation_value,
    parse_floats,
    parse_ints,
    reach_summary,
    summarize_seed,
)


@dataclass(frozen=True)
class FoodCondition:
    name: str
    food_site_count: int
    food_site_width: float
    food_site_capacity: int
    food_value: float


FOOD_CONDITIONS = {
    "baseline": FoodCondition(
        name="baseline",
        food_site_count=2,
        food_site_width=0.20,
        food_site_capacity=6,
        food_value=1.0,
    ),
    "broad": FoodCondition(
        name="broad",
        food_site_count=2,
        food_site_width=0.35,
        food_site_capacity=6,
        food_value=1.0,
    ),
    "broad_high_capacity": FoodCondition(
        name="broad_high_capacity",
        food_site_count=2,
        food_site_width=0.35,
        food_site_capacity=12,
        food_value=1.0,
    ),
    "broad_rich": FoodCondition(
        name="broad_rich",
        food_site_count=2,
        food_site_width=0.35,
        food_site_capacity=6,
        food_value=1.5,
    ),
    "broad_rich_high_capacity": FoodCondition(
        name="broad_rich_high_capacity",
        food_site_count=2,
        food_site_width=0.35,
        food_site_capacity=12,
        food_value=1.5,
    ),
    "moderate_dense": FoodCondition(
        name="moderate_dense",
        food_site_count=4,
        food_site_width=0.30,
        food_site_capacity=10,
        food_value=1.0,
    ),
}


SUMMARY_FIELDNAMES = [
    "condition",
    "food_site_count",
    "food_site_width",
    "food_site_capacity",
    "food_value",
    "initial_comb_tilt",
    "vertical_comb_benefit",
    "vertical_comb_modifier",
    "seeds",
    "generations",
    "comb_orientation_axial",
    "transposition_mutation_correlation",
    "reached_gravity_fraction",
    "mean_gravity_reach_generation",
    "gravity_reach_generation_sd",
    "final_gravity_fraction",
    "reached_vertical_fraction",
    "mean_vertical_reach_generation",
    "vertical_reach_generation_sd",
    "retained_vertical_fraction",
    "stable_vertical_gravity_fraction",
    "collapse_fraction",
    "recovered_from_collapse_fraction",
    "mean_initial_success",
    "mean_min_success",
    "mean_min_success_generation",
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

SEED_FIELDNAMES = [
    "condition",
    "food_site_count",
    "food_site_width",
    "food_site_capacity",
    "food_value",
    "initial_comb_tilt",
    "vertical_comb_benefit",
    "vertical_comb_modifier",
    "seed",
    "generations",
    "gravity_reach_generation",
    "vertical_reach_generation",
    "collapsed",
    "recovered_from_collapse",
    "retained_vertical",
    "final_gravity",
    "stable_vertical_gravity",
    "initial_success",
    "min_success",
    "min_success_generation",
    "final_bias",
    "final_attention",
    "final_sender_transposition",
    "final_receiver_transposition",
    "final_min_transposition",
    "final_transposition_gap",
    "final_comb_tilt",
    "final_comb_orientation_alignment",
    "final_search_limit",
    "final_success",
    "final_payoff",
    "success_delta",
    "elapsed_seconds",
]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    seeds = parse_ints(args.seeds)
    conditions = select_conditions(args.conditions)
    benefits = parse_floats(args.vertical_comb_benefits)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.seed_output.parent.mkdir(parents=True, exist_ok=True)

    with (
        args.summary_output.open("w", newline="") as summary_file,
        args.seed_output.open("w", newline="") as seed_file,
    ):
        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=SUMMARY_FIELDNAMES,
            lineterminator="\n",
        )
        seed_writer = csv.DictWriter(
            seed_file,
            fieldnames=SEED_FIELDNAMES,
            lineterminator="\n",
        )
        summary_writer.writeheader()
        seed_writer.writeheader()
        summary_file.flush()
        seed_file.flush()

        total_conditions = len(conditions) * len(benefits)
        condition_index = 0
        for condition in conditions:
            for vertical_comb_benefit in benefits:
                condition_index += 1
                started = perf_counter()
                settings = replace(
                    base_settings,
                    initial_comb_tilt=0.0,
                    vertical_comb_benefit=vertical_comb_benefit,
                    vertical_comb_modifier=args.vertical_comb_modifier,
                    food_site_count=condition.food_site_count,
                    food_site_width=condition.food_site_width,
                    food_site_capacity=condition.food_site_capacity,
                    food_value=condition.food_value,
                )
                print(
                    (
                        f"condition {condition_index}/{total_conditions}: "
                        f"{condition.name}, alpha={vertical_comb_benefit:.3f}, "
                        f"generations={settings.generations}, seeds={len(seeds)}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

                metrics = []
                for seed in seeds:
                    seed_started = perf_counter()
                    history = simulate(settings, seed=seed)
                    seed_metrics = summarize_seed(
                        initial_comb_tilt=0.0,
                        vertical_comb_benefit=vertical_comb_benefit,
                        vertical_comb_modifier=args.vertical_comb_modifier,
                        seed=seed,
                        settings=settings,
                        history=history,
                        gravity_threshold=args.gravity_threshold,
                        directional_threshold=args.directional_threshold,
                        vertical_threshold=args.vertical_threshold,
                        collapse_success_threshold=args.collapse_success_threshold,
                        recovery_success_threshold=args.recovery_success_threshold,
                        elapsed_seconds=perf_counter() - seed_started,
                    )
                    metrics.append(seed_metrics)
                    seed_writer.writerow(
                        seed_row(
                            condition=condition,
                            metric=seed_metrics,
                            gravity_threshold=args.gravity_threshold,
                            vertical_threshold=args.vertical_threshold,
                        )
                    )
                    seed_file.flush()

                row = summarize_condition(
                    condition=condition,
                    vertical_comb_benefit=vertical_comb_benefit,
                    vertical_comb_modifier=args.vertical_comb_modifier,
                    settings=settings,
                    metrics=metrics,
                    gravity_threshold=args.gravity_threshold,
                    vertical_threshold=args.vertical_threshold,
                    elapsed_seconds=perf_counter() - started,
                )
                summary_writer.writerow(row)
                summary_file.flush()
                print(
                    (
                        f"  summary: stable={row['stable_vertical_gravity_fraction']}, "
                        f"gravity={row['reached_gravity_fraction']}, "
                        f"vertical={row['retained_vertical_fraction']}, "
                        f"success={row['mean_final_success']}, "
                        f"tilt={row['mean_final_comb_tilt']}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    print(f"wrote {args.summary_output}", file=sys.stderr, flush=True)
    print(f"wrote {args.seed_output}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run horizontal-start food-distribution probes for the "
            "sun-to-gravity communication transition."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "long_vertical_transition.json",
        help="Path to the long-transition model config.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "food_transition_summary.csv",
        help="Path for aggregate condition-level CSV output.",
    )
    parser.add_argument(
        "--seed-output",
        type=Path,
        default=ROOT / "results" / "food_transition_seeds.csv",
        help="Path for per-seed CSV output.",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(FOOD_CONDITIONS),
        help="Comma-separated condition names to run.",
    )
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105,106,107,108,109,110",
        help="Comma-separated random seeds to run for each condition.",
    )
    parser.add_argument(
        "--vertical-comb-benefits",
        default="0.25",
        help="Comma-separated proportional vertical-comb advantage values.",
    )
    parser.add_argument(
        "--vertical-comb-modifier",
        default="linear",
        choices=("linear", "threshold_0.8"),
        help="Vertical-comb modifier function.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override the generation count from the config.",
    )
    parser.add_argument(
        "--gravity-threshold",
        type=float,
        default=0.50,
        help="Sender and receiver transposition threshold for gravity code.",
    )
    parser.add_argument(
        "--directional-threshold",
        type=float,
        default=0.30,
        help="Directional-bias threshold used for compatibility metrics.",
    )
    parser.add_argument(
        "--vertical-threshold",
        type=float,
        default=0.80,
        help="Mean comb-tilt threshold for vertical reach and retention.",
    )
    parser.add_argument(
        "--collapse-success-threshold",
        type=float,
        default=0.02,
        help="A seed is marked collapsed if mean success falls at or below this.",
    )
    parser.add_argument(
        "--recovery-success-threshold",
        type=float,
        default=0.10,
        help="A collapsed seed is marked recovered above this final success.",
    )
    return parser.parse_args()


def load_settings(config_path: Path) -> DirectionSettings:
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    return DirectionSettings(**config)


def select_conditions(raw: str) -> list[FoodCondition]:
    conditions = []
    for name in (item.strip() for item in raw.split(",")):
        if not name:
            continue
        if name not in FOOD_CONDITIONS:
            known = ", ".join(FOOD_CONDITIONS)
            raise ValueError(f"unknown food condition {name!r}; known: {known}")
        conditions.append(FOOD_CONDITIONS[name])
    return conditions


def summarize_condition(
    condition: FoodCondition,
    vertical_comb_benefit: float,
    vertical_comb_modifier: str,
    settings: DirectionSettings,
    metrics: list[SeedMetrics],
    gravity_threshold: float,
    vertical_threshold: float,
    elapsed_seconds: float,
) -> dict[str, str]:
    row = {
        "condition": condition.name,
        "food_site_count": str(condition.food_site_count),
        "food_site_width": format_float(condition.food_site_width),
        "food_site_capacity": str(condition.food_site_capacity),
        "food_value": format_float(condition.food_value),
        "initial_comb_tilt": "0.000",
        "vertical_comb_benefit": format_float(vertical_comb_benefit),
        "vertical_comb_modifier": vertical_comb_modifier,
        "seeds": str(len(metrics)),
        "generations": str(settings.generations),
        "comb_orientation_axial": str(settings.comb_orientation_axial).lower(),
        "transposition_mutation_correlation": format_float(
            settings.transposition_mutation_correlation,
        ),
        "final_gravity_fraction": format_float(
            fraction(
                is_final_gravity(metric, gravity_threshold) for metric in metrics
            )
        ),
        "retained_vertical_fraction": format_float(
            fraction(metric.retained_vertical for metric in metrics)
        ),
        "stable_vertical_gravity_fraction": format_float(
            fraction(
                is_stable_vertical_gravity(metric, gravity_threshold, vertical_threshold)
                for metric in metrics
            )
        ),
        "collapse_fraction": format_float(
            fraction(metric.collapsed for metric in metrics)
        ),
        "recovered_from_collapse_fraction": format_float(
            fraction(metric.recovered_from_collapse for metric in metrics)
        ),
        "mean_initial_success": mean_metric(metrics, "initial_success"),
        "mean_min_success": mean_metric(metrics, "min_success"),
        "mean_min_success_generation": mean_metric(metrics, "min_success_generation"),
        "mean_final_bias": mean_metric(metrics, "final_bias"),
        "mean_final_attention": mean_metric(metrics, "final_attention"),
        "mean_final_sender_transposition": mean_metric(
            metrics,
            "final_sender_transposition",
        ),
        "mean_final_receiver_transposition": mean_metric(
            metrics,
            "final_receiver_transposition",
        ),
        "mean_final_min_transposition": mean_metric(
            metrics,
            "final_min_transposition",
        ),
        "mean_final_transposition_gap": mean_metric(
            metrics,
            "final_transposition_gap",
        ),
        "mean_final_comb_tilt": mean_metric(metrics, "final_comb_tilt"),
        "mean_final_comb_orientation_alignment": mean_metric(
            metrics,
            "final_comb_orientation_alignment",
        ),
        "mean_final_search_limit": mean_metric(metrics, "final_search_limit"),
        "mean_final_success": mean_metric(metrics, "final_success"),
        "mean_final_payoff": mean_metric(metrics, "final_payoff"),
        "mean_success_delta": mean_metric(metrics, "success_delta"),
        "elapsed_seconds": format_float(elapsed_seconds),
    }
    row.update(
        reach_summary(
            "gravity",
            [metric.gravity_reach_generation for metric in metrics],
        )
    )
    row.update(
        reach_summary(
            "vertical",
            [metric.vertical_reach_generation for metric in metrics],
        )
    )
    return row


def seed_row(
    condition: FoodCondition,
    metric: SeedMetrics,
    gravity_threshold: float,
    vertical_threshold: float,
) -> dict[str, str]:
    final_gravity = is_final_gravity(metric, gravity_threshold)
    stable = is_stable_vertical_gravity(metric, gravity_threshold, vertical_threshold)
    return {
        "condition": condition.name,
        "food_site_count": str(condition.food_site_count),
        "food_site_width": format_float(condition.food_site_width),
        "food_site_capacity": str(condition.food_site_capacity),
        "food_value": format_float(condition.food_value),
        "initial_comb_tilt": "0.000",
        "vertical_comb_benefit": format_float(metric.vertical_comb_benefit),
        "vertical_comb_modifier": metric.vertical_comb_modifier,
        "seed": str(metric.seed),
        "generations": str(metric.generations),
        "gravity_reach_generation": generation_value(metric.gravity_reach_generation),
        "vertical_reach_generation": generation_value(metric.vertical_reach_generation),
        "collapsed": str(metric.collapsed).lower(),
        "recovered_from_collapse": str(metric.recovered_from_collapse).lower(),
        "retained_vertical": str(metric.retained_vertical).lower(),
        "final_gravity": str(final_gravity).lower(),
        "stable_vertical_gravity": str(stable).lower(),
        "initial_success": format_float(metric.initial_success),
        "min_success": format_float(metric.min_success),
        "min_success_generation": str(metric.min_success_generation),
        "final_bias": format_float(metric.final_bias),
        "final_attention": format_float(metric.final_attention),
        "final_sender_transposition": format_float(metric.final_sender_transposition),
        "final_receiver_transposition": format_float(metric.final_receiver_transposition),
        "final_min_transposition": format_float(metric.final_min_transposition),
        "final_transposition_gap": format_float(metric.final_transposition_gap),
        "final_comb_tilt": format_float(metric.final_comb_tilt),
        "final_comb_orientation_alignment": format_float(
            metric.final_comb_orientation_alignment,
        ),
        "final_search_limit": format_float(metric.final_search_limit),
        "final_success": format_float(metric.final_success),
        "final_payoff": format_float(metric.final_payoff),
        "success_delta": format_float(metric.success_delta),
        "elapsed_seconds": format_float(metric.elapsed_seconds),
    }


def is_final_gravity(metric: SeedMetrics, gravity_threshold: float) -> bool:
    return metric.final_min_transposition >= gravity_threshold


def is_stable_vertical_gravity(
    metric: SeedMetrics,
    gravity_threshold: float,
    vertical_threshold: float,
) -> bool:
    return (
        metric.final_comb_tilt >= vertical_threshold
        and metric.final_min_transposition >= gravity_threshold
    )


def mean_metric(metrics: list[SeedMetrics], attribute: str) -> str:
    return format_float(mean(getattr(metric, attribute) for metric in metrics))


def fraction(values) -> float:
    items = list(values)
    return sum(items) / len(items)


if __name__ == "__main__":
    main()
