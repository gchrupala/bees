from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate

SUMMARY_FIELDNAMES = [
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
    "reached_directional_fraction",
    "mean_directional_reach_generation",
    "directional_reach_generation_sd",
    "reached_vertical_fraction",
    "mean_vertical_reach_generation",
    "vertical_reach_generation_sd",
    "retained_vertical_fraction",
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
    "initial_comb_tilt",
    "vertical_comb_benefit",
    "vertical_comb_modifier",
    "seed",
    "generations",
    "comb_orientation_axial",
    "transposition_mutation_correlation",
    "gravity_reach_generation",
    "directional_reach_generation",
    "vertical_reach_generation",
    "collapsed",
    "recovered_from_collapse",
    "retained_vertical",
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


@dataclass(frozen=True)
class SeedMetrics:
    initial_comb_tilt: float
    vertical_comb_benefit: float
    vertical_comb_modifier: str
    seed: int
    generations: int
    comb_orientation_axial: bool
    transposition_mutation_correlation: float
    gravity_reach_generation: int | None
    directional_reach_generation: int | None
    vertical_reach_generation: int | None
    collapsed: bool
    recovered_from_collapse: bool
    retained_vertical: bool
    initial_success: float
    min_success: float
    min_success_generation: int
    final_bias: float
    final_attention: float
    final_sender_transposition: float
    final_receiver_transposition: float
    final_min_transposition: float
    final_transposition_gap: float
    final_comb_tilt: float
    final_comb_orientation_alignment: float
    final_search_limit: float
    final_success: float
    final_payoff: float
    success_delta: float
    elapsed_seconds: float


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    seeds = parse_ints(args.seeds)
    initial_comb_tilts = parse_floats(args.initial_comb_tilts)
    vertical_comb_benefits = parse_floats(args.vertical_comb_benefits)
    vertical_comb_modifiers = parse_strings(args.vertical_comb_modifiers)

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

        condition_count = (
            len(initial_comb_tilts)
            * len(vertical_comb_benefits)
            * len(vertical_comb_modifiers)
        )
        condition_index = 0
        for initial_comb_tilt in initial_comb_tilts:
            for vertical_comb_modifier in vertical_comb_modifiers:
                validate_vertical_comb_modifier(vertical_comb_modifier)
                for vertical_comb_benefit in vertical_comb_benefits:
                    condition_index += 1
                    condition_started = perf_counter()
                    settings = replace(
                        base_settings,
                        initial_comb_tilt=initial_comb_tilt,
                        vertical_comb_benefit=vertical_comb_benefit,
                        vertical_comb_modifier=vertical_comb_modifier,
                    )
                    print(
                        (
                            f"condition {condition_index}/{condition_count}: "
                            f"tilt={initial_comb_tilt:.3f}, "
                            f"modifier={vertical_comb_modifier}, "
                            f"benefit={vertical_comb_benefit:.3f}, "
                            f"generations={settings.generations}, "
                            f"seeds={len(seeds)}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

                    metrics = []
                    for seed_index, seed in enumerate(seeds, start=1):
                        seed_started = perf_counter()
                        history = simulate(settings, seed=seed)
                        seed_metrics = summarize_seed(
                            initial_comb_tilt=initial_comb_tilt,
                            vertical_comb_benefit=vertical_comb_benefit,
                            vertical_comb_modifier=vertical_comb_modifier,
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
                        seed_writer.writerow(seed_row(seed_metrics))
                        seed_file.flush()
                        print(
                            (
                                f"  seed {seed_index}/{len(seeds)} ({seed}) done: "
                                f"final_success={seed_metrics.final_success:.3f}, "
                                f"final_tilt={seed_metrics.final_comb_tilt:.3f}, "
                                f"final_min_transposition="
                                f"{seed_metrics.final_min_transposition:.3f}, "
                                f"collapsed={str(seed_metrics.collapsed).lower()}, "
                                f"elapsed={seed_metrics.elapsed_seconds:.1f}s"
                            ),
                            file=sys.stderr,
                            flush=True,
                        )

                    row = summarize_condition(
                        initial_comb_tilt=initial_comb_tilt,
                        vertical_comb_benefit=vertical_comb_benefit,
                        vertical_comb_modifier=vertical_comb_modifier,
                        settings=settings,
                        metrics=metrics,
                        elapsed_seconds=perf_counter() - condition_started,
                    )
                    summary_writer.writerow(row)
                    summary_file.flush()
                    print(
                        (
                            f"condition {condition_index}/{condition_count} summary: "
                            f"gravity={row['reached_gravity_fraction']}, "
                            f"vertical={row['retained_vertical_fraction']}, "
                            f"collapse={row['collapse_fraction']}, "
                            f"final_success={row['mean_final_success']}, "
                            f"elapsed={row['elapsed_seconds']}s"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

    print(f"wrote {args.summary_output}", file=sys.stderr, flush=True)
    print(f"wrote {args.seed_output}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run longer axial-orientation transition experiments from "
            "horizontal, tilted, and vertical comb starts."
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
        default=ROOT / "results" / "long_vertical_transition_summary.csv",
        help="Path for aggregate condition-level CSV output.",
    )
    parser.add_argument(
        "--seed-output",
        type=Path,
        default=ROOT / "results" / "long_vertical_transition_seeds.csv",
        help="Path for per-seed CSV output.",
    )
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105,106,107,108,109,110",
        help="Comma-separated random seeds to run for each condition.",
    )
    parser.add_argument(
        "--initial-comb-tilts",
        default="0.0,0.5,1.0",
        help="Comma-separated initial comb tilt values.",
    )
    parser.add_argument(
        "--vertical-comb-benefits",
        default="0.05,0.10,0.15,0.20,0.25",
        help="Comma-separated proportional vertical-comb advantage values.",
    )
    parser.add_argument(
        "--vertical-comb-modifiers",
        default="linear",
        help=(
            "Comma-separated vertical-comb modifier functions. Supported "
            "values: linear, threshold_0.8."
        ),
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
        "--vertical-threshold",
        type=float,
        default=0.80,
        help="Mean comb-tilt threshold for retained or reached verticality.",
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
        help=(
            "A collapsed seed is marked recovered if final mean success reaches "
            "at least this value."
        ),
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


def parse_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def validate_vertical_comb_modifier(modifier: str) -> None:
    if modifier not in {"linear", "threshold_0.8"}:
        raise ValueError(f"unknown vertical comb modifier: {modifier}")


def summarize_seed(
    initial_comb_tilt: float,
    vertical_comb_benefit: float,
    vertical_comb_modifier: str,
    seed: int,
    settings: DirectionSettings,
    history: list[GenerationSummary],
    gravity_threshold: float,
    directional_threshold: float,
    vertical_threshold: float,
    collapse_success_threshold: float,
    recovery_success_threshold: float,
    elapsed_seconds: float,
) -> SeedMetrics:
    initial = history[0]
    final = history[-1]
    min_success_state = min(history, key=lambda state: state.average_success_rate)
    collapsed = min_success_state.average_success_rate <= collapse_success_threshold
    recovered_from_collapse = (
        collapsed and final.average_success_rate >= recovery_success_threshold
    )

    return SeedMetrics(
        initial_comb_tilt=initial_comb_tilt,
        vertical_comb_benefit=vertical_comb_benefit,
        vertical_comb_modifier=vertical_comb_modifier,
        seed=seed,
        generations=settings.generations,
        comb_orientation_axial=settings.comb_orientation_axial,
        transposition_mutation_correlation=(
            settings.transposition_mutation_correlation
        ),
        gravity_reach_generation=first_gravity_reach_generation(
            history,
            gravity_threshold,
        ),
        directional_reach_generation=first_directional_reach_generation(
            history,
            directional_threshold,
        ),
        vertical_reach_generation=first_vertical_reach_generation(
            history,
            vertical_threshold,
        ),
        collapsed=collapsed,
        recovered_from_collapse=recovered_from_collapse,
        retained_vertical=final.average_comb_tilt >= vertical_threshold,
        initial_success=initial.average_success_rate,
        min_success=min_success_state.average_success_rate,
        min_success_generation=min_success_state.generation,
        final_bias=final.average_directional_bias,
        final_attention=final.average_receiver_attention,
        final_sender_transposition=final.average_sender_transposition,
        final_receiver_transposition=final.average_receiver_transposition,
        final_min_transposition=min(
            final.average_sender_transposition,
            final.average_receiver_transposition,
        ),
        final_transposition_gap=abs(
            final.average_sender_transposition
            - final.average_receiver_transposition,
        ),
        final_comb_tilt=final.average_comb_tilt,
        final_comb_orientation_alignment=final.comb_orientation_alignment,
        final_search_limit=final.average_search_limit,
        final_success=final.average_success_rate,
        final_payoff=final.average_payoff,
        success_delta=final.average_success_rate - initial.average_success_rate,
        elapsed_seconds=elapsed_seconds,
    )


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


def first_vertical_reach_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if state.average_comb_tilt >= threshold:
            return state.generation

    return None


def summarize_condition(
    initial_comb_tilt: float,
    vertical_comb_benefit: float,
    vertical_comb_modifier: str,
    settings: DirectionSettings,
    metrics: list[SeedMetrics],
    elapsed_seconds: float,
) -> dict[str, str]:
    row = {
        "initial_comb_tilt": format_float(initial_comb_tilt),
        "vertical_comb_benefit": format_float(vertical_comb_benefit),
        "vertical_comb_modifier": vertical_comb_modifier,
        "seeds": str(len(metrics)),
        "generations": str(settings.generations),
        "comb_orientation_axial": str(settings.comb_orientation_axial).lower(),
        "transposition_mutation_correlation": format_float(
            settings.transposition_mutation_correlation,
        ),
        "retained_vertical_fraction": format_float(
            fraction(metric.retained_vertical for metric in metrics),
        ),
        "collapse_fraction": format_float(
            fraction(metric.collapsed for metric in metrics),
        ),
        "recovered_from_collapse_fraction": format_float(
            fraction(metric.recovered_from_collapse for metric in metrics),
        ),
        "mean_initial_success": mean_summary(metrics, "initial_success"),
        "mean_min_success": mean_summary(metrics, "min_success"),
        "mean_min_success_generation": mean_summary(
            metrics,
            "min_success_generation",
        ),
        "mean_final_bias": mean_summary(metrics, "final_bias"),
        "mean_final_attention": mean_summary(metrics, "final_attention"),
        "mean_final_sender_transposition": mean_summary(
            metrics,
            "final_sender_transposition",
        ),
        "mean_final_receiver_transposition": mean_summary(
            metrics,
            "final_receiver_transposition",
        ),
        "mean_final_min_transposition": mean_summary(
            metrics,
            "final_min_transposition",
        ),
        "mean_final_transposition_gap": mean_summary(
            metrics,
            "final_transposition_gap",
        ),
        "mean_final_comb_tilt": mean_summary(metrics, "final_comb_tilt"),
        "mean_final_comb_orientation_alignment": mean_summary(
            metrics,
            "final_comb_orientation_alignment",
        ),
        "mean_final_search_limit": mean_summary(metrics, "final_search_limit"),
        "mean_final_success": mean_summary(metrics, "final_success"),
        "mean_final_payoff": mean_summary(metrics, "final_payoff"),
        "mean_success_delta": mean_summary(metrics, "success_delta"),
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
            "directional",
            [metric.directional_reach_generation for metric in metrics],
        )
    )
    row.update(
        reach_summary(
            "vertical",
            [metric.vertical_reach_generation for metric in metrics],
        )
    )
    return row


def seed_row(metric: SeedMetrics) -> dict[str, str]:
    return {
        "initial_comb_tilt": format_float(metric.initial_comb_tilt),
        "vertical_comb_benefit": format_float(metric.vertical_comb_benefit),
        "vertical_comb_modifier": metric.vertical_comb_modifier,
        "seed": str(metric.seed),
        "generations": str(metric.generations),
        "comb_orientation_axial": str(metric.comb_orientation_axial).lower(),
        "transposition_mutation_correlation": format_float(
            metric.transposition_mutation_correlation,
        ),
        "gravity_reach_generation": generation_value(
            metric.gravity_reach_generation,
        ),
        "directional_reach_generation": generation_value(
            metric.directional_reach_generation,
        ),
        "vertical_reach_generation": generation_value(
            metric.vertical_reach_generation,
        ),
        "collapsed": str(metric.collapsed).lower(),
        "recovered_from_collapse": str(metric.recovered_from_collapse).lower(),
        "retained_vertical": str(metric.retained_vertical).lower(),
        "initial_success": format_float(metric.initial_success),
        "min_success": format_float(metric.min_success),
        "min_success_generation": str(metric.min_success_generation),
        "final_bias": format_float(metric.final_bias),
        "final_attention": format_float(metric.final_attention),
        "final_sender_transposition": format_float(metric.final_sender_transposition),
        "final_receiver_transposition": format_float(
            metric.final_receiver_transposition,
        ),
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


def mean_summary(metrics: list[SeedMetrics], attribute: str) -> str:
    return format_float(mean(getattr(metric, attribute) for metric in metrics))


def fraction(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(items) / len(items)


def generation_value(generation: int | None) -> str:
    if generation is None:
        return "not_reached"

    return str(generation)


def format_float(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
