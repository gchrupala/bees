from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate


TRAJECTORY_FIELDNAMES = [
    "seed",
    "comparison_group",
    "outcome",
    "generation",
    "average_directional_bias",
    "average_receiver_attention",
    "average_sender_transposition",
    "average_receiver_transposition",
    "average_min_transposition",
    "average_transposition_gap",
    "average_comb_tilt",
    "average_comb_orientation",
    "comb_orientation_alignment",
    "average_search_limit",
    "average_success",
    "average_payoff",
]

EVENT_FIELDNAMES = [
    "seed",
    "comparison_group",
    "outcome",
    "stable_vertical_gravity",
    "reached_gravity",
    "retained_vertical",
    "collapsed",
    "gravity_reach_generation",
    "vertical_reach_generation",
    "partial_tilt_generation",
    "partial_transposition_generation",
    "max_comb_tilt",
    "max_comb_tilt_generation",
    "max_min_transposition",
    "max_min_transposition_generation",
    "min_success",
    "min_success_generation",
    "final_directional_bias",
    "final_attention",
    "final_sender_transposition",
    "final_receiver_transposition",
    "final_min_transposition",
    "final_transposition_gap",
    "final_comb_tilt",
    "final_success",
    "final_payoff",
    "elapsed_seconds",
]

GROUP_SUMMARY_FIELDNAMES = [
    "group_kind",
    "group",
    "runs",
    "seeds",
    "stable_fraction",
    "reached_gravity_fraction",
    "retained_vertical_fraction",
    "collapse_fraction",
    "mean_gravity_reach_generation",
    "mean_vertical_reach_generation",
    "mean_partial_tilt_generation",
    "mean_partial_transposition_generation",
    "mean_max_comb_tilt",
    "mean_max_min_transposition",
    "mean_min_success",
    "mean_final_directional_bias",
    "mean_final_attention",
    "mean_final_min_transposition",
    "mean_final_transposition_gap",
    "mean_final_comb_tilt",
    "mean_final_success",
    "mean_final_payoff",
]

GENERATION_SUMMARY_FIELDNAMES = [
    "group_kind",
    "group",
    "generation",
    "runs",
    "mean_directional_bias",
    "mean_receiver_attention",
    "mean_sender_transposition",
    "mean_receiver_transposition",
    "mean_min_transposition",
    "mean_transposition_gap",
    "mean_comb_tilt",
    "mean_success",
    "mean_payoff",
]


@dataclass(frozen=True)
class Thresholds:
    gravity: float
    vertical: float
    partial_tilt: float
    partial_transposition: float
    collapse_success: float


@dataclass(frozen=True)
class SeedResult:
    seed: int
    event_row: dict[str, str]
    trajectory_rows: list[dict[str, str]]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    settings = replace(
        base_settings,
        generations=args.generations or base_settings.generations,
        initial_comb_tilt=0.0,
        vertical_comb_modifier="linear",
        food_site_count=args.food_site_count,
        food_site_width=args.food_site_width,
        food_site_capacity=args.food_site_capacity,
        food_value=1.0,
        food_site_max_distance=args.food_site_max_distance,
        vertical_comb_benefit=args.vertical_comb_benefit,
        travel_cost_per_distance=args.travel_cost_per_distance,
    )
    seeds = parse_seed_spec(args.seeds)
    thresholds = Thresholds(
        gravity=args.gravity_threshold,
        vertical=args.vertical_threshold,
        partial_tilt=args.partial_tilt_threshold,
        partial_transposition=args.partial_transposition_threshold,
        collapse_success=args.collapse_success_threshold,
    )

    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    args.event_output.parent.mkdir(parents=True, exist_ok=True)
    args.group_summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.generation_summary_output.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    print(
        (
            f"running Optuna-best trajectory panel: seeds={len(seeds)} "
            f"workers={args.max_workers} n={settings.food_site_count} "
            f"width={settings.food_site_width:.3f} "
            f"cap={settings.food_site_capacity} "
            f"alpha={settings.vertical_comb_benefit:.3f} "
            f"max_distance={settings.food_site_max_distance:.3f} "
            f"travel_cost={settings.travel_cost_per_distance:.3f}"
        ),
        file=sys.stderr,
        flush=True,
    )

    results: list[SeedResult] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_seed, seed, settings, thresholds): seed
            for seed in seeds
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            results.append(result)
            event = result.event_row
            print(
                (
                    f"{completed}/{len(seeds)} seed={result.seed} "
                    f"outcome={event['outcome']} "
                    f"gravity={event['gravity_reach_generation']} "
                    f"vertical={event['vertical_reach_generation']} "
                    f"tilt={event['final_comb_tilt']} "
                    f"m={event['final_min_transposition']} "
                    f"success={event['final_success']} "
                    f"elapsed={perf_counter() - started:.1f}s"
                ),
                file=sys.stderr,
                flush=True,
            )

    results.sort(key=lambda result: result.seed)
    event_rows = [result.event_row for result in results]
    trajectory_rows = [
        row
        for result in results
        for row in sorted(
            result.trajectory_rows,
            key=lambda item: int(item["generation"]),
        )
    ]
    write_rows(args.event_output, EVENT_FIELDNAMES, event_rows)
    write_rows(args.trajectory_output, TRAJECTORY_FIELDNAMES, trajectory_rows)
    write_rows(
        args.group_summary_output,
        GROUP_SUMMARY_FIELDNAMES,
        summarize_event_groups(event_rows),
    )
    write_rows(
        args.generation_summary_output,
        GENERATION_SUMMARY_FIELDNAMES,
        summarize_generation_groups(trajectory_rows),
    )
    print(f"wrote {relative(args.event_output)}", file=sys.stderr, flush=True)
    print(f"wrote {relative(args.trajectory_output)}", file=sys.stderr, flush=True)
    print(f"wrote {relative(args.group_summary_output)}", file=sys.stderr, flush=True)
    print(
        f"wrote {relative(args.generation_summary_output)}",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare successful and unsuccessful trajectories in the best "
            "multi-seed Optuna food-transition pocket."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "long_vertical_transition.json",
        help="Path to the base long-transition model config.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=ROOT / "results" / "food_transition_optuna_best_trajectories.csv",
        help="Path for generation-level trajectory CSV output.",
    )
    parser.add_argument(
        "--event-output",
        type=Path,
        default=ROOT / "results" / "food_transition_optuna_best_events.csv",
        help="Path for per-seed event and final-state CSV output.",
    )
    parser.add_argument(
        "--group-summary-output",
        type=Path,
        default=ROOT / "results" / "food_transition_optuna_best_group_summary.csv",
        help="Path for outcome-level event summaries.",
    )
    parser.add_argument(
        "--generation-summary-output",
        type=Path,
        default=ROOT
        / "results"
        / "food_transition_optuna_best_generation_summary.csv",
        help="Path for per-generation outcome means.",
    )
    parser.add_argument(
        "--seeds",
        default="96-120",
        help="Comma-separated seeds and inclusive ranges, e.g. 96-120,130.",
    )
    parser.add_argument("--food-site-count", type=int, default=6)
    parser.add_argument("--food-site-width", type=float, default=0.32)
    parser.add_argument("--food-site-capacity", type=int, default=9)
    parser.add_argument("--vertical-comb-benefit", type=float, default=0.34)
    parser.add_argument("--food-site-max-distance", type=float, default=6.0)
    parser.add_argument("--travel-cost-per-distance", type=float, default=0.03)
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override the generation count from the config.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel worker processes.",
    )
    parser.add_argument("--gravity-threshold", type=float, default=0.50)
    parser.add_argument("--vertical-threshold", type=float, default=0.80)
    parser.add_argument("--partial-tilt-threshold", type=float, default=0.50)
    parser.add_argument("--partial-transposition-threshold", type=float, default=0.25)
    parser.add_argument("--collapse-success-threshold", type=float, default=0.02)
    return parser.parse_args()


def run_seed(
    seed: int,
    settings: DirectionSettings,
    thresholds: Thresholds,
) -> SeedResult:
    started = perf_counter()
    history = simulate(settings, seed=seed)
    event = summarize_events(
        seed=seed,
        history=history,
        thresholds=thresholds,
        elapsed_seconds=perf_counter() - started,
    )
    trajectory = [
        trajectory_row(
            seed=seed,
            comparison_group=event["comparison_group"],
            outcome=event["outcome"],
            state=state,
        )
        for state in history
    ]
    return SeedResult(seed=seed, event_row=event, trajectory_rows=trajectory)


def summarize_events(
    seed: int,
    history: list[GenerationSummary],
    thresholds: Thresholds,
    elapsed_seconds: float,
) -> dict[str, str]:
    final = history[-1]
    final_min_transposition = min_transposition(final)
    final_gap = transposition_gap(final)
    stable = (
        final.average_comb_tilt >= thresholds.vertical
        and final_min_transposition >= thresholds.gravity
    )
    reached_gravity = first_gravity_reach_generation(history, thresholds.gravity)
    reached_vertical = first_comb_tilt_generation(history, thresholds.vertical)
    partial_tilt = first_comb_tilt_generation(history, thresholds.partial_tilt)
    partial_transposition = first_min_transposition_generation(
        history,
        thresholds.partial_transposition,
    )
    min_success_state = min(history, key=lambda state: state.average_success_rate)
    max_tilt_state = max(history, key=lambda state: state.average_comb_tilt)
    max_transposition_state = max(history, key=min_transposition)
    retained_vertical = final.average_comb_tilt >= thresholds.vertical
    collapsed = min_success_state.average_success_rate <= thresholds.collapse_success
    outcome = classify_outcome(
        stable=stable,
        reached_gravity=reached_gravity is not None,
        retained_vertical=retained_vertical,
        final_tilt=final.average_comb_tilt,
        final_min_transposition=final_min_transposition,
        thresholds=thresholds,
    )
    comparison_group = "successful" if stable else "unsuccessful"

    return {
        "seed": str(seed),
        "comparison_group": comparison_group,
        "outcome": outcome,
        "stable_vertical_gravity": bool_value(stable),
        "reached_gravity": bool_value(reached_gravity is not None),
        "retained_vertical": bool_value(retained_vertical),
        "collapsed": bool_value(collapsed),
        "gravity_reach_generation": generation_value(reached_gravity),
        "vertical_reach_generation": generation_value(reached_vertical),
        "partial_tilt_generation": generation_value(partial_tilt),
        "partial_transposition_generation": generation_value(partial_transposition),
        "max_comb_tilt": format_float(max_tilt_state.average_comb_tilt),
        "max_comb_tilt_generation": str(max_tilt_state.generation),
        "max_min_transposition": format_float(min_transposition(max_transposition_state)),
        "max_min_transposition_generation": str(max_transposition_state.generation),
        "min_success": format_float(min_success_state.average_success_rate),
        "min_success_generation": str(min_success_state.generation),
        "final_directional_bias": format_float(final.average_directional_bias),
        "final_attention": format_float(final.average_receiver_attention),
        "final_sender_transposition": format_float(
            final.average_sender_transposition,
        ),
        "final_receiver_transposition": format_float(
            final.average_receiver_transposition,
        ),
        "final_min_transposition": format_float(final_min_transposition),
        "final_transposition_gap": format_float(final_gap),
        "final_comb_tilt": format_float(final.average_comb_tilt),
        "final_success": format_float(final.average_success_rate),
        "final_payoff": format_float(final.average_payoff),
        "elapsed_seconds": format_float(elapsed_seconds),
    }


def trajectory_row(
    seed: int,
    comparison_group: str,
    outcome: str,
    state: GenerationSummary,
) -> dict[str, str]:
    return {
        "seed": str(seed),
        "comparison_group": comparison_group,
        "outcome": outcome,
        "generation": str(state.generation),
        "average_directional_bias": format_float(state.average_directional_bias),
        "average_receiver_attention": format_float(state.average_receiver_attention),
        "average_sender_transposition": format_float(
            state.average_sender_transposition,
        ),
        "average_receiver_transposition": format_float(
            state.average_receiver_transposition,
        ),
        "average_min_transposition": format_float(min_transposition(state)),
        "average_transposition_gap": format_float(transposition_gap(state)),
        "average_comb_tilt": format_float(state.average_comb_tilt),
        "average_comb_orientation": format_float(state.average_comb_orientation),
        "comb_orientation_alignment": format_float(state.comb_orientation_alignment),
        "average_search_limit": format_float(state.average_search_limit),
        "average_success": format_float(state.average_success_rate),
        "average_payoff": format_float(state.average_payoff),
    }


def summarize_event_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = []
    for group_kind, group_key in (
        ("comparison", "comparison_group"),
        ("outcome", "outcome"),
    ):
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(row[group_key], []).append(row)
        for group, group_rows in sorted(groups.items()):
            summary.append(event_group_row(group_kind, group, group_rows))
    return summary


def event_group_row(
    group_kind: str,
    group: str,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    return {
        "group_kind": group_kind,
        "group": group,
        "runs": str(len(rows)),
        "seeds": ",".join(row["seed"] for row in rows),
        "stable_fraction": fraction_field(rows, "stable_vertical_gravity"),
        "reached_gravity_fraction": fraction_field(rows, "reached_gravity"),
        "retained_vertical_fraction": fraction_field(rows, "retained_vertical"),
        "collapse_fraction": fraction_field(rows, "collapsed"),
        "mean_gravity_reach_generation": mean_optional_field(
            rows,
            "gravity_reach_generation",
        ),
        "mean_vertical_reach_generation": mean_optional_field(
            rows,
            "vertical_reach_generation",
        ),
        "mean_partial_tilt_generation": mean_optional_field(
            rows,
            "partial_tilt_generation",
        ),
        "mean_partial_transposition_generation": mean_optional_field(
            rows,
            "partial_transposition_generation",
        ),
        "mean_max_comb_tilt": mean_float_field(rows, "max_comb_tilt"),
        "mean_max_min_transposition": mean_float_field(rows, "max_min_transposition"),
        "mean_min_success": mean_float_field(rows, "min_success"),
        "mean_final_directional_bias": mean_float_field(rows, "final_directional_bias"),
        "mean_final_attention": mean_float_field(rows, "final_attention"),
        "mean_final_min_transposition": mean_float_field(
            rows,
            "final_min_transposition",
        ),
        "mean_final_transposition_gap": mean_float_field(
            rows,
            "final_transposition_gap",
        ),
        "mean_final_comb_tilt": mean_float_field(rows, "final_comb_tilt"),
        "mean_final_success": mean_float_field(rows, "final_success"),
        "mean_final_payoff": mean_float_field(rows, "final_payoff"),
    }


def summarize_generation_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = []
    for group_kind, group_key in (
        ("comparison", "comparison_group"),
        ("outcome", "outcome"),
    ):
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            key = (row[group_key], row["generation"])
            groups.setdefault(key, []).append(row)
        for (group, generation), group_rows in sorted(
            groups.items(),
            key=lambda item: (item[0][0], int(item[0][1])),
        ):
            summary.append(generation_group_row(group_kind, group, generation, group_rows))
    return summary


def generation_group_row(
    group_kind: str,
    group: str,
    generation: str,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    return {
        "group_kind": group_kind,
        "group": group,
        "generation": generation,
        "runs": str(len(rows)),
        "mean_directional_bias": mean_float_field(rows, "average_directional_bias"),
        "mean_receiver_attention": mean_float_field(rows, "average_receiver_attention"),
        "mean_sender_transposition": mean_float_field(
            rows,
            "average_sender_transposition",
        ),
        "mean_receiver_transposition": mean_float_field(
            rows,
            "average_receiver_transposition",
        ),
        "mean_min_transposition": mean_float_field(rows, "average_min_transposition"),
        "mean_transposition_gap": mean_float_field(rows, "average_transposition_gap"),
        "mean_comb_tilt": mean_float_field(rows, "average_comb_tilt"),
        "mean_success": mean_float_field(rows, "average_success"),
        "mean_payoff": mean_float_field(rows, "average_payoff"),
    }


def classify_outcome(
    stable: bool,
    reached_gravity: bool,
    retained_vertical: bool,
    final_tilt: float,
    final_min_transposition: float,
    thresholds: Thresholds,
) -> str:
    if stable:
        return "success"
    if retained_vertical:
        return "vertical_without_gravity"
    if reached_gravity:
        return "gravity_without_vertical"
    if final_tilt >= thresholds.partial_tilt:
        return "partial_tilt"
    if final_min_transposition >= thresholds.partial_transposition:
        return "partial_transposition"
    return "flat"


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


def first_comb_tilt_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if state.average_comb_tilt >= threshold:
            return state.generation
    return None


def first_min_transposition_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if min_transposition(state) >= threshold:
            return state.generation
    return None


def load_settings(config_path: Path) -> DirectionSettings:
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    return DirectionSettings(**config)


def parse_seed_spec(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in (item.strip() for item in raw.split(",")):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"seed range must be increasing: {part}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, str]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def min_transposition(state: GenerationSummary) -> float:
    return min(
        state.average_sender_transposition,
        state.average_receiver_transposition,
    )


def transposition_gap(state: GenerationSummary) -> float:
    return abs(
        state.average_sender_transposition
        - state.average_receiver_transposition,
    )


def fraction_field(rows: list[dict[str, str]], field: str) -> str:
    return format_float(
        sum(row[field] == "true" for row in rows) / len(rows),
    )


def mean_float_field(rows: list[dict[str, str]], field: str) -> str:
    return format_float(mean(float(row[field]) for row in rows))


def mean_optional_field(rows: list[dict[str, str]], field: str) -> str:
    values = [float(row[field]) for row in rows if row[field] != "not_reached"]
    if not values:
        return "not_reached"
    return format_float(mean(values))


def generation_value(generation: int | None) -> str:
    if generation is None:
        return "not_reached"
    return str(generation)


def bool_value(value: bool) -> str:
    return str(value).lower()


def format_float(value: float) -> str:
    return f"{value:.3f}"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
