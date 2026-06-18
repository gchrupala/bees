from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, simulate

from analyze_optuna_best_trajectories import (
    EVENT_FIELDNAMES,
    GENERATION_SUMMARY_FIELDNAMES,
    GROUP_SUMMARY_FIELDNAMES,
    TRAJECTORY_FIELDNAMES,
    Thresholds,
    event_group_row,
    format_float,
    generation_group_row,
    load_settings,
    parse_seed_spec,
    relative,
    summarize_events,
    trajectory_row,
    write_rows,
)


PARAM_FIELDNAMES = [
    "food_site_count",
    "food_site_width",
    "food_site_capacity",
    "food_value",
    "vertical_comb_benefit",
    "food_site_max_distance",
    "travel_cost_per_distance",
    "mutation_sd",
    "transposition_mutation_correlation",
]
POINT_FIELDNAMES = ["point", "parameter", "parameter_value", "is_baseline"]

POINT_OUTPUT_FIELDNAMES = [*POINT_FIELDNAMES, *PARAM_FIELDNAMES]
EVENT_OUTPUT_FIELDNAMES = [*POINT_FIELDNAMES, *PARAM_FIELDNAMES, *EVENT_FIELDNAMES]
TRAJECTORY_OUTPUT_FIELDNAMES = [*POINT_FIELDNAMES, *TRAJECTORY_FIELDNAMES]
GROUP_OUTPUT_FIELDNAMES = [*POINT_FIELDNAMES, *PARAM_FIELDNAMES, *GROUP_SUMMARY_FIELDNAMES]
GENERATION_OUTPUT_FIELDNAMES = [
    *POINT_FIELDNAMES,
    *GENERATION_SUMMARY_FIELDNAMES,
]

BASELINE_VALUES: dict[str, int | float] = {
    "food_site_count": 7,
    "food_site_width": 0.37,
    "food_site_capacity": 10,
    "food_value": 0.9,
    "vertical_comb_benefit": 0.44,
    "food_site_max_distance": 5.0,
    "travel_cost_per_distance": 0.035,
    "mutation_sd": 0.09,
    "transposition_mutation_correlation": 0.9,
}

OAT_LADDERS: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("vertical_comb_benefit", (0.34, 0.40, 0.44, 0.48)),
    ("food_site_max_distance", (5.0, 6.0, 7.0)),
    ("transposition_mutation_correlation", (0.5, 0.7, 0.9)),
    ("mutation_sd", (0.06, 0.09, 0.12)),
    ("food_site_width", (0.30, 0.37, 0.40)),
    ("food_site_count", (5, 7, 8)),
    ("travel_cost_per_distance", (0.020, 0.035, 0.050)),
)

REFINEMENT_LADDERS: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("food_site_count", (4, 6, 9)),
    ("mutation_sd", (0.045, 0.075, 0.105, 0.135)),
    ("vertical_comb_benefit", (0.30, 0.37, 0.42, 0.46, 0.52)),
    ("transposition_mutation_correlation", (0.30, 0.60, 0.80, 1.00)),
)

SENSITIVITY_PANELS = {
    "coarse": OAT_LADDERS,
    "refinement": REFINEMENT_LADDERS,
}


@dataclass(frozen=True)
class SensitivityPoint:
    point: str
    parameter: str
    parameter_value: str
    is_baseline: bool
    values: dict[str, int | float]


@dataclass(frozen=True)
class PointSeedResult:
    point: str
    seed: int
    event_row: dict[str, str]
    trajectory_rows: list[dict[str, str]]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    baseline_values = load_baseline_values(args)
    points = build_points(panel_ladders(args.panel, baseline_values), baseline_values)
    seeds = heldout_seeds(args.seeds, args.exclude_seeds)
    thresholds = Thresholds(
        gravity=args.gravity_threshold,
        vertical=args.vertical_threshold,
        partial_tilt=args.partial_tilt_threshold,
        partial_transposition=args.partial_transposition_threshold,
        collapse_success=args.collapse_success_threshold,
    )
    outputs = output_paths(args.output_prefix or default_output_prefix(args.panel))
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    write_rows(
        outputs["points"],
        POINT_OUTPUT_FIELDNAMES,
        (point_row(point) for point in points),
    )
    print(f"wrote {relative(outputs['points'])}", file=sys.stderr, flush=True)

    started = perf_counter()
    total_runs = len(points) * len(seeds)
    print(
        (
            f"running {args.panel} one-parameter food-transition sensitivity: "
            f"points={len(points)} seeds={len(seeds)} total_runs={total_runs} "
            f"workers={args.max_workers} generations={base_settings.generations}"
        ),
        file=sys.stderr,
        flush=True,
    )

    results: list[PointSeedResult] = []
    with (
        outputs["events"].open("w", newline="") as event_handle,
        outputs["trajectories"].open("w", newline="") as trajectory_handle,
        ProcessPoolExecutor(max_workers=args.max_workers) as executor,
    ):
        event_writer = csv.DictWriter(
            event_handle,
            fieldnames=EVENT_OUTPUT_FIELDNAMES,
            lineterminator="\n",
        )
        trajectory_writer = csv.DictWriter(
            trajectory_handle,
            fieldnames=TRAJECTORY_OUTPUT_FIELDNAMES,
            lineterminator="\n",
        )
        event_writer.writeheader()
        trajectory_writer.writeheader()
        event_handle.flush()
        trajectory_handle.flush()
        print(
            (
                f"streaming event rows to {relative(outputs['events'])}; "
                f"trajectory rows to {relative(outputs['trajectories'])}"
            ),
            file=sys.stderr,
            flush=True,
        )

        futures = {}
        for point in points:
            settings = point_settings(base_settings, point)
            for seed in seeds:
                future = executor.submit(run_point_seed, point, settings, seed, thresholds)
                futures[future] = (point.point, seed)

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            results.append(result)
            event = result.event_row
            write_partial_result(event_writer, trajectory_writer, result)
            event_handle.flush()
            trajectory_handle.flush()
            print(
                (
                    f"{completed}/{total_runs} point={result.point} "
                    f"seed={result.seed} outcome={event['outcome']} "
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

    point_order = {point.point: index for index, point in enumerate(points)}
    point_lookup = {point.point: point for point in points}
    results.sort(key=lambda result: (point_order[result.point], result.seed))
    event_rows = [result.event_row for result in results]
    trajectory_rows = [
        row
        for result in results
        for row in sorted(result.trajectory_rows, key=lambda item: int(item["generation"]))
    ]

    write_rows(
        outputs["group_summary"],
        GROUP_OUTPUT_FIELDNAMES,
        summarize_event_groups_by_point(event_rows, point_order, point_lookup),
    )
    write_rows(
        outputs["generation_summary"],
        GENERATION_OUTPUT_FIELDNAMES,
        summarize_generation_groups_by_point(trajectory_rows, point_order, point_lookup),
    )
    for name, path in outputs.items():
        if name not in {"points", "events", "trajectories"}:
            print(f"wrote {relative(path)}", file=sys.stderr, flush=True)
    print(f"completed {relative(outputs['events'])}", file=sys.stderr, flush=True)
    print(
        f"completed {relative(outputs['trajectories'])}",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-parameter-at-a-time sensitivity panel around the "
            "validated v2 food-transition candidate."
        )
    )
    parser.add_argument(
        "--panel",
        choices=sorted(SENSITIVITY_PANELS),
        default="coarse",
        help="Sensitivity point set to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "long_vertical_transition.json",
        help="Path to the base long-transition model config.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Prefix for point, event, trajectory, and summary CSV outputs.",
    )
    parser.add_argument(
        "--baseline-points",
        type=Path,
        default=None,
        help="Candidate point CSV used to select a dynamic sensitivity baseline.",
    )
    parser.add_argument(
        "--baseline-group-summary",
        type=Path,
        default=None,
        help="Candidate group-summary CSV used to select the best baseline row.",
    )
    parser.add_argument(
        "--seeds",
        default="300-399",
        help="Comma-separated seeds and inclusive ranges, e.g. 300-399,420.",
    )
    parser.add_argument(
        "--exclude-seeds",
        default="",
        help="Seeds to remove from the validation panel.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=120,
        help="Generation count for sensitivity runs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum number of parallel worker processes.",
    )
    parser.add_argument("--gravity-threshold", type=float, default=0.50)
    parser.add_argument("--vertical-threshold", type=float, default=0.80)
    parser.add_argument("--partial-tilt-threshold", type=float, default=0.50)
    parser.add_argument("--partial-transposition-threshold", type=float, default=0.25)
    parser.add_argument("--collapse-success-threshold", type=float, default=0.02)
    return parser.parse_args()


def default_output_prefix(panel: str) -> Path:
    if panel == "coarse":
        return ROOT / "results" / "food_transition_v2_oat_sensitivity"
    return ROOT / f"results/food_transition_v2_sensitivity_{panel}"


def load_baseline_values(args: argparse.Namespace) -> dict[str, int | float]:
    if args.baseline_points is None and args.baseline_group_summary is None:
        return dict(BASELINE_VALUES)
    if args.baseline_points is None or args.baseline_group_summary is None:
        raise ValueError(
            "--baseline-points and --baseline-group-summary must be used together"
        )

    point_rows = {
        row["candidate"]: row
        for row in read_csv_rows(args.baseline_points)
    }
    summary_rows = [
        row
        for row in read_csv_rows(args.baseline_group_summary)
        if row["group_kind"] == "all" and row["group"] == "all"
    ]
    if not summary_rows:
        raise ValueError(f"no all-candidate rows in {args.baseline_group_summary}")
    best = min(summary_rows, key=baseline_sort_key)
    row = point_rows[best["candidate"]]
    return {
        name: parse_param_value(name, row[name])
        for name in PARAM_FIELDNAMES
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def baseline_sort_key(row: dict[str, str]) -> tuple[float, float, float, float, str]:
    return (
        -float(row["stable_fraction"]),
        float(row["collapse_fraction"]),
        -float(row["mean_final_min_transposition"]),
        -float(row["mean_final_success"]),
        row["candidate"],
    )


def parse_param_value(name: str, raw: str) -> int | float:
    if name in {"food_site_count", "food_site_capacity"}:
        return int(raw)
    return float(raw)


def build_points(
    ladders: tuple[tuple[str, tuple[int | float, ...]], ...],
    baseline_values: dict[str, int | float],
) -> list[SensitivityPoint]:
    baseline = SensitivityPoint(
        point="baseline",
        parameter="baseline",
        parameter_value="baseline",
        is_baseline=True,
        values=dict(baseline_values),
    )
    points = [baseline]
    seen = {values_key(baseline.values)}

    for parameter, values in ladders:
        for value in values:
            point_values = {**baseline_values, parameter: value}
            key = values_key(point_values)
            if key in seen:
                continue
            seen.add(key)
            points.append(
                SensitivityPoint(
                    point=f"{parameter}_{slug_value(value)}",
                    parameter=parameter,
                    parameter_value=parameter_value(value),
                    is_baseline=False,
                    values=point_values,
                )
            )
    return points


def panel_ladders(
    panel: str,
    baseline_values: dict[str, int | float],
) -> tuple[tuple[str, tuple[int | float, ...]], ...]:
    if panel == "coarse":
        return coarse_ladders(baseline_values)
    if panel == "refinement":
        return refinement_ladders(baseline_values)
    raise ValueError(f"unknown sensitivity panel: {panel}")


def coarse_ladders(
    baseline: dict[str, int | float],
) -> tuple[tuple[str, tuple[int | float, ...]], ...]:
    return (
        ("vertical_comb_benefit", float_ladder(baseline, "vertical_comb_benefit", (-0.08, -0.04, 0.04, 0.08), 0.10, 0.60)),
        ("food_site_max_distance", float_ladder(baseline, "food_site_max_distance", (-1.0, -0.5, 0.5, 1.0), 4.5, 9.0)),
        ("transposition_mutation_correlation", float_ladder(baseline, "transposition_mutation_correlation", (-0.2, -0.1, 0.1, 0.2), 0.0, 1.0)),
        ("mutation_sd", float_ladder(baseline, "mutation_sd", (-0.03, -0.01, 0.01, 0.03), 0.04, 0.14)),
        ("food_site_width", float_ladder(baseline, "food_site_width", (-0.07, -0.03, 0.03, 0.07), 0.18, 0.45)),
        ("food_site_count", int_ladder(baseline, "food_site_count", (-2, -1, 1, 2), 3, 10)),
        ("food_site_capacity", int_ladder(baseline, "food_site_capacity", (-4, -2, 2, 4), 4, 16)),
        ("travel_cost_per_distance", float_ladder(baseline, "travel_cost_per_distance", (-0.015, -0.005, 0.005, 0.015), 0.01, 0.06)),
    )


def refinement_ladders(
    baseline: dict[str, int | float],
) -> tuple[tuple[str, tuple[int | float, ...]], ...]:
    return (
        ("food_site_count", int_ladder(baseline, "food_site_count", (-3, -2, -1, 1, 2, 3), 3, 10)),
        ("mutation_sd", float_ladder(baseline, "mutation_sd", (-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04), 0.04, 0.14)),
        ("vertical_comb_benefit", float_ladder(baseline, "vertical_comb_benefit", (-0.12, -0.08, -0.04, 0.04, 0.08, 0.12), 0.10, 0.60)),
        ("transposition_mutation_correlation", float_ladder(baseline, "transposition_mutation_correlation", (-0.4, -0.2, -0.1, 0.1, 0.2, 0.4), 0.0, 1.0)),
    )


def float_ladder(
    baseline: dict[str, int | float],
    parameter: str,
    offsets: tuple[float, ...],
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    base = float(baseline[parameter])
    values = {
        round(min(max(base + offset, minimum), maximum), 3)
        for offset in offsets
    }
    values.add(round(base, 3))
    return tuple(sorted(values))


def int_ladder(
    baseline: dict[str, int | float],
    parameter: str,
    offsets: tuple[int, ...],
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    base = int(baseline[parameter])
    values = {min(max(base + offset, minimum), maximum) for offset in offsets}
    values.add(base)
    return tuple(sorted(values))


def values_key(values: dict[str, int | float]) -> tuple[int | float, ...]:
    return tuple(values[name] for name in PARAM_FIELDNAMES)


def output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "points": prefix.with_name(f"{prefix.name}_points.csv"),
        "events": prefix.with_name(f"{prefix.name}_events.csv"),
        "trajectories": prefix.with_name(f"{prefix.name}_trajectories.csv"),
        "group_summary": prefix.with_name(f"{prefix.name}_group_summary.csv"),
        "generation_summary": prefix.with_name(f"{prefix.name}_generation_summary.csv"),
    }


def heldout_seeds(seed_spec: str, exclude_spec: str) -> list[int]:
    seeds = parse_seed_spec(seed_spec)
    excluded = set(parse_seed_spec(exclude_spec)) if exclude_spec.strip() else set()
    heldout = [seed for seed in seeds if seed not in excluded]
    if not heldout:
        raise ValueError("held-out seed panel is empty")
    return heldout


def point_settings(
    base_settings: DirectionSettings,
    point: SensitivityPoint,
) -> DirectionSettings:
    return replace(
        base_settings,
        initial_comb_tilt=0.0,
        vertical_comb_modifier="linear",
        **point.values,
    )


def run_point_seed(
    point: SensitivityPoint,
    settings: DirectionSettings,
    seed: int,
    thresholds: Thresholds,
) -> PointSeedResult:
    started = perf_counter()
    history = simulate(settings, seed=seed)
    event = summarize_events(
        seed=seed,
        history=history,
        thresholds=thresholds,
        elapsed_seconds=perf_counter() - started,
    )
    metadata = point_metadata(point)
    event_row = {
        **metadata,
        **point_param_values(point),
        **event,
    }
    trajectory_rows = [
        {
            **metadata,
            **trajectory_row(
                seed=seed,
                comparison_group=event["comparison_group"],
                outcome=event["outcome"],
                state=state,
            ),
        }
        for state in history
    ]
    return PointSeedResult(
        point=point.point,
        seed=seed,
        event_row=event_row,
        trajectory_rows=trajectory_rows,
    )


def write_partial_result(
    event_writer: csv.DictWriter,
    trajectory_writer: csv.DictWriter,
    result: PointSeedResult,
) -> None:
    event_writer.writerow(result.event_row)
    for row in sorted(result.trajectory_rows, key=lambda item: int(item["generation"])):
        trajectory_writer.writerow(row)


def summarize_event_groups_by_point(
    rows: list[dict[str, str]],
    point_order: dict[str, int],
    point_lookup: dict[str, SensitivityPoint],
) -> list[dict[str, str]]:
    summary = []
    for point_name in sorted(point_order, key=point_order.get):
        point_rows = [row for row in rows if row["point"] == point_name]
        point = point_lookup[point_name]
        context = {**point_metadata(point), **point_param_values(point)}
        summary.append({**context, **event_group_row("all", "all", point_rows)})
        for group_kind, group_key in (
            ("comparison", "comparison_group"),
            ("outcome", "outcome"),
        ):
            for group in sorted({row[group_key] for row in point_rows}):
                group_rows = [row for row in point_rows if row[group_key] == group]
                summary.append(
                    {
                        **context,
                        **event_group_row(group_kind, group, group_rows),
                    }
                )
    return summary


def summarize_generation_groups_by_point(
    rows: list[dict[str, str]],
    point_order: dict[str, int],
    point_lookup: dict[str, SensitivityPoint],
) -> list[dict[str, str]]:
    summary = []
    for point_name in sorted(point_order, key=point_order.get):
        point_rows = [row for row in rows if row["point"] == point_name]
        context = point_metadata(point_lookup[point_name])
        for group_kind, group_key in (
            ("all", None),
            ("comparison", "comparison_group"),
            ("outcome", "outcome"),
        ):
            groups: dict[tuple[str, str], list[dict[str, str]]] = {}
            for row in point_rows:
                group = "all" if group_key is None else row[group_key]
                key = (group, row["generation"])
                groups.setdefault(key, []).append(row)
            for (group, generation), group_rows in sorted(
                groups.items(),
                key=lambda item: (item[0][0], int(item[0][1])),
            ):
                summary.append(
                    {
                        **context,
                        **generation_group_row(
                            group_kind,
                            group,
                            generation,
                            group_rows,
                        ),
                    }
                )
    return summary


def point_row(point: SensitivityPoint) -> dict[str, str]:
    return {**point_metadata(point), **point_param_values(point)}


def point_metadata(point: SensitivityPoint) -> dict[str, str]:
    return {
        "point": point.point,
        "parameter": point.parameter,
        "parameter_value": point.parameter_value,
        "is_baseline": str(point.is_baseline).lower(),
    }


def point_param_values(point: SensitivityPoint) -> dict[str, str]:
    return {name: parameter_value(point.values[name]) for name in PARAM_FIELDNAMES}


def parameter_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return format_float(value)


def slug_value(value: int | float) -> str:
    return parameter_value(value).replace(".", "p")


if __name__ == "__main__":
    main()
