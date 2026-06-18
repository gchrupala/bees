from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_optuna_best_trajectories import (  # noqa: E402
    Thresholds,
    load_settings,
    parse_seed_spec,
    relative,
    write_rows,
)
from validate_extensive_optuna_candidates import (  # noqa: E402
    EVENT_OUTPUT_FIELDNAMES,
    GENERATION_OUTPUT_FIELDNAMES,
    GROUP_OUTPUT_FIELDNAMES,
    PARAM_FIELDNAMES,
    TRAJECTORY_OUTPUT_FIELDNAMES,
    Candidate,
    candidate_param_values,
    candidate_settings,
    run_candidate_seed,
    summarize_event_groups_by_candidate,
    summarize_generation_groups_by_candidate,
)

POINT_FIELDNAMES = ["candidate", "source", *PARAM_FIELDNAMES]
POINT_OUTPUT_FIELDNAMES = POINT_FIELDNAMES


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        from dataclasses import replace

        base_settings = replace(base_settings, generations=args.generations)

    candidates = select_candidates(args)
    seeds = heldout_seeds(args.seeds, args.exclude_seeds)
    thresholds = Thresholds(
        gravity=args.gravity_threshold,
        vertical=args.vertical_threshold,
        partial_tilt=args.partial_tilt_threshold,
        partial_transposition=args.partial_transposition_threshold,
        collapse_success=args.collapse_success_threshold,
    )
    outputs = output_paths(args.output_prefix)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    write_rows(
        outputs["points"],
        POINT_OUTPUT_FIELDNAMES,
        (candidate_row(candidate, source) for candidate, source in candidates),
    )
    print(f"wrote {relative(outputs['points'])}", file=sys.stderr, flush=True)

    started = perf_counter()
    total_runs = len(candidates) * len(seeds)
    print(
        (
            f"running v2 candidate panel: candidates={len(candidates)} "
            f"seeds={len(seeds)} total_runs={total_runs} "
            f"workers={args.max_workers} generations={base_settings.generations}"
        ),
        file=sys.stderr,
        flush=True,
    )

    results = []
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

        futures = {}
        for candidate, _source in candidates:
            settings = candidate_settings(base_settings, candidate)
            for seed in seeds:
                future = executor.submit(
                    run_candidate_seed,
                    candidate,
                    settings,
                    seed,
                    thresholds,
                )
                futures[future] = (candidate.name, seed)

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            results.append(result)
            write_partial_result(event_writer, trajectory_writer, result)
            event_handle.flush()
            trajectory_handle.flush()
            event = result.event_row
            print(
                (
                    f"{completed}/{total_runs} candidate={result.candidate} "
                    f"seed={result.seed} outcome={event['outcome']} "
                    f"tilt={event['final_comb_tilt']} "
                    f"m={event['final_min_transposition']} "
                    f"success={event['final_success']} "
                    f"elapsed={perf_counter() - started:.1f}s"
                ),
                file=sys.stderr,
                flush=True,
            )

    candidate_order = {
        candidate.name: index for index, (candidate, _source) in enumerate(candidates)
    }
    results.sort(key=lambda result: (candidate_order[result.candidate], result.seed))
    event_rows = [result.event_row for result in results]
    trajectory_rows = [
        row
        for result in results
        for row in sorted(result.trajectory_rows, key=lambda item: int(item["generation"]))
    ]
    write_rows(
        outputs["group_summary"],
        GROUP_OUTPUT_FIELDNAMES,
        summarize_event_groups_by_candidate(event_rows, candidate_order),
    )
    write_rows(
        outputs["generation_summary"],
        GENERATION_OUTPUT_FIELDNAMES,
        summarize_generation_groups_by_candidate(trajectory_rows, candidate_order),
    )
    for name, path in outputs.items():
        if name not in {"points", "events", "trajectories"}:
            print(f"wrote {relative(path)}", file=sys.stderr, flush=True)
    print(f"completed {relative(outputs['events'])}", file=sys.stderr, flush=True)
    print(f"completed {relative(outputs['trajectories'])}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run v2 confirmation or validation panels for candidates selected "
            "from Optuna trials or an earlier candidate panel."
        )
    )
    parser.add_argument(
        "--source",
        choices=("trials", "panel"),
        default="trials",
        help="Select candidates from Optuna trials or a previous candidate panel.",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=ROOT / "results" / "food_transition_v2_optuna_trials.csv",
        help="Trial CSV used when --source=trials.",
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=ROOT / "results" / "food_transition_v2_confirmation_points.csv",
        help="Candidate point CSV used when --source=panel.",
    )
    parser.add_argument(
        "--group-summary",
        type=Path,
        default=ROOT / "results" / "food_transition_v2_confirmation_group_summary.csv",
        help="Candidate group-summary CSV used when --source=panel.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "results" / "food_transition_v2_confirmation",
        help="Prefix for point, event, trajectory, and summary CSV outputs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "long_vertical_transition.json",
        help="Path to the base long-transition model config.",
    )
    parser.add_argument(
        "--seeds",
        default="110-149",
        help="Comma-separated seeds and inclusive ranges.",
    )
    parser.add_argument(
        "--exclude-seeds",
        default="",
        help="Seeds to remove from the panel.",
    )
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--gravity-threshold", type=float, default=0.50)
    parser.add_argument("--vertical-threshold", type=float, default=0.80)
    parser.add_argument("--partial-tilt-threshold", type=float, default=0.50)
    parser.add_argument("--partial-transposition-threshold", type=float, default=0.25)
    parser.add_argument("--collapse-success-threshold", type=float, default=0.02)
    return parser.parse_args()


def select_candidates(args: argparse.Namespace) -> list[tuple[Candidate, str]]:
    if args.source == "trials":
        return select_from_trials(args.trials, args.max_candidates)
    return select_from_panel(args.points, args.group_summary, args.max_candidates)


def select_from_trials(
    path: Path,
    max_candidates: int,
) -> list[tuple[Candidate, str]]:
    rows = read_rows(path)
    completed = [row for row in rows if row.get("state") == "COMPLETE"]
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for row in sorted(completed, key=trial_sort_key):
        key = tuple(row[name] for name in PARAM_FIELDNAMES)
        unique.setdefault(key, row)
    selected = list(unique.values())[:max_candidates]
    return [
        (candidate_from_row(f"trial_{row['number']}", row), f"trial_{row['number']}")
        for row in selected
    ]


def select_from_panel(
    points_path: Path,
    group_summary_path: Path,
    max_candidates: int,
) -> list[tuple[Candidate, str]]:
    point_rows = {row["candidate"]: row for row in read_rows(points_path)}
    summary_rows = [
        row
        for row in read_rows(group_summary_path)
        if row["group_kind"] == "all" and row["group"] == "all"
    ]
    selected = sorted(summary_rows, key=panel_sort_key)[:max_candidates]
    return [
        (
            candidate_from_row(row["candidate"], point_rows[row["candidate"]]),
            row["candidate"],
        )
        for row in selected
    ]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def trial_sort_key(row: dict[str, str]) -> tuple[float, int, float, float, float, float, int]:
    seed_count = max(int(row.get("seed_count") or 0), 1)
    stable_fraction = int(row.get("stable_count") or 0) / seed_count
    collapse_count = int(row.get("collapse_count") or 0)
    return (
        -stable_fraction,
        collapse_count,
        -float_value(row, "mean_final_min_transposition"),
        -float_value(row, "mean_final_success"),
        -float_value(row, "mean_final_payoff"),
        -float_value(row, "value"),
        int(row["number"]),
    )


def panel_sort_key(row: dict[str, str]) -> tuple[float, float, float, float, str]:
    return (
        -float_value(row, "stable_fraction"),
        float_value(row, "collapse_fraction"),
        -float_value(row, "mean_final_min_transposition"),
        -float_value(row, "mean_final_success"),
        row["candidate"],
    )


def float_value(row: dict[str, str], name: str) -> float:
    raw = row.get(name, "")
    if raw in {"", "not_reached"}:
        return 0.0
    return float(raw)


def candidate_from_row(name: str, row: dict[str, str]) -> Candidate:
    return Candidate(
        name=name,
        food_site_count=int(row["food_site_count"]),
        food_site_width=float(row["food_site_width"]),
        food_site_capacity=int(row["food_site_capacity"]),
        food_value=float(row["food_value"]),
        vertical_comb_benefit=float(row["vertical_comb_benefit"]),
        food_site_max_distance=float(row["food_site_max_distance"]),
        travel_cost_per_distance=float(row["travel_cost_per_distance"]),
        mutation_sd=float(row["mutation_sd"]),
        transposition_mutation_correlation=float(
            row["transposition_mutation_correlation"]
        ),
    )


def heldout_seeds(seed_spec: str, exclude_spec: str) -> list[int]:
    seeds = parse_seed_spec(seed_spec)
    excluded = set(parse_seed_spec(exclude_spec)) if exclude_spec.strip() else set()
    heldout = [seed for seed in seeds if seed not in excluded]
    if not heldout:
        raise ValueError("seed panel is empty")
    return heldout


def output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "points": prefix.with_name(f"{prefix.name}_points.csv"),
        "events": prefix.with_name(f"{prefix.name}_events.csv"),
        "trajectories": prefix.with_name(f"{prefix.name}_trajectories.csv"),
        "group_summary": prefix.with_name(f"{prefix.name}_group_summary.csv"),
        "generation_summary": prefix.with_name(f"{prefix.name}_generation_summary.csv"),
    }


def candidate_row(candidate: Candidate, source: str) -> dict[str, str]:
    return {"candidate": candidate.name, "source": source, **candidate_param_values(candidate)}


def write_partial_result(
    event_writer: csv.DictWriter,
    trajectory_writer: csv.DictWriter,
    result,
) -> None:
    event_writer.writerow(result.event_row)
    trajectory_writer.writerows(
        sorted(result.trajectory_rows, key=lambda item: int(item["generation"]))
    )


if __name__ == "__main__":
    main()
