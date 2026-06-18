from __future__ import annotations

import argparse
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

EVENT_OUTPUT_FIELDNAMES = ["candidate", *PARAM_FIELDNAMES, *EVENT_FIELDNAMES]
TRAJECTORY_OUTPUT_FIELDNAMES = ["candidate", *TRAJECTORY_FIELDNAMES]
GROUP_OUTPUT_FIELDNAMES = ["candidate", *GROUP_SUMMARY_FIELDNAMES]
GENERATION_OUTPUT_FIELDNAMES = ["candidate", *GENERATION_SUMMARY_FIELDNAMES]


@dataclass(frozen=True)
class Candidate:
    name: str
    food_site_count: int
    food_site_width: float
    food_site_capacity: int
    food_value: float
    vertical_comb_benefit: float
    food_site_max_distance: float
    travel_cost_per_distance: float
    mutation_sd: float
    transposition_mutation_correlation: float


@dataclass(frozen=True)
class CandidateSeedResult:
    candidate: str
    seed: int
    event_row: dict[str, str]
    trajectory_rows: list[dict[str, str]]


CANDIDATES = {
    "high_success_8site": Candidate(
        name="high_success_8site",
        food_site_count=8,
        food_site_width=0.39,
        food_site_capacity=10,
        food_value=1.0,
        vertical_comb_benefit=0.41,
        food_site_max_distance=5.0,
        travel_cost_per_distance=0.035,
        mutation_sd=0.08,
        transposition_mutation_correlation=0.9,
    ),
    "repeated_7site": Candidate(
        name="repeated_7site",
        food_site_count=7,
        food_site_width=0.37,
        food_site_capacity=10,
        food_value=0.9,
        vertical_comb_benefit=0.44,
        food_site_max_distance=5.0,
        travel_cost_per_distance=0.035,
        mutation_sd=0.09,
        transposition_mutation_correlation=0.9,
    ),
}


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    candidates = selected_candidates(args.candidates)
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

    started = perf_counter()
    total_runs = len(candidates) * len(seeds)
    print(
        (
            "validating extensive Optuna candidates: "
            f"candidates={','.join(candidate.name for candidate in candidates)} "
            f"seeds={len(seeds)} total_runs={total_runs} "
            f"workers={args.max_workers} generations={base_settings.generations}"
        ),
        file=sys.stderr,
        flush=True,
    )

    results: list[CandidateSeedResult] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for candidate in candidates:
            settings = candidate_settings(base_settings, candidate)
            for seed in seeds:
                future = executor.submit(run_candidate_seed, candidate, settings, seed, thresholds)
                futures[future] = (candidate.name, seed)

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            results.append(result)
            event = result.event_row
            print(
                (
                    f"{completed}/{total_runs} candidate={result.candidate} "
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

    candidate_order = {candidate.name: index for index, candidate in enumerate(candidates)}
    results.sort(key=lambda result: (candidate_order[result.candidate], result.seed))
    event_rows = [result.event_row for result in results]
    trajectory_rows = [
        row
        for result in results
        for row in sorted(result.trajectory_rows, key=lambda item: int(item["generation"]))
    ]

    write_rows(outputs["events"], EVENT_OUTPUT_FIELDNAMES, event_rows)
    write_rows(outputs["trajectories"], TRAJECTORY_OUTPUT_FIELDNAMES, trajectory_rows)
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
    for path in outputs.values():
        print(f"wrote {relative(path)}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate selected extensive Optuna transition candidates on a "
            "held-out seed panel."
        )
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
        default=ROOT / "results" / "food_transition_v2_manual_validation",
        help="Prefix for event, trajectory, and summary CSV outputs.",
    )
    parser.add_argument(
        "--candidates",
        default="high_success_8site,repeated_7site",
        help=(
            "Comma-separated candidate names, or 'all'. Available names: "
            f"{','.join(CANDIDATES)}."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="200-299",
        help="Comma-separated seeds and inclusive ranges, e.g. 200-299,420.",
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
        help="Generation count for validation runs.",
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


def selected_candidates(raw: str) -> list[Candidate]:
    if raw.strip() == "all":
        return list(CANDIDATES.values())

    candidates = []
    for name in (item.strip() for item in raw.split(",")):
        if not name:
            continue
        try:
            candidates.append(CANDIDATES[name])
        except KeyError as exc:
            available = ", ".join(CANDIDATES)
            raise ValueError(f"unknown candidate {name!r}; choose from {available}") from exc
    if not candidates:
        raise ValueError("at least one candidate is required")
    return candidates


def heldout_seeds(seed_spec: str, exclude_spec: str) -> list[int]:
    seeds = parse_seed_spec(seed_spec)
    excluded = set(parse_seed_spec(exclude_spec))
    heldout = [seed for seed in seeds if seed not in excluded]
    if not heldout:
        raise ValueError("held-out seed panel is empty")
    return heldout


def output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "events": prefix.with_name(f"{prefix.name}_events.csv"),
        "trajectories": prefix.with_name(f"{prefix.name}_trajectories.csv"),
        "group_summary": prefix.with_name(f"{prefix.name}_group_summary.csv"),
        "generation_summary": prefix.with_name(f"{prefix.name}_generation_summary.csv"),
    }


def candidate_settings(
    base_settings: DirectionSettings,
    candidate: Candidate,
) -> DirectionSettings:
    return replace(
        base_settings,
        initial_comb_tilt=0.0,
        vertical_comb_modifier="linear",
        food_site_count=candidate.food_site_count,
        food_site_width=candidate.food_site_width,
        food_site_capacity=candidate.food_site_capacity,
        food_value=candidate.food_value,
        vertical_comb_benefit=candidate.vertical_comb_benefit,
        food_site_max_distance=candidate.food_site_max_distance,
        travel_cost_per_distance=candidate.travel_cost_per_distance,
        mutation_sd=candidate.mutation_sd,
        transposition_mutation_correlation=candidate.transposition_mutation_correlation,
    )


def run_candidate_seed(
    candidate: Candidate,
    settings: DirectionSettings,
    seed: int,
    thresholds: Thresholds,
) -> CandidateSeedResult:
    started = perf_counter()
    history = simulate(settings, seed=seed)
    event = summarize_events(
        seed=seed,
        history=history,
        thresholds=thresholds,
        elapsed_seconds=perf_counter() - started,
    )
    event_row = {
        "candidate": candidate.name,
        **candidate_param_values(candidate),
        **event,
    }
    trajectory_rows = [
        {
            "candidate": candidate.name,
            **trajectory_row(
                seed=seed,
                comparison_group=event["comparison_group"],
                outcome=event["outcome"],
                state=state,
            ),
        }
        for state in history
    ]
    return CandidateSeedResult(
        candidate=candidate.name,
        seed=seed,
        event_row=event_row,
        trajectory_rows=trajectory_rows,
    )


def candidate_param_values(candidate: Candidate) -> dict[str, str]:
    return {
        "food_site_count": str(candidate.food_site_count),
        "food_site_width": format_float(candidate.food_site_width),
        "food_site_capacity": str(candidate.food_site_capacity),
        "food_value": format_float(candidate.food_value),
        "vertical_comb_benefit": format_float(candidate.vertical_comb_benefit),
        "food_site_max_distance": format_float(candidate.food_site_max_distance),
        "travel_cost_per_distance": format_float(candidate.travel_cost_per_distance),
        "mutation_sd": format_float(candidate.mutation_sd),
        "transposition_mutation_correlation": format_float(
            candidate.transposition_mutation_correlation,
        ),
    }


def summarize_event_groups_by_candidate(
    rows: list[dict[str, str]],
    candidate_order: dict[str, int],
) -> list[dict[str, str]]:
    summary = []
    for candidate in sorted(candidate_order, key=candidate_order.get):
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        summary.append({"candidate": candidate, **event_group_row("all", "all", candidate_rows)})
        for group_kind, group_key in (
            ("comparison", "comparison_group"),
            ("outcome", "outcome"),
        ):
            for group in sorted({row[group_key] for row in candidate_rows}):
                group_rows = [row for row in candidate_rows if row[group_key] == group]
                summary.append(
                    {
                        "candidate": candidate,
                        **event_group_row(group_kind, group, group_rows),
                    }
                )
    return summary


def summarize_generation_groups_by_candidate(
    rows: list[dict[str, str]],
    candidate_order: dict[str, int],
) -> list[dict[str, str]]:
    summary = []
    for candidate in sorted(candidate_order, key=candidate_order.get):
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        for group_kind, group_key in (
            ("all", None),
            ("comparison", "comparison_group"),
            ("outcome", "outcome"),
        ):
            groups: dict[tuple[str, str], list[dict[str, str]]] = {}
            for row in candidate_rows:
                group = "all" if group_key is None else row[group_key]
                key = (group, row["generation"])
                groups.setdefault(key, []).append(row)
            for (group, generation), group_rows in sorted(
                groups.items(),
                key=lambda item: (item[0][0], int(item[0][1])),
            ):
                summary.append(
                    {
                        "candidate": candidate,
                        **generation_group_row(group_kind, group, generation, group_rows),
                    }
                )
    return summary


if __name__ == "__main__":
    main()
