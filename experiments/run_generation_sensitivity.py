from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_optuna_best_trajectories import (
    EVENT_FIELDNAMES,
    GENERATION_SUMMARY_FIELDNAMES,
    GROUP_SUMMARY_FIELDNAMES,
    TRAJECTORY_FIELDNAMES,
    Thresholds,
    event_group_row,
    generation_group_row,
    load_settings,
)
from evolutionary_interaction import slug_value
from run_food_transition_oat_sensitivity import (
    BASELINE_VALUES,
    PARAM_FIELDNAMES,
    PointSeedResult,
    heldout_seeds,
    load_baseline_values,
    point_param_values,
    run_point_seed,
    write_partial_result,
    write_rows,
)
from bees.model import DirectionSettings


RESULTS = ROOT / "results"
DEFAULT_PREFIX = RESULTS / "food_transition_v2_low_regime_generation_sensitivity"
DEFAULT_GENERATIONS = (240, 480, 960)
POINT_FIELDNAMES = ["point", "parameter", "parameter_value", "is_baseline"]
EXPERIMENT_FIELDNAMES = ["generations", *PARAM_FIELDNAMES]
POINT_OUTPUT_FIELDNAMES = [*POINT_FIELDNAMES, *EXPERIMENT_FIELDNAMES]
JOB_OUTPUT_FIELDNAMES = ["job_index", *POINT_FIELDNAMES, *EXPERIMENT_FIELDNAMES, "seed"]
EVENT_OUTPUT_FIELDNAMES = [*POINT_FIELDNAMES, *EXPERIMENT_FIELDNAMES, *EVENT_FIELDNAMES]
TRAJECTORY_OUTPUT_FIELDNAMES = [
    *POINT_FIELDNAMES,
    "generations",
    *TRAJECTORY_FIELDNAMES,
]
GROUP_OUTPUT_FIELDNAMES = [
    *POINT_FIELDNAMES,
    *EXPERIMENT_FIELDNAMES,
    *GROUP_SUMMARY_FIELDNAMES,
]
GENERATION_OUTPUT_FIELDNAMES = [
    *POINT_FIELDNAMES,
    *EXPERIMENT_FIELDNAMES,
    *GENERATION_SUMMARY_FIELDNAMES,
]


@dataclass(frozen=True)
class GenerationPoint:
    point: str
    parameter: str
    parameter_value: str
    is_baseline: bool
    generations: int
    values: dict[str, int | float]


@dataclass(frozen=True)
class PointSeedJob:
    index: int
    point: GenerationPoint
    seed: int


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    baseline_values = load_baseline_values(args)
    points = build_generation_points(
        generations=parse_generations(args.generations),
        baseline_values=baseline_values,
        vertical_comb_benefit=args.vertical_comb_benefit,
        mutation_sd=args.mutation_sd,
        transposition_mutation_correlation=args.transposition_mutation_correlation,
    )
    seeds = heldout_seeds(args.seeds, args.exclude_seeds)
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    jobs = build_jobs(points, seeds)
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

    write_rows(outputs["points"], POINT_OUTPUT_FIELDNAMES, map(point_row, points))
    write_rows(outputs["jobs"], JOB_OUTPUT_FIELDNAMES, job_rows(jobs))
    print(f"wrote {relative(outputs['points'])}", file=sys.stderr, flush=True)
    print(f"wrote {relative(outputs['jobs'])}", file=sys.stderr, flush=True)
    print(
        (
            "running low-regime generation sensitivity: "
            f"points={len(points)} seeds={len(seeds)} total_runs={len(jobs)} "
            f"workers={args.max_workers} generations={args.generations} "
            f"alpha={args.vertical_comb_benefit:.3f} "
            f"mutation_sd={args.mutation_sd:.3f} "
            f"rho={args.transposition_mutation_correlation:.3f}"
        ),
        file=sys.stderr,
        flush=True,
    )

    results = run_jobs(
        jobs=jobs,
        base_settings=base_settings,
        thresholds=thresholds,
        outputs=outputs,
        max_workers=args.max_workers,
    )
    write_summaries(points, results, outputs)
    for name in ("events", "trajectories", "group_summary", "generation_summary"):
        print(f"wrote {relative(outputs[name])}", file=sys.stderr, flush=True)

    if args.commit or args.push:
        commit_results(args.commit_message, outputs)
    if args.push:
        run(["git", "push"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe generation-count sensitivity in the low-benefit, low-mutation, "
            "low non-zero sender-receiver correlation regime."
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
        default=DEFAULT_PREFIX,
        help="Prefix for generated point, job, event, trajectory, and summary CSVs.",
    )
    parser.add_argument(
        "--baseline-points",
        type=Path,
        default=ROOT / "results" / "food_transition_v2_validation_points.csv",
        help="Candidate point CSV used to select the v2 ecology baseline.",
    )
    parser.add_argument(
        "--baseline-group-summary",
        type=Path,
        default=ROOT / "results" / "food_transition_v2_validation_group_summary.csv",
        help="Candidate group-summary CSV used to select the v2 ecology baseline.",
    )
    parser.add_argument(
        "--generations",
        default=",".join(str(value) for value in DEFAULT_GENERATIONS),
        help="Comma-separated generation counts to run.",
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
        "--max-seeds",
        type=int,
        default=None,
        help="Limit held-out seeds for local smoke tests.",
    )
    parser.add_argument("--vertical-comb-benefit", type=float, default=0.10)
    parser.add_argument("--mutation-sd", type=float, default=0.045)
    parser.add_argument("--transposition-mutation-correlation", type=float, default=0.30)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum worker processes.",
    )
    parser.add_argument("--gravity-threshold", type=float, default=0.50)
    parser.add_argument("--vertical-threshold", type=float, default=0.80)
    parser.add_argument("--partial-tilt-threshold", type=float, default=0.50)
    parser.add_argument("--partial-transposition-threshold", type=float, default=0.25)
    parser.add_argument("--collapse-success-threshold", type=float, default=0.02)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit generated result CSVs after a successful run.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add low-regime generation sensitivity results",
        help="Commit message used with --commit or --push.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the result commit after a successful run.",
    )
    return parser.parse_args()


def default_max_workers() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))


def parse_generations(raw: str) -> list[int]:
    generations = []
    for part in (item.strip() for item in raw.split(",")):
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"generation counts must be positive: {value}")
        generations.append(value)
    if not generations:
        raise ValueError("generation panel is empty")
    return generations


def build_generation_points(
    generations: list[int],
    baseline_values: dict[str, int | float] | None = None,
    vertical_comb_benefit: float = 0.10,
    mutation_sd: float = 0.045,
    transposition_mutation_correlation: float = 0.30,
) -> list[GenerationPoint]:
    values = dict(BASELINE_VALUES if baseline_values is None else baseline_values)
    values.update(
        {
            "vertical_comb_benefit": vertical_comb_benefit,
            "mutation_sd": mutation_sd,
            "transposition_mutation_correlation": (
                transposition_mutation_correlation
            ),
        }
    )
    points = []
    for generation_count in generations:
        points.append(
            GenerationPoint(
                point=(
                    f"generations_{generation_count}"
                    f"_alpha_{slug_value(vertical_comb_benefit)}"
                    f"_mut_{slug_value(mutation_sd)}"
                    f"_rho_{slug_value(transposition_mutation_correlation)}"
                ),
                parameter="generations",
                parameter_value=str(generation_count),
                is_baseline=False,
                generations=generation_count,
                values=dict(values),
            )
        )
    return points


def build_jobs(points: list[GenerationPoint], seeds: list[int]) -> list[PointSeedJob]:
    return [
        PointSeedJob(index=index, point=point, seed=seed)
        for index, (point, seed) in enumerate(
            (point, seed) for point in points for seed in seeds
        )
    ]


def run_jobs(
    jobs: list[PointSeedJob],
    base_settings: DirectionSettings,
    thresholds: Thresholds,
    outputs: dict[str, Path],
    max_workers: int,
) -> list[PointSeedResult]:
    started = perf_counter()
    results = []
    with (
        outputs["events"].open("w", newline="") as event_handle,
        outputs["trajectories"].open("w", newline="") as trajectory_handle,
        ProcessPoolExecutor(max_workers=max_workers) as executor,
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

        futures = {
            executor.submit(
                run_generation_seed,
                job.point,
                point_settings(base_settings, job.point),
                job.seed,
                thresholds,
            ): job
            for job in jobs
        }
        completed = 0
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            completed += 1
            results.append(result)
            event = result.event_row
            write_partial_result(event_writer, trajectory_writer, result)
            event_handle.flush()
            trajectory_handle.flush()
            print(
                (
                    f"{completed}/{len(jobs)} global_job={job.index} "
                    f"generations={job.point.generations} seed={result.seed} "
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
    return results


def run_generation_seed(
    point: GenerationPoint,
    settings: DirectionSettings,
    seed: int,
    thresholds: Thresholds,
) -> PointSeedResult:
    result = run_point_seed(point, settings, seed, thresholds)
    generation_context = {"generations": str(point.generations)}
    return PointSeedResult(
        point=result.point,
        seed=result.seed,
        event_row={**result.event_row, **generation_context},
        trajectory_rows=[
            {**row, **generation_context}
            for row in result.trajectory_rows
        ],
    )


def point_settings(
    base_settings: DirectionSettings,
    point: GenerationPoint,
) -> DirectionSettings:
    return replace(
        base_settings,
        generations=point.generations,
        initial_comb_tilt=0.0,
        vertical_comb_modifier="linear",
        **point.values,
    )


def write_summaries(
    points: list[GenerationPoint],
    results: list[PointSeedResult],
    outputs: dict[str, Path],
) -> None:
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
        summarize_generation_groups_by_point(
            trajectory_rows,
            point_order,
            point_lookup,
        ),
    )


def summarize_event_groups_by_point(
    rows: list[dict[str, str]],
    point_order: dict[str, int],
    point_lookup: dict[str, GenerationPoint],
) -> list[dict[str, str]]:
    summary = []
    for point_name in sorted(point_order, key=point_order.get):
        point_rows = [row for row in rows if row["point"] == point_name]
        point = point_lookup[point_name]
        context = point_context(point)
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
    point_lookup: dict[str, GenerationPoint],
) -> list[dict[str, str]]:
    summary = []
    for point_name in sorted(point_order, key=point_order.get):
        point_rows = [row for row in rows if row["point"] == point_name]
        context = point_context(point_lookup[point_name])
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


def output_paths(prefix: Path = DEFAULT_PREFIX) -> dict[str, Path]:
    return {
        "points": prefix.with_name(f"{prefix.name}_points.csv"),
        "jobs": prefix.with_name(f"{prefix.name}_jobs.csv"),
        "events": prefix.with_name(f"{prefix.name}_events.csv"),
        "trajectories": prefix.with_name(f"{prefix.name}_trajectories.csv"),
        "group_summary": prefix.with_name(f"{prefix.name}_group_summary.csv"),
        "generation_summary": prefix.with_name(f"{prefix.name}_generation_summary.csv"),
    }


def point_row(point: GenerationPoint) -> dict[str, str]:
    return point_context(point)


def point_context(point: GenerationPoint) -> dict[str, str]:
    return {
        **point_metadata(point),
        "generations": str(point.generations),
        **point_param_values(point),
    }


def point_metadata(point: GenerationPoint) -> dict[str, str]:
    return {
        "point": point.point,
        "parameter": point.parameter,
        "parameter_value": point.parameter_value,
        "is_baseline": str(point.is_baseline).lower(),
    }


def job_rows(jobs: list[PointSeedJob]) -> list[dict[str, str]]:
    return [
        {
            "job_index": str(job.index),
            **point_context(job.point),
            "seed": str(job.seed),
        }
        for job in jobs
    ]


def commit_results(commit_message: str, outputs: dict[str, Path]) -> None:
    paths = [str(path.relative_to(ROOT)) for path in outputs.values()]
    run(["git", "add", *paths])
    run(["git", "commit", "-m", commit_message])


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
