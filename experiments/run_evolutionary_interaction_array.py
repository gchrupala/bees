from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_optuna_best_trajectories import Thresholds, load_settings
from evolutionary_interaction import (
    DEFAULT_PREFIX,
    DEFAULT_SHARD_DIR,
    MUTATION_SD_VALUES,
    TRANSPOSITION_MUTATION_CORRELATION_VALUES,
    VERTICAL_COMB_BENEFIT_VALUES,
    shard_paths,
    slug_value,
)
from run_food_transition_oat_sensitivity import (
    BASELINE_VALUES,
    EVENT_OUTPUT_FIELDNAMES,
    POINT_OUTPUT_FIELDNAMES,
    TRAJECTORY_OUTPUT_FIELDNAMES,
    SensitivityPoint,
    heldout_seeds,
    point_row,
    point_settings,
    run_point_seed,
    write_partial_result,
    write_rows,
)


@dataclass(frozen=True)
class PointSeedJob:
    index: int
    point: SensitivityPoint
    seed: int


def main() -> None:
    args = parse_args()
    task_index, task_count = task_coordinates(args)
    points = build_interaction_points(args.max_points)
    seeds = heldout_seeds(args.seeds, args.exclude_seeds)
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    jobs = build_jobs(points, seeds)
    selected_jobs = [
        job for job in jobs if job.index % task_count == task_index
    ]
    if not selected_jobs:
        raise SystemExit(
            f"task {task_index}/{task_count} has no jobs; total jobs={len(jobs)}"
        )

    base_settings = load_settings(args.config)
    if args.generations is not None:
        from dataclasses import replace

        base_settings = replace(base_settings, generations=args.generations)
    thresholds = Thresholds(
        gravity=args.gravity_threshold,
        vertical=args.vertical_threshold,
        partial_tilt=args.partial_tilt_threshold,
        partial_transposition=args.partial_transposition_threshold,
        collapse_success=args.collapse_success_threshold,
    )

    outputs = shard_paths(args.shard_dir, task_index)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if task_index == 0:
        points_output = args.output_prefix.with_name(
            f"{args.output_prefix.name}_points.csv"
        )
        points_output.parent.mkdir(parents=True, exist_ok=True)
        write_rows(
            points_output,
            POINT_OUTPUT_FIELDNAMES,
            (point_row(point) for point in points),
        )
        print(f"wrote {relative(points_output)}", file=sys.stderr, flush=True)

    print(
        (
            "running evolutionary interaction shard: "
            f"task_index={task_index} task_count={task_count} "
            f"points={len(points)} seeds={len(seeds)} "
            f"total_jobs={len(jobs)} selected_jobs={len(selected_jobs)} "
            f"workers={args.max_workers} generations={base_settings.generations}"
        ),
        file=sys.stderr,
        flush=True,
    )

    started = perf_counter()
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

        futures = {
            executor.submit(
                run_point_seed,
                job.point,
                point_settings(base_settings, job.point),
                job.seed,
                thresholds,
            ): job
            for job in selected_jobs
        }
        completed = 0
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            completed += 1
            event = result.event_row
            write_partial_result(event_writer, trajectory_writer, result)
            event_handle.flush()
            trajectory_handle.flush()
            print(
                (
                    f"{completed}/{len(selected_jobs)} global_job={job.index} "
                    f"point={result.point} seed={result.seed} "
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

    print(f"completed {relative(outputs['events'])}", file=sys.stderr, flush=True)
    print(f"completed {relative(outputs['trajectories'])}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Slurm-array shard of the evolutionary-parameter interaction "
            "panel under the favorable repeated_7site ecology."
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
        help="Canonical output prefix used for the shared points CSV.",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=DEFAULT_SHARD_DIR,
        help="Directory for per-array-task partial CSV outputs.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Slurm array task id. Defaults to SLURM_ARRAY_TASK_ID or 0.",
    )
    parser.add_argument(
        "--task-min",
        type=int,
        default=None,
        help="Slurm array minimum task id. Defaults to SLURM_ARRAY_TASK_MIN or 0.",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=None,
        help="Slurm array task count. Defaults to SLURM_ARRAY_TASK_COUNT or 1.",
    )
    parser.add_argument(
        "--seeds",
        default="96-195",
        help="Comma-separated seeds and inclusive ranges, e.g. 96-195,220.",
    )
    parser.add_argument(
        "--exclude-seeds",
        default="100-104",
        help="Seeds to remove from the validation panel.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=120,
        help="Generation count for interaction runs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum worker processes for this shard.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Limit parameter points for local smoke tests.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Limit held-out seeds for local smoke tests.",
    )
    parser.add_argument("--gravity-threshold", type=float, default=0.50)
    parser.add_argument("--vertical-threshold", type=float, default=0.80)
    parser.add_argument("--partial-tilt-threshold", type=float, default=0.50)
    parser.add_argument("--partial-transposition-threshold", type=float, default=0.25)
    parser.add_argument("--collapse-success-threshold", type=float, default=0.02)
    return parser.parse_args()


def default_max_workers() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))


def task_coordinates(args: argparse.Namespace) -> tuple[int, int]:
    task_id = env_int("SLURM_ARRAY_TASK_ID", args.task_id, 0)
    task_min = env_int("SLURM_ARRAY_TASK_MIN", args.task_min, 0)
    task_count = env_int("SLURM_ARRAY_TASK_COUNT", args.task_count, 1)
    task_index = task_id - task_min
    if task_count <= 0:
        raise SystemExit(f"task count must be positive, got {task_count}")
    if not 0 <= task_index < task_count:
        raise SystemExit(
            f"task id {task_id} with min {task_min} gives invalid index "
            f"{task_index} for count {task_count}"
        )
    return task_index, task_count


def env_int(name: str, argument: int | None, default: int) -> int:
    if argument is not None:
        return argument
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def build_interaction_points(max_points: int | None = None) -> list[SensitivityPoint]:
    points = [
        make_point(alpha, mutation_sd, correlation)
        for alpha in VERTICAL_COMB_BENEFIT_VALUES
        for mutation_sd in MUTATION_SD_VALUES
        for correlation in TRANSPOSITION_MUTATION_CORRELATION_VALUES
    ]
    if max_points is not None:
        return points[:max_points]
    return points


def make_point(
    vertical_comb_benefit: float,
    mutation_sd: float,
    transposition_mutation_correlation: float,
) -> SensitivityPoint:
    values = {
        **BASELINE_VALUES,
        "vertical_comb_benefit": vertical_comb_benefit,
        "mutation_sd": mutation_sd,
        "comb_tilt_mutation_sd": mutation_sd,
        "transposition_mutation_correlation": transposition_mutation_correlation,
    }
    is_baseline = all(values[name] == BASELINE_VALUES[name] for name in BASELINE_VALUES)
    point = (
        f"alpha_{slug_value(vertical_comb_benefit)}"
        f"_mut_{slug_value(mutation_sd)}"
        f"_rho_{slug_value(transposition_mutation_correlation)}"
    )
    return SensitivityPoint(
        point=point,
        parameter="evolutionary_interaction",
        parameter_value=point,
        is_baseline=is_baseline,
        values=values,
    )


def build_jobs(
    points: list[SensitivityPoint],
    seeds: list[int],
) -> list[PointSeedJob]:
    return [
        PointSeedJob(index=index, point=point, seed=seed)
        for index, (point, seed) in enumerate(
            (point, seed) for point in points for seed in seeds
        )
    ]


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
