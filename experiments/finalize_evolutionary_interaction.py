from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evolutionary_interaction import (
    DEFAULT_PREFIX,
    DEFAULT_SHARD_DIR,
    JOB_FIELDNAMES,
    output_paths,
    shard_paths,
)
from run_evolutionary_interaction_array import (
    PointSeedJob,
    build_interaction_points,
    build_jobs,
)
from run_food_transition_oat_sensitivity import (
    EVENT_OUTPUT_FIELDNAMES,
    GENERATION_OUTPUT_FIELDNAMES,
    GROUP_OUTPUT_FIELDNAMES,
    POINT_OUTPUT_FIELDNAMES,
    TRAJECTORY_OUTPUT_FIELDNAMES,
    SensitivityPoint,
    heldout_seeds,
    point_metadata,
    point_param_values,
    point_row,
    summarize_event_groups_by_point,
    summarize_generation_groups_by_point,
    write_rows,
)


def main() -> None:
    args = parse_args()
    points = build_interaction_points(args.max_points)
    seeds = heldout_seeds(args.seeds, args.exclude_seeds)
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    jobs = build_jobs(points, seeds)
    expected_runs = len(jobs)
    task_count = args.task_count
    if args.wait_for_shards:
        wait_for_complete_shards(
            shard_dir=args.shard_dir,
            task_count=task_count,
            expected_runs=expected_runs,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_hours * 3600,
        )

    event_rows = read_shard_rows(args.shard_dir, task_count, "events")
    trajectory_rows = read_shard_rows(args.shard_dir, task_count, "trajectories")
    validate_events(event_rows, jobs)

    point_order = {point.point: index for index, point in enumerate(points)}
    point_lookup = {point.point: point for point in points}
    sorted_event_rows = sorted(
        event_rows,
        key=lambda row: (point_order[row["point"]], int(row["seed"])),
    )
    sorted_trajectory_rows = sorted(
        trajectory_rows,
        key=lambda row: (
            point_order[row["point"]],
            int(row["seed"]),
            int(row["generation"]),
        ),
    )
    outputs = output_paths(args.output_prefix)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(outputs["points"], POINT_OUTPUT_FIELDNAMES, (point_row(point) for point in points))
    write_rows(outputs["jobs"], job_fieldnames(), job_rows(jobs))
    write_rows(outputs["events"], EVENT_OUTPUT_FIELDNAMES, sorted_event_rows)
    write_rows(outputs["trajectories"], TRAJECTORY_OUTPUT_FIELDNAMES, sorted_trajectory_rows)
    write_rows(
        outputs["group_summary"],
        GROUP_OUTPUT_FIELDNAMES,
        summarize_event_groups_by_point(sorted_event_rows, point_order, point_lookup),
    )
    write_rows(
        outputs["generation_summary"],
        GENERATION_OUTPUT_FIELDNAMES,
        summarize_generation_groups_by_point(
            sorted_trajectory_rows,
            point_order,
            point_lookup,
        ),
    )
    for path in outputs.values():
        print(f"wrote {relative(path)}", flush=True)

    if args.dry_run:
        return

    commit_results(args.commit_message, outputs)
    if args.push:
        run(["git", "push"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Slurm-array shards for the evolutionary interaction panel "
            "and write canonical result CSVs."
        )
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help="Prefix for merged output CSVs.",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=DEFAULT_SHARD_DIR,
        help="Directory containing per-array-task partial CSV outputs.",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=64,
        help="Expected number of array tasks/shards.",
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
    parser.add_argument(
        "--wait-for-shards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait until shard event rows reach the expected total.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Seconds between shard completeness checks.",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=48.0,
        help="Maximum wait before failing rather than merging partial results.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add evolutionary interaction results",
        help="Commit message for merged result CSVs.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the result commit after finalization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Merge and validate outputs without committing.",
    )
    return parser.parse_args()


def wait_for_complete_shards(
    shard_dir: Path,
    task_count: int,
    expected_runs: int,
    poll_seconds: int,
    max_wait_seconds: float,
) -> None:
    waited = 0.0
    while True:
        rows = count_event_rows(shard_dir, task_count)
        print(
            f"waiting for {expected_runs} shard event rows; found {rows}",
            flush=True,
        )
        if rows >= expected_runs:
            return
        sleep(poll_seconds)
        waited += poll_seconds
        if waited > max_wait_seconds:
            raise SystemExit(
                f"timed out waiting for {expected_runs} event rows in {shard_dir}"
            )


def count_event_rows(shard_dir: Path, task_count: int) -> int:
    return sum(
        count_data_rows(shard_paths(shard_dir, index)["events"])
        for index in range(task_count)
    )


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_shard_rows(
    shard_dir: Path,
    task_count: int,
    kind: str,
) -> list[dict[str, str]]:
    rows = []
    for index in range(task_count):
        path = shard_paths(shard_dir, index)[kind]
        if not path.exists():
            raise SystemExit(f"missing shard output: {path}")
        rows.extend(read_csv(path))
    return rows


def validate_events(
    rows: list[dict[str, str]],
    jobs: list[PointSeedJob],
) -> None:
    expected_keys = {(job.point.point, job.seed) for job in jobs}
    found_keys = {(row["point"], int(row["seed"])) for row in rows}
    if len(rows) != len(jobs):
        raise SystemExit(f"expected {len(jobs)} event rows, found {len(rows)}")
    if len(found_keys) != len(rows):
        raise SystemExit(
            f"expected {len(rows)} unique point-seed event rows, found {len(found_keys)}"
        )
    missing = expected_keys - found_keys
    unexpected = found_keys - expected_keys
    if missing:
        raise SystemExit(f"missing {len(missing)} expected point-seed event rows")
    if unexpected:
        raise SystemExit(f"found {len(unexpected)} unexpected point-seed event rows")


def job_fieldnames() -> list[str]:
    return [
        *JOB_FIELDNAMES,
        "parameter",
        "parameter_value",
        "is_baseline",
        *[
            name
            for name in POINT_OUTPUT_FIELDNAMES
            if name
            not in {
                "point",
                "parameter",
                "parameter_value",
                "is_baseline",
            }
        ],
    ]


def job_rows(jobs: list[PointSeedJob]) -> list[dict[str, str]]:
    return [
        {
            "job_index": str(job.index),
            "seed": str(job.seed),
            **point_context(job.point),
        }
        for job in jobs
    ]


def point_context(point: SensitivityPoint) -> dict[str, str]:
    return {**point_metadata(point), **point_param_values(point)}


def commit_results(commit_message: str, outputs: dict[str, Path]) -> None:
    paths = [str(path.relative_to(ROOT)) for path in outputs.values()]
    run(["git", "add", *paths])
    run(["git", "commit", "-m", commit_message])


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
