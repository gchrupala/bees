from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "report.md"
SECTION_START = "<!-- sensitivity-refinement-start -->"
SECTION_END = "<!-- sensitivity-refinement-end -->"
OAT_SECTION_END = "<!-- oat-sensitivity-end -->"

COARSE_EVENTS = RESULTS / "food_transition_oat_sensitivity_events.csv"
REFINEMENT_PREFIX = RESULTS / "food_transition_sensitivity_refinement"

REFINED_PARAMETERS = (
    "food_site_count",
    "mutation_sd",
    "vertical_comb_benefit",
    "transposition_mutation_correlation",
)
PARAMETER_ORDER = {parameter: index for index, parameter in enumerate(REFINED_PARAMETERS)}


@dataclass(frozen=True)
class CurveSummary:
    parameter: str
    parameter_value: str
    is_baseline: bool
    runs: int
    stable_count: int
    reached_gravity_count: int
    retained_vertical_count: int
    collapse_count: int
    mean_final_success: float
    mean_final_payoff: float
    mean_final_comb_tilt: float
    mean_final_min_transposition: float

    @property
    def stable_fraction(self) -> float:
        return self.stable_count / self.runs


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT / "experiments"))
    from run_food_transition_oat_sensitivity import output_paths

    outputs = output_paths(args.output_prefix)
    if args.wait_for_events:
        wait_for_complete_events(
            path=outputs["events"],
            expected_runs=args.expected_runs,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_hours * 3600,
        )

    events = read_csv(outputs["events"])
    validate_events(events, args.expected_runs)
    write_summary_files(args.output_prefix)

    combined_events = read_csv(COARSE_EVENTS) + events
    summaries = summarize_curve_points(combined_events)
    if args.dry_run:
        print(report_section(summaries, args.output_prefix), flush=True)
        return

    update_report(summaries, args.output_prefix)
    render_report()
    commit_results(args.commit_message, args.output_prefix)
    if args.push:
        run(["git", "push"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the refined one-parameter sensitivity panel and update "
            "the working report."
        )
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=REFINEMENT_PREFIX,
        help="Prefix for the refinement point, event, trajectory, and summary CSVs.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=1615,
        help="Expected number of completed refinement point-seed event rows.",
    )
    parser.add_argument(
        "--wait-for-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait for the refinement event CSV to reach --expected-runs rows.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Seconds between output completeness checks.",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=36.0,
        help="Maximum wait before failing rather than committing partial results.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add refined sensitivity results",
        help="Commit message for generated results and report updates.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the commit after successful finalization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the report section without editing files.",
    )
    return parser.parse_args()


def wait_for_complete_events(
    path: Path,
    expected_runs: int,
    poll_seconds: int,
    max_wait_seconds: float,
) -> None:
    waited = 0.0
    while count_data_rows(path) < expected_runs:
        rows = count_data_rows(path)
        print(
            f"waiting for {expected_runs} event rows in {relative(path)}; found {rows}",
            flush=True,
        )
        sleep(poll_seconds)
        waited += poll_seconds
        if waited > max_wait_seconds:
            raise SystemExit(
                f"timed out waiting for {expected_runs} event rows in {path}"
            )


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def validate_events(rows: list[dict[str, str]], expected_runs: int) -> None:
    if len(rows) != expected_runs:
        raise SystemExit(f"expected {expected_runs} event rows, found {len(rows)}")
    keys = {(row["point"], row["seed"]) for row in rows}
    if len(keys) != expected_runs:
        raise SystemExit(
            f"expected {expected_runs} unique point-seed rows, found {len(keys)}"
        )


def write_summary_files(output_prefix: Path) -> None:
    from analyze_optuna_best_trajectories import write_rows
    from run_food_transition_oat_sensitivity import (
        GENERATION_OUTPUT_FIELDNAMES,
        GROUP_OUTPUT_FIELDNAMES,
        PARAM_FIELDNAMES,
        SensitivityPoint,
        output_paths,
        summarize_event_groups_by_point,
        summarize_generation_groups_by_point,
    )

    outputs = output_paths(output_prefix)
    points = read_csv(outputs["points"])
    event_rows = read_csv(outputs["events"])
    trajectory_rows = read_csv(outputs["trajectories"])
    point_objects = [
        SensitivityPoint(
            point=row["point"],
            parameter=row["parameter"],
            parameter_value=row["parameter_value"],
            is_baseline=row["is_baseline"] == "true",
            values={name: parse_parameter(row[name]) for name in PARAM_FIELDNAMES},
        )
        for row in points
    ]
    point_order = {point.point: index for index, point in enumerate(point_objects)}
    point_lookup = {point.point: point for point in point_objects}
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
    print(f"wrote {relative(outputs['group_summary'])}", flush=True)
    print(f"wrote {relative(outputs['generation_summary'])}", flush=True)


def parse_parameter(raw: str) -> int | float:
    value = float(raw)
    if value.is_integer():
        return int(value)
    return value


def summarize_curve_points(rows: list[dict[str, str]]) -> list[CurveSummary]:
    grouped: dict[tuple[str, str, bool], dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["point"] == "baseline":
            for parameter in REFINED_PARAMETERS:
                key = (parameter, row[parameter], True)
                store_seed_row(grouped.setdefault(key, {}), row)
        elif row["parameter"] in PARAMETER_ORDER:
            key = (row["parameter"], row["parameter_value"], False)
            store_seed_row(grouped.setdefault(key, {}), row)

    summaries = [
        summarize_group(parameter, value, is_baseline, list(seed_rows.values()))
        for (parameter, value, is_baseline), seed_rows in grouped.items()
    ]
    return sorted(
        summaries,
        key=lambda summary: (
            PARAMETER_ORDER[summary.parameter],
            float(summary.parameter_value),
            not summary.is_baseline,
        ),
    )


def store_seed_row(seed_rows: dict[str, dict[str, str]], row: dict[str, str]) -> None:
    previous = seed_rows.get(row["seed"])
    if previous is None or elapsed_seconds(row) > elapsed_seconds(previous):
        seed_rows[row["seed"]] = row


def elapsed_seconds(row: dict[str, str]) -> float:
    return float(row.get("elapsed_seconds", "0"))


def summarize_group(
    parameter: str,
    value: str,
    is_baseline: bool,
    rows: list[dict[str, str]],
) -> CurveSummary:
    return CurveSummary(
        parameter=parameter,
        parameter_value=value,
        is_baseline=is_baseline,
        runs=len(rows),
        stable_count=count_true(rows, "stable_vertical_gravity"),
        reached_gravity_count=count_true(rows, "reached_gravity"),
        retained_vertical_count=count_true(rows, "retained_vertical"),
        collapse_count=count_true(rows, "collapsed"),
        mean_final_success=mean_float(rows, "final_success"),
        mean_final_payoff=mean_float(rows, "final_payoff"),
        mean_final_comb_tilt=mean_float(rows, "final_comb_tilt"),
        mean_final_min_transposition=mean_float(rows, "final_min_transposition"),
    )


def count_true(rows: list[dict[str, str]], field: str) -> int:
    return sum(row[field] == "true" for row in rows)


def mean_float(rows: list[dict[str, str]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def update_report(summaries: list[CurveSummary], output_prefix: Path) -> None:
    section = report_section(summaries, output_prefix)
    markdown = REPORT.read_text()
    if SECTION_START in markdown and SECTION_END in markdown:
        start = markdown.index(SECTION_START)
        end = markdown.index(SECTION_END) + len(SECTION_END)
        updated = f"{markdown[:start]}{section}{markdown[end:]}"
    elif OAT_SECTION_END in markdown:
        insert_at = markdown.index(OAT_SECTION_END) + len(OAT_SECTION_END)
        updated = f"{markdown[:insert_at]}\n\n{section}{markdown[insert_at:]}"
    else:
        updated = markdown.replace("\n# Conclusion\n", f"\n{section}\n# Conclusion\n")
    REPORT.write_text(updated)
    print(f"updated {relative(REPORT)}", flush=True)


def report_section(summaries: list[CurveSummary], output_prefix: Path) -> str:
    rows = "\n".join(table_row(summary) for summary in summaries)
    nonbaseline = [summary for summary in summaries if not summary.is_baseline]
    lowest = min(nonbaseline, key=lambda summary: summary.stable_fraction)
    highest = max(nonbaseline, key=lambda summary: summary.stable_fraction)
    outputs = output_file_descriptions(output_prefix)
    return "\n".join(
        [
            SECTION_START,
            "## Refined parameter sensitivity",
            "",
            (
                "We then refined the one-parameter panel around the coarse ranges "
                "with the largest stability changes: food-site count, mutation scale, "
                "vertical-comb benefit, and sender-receiver mutation coupling. The "
                "table combines the original coarse anchors with the new refinement "
                f"points; each row uses 100 held-out seeds. New raw rows are saved in "
                f"`{outputs['events']}` and `{outputs['trajectories']}`."
            ),
            "",
            (
                f"Across these refined points, the lowest stability occurred for "
                f"`{lowest.parameter}={lowest.parameter_value}` "
                f"({lowest.stable_count}/{lowest.runs}, "
                f"{percent(lowest.stable_fraction)}), and the highest occurred for "
                f"`{highest.parameter}={highest.parameter_value}` "
                f"({highest.stable_count}/{highest.runs}, "
                f"{percent(highest.stable_fraction)}). Baseline rows are repeated as "
                "within-parameter anchors rather than as independent perturbations."
            ),
            "",
            "| Parameter | Value | Baseline | Stable | Gravity reached | Vertical retained | Collapse | Mean succ. | Mean payoff | Mean $t_f$ | Mean $m_f$ |",
            "|:----------|:------|:---------|-------:|----------------:|------------------:|---------:|-----------:|------------:|-----------:|-----------:|",
            rows,
            "",
            (
                "These are still one-parameter perturbations, so they map local "
                "stability boundaries rather than interactions among parameters."
            ),
            SECTION_END,
        ]
    )


def output_file_descriptions(output_prefix: Path) -> dict[str, str]:
    sys.path.insert(0, str(ROOT / "experiments"))
    from run_food_transition_oat_sensitivity import output_paths

    return {
        name: relative(path)
        for name, path in output_paths(output_prefix).items()
    }


def table_row(summary: CurveSummary) -> str:
    baseline = "yes" if summary.is_baseline else "no"
    return (
        f"| {summary.parameter} | {summary.parameter_value} | {baseline} | "
        f"{summary.stable_count}/{summary.runs} | "
        f"{summary.reached_gravity_count}/{summary.runs} | "
        f"{summary.retained_vertical_count}/{summary.runs} | "
        f"{summary.collapse_count}/{summary.runs} | "
        f"{summary.mean_final_success:.3f} | "
        f"{summary.mean_final_payoff:.3f} | "
        f"{summary.mean_final_comb_tilt:.3f} | "
        f"{summary.mean_final_min_transposition:.3f} |"
    )


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_report() -> None:
    if shutil.which("pandoc") is None:
        print("pandoc unavailable; leaving report/report.html unchanged", flush=True)
        return
    run([sys.executable, "-u", "experiments/render_report_html.py"])


def commit_results(message: str, output_prefix: Path) -> None:
    sys.path.insert(0, str(ROOT / "experiments"))
    from run_food_transition_oat_sensitivity import output_paths

    outputs = output_paths(output_prefix)
    paths = [
        REPORT,
        ROOT / "report" / "report.html",
        *outputs.values(),
    ]
    git_paths = []
    for path in paths:
        if not path.exists():
            continue
        try:
            git_paths.append(str(path.relative_to(ROOT)))
        except ValueError:
            continue
    run(["git", "add", *git_paths])
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
        check=False,
    )
    if status.returncode == 0:
        print("no staged changes to commit", flush=True)
        return
    run(["git", "commit", "-m", message])


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
