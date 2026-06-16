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
SECTION_START = "<!-- oat-sensitivity-start -->"
SECTION_END = "<!-- oat-sensitivity-end -->"

EVENTS = RESULTS / "food_transition_oat_sensitivity_events.csv"
TRAJECTORIES = RESULTS / "food_transition_oat_sensitivity_trajectories.csv"
POINTS = RESULTS / "food_transition_oat_sensitivity_points.csv"
GROUP_SUMMARY = RESULTS / "food_transition_oat_sensitivity_group_summary.csv"
GENERATION_SUMMARY = RESULTS / "food_transition_oat_sensitivity_generation_summary.csv"


@dataclass(frozen=True)
class PointSummary:
    point: str
    parameter: str
    parameter_value: str
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
    if args.wait_for_service:
        wait_for_service(
            service=args.service,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_hours * 3600,
        )
        found = count_data_rows(EVENTS)
        if found != args.expected_runs:
            raise SystemExit(
                f"{args.service} stopped with {found}/{args.expected_runs} event rows"
            )
    else:
        wait_for_complete_events(
            expected_runs=args.expected_runs,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_hours * 3600,
        )

    points = read_csv(POINTS)
    events = read_csv(EVENTS)
    trajectories = read_csv(TRAJECTORIES)
    validate_events(events, args.expected_runs)
    write_summaries(points, events, trajectories)
    update_report(summarize_points(events, points))
    render_report()
    commit_results(args.commit_message)
    if args.push:
        run(["git", "push"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the coarse OAT sensitivity run after the unattended "
            "simulation completes."
        )
    )
    parser.add_argument(
        "--service",
        default="bees-oat-sensitivity-streaming.service",
        help="User systemd service to wait for before finalizing.",
    )
    parser.add_argument(
        "--wait-for-service",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for the systemd service to stop before reading final outputs.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=1520,
        help="Expected number of completed point-seed event rows.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Seconds between service/output completeness checks.",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=36.0,
        help="Maximum wait before failing rather than committing partial results.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add OAT sensitivity results",
        help="Commit message for generated results and report updates.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the commit after successful finalization.",
    )
    return parser.parse_args()


def wait_for_service(service: str, poll_seconds: int, max_wait_seconds: float) -> None:
    waited = 0.0
    while service_active(service):
        print(f"{service} still active; waited {waited / 3600:.2f}h", flush=True)
        sleep(poll_seconds)
        waited += poll_seconds
        if waited > max_wait_seconds:
            raise SystemExit(f"timed out waiting for {service}")
    print(f"{service} is no longer active", flush=True)


def service_active(service: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", service],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() == "active"


def wait_for_complete_events(
    expected_runs: int,
    poll_seconds: int,
    max_wait_seconds: float,
) -> None:
    waited = 0.0
    while count_data_rows(EVENTS) < expected_runs:
        rows = count_data_rows(EVENTS)
        print(
            f"waiting for {expected_runs} event rows; found {rows}",
            flush=True,
        )
        sleep(poll_seconds)
        waited += poll_seconds
        if waited > max_wait_seconds:
            raise SystemExit(
                f"timed out waiting for {expected_runs} event rows in {EVENTS}"
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


def write_summaries(
    points: list[dict[str, str]],
    events: list[dict[str, str]],
    trajectories: list[dict[str, str]],
) -> None:
    write_summary_files()


def write_summary_files() -> None:
    sys.path.insert(0, str(ROOT / "experiments"))
    from run_food_transition_oat_sensitivity import (
        GENERATION_OUTPUT_FIELDNAMES,
        GROUP_OUTPUT_FIELDNAMES,
        PARAM_FIELDNAMES,
        SensitivityPoint,
        summarize_event_groups_by_point,
        summarize_generation_groups_by_point,
    )
    from analyze_optuna_best_trajectories import write_rows

    points = read_csv(POINTS)
    event_rows = read_csv(EVENTS)
    trajectory_rows = read_csv(TRAJECTORIES)
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
        GROUP_SUMMARY,
        GROUP_OUTPUT_FIELDNAMES,
        summarize_event_groups_by_point(event_rows, point_order, point_lookup),
    )
    write_rows(
        GENERATION_SUMMARY,
        GENERATION_OUTPUT_FIELDNAMES,
        summarize_generation_groups_by_point(
            trajectory_rows,
            point_order,
            point_lookup,
        ),
    )
    print(f"wrote {relative(GROUP_SUMMARY)}", flush=True)
    print(f"wrote {relative(GENERATION_SUMMARY)}", flush=True)


def parse_parameter(raw: str) -> int | float:
    value = float(raw)
    if value.is_integer():
        return int(value)
    return value


def summarize_points(
    events: list[dict[str, str]],
    points: list[dict[str, str]],
) -> list[PointSummary]:
    point_order = {row["point"]: index for index, row in enumerate(points)}
    rows_by_point: dict[str, list[dict[str, str]]] = {}
    for row in events:
        rows_by_point.setdefault(row["point"], []).append(row)

    summaries = []
    for point in sorted(rows_by_point, key=point_order.get):
        point_rows = rows_by_point[point]
        first = point_rows[0]
        summaries.append(
            PointSummary(
                point=point,
                parameter=first["parameter"],
                parameter_value=first["parameter_value"],
                runs=len(point_rows),
                stable_count=count_true(point_rows, "stable_vertical_gravity"),
                reached_gravity_count=count_true(point_rows, "reached_gravity"),
                retained_vertical_count=count_true(point_rows, "retained_vertical"),
                collapse_count=count_true(point_rows, "collapsed"),
                mean_final_success=mean_float(point_rows, "final_success"),
                mean_final_payoff=mean_float(point_rows, "final_payoff"),
                mean_final_comb_tilt=mean_float(point_rows, "final_comb_tilt"),
                mean_final_min_transposition=mean_float(
                    point_rows,
                    "final_min_transposition",
                ),
            )
        )
    return summaries


def count_true(rows: list[dict[str, str]], field: str) -> int:
    return sum(row[field] == "true" for row in rows)


def mean_float(rows: list[dict[str, str]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def update_report(summaries: list[PointSummary]) -> None:
    section = report_section(summaries)
    markdown = REPORT.read_text()
    if SECTION_START in markdown and SECTION_END in markdown:
        start = markdown.index(SECTION_START)
        end = markdown.index(SECTION_END) + len(SECTION_END)
        updated = f"{markdown[:start]}{section}{markdown[end:]}"
    else:
        updated = markdown.replace("\n# Conclusion\n", f"\n{section}\n# Conclusion\n")
    REPORT.write_text(updated)
    print(f"updated {relative(REPORT)}", flush=True)


def report_section(summaries: list[PointSummary]) -> str:
    baseline = next(summary for summary in summaries if summary.point == "baseline")
    nonbaseline = [summary for summary in summaries if summary.point != "baseline"]
    lowest = min(nonbaseline, key=lambda summary: summary.stable_fraction)
    highest = max(nonbaseline, key=lambda summary: summary.stable_fraction)
    rows = "\n".join(table_row(summary) for summary in summaries)
    return "\n".join(
        [
            SECTION_START,
            "## Coarse one-parameter sensitivity",
            "",
            (
                "We next ran a coarse one-parameter-at-a-time sensitivity panel around "
                "the held-out-validated `repeated_7site` candidate. Each point used "
                f"{baseline.runs} held-out seeds, with all other parameters fixed at "
                "the validated baseline. Raw event and trajectory rows are saved in "
                "`results/food_transition_oat_sensitivity_events.csv` and "
                "`results/food_transition_oat_sensitivity_trajectories.csv`; point, "
                "group-summary, and generation-summary files use the same prefix."
            ),
            "",
            (
                f"The baseline reproduced {baseline.stable_count}/{baseline.runs} "
                f"stable vertical gravity-code outcomes ({percent(baseline.stable_fraction)}), "
                f"with mean final success {baseline.mean_final_success:.3f}, mean final "
                f"comb tilt {baseline.mean_final_comb_tilt:.3f}, and mean final minimum "
                f"sender-receiver transposition {baseline.mean_final_min_transposition:.3f}. "
                f"Across the coarse perturbations, the lowest stability occurred for "
                f"`{lowest.parameter}={lowest.parameter_value}` "
                f"({lowest.stable_count}/{lowest.runs}, {percent(lowest.stable_fraction)}), "
                f"whereas the highest non-baseline stability occurred for "
                f"`{highest.parameter}={highest.parameter_value}` "
                f"({highest.stable_count}/{highest.runs}, {percent(highest.stable_fraction)})."
            ),
            "",
            "| Parameter | Value | Stable | Gravity reached | Vertical retained | Collapse | Mean succ. | Mean payoff | Mean $t_f$ | Mean $m_f$ |",
            "|:----------|:------|-------:|----------------:|------------------:|---------:|-----------:|------------:|-----------:|-----------:|",
            rows,
            "",
            (
                "This first pass is deliberately coarse: it identifies broad cliffs and "
                "candidate robust ranges rather than estimating smooth response curves. "
                "The next step is to refine only the parameter ranges where the stable "
                "fraction changes sharply."
            ),
            SECTION_END,
        ]
    )


def table_row(summary: PointSummary) -> str:
    parameter = "baseline" if summary.point == "baseline" else summary.parameter
    value = "-" if summary.point == "baseline" else summary.parameter_value
    return (
        f"| {parameter} | {value} | {summary.stable_count}/{summary.runs} | "
        f"{summary.reached_gravity_count}/{summary.runs} | "
        f"{summary.retained_vertical_count}/{summary.runs} | "
        f"{summary.collapse_count}/{summary.runs} | "
        f"{summary.mean_final_success:.3f} | {summary.mean_final_payoff:.3f} | "
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


def commit_results(message: str) -> None:
    paths = [
        REPORT,
        ROOT / "report" / "report.html",
        POINTS,
        EVENTS,
        TRAJECTORIES,
        GROUP_SUMMARY,
        GENERATION_SUMMARY,
    ]
    run(["git", "add", *[str(path.relative_to(ROOT)) for path in paths if path.exists()]])
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
