from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "report" / "tables"
FIGURES = ROOT / "report" / "figures"
SOURCE_MANIFEST = ROOT / "report" / "artifact_sources.csv"


@dataclass(frozen=True)
class ResultSpec:
    name: str
    command: tuple[str, ...]
    result_files: tuple[Path, ...]
    stdout_file: Path | None = None


@dataclass(frozen=True)
class TableSpec:
    name: str
    label: str
    result_name: str
    result_file: Path
    table_file: Path
    builder: Callable[[], str]


@dataclass(frozen=True)
class FigureSpec:
    name: str
    label: str
    result_name: str
    result_file: Path
    figure_file: Path
    builder: Callable[[], str]


def main() -> None:
    args = parse_args()
    selected = select_result_names(args.only)

    if args.action == "list":
        write_source_manifest()
        print_source_manifest()
        return

    if args.action in {"results", "all"}:
        run_results(selected)

    if args.action in {"tables", "artifacts", "all"}:
        build_artifacts(selected)
        write_source_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate report result CSVs and LaTeX fragments.",
    )
    parser.add_argument(
        "action",
        choices=("results", "tables", "artifacts", "all", "list"),
        help=(
            "Use 'results' to run experiments, 'artifacts' or 'tables' to "
            "rebuild LaTeX fragments from existing CSVs, 'all' for both, or "
            "'list' to show provenance."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=tuple(result_specs()),
        help="Limit work to one or more result sets.",
    )
    return parser.parse_args()


def select_result_names(only: list[str] | None) -> set[str]:
    if only:
        return set(only)

    return set(result_specs())


def result_specs() -> dict[str, ResultSpec]:
    return {
        "report_conditions": ResultSpec(
            name="report_conditions",
            command=("python", "-u", "experiments/run_report_conditions.py"),
            stdout_file=RESULTS / "report_conditions.csv",
            result_files=(RESULTS / "report_conditions.csv",),
        ),
        "tilt_geometry_calibration": ResultSpec(
            name="tilt_geometry_calibration",
            command=(
                "python",
                "-u",
                "experiments/run_tilt_geometry_sanity.py",
                "--seeds",
                "101,102,103,104,105,106,107,108,109,110",
                "--initial-comb-tilts",
                "0.0,0.5,1.0",
                "--vertical-comb-benefits",
                "0.15,0.20,0.25",
            ),
            stdout_file=RESULTS / "tilt_geometry_calibration.csv",
            result_files=(RESULTS / "tilt_geometry_calibration.csv",),
        ),
        "orientation_calibration": ResultSpec(
            name="orientation_calibration",
            command=("python", "-u", "experiments/run_orientation_sanity.py"),
            stdout_file=RESULTS / "orientation_calibration.csv",
            result_files=(RESULTS / "orientation_calibration.csv",),
        ),
        "vertical_coupling_probe": ResultSpec(
            name="vertical_coupling_probe",
            command=("python", "-u", "experiments/run_vertical_coupling_probe.py"),
            stdout_file=RESULTS / "vertical_coupling_probe.csv",
            result_files=(RESULTS / "vertical_coupling_probe.csv",),
        ),
        "long_vertical_transition": ResultSpec(
            name="long_vertical_transition",
            command=(
                "python",
                "-u",
                "experiments/run_long_vertical_transition.py",
                "--summary-output",
                "results/long_vertical_transition_summary.csv",
                "--seed-output",
                "results/long_vertical_transition_seeds.csv",
                "--vertical-comb-modifiers",
                "linear,threshold_0.8",
            ),
            stdout_file=None,
            result_files=(
                RESULTS / "long_vertical_transition_summary.csv",
                RESULTS / "long_vertical_transition_seeds.csv",
            ),
        ),
    }


def table_specs() -> dict[str, TableSpec]:
    return {
        "food_distribution": TableSpec(
            name="food_distribution",
            label="tab:food-distribution",
            result_name="report_conditions",
            result_file=RESULTS / "report_conditions.csv",
            table_file=TABLES / "food_distribution.tex",
            builder=build_food_distribution_table,
        ),
        "comb_tilt": TableSpec(
            name="comb_tilt",
            label="tab:comb-tilt",
            result_name="report_conditions",
            result_file=RESULTS / "report_conditions.csv",
            table_file=TABLES / "comb_tilt.tex",
            builder=build_comb_tilt_table,
        ),
        "tilt_benefit_calibration": TableSpec(
            name="tilt_benefit_calibration",
            label="tab:tilt-benefit-calibration",
            result_name="tilt_geometry_calibration",
            result_file=RESULTS / "tilt_geometry_calibration.csv",
            table_file=TABLES / "tilt_benefit_calibration.tex",
            builder=build_tilt_benefit_calibration_table,
        ),
        "orientation_calibration": TableSpec(
            name="orientation_calibration",
            label="tab:orientation-calibration",
            result_name="orientation_calibration",
            result_file=RESULTS / "orientation_calibration.csv",
            table_file=TABLES / "orientation_calibration.tex",
            builder=build_orientation_calibration_table,
        ),
        "vertical_coupling_probe": TableSpec(
            name="vertical_coupling_probe",
            label="tab:vertical-coupling-probe",
            result_name="vertical_coupling_probe",
            result_file=RESULTS / "vertical_coupling_probe.csv",
            table_file=TABLES / "vertical_coupling_probe.tex",
            builder=build_vertical_coupling_probe_table,
        ),
    }


def figure_specs() -> dict[str, FigureSpec]:
    return {
        "long_transition_heatmap": FigureSpec(
            name="long_transition_heatmap",
            label="fig:long-transition-heatmap",
            result_name="long_vertical_transition",
            result_file=RESULTS / "long_vertical_transition_summary.csv",
            figure_file=FIGURES / "long_transition_heatmap.tex",
            builder=build_long_transition_heatmap,
        ),
    }


def run_results(selected: set[str]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for spec in result_specs().values():
        if spec.name not in selected:
            continue

        print(f"running {spec.name}: {format_command(spec.command)}", file=sys.stderr)
        command = resolve_python(spec.command)
        if spec.stdout_file is None:
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            spec.stdout_file.parent.mkdir(parents=True, exist_ok=True)
            stream_stdout_to_file(spec, command)


def stream_stdout_to_file(spec: ResultSpec, command: list[str]) -> None:
    assert spec.stdout_file is not None
    temp_file = spec.stdout_file.with_suffix(spec.stdout_file.suffix + ".tmp")
    with (
        temp_file.open("w", newline="") as output,
        subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=None,
        ) as process,
    ):
        assert process.stdout is not None
        expected_fields = None
        row_number = 0
        for raw_line in process.stdout:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line:
                continue

            field_count = len(next(csv.reader([line])))
            if expected_fields is None:
                expected_fields = field_count
                output.write(line + "\n")
                output.flush()
                continue

            if field_count != expected_fields:
                print(
                    (
                        f"  skipped malformed {spec.name} line: "
                        f"expected {expected_fields} fields, got {field_count}: "
                        f"{line}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                continue

            row_number += 1
            output.write(line + "\n")
            output.flush()
            print(
                f"  {spec.name} row {row_number}: {line}",
                file=sys.stderr,
                flush=True,
            )

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    temp_file.replace(spec.stdout_file)


def build_artifacts(selected: set[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for spec in table_specs().values():
        if spec.result_name not in selected:
            continue

        assert_result_files_exist(result_specs()[spec.result_name])
        spec.table_file.write_text(spec.builder())
        print(f"wrote {relative(spec.table_file)}", file=sys.stderr)

    for spec in figure_specs().values():
        if spec.result_name not in selected:
            continue

        assert_result_files_exist(result_specs()[spec.result_name])
        spec.figure_file.write_text(spec.builder())
        print(f"wrote {relative(spec.figure_file)}", file=sys.stderr)


def assert_result_files_exist(spec: ResultSpec) -> None:
    missing = [path for path in spec.result_files if not path.exists()]
    if missing:
        paths = ", ".join(relative(path) for path in missing)
        raise FileNotFoundError(
            f"missing result file(s): {paths}; run "
            f"`python -u experiments/run_report_artifacts.py results "
            f"--only {spec.name}` first"
        )


def build_food_distribution_table() -> str:
    rows = rows_by(read_csv(RESULTS / "report_conditions.csv"), "condition")
    body = "\n".join(
        table_row(
            [
                label,
                count_from_fraction(rows[key]["reached_fraction"], rows[key]["seeds"]),
                rows[key]["mean_reach_generation"],
                rows[key]["final_bias"],
                rows[key]["final_attention"],
                rows[key]["final_search_limit"],
                rows[key]["final_success"],
                rows[key]["final_payoff"],
            ]
        )
        for key, label in (("hard", "Hard"), ("baseline", "Baseline"), ("easy", "Easy"))
    )
    return generated_table(
        name="food_distribution",
        result_name="report_conditions",
        column_spec="lrrrrrrr",
        header=(
            "Condition & Reached & Reach gen. & Prec. & Attention & Search "
            "& Success & Payoff \\\\"
        ),
        body=body,
        caption=(
            "Food-distribution experiment under the current dynamic recruitment\n"
            "model. ``Reached'' is the number of seeds in which mean directional "
            "precision\n"
            "investment reached 0.30 by generation 30. ``Reach gen.'' is averaged "
            "over\n"
            "reached seeds only. Prec. is final mean directional precision "
            "investment;\n"
            "attention, search limit, success, and payoff values are also "
            "final-generation\n"
            "means over all ten seeds."
        ),
        label="tab:food-distribution",
    )


def build_comb_tilt_table() -> str:
    rows = rows_by(read_csv(RESULTS / "report_conditions.csv"), "condition")
    body = "\n".join(
        table_row(
            [
                label,
                count_from_fraction(rows[key]["reached_fraction"], rows[key]["seeds"]),
                rows[key]["mean_reach_generation"],
                rows[key]["final_bias"],
                rows[key]["final_attention"],
                rows[key]["final_sender_transposition"],
                rows[key]["final_receiver_transposition"],
                rows[key]["final_success"],
                rows[key]["final_payoff"],
            ]
        )
        for key, label in (
            ("horizontal", "Horizontal"),
            ("tilted", "Tilted"),
            ("vertical", "Vertical"),
        )
    )
    return generated_table(
        name="comb_tilt",
        result_name="report_conditions",
        column_spec="lrrrrrrrr",
        header=(
            "Comb & Reached & Reach gen. & Prec. & Attention & Sender & Receiver "
            "& Success & Payoff \\\\"
        ),
        body=body,
        caption=(
            "Comb-tilt comparison using the baseline food distribution and ten\n"
            "seeds. Sender and receiver denote final mean sender and receiver "
            "transposition\n"
            "traits."
        ),
        label="tab:comb-tilt",
    )


def build_tilt_benefit_calibration_table() -> str:
    rows = read_csv(RESULTS / "tilt_geometry_calibration.csv")
    body = "\n".join(
        table_row(
            [
                row["initial_comb_tilt"],
                row["vertical_comb_benefit"],
                row["mean_final_comb_tilt"],
                row["mean_final_sender_transposition"],
                row["mean_final_receiver_transposition"],
                row["mean_final_success"],
                row["mean_final_payoff"],
                count_from_fraction(row["reached_fraction"], row["seeds"]),
            ]
        )
        for row in rows
    )
    return generated_table(
        name="tilt_benefit_calibration",
        result_name="tilt_geometry_calibration",
        column_spec="rrrrrrrr",
        header=(
            "Initial tilt & $\\alpha$ & Final tilt & Sender & Receiver & Success "
            "& Payoff & Reached \\\\"
        ),
        body=body,
        caption=(
            "Ten-seed calibration of the proportional vertical-comb advantage under the explicit\n"
            "tilt-geometry model. Sender and receiver are final mean transposition "
            "traits.\n"
            "Reached is the number of seeds in which mean directional precision "
            "investment\n"
            "reached 0.30 by generation 30."
        ),
        label="tab:tilt-benefit-calibration",
    )


def build_orientation_calibration_table() -> str:
    rows = read_csv(RESULTS / "orientation_calibration.csv")
    body = "\n".join(
        table_row(
            [
                row["initial_comb_tilt"],
                row["vertical_comb_benefit"],
                row["mean_final_tilt"],
                row["mean_final_success"],
                row["mean_within_orientation_alignment"],
                row["weighted_mean_orientation_degrees"],
                sun_offset(row["weighted_mean_orientation_degrees"]),
                row["weighted_across_seed_orientation_alignment"],
            ]
        )
        for row in rows
    )
    return generated_table(
        name="orientation_calibration",
        result_name="orientation_calibration",
        column_spec="rrrrrrrr",
        header=(
            "Initial tilt & $\\alpha$ & Final tilt & Success & Within align. "
            "& Mean orient. & Sun offset & Across align. \\\\"
        ),
        body=body,
        caption=(
            "Orientation sanity check under the explicit tilt-geometry model.\n"
            "Mean orientation and sun offset are measured in degrees. Across "
            "alignment is\n"
            "the weighted across-seed orientation alignment; values near 0 indicate "
            "that\n"
            "different seeds choose different absolute orientations."
        ),
        label="tab:orientation-calibration",
    )


def build_vertical_coupling_probe_table() -> str:
    rows = read_csv(RESULTS / "vertical_coupling_probe.csv")
    body = "\n".join(
        table_row(
            [
                row["transposition_mutation_correlation"],
                count_from_fraction(row["reached_gravity_fraction"], row["seeds"]),
                row["mean_gravity_reach_generation"],
                row["mean_final_bias"],
                row["mean_final_sender_transposition"],
                row["mean_final_receiver_transposition"],
                row["mean_final_transposition_gap"],
                row["mean_final_success"],
                row["mean_final_payoff"],
            ]
        )
        for row in rows
    )
    return generated_table(
        name="vertical_coupling_probe",
        result_name="vertical_coupling_probe",
        column_spec="rrrrrrrrr",
        header=(
            "Corr. & Reached & Reach gen. & Prec. & Sender & Receiver & Gap "
            "& Success & Payoff \\\\"
        ),
        body=body,
        caption=(
            "Constrained near-vertical coupling probe. Corr. is the correlation\n"
            "between sender and receiver transposition mutation increments. "
            "``Reached''\n"
            "counts seeds in which both mean sender and receiver transposition "
            "reached 0.50\n"
            "by generation 120. Reach generation is averaged over reached seeds "
            "only.\n"
            "Gap is the final mean absolute sender--receiver transposition "
            "difference."
        ),
        label="tab:vertical-coupling-probe",
    )


def build_long_transition_heatmap() -> str:
    rows = read_csv(RESULTS / "long_vertical_transition_summary.csv")
    for row in rows:
        row.setdefault("vertical_comb_modifier", "linear")

    modifiers = sorted(
        {row["vertical_comb_modifier"] for row in rows},
        key=vertical_comb_modifier_sort_key,
    )
    metrics = (
        (
            "Gravity",
            "reached_gravity_fraction",
            "count",
            "blue",
            "Seeds reaching gravity-channel transposition",
        ),
        (
            "Vertical",
            "retained_vertical_fraction",
            "count",
            "teal",
            "Seeds retaining final comb tilt at least 0.80",
        ),
        (
            "Collapse",
            "collapse_fraction",
            "count",
            "red",
            "Seeds whose success fell to 0.02 or below",
        ),
        (
            "Recovered",
            "recovered_from_collapse_fraction",
            "count",
            "green",
            "Collapsed seeds ending with success at least 0.10",
        ),
        (
            "Final success",
            "mean_final_success",
            "success",
            "violet",
            "Final-generation mean foraging success",
        ),
        (
            "Final tilt",
            "mean_final_comb_tilt",
            "unit",
            "orange",
            "Final-generation mean comb tilt",
        ),
    )

    blocks = []
    for modifier_index, modifier in enumerate(modifiers):
        modifier_rows = [
            row for row in rows if row["vertical_comb_modifier"] == modifier
        ]
        lookup = {
            (float(row["initial_comb_tilt"]), float(row["vertical_comb_benefit"])): row
            for row in modifier_rows
        }
        initial_tilts = sorted({key[0] for key in lookup}, reverse=True)
        benefits = sorted({key[1] for key in lookup})
        block_y = -7.30 * modifier_index
        blocks.append(
            f"\\node[modifierlabel] at (0.0,{block_y + 0.45:.2f}) "
            f"{{{vertical_comb_modifier_label(modifier)}}};"
        )
        blocks.extend(
            build_heatmap_panel(
                title=title,
                field=field,
                value_kind=value_kind,
                color=color,
                rows=lookup,
                initial_tilts=initial_tilts,
                benefits=benefits,
                origin_x=0.0 + 4.85 * (index % 3),
                origin_y=block_y - 0.10 - 3.05 * (index // 3),
            )
            for index, (
                title,
                field,
                value_kind,
                color,
                _description,
            ) in enumerate(metrics)
        )

    legend_y = -6.95 - 7.30 * (len(modifiers) - 1)
    legend = "\n".join(
        f"\\node[legend] at (0.0,{legend_y - 0.25 * index:.2f})"
        f" {{{title}: {description}.}};"
        for index, (title, _field, _kind, _color, description) in enumerate(metrics)
    )

    return generated_figure(
        name="long_transition_heatmap",
        result_name="long_vertical_transition",
        body="\n".join(blocks) + "\n" + legend,
        caption=(
            "Long axial-orientation transition experiment. Rows are initial "
            "comb tilt, columns are proportional vertical-comb advantage "
            "$\\alpha$, and modifier headings give $f(t)$. Count panels report "
            "seeds out of ten; continuous panels report final-generation means. "
            "The generated fragment records the source CSV."
        ),
        label="fig:long-transition-heatmap",
    )


def vertical_comb_modifier_sort_key(modifier: str) -> tuple[int, str]:
    order = {"linear": 0, "threshold_0.8": 1}
    return (order.get(modifier, len(order)), modifier)


def vertical_comb_modifier_label(modifier: str) -> str:
    if modifier == "linear":
        return "Modifier: $f(t)=t$"

    if modifier == "threshold_0.8":
        return "Modifier: $f(t)=\\mathbf{1}[t \\geq 0.8]$"

    escaped_modifier = modifier.replace("_", "\\_")
    return f"Modifier: {escaped_modifier}"


def generated_table(
    name: str,
    result_name: str,
    column_spec: str,
    header: str,
    body: str,
    caption: str,
    label: str,
) -> str:
    command = f"python -u experiments/run_report_artifacts.py results --only {result_name}"
    return (
        f"% Generated by experiments/run_report_artifacts.py; do not edit by hand.\n"
        f"% Result command: {command}\n"
        f"% Result file: {relative(table_specs()[name].result_file)}\n"
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\small\n"
        f"\\begin{{tabular}}{{{column_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def generated_figure(
    name: str,
    result_name: str,
    body: str,
    caption: str,
    label: str,
) -> str:
    command = f"python -u experiments/run_report_artifacts.py results --only {result_name}"
    return (
        "% Generated by experiments/run_report_artifacts.py; do not edit by hand.\n"
        f"% Result command: {command}\n"
        f"% Result file: {relative(figure_specs()[name].result_file)}\n"
        "\\begin{figure}[h]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tikzpicture}[\n"
        "  cell/.style={minimum width=0.82cm, minimum height=0.56cm, "
        "anchor=center, draw=white, line width=0.7pt, font=\\scriptsize},\n"
        "  paneltitle/.style={font=\\footnotesize\\bfseries, anchor=west},\n"
        "  modifierlabel/.style={font=\\footnotesize\\bfseries, anchor=west},\n"
        "  axislabel/.style={font=\\scriptsize},\n"
        "  legend/.style={font=\\scriptsize, anchor=west}\n"
        "]\n"
        f"{body}\n"
        "\\end{tikzpicture}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )


def build_heatmap_panel(
    title: str,
    field: str,
    value_kind: str,
    color: str,
    rows: dict[tuple[float, float], dict[str, str]],
    initial_tilts: list[float],
    benefits: list[float],
    origin_x: float,
    origin_y: float,
) -> str:
    lines = [
        f"\\node[paneltitle] at ({origin_x:.2f},{origin_y:.2f}) {{{title}}};",
        (
            f"\\node[axislabel, anchor=east] at ({origin_x + 0.72:.2f},"
            f"{origin_y - 2.55:.2f}) {{$t_0$}};"
        ),
        (
            f"\\node[axislabel] at ({origin_x + 2.05:.2f},"
            f"{origin_y - 2.55:.2f}) {{$\\alpha$}};"
        ),
    ]
    for column, benefit in enumerate(benefits):
        x = origin_x + 1.15 + column * 0.82
        lines.append(
            f"\\node[axislabel] at ({x:.2f},{origin_y - 0.43:.2f}) "
            f"{{{benefit:.2f}}};"
        )

    for row_index, initial_tilt in enumerate(initial_tilts):
        y = origin_y - 0.88 - row_index * 0.56
        lines.append(
            f"\\node[axislabel, anchor=east] at ({origin_x + 0.75:.2f},{y:.2f}) "
            f"{{{initial_tilt:.1f}}};"
        )
        for column, benefit in enumerate(benefits):
            x = origin_x + 1.15 + column * 0.82
            row = rows[(initial_tilt, benefit)]
            text, intensity = heatmap_cell_value(row, field, value_kind)
            lines.append(
                f"\\node[cell, fill={color}!{intensity}!white] at "
                f"({x:.2f},{y:.2f}) {{{text}}};"
            )

    return "\n".join(lines)


def heatmap_cell_value(row: dict[str, str], field: str, value_kind: str) -> tuple[str, int]:
    if value_kind == "count":
        fraction = float(row[field])
        seeds = int(row["seeds"])
        count = round(fraction * seeds)
        return f"{count}/{seeds}", round(100 * fraction)

    value = float(row[field])
    if value_kind == "success":
        return f"{value:.2f}", round(100 * min(value / 0.2, 1.0))

    if value_kind == "unit":
        return f"{value:.2f}", round(100 * min(value, 1.0))

    raise ValueError(f"unknown heatmap value kind: {value_kind}")


def write_source_manifest() -> None:
    SOURCE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "artifact_kind",
                "artifact_label",
                "artifact_file",
                "result_file",
                "result_command",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for table in table_specs().values():
            writer.writerow(
                {
                    "artifact_kind": "table",
                    "artifact_label": table.label,
                    "artifact_file": relative(table.table_file),
                    "result_file": relative(table.result_file),
                    "result_command": (
                        "python -u experiments/run_report_artifacts.py "
                        f"results --only {table.result_name}"
                    ),
                }
            )
        for figure in figure_specs().values():
            writer.writerow(
                {
                    "artifact_kind": "figure",
                    "artifact_label": figure.label,
                    "artifact_file": relative(figure.figure_file),
                    "result_file": relative(figure.result_file),
                    "result_command": (
                        "python -u experiments/run_report_artifacts.py "
                        f"results --only {figure.result_name}"
                    ),
                }
            )


def print_source_manifest() -> None:
    with SOURCE_MANIFEST.open() as handle:
        print(handle.read(), end="")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rows_by(rows: Iterable[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def table_row(values: list[str]) -> str:
    return " & ".join(values) + r" \\"


def count_from_fraction(fraction: str, seeds: str) -> str:
    seed_count = int(seeds)
    count = round(float(fraction) * seed_count)
    return f"{count}/{seed_count}"


def sun_offset(mean_orientation_degrees: str) -> str:
    offset = (float(mean_orientation_degrees) - 90.0 + 180.0) % 360.0 - 180.0
    return f"{offset:.1f}"


def resolve_python(command: tuple[str, ...]) -> list[str]:
    if command[0] != "python":
        return list(command)

    return [sys.executable, *command[1:]]


def format_command(command: tuple[str, ...]) -> str:
    return " ".join(command)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
