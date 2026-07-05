"""One-parameter sensitivity boxplots for the disk transition, both decodes.

Reads the per-seed sensitivity metrics written by
``run_food_transition_disk_sensitivity.py`` (``<prefix>_seed_metrics.csv`` plus the
matching ``<prefix>_points.csv``) for the flatten and unproject decodes, and draws,
for every one-at-a-time perturbation point, a seed-bootstrap distribution of the
stable-transition rate (box) alongside the observed rate (dot) and the baseline
(dashed line). The two decodes are faceted side by side; because each decode's
ladder is centred on its own validated baseline the perturbation values differ, so
each facet carries its own set of rows.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "bees-matplotlib-cache"))

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_hline,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
)

# Reuse the seed-bootstrap machinery from the shared OAT plotter.
import sys

sys.path.insert(0, str(ROOT / "experiments"))
from plot_oat_sensitivity_effects import bootstrap_rate_percentages  # noqa: E402

RESULTS = ROOT / "results"
REPORT_FIGURES = ROOT / "report" / "figures"
DEFAULT_OUTPUT = REPORT_FIGURES / "food_transition_disk_sensitivity_boxplot"

DECODES = (
    ("flatten", RESULTS / "food_transition_disk_sensitivity"),
    ("unproject", RESULTS / "food_transition_disk_unproject_sensitivity"),
)

# Order parameters top-to-bottom; baseline is pinned first.
PARAMETER_ORDER = [
    "food_site_count",
    "food_site_radius",
    "food_site_capacity",
    "food_site_max_distance",
    "travel_cost_per_distance",
    "vertical_comb_benefit",
    "mutation_sd",
    "transposition_mutation_correlation",
]
PARAMETER_LABELS = {
    "food_site_count": "food-site count",
    "food_site_radius": "patch radius",
    "food_site_capacity": "capacity",
    "food_site_max_distance": "max distance",
    "travel_cost_per_distance": "travel cost",
    "vertical_comb_benefit": "vertical benefit",
    "mutation_sd": "mutation scale",
    "transposition_mutation_correlation": "corr.",
}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)
    point_rows: list[dict] = []
    sample_rows: list[dict] = []
    baselines: dict[str, float] = {}
    order_keys: list[tuple[int, float, str]] = []
    for decode, prefix in DECODES:
        points = read_csv(Path(f"{prefix}_points.csv"))
        outcomes = stable_by_point(read_csv(Path(f"{prefix}_seed_metrics.csv")))
        baseline_pct, prows, srows, keys = decode_frames(
            decode, points, outcomes, args.bootstrap_samples, rng
        )
        baselines[decode] = baseline_pct
        point_rows.extend(prows)
        sample_rows.extend(srows)
        order_keys.extend(keys)

    write_figure(point_rows, sample_rows, baselines, order_keys, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260705)
    return parser.parse_args()


def decode_frames(
    decode: str,
    points: list[dict[str, str]],
    outcomes: dict[str, list[bool]],
    bootstrap_samples: int,
    rng: random.Random,
) -> tuple[float, list[dict], list[dict], list[str]]:
    meta = {row["point"]: row for row in points}
    baseline_point = next(p["point"] for p in points if p["is_baseline"] == "true")
    baseline_pct = 100 * sum(outcomes[baseline_point]) / len(outcomes[baseline_point])

    ordered = order_points(points, baseline_point)
    prows: list[dict] = []
    srows: list[dict] = []
    order_keys: list[tuple[int, float, str]] = []
    for point in ordered:
        is_baseline = point == baseline_point
        label = point_label(decode, meta[point], is_baseline)
        order_keys.append((*sort_key(meta[point], is_baseline), label))
        observed = 100 * sum(outcomes[point]) / len(outcomes[point])
        change = "baseline" if is_baseline else (
            "below" if observed < baseline_pct else "above"
        )
        prows.append(
            {"decode": decode, "label": label, "observed_percent": observed, "change": change}
        )
        for pct in bootstrap_rate_percentages(outcomes[point], bootstrap_samples, rng):
            srows.append(
                {"decode": decode, "label": label, "stable_percent": pct, "change": change}
            )
    return baseline_pct, prows, srows, order_keys


def sort_key(row: dict[str, str], is_baseline: bool) -> tuple[int, float]:
    """Global top-to-bottom order: baseline first, then by parameter group and
    value, so both decode facets read in the same parameter order."""
    if is_baseline:
        return (-1, 0.0)
    param = row["parameter"]
    group = PARAMETER_ORDER.index(param) if param in PARAMETER_ORDER else len(PARAMETER_ORDER)
    try:
        value = float(row["parameter_value"])
    except ValueError:
        value = 0.0
    return (group, value)


def order_points(points: list[dict[str, str]], baseline_point: str) -> list[str]:
    non_baseline = sorted(
        (p for p in points if p["point"] != baseline_point),
        key=lambda row: sort_key(row, False),
    )
    return [baseline_point] + [p["point"] for p in non_baseline]


def point_label(decode: str, row: dict[str, str], is_baseline: bool) -> str:
    if is_baseline:
        return "baseline"
    param = row["parameter"]
    plabel = PARAMETER_LABELS.get(param, param.replace("_", " "))
    return f"{plabel} = {format_value(row['parameter_value'])}"


def format_value(raw: str) -> str:
    value = float(raw)
    if value.is_integer():
        return str(int(value))
    if abs(value) < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def write_figure(
    point_rows: list[dict],
    sample_rows: list[dict],
    baselines: dict[str, float],
    order_keys: list[tuple[int, float, str]],
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    point_frame = pd.DataFrame(point_rows)
    sample_frame = pd.DataFrame(sample_rows)

    # One global category order by (parameter group, value) so both facets read in
    # the same parameter order; reverse so the first parameter sits at the top after
    # coord_flip. Labels shared across decodes carry the same key, so they collapse.
    key_by_label: dict[str, tuple[int, float]] = {}
    for group, value, label in order_keys:
        key_by_label.setdefault(label, (group, value))
    categories = [
        label for label, _ in sorted(key_by_label.items(), key=lambda kv: kv[1])
    ]
    categories = list(reversed(categories))
    for frame in (point_frame, sample_frame):
        frame["label"] = pd.Categorical(frame["label"], categories=categories, ordered=True)
        frame["decode"] = pd.Categorical(
            frame["decode"], categories=["flatten", "unproject"], ordered=True
        )

    baseline_frame = pd.DataFrame(
        {
            "decode": pd.Categorical(
                list(baselines), categories=["flatten", "unproject"], ordered=True
            ),
            "baseline_percent": list(baselines.values()),
        }
    )

    palette = {"below": "#d95f02", "baseline": "#6b7280", "above": "#1b9e77"}
    width = 9.6
    height = 6.6
    plot = (
        ggplot(sample_frame, aes(x="label", y="stable_percent", fill="change"))
        + geom_hline(
            aes(yintercept="baseline_percent"),
            data=baseline_frame,
            linetype="dashed",
            color="#6b7280",
            size=0.55,
        )
        + geom_boxplot(width=0.6, alpha=0.42, color="#4b5563", outlier_alpha=0.12)
        + geom_point(
            aes(x="label", y="observed_percent", color="change"),
            data=point_frame,
            inherit_aes=False,
            size=2.4,
            show_legend=False,
        )
        + facet_wrap("~decode", nrow=1, scales="free_y")
        + scale_fill_manual(
            values=palette,
            breaks=["below", "baseline", "above"],
            labels=["below baseline", "baseline", "above baseline"],
            name="",
        )
        + scale_color_manual(values=palette)
        + coord_flip()
        + labs(
            title="One-parameter sensitivity of the disk vertical transition",
            subtitle=(
                "Boxes show seed-bootstrap distributions; dots show observed rates; "
                "dashed line is each decode's baseline."
            ),
            x="",
            y="Stable vertical gravity-code outcomes (%)",
        )
        + theme_minimal(base_size=10)
        + theme(
            figure_size=(width, height),
            legend_position="top",
            panel_grid_major_y=element_blank(),
            panel_grid_minor=element_blank(),
            axis_text_y=element_text(size=8),
            axis_title_y=element_blank(),
            axis_title_x=element_text(margin={"t": 8}),
            strip_text=element_text(weight="bold", size=10),
            plot_title=element_text(weight="bold", size=12),
            plot_subtitle=element_text(size=9),
        )
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    plot.save(png_path, width=width, height=height, units="in", dpi=220, verbose=False)
    plot.save(
        pdf_path,
        width=width,
        height=height,
        units="in",
        verbose=False,
        metadata={"CreationDate": None, "ModDate": None},
    )
    print(f"wrote {relative(png_path)}", flush=True)
    print(f"wrote {relative(pdf_path)}", flush=True)


def stable_by_point(rows: list[dict[str, str]]) -> dict[str, list[bool]]:
    outcomes: dict[str, list[bool]] = {}
    for row in rows:
        outcomes.setdefault(row["candidate"], []).append(row["stable"] == "True")
    return outcomes


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
