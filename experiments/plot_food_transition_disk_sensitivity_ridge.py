"""Per-parameter sensitivity distributions for the disk transition (prototype).

Keeps the eight-parameter reduction of the tornado but shows the effect
distribution directly instead of a bar+whisker summary. For each parameter it
pools the seed-bootstrap stable-rate draws over that parameter's (non-baseline)
swept values into one horizontal violin, overlays a dot at each swept value's
observed rate (coloured below/above baseline), and marks the decode's baseline as
a dashed line. Parameters are sorted by worst-case drop so the biggest levers sit
on top; both decodes share the parameter axis.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
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
    geom_hline,
    geom_point,
    geom_violin,
    ggplot,
    labs,
    scale_color_manual,
    theme,
    theme_minimal,
)

sys.path.insert(0, str(ROOT / "experiments"))
from plot_oat_sensitivity_effects import bootstrap_rate_percentages  # noqa: E402

RESULTS = ROOT / "results"
REPORT_FIGURES = ROOT / "report" / "figures"
OUTPUT = REPORT_FIGURES / "food_transition_disk_sensitivity_ridge"

DECODES = (
    ("flatten", RESULTS / "food_transition_disk_sensitivity"),
    ("unproject", RESULTS / "food_transition_disk_unproject_sensitivity"),
)

PARAMETER_LABELS = {
    "food_site_count": "food-site count",
    "food_site_radius": "patch radius",
    "food_site_capacity": "capacity",
    "food_site_max_distance": "max distance",
    "travel_cost_per_distance": "travel cost",
    "vertical_comb_benefit": "vertical benefit",
    "mutation_sd": "mutation scale",
    "transposition_mutation_correlation": "sender-receiver corr.",
}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)
    draw_rows: list[dict] = []
    dot_rows: list[dict] = []
    baseline_rows: list[dict] = []
    worst_by_param: dict[str, float] = {}
    for decode, prefix in DECODES:
        points = read_csv(Path(f"{prefix}_points.csv"))
        outcomes = stable_by_point(read_csv(Path(f"{prefix}_seed_metrics.csv")))
        baseline_point = next(p["point"] for p in points if p["is_baseline"] == "true")
        baseline_rate = rate(outcomes[baseline_point])
        baseline_rows.append({"decode": decode, "baseline_rate": baseline_rate})

        by_param: dict[str, list[str]] = {}
        for row in points:
            if row["point"] == baseline_point:
                continue
            by_param.setdefault(row["parameter"], []).append(row["point"])

        for parameter, pts in by_param.items():
            plabel = PARAMETER_LABELS.get(parameter, parameter)
            value_rates = []
            for point in pts:
                r = rate(outcomes[point])
                value_rates.append(r)
                dot_rows.append(
                    {
                        "decode": decode,
                        "label": plabel,
                        "rate": r,
                        "change": "below" if r < baseline_rate else "above",
                    }
                )
                for draw in bootstrap_rate_percentages(
                    outcomes[point], args.bootstrap_samples, rng
                ):
                    draw_rows.append({"decode": decode, "label": plabel, "rate": draw})
            worst_by_param[parameter] = max(
                worst_by_param.get(parameter, 0.0), baseline_rate - min(value_rates)
            )

    order = sorted(worst_by_param, key=lambda p: worst_by_param[p])
    labels = [PARAMETER_LABELS.get(p, p) for p in order]
    write_figure(draw_rows, dot_rows, baseline_rows, labels, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260705)
    return parser.parse_args()


def write_figure(
    draw_rows: list[dict],
    dot_rows: list[dict],
    baseline_rows: list[dict],
    labels: list[str],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    draws = pd.DataFrame(draw_rows)
    dots = pd.DataFrame(dot_rows)
    baselines = pd.DataFrame(baseline_rows)
    for frame in (draws, dots):
        frame["label"] = pd.Categorical(frame["label"], categories=labels, ordered=True)
        frame["decode"] = pd.Categorical(
            frame["decode"], categories=["flatten", "unproject"], ordered=True
        )
    baselines["decode"] = pd.Categorical(
        baselines["decode"], categories=["flatten", "unproject"], ordered=True
    )

    palette = {"below": "#d95f02", "above": "#1b9e77"}
    plot = (
        ggplot(draws, aes(x="label", y="rate"))
        + geom_violin(
            fill="#c6dbef", color="#6b7280", width=0.9, alpha=0.7, scale="width"
        )
        + geom_hline(
            aes(yintercept="baseline_rate"),
            data=baselines,
            linetype="dashed",
            color="#374151",
            size=0.5,
        )
        + geom_point(
            aes(x="label", y="rate", color="change"),
            data=dots,
            inherit_aes=False,
            size=2.1,
            alpha=0.9,
        )
        + coord_flip()
        + facet_wrap("~decode", nrow=1)
        + scale_color_manual(
            values=palette,
            breaks=["below", "above"],
            labels=["below baseline", "above baseline"],
            name="",
        )
        + labs(
            title="Per-parameter sensitivity of the disk vertical transition",
            subtitle=(
                "Violin: seed-bootstrap stable-rate density over each parameter's "
                "swept values.  Dots: per-value rates.  Dashed: baseline."
            ),
            x="",
            y="Stable vertical gravity-code outcomes (%)",
        )
        + theme_minimal(base_size=10)
        + theme(
            figure_size=(9.4, 4.6),
            legend_position="top",
            panel_grid_major_y=element_blank(),
            panel_grid_minor=element_blank(),
            axis_text_y=element_text(size=9),
            axis_title_x=element_text(margin={"t": 8}),
            strip_text=element_text(weight="bold", size=10),
            plot_title=element_text(weight="bold", size=12),
            plot_subtitle=element_text(size=9),
        )
    )
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    plot.save(png_path, width=9.4, height=4.6, units="in", dpi=220, verbose=False)
    plot.save(
        pdf_path,
        width=9.4,
        height=4.6,
        units="in",
        verbose=False,
        metadata={"CreationDate": None, "ModDate": None},
    )
    print(f"wrote {relative(png_path)}", flush=True)
    print(f"wrote {relative(pdf_path)}", flush=True)


def rate(outcomes: list[bool]) -> float:
    return 100 * sum(outcomes) / len(outcomes)


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
