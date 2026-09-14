"""Heatmap of the disk-ecology evolutionary-interaction grid, both decodes.

Reads the per-cell interaction summaries written by
``run_food_transition_disk_interaction.py`` for the flatten and unproject decodes
and renders stable-transition rate as tiles over the sender--receiver correlation
(x) against the mutation scale (y), faceted by decode (rows) and vertical-comb
benefit (columns). Unlike the angular heatmap script this one works in the disk
schema (``stable_count``/``seed_count`` rather than a pre-computed
``stable_fraction``) and folds the two decodes into one figure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "bees-matplotlib-cache"))

import pandas as pd
from plotnine import (
    aes,
    coord_fixed,
    element_blank,
    element_text,
    facet_grid,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_identity,
    scale_fill_gradientn,
    scale_x_discrete,
    scale_y_discrete,
    theme,
    theme_minimal,
)

RESULTS = ROOT / "results"
REPORT_FIGURES = ROOT / "report" / "figures"
DEFAULT_FLATTEN = RESULTS / "food_transition_disk_interaction.csv"
DEFAULT_UNPROJECT = RESULTS / "food_transition_disk_unproject_interaction.csv"
DEFAULT_OUTPUT = REPORT_FIGURES / "food_transition_disk_interaction_heatmap"


def main() -> None:
    args = parse_args()
    frames = []
    for decode, path in (("flatten", args.flatten), ("unproject", args.unproject)):
        frames.append(load_grid(path, decode))
    frame = pd.concat(frames, ignore_index=True)
    frame = add_labels(frame)
    write_figure(frame, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flatten", type=Path, default=DEFAULT_FLATTEN)
    parser.add_argument("--unproject", type=Path, default=DEFAULT_UNPROJECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_grid(path: Path, decode: str) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows["decode"] = decode
    rows["stable_percent"] = 100 * rows["stable_count"].astype(float) / rows[
        "seed_count"
    ].astype(float)
    return rows


def add_labels(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid.copy()
    grid["percent_label"] = grid["stable_percent"].round(0).astype(int).astype(str)
    grid["text_color"] = grid["stable_percent"].map(
        lambda value: "white" if value >= 58 else "#111827"
    )

    benefit_values = sorted(grid["vertical_comb_benefit"].astype(float).unique())
    mutation_values = sorted(grid["mutation_sd"].astype(float).unique())
    rho_values = sorted(grid["transposition_mutation_correlation"].astype(float).unique())

    grid["vertical_label"] = pd.Categorical(
        grid["vertical_comb_benefit"].astype(float).map(
            lambda value: f"benefit = {format_value(value)}"
        ),
        categories=[f"benefit = {format_value(v)}" for v in benefit_values],
        ordered=True,
    )
    grid["decode_label"] = pd.Categorical(
        grid["decode"], categories=["flatten", "unproject"], ordered=True
    )
    grid["mutation_label"] = pd.Categorical(
        grid["mutation_sd"].astype(float).map(format_value),
        categories=[format_value(v) for v in mutation_values],
        ordered=True,
    )
    grid["rho_label"] = pd.Categorical(
        grid["transposition_mutation_correlation"].astype(float).map(format_value),
        categories=[format_value(v) for v in rho_values],
        ordered=True,
    )
    return grid


def write_figure(frame: pd.DataFrame, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    width = 10.5
    height = 5.2
    plot = (
        ggplot(frame, aes(x="rho_label", y="mutation_label", fill="stable_percent"))
        + geom_tile(color="white", size=0.8)
        + geom_text(
            aes(label="percent_label", color="text_color"),
            size=8.5,
            fontweight="bold",
            show_legend=False,
        )
        + facet_grid("decode_label ~ vertical_label")
        + scale_fill_gradientn(
            colors=["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
            limits=(0, 100),
            breaks=[0, 25, 50, 75, 100],
            labels=["0", "25", "50", "75", "100"],
            name="Stable\ntransition (%)",
        )
        + scale_color_identity()
        + scale_x_discrete(drop=False)
        + scale_y_discrete(drop=False)
        + coord_fixed()
        + labs(
            title="Stable transition rate across evolutionary-parameter interactions",
            x="Sender-receiver mutation correlation",
            y="Mutation scale",
        )
        + theme_minimal(base_size=15)
        + theme(
            figure_size=(width, height),
            legend_position="right",
            panel_grid=element_blank(),
            panel_spacing=0.08,
            strip_text=element_text(weight="bold", size=14),
            axis_text_x=element_text(size=12),
            axis_text_y=element_text(size=12),
            axis_title_x=element_text(size=15, margin={"t": 8}),
            axis_title_y=element_text(size=15, margin={"r": 8}),
            plot_title=element_text(weight="bold", size=17),
            legend_title=element_text(size=13),
            legend_text=element_text(size=12),
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


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
