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
    facet_wrap,
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
GROUP_SUMMARY = RESULTS / "food_transition_evolutionary_interaction_group_summary.csv"
DEFAULT_OUTPUT = REPORT_FIGURES / "evolutionary_interaction_stable_heatmap"


def main() -> None:
    args = parse_args()
    frame = load_grid(args.group_summary)
    write_figure(frame, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the evolutionary-parameter interaction grid as stable-transition "
            "rate heatmaps."
        )
    )
    parser.add_argument(
        "--group-summary",
        type=Path,
        default=GROUP_SUMMARY,
        help="Evolutionary interaction group-summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path prefix. The script writes .png and .pdf.",
    )
    return parser.parse_args()


def load_grid(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    grid = rows[(rows["group_kind"] == "all") & (rows["group"] == "all")].copy()
    if grid.empty:
        raise SystemExit(f"no all-run summaries found in {path}")

    grid["stable_percent"] = 100 * grid["stable_fraction"].astype(float)
    grid["percent_label"] = grid["stable_percent"].round(0).astype(int).astype(str) + "%"
    grid["text_color"] = grid["stable_percent"].map(
        lambda value: "white" if value >= 58 else "#111827"
    )

    alpha_values = sorted(grid["vertical_comb_benefit"].astype(float).unique())
    mutation_values = sorted(grid["mutation_sd"].astype(float).unique())
    rho_values = sorted(
        grid["transposition_mutation_correlation"].astype(float).unique()
    )
    expected_points = len(alpha_values) * len(mutation_values) * len(rho_values)
    if len(grid) != expected_points:
        raise SystemExit(
            f"expected {expected_points} grid rows from parameter levels, found {len(grid)}"
        )

    grid["vertical_label"] = pd.Categorical(
        grid["vertical_comb_benefit"].astype(float).map(
            lambda value: f"vertical benefit = {format_value(value)}"
        ),
        categories=[
            f"vertical benefit = {format_value(value)}" for value in alpha_values
        ],
        ordered=True,
    )
    grid["mutation_label"] = pd.Categorical(
        grid["mutation_sd"].astype(float).map(format_value),
        categories=[format_value(value) for value in mutation_values],
        ordered=True,
    )
    grid["rho_label"] = pd.Categorical(
        grid["transposition_mutation_correlation"].astype(float).map(format_value),
        categories=[format_value(value) for value in rho_values],
        ordered=True,
    )
    return grid


def write_figure(frame: pd.DataFrame, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    width = 9.0
    height = 3.9
    plot = (
        ggplot(frame, aes(x="rho_label", y="mutation_label", fill="stable_percent"))
        + geom_tile(color="white", size=0.8)
        + geom_text(
            aes(label="percent_label", color="text_color"),
            size=8.5,
            fontweight="bold",
            show_legend=False,
        )
        + facet_wrap("~vertical_label", nrow=1)
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
            title="Stable transition rate across evolutionary parameter interactions",
            subtitle="Each cell summarizes 100 held-out seeds; labels show percent stable.",
            x="Sender-receiver mutation correlation",
            y="Mutation scale",
        )
        + theme_minimal(base_size=10)
        + theme(
            figure_size=(width, height),
            legend_position="right",
            panel_grid=element_blank(),
            panel_spacing=0.08,
            strip_text=element_text(weight="bold", size=9),
            axis_text_x=element_text(size=8),
            axis_text_y=element_text(size=8),
            axis_title_x=element_text(margin={"t": 8}),
            axis_title_y=element_text(margin={"r": 8}),
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


def format_value(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
