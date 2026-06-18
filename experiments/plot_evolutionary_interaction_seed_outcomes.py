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
    geom_point,
    geom_tile,
    ggplot,
    guides,
    guide_legend,
    labs,
    scale_color_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)


RESULTS = ROOT / "results"
REPORT_FIGURES = ROOT / "report" / "figures"
EVENTS = RESULTS / "food_transition_evolutionary_interaction_events.csv"
DEFAULT_OUTPUT = REPORT_FIGURES / "evolutionary_interaction_seed_outcomes_binary"

OUTCOME_COLORS = {
    "stable transition": "#111111",
    "other outcome": "#d7d7d7",
}


def main() -> None:
    args = parse_args()
    seed_frame, cell_frame = load_seed_outcomes(args.events)
    write_figure(seed_frame, cell_frame, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot every held-out seed inside each evolutionary interaction grid "
            "cell, colored by transition outcome."
        )
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=EVENTS,
        help="Evolutionary interaction per-seed event CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path prefix. The script writes .png and .pdf.",
    )
    return parser.parse_args()


def load_seed_outcomes(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(path)
    required = {
        "point",
        "seed",
        "stable_vertical_gravity",
        "vertical_comb_benefit",
        "mutation_sd",
        "transposition_mutation_correlation",
    }
    missing = required - set(events.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")

    alpha_values = sorted(events["vertical_comb_benefit"].astype(float).unique())
    mutation_values = sorted(events["mutation_sd"].astype(float).unique())
    rho_values = sorted(
        events["transposition_mutation_correlation"].astype(float).unique()
    )
    seeds = sorted(events["seed"].astype(int).unique())
    expected_rows = len(alpha_values) * len(mutation_values) * len(rho_values) * len(seeds)
    if len(events) != expected_rows:
        raise SystemExit(f"expected {expected_rows} event rows, found {len(events)}")

    alpha_lookup = {
        value: f"vertical benefit = {format_value(value)}" for value in alpha_values
    }
    mutation_lookup = {value: index for index, value in enumerate(mutation_values)}
    rho_lookup = {value: index for index, value in enumerate(rho_values)}
    seed_lookup = {seed: index for index, seed in enumerate(seeds)}

    events["vertical_label"] = pd.Categorical(
        events["vertical_comb_benefit"].astype(float).map(alpha_lookup),
        categories=[alpha_lookup[value] for value in alpha_values],
        ordered=True,
    )
    events["mutation_index"] = events["mutation_sd"].astype(float).map(mutation_lookup)
    events["rho_index"] = (
        events["transposition_mutation_correlation"].astype(float).map(rho_lookup)
    )
    events["seed_index"] = events["seed"].astype(int).map(seed_lookup)
    stable = events["stable_vertical_gravity"].astype(str).str.lower() == "true"
    events["outcome_label"] = pd.Categorical(
        stable.map(lambda value: "stable transition" if value else "other outcome"),
        categories=list(OUTCOME_COLORS),
        ordered=True,
    )

    seed_columns = 10
    seed_spacing = 0.08
    events["seed_col"] = events["seed_index"] % seed_columns
    events["seed_row"] = events["seed_index"] // seed_columns
    events["plot_x"] = events["rho_index"] + (events["seed_col"] - 4.5) * seed_spacing
    events["plot_y"] = events["mutation_index"] + (4.5 - events["seed_row"]) * seed_spacing

    cell_frame = (
        events[
            [
                "vertical_label",
                "mutation_index",
                "rho_index",
                "mutation_sd",
                "transposition_mutation_correlation",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    cell_frame["plot_x"] = cell_frame["rho_index"]
    cell_frame["plot_y"] = cell_frame["mutation_index"]

    events.attrs["rho_breaks"] = list(range(len(rho_values)))
    events.attrs["rho_labels"] = [format_value(value) for value in rho_values]
    events.attrs["mutation_breaks"] = list(range(len(mutation_values)))
    events.attrs["mutation_labels"] = [format_value(value) for value in mutation_values]
    events.attrs["x_limits"] = (-0.5, len(rho_values) - 0.5)
    events.attrs["y_limits"] = (-0.5, len(mutation_values) - 0.5)
    return events, cell_frame


def write_figure(
    seed_frame: pd.DataFrame,
    cell_frame: pd.DataFrame,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    width = 10.4
    height = 4.4
    plot = (
        ggplot(seed_frame, aes(x="plot_x", y="plot_y"))
        + geom_tile(
            aes(x="plot_x", y="plot_y"),
            data=cell_frame,
            inherit_aes=False,
            width=0.98,
            height=0.98,
            fill="#f8fafc",
            color="#d1d5db",
            size=0.35,
        )
        + geom_point(
            aes(color="outcome_label"),
            size=1.05,
            alpha=1,
            stroke=0,
        )
        + facet_wrap("~vertical_label", nrow=1)
        + scale_color_manual(
            values=OUTCOME_COLORS,
            breaks=list(OUTCOME_COLORS),
            name="Outcome",
        )
        + scale_x_continuous(
            breaks=seed_frame.attrs["rho_breaks"],
            labels=seed_frame.attrs["rho_labels"],
            limits=seed_frame.attrs["x_limits"],
            expand=(0, 0),
        )
        + scale_y_continuous(
            breaks=seed_frame.attrs["mutation_breaks"],
            labels=seed_frame.attrs["mutation_labels"],
            limits=seed_frame.attrs["y_limits"],
            expand=(0, 0),
        )
        + coord_fixed()
        + labs(
            title="Seed-level outcomes across evolutionary parameter interactions",
            subtitle=(
                "Each dot is one held-out seed; black dots are stable transitions."
            ),
            x="Sender-receiver mutation correlation",
            y="Mutation scale",
        )
        + guides(color=guide_legend(override_aes={"size": 3.2, "alpha": 1}))
        + theme_minimal(base_size=10)
        + theme(
            figure_size=(width, height),
            legend_position="right",
            legend_title=element_text(weight="bold"),
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
