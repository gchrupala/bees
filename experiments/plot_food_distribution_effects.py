"""Plot recruitment advantage across the v2 food-distribution sweeps.

Two panels, one per one-dimensional difficulty sweep (food-site count and patch
angular width). Each panel shows the per-seed in-run recruitment advantage
(dance-follower minus matched non-follower success rate, averaged over the final
generations) and the across-seed mean. Higher means the dance buys more.
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
    element_text,
    facet_wrap,
    geom_jitter,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_x_continuous,
    theme,
    theme_minimal,
)

RESULTS = ROOT / "results"
REPORT_FIGURES = ROOT / "report" / "figures"
DEFAULT_EVENTS = RESULTS / "food_distribution_v2_events.csv"
DEFAULT_OUTPUT = REPORT_FIGURES / "food_distribution_recruitment_advantage"

SWEEPS = {
    "site_count": {
        "x_column": "food_site_count",
        "title": "Food-site count (width 0.20)",
    },
    "patch_width": {
        "x_column": "food_site_width",
        "title": "Patch angular width (2 sites)",
    },
}


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events)
    seed_frame, mean_frame = build_frames(events)
    write_figure(seed_frame, mean_frame, args.output)


def build_frames(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = events[events["sweep"].isin(SWEEPS)].copy()
    rows["x_value"] = rows.apply(
        lambda row: row[SWEEPS[row["sweep"]]["x_column"]], axis=1
    )
    rows["panel"] = rows["sweep"].map(lambda sweep: SWEEPS[sweep]["title"])
    panel_order = [SWEEPS[name]["title"] for name in SWEEPS]
    rows["panel"] = pd.Categorical(rows["panel"], categories=panel_order, ordered=True)

    mean_frame = (
        rows.groupby(["panel", "x_value"], observed=True)["tail_recruitment_advantage"]
        .mean()
        .reset_index()
        .rename(columns={"tail_recruitment_advantage": "mean_advantage"})
    )
    return rows, mean_frame


def write_figure(
    seed_frame: pd.DataFrame,
    mean_frame: pd.DataFrame,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    width = 9.0
    height = 4.4

    plot = (
        ggplot(seed_frame, aes(x="x_value", y="tail_recruitment_advantage"))
        + geom_jitter(
            width=0.04,
            height=0.0,
            alpha=0.28,
            size=1.1,
            color="#6b7280",
        )
        + geom_line(
            aes(x="x_value", y="mean_advantage"),
            data=mean_frame,
            color="#1b9e77",
            size=1.0,
            inherit_aes=False,
        )
        + geom_point(
            aes(x="x_value", y="mean_advantage"),
            data=mean_frame,
            color="#1b9e77",
            size=3.0,
            inherit_aes=False,
        )
        + facet_wrap("panel", scales="free_x")
        + scale_x_continuous(expand=(0.08, 0))
        + labs(
            title="Where direct-pointing communication is favored",
            subtitle=(
                "In-run recruitment advantage on a flat comb, 50 seeds. "
                "Green = across-seed mean."
            ),
            x="ecological parameter value",
            y="recruitment advantage\n(follower − matched searcher success)",
        )
        + theme_minimal(base_size=10)
        + theme(
            figure_size=(width, height),
            plot_title=element_text(weight="bold", size=12),
            plot_subtitle=element_text(size=9),
            strip_text=element_text(size=10),
            axis_title_x=element_text(margin={"t": 8}),
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


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
