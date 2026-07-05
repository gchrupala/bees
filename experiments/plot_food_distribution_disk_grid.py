"""Grid heatmaps of the disk-geometry food-distribution matrix.

Reads the per-cell group summary from the disk food-distribution experiment and
renders recruitment advantage and evolved directional bias as heatmaps over the
mean-site-count x median-patch-radius grid (see ``run_food_distribution_disk``).

Recruitment advantage is the tail-window difference between dance-follower and
matched-searcher success rates. Styling (blue sequential fill, white tile
borders, bold labels, square tiles) matches the evolutionary-parameter
interaction heatmap (see ``plot_evolutionary_interaction_heatmap``). The two
panels are different quantities on different scales, so each is shaded on its
own min-max range with the true value printed in every cell; a single continuous
legend would not be comparable across panels, so it is omitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "food_distribution_disk_group_summary.csv"
DEFAULT_OUTPUT = ROOT / "report" / "figures" / "food_distribution_disk_grid.png"

# Blue sequential palette shared with the interaction heatmap.
BLUES = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]

METRICS = {
    "mean_tail_recruitment_advantage": "Recruitment advantage",
    "mean_final_directional_bias": "Evolved directional bias",
}


def main() -> None:
    summary = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SUMMARY
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    frame = pd.read_csv(summary)
    frame["count"] = frame["food_site_count"].astype(int)
    frame["radius"] = frame["food_site_radius"].astype(float)

    counts = sorted(frame["count"].unique())
    radii = sorted(frame["radius"].unique())
    frame["count_label"] = pd.Categorical(
        frame["count"].astype(str), categories=[str(c) for c in counts], ordered=True
    )
    frame["radius_label"] = pd.Categorical(
        frame["radius"].map(lambda r: f"{r:g}"),
        categories=[f"{r:g}" for r in radii],
        ordered=True,
    )

    long = frame.melt(
        id_vars=["count_label", "radius_label"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )
    long["metric"] = pd.Categorical(
        long["metric"].map(METRICS), categories=list(METRICS.values()), ordered=True
    )
    long["label"] = long["value"].map(lambda v: f"{v:.2f}")

    # The two panels are on different scales, so shade each on its own min-max
    # range and print the true value; a shared continuous legend would not be
    # comparable across panels, so it is omitted.
    long["fill_norm"] = long.groupby("metric", observed=True)["value"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    )
    long["text_color"] = ["white" if x >= 0.5 else "#111827" for x in long["fill_norm"]]

    width = 11.0
    height = 7.0
    plot = (
        ggplot(long, aes("radius_label", "count_label", fill="fill_norm"))
        + geom_tile(color="white", size=0.8)
        + geom_text(
            aes(label="label", color="text_color"),
            size=7.5,
            fontweight="bold",
            show_legend=False,
        )
        + facet_wrap("~metric", nrow=1)
        + scale_fill_gradientn(colors=BLUES, limits=(0, 1))
        + scale_color_identity()
        + scale_x_discrete(drop=False)
        + scale_y_discrete(drop=False)
        + coord_fixed()
        + labs(
            x="median patch radius (m)",
            y="mean site count",
            title="Disk-geometry food-distribution matrix",
            subtitle=(
                "Flat comb, 50 seeds/cell. Recruitment advantage is the "
                "follower−searcher success difference; shading normalized per panel."
            ),
        )
        + theme_minimal(base_size=10)
        + theme(
            figure_size=(width, height),
            legend_position="none",
            panel_grid=element_blank(),
            panel_spacing=0.08,
            strip_text=element_text(weight="bold", size=10),
            axis_text_x=element_text(size=8),
            axis_text_y=element_text(size=8),
            axis_title_x=element_text(margin={"t": 8}),
            axis_title_y=element_text(margin={"r": 8}),
            plot_title=element_text(weight="bold", size=12),
            plot_subtitle=element_text(size=9),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    plot.save(output, width=width, height=height, units="in", dpi=180, verbose=False)
    plot.save(
        output.with_suffix(".pdf"), width=width, height=height, units="in", verbose=False
    )
    print(f"wrote {output} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
