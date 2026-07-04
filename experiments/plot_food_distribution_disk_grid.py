"""Grid heatmaps of the disk-geometry food-distribution matrix.

Reads the per-cell group summary from the disk food-distribution experiment and
renders recruitment advantage and evolved directional bias as heatmaps over the
mean-site-count x median-patch-radius grid (see ``run_food_distribution_disk``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_identity,
    scale_fill_cmap,
    theme,
    theme_minimal,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "food_distribution_disk_group_summary.csv"
DEFAULT_OUTPUT = ROOT / "report" / "figures" / "food_distribution_disk_grid.png"

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
    # Per-metric min-max so the text stays legible on both ends of viridis.
    span = long.groupby("metric", observed=True)["value"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    )
    long["text_color"] = ["black" if x > 0.55 else "white" for x in span]

    plot = (
        ggplot(long, aes("radius_label", "count_label", fill="value"))
        + geom_tile()
        + geom_text(aes(label="label", color="text_color"), size=7)
        + facet_wrap("metric")
        + scale_fill_cmap(cmap_name="viridis")
        + scale_color_identity()
        + labs(
            x="median patch radius (m)",
            y="mean site count",
            fill="value",
            title="Disk-geometry food-distribution matrix (flat comb, 50 seeds/cell)",
        )
        + theme_minimal()
        + theme(
            figure_size=(11, 5),
            plot_title=element_text(size=12),
            strip_text=element_text(size=11),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    plot.save(output, dpi=180, verbose=False)
    plot.save(output.with_suffix(".pdf"), verbose=False)
    print(f"wrote {output} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
