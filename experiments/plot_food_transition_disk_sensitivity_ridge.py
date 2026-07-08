"""Per-parameter sensitivity distributions for the disk transition.

For each decode (flatten, unproject) this writes a separate figure. For every
parameter it pools the seed-bootstrap stable-rate draws over that parameter's
(non-baseline) swept values into one horizontal violin, overlays a dot at each
swept value's observed rate (coloured below/above baseline), and marks the
decode's baseline as a dashed line. Parameters are sorted by worst-case drop
(shared order across decodes) and the two figures share an x-range, so the
separate panels stay directly comparable.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from math import floor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "bees-matplotlib-cache"))

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
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
OUTPUT_STEM = REPORT_FIGURES / "food_transition_disk_sensitivity_ridge"

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
    per_decode: dict[str, dict] = {}
    worst_by_param: dict[str, float] = {}
    global_min = 100.0
    for decode, prefix in DECODES:
        points = read_csv(Path(f"{prefix}_points.csv"))
        outcomes = stable_by_point(read_csv(Path(f"{prefix}_seed_metrics.csv")))
        data, worst, low = decode_data(decode, points, outcomes, args.bootstrap_samples, rng)
        per_decode[decode] = data
        global_min = min(global_min, low)
        for parameter, drop in worst.items():
            worst_by_param[parameter] = max(worst_by_param.get(parameter, 0.0), drop)

    order = sorted(worst_by_param, key=lambda p: worst_by_param[p])
    labels = [PARAMETER_LABELS.get(p, p) for p in order]
    lower = max(0.0, 5 * floor(global_min / 5) - 5)
    for decode, _ in DECODES:
        write_figure(decode, per_decode[decode], labels, lower, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_STEM)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260705)
    return parser.parse_args()


def decode_data(
    decode: str,
    points: list[dict[str, str]],
    outcomes: dict[str, list[bool]],
    bootstrap_samples: int,
    rng: random.Random,
) -> tuple[dict, dict[str, float], float]:
    baseline_point = next(p["point"] for p in points if p["is_baseline"] == "true")
    baseline_rate = rate(outcomes[baseline_point])

    by_param: dict[str, list[str]] = {}
    for row in points:
        if row["point"] == baseline_point:
            continue
        by_param.setdefault(row["parameter"], []).append(row["point"])

    draw_rows: list[dict] = []
    dot_rows: list[dict] = []
    worst: dict[str, float] = {}
    low = baseline_rate
    for parameter, pts in by_param.items():
        plabel = PARAMETER_LABELS.get(parameter, parameter)
        value_rates = []
        for point in pts:
            r = rate(outcomes[point])
            value_rates.append(r)
            dot_rows.append(
                {
                    "label": plabel,
                    "rate": r,
                    "change": "below" if r < baseline_rate else "above",
                }
            )
            for draw in bootstrap_rate_percentages(outcomes[point], bootstrap_samples, rng):
                draw_rows.append({"label": plabel, "rate": draw})
                low = min(low, draw)
        worst[parameter] = baseline_rate - min(value_rates)
    return {"draws": draw_rows, "dots": dot_rows, "baseline": baseline_rate}, worst, low


def write_figure(
    decode: str,
    data: dict,
    labels: list[str],
    lower: float,
    output_stem: Path,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    draws = pd.DataFrame(data["draws"])
    dots = pd.DataFrame(data["dots"])
    for frame in (draws, dots):
        frame["label"] = pd.Categorical(frame["label"], categories=labels, ordered=True)

    palette = {"below": "#d95f02", "above": "#1b9e77"}
    plot = (
        ggplot(draws, aes(x="label", y="rate"))
        + geom_violin(fill="#c6dbef", color="#6b7280", width=0.9, alpha=0.7, scale="width")
        + geom_hline(yintercept=data["baseline"], linetype="dashed", color="#374151", size=0.5)
        + geom_point(
            aes(x="label", y="rate", color="change"),
            data=dots,
            inherit_aes=False,
            size=2.3,
            alpha=0.9,
        )
        + coord_flip(ylim=(lower, 100))
        + scale_color_manual(
            values=palette,
            breaks=["below", "above"],
            labels=["below baseline", "above baseline"],
            name="",
        )
        + labs(
            title=f"Per-parameter sensitivity: {decode} decode",
            x="",
            y="Stable vertical gravity-code outcomes (%)",
        )
        + theme_minimal(base_size=13)
        + theme(
            figure_size=(6.6, 4.2),
            legend_position="top",
            panel_grid_major_y=element_blank(),
            panel_grid_minor=element_blank(),
            axis_text_y=element_text(size=12),
            axis_title_x=element_text(size=13, margin={"t": 8}),
            plot_title=element_text(weight="bold", size=13),
            legend_text=element_text(size=12),
        )
    )
    png_path = Path(f"{output_stem}_{decode}.png")
    pdf_path = Path(f"{output_stem}_{decode}.pdf")
    plot.save(png_path, width=6.6, height=4.2, units="in", dpi=220, verbose=False)
    plot.save(
        pdf_path,
        width=6.6,
        height=4.2,
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
