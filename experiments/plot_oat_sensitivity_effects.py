from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "bees-matplotlib-cache"))

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    geom_boxplot,
    geom_hline,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_y_continuous,
    theme,
    theme_minimal,
)


RESULTS = ROOT / "results"
REPORT_FIGURES = ROOT / "report" / "figures"
EVENTS = RESULTS / "food_transition_oat_sensitivity_events.csv"
POINTS = RESULTS / "food_transition_oat_sensitivity_points.csv"
DEFAULT_OUTPUT = REPORT_FIGURES / "oat_sensitivity_stable_delta"


PARAMETER_LABELS = {
    "vertical_comb_benefit": "vertical benefit",
    "food_site_max_distance": "max distance",
    "transposition_mutation_correlation": "transposition corr.",
    "mutation_sd": "mutation sd",
    "food_site_width": "site width",
    "food_site_count": "site count",
    "travel_cost_per_distance": "travel cost",
}


@dataclass(frozen=True)
class Effect:
    point: str
    label: str
    stable_count: int
    runs: int
    delta: float
    bootstrap_percentages: tuple[float, ...]
    is_baseline: bool = False

    @property
    def stable_percent(self) -> float:
        return 100 * self.stable_count / self.runs


def main() -> None:
    args = parse_args()
    effects, baseline_count, runs = compute_effects(
        points_path=args.points,
        events_path=args.events,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    write_figure(effects, baseline_count, runs, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot OAT sensitivity stable-transition rates as bootstrap boxplots."
        )
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=POINTS,
        help="CSV describing OAT parameter points.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=EVENTS,
        help="Per-seed OAT event CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path prefix. The script writes .png and .pdf.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Seed-bootstrap resamples for boxplot distributions.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260617,
        help="Random seed for deterministic bootstrap boxplots.",
    )
    return parser.parse_args()


def compute_effects(
    points_path: Path,
    events_path: Path,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[list[Effect], int, int]:
    points = read_csv(points_path)
    events = read_csv(events_path)
    baseline_point = find_baseline_point(points)
    point_order = {row["point"]: index for index, row in enumerate(points)}
    point_metadata = {row["point"]: row for row in points}
    stable_by_point = stable_outcomes_by_point(events)
    baseline_outcomes = stable_by_point[baseline_point]
    baseline_seeds = set(baseline_outcomes)
    runs = len(baseline_outcomes)
    rng = random.Random(random_seed)

    effects = [
        Effect(
            point=baseline_point,
            label="baseline",
            stable_count=sum(baseline_outcomes.values()),
            runs=runs,
            delta=0.0,
            bootstrap_percentages=bootstrap_rate_percentages(
                list(baseline_outcomes.values()),
                samples=bootstrap_samples,
                rng=rng,
            ),
            is_baseline=True,
        )
    ]
    for point, outcomes in stable_by_point.items():
        if point == baseline_point:
            continue
        if set(outcomes) != baseline_seeds:
            raise SystemExit(
                f"{point} does not have the same seed panel as {baseline_point}"
            )
        seed_deltas = [
            int(outcomes[seed]) - int(baseline_outcomes[seed])
            for seed in sorted(baseline_seeds)
        ]
        delta = sum(seed_deltas) / runs
        bootstrap_percentages = bootstrap_rate_percentages(
            list(outcomes.values()),
            samples=bootstrap_samples,
            rng=rng,
        )
        metadata = point_metadata[point]
        effects.append(
            Effect(
                point=point,
                label=format_point_label(metadata),
                stable_count=sum(outcomes.values()),
                runs=runs,
                delta=delta,
                bootstrap_percentages=bootstrap_percentages,
            )
        )

    effects.sort(
        key=lambda effect: (
            effect.delta,
            0 if effect.is_baseline else 1,
            point_order[effect.point],
        )
    )
    return effects, sum(baseline_outcomes.values()), runs


def find_baseline_point(points: list[dict[str, str]]) -> str:
    baselines = [row["point"] for row in points if row["is_baseline"] == "true"]
    if len(baselines) != 1:
        raise SystemExit(f"expected one baseline point, found {len(baselines)}")
    return baselines[0]


def stable_outcomes_by_point(
    rows: list[dict[str, str]],
) -> dict[str, dict[int, bool]]:
    outcomes: dict[str, dict[int, bool]] = {}
    for row in rows:
        point = row["point"]
        seed = int(row["seed"])
        if seed in outcomes.setdefault(point, {}):
            raise SystemExit(f"duplicate event row for point={point}, seed={seed}")
        outcomes[point][seed] = row["stable_vertical_gravity"] == "true"
    return outcomes


def bootstrap_rate_percentages(
    outcomes: list[bool],
    samples: int,
    rng: random.Random,
) -> tuple[float, ...]:
    n = len(outcomes)
    observed_percent = 100 * sum(outcomes) / n
    if samples <= 0:
        return (observed_percent,)

    return tuple(
        100 * sum(outcomes[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )


def format_point_label(row: dict[str, str]) -> str:
    parameter = row["parameter"]
    label = PARAMETER_LABELS.get(parameter, parameter.replace("_", " "))
    return f"{label} = {format_value(row['parameter_value'])}"


def format_value(raw: str) -> str:
    value = float(raw)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def write_figure(
    effects: list[Effect],
    baseline_count: int,
    runs: int,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    point_frame, sample_frame, x_limits, x_breaks = plot_frames(effects)
    baseline_percent = 100 * baseline_count / runs
    width = 8.4
    height = max(5.2, 0.34 * len(effects) + 1.6)

    plot = (
        ggplot(sample_frame, aes(x="label", y="stable_percent", fill="change"))
        + geom_hline(
            yintercept=baseline_percent,
            linetype="dashed",
            color="#6b7280",
            size=0.55,
        )
        + geom_boxplot(
            width=0.58,
            alpha=0.42,
            color="#4b5563",
            outlier_alpha=0.12,
        )
        + geom_point(
            aes(x="label", y="observed_percent", color="change"),
            data=point_frame,
            inherit_aes=False,
            size=2.6,
            show_legend=False,
        )
        + scale_fill_manual(
            values={
                "below": "#d95f02",
                "baseline": "#6b7280",
                "above": "#1b9e77",
            },
            breaks=["below", "baseline", "above"],
            labels=["below baseline", "baseline", "above baseline"],
            name="",
        )
        + scale_color_manual(
            values={
                "below": "#d95f02",
                "baseline": "#6b7280",
                "above": "#1b9e77",
            },
        )
        + scale_y_continuous(breaks=x_breaks)
        + coord_flip(ylim=x_limits)
        + labs(
            title="Stable transition rate under one-parameter perturbations",
            subtitle=(
                f"Baseline is {baseline_percent:.1f}% stable. "
                "Boxes show bootstrap distributions; points show observed rates."
            ),
            x="",
            y="Stable vertical gravity-code outcomes (%)",
        )
        + theme_minimal(base_size=10)
        + theme(
            figure_size=(width, height),
            legend_position="top",
            legend_box_spacing=0.02,
            panel_grid_major_y=element_blank(),
            panel_grid_minor=element_blank(),
            axis_text_y=element_text(size=9),
            axis_title_y=element_blank(),
            axis_title_x=element_text(margin={"t": 8}),
            legend_title=element_blank(),
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


def plot_frames(
    effects: list[Effect],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float], list[int]]:
    point_rows = []
    sample_rows = []
    for effect in effects:
        change = effect_change(effect)
        point_rows.append(
            {
                "label": effect.label,
                "observed_percent": effect.stable_percent,
                "change": change,
            }
        )
        sample_rows.extend(
            {
                "label": effect.label,
                "stable_percent": stable_percent,
                "change": change,
            }
            for stable_percent in effect.bootstrap_percentages
        )

    point_frame = pd.DataFrame(point_rows)
    sample_frame = pd.DataFrame(sample_rows)
    labels = [row["label"] for row in point_rows]
    categories = list(reversed(labels))
    point_frame["label"] = pd.Categorical(
        point_frame["label"],
        categories=categories,
        ordered=True,
    )
    sample_frame["label"] = pd.Categorical(
        sample_frame["label"],
        categories=categories,
        ordered=True,
    )
    min_x = min(sample_frame["stable_percent"].min(), point_frame["observed_percent"].min())
    max_x = max(
        sample_frame["stable_percent"].max(),
        point_frame["observed_percent"].max(),
    )
    x_limits = (max(0, 5 * floor(min_x / 5) - 5), 5 * ceil(max_x / 5))
    x_breaks = list(range(int(10 * floor(x_limits[0] / 10)), int(x_limits[1]) + 1, 10))
    return point_frame, sample_frame, x_limits, x_breaks


def effect_change(effect: Effect) -> str:
    if effect.is_baseline:
        return "baseline"
    if effect.delta < 0:
        return "below"
    return "above"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
