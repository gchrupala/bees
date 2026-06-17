from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "bees-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


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
    ci_low: float
    ci_high: float
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
            "Plot OAT sensitivity effects as paired changes in stable-transition "
            "rate relative to the baseline."
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
        help="Output path prefix. The script writes .svg and .png.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Paired bootstrap resamples for uncertainty intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260617,
        help="Random seed for deterministic bootstrap intervals.",
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
            ci_low=0.0,
            ci_high=0.0,
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
        ci_low, ci_high = paired_bootstrap_interval(
            seed_deltas,
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
                ci_low=ci_low,
                ci_high=ci_high,
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


def paired_bootstrap_interval(
    deltas: list[int],
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if samples <= 0:
        point = sum(deltas) / len(deltas)
        return point, point

    n = len(deltas)
    estimates = []
    for _ in range(samples):
        total = sum(deltas[rng.randrange(n)] for _ in range(n))
        estimates.append(total / n)
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty list")
    index = probability * (len(values) - 1)
    low_index = int(index)
    high_index = min(low_index + 1, len(values) - 1)
    weight = index - low_index
    return values[low_index] * (1 - weight) + values[high_index] * weight


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

    y_positions = list(range(len(effects)))
    deltas = [effect.delta * 100 for effect in effects]
    ci_low = [effect.ci_low * 100 for effect in effects]
    ci_high = [effect.ci_high * 100 for effect in effects]
    colors = [effect_color(effect) for effect in effects]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "bees-oat-sensitivity-stable-delta",
        }
    )
    fig_height = max(5.2, 0.36 * len(effects) + 1.8)
    fig, ax = plt.subplots(figsize=(8.4, fig_height), constrained_layout=True)

    for y, low, high, color, effect in zip(
        y_positions,
        ci_low,
        ci_high,
        colors,
        effects,
    ):
        if not effect.is_baseline:
            ax.hlines(y, low, high, color=color, linewidth=1.8, alpha=0.95, zorder=2)
            ax.vlines(
                [low, high],
                y - 0.08,
                y + 0.08,
                color=color,
                linewidth=1.8,
                alpha=0.95,
                zorder=2,
            )
    ax.scatter(
        deltas,
        y_positions,
        s=54,
        c=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    for y, delta, effect in zip(y_positions, deltas, effects):
        offset = -1.2 if delta < 0 and not effect.is_baseline else 1.2
        alignment = "right" if delta < 0 and not effect.is_baseline else "left"
        ax.text(
            delta + offset,
            y,
            f"{effect.stable_percent:.1f}%",
            va="center",
            ha=alignment,
            fontsize=9,
            color="#111827",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "boxstyle": "round,pad=0.15",
                "alpha": 0.85,
            },
        )

    min_x = min(ci_low + deltas + [0]) - 6
    max_x = max(ci_high + deltas + [0]) + 6
    ax.set_xlim(min_x, max_x)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([effect.label for effect in effects])
    ax.invert_yaxis()
    ax.axvline(0, color="#111827", linewidth=1)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Change in stable outcomes vs baseline (percentage points)")
    ax.set_title(
        (
            "Coarse one-parameter sensitivity: stable vertical gravity-code outcomes\n"
            f"Baseline: {100 * baseline_count / runs:.1f}% stable; "
            "whiskers are paired seed-bootstrap intervals"
        ),
        loc="left",
        fontsize=11,
        pad=12,
    )

    svg_path = output_prefix.with_suffix(".svg")
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(svg_path, metadata={"Date": None})
    strip_trailing_whitespace(svg_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    print(f"wrote {relative(svg_path)}", flush=True)
    print(f"wrote {relative(png_path)}", flush=True)


def effect_color(effect: Effect) -> str:
    if effect.is_baseline:
        return "#4b5563"
    if effect.delta < 0:
        return "#c2410c"
    return "#047857"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
