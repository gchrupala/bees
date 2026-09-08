"""Sensitivity of the stage-1 food-distribution result to two modelling choices.

The horizontal-comb grid (mean patch count x median patch radius) is re-run under
several variants of the model. Each variant changes exactly one assumption, so a
variant's grid can be compared cell-by-cell against the baseline.

Recruitment rule (github issue #9). A patch's area-scaled capacity is clamped
below at one load, so every patch under 75 m radius holds exactly one. A one-load
patch is emptied by the forager that finds it, and the recruitment rule reads the
capacity *remaining* after her visit, so she never dances. Recruitment is
therefore mechanically impossible in the small-radius columns, independently of
whether patches there are worth advertising.

  * ``pre_visit_dance`` reads the capacity the scout found rather than what she
    left, so a one-load patch can seed a dance. Ecology is untouched.
  * ``no_capacity_floor`` drops the clamp, so sub-threshold patches hold nothing
    and cannot be foraged at all. Ecology changes; recruitment rule is untouched.

Selection (github issue #10). Colony fitness is clipped at MIN_COLONY_PAYOFF and
parents are drawn proportionally to it, so colonies at the floor are
indistinguishable; and every colony is scored on its own environment draw, so
near the floor which colonies clear it is largely luck.

  * ``tournament`` selects on payoff order instead, which separates colonies that
    proportional selection cannot.
  * ``common_draws`` scores every colony in a generation on the same environments,
    removing the environment draw as a source of between-colony variance.

Every variant, including the baseline, is run here rather than compared against
the stored grid, so all cells come from one code path and one seed panel.

Rows are streamed as each run finishes, so a partial CSV survives an interrupted
job; the per-cell summary is written at the end.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bees.model import DirectionSettings, simulate  # noqa: E402
from run_food_distribution_disk import (  # noqa: E402
    PATCH_RADII,
    SITE_COUNTS,
    Condition,
    _fmt,
    load_settings,
    parse_seed_spec,
    relative,
    write_rows,
)

DEFAULT_PREFIX = ROOT / "results" / "assumption_sensitivity"
DEFAULT_CONFIG = ROOT / "configs" / "food_distribution_disk.json"

# Each variant is a set of DirectionSettings overrides applied on top of the
# baseline configuration. "baseline" is deliberately empty.
VARIANTS: dict[str, dict[str, object]] = {
    "baseline": {},
    "pre_visit_dance": {"dance_capacity_basis": "pre_visit"},
    "no_capacity_floor": {"food_capacity_floor": 0},
    "tournament": {"selection": "tournament", "tournament_size": 3},
    "common_draws": {"common_episode_draws": True},
}

EVENT_FIELDNAMES = [
    "variant",
    "condition",
    "food_site_count",
    "food_site_radius",
    "seed",
    "final_recruitment_advantage",
    "tail_recruitment_advantage",
    "final_directional_bias",
    "final_receiver_attention",
    "final_dance_propensity",
    "final_success",
    "final_payoff",
    "final_floor_fraction",
    "tail_floor_fraction",
    "tail_dance_follow_share",
]

GROUP_FIELDNAMES = [
    "variant",
    "condition",
    "food_site_count",
    "food_site_radius",
    "seeds",
    "mean_final_recruitment_advantage",
    "median_final_recruitment_advantage",
    "mean_tail_recruitment_advantage",
    "median_tail_recruitment_advantage",
    "useful_fraction",
    "mean_final_directional_bias",
    "mean_final_receiver_attention",
    "mean_final_dance_propensity",
    "mean_final_success",
    "mean_final_payoff",
    "mean_final_floor_fraction",
    "mean_tail_floor_fraction",
    "mean_tail_dance_follow_share",
]

SUMMARY_METRICS = (
    "final_recruitment_advantage",
    "tail_recruitment_advantage",
    "final_directional_bias",
    "final_receiver_attention",
    "final_dance_propensity",
    "final_success",
    "final_payoff",
    "final_floor_fraction",
    "tail_floor_fraction",
    "tail_dance_follow_share",
)


@dataclass(frozen=True)
class SeedResult:
    variant: str
    condition: str
    seed: int
    metrics: dict[str, float]
    row: dict[str, str]


def build_conditions() -> list[Condition]:
    """The stage-1 grid, identical to the published food-distribution run."""
    return [
        Condition(f"c{count}_r{radius:g}m", "grid", count, radius, 6, 1.0)
        for count in SITE_COUNTS
        for radius in PATCH_RADII
    ]


def variant_settings(
    base_settings: DirectionSettings,
    condition: Condition,
    variant: str,
) -> DirectionSettings:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}")
    return replace(
        base_settings,
        evolve_comb_tilt=False,
        initial_comb_tilt=0.0,
        **condition.param_values(),
        **VARIANTS[variant],
    )


def run_variant_seed(
    variant: str,
    condition: Condition,
    settings: DirectionSettings,
    seed: int,
    tail_window: int,
) -> SeedResult:
    history = simulate(settings, seed=seed)
    final = history[-1]
    tail = history[-tail_window:] if tail_window > 0 else [final]

    metrics = {
        "final_recruitment_advantage": final.recruitment_advantage,
        "tail_recruitment_advantage": mean(s.recruitment_advantage for s in tail),
        "final_directional_bias": final.average_directional_bias,
        "final_receiver_attention": final.average_receiver_attention,
        "final_dance_propensity": final.average_dance_propensity,
        "final_success": final.average_success_rate,
        "final_payoff": final.average_payoff,
        "final_floor_fraction": final.floor_fraction,
        "tail_floor_fraction": mean(s.floor_fraction for s in tail),
        "tail_dance_follow_share": mean(s.dance_follow_share for s in tail),
    }
    row = {
        "variant": variant,
        "condition": condition.name,
        "food_site_count": _fmt(condition.food_site_count),
        "food_site_radius": _fmt(condition.food_site_radius),
        "seed": str(seed),
        **{key: _fmt(value) for key, value in metrics.items()},
    }
    return SeedResult(variant, condition.name, seed, metrics, row)


def group_summary_row(
    variant: str,
    condition: Condition,
    results: list[SeedResult],
    useful_threshold: float,
) -> dict[str, str]:
    tail_adv = [r.metrics["tail_recruitment_advantage"] for r in results]
    final_adv = [r.metrics["final_recruitment_advantage"] for r in results]
    row = {
        "variant": variant,
        "condition": condition.name,
        "food_site_count": _fmt(condition.food_site_count),
        "food_site_radius": _fmt(condition.food_site_radius),
        "seeds": str(len(results)),
        "median_final_recruitment_advantage": _fmt(median(final_adv)),
        "median_tail_recruitment_advantage": _fmt(median(tail_adv)),
        "useful_fraction": _fmt(mean(v >= useful_threshold for v in tail_adv)),
    }
    for metric in SUMMARY_METRICS:
        key = f"mean_{metric}"
        if key in GROUP_FIELDNAMES:
            row[key] = _fmt(mean(r.metrics[metric] for r in results))
    return row


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {', '.join(unknown)}")

    conditions = build_conditions()
    seeds = parse_seed_spec(args.seeds)
    base = Path(args.output_prefix)
    events_path = base.with_name(base.name + "_events.csv")
    summary_path = base.with_name(base.name + "_group_summary.csv")
    events_path.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    total_runs = len(variants) * len(conditions) * len(seeds)
    print(
        (
            f"running assumption sensitivity: variants={len(variants)} "
            f"({', '.join(variants)}) conditions={len(conditions)} "
            f"seeds={len(seeds)} total_runs={total_runs} "
            f"workers={args.max_workers} generations={base_settings.generations}"
        ),
        file=sys.stderr,
        flush=True,
    )

    results: list[SeedResult] = []
    with (
        events_path.open("w", newline="") as event_handle,
        ProcessPoolExecutor(max_workers=args.max_workers) as executor,
    ):
        event_writer = csv.DictWriter(
            event_handle, fieldnames=EVENT_FIELDNAMES, lineterminator="\n"
        )
        event_writer.writeheader()
        event_handle.flush()

        futures = {}
        for variant in variants:
            for condition in conditions:
                settings = variant_settings(base_settings, condition, variant)
                for seed in seeds:
                    future = executor.submit(
                        run_variant_seed,
                        variant,
                        condition,
                        settings,
                        seed,
                        args.tail_window,
                    )
                    futures[future] = (variant, condition.name, seed)

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            results.append(result)
            event_writer.writerow(result.row)
            event_handle.flush()
            print(
                (
                    f"{completed}/{total_runs} variant={result.variant} "
                    f"condition={result.condition} seed={result.seed} "
                    f"adv={result.row['tail_recruitment_advantage']} "
                    f"floor={result.row['tail_floor_fraction']} "
                    f"elapsed={perf_counter() - started:.1f}s"
                ),
                file=sys.stderr,
                flush=True,
            )

    grouped: dict[tuple[str, str], list[SeedResult]] = {}
    for result in results:
        grouped.setdefault((result.variant, result.condition), []).append(result)
    for bucket in grouped.values():
        bucket.sort(key=lambda r: r.seed)

    write_rows(
        summary_path,
        GROUP_FIELDNAMES,
        (
            group_summary_row(
                variant, condition, grouped[(variant, condition.name)], args.useful_threshold
            )
            for variant in variants
            for condition in conditions
            if (variant, condition.name) in grouped
        ),
    )

    print(
        (
            f"wrote {relative(events_path)}, {relative(summary_path)} "
            f"in {perf_counter() - started:.1f}s"
        ),
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"comma-separated subset of: {', '.join(VARIANTS)}",
    )
    parser.add_argument("--seeds", default="400-449")
    parser.add_argument("--output-prefix", default=str(DEFAULT_PREFIX))
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--tail-window", type=int, default=10)
    parser.add_argument("--useful-threshold", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    main()
