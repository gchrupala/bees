"""Food-distribution / communication experiment on the current v2 model.

This experiment asks where direct-pointing dance communication is favored as the
food ecology is varied, on a comb held flat (``evolve_comb_tilt=False``,
``initial_comb_tilt=0``) so the vertical-comb / gravity-code machinery stays
inert and only the direct-pointing dance is in play.

The headline outcome is the in-run *recruitment advantage*: among foraging
attempts where a dance was available, the success rate of dance-followers minus
the success rate of matched non-followers. This is a contemporaneous,
near-randomized contrast (the follow decision is a coin flip on receiver
attention), so it measures how much the dance actually buys in each ecology
rather than whether a trait has merely drifted up.

Conditions reproduce the v1 hard/baseline/easy anchors and add two orthogonal
one-dimensional sweeps through the difficulty space (food-site count and patch
angular width).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate  # noqa: E402

RESULTS = ROOT / "results"
DEFAULT_PREFIX = RESULTS / "food_distribution_v2"
DEFAULT_CONFIG = ROOT / "configs" / "food_distribution_v2.json"

PARAM_FIELDNAMES = [
    "food_site_count",
    "food_site_width",
    "food_site_capacity",
    "food_value",
]
POINT_FIELDNAMES = ["condition", "sweep", *PARAM_FIELDNAMES]
EVENT_FIELDNAMES = [
    *POINT_FIELDNAMES,
    "seed",
    "generations",
    "final_directional_bias",
    "final_receiver_attention",
    "final_search_limit",
    "final_comb_tilt",
    "final_success",
    "final_payoff",
    "final_follower_success_rate",
    "final_matched_searcher_success_rate",
    "final_recruitment_advantage",
    "final_dance_follow_share",
    "tail_recruitment_advantage",
    "tail_follower_success_rate",
    "tail_matched_searcher_success_rate",
    "tail_dance_follow_share",
]
TRAJECTORY_FIELDNAMES = [
    *POINT_FIELDNAMES,
    "seed",
    "generation",
    "directional_bias",
    "receiver_attention",
    "search_limit",
    "comb_tilt",
    "success_rate",
    "payoff",
    "follower_success_rate",
    "matched_searcher_success_rate",
    "recruitment_advantage",
    "dance_follow_share",
]
GROUP_FIELDNAMES = [
    *POINT_FIELDNAMES,
    "seeds",
    "mean_final_recruitment_advantage",
    "median_final_recruitment_advantage",
    "mean_tail_recruitment_advantage",
    "median_tail_recruitment_advantage",
    "useful_fraction",
    "mean_final_directional_bias",
    "mean_final_receiver_attention",
    "mean_final_success",
    "mean_final_payoff",
    "mean_tail_follower_success_rate",
    "mean_tail_matched_searcher_success_rate",
    "mean_tail_dance_follow_share",
]


@dataclass(frozen=True)
class Condition:
    name: str
    sweep: str
    food_site_count: int
    food_site_width: float
    food_site_capacity: int
    food_value: float

    def param_values(self) -> dict[str, float | int]:
        return {
            "food_site_count": self.food_site_count,
            "food_site_width": self.food_site_width,
            "food_site_capacity": self.food_site_capacity,
            "food_value": self.food_value,
        }


@dataclass(frozen=True)
class SeedResult:
    condition: str
    sweep: str
    seed: int
    event_row: dict[str, str]
    trajectory_rows: list[dict[str, str]]
    tail_recruitment_advantage: float
    final_recruitment_advantage: float
    final_directional_bias: float
    final_receiver_attention: float
    final_success: float
    final_payoff: float
    tail_follower_success_rate: float
    tail_matched_searcher_success_rate: float
    tail_dance_follow_share: float


def build_conditions() -> list[Condition]:
    conditions: list[Condition] = [
        Condition("hard", "anchor", 1, 0.08, 6, 1.0),
        Condition("baseline", "anchor", 2, 0.20, 6, 1.0),
        Condition("easy", "anchor", 8, 0.50, 6, 1.0),
    ]
    for count in (1, 2, 3, 4, 6, 8):
        conditions.append(
            Condition(f"sites_{count}", "site_count", count, 0.20, 6, 1.0)
        )
    for width in (0.08, 0.15, 0.20, 0.30, 0.50):
        conditions.append(
            Condition(f"width_{width:.2f}", "patch_width", 2, width, 6, 1.0)
        )
    return conditions


def condition_settings(
    base_settings: DirectionSettings,
    condition: Condition,
) -> DirectionSettings:
    # The comb is held flat for this experiment regardless of the base config.
    return replace(
        base_settings,
        evolve_comb_tilt=False,
        initial_comb_tilt=0.0,
        **condition.param_values(),
    )


def run_condition_seed(
    condition: Condition,
    settings: DirectionSettings,
    seed: int,
    tail_window: int,
    useful_threshold: float,
) -> SeedResult:
    history = simulate(settings, seed=seed)
    final = history[-1]
    tail = history[-tail_window:] if tail_window > 0 else [final]

    tail_recruitment = mean(state.recruitment_advantage for state in tail)
    tail_follower = mean(state.follower_success_rate for state in tail)
    tail_searcher = mean(state.matched_searcher_success_rate for state in tail)
    tail_follow_share = mean(state.dance_follow_share for state in tail)

    point = {
        "condition": condition.name,
        "sweep": condition.sweep,
        **{key: _fmt(value) for key, value in condition.param_values().items()},
    }
    event_row = {
        **point,
        "seed": str(seed),
        "generations": str(settings.generations),
        "final_directional_bias": _fmt(final.average_directional_bias),
        "final_receiver_attention": _fmt(final.average_receiver_attention),
        "final_search_limit": _fmt(final.average_search_limit),
        "final_comb_tilt": _fmt(final.average_comb_tilt),
        "final_success": _fmt(final.average_success_rate),
        "final_payoff": _fmt(final.average_payoff),
        "final_follower_success_rate": _fmt(final.follower_success_rate),
        "final_matched_searcher_success_rate": _fmt(
            final.matched_searcher_success_rate
        ),
        "final_recruitment_advantage": _fmt(final.recruitment_advantage),
        "final_dance_follow_share": _fmt(final.dance_follow_share),
        "tail_recruitment_advantage": _fmt(tail_recruitment),
        "tail_follower_success_rate": _fmt(tail_follower),
        "tail_matched_searcher_success_rate": _fmt(tail_searcher),
        "tail_dance_follow_share": _fmt(tail_follow_share),
    }
    trajectory_rows = [
        {
            **point,
            "seed": str(seed),
            "generation": str(state.generation),
            "directional_bias": _fmt(state.average_directional_bias),
            "receiver_attention": _fmt(state.average_receiver_attention),
            "search_limit": _fmt(state.average_search_limit),
            "comb_tilt": _fmt(state.average_comb_tilt),
            "success_rate": _fmt(state.average_success_rate),
            "payoff": _fmt(state.average_payoff),
            "follower_success_rate": _fmt(state.follower_success_rate),
            "matched_searcher_success_rate": _fmt(
                state.matched_searcher_success_rate
            ),
            "recruitment_advantage": _fmt(state.recruitment_advantage),
            "dance_follow_share": _fmt(state.dance_follow_share),
        }
        for state in history
    ]

    return SeedResult(
        condition=condition.name,
        sweep=condition.sweep,
        seed=seed,
        event_row=event_row,
        trajectory_rows=trajectory_rows,
        tail_recruitment_advantage=tail_recruitment,
        final_recruitment_advantage=final.recruitment_advantage,
        final_directional_bias=final.average_directional_bias,
        final_receiver_attention=final.average_receiver_attention,
        final_success=final.average_success_rate,
        final_payoff=final.average_payoff,
        tail_follower_success_rate=tail_follower,
        tail_matched_searcher_success_rate=tail_searcher,
        tail_dance_follow_share=tail_follow_share,
    )


def group_summary_row(
    condition: Condition,
    results: list[SeedResult],
    useful_threshold: float,
) -> dict[str, str]:
    point = {
        "condition": condition.name,
        "sweep": condition.sweep,
        **{key: _fmt(value) for key, value in condition.param_values().items()},
    }
    final_adv = [result.final_recruitment_advantage for result in results]
    tail_adv = [result.tail_recruitment_advantage for result in results]
    useful = [value >= useful_threshold for value in tail_adv]

    return {
        **point,
        "seeds": str(len(results)),
        "mean_final_recruitment_advantage": _fmt(mean(final_adv)),
        "median_final_recruitment_advantage": _fmt(median(final_adv)),
        "mean_tail_recruitment_advantage": _fmt(mean(tail_adv)),
        "median_tail_recruitment_advantage": _fmt(median(tail_adv)),
        "useful_fraction": _fmt(mean(useful)),
        "mean_final_directional_bias": _fmt(
            mean(result.final_directional_bias for result in results)
        ),
        "mean_final_receiver_attention": _fmt(
            mean(result.final_receiver_attention for result in results)
        ),
        "mean_final_success": _fmt(
            mean(result.final_success for result in results)
        ),
        "mean_final_payoff": _fmt(
            mean(result.final_payoff for result in results)
        ),
        "mean_tail_follower_success_rate": _fmt(
            mean(result.tail_follower_success_rate for result in results)
        ),
        "mean_tail_matched_searcher_success_rate": _fmt(
            mean(result.tail_matched_searcher_success_rate for result in results)
        ),
        "mean_tail_dance_follow_share": _fmt(
            mean(result.tail_dance_follow_share for result in results)
        ),
    }


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    conditions = build_conditions()
    seeds = parse_seed_spec(args.seeds)
    outputs = output_paths(args.output_prefix)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    write_rows(
        outputs["points"],
        POINT_FIELDNAMES,
        (condition_point_row(condition) for condition in conditions),
    )
    print(f"wrote {relative(outputs['points'])}", file=sys.stderr, flush=True)

    started = perf_counter()
    total_runs = len(conditions) * len(seeds)
    print(
        (
            f"running food-distribution v2: conditions={len(conditions)} "
            f"seeds={len(seeds)} total_runs={total_runs} "
            f"workers={args.max_workers} generations={base_settings.generations}"
        ),
        file=sys.stderr,
        flush=True,
    )

    results: list[SeedResult] = []
    with (
        outputs["events"].open("w", newline="") as event_handle,
        outputs["trajectories"].open("w", newline="") as trajectory_handle,
        ProcessPoolExecutor(max_workers=args.max_workers) as executor,
    ):
        event_writer = csv.DictWriter(
            event_handle,
            fieldnames=EVENT_FIELDNAMES,
            lineterminator="\n",
        )
        trajectory_writer = csv.DictWriter(
            trajectory_handle,
            fieldnames=TRAJECTORY_FIELDNAMES,
            lineterminator="\n",
        )
        event_writer.writeheader()
        trajectory_writer.writeheader()
        event_handle.flush()
        trajectory_handle.flush()

        futures = {}
        for condition in conditions:
            settings = condition_settings(base_settings, condition)
            for seed in seeds:
                future = executor.submit(
                    run_condition_seed,
                    condition,
                    settings,
                    seed,
                    args.tail_window,
                    args.useful_threshold,
                )
                futures[future] = (condition.name, seed)

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            results.append(result)
            event_writer.writerow(result.event_row)
            trajectory_writer.writerows(result.trajectory_rows)
            event_handle.flush()
            trajectory_handle.flush()
            print(
                (
                    f"{completed}/{total_runs} condition={result.condition} "
                    f"seed={result.seed} "
                    f"adv={result.event_row['tail_recruitment_advantage']} "
                    f"bias={result.event_row['final_directional_bias']} "
                    f"success={result.event_row['final_success']} "
                    f"elapsed={perf_counter() - started:.1f}s"
                ),
                file=sys.stderr,
                flush=True,
            )

    condition_order = {condition.name: index for index, condition in enumerate(conditions)}
    results.sort(key=lambda result: (condition_order[result.condition], result.seed))

    grouped = {condition.name: [] for condition in conditions}
    for result in results:
        grouped[result.condition].append(result)
    write_rows(
        outputs["group_summary"],
        GROUP_FIELDNAMES,
        (
            group_summary_row(condition, grouped[condition.name], args.useful_threshold)
            for condition in conditions
            if grouped[condition.name]
        ),
    )

    print(
        (
            f"wrote {relative(outputs['events'])}, "
            f"{relative(outputs['trajectories'])}, "
            f"{relative(outputs['group_summary'])} "
            f"in {perf_counter() - started:.1f}s"
        ),
        file=sys.stderr,
        flush=True,
    )


def condition_point_row(condition: Condition) -> dict[str, str]:
    return {
        "condition": condition.name,
        "sweep": condition.sweep,
        **{key: _fmt(value) for key, value in condition.param_values().items()},
    }


def output_paths(prefix: str) -> dict[str, Path]:
    base = Path(prefix)
    return {
        "points": base.with_name(base.name + "_points.csv"),
        "events": base.with_name(base.name + "_events.csv"),
        "trajectories": base.with_name(base.name + "_trajectories.csv"),
        "group_summary": base.with_name(base.name + "_group_summary.csv"),
    }


def write_rows(path: Path, fieldnames: list[str], rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_settings(config_path: Path) -> DirectionSettings:
    config = json.loads(Path(config_path).read_text())
    config.pop("seed", None)
    return DirectionSettings(**config)


def parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(token))
    return seeds


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", default="400-449")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument(
        "--tail-window",
        type=int,
        default=10,
        help="number of final generations averaged for the tail metrics",
    )
    parser.add_argument(
        "--useful-threshold",
        type=float,
        default=0.05,
        help="tail recruitment advantage above which a seed counts as useful",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-prefix", default=str(DEFAULT_PREFIX))
    return parser.parse_args()


if __name__ == "__main__":
    main()
