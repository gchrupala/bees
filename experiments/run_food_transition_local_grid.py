from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate


FIELDNAMES = [
    "case_index",
    "seed",
    "food_site_count",
    "food_site_width",
    "food_site_capacity",
    "food_value",
    "vertical_comb_benefit",
    "vertical_comb_modifier",
    "generations",
    "initial_comb_tilt",
    "comb_orientation_axial",
    "transposition_mutation_correlation",
    "mutation_sd",
    "gravity_reach_generation",
    "vertical_reach_generation",
    "stable_vertical_gravity",
    "reached_gravity",
    "retained_vertical",
    "collapsed",
    "recovered_from_collapse",
    "initial_success",
    "min_success",
    "min_success_generation",
    "final_bias",
    "final_attention",
    "final_sender_transposition",
    "final_receiver_transposition",
    "final_min_transposition",
    "final_transposition_gap",
    "final_comb_tilt",
    "final_comb_orientation_alignment",
    "final_search_limit",
    "final_success",
    "final_payoff",
    "success_delta",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class GridCase:
    case_index: int
    seed: int
    food_site_count: int
    food_site_width: float
    food_site_capacity: int
    food_value: float
    vertical_comb_benefit: float


@dataclass(frozen=True)
class Thresholds:
    gravity: float
    vertical: float
    collapse_success: float
    recovery_success: float


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    if args.case_specs:
        cases = list(
            make_cases_from_specs(
                seeds=parse_ints(args.seeds),
                case_specs=parse_case_specs(args.case_specs),
            )
        )
    else:
        cases = list(
            make_cases(
                seeds=parse_ints(args.seeds),
                food_site_counts=parse_ints(args.food_site_counts),
                food_site_widths=parse_floats(args.food_site_widths),
                food_site_capacities=parse_ints(args.food_site_capacities),
                food_values=parse_floats(args.food_values),
                vertical_comb_benefits=parse_floats(args.vertical_comb_benefits),
            )
        )
    thresholds = Thresholds(
        gravity=args.gravity_threshold,
        vertical=args.vertical_threshold,
        collapse_success=args.collapse_success_threshold,
        recovery_success=args.recovery_success_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        handle.flush()

        print(
            (
                f"running {len(cases)} cases with {args.max_workers} workers; "
                f"output={relative(args.output)}"
            ),
            file=sys.stderr,
            flush=True,
        )
        completed = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(run_case, case, base_settings, thresholds)
                for case in cases
            ]
            for future in as_completed(futures):
                row = future.result()
                completed += 1
                writer.writerow(row)
                handle.flush()
                print(
                    progress_line(
                        completed=completed,
                        total=len(cases),
                        row=row,
                        elapsed=perf_counter() - started,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    print(f"wrote {args.output}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a parallel local ecology grid around the successful seed-101 "
            "horizontal-to-vertical transition."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "long_vertical_transition.json",
        help="Path to the base long-transition model config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "food_transition_local_grid.csv",
        help="Path for streamed per-case CSV output.",
    )
    parser.add_argument(
        "--seeds",
        default="101",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--food-site-counts",
        default="3,4,5",
        help="Comma-separated food-site counts.",
    )
    parser.add_argument(
        "--food-site-widths",
        default="0.26,0.30,0.34",
        help="Comma-separated food-site angular widths.",
    )
    parser.add_argument(
        "--food-site-capacities",
        default="8,10,12",
        help="Comma-separated food-site capacities.",
    )
    parser.add_argument(
        "--food-values",
        default="1.0",
        help="Comma-separated food-site values.",
    )
    parser.add_argument(
        "--vertical-comb-benefits",
        default="0.20,0.25,0.30",
        help="Comma-separated proportional vertical-comb advantages.",
    )
    parser.add_argument(
        "--case-specs",
        default="",
        help=(
            "Optional semicolon-separated explicit cases, each formatted as "
            "count:width:capacity:value:alpha. When set, grid arguments are "
            "ignored except seeds."
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override the generation count from the config.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel worker processes.",
    )
    parser.add_argument(
        "--gravity-threshold",
        type=float,
        default=0.50,
        help="Sender and receiver transposition threshold for gravity code.",
    )
    parser.add_argument(
        "--vertical-threshold",
        type=float,
        default=0.80,
        help="Mean comb-tilt threshold for verticality.",
    )
    parser.add_argument(
        "--collapse-success-threshold",
        type=float,
        default=0.02,
        help="A case is marked collapsed if success falls at or below this.",
    )
    parser.add_argument(
        "--recovery-success-threshold",
        type=float,
        default=0.10,
        help="A collapsed case is marked recovered above this final success.",
    )
    return parser.parse_args()


def run_case(
    case: GridCase,
    base_settings: DirectionSettings,
    thresholds: Thresholds,
) -> dict[str, str]:
    started = perf_counter()
    settings = replace(
        base_settings,
        initial_comb_tilt=0.0,
        vertical_comb_benefit=case.vertical_comb_benefit,
        vertical_comb_modifier="linear",
        food_site_count=case.food_site_count,
        food_site_width=case.food_site_width,
        food_site_capacity=case.food_site_capacity,
        food_value=case.food_value,
    )
    history = simulate(settings, seed=case.seed)
    initial = history[0]
    final = history[-1]
    min_success_state = min(history, key=lambda state: state.average_success_rate)
    gravity_reach = first_gravity_reach_generation(history, thresholds.gravity)
    vertical_reach = first_vertical_reach_generation(history, thresholds.vertical)
    collapsed = min_success_state.average_success_rate <= thresholds.collapse_success
    recovered = collapsed and final.average_success_rate >= thresholds.recovery_success
    final_min_transposition = min(
        final.average_sender_transposition,
        final.average_receiver_transposition,
    )
    stable = (
        final.average_comb_tilt >= thresholds.vertical
        and final_min_transposition >= thresholds.gravity
    )

    return {
        "case_index": str(case.case_index),
        "seed": str(case.seed),
        "food_site_count": str(case.food_site_count),
        "food_site_width": format_float(case.food_site_width),
        "food_site_capacity": str(case.food_site_capacity),
        "food_value": format_float(case.food_value),
        "vertical_comb_benefit": format_float(case.vertical_comb_benefit),
        "vertical_comb_modifier": settings.vertical_comb_modifier,
        "generations": str(settings.generations),
        "initial_comb_tilt": format_float(settings.initial_comb_tilt),
        "comb_orientation_axial": str(settings.comb_orientation_axial).lower(),
        "transposition_mutation_correlation": format_float(
            settings.transposition_mutation_correlation,
        ),
        "mutation_sd": format_float(settings.mutation_sd),
        "gravity_reach_generation": generation_value(gravity_reach),
        "vertical_reach_generation": generation_value(vertical_reach),
        "stable_vertical_gravity": str(stable).lower(),
        "reached_gravity": str(gravity_reach is not None).lower(),
        "retained_vertical": str(final.average_comb_tilt >= thresholds.vertical).lower(),
        "collapsed": str(collapsed).lower(),
        "recovered_from_collapse": str(recovered).lower(),
        "initial_success": format_float(initial.average_success_rate),
        "min_success": format_float(min_success_state.average_success_rate),
        "min_success_generation": str(min_success_state.generation),
        "final_bias": format_float(final.average_directional_bias),
        "final_attention": format_float(final.average_receiver_attention),
        "final_sender_transposition": format_float(
            final.average_sender_transposition,
        ),
        "final_receiver_transposition": format_float(
            final.average_receiver_transposition,
        ),
        "final_min_transposition": format_float(final_min_transposition),
        "final_transposition_gap": format_float(
            abs(
                final.average_sender_transposition
                - final.average_receiver_transposition
            )
        ),
        "final_comb_tilt": format_float(final.average_comb_tilt),
        "final_comb_orientation_alignment": format_float(
            final.comb_orientation_alignment,
        ),
        "final_search_limit": format_float(final.average_search_limit),
        "final_success": format_float(final.average_success_rate),
        "final_payoff": format_float(final.average_payoff),
        "success_delta": format_float(
            final.average_success_rate - initial.average_success_rate,
        ),
        "elapsed_seconds": format_float(perf_counter() - started),
    }


def make_cases(
    seeds: list[int],
    food_site_counts: list[int],
    food_site_widths: list[float],
    food_site_capacities: list[int],
    food_values: list[float],
    vertical_comb_benefits: list[float],
):
    case_index = 0
    for seed in seeds:
        for food_site_count in food_site_counts:
            for food_site_width in food_site_widths:
                for food_site_capacity in food_site_capacities:
                    for food_value in food_values:
                        for vertical_comb_benefit in vertical_comb_benefits:
                            case_index += 1
                            yield GridCase(
                                case_index=case_index,
                                seed=seed,
                                food_site_count=food_site_count,
                                food_site_width=food_site_width,
                                food_site_capacity=food_site_capacity,
                                food_value=food_value,
                                vertical_comb_benefit=vertical_comb_benefit,
                            )


def make_cases_from_specs(
    seeds: list[int],
    case_specs: list[tuple[int, float, int, float, float]],
):
    case_index = 0
    for seed in seeds:
        for (
            food_site_count,
            food_site_width,
            food_site_capacity,
            food_value,
            vertical_comb_benefit,
        ) in case_specs:
            case_index += 1
            yield GridCase(
                case_index=case_index,
                seed=seed,
                food_site_count=food_site_count,
                food_site_width=food_site_width,
                food_site_capacity=food_site_capacity,
                food_value=food_value,
                vertical_comb_benefit=vertical_comb_benefit,
            )


def first_gravity_reach_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if (
            state.average_sender_transposition >= threshold
            and state.average_receiver_transposition >= threshold
        ):
            return state.generation

    return None


def first_vertical_reach_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if state.average_comb_tilt >= threshold:
            return state.generation

    return None


def load_settings(config_path: Path) -> DirectionSettings:
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    return DirectionSettings(**config)


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_case_specs(raw: str) -> list[tuple[int, float, int, float, float]]:
    specs = []
    for item in (part.strip() for part in raw.split(";")):
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 5:
            raise ValueError(
                "case specs must have five fields: "
                "count:width:capacity:value:alpha"
            )
        specs.append(
            (
                int(parts[0]),
                float(parts[1]),
                int(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        )
    return specs


def generation_value(generation: int | None) -> str:
    if generation is None:
        return "not_reached"

    return str(generation)


def progress_line(
    completed: int,
    total: int,
    row: dict[str, str],
    elapsed: float,
) -> str:
    return (
        f"{completed}/{total} case={row['case_index']} "
        f"n={row['food_site_count']} width={row['food_site_width']} "
        f"cap={row['food_site_capacity']} alpha={row['vertical_comb_benefit']} "
        f"stable={row['stable_vertical_gravity']} "
        f"gravity={row['reached_gravity']} vertical={row['retained_vertical']} "
        f"tilt={row['final_comb_tilt']} m={row['final_min_transposition']} "
        f"success={row['final_success']} elapsed={elapsed:.1f}s"
    )


def format_float(value: float) -> str:
    return f"{value:.3f}"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
