from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate

TRAJECTORY_FIELDNAMES = [
    "initial_comb_tilt",
    "vertical_comb_benefit",
    "vertical_comb_modifier",
    "seed",
    "outcome",
    "generation",
    "average_directional_bias",
    "average_receiver_attention",
    "average_sender_transposition",
    "average_receiver_transposition",
    "average_min_transposition",
    "average_transposition_gap",
    "average_comb_tilt",
    "average_comb_orientation",
    "comb_orientation_alignment",
    "average_search_limit",
    "average_success",
    "average_payoff",
]

EVENT_FIELDNAMES = [
    "initial_comb_tilt",
    "vertical_comb_benefit",
    "vertical_comb_modifier",
    "seed",
    "outcome",
    "reached_gravity",
    "retained_vertical",
    "collapsed",
    "recovered_from_collapse",
    "gravity_reach_generation",
    "partial_transposition_generation",
    "directional_reach_generation",
    "vertical_reach_generation",
    "vertical_loss_generation",
    "last_below_vertical_generation",
    "vertical_below_generations",
    "recovered_vertical_after_loss",
    "tilt_below_half_generation",
    "tilt_below_quarter_generation",
    "min_success",
    "min_success_generation",
    "final_success",
    "final_payoff",
    "final_comb_tilt",
    "final_min_transposition",
    "final_transposition_gap",
    "tilt_at_gravity_reach",
    "success_at_gravity_reach",
    "gap_at_gravity_reach",
    "max_min_transposition_while_vertical",
    "max_min_transposition_while_vertical_generation",
    "tilt_at_max_min_transposition_while_vertical",
    "success_at_max_min_transposition_while_vertical",
    "gravity_before_vertical_loss",
]

SUMMARY_FIELDNAMES = [
    "initial_comb_tilt",
    "vertical_comb_benefit",
    "vertical_comb_modifier",
    "outcome",
    "runs",
    "mean_gravity_reach_generation",
    "mean_partial_transposition_generation",
    "mean_vertical_loss_generation",
    "mean_last_below_vertical_generation",
    "mean_vertical_below_generations",
    "recovered_vertical_after_loss_fraction",
    "mean_min_success",
    "mean_final_success",
    "mean_final_payoff",
    "mean_final_comb_tilt",
    "mean_final_min_transposition",
    "mean_max_min_transposition_while_vertical",
    "gravity_before_vertical_loss_fraction",
]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    seeds = parse_ints(args.seeds)
    initial_comb_tilts = parse_floats(args.initial_comb_tilts)
    vertical_comb_benefits = parse_floats(args.vertical_comb_benefits)

    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    args.event_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    event_rows: list[dict[str, str]] = []
    condition_count = len(initial_comb_tilts) * len(vertical_comb_benefits)
    condition_index = 0

    with args.trajectory_output.open("w", newline="") as trajectory_file:
        trajectory_writer = csv.DictWriter(
            trajectory_file,
            fieldnames=TRAJECTORY_FIELDNAMES,
            lineterminator="\n",
        )
        trajectory_writer.writeheader()
        trajectory_file.flush()

        for initial_comb_tilt in initial_comb_tilts:
            for vertical_comb_benefit in vertical_comb_benefits:
                condition_index += 1
                settings = replace(
                    base_settings,
                    initial_comb_tilt=initial_comb_tilt,
                    vertical_comb_benefit=vertical_comb_benefit,
                    vertical_comb_modifier=args.vertical_comb_modifier,
                )
                print(
                    (
                        f"condition {condition_index}/{condition_count}: "
                        f"tilt={initial_comb_tilt:.3f}, "
                        f"modifier={args.vertical_comb_modifier}, "
                        f"alpha={vertical_comb_benefit:.3f}, "
                        f"seeds={len(seeds)}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

                for seed_index, seed in enumerate(seeds, start=1):
                    started = perf_counter()
                    history = simulate(settings, seed=seed)
                    event_row = summarize_events(
                        initial_comb_tilt=initial_comb_tilt,
                        vertical_comb_benefit=vertical_comb_benefit,
                        vertical_comb_modifier=args.vertical_comb_modifier,
                        seed=seed,
                        history=history,
                        gravity_threshold=args.gravity_threshold,
                        directional_threshold=args.directional_threshold,
                        vertical_threshold=args.vertical_threshold,
                        collapse_success_threshold=args.collapse_success_threshold,
                        recovery_success_threshold=args.recovery_success_threshold,
                    )
                    event_rows.append(event_row)
                    for state in history:
                        trajectory_writer.writerow(
                            trajectory_row(
                                initial_comb_tilt=initial_comb_tilt,
                                vertical_comb_benefit=vertical_comb_benefit,
                                vertical_comb_modifier=args.vertical_comb_modifier,
                                seed=seed,
                                outcome=event_row["outcome"],
                                state=state,
                            )
                        )
                    trajectory_file.flush()
                    print(
                        (
                            f"  seed {seed_index}/{len(seeds)} ({seed}) done: "
                            f"outcome={event_row['outcome']}, "
                            f"gravity={event_row['gravity_reach_generation']}, "
                            f"vertical_loss={event_row['vertical_loss_generation']}, "
                            f"elapsed={perf_counter() - started:.1f}s"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

    write_rows(args.event_output, EVENT_FIELDNAMES, event_rows)
    write_rows(args.summary_output, SUMMARY_FIELDNAMES, summarize_event_rows(event_rows))
    print(f"wrote {args.trajectory_output}", file=sys.stderr, flush=True)
    print(f"wrote {args.event_output}", file=sys.stderr, flush=True)
    print(f"wrote {args.summary_output}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun long-transition conditions and export per-generation "
            "trajectories plus event-order summaries."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "long_vertical_transition.json",
        help="Path to the long-transition model config.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=ROOT / "results" / "long_vertical_transition_linear_trajectories.csv",
        help="Path for generation-level trajectory CSV output.",
    )
    parser.add_argument(
        "--event-output",
        type=Path,
        default=ROOT / "results" / "long_vertical_transition_linear_events.csv",
        help="Path for per-seed event-order CSV output.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "long_vertical_transition_linear_event_summary.csv",
        help="Path for outcome-level event summary CSV output.",
    )
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105,106,107,108,109,110",
        help="Comma-separated random seeds to run for each condition.",
    )
    parser.add_argument(
        "--initial-comb-tilts",
        default="1.0",
        help="Comma-separated initial comb tilt values.",
    )
    parser.add_argument(
        "--vertical-comb-benefits",
        default="0.05,0.10,0.15,0.20,0.25",
        help="Comma-separated proportional vertical-comb advantage values.",
    )
    parser.add_argument(
        "--vertical-comb-modifier",
        default="linear",
        choices=("linear", "threshold_0.8"),
        help="Vertical-comb modifier function to analyze.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override the generation count from the config.",
    )
    parser.add_argument(
        "--gravity-threshold",
        type=float,
        default=0.50,
        help="Sender and receiver transposition threshold for gravity communication.",
    )
    parser.add_argument(
        "--directional-threshold",
        type=float,
        default=0.30,
        help="Directional-bias threshold used for the reach-generation metric.",
    )
    parser.add_argument(
        "--vertical-threshold",
        type=float,
        default=0.80,
        help="Mean comb-tilt threshold for retained or reached verticality.",
    )
    parser.add_argument(
        "--collapse-success-threshold",
        type=float,
        default=0.02,
        help="A seed is marked collapsed if mean success falls at or below this.",
    )
    parser.add_argument(
        "--recovery-success-threshold",
        type=float,
        default=0.10,
        help="A collapsed seed is marked recovered if final success reaches this.",
    )
    return parser.parse_args()


def load_settings(config_path: Path) -> DirectionSettings:
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    return DirectionSettings(**config)


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def summarize_events(
    initial_comb_tilt: float,
    vertical_comb_benefit: float,
    vertical_comb_modifier: str,
    seed: int,
    history: list[GenerationSummary],
    gravity_threshold: float,
    directional_threshold: float,
    vertical_threshold: float,
    collapse_success_threshold: float,
    recovery_success_threshold: float,
) -> dict[str, str]:
    final = history[-1]
    min_success_state = min(history, key=lambda state: state.average_success_rate)
    gravity_reach = first_gravity_reach_generation(history, gravity_threshold)
    partial_transposition = first_min_transposition_generation(history, 0.25)
    directional_reach = first_directional_reach_generation(history, directional_threshold)
    vertical_reach = first_vertical_reach_generation(history, vertical_threshold)
    vertical_loss = first_below_tilt_generation(history, vertical_threshold)
    last_below_vertical = last_below_tilt_generation(history, vertical_threshold)
    vertical_below_count = count_below_tilt(history, vertical_threshold)
    tilt_below_half = first_below_tilt_generation(history, 0.5)
    tilt_below_quarter = first_below_tilt_generation(history, 0.25)
    retained_vertical = final.average_comb_tilt >= vertical_threshold
    reached_gravity = gravity_reach is not None
    collapsed = min_success_state.average_success_rate <= collapse_success_threshold
    recovered = collapsed and final.average_success_rate >= recovery_success_threshold
    outcome = classify_outcome(reached_gravity, retained_vertical, collapsed, recovered)
    gravity_state = state_at_generation(history, gravity_reach)
    max_vertical_state = max_min_transposition_while_vertical(history, vertical_threshold)

    gravity_before_vertical_loss = (
        reached_gravity
        and (vertical_loss is None or gravity_reach <= vertical_loss)
    )
    recovered_vertical_after_loss = retained_vertical and vertical_loss is not None

    return {
        "initial_comb_tilt": format_float(initial_comb_tilt),
        "vertical_comb_benefit": format_float(vertical_comb_benefit),
        "vertical_comb_modifier": vertical_comb_modifier,
        "seed": str(seed),
        "outcome": outcome,
        "reached_gravity": bool_value(reached_gravity),
        "retained_vertical": bool_value(retained_vertical),
        "collapsed": bool_value(collapsed),
        "recovered_from_collapse": bool_value(recovered),
        "gravity_reach_generation": generation_value(gravity_reach),
        "partial_transposition_generation": generation_value(partial_transposition),
        "directional_reach_generation": generation_value(directional_reach),
        "vertical_reach_generation": generation_value(vertical_reach),
        "vertical_loss_generation": generation_value(vertical_loss),
        "last_below_vertical_generation": generation_value(last_below_vertical),
        "vertical_below_generations": str(vertical_below_count),
        "recovered_vertical_after_loss": bool_value(recovered_vertical_after_loss),
        "tilt_below_half_generation": generation_value(tilt_below_half),
        "tilt_below_quarter_generation": generation_value(tilt_below_quarter),
        "min_success": format_float(min_success_state.average_success_rate),
        "min_success_generation": str(min_success_state.generation),
        "final_success": format_float(final.average_success_rate),
        "final_payoff": format_float(final.average_payoff),
        "final_comb_tilt": format_float(final.average_comb_tilt),
        "final_min_transposition": format_float(min_transposition(final)),
        "final_transposition_gap": format_float(transposition_gap(final)),
        "tilt_at_gravity_reach": state_float(gravity_state, "average_comb_tilt"),
        "success_at_gravity_reach": state_float(gravity_state, "average_success_rate"),
        "gap_at_gravity_reach": (
            format_float(transposition_gap(gravity_state))
            if gravity_state is not None
            else "not_reached"
        ),
        "max_min_transposition_while_vertical": state_float(
            max_vertical_state,
            "average_min_transposition",
        ),
        "max_min_transposition_while_vertical_generation": (
            str(max_vertical_state.generation)
            if max_vertical_state is not None
            else "not_applicable"
        ),
        "tilt_at_max_min_transposition_while_vertical": state_float(
            max_vertical_state,
            "average_comb_tilt",
        ),
        "success_at_max_min_transposition_while_vertical": state_float(
            max_vertical_state,
            "average_success_rate",
        ),
        "gravity_before_vertical_loss": bool_value(gravity_before_vertical_loss),
    }


def trajectory_row(
    initial_comb_tilt: float,
    vertical_comb_benefit: float,
    vertical_comb_modifier: str,
    seed: int,
    outcome: str,
    state: GenerationSummary,
) -> dict[str, str]:
    return {
        "initial_comb_tilt": format_float(initial_comb_tilt),
        "vertical_comb_benefit": format_float(vertical_comb_benefit),
        "vertical_comb_modifier": vertical_comb_modifier,
        "seed": str(seed),
        "outcome": outcome,
        "generation": str(state.generation),
        "average_directional_bias": format_float(state.average_directional_bias),
        "average_receiver_attention": format_float(state.average_receiver_attention),
        "average_sender_transposition": format_float(
            state.average_sender_transposition,
        ),
        "average_receiver_transposition": format_float(
            state.average_receiver_transposition,
        ),
        "average_min_transposition": format_float(min_transposition(state)),
        "average_transposition_gap": format_float(transposition_gap(state)),
        "average_comb_tilt": format_float(state.average_comb_tilt),
        "average_comb_orientation": format_float(state.average_comb_orientation),
        "comb_orientation_alignment": format_float(state.comb_orientation_alignment),
        "average_search_limit": format_float(state.average_search_limit),
        "average_success": format_float(state.average_success_rate),
        "average_payoff": format_float(state.average_payoff),
    }


def summarize_event_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["initial_comb_tilt"],
            row["vertical_comb_benefit"],
            row["vertical_comb_modifier"],
            row["outcome"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        initial_comb_tilt, vertical_comb_benefit, vertical_comb_modifier, outcome = key
        summary_rows.append(
            {
                "initial_comb_tilt": initial_comb_tilt,
                "vertical_comb_benefit": vertical_comb_benefit,
                "vertical_comb_modifier": vertical_comb_modifier,
                "outcome": outcome,
                "runs": str(len(group)),
                "mean_gravity_reach_generation": mean_field(
                    group,
                    "gravity_reach_generation",
                ),
                "mean_partial_transposition_generation": mean_field(
                    group,
                    "partial_transposition_generation",
                ),
                "mean_vertical_loss_generation": mean_field(
                    group,
                    "vertical_loss_generation",
                ),
                "mean_last_below_vertical_generation": mean_field(
                    group,
                    "last_below_vertical_generation",
                ),
                "mean_vertical_below_generations": mean_float_field(
                    group,
                    "vertical_below_generations",
                ),
                "recovered_vertical_after_loss_fraction": format_float(
                    fraction(
                        row["recovered_vertical_after_loss"] == "true"
                        for row in group
                    )
                ),
                "mean_min_success": mean_float_field(group, "min_success"),
                "mean_final_success": mean_float_field(group, "final_success"),
                "mean_final_payoff": mean_float_field(group, "final_payoff"),
                "mean_final_comb_tilt": mean_float_field(group, "final_comb_tilt"),
                "mean_final_min_transposition": mean_float_field(
                    group,
                    "final_min_transposition",
                ),
                "mean_max_min_transposition_while_vertical": mean_field(
                    group,
                    "max_min_transposition_while_vertical",
                ),
                "gravity_before_vertical_loss_fraction": format_float(
                    fraction(
                        row["gravity_before_vertical_loss"] == "true"
                        for row in group
                    )
                ),
            }
        )
    return summary_rows


def classify_outcome(
    reached_gravity: bool,
    retained_vertical: bool,
    collapsed: bool,
    recovered: bool,
) -> str:
    if reached_gravity and retained_vertical:
        return "success_after_collapse" if collapsed else "success"

    if reached_gravity:
        return "gravity_without_vertical"

    if retained_vertical:
        return "vertical_without_gravity"

    if collapsed and recovered:
        return "flat_recovered_collapse"

    if collapsed:
        return "flat_collapsed"

    return "flat"


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


def first_min_transposition_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if min_transposition(state) >= threshold:
            return state.generation
    return None


def first_directional_reach_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if state.average_directional_bias >= threshold:
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


def first_below_tilt_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    for state in history:
        if state.average_comb_tilt < threshold:
            return state.generation
    return None


def last_below_tilt_generation(
    history: list[GenerationSummary],
    threshold: float,
) -> int | None:
    last_generation = None
    for state in history:
        if state.average_comb_tilt < threshold:
            last_generation = state.generation
    return last_generation


def count_below_tilt(
    history: list[GenerationSummary],
    threshold: float,
) -> int:
    return sum(state.average_comb_tilt < threshold for state in history)


def max_min_transposition_while_vertical(
    history: list[GenerationSummary],
    vertical_threshold: float,
) -> GenerationSummary | None:
    vertical_states = [
        state for state in history if state.average_comb_tilt >= vertical_threshold
    ]
    if not vertical_states:
        return None

    return max(vertical_states, key=min_transposition)


def state_at_generation(
    history: list[GenerationSummary],
    generation: int | None,
) -> GenerationSummary | None:
    if generation is None:
        return None

    return history[generation]


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, str]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def min_transposition(state: GenerationSummary) -> float:
    return min(
        state.average_sender_transposition,
        state.average_receiver_transposition,
    )


def transposition_gap(state: GenerationSummary) -> float:
    return abs(
        state.average_sender_transposition
        - state.average_receiver_transposition,
    )


def state_float(state: GenerationSummary | None, attribute: str) -> str:
    if state is None:
        return "not_reached"

    if attribute == "average_min_transposition":
        return format_float(min_transposition(state))

    return format_float(getattr(state, attribute))


def mean_float_field(rows: list[dict[str, str]], field: str) -> str:
    return format_float(mean(float(row[field]) for row in rows))


def mean_field(rows: list[dict[str, str]], field: str) -> str:
    values = [
        float(row[field])
        for row in rows
        if row[field] not in {"not_reached", "not_applicable"}
    ]
    if not values:
        return "not_reached"

    return format_float(mean(values))


def generation_value(generation: int | None) -> str:
    if generation is None:
        return "not_reached"

    return str(generation)


def bool_value(value: bool) -> str:
    return str(value).lower()


def fraction(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(items) / len(items)


def format_float(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
