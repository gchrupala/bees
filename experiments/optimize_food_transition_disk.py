"""Optimize the vertical-gravity transition under the *disk* food ecology.

This is the disk-geometry sibling of ``optimize_food_transition.py``. The
stability / collapse / progress objective is identical -- a seed is "stable"
when the comb has tilted vertical and both sender and receiver read a
gravity-referenced (transposition) code -- so the shared thresholds and
per-seed evaluation are imported rather than re-derived. What differs is the
sampled parameter space: the disk model works in metres with lognormal patch
radii, gamma forays, area-scaled capacity and Poisson site counts, so the
angular-scale ranges of the legacy optimizer do not apply here.

Multi-node friendliness: many array tasks may share a single Optuna
``JournalStorage`` file on a shared filesystem. Pass ``--sampler-seed-offset``
(e.g. ``SLURM_ARRAY_TASK_ID * workers``) so each task's worker processes draw
distinct sampler seeds instead of re-exploring the same TPE start. Use
``--no-export`` on the optimizing tasks and a final ``--n-trials 0 --export``
pass to write the merged trial/seed CSVs once from the shared study.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from multiprocessing import Process
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
except ModuleNotFoundError as exc:  # pragma: no cover - exercised before install.
    raise SystemExit(
        "Optuna is required for this experiment. Install project dependencies "
        "with `python -m pip install -e .`."
    ) from exc

from bees.model import DirectionSettings, simulate

# Reuse the shared stability definition and formatting helpers so the disk
# search stays consistent with the angular optimizer.
from optimize_food_transition import (  # noqa: E402
    Thresholds,
    attr_value,
    bool_string,
    format_attr_float,
    format_metric_float,
    format_optional_float,
    param_value,
    parse_ints,
    relative,
    split_trials,
    suggest_float_or_fixed,
)


TRIAL_FIELDNAMES = [
    "number",
    "state",
    "value",
    "food_site_count",
    "food_site_radius",
    "food_site_radius_log_sd",
    "food_site_capacity",
    "food_value",
    "vertical_comb_benefit",
    "food_site_min_distance",
    "food_site_max_distance",
    "foray_shape",
    "travel_cost_per_distance",
    "mutation_sd",
    "transposition_mutation_correlation",
    "stable_count",
    "seed_count",
    "collapse_count",
    "mean_progress",
    "mean_final_success",
    "mean_final_payoff",
    "mean_final_comb_tilt",
    "mean_final_min_transposition",
    "mean_final_dance_propensity",
    "elapsed_seconds",
]

SEED_FIELDNAMES = [
    "number",
    "state",
    "value",
    "seed",
    "food_site_count",
    "food_site_radius",
    "food_site_capacity",
    "vertical_comb_benefit",
    "food_site_max_distance",
    "travel_cost_per_distance",
    "mutation_sd",
    "transposition_mutation_correlation",
    "stable",
    "collapsed",
    "progress",
    "final_success",
    "final_payoff",
    "final_comb_tilt",
    "final_sender_transposition",
    "final_receiver_transposition",
    "final_min_transposition",
    "final_dance_propensity",
]


@dataclass(frozen=True)
class DiskSearchSpace:
    food_site_count_min: int
    food_site_count_max: int
    food_site_radius_min: float
    food_site_radius_max: float
    food_site_radius_step: float
    food_site_capacity_min: int
    food_site_capacity_max: int
    food_value_min: float
    food_value_max: float
    food_value_step: float
    vertical_comb_benefit_min: float
    vertical_comb_benefit_max: float
    vertical_comb_benefit_step: float
    food_site_max_distance_min: float
    food_site_max_distance_max: float
    food_site_max_distance_step: float
    travel_cost_min: float
    travel_cost_max: float
    travel_cost_step: float
    mutation_sd_min: float
    mutation_sd_max: float
    mutation_sd_step: float
    transposition_mutation_correlation_min: float
    transposition_mutation_correlation_max: float
    transposition_mutation_correlation_step: float


@dataclass(frozen=True)
class SampledSettings:
    settings: DirectionSettings
    values: dict[str, int | float]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if base_settings.food_geometry != "disk":
        raise SystemExit(
            "optimize_food_transition_disk expects a disk-geometry config "
            f"(food_geometry='disk'); got {base_settings.food_geometry!r}."
        )
    if args.generations is not None:
        base_settings = replace(base_settings, generations=args.generations)

    seeds = parse_ints(args.seeds)
    thresholds = Thresholds(
        gravity=args.gravity_threshold,
        vertical=args.vertical_threshold,
        collapse_success=args.collapse_success_threshold,
    )
    search_space = build_search_space(args)

    args.journal_output.parent.mkdir(parents=True, exist_ok=True)
    args.trials_output.parent.mkdir(parents=True, exist_ok=True)
    if args.seed_output is not None:
        args.seed_output.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    worker_trial_counts = split_trials(args.n_trials, args.workers)
    workers = [
        Process(
            target=run_worker,
            kwargs={
                "worker_index": worker_index,
                "n_trials": n_trials,
                "args": args,
                "base_settings": base_settings,
                "seeds": seeds,
                "thresholds": thresholds,
                "search_space": search_space,
            },
        )
        for worker_index, n_trials in enumerate(worker_trial_counts)
        if n_trials > 0
    ]
    if len(workers) == 1:
        # Single worker: run inline so exceptions surface directly.
        run_worker(
            worker_index=0,
            n_trials=worker_trial_counts[0],
            args=args,
            base_settings=base_settings,
            seeds=seeds,
            thresholds=thresholds,
            search_space=search_space,
        )
    else:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise SystemExit(
                    f"worker process {worker.pid} exited with {worker.exitcode}"
                )

    if not args.export:
        print(
            f"optimized {args.n_trials} trials into "
            f"{relative(args.journal_output)} in {perf_counter() - started:.1f}s "
            f"(export skipped)",
            file=sys.stderr,
            flush=True,
        )
        return

    study = create_study(args, worker_index=0)
    export_trials(study, args.trials_output)
    if args.seed_output is not None:
        export_seed_metrics(study, args.seed_output)
    print(
        f"wrote {relative(args.trials_output)} from {len(study.trials)} trials "
        f"in {perf_counter() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize the disk-ecology vertical transition with Optuna.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "long_vertical_transition_disk.json",
        help="Path to the base disk-geometry long-transition config.",
    )
    parser.add_argument(
        "--journal-output",
        type=Path,
        default=ROOT / "results" / "food_transition_disk_optuna.journal",
        help="Optuna JournalStorage file for the persistent study.",
    )
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=ROOT / "results" / "food_transition_disk_optuna_trials.csv",
        help="CSV export of all completed/pruned trials.",
    )
    parser.add_argument(
        "--seed-output",
        type=Path,
        default=ROOT / "results" / "food_transition_disk_optuna_seed_metrics.csv",
        help="Optional CSV export of per-seed metrics for each completed trial.",
    )
    parser.add_argument(
        "--study-name",
        default="food_transition_disk_optuna",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=512,
        help="Total number of trials across this task's workers (0 = export only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes sharing the study journal.",
    )
    parser.add_argument(
        "--seeds",
        default="100-109",
        help="Comma-separated simulation seeds evaluated for each trial.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override the generation count from the config.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=2026,
        help="Base random seed for Optuna samplers.",
    )
    parser.add_argument(
        "--sampler-seed-offset",
        type=int,
        default=0,
        help=(
            "Added to the sampler seed for every worker in this task. Set to "
            "SLURM_ARRAY_TASK_ID * workers so array tasks sharing one journal "
            "explore distinct TPE trajectories."
        ),
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=64,
        help="Number of initial random TPE trials.",
    )
    parser.add_argument(
        "--pruner",
        choices=("none", "median"),
        default="none",
        help="Optional Optuna pruner. Median pruning compares partial seed panels.",
    )
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument(
        "--export",
        dest="export",
        action="store_true",
        help="Export merged trial/seed CSVs after optimizing (default).",
    )
    export_group.add_argument(
        "--no-export",
        dest="export",
        action="store_false",
        help="Optimize only; skip CSV export (for array tasks sharing a journal).",
    )
    parser.set_defaults(export=True)

    # Disk search space. Distances/radii are in metres; the config's operating
    # point (radius 150 m, distances 750-6000 m, travel cost ~2.7e-5) sits
    # inside these ranges. food_site_count is the Poisson mean.
    parser.add_argument("--food-site-count-min", type=int, default=1)
    parser.add_argument("--food-site-count-max", type=int, default=8)
    parser.add_argument("--food-site-radius-min", type=float, default=60.0)
    parser.add_argument("--food-site-radius-max", type=float, default=360.0)
    parser.add_argument("--food-site-radius-step", type=float, default=15.0)
    parser.add_argument("--food-site-capacity-min", type=int, default=2)
    parser.add_argument("--food-site-capacity-max", type=int, default=12)
    parser.add_argument("--food-value-min", type=float, default=1.0)
    parser.add_argument("--food-value-max", type=float, default=1.0)
    parser.add_argument("--food-value-step", type=float, default=0.1)
    parser.add_argument("--vertical-comb-benefit-min", type=float, default=0.10)
    parser.add_argument("--vertical-comb-benefit-max", type=float, default=0.60)
    parser.add_argument("--vertical-comb-benefit-step", type=float, default=0.02)
    parser.add_argument("--food-site-max-distance-min", type=float, default=3000.0)
    parser.add_argument("--food-site-max-distance-max", type=float, default=7500.0)
    parser.add_argument("--food-site-max-distance-step", type=float, default=250.0)
    parser.add_argument("--travel-cost-min", type=float, default=1.0e-5)
    parser.add_argument("--travel-cost-max", type=float, default=6.0e-5)
    parser.add_argument("--travel-cost-step", type=float, default=2.5e-6)
    parser.add_argument("--mutation-sd-min", type=float, default=0.04)
    parser.add_argument("--mutation-sd-max", type=float, default=0.14)
    parser.add_argument("--mutation-sd-step", type=float, default=0.01)
    parser.add_argument(
        "--transposition-mutation-correlation-min", type=float, default=0.0
    )
    parser.add_argument(
        "--transposition-mutation-correlation-max", type=float, default=1.0
    )
    parser.add_argument(
        "--transposition-mutation-correlation-step", type=float, default=0.1
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
        help="A seed is marked collapsed if success falls at or below this.",
    )
    return parser.parse_args()


def build_search_space(args: argparse.Namespace) -> DiskSearchSpace:
    return DiskSearchSpace(
        food_site_count_min=args.food_site_count_min,
        food_site_count_max=args.food_site_count_max,
        food_site_radius_min=args.food_site_radius_min,
        food_site_radius_max=args.food_site_radius_max,
        food_site_radius_step=args.food_site_radius_step,
        food_site_capacity_min=args.food_site_capacity_min,
        food_site_capacity_max=args.food_site_capacity_max,
        food_value_min=args.food_value_min,
        food_value_max=args.food_value_max,
        food_value_step=args.food_value_step,
        vertical_comb_benefit_min=args.vertical_comb_benefit_min,
        vertical_comb_benefit_max=args.vertical_comb_benefit_max,
        vertical_comb_benefit_step=args.vertical_comb_benefit_step,
        food_site_max_distance_min=args.food_site_max_distance_min,
        food_site_max_distance_max=args.food_site_max_distance_max,
        food_site_max_distance_step=args.food_site_max_distance_step,
        travel_cost_min=args.travel_cost_min,
        travel_cost_max=args.travel_cost_max,
        travel_cost_step=args.travel_cost_step,
        mutation_sd_min=args.mutation_sd_min,
        mutation_sd_max=args.mutation_sd_max,
        mutation_sd_step=args.mutation_sd_step,
        transposition_mutation_correlation_min=(
            args.transposition_mutation_correlation_min
        ),
        transposition_mutation_correlation_max=(
            args.transposition_mutation_correlation_max
        ),
        transposition_mutation_correlation_step=(
            args.transposition_mutation_correlation_step
        ),
    )


def run_worker(
    worker_index: int,
    n_trials: int,
    args: argparse.Namespace,
    base_settings: DirectionSettings,
    seeds: list[int],
    thresholds: Thresholds,
    search_space: DiskSearchSpace,
) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = create_study(args, worker_index)
    objective = DiskTransitionObjective(
        base_settings=base_settings,
        seeds=seeds,
        thresholds=thresholds,
        search_space=search_space,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[print_progress],
        gc_after_trial=True,
    )


class DiskTransitionObjective:
    def __init__(
        self,
        base_settings: DirectionSettings,
        seeds: list[int],
        thresholds: Thresholds,
        search_space: DiskSearchSpace,
    ) -> None:
        self.base_settings = base_settings
        self.seeds = seeds
        self.thresholds = thresholds
        self.search_space = search_space

    def __call__(self, trial: optuna.Trial) -> float:
        started = perf_counter()
        sample = sample_settings(trial, self.base_settings, self.search_space)
        for name, value in sample.values.items():
            trial.set_user_attr(f"setting_{name}", value)
        # Record disk settings held fixed from the config so the exported CSV
        # is self-describing even though these are not sampled.
        trial.set_user_attr(
            "setting_food_site_min_distance", sample.settings.food_site_min_distance
        )
        trial.set_user_attr("setting_foray_shape", sample.settings.foray_shape)
        trial.set_user_attr(
            "setting_food_site_radius_log_sd",
            sample.settings.food_site_radius_log_sd,
        )

        seed_scores = []
        seed_metrics = []
        for step, seed in enumerate(self.seeds, start=1):
            metrics = evaluate_seed(sample.settings, seed, self.thresholds)
            metrics["seed"] = seed
            seed_metrics.append(metrics)
            seed_scores.append(metrics["score"])
            trial.report(mean(seed_scores), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        stable_count = sum(metric["stable"] for metric in seed_metrics)
        collapse_count = sum(metric["collapsed"] for metric in seed_metrics)
        trial.set_user_attr("stable_count", stable_count)
        trial.set_user_attr("seed_count", len(seed_metrics))
        trial.set_user_attr("collapse_count", collapse_count)
        trial.set_user_attr(
            "mean_progress", mean(m["progress"] for m in seed_metrics)
        )
        trial.set_user_attr(
            "mean_final_success", mean(m["final_success"] for m in seed_metrics)
        )
        trial.set_user_attr(
            "mean_final_payoff", mean(m["final_payoff"] for m in seed_metrics)
        )
        trial.set_user_attr(
            "mean_final_comb_tilt", mean(m["final_comb_tilt"] for m in seed_metrics)
        )
        trial.set_user_attr(
            "mean_final_min_transposition",
            mean(m["final_min_transposition"] for m in seed_metrics),
        )
        trial.set_user_attr(
            "mean_final_dance_propensity",
            mean(m["final_dance_propensity"] for m in seed_metrics),
        )
        trial.set_user_attr("seed_metrics", seed_metrics)
        trial.set_user_attr("elapsed_seconds", perf_counter() - started)

        # Same objective shape as the angular search: an extra stable seed
        # always dominates near-miss progress; collapses subtract.
        return (
            stable_count
            + mean(m["progress"] for m in seed_metrics)
            - collapse_count
        )


def sample_settings(
    trial: optuna.Trial,
    base_settings: DirectionSettings,
    search_space: DiskSearchSpace,
) -> SampledSettings:
    food_site_count = trial.suggest_int(
        "food_site_count",
        search_space.food_site_count_min,
        search_space.food_site_count_max,
    )
    food_site_radius = trial.suggest_float(
        "food_site_radius",
        search_space.food_site_radius_min,
        search_space.food_site_radius_max,
        step=search_space.food_site_radius_step,
    )
    food_site_capacity = trial.suggest_int(
        "food_site_capacity",
        search_space.food_site_capacity_min,
        search_space.food_site_capacity_max,
    )
    food_value = suggest_float_or_fixed(
        trial,
        "food_value",
        search_space.food_value_min,
        search_space.food_value_max,
        search_space.food_value_step,
    )
    vertical_comb_benefit = trial.suggest_float(
        "vertical_comb_benefit",
        search_space.vertical_comb_benefit_min,
        search_space.vertical_comb_benefit_max,
        step=search_space.vertical_comb_benefit_step,
    )
    food_site_max_distance = trial.suggest_float(
        "food_site_max_distance",
        search_space.food_site_max_distance_min,
        search_space.food_site_max_distance_max,
        step=search_space.food_site_max_distance_step,
    )
    travel_cost = trial.suggest_float(
        "travel_cost_per_distance",
        search_space.travel_cost_min,
        search_space.travel_cost_max,
        step=search_space.travel_cost_step,
    )
    mutation_sd = suggest_float_or_fixed(
        trial,
        "mutation_sd",
        search_space.mutation_sd_min,
        search_space.mutation_sd_max,
        search_space.mutation_sd_step,
    )
    transposition_mutation_correlation = suggest_float_or_fixed(
        trial,
        "transposition_mutation_correlation",
        search_space.transposition_mutation_correlation_min,
        search_space.transposition_mutation_correlation_max,
        search_space.transposition_mutation_correlation_step,
    )

    values = {
        "food_site_count": food_site_count,
        "food_site_radius": food_site_radius,
        "food_site_capacity": food_site_capacity,
        "food_value": food_value,
        "vertical_comb_benefit": vertical_comb_benefit,
        "food_site_max_distance": food_site_max_distance,
        "travel_cost_per_distance": travel_cost,
        "mutation_sd": mutation_sd,
        "transposition_mutation_correlation": transposition_mutation_correlation,
    }
    return SampledSettings(
        settings=replace(
            base_settings,
            initial_comb_tilt=0.0,
            vertical_comb_modifier="linear",
            food_geometry="disk",
            food_site_count=food_site_count,
            food_site_radius=food_site_radius,
            food_site_capacity=food_site_capacity,
            food_value=food_value,
            vertical_comb_benefit=vertical_comb_benefit,
            food_site_max_distance=food_site_max_distance,
            # Keep foray range able to reach the farthest patches.
            max_search_distance=food_site_max_distance,
            travel_cost_per_distance=travel_cost,
            mutation_sd=mutation_sd,
            transposition_mutation_correlation=transposition_mutation_correlation,
        ),
        values=values,
    )


def evaluate_seed(
    settings: DirectionSettings,
    seed: int,
    thresholds: Thresholds,
) -> dict[str, float | bool]:
    """Disk-ecology per-seed evaluation. The stability / collapse / progress
    definitions match the angular optimizer; this variant also records the
    evolved dance-propensity trait, which is inert in the angular model."""
    history = simulate(settings, seed=seed)
    final = history[-1]
    min_success = min(state.average_success_rate for state in history)
    final_min_transposition = min(
        final.average_sender_transposition,
        final.average_receiver_transposition,
    )
    stable = (
        final.average_comb_tilt >= thresholds.vertical
        and final.average_sender_transposition >= thresholds.gravity
        and final.average_receiver_transposition >= thresholds.gravity
    )
    collapsed = min_success <= thresholds.collapse_success
    progress = min(
        1.0,
        final.average_comb_tilt / thresholds.vertical,
        final.average_sender_transposition / thresholds.gravity,
        final.average_receiver_transposition / thresholds.gravity,
    )
    return {
        "score": progress + (1.0 if stable else 0.0) - (1.0 if collapsed else 0.0),
        "progress": progress,
        "stable": stable,
        "collapsed": collapsed,
        "final_success": final.average_success_rate,
        "final_payoff": final.average_payoff,
        "final_comb_tilt": final.average_comb_tilt,
        "final_sender_transposition": final.average_sender_transposition,
        "final_receiver_transposition": final.average_receiver_transposition,
        "final_min_transposition": final_min_transposition,
        "final_dance_propensity": final.average_dance_propensity,
    }


def create_study(args: argparse.Namespace, worker_index: int) -> optuna.Study:
    storage = JournalStorage(JournalFileBackend(str(args.journal_output)))
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed + args.sampler_seed_offset + worker_index,
        n_startup_trials=args.startup_trials,
    )
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=args.startup_trials)
    else:
        pruner = optuna.pruners.NopPruner()

    return optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )


def print_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    print(
        f"trial={trial.number} value={trial.value:.3f} "
        f"stable={trial.user_attrs.get('stable_count')}/"
        f"{trial.user_attrs.get('seed_count')} "
        f"progress={trial.user_attrs.get('mean_progress'):.3f} "
        f"tilt={trial.user_attrs.get('mean_final_comb_tilt'):.3f} "
        f"m={trial.user_attrs.get('mean_final_min_transposition'):.3f} "
        f"success={trial.user_attrs.get('mean_final_success'):.3f}",
        file=sys.stderr,
        flush=True,
    )


def export_trials(study: optuna.Study, output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=TRIAL_FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for trial in study.trials:
            writer.writerow(trial_row(trial))


def export_seed_metrics(study: optuna.Study, output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SEED_FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for trial in study.trials:
            for metric in trial.user_attrs.get("seed_metrics", []):
                writer.writerow(seed_row(trial, metric))


def trial_row(trial: optuna.trial.FrozenTrial) -> dict[str, str]:
    return {
        "number": str(trial.number),
        "state": trial.state.name,
        "value": format_optional_float(trial.value),
        "food_site_count": param_value(trial, "food_site_count"),
        "food_site_radius": param_value(trial, "food_site_radius"),
        "food_site_radius_log_sd": attr_value(
            trial, "setting_food_site_radius_log_sd"
        ),
        "food_site_capacity": param_value(trial, "food_site_capacity"),
        "food_value": param_value(trial, "food_value"),
        "vertical_comb_benefit": param_value(trial, "vertical_comb_benefit"),
        "food_site_min_distance": attr_value(
            trial, "setting_food_site_min_distance"
        ),
        "food_site_max_distance": param_value(trial, "food_site_max_distance"),
        "foray_shape": attr_value(trial, "setting_foray_shape"),
        "travel_cost_per_distance": format_travel_cost(trial),
        "mutation_sd": param_value(trial, "mutation_sd"),
        "transposition_mutation_correlation": param_value(
            trial, "transposition_mutation_correlation"
        ),
        "stable_count": attr_value(trial, "stable_count"),
        "seed_count": attr_value(trial, "seed_count"),
        "collapse_count": attr_value(trial, "collapse_count"),
        "mean_progress": format_attr_float(trial, "mean_progress"),
        "mean_final_success": format_attr_float(trial, "mean_final_success"),
        "mean_final_payoff": format_attr_float(trial, "mean_final_payoff"),
        "mean_final_comb_tilt": format_attr_float(trial, "mean_final_comb_tilt"),
        "mean_final_min_transposition": format_attr_float(
            trial, "mean_final_min_transposition"
        ),
        "mean_final_dance_propensity": format_attr_float(
            trial, "mean_final_dance_propensity"
        ),
        "elapsed_seconds": format_attr_float(trial, "elapsed_seconds"),
    }


def seed_row(
    trial: optuna.trial.FrozenTrial,
    metric: dict[str, int | float | bool],
) -> dict[str, str]:
    return {
        "number": str(trial.number),
        "state": trial.state.name,
        "value": format_optional_float(trial.value),
        "seed": str(metric["seed"]),
        "food_site_count": param_value(trial, "food_site_count"),
        "food_site_radius": param_value(trial, "food_site_radius"),
        "food_site_capacity": param_value(trial, "food_site_capacity"),
        "vertical_comb_benefit": param_value(trial, "vertical_comb_benefit"),
        "food_site_max_distance": param_value(trial, "food_site_max_distance"),
        "travel_cost_per_distance": format_travel_cost(trial),
        "mutation_sd": param_value(trial, "mutation_sd"),
        "transposition_mutation_correlation": param_value(
            trial, "transposition_mutation_correlation"
        ),
        "stable": bool_string(metric["stable"]),
        "collapsed": bool_string(metric["collapsed"]),
        "progress": format_metric_float(metric, "progress"),
        "final_success": format_metric_float(metric, "final_success"),
        "final_payoff": format_metric_float(metric, "final_payoff"),
        "final_comb_tilt": format_metric_float(metric, "final_comb_tilt"),
        "final_sender_transposition": format_metric_float(
            metric, "final_sender_transposition"
        ),
        "final_receiver_transposition": format_metric_float(
            metric, "final_receiver_transposition"
        ),
        "final_min_transposition": format_metric_float(
            metric, "final_min_transposition"
        ),
        "final_dance_propensity": format_metric_float(
            metric, "final_dance_propensity"
        ),
    }


def format_travel_cost(trial: optuna.trial.FrozenTrial) -> str:
    """Travel cost is ~1e-5 in the disk model, so the shared ``.3f`` cell
    formatter would round it to zero. Keep scientific precision instead."""
    value = trial.params.get(
        "travel_cost_per_distance",
        trial.user_attrs.get("setting_travel_cost_per_distance"),
    )
    if value is None or value == "":
        return ""
    return f"{float(value):.6e}"


def load_settings(config_path: Path) -> DirectionSettings:
    config = json.loads(config_path.read_text())
    config.pop("seed", None)
    return DirectionSettings(**config)


if __name__ == "__main__":
    main()
