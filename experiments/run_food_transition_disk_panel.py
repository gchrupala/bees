"""Confirmation / validation panel for the disk-ecology vertical transition.

The disk sibling of ``run_food_transition_v2_candidate_panel.py``. The angular
panel is wired to the angular parameter schema (``food_site_width`` etc.); this
one works in the disk schema and reuses the single disk stability definition
(``optimize_food_transition_disk.evaluate_seed``) so search, confirmation, and
validation all agree on what "stable" means.

Two stages, one script:

* ``--source trials``  -- confirmation. Read the Optuna trials CSV, take the top
  distinct candidates by stable-seed count, and rerun them on a held-out seed
  panel.
* ``--source panel``   -- validation. Read a previous panel's points and group
  summary, take the top candidates by stable count, and rerun them on a larger
  held-out panel.

Each stage writes ``<prefix>_points.csv`` (candidate parameters) and
``<prefix>_group_summary.csv`` (per-candidate aggregates over the panel).
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from bees.model import DirectionSettings  # noqa: E402

from optimize_food_transition import Thresholds, parse_ints, relative  # noqa: E402
from optimize_food_transition_disk import (  # noqa: E402
    DISK_SEARCH_PARAMS,
    build_disk_settings,
    evaluate_seed,
    load_settings,
)

INT_PARAMS = {"food_site_count", "food_site_capacity"}

POINT_FIELDNAMES = ["candidate", "source", *DISK_SEARCH_PARAMS]
SUMMARY_FIELDNAMES = [
    "candidate",
    "seed_count",
    "stable_count",
    "collapse_count",
    "mean_final_success",
    "mean_final_payoff",
    "mean_final_comb_tilt",
    "mean_final_min_transposition",
    "mean_final_dance_propensity",
]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if base_settings.food_geometry != "disk":
        raise SystemExit(
            "run_food_transition_disk_panel expects a disk-geometry config; "
            f"got food_geometry={base_settings.food_geometry!r}."
        )
    if args.generations is not None:
        from dataclasses import replace

        base_settings = replace(base_settings, generations=args.generations)

    seeds = parse_ints(args.seeds)
    thresholds = Thresholds(
        gravity=args.gravity_threshold,
        vertical=args.vertical_threshold,
        collapse_success=args.collapse_success_threshold,
    )
    candidates = select_candidates(args, base_settings)
    if not candidates:
        raise SystemExit("no candidates selected")

    points_path = Path(f"{args.output_prefix}_points.csv")
    summary_path = Path(f"{args.output_prefix}_group_summary.csv")
    points_path.parent.mkdir(parents=True, exist_ok=True)
    write_points(points_path, candidates, args.source)

    started = perf_counter()
    summaries = evaluate_candidates(
        base_settings, candidates, seeds, thresholds, args.max_workers
    )
    write_summary(summary_path, summaries)
    print(
        f"wrote {relative(summary_path)} for {len(candidates)} candidates over "
        f"{len(seeds)} seeds in {perf_counter() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("trials", "panel"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=Path, help="Optuna trials CSV (source=trials).")
    parser.add_argument("--points", type=Path, help="Prior points CSV (source=panel).")
    parser.add_argument(
        "--group-summary", type=Path, help="Prior group summary CSV (source=panel)."
    )
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--seeds", required=True, help="Held-out seed range, e.g. 110-149.")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--gravity-threshold", type=float, default=0.50)
    parser.add_argument("--vertical-threshold", type=float, default=0.80)
    parser.add_argument("--collapse-success-threshold", type=float, default=0.02)
    return parser.parse_args()


def cast_param(name: str, raw: str) -> int | float:
    return int(round(float(raw))) if name in INT_PARAMS else float(raw)


def candidate_params(
    row: dict[str, str], base_settings: DirectionSettings
) -> dict[str, int | float]:
    # Fixed parameters (e.g. food_value) are neither Optuna params nor stored
    # attrs, so their trial-CSV cell can be blank; fall back to the base config.
    params: dict[str, int | float] = {}
    for name in DISK_SEARCH_PARAMS:
        raw = (row.get(name) or "").strip()
        params[name] = cast_param(name, raw) if raw else getattr(base_settings, name)
    return params


def select_candidates(
    args: argparse.Namespace, base_settings: DirectionSettings
) -> list[tuple[str, dict]]:
    if args.source == "trials":
        if args.trials is None:
            raise SystemExit("--trials is required with --source trials")
        return select_from_trials(args.trials, args.max_candidates, base_settings)
    if args.points is None or args.group_summary is None:
        raise SystemExit("--points and --group-summary are required with --source panel")
    return select_from_panel(
        args.points, args.group_summary, args.max_candidates, base_settings
    )


def select_from_trials(
    path: Path, max_candidates: int, base_settings: DirectionSettings
) -> list[tuple[str, dict]]:
    rows = [r for r in read_rows(path) if r.get("state") == "COMPLETE"]

    def sort_key(row: dict[str, str]) -> tuple[float, float]:
        return (float(row.get("stable_count") or 0), float(row.get("value") or 0))

    unique: dict[tuple, dict] = {}
    for row in sorted(rows, key=sort_key, reverse=True):
        key = tuple(row[name] for name in DISK_SEARCH_PARAMS)
        unique.setdefault(key, row)
    selected = list(unique.values())[:max_candidates]
    return [
        (f"trial_{row['number']}", candidate_params(row, base_settings))
        for row in selected
    ]


def select_from_panel(
    points: Path,
    group_summary: Path,
    max_candidates: int,
    base_settings: DirectionSettings,
) -> list[tuple[str, dict]]:
    params_by_name = {
        r["candidate"]: candidate_params(r, base_settings) for r in read_rows(points)
    }
    summary = read_rows(group_summary)
    summary.sort(key=lambda r: float(r.get("stable_count") or 0), reverse=True)
    selected: list[tuple[str, dict]] = []
    for row in summary[:max_candidates]:
        name = row["candidate"]
        if name in params_by_name:
            selected.append((name, params_by_name[name]))
    return selected


def evaluate_candidates(
    base_settings: DirectionSettings,
    candidates: list[tuple[str, dict]],
    seeds: list[int],
    thresholds: Thresholds,
    max_workers: int,
) -> list[dict[str, str]]:
    jobs = [(name, params, seed) for name, params in candidates for seed in seeds]
    results: dict[str, list[dict]] = {name: [] for name, _ in candidates}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, base_settings, params, seed, thresholds): name
            for name, params, seed in jobs
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name].append(future.result())
    order = [name for name, _ in candidates]
    return [summarize(name, results[name]) for name in order]


def _run_one(
    base_settings: DirectionSettings,
    params: dict,
    seed: int,
    thresholds: Thresholds,
) -> dict:
    settings = build_disk_settings(base_settings, **params)
    return evaluate_seed(settings, seed, thresholds)


def summarize(name: str, metrics: list[dict]) -> dict[str, str]:
    return {
        "candidate": name,
        "seed_count": str(len(metrics)),
        "stable_count": str(sum(1 for m in metrics if m["stable"])),
        "collapse_count": str(sum(1 for m in metrics if m["collapsed"])),
        "mean_final_success": f"{mean(m['final_success'] for m in metrics):.3f}",
        "mean_final_payoff": f"{mean(m['final_payoff'] for m in metrics):.3f}",
        "mean_final_comb_tilt": f"{mean(m['final_comb_tilt'] for m in metrics):.3f}",
        "mean_final_min_transposition": (
            f"{mean(m['final_min_transposition'] for m in metrics):.3f}"
        ),
        "mean_final_dance_propensity": (
            f"{mean(m['final_dance_propensity'] for m in metrics):.3f}"
        ),
    }


def format_param(name: str, value: int | float) -> str:
    if name in INT_PARAMS:
        return str(int(value))
    if name == "travel_cost_per_distance":
        return f"{value:.6e}"
    return f"{value:.4f}"


def write_points(path: Path, candidates: list[tuple[str, dict]], source: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for name, params in candidates:
            row = {"candidate": name, "source": source}
            row.update({k: format_param(k, params[k]) for k in DISK_SEARCH_PARAMS})
            writer.writerow(row)


def write_summary(path: Path, summaries: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
