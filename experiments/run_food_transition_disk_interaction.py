"""Evolutionary-parameter interaction grid for the disk-ecology transition.

The disk sibling of the angular interaction experiment. It holds the disk
ecology fixed at a baseline candidate and crosses the three evolutionary
parameters -- vertical-comb benefit, mutation scale, and sender-receiver
mutation correlation -- to map whether mutation parameters can compensate for
weaker architectural benefit.

The grid is re-centred on the disk stable region (higher benefit and mutation
than the angular grid), where disk transitions actually occur. Each cell is run
over a held-out seed panel with the shared disk stability definition. Output is
one summary row per cell.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from optimize_food_transition import Thresholds, parse_ints, relative  # noqa: E402
from optimize_food_transition_disk import DISK_SEARCH_PARAMS, load_settings  # noqa: E402
from run_food_transition_disk_panel import (  # noqa: E402
    candidate_params,
    evaluate_candidates,
    format_param,
    read_rows,
)

# Grid re-centred on the disk stable region (angular used 0.10/0.25/0.44 benefit,
# 0.045-0.135 mutation). Disk transitions concentrate near benefit ~0.56 and
# mutation ~0.11, so the grid brackets that with weaker values to test rescue.
VERTICAL_COMB_BENEFIT_VALUES = (0.30, 0.44, 0.56)
MUTATION_SD_VALUES = (0.05, 0.07, 0.09, 0.11)
TRANSPOSITION_MUTATION_CORRELATION_VALUES = (0.0, 0.3, 0.6, 0.9)

GRID_PARAMS = (
    "vertical_comb_benefit",
    "mutation_sd",
    "transposition_mutation_correlation",
)

CELL_FIELDNAMES = [
    "cell",
    *GRID_PARAMS,
    "seed_count",
    "stable_count",
    "collapse_count",
    "mean_final_success",
    "mean_final_comb_tilt",
    "mean_final_min_transposition",
    "mean_final_dance_propensity",
]


def main() -> None:
    args = parse_args()
    base_settings = load_settings(args.config)
    if base_settings.food_geometry != "disk":
        raise SystemExit("expects a disk-geometry config")
    if args.generations is not None:
        from dataclasses import replace

        base_settings = replace(base_settings, generations=args.generations)

    seeds = parse_ints(args.seeds)
    thresholds = Thresholds(
        gravity=args.gravity_threshold,
        vertical=args.vertical_threshold,
        collapse_success=args.collapse_success_threshold,
    )
    baseline = resolve_baseline(args, base_settings)
    cells = build_cells(baseline)

    candidates = [(name, params) for name, params, _ in cells]
    started = perf_counter()
    summaries = evaluate_candidates(
        base_settings, candidates, seeds, thresholds, args.max_workers
    )
    summary_by_name = {s["candidate"]: s for s in summaries}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_cells(output, cells, summary_by_name)
    print(
        f"wrote {relative(output)} for {len(cells)} cells over {len(seeds)} seeds "
        f"in {perf_counter() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-points", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--gravity-threshold", type=float, default=0.50)
    parser.add_argument("--vertical-threshold", type=float, default=0.80)
    parser.add_argument("--collapse-success-threshold", type=float, default=0.02)
    return parser.parse_args()


def resolve_baseline(args: argparse.Namespace, base_settings) -> dict[str, int | float]:
    if args.baseline_points and args.baseline_summary:
        summary = read_rows(args.baseline_summary)
        summary.sort(key=lambda r: float(r.get("stable_count") or 0), reverse=True)
        top = summary[0]["candidate"]
        params_by_name = {
            r["candidate"]: candidate_params(r, base_settings)
            for r in read_rows(args.baseline_points)
        }
        return params_by_name[top]
    return {name: getattr(base_settings, name) for name in DISK_SEARCH_PARAMS}


def build_cells(baseline: dict[str, int | float]) -> list[tuple]:
    """Return (cell_name, params, grid_values) for every grid combination, with
    the ecology fixed at the baseline and the three evolutionary params varied."""
    cells: list[tuple] = []
    for benefit in VERTICAL_COMB_BENEFIT_VALUES:
        for mutation in MUTATION_SD_VALUES:
            for correlation in TRANSPOSITION_MUTATION_CORRELATION_VALUES:
                grid = {
                    "vertical_comb_benefit": benefit,
                    "mutation_sd": mutation,
                    "transposition_mutation_correlation": correlation,
                }
                params = dict(baseline)
                params.update(grid)
                name = "_".join(
                    f"{p}{format_param(p, grid[p])}" for p in GRID_PARAMS
                )
                cells.append((name, params, grid))
    return cells


def write_cells(path: Path, cells: list[tuple], summary_by_name: dict) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CELL_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for name, _params, grid in cells:
            summary = summary_by_name[name]
            row = {"cell": name}
            row.update({p: format_param(p, grid[p]) for p in GRID_PARAMS})
            for key in CELL_FIELDNAMES[len(GRID_PARAMS) + 1 :]:
                row[key] = summary[key]
            writer.writerow(row)


if __name__ == "__main__":
    main()
