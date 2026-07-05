"""One-at-a-time (OAT) sensitivity around a disk-ecology transition baseline.

The disk sibling of ``run_food_transition_oat_sensitivity.py``. It takes a
baseline candidate (the top validated disk candidate, or the base config's own
parameters), perturbs each disk parameter one at a time along a coarse ladder,
and reruns every point on a held-out seed panel using the shared disk stability
definition. Distances and radii are in metres, so the ladders are metric-scale.

Outputs ``<prefix>_points.csv`` (one row per perturbation point, tagged with the
varied parameter and whether it is the baseline) and ``<prefix>_group_summary.csv``
(stable/collapse/success aggregates per point), reusing the panel evaluator.
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
    INT_PARAMS,
    candidate_params,
    evaluate_candidates,
    format_param,
    read_rows,
)

# Per-parameter OAT ladder: additive deltas around the baseline, clamped to the
# search bounds. food_value is held fixed (not a search dimension).
LADDERS: dict[str, tuple[tuple[float, ...], float, float]] = {
    "food_site_count": ((-3, -2, -1, 1, 2), 1, 10),
    "food_site_radius": ((-90, -45, 45, 90), 60, 360),
    "food_site_capacity": ((-3, -2, -1, 1, 2), 1, 14),
    "vertical_comb_benefit": ((-0.12, -0.08, -0.04, 0.04), 0.10, 0.60),
    "food_site_max_distance": ((-1500, -750, 750, 1500), 3000, 7500),
    "travel_cost_per_distance": ((-1.5e-5, -0.5e-5, 0.5e-5, 1.5e-5), 1.0e-5, 6.0e-5),
    "mutation_sd": ((-0.05, -0.03, -0.02, 0.02), 0.04, 0.14),
    "transposition_mutation_correlation": ((-0.5, -0.25, 0.25, 0.5), 0.0, 1.0),
}

POINT_FIELDNAMES = ["point", "parameter", "parameter_value", "is_baseline", *DISK_SEARCH_PARAMS]


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
    points = build_points(baseline)

    prefix = args.output_prefix
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    write_points(Path(f"{prefix}_points.csv"), points)

    started = perf_counter()
    candidates = [(name, params) for name, params, *_ in points]
    summaries = evaluate_candidates(
        base_settings, candidates, seeds, thresholds, args.max_workers
    )
    write_summary(Path(f"{prefix}_group_summary.csv"), summaries)
    print(
        f"wrote {relative(Path(f'{prefix}_group_summary.csv'))} for {len(points)} "
        f"points over {len(seeds)} seeds in {perf_counter() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline-points",
        type=Path,
        help="Validation/confirmation points CSV to draw the baseline candidate from.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        help="Matching group summary; the top-stable candidate becomes the baseline.",
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--output-prefix", required=True)
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
    # Fallback: the base config's own parameters.
    return {name: getattr(base_settings, name) for name in DISK_SEARCH_PARAMS}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_points(baseline: dict[str, int | float]) -> list[tuple]:
    """Return (name, params, parameter, parameter_value, is_baseline) tuples: the
    baseline once, then each parameter's clamped, de-duplicated ladder."""
    points: list[tuple] = [
        ("baseline", dict(baseline), "baseline", "", True),
    ]
    for name, (deltas, lo, hi) in LADDERS.items():
        seen: set[float] = {float(baseline[name])}
        for delta in deltas:
            raw = baseline[name] + delta
            value = round(clamp(raw, lo, hi)) if name in INT_PARAMS else clamp(raw, lo, hi)
            if float(value) in seen:
                continue
            seen.add(float(value))
            params = dict(baseline)
            params[name] = value
            label = format_param(name, value)
            points.append((f"{name}={label}", params, name, label, False))
    return points


def write_points(path: Path, points: list[tuple]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for name, params, parameter, parameter_value, is_baseline in points:
            row = {
                "point": name,
                "parameter": parameter,
                "parameter_value": parameter_value,
                "is_baseline": str(is_baseline).lower(),
            }
            row.update({k: format_param(k, params[k]) for k in DISK_SEARCH_PARAMS})
            writer.writerow(row)


def write_summary(path: Path, summaries: list[dict[str, str]]) -> None:
    from run_food_transition_disk_panel import SUMMARY_FIELDNAMES

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
