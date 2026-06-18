from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_PREFIX = RESULTS / "food_transition_v2_evolutionary_interaction"
DEFAULT_SHARD_DIR = RESULTS / "food_transition_v2_evolutionary_interaction_shards"

EVOLUTIONARY_PARAMETERS = (
    "vertical_comb_benefit",
    "mutation_sd",
    "transposition_mutation_correlation",
)
JOB_FIELDNAMES = ["job_index", "point", "seed"]

VERTICAL_COMB_BENEFIT_VALUES = (0.10, 0.25, 0.44)
MUTATION_SD_VALUES = (0.045, 0.075, 0.090, 0.135)
TRANSPOSITION_MUTATION_CORRELATION_VALUES = (0.0, 0.3, 0.6, 0.9)


def expected_point_count() -> int:
    return (
        len(VERTICAL_COMB_BENEFIT_VALUES)
        * len(MUTATION_SD_VALUES)
        * len(TRANSPOSITION_MUTATION_CORRELATION_VALUES)
    )


def output_paths(prefix: Path = DEFAULT_PREFIX) -> dict[str, Path]:
    return {
        "points": prefix.with_name(f"{prefix.name}_points.csv"),
        "jobs": prefix.with_name(f"{prefix.name}_jobs.csv"),
        "events": prefix.with_name(f"{prefix.name}_events.csv"),
        "trajectories": prefix.with_name(f"{prefix.name}_trajectories.csv"),
        "group_summary": prefix.with_name(f"{prefix.name}_group_summary.csv"),
        "generation_summary": prefix.with_name(f"{prefix.name}_generation_summary.csv"),
    }


def shard_paths(shard_dir: Path, task_index: int) -> dict[str, Path]:
    stem = f"task_{task_index:05d}"
    return {
        "events": shard_dir / f"{stem}_events.csv",
        "trajectories": shard_dir / f"{stem}_trajectories.csv",
    }


def slug_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")
