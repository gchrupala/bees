from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RUNS = 1615


def main() -> None:
    args = parse_args()
    run(
        [
            sys.executable,
            "-u",
            "experiments/run_food_transition_oat_sensitivity.py",
            "--panel",
            "refinement",
            "--max-workers",
            str(args.max_workers),
        ]
    )

    finalize_command = [
        sys.executable,
        "-u",
        "experiments/finalize_sensitivity_refinement.py",
        "--expected-runs",
        str(EXPECTED_RUNS),
        "--commit-message",
        args.commit_message,
    ]
    if args.push:
        finalize_command.append("--push")
    run(finalize_command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and finalize the refined food-transition sensitivity panel."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum worker processes for the simulation phase.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add refined sensitivity results",
        help="Commit message for the generated results and report update.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the result commit after finalization.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
