#!/usr/bin/env bash
set -euo pipefail

job="$(
    sbatch --parsable \
        experiments/run_generation_sensitivity_snellius.sbatch
)"

echo "submitted low-regime generation sensitivity job ${job}"
