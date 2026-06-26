#!/usr/bin/env bash
set -euo pipefail

job="$(
    sbatch --parsable \
        experiments/run_food_distribution_v2_snellius.sbatch
)"

echo "submitted food-distribution v2 job ${job}"
