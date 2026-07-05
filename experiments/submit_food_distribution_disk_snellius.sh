#!/usr/bin/env bash
set -euo pipefail

job="$(
    sbatch --parsable \
        experiments/run_food_distribution_disk_snellius.sbatch
)"

echo "submitted food-distribution disk job ${job}"
