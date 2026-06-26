#!/usr/bin/env bash
#SBATCH --job-name=bees-food-dist
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

BEES_MODULE_LOAD="${BEES_MODULE_LOAD:-2025 Python/3.13.5-GCCcore-14.3.0}"
if [[ "${BEES_MODULE_LOAD}" != "none" ]]; then
    module load ${BEES_MODULE_LOAD}
fi

if [[ -n "${BEES_VENV:-}" ]]; then
    source "${BEES_VENV}/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${BEES_PYTHON:-python}"

"${PYTHON_BIN}" -u experiments/run_food_distribution_v2.py \
    --seeds "${BEES_FD_SEEDS:-400-449}" \
    --output-prefix results/food_distribution_v2 \
    --max-workers "${SLURM_CPUS_PER_TASK}"
