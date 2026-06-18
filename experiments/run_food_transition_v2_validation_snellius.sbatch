#!/usr/bin/env bash
#SBATCH --job-name=bees-v2-valid
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
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

"${PYTHON_BIN}" -u experiments/run_food_transition_v2_candidate_panel.py \
    --source panel \
    --points results/food_transition_v2_confirmation_points.csv \
    --group-summary results/food_transition_v2_confirmation_group_summary.csv \
    --max-candidates "${BEES_V2_VALIDATION_CANDIDATES:-5}" \
    --seeds "${BEES_V2_VALIDATION_SEEDS:-200-299}" \
    --output-prefix results/food_transition_v2_validation \
    --max-workers "${SLURM_CPUS_PER_TASK}"
