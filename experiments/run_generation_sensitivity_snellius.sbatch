#!/usr/bin/env bash
#SBATCH --job-name=bees-gen-sens
#SBATCH --time=36:00:00
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

generation_arg=()
if [[ -n "${BEES_GENERATIONS:-}" ]]; then
    generation_arg=(--generations "${BEES_GENERATIONS}")
fi

seed_arg=()
if [[ -n "${BEES_SEEDS:-}" ]]; then
    seed_arg=(--seeds "${BEES_SEEDS}")
fi

push_arg=()
if [[ "${BEES_PUSH:-0}" == "1" ]]; then
    push_arg=(--push)
fi

"${PYTHON_BIN}" -u experiments/run_generation_sensitivity.py \
    --baseline-points results/food_transition_v2_validation_points.csv \
    --baseline-group-summary results/food_transition_v2_validation_group_summary.csv \
    --max-workers "${SLURM_CPUS_PER_TASK}" \
    "${generation_arg[@]}" \
    "${seed_arg[@]}" \
    "${push_arg[@]}"
