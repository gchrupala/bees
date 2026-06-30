#!/usr/bin/env bash
#SBATCH --job-name=bees-v2-opt
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

"${PYTHON_BIN}" -u experiments/optimize_food_transition.py \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --n-trials "${BEES_V2_OPTUNA_TRIALS:-512}" \
    --seeds "${BEES_V2_OPTUNA_SEEDS:-100-109}" \
    --config "${BEES_CONFIG:-configs/long_vertical_transition.json}" \
    --journal-output "${BEES_OPTUNA_JOURNAL:-results/food_transition_v2_optuna.journal}" \
    --trials-output "${BEES_OPTUNA_TRIALS_CSV:-results/food_transition_v2_optuna_trials.csv}" \
    --seed-output "${BEES_OPTUNA_SEED_METRICS:-results/food_transition_v2_optuna_seed_metrics.csv}"
