#!/usr/bin/env bash
#SBATCH --job-name=bees-disk-finalize
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Export the merged trial and per-seed CSVs once from the shared study journal
# after the optimizing array finishes. Runs no simulations (--n-trials 0).

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

PYTHON_BIN="${BEES_PYTHON:-python}"

"${PYTHON_BIN}" -u experiments/optimize_food_transition_disk.py \
    --n-trials 0 \
    --export \
    --config "${BEES_CONFIG:-configs/long_vertical_transition_disk.json}" \
    --journal-output "${BEES_OPTUNA_JOURNAL:-results/food_transition_disk_optuna.journal}" \
    --trials-output "${BEES_OPTUNA_TRIALS_CSV:-results/food_transition_disk_optuna_trials.csv}" \
    --seed-output "${BEES_OPTUNA_SEED_METRICS:-results/food_transition_disk_optuna_seed_metrics.csv}"

if [[ "${BEES_PUSH:-0}" == "1" ]]; then
    git add \
        "${BEES_OPTUNA_TRIALS_CSV:-results/food_transition_disk_optuna_trials.csv}" \
        "${BEES_OPTUNA_SEED_METRICS:-results/food_transition_disk_optuna_seed_metrics.csv}"
    git commit -m "Add disk-ecology food-transition optuna results" || true
    git push || true
fi
