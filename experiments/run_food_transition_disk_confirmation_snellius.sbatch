#!/usr/bin/env bash
#SBATCH --job-name=bees-disk-confirm
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Confirmation: rerun the top distinct Optuna candidates on a held-out seed panel.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BEES_MODULE_LOAD="${BEES_MODULE_LOAD:-2025 Python/3.13.5-GCCcore-14.3.0}"
if [[ "${BEES_MODULE_LOAD}" != "none" ]]; then module load ${BEES_MODULE_LOAD}; fi
if [[ -n "${BEES_VENV:-}" ]]; then source "${BEES_VENV}/bin/activate"; elif [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON_BIN="${BEES_PYTHON:-python}"

"${PYTHON_BIN}" -u experiments/run_food_transition_disk_panel.py \
    --source trials \
    --config "${BEES_CONFIG:-configs/long_vertical_transition_disk.json}" \
    --trials "${BEES_OPTUNA_TRIALS_CSV:-results/food_transition_disk_optuna_trials.csv}" \
    --max-candidates "${BEES_DISK_CONFIRMATION_CANDIDATES:-20}" \
    --seeds "${BEES_DISK_CONFIRMATION_SEEDS:-110-149}" \
    --output-prefix "${BEES_CONFIRMATION_PREFIX:-results/food_transition_disk_confirmation}" \
    --max-workers "${SLURM_CPUS_PER_TASK}"
