#!/usr/bin/env bash
#SBATCH --job-name=bees-disk-valid
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Validation: rerun the top confirmation candidates on a larger held-out panel.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BEES_MODULE_LOAD="${BEES_MODULE_LOAD:-2025 Python/3.13.5-GCCcore-14.3.0}"
if [[ "${BEES_MODULE_LOAD}" != "none" ]]; then module load ${BEES_MODULE_LOAD}; fi
if [[ -n "${BEES_VENV:-}" ]]; then source "${BEES_VENV}/bin/activate"; elif [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON_BIN="${BEES_PYTHON:-python}"

CONFIRM_PREFIX="${BEES_CONFIRMATION_PREFIX:-results/food_transition_disk_confirmation}"

"${PYTHON_BIN}" -u experiments/run_food_transition_disk_panel.py \
    --source panel \
    --config "${BEES_CONFIG:-configs/long_vertical_transition_disk.json}" \
    --points "${CONFIRM_PREFIX}_points.csv" \
    --group-summary "${CONFIRM_PREFIX}_group_summary.csv" \
    --max-candidates "${BEES_DISK_VALIDATION_CANDIDATES:-5}" \
    --seeds "${BEES_DISK_VALIDATION_SEEDS:-200-299}" \
    --output-prefix "${BEES_VALIDATION_PREFIX:-results/food_transition_disk_validation}" \
    --max-workers "${SLURM_CPUS_PER_TASK}"
