#!/usr/bin/env bash
#SBATCH --job-name=bees-disk-inter
#SBATCH --time=24:00:00
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Evolutionary-parameter interaction grid on the validated disk baseline.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BEES_MODULE_LOAD="${BEES_MODULE_LOAD:-2025 Python/3.13.5-GCCcore-14.3.0}"
if [[ "${BEES_MODULE_LOAD}" != "none" ]]; then module load ${BEES_MODULE_LOAD}; fi
if [[ -n "${BEES_VENV:-}" ]]; then source "${BEES_VENV}/bin/activate"; elif [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON_BIN="${BEES_PYTHON:-python}"

VALID_PREFIX="${BEES_VALIDATION_PREFIX:-results/food_transition_disk_validation}"

"${PYTHON_BIN}" -u experiments/run_food_transition_disk_interaction.py \
    --config "${BEES_CONFIG:-configs/long_vertical_transition_disk.json}" \
    --baseline-points "${VALID_PREFIX}_points.csv" \
    --baseline-summary "${VALID_PREFIX}_group_summary.csv" \
    --seeds "${BEES_DISK_INTERACTION_SEEDS:-200-299}" \
    --output "${BEES_INTERACTION_OUTPUT:-results/food_transition_disk_interaction.csv}" \
    --max-workers "${SLURM_CPUS_PER_TASK}"
