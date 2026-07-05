#!/usr/bin/env bash
#SBATCH --job-name=bees-disk-opt
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# One array task = one node's worth of Optuna workers. Every task in the array
# optimizes the SAME study by pointing at a single JournalStorage file on the
# shared filesystem (Optuna's symlink lock is NFS/GPFS-safe). The per-task
# sampler-seed offset keeps each node exploring a distinct TPE trajectory
# instead of re-drawing the same startup trials. Tasks run with --no-export;
# the finalizer writes the merged trial/seed CSVs once.

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

task_id="${SLURM_ARRAY_TASK_ID:-0}"
workers="${SLURM_CPUS_PER_TASK:-32}"
# Distinct sampler seeds per array task: task 0 -> [base+0 .. base+W-1],
# task 1 -> [base+W .. base+2W-1], and so on.
seed_offset=$(( task_id * workers ))

"${PYTHON_BIN}" -u experiments/optimize_food_transition_disk.py \
    --no-export \
    --workers "${workers}" \
    --sampler-seed-offset "${seed_offset}" \
    --n-trials "${BEES_DISK_OPTUNA_TRIALS_PER_TASK:-64}" \
    --startup-trials "${BEES_DISK_OPTUNA_STARTUP:-64}" \
    --seeds "${BEES_DISK_OPTUNA_SEEDS:-100-109}" \
    ${BEES_DISK_OPTUNA_GENERATIONS:+--generations "${BEES_DISK_OPTUNA_GENERATIONS}"} \
    --config "${BEES_CONFIG:-configs/long_vertical_transition_disk.json}" \
    --journal-output "${BEES_OPTUNA_JOURNAL:-results/food_transition_disk_optuna.journal}"
