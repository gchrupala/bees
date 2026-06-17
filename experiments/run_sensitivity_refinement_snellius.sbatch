#!/usr/bin/env bash
#SBATCH --job-name=bees-refine
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${BEES_MODULE_LOAD:-}" ]]; then
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

push_arg=()
if [[ "${BEES_PUSH:-0}" == "1" ]]; then
    push_arg=(--push)
fi

"${PYTHON_BIN}" -u experiments/run_sensitivity_refinement_pipeline.py "${push_arg[@]}"
