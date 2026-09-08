#!/usr/bin/env bash
#SBATCH --job-name=bees-assumption-sensitivity
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# Assumption-sensitivity re-run of the stage-1 food-distribution grid; see
# experiments/run_assumption_sensitivity.py for what each variant changes.
#
# One array task per variant, each writing its own pair of CSVs, so a failed or
# slow variant cannot hold up or corrupt the others. Every file carries a
# "variant" column, so the per-variant CSVs concatenate directly.
#
# Defaults need no --export; override with
#   sbatch --export=ALL,BEES_AS_SEEDS=400-449,BEES_PUSH=1 <this file>
# since Slurm does not inherit the submitting shell's environment here.

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

# Index must match VARIANTS order in run_assumption_sensitivity.py.
VARIANTS=(baseline pre_visit_dance no_capacity_floor tournament common_draws)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
PREFIX="results/assumption_sensitivity_${VARIANT}"

echo "array task ${SLURM_ARRAY_TASK_ID:-0} -> variant ${VARIANT}" >&2

"${PYTHON_BIN}" -u experiments/run_assumption_sensitivity.py \
    --variants "${VARIANT}" \
    --seeds "${BEES_AS_SEEDS:-400-449}" \
    --output-prefix "${PREFIX}" \
    --max-workers "${SLURM_CPUS_PER_TASK}"

if [[ "${BEES_PUSH:-0}" == "1" ]]; then
    # Serialize the racing array tasks; rebase picks up siblings' commits.
    flock results/.assumption_sensitivity.lock bash -c "
        git add '${PREFIX}_events.csv' '${PREFIX}_group_summary.csv' &&
        git commit -m 'Add assumption-sensitivity results (${VARIANT})' || true
        git pull --rebase --autostash || true
        git push || true
    "
fi
