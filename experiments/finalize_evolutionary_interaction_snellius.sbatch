#!/usr/bin/env bash
#SBATCH --job-name=bees-evo-final
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
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

PYTHON_BIN="${BEES_PYTHON:-python}"
TASK_COUNT="${BEES_ARRAY_TASKS:-64}"

push_arg=()
if [[ "${BEES_PUSH:-0}" == "1" ]]; then
    push_arg=(--push)
fi

"${PYTHON_BIN}" -u experiments/finalize_evolutionary_interaction.py \
    --task-count "${TASK_COUNT}" \
    --baseline-points "${BEES_VALIDATION_POINTS:-results/food_transition_v2_validation_points.csv}" \
    --baseline-group-summary "${BEES_VALIDATION_GROUP_SUMMARY:-results/food_transition_v2_validation_group_summary.csv}" \
    --wait-for-shards \
    "${push_arg[@]}"
