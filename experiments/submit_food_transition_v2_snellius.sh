#!/usr/bin/env bash
set -euo pipefail

array_tasks="${BEES_ARRAY_TASKS:-64}"
concurrency="${BEES_ARRAY_CONCURRENCY:-}"

if (( array_tasks < 1 )); then
    echo "BEES_ARRAY_TASKS must be at least 1" >&2
    exit 2
fi

array_spec="0-$((array_tasks - 1))"
if [[ -n "${concurrency}" ]]; then
    array_spec="${array_spec}%${concurrency}"
fi

optuna_job="$(
    sbatch --parsable experiments/run_food_transition_v2_optuna_snellius.sbatch
)"
confirmation_job="$(
    sbatch --parsable \
        --dependency=afterok:"${optuna_job}" \
        experiments/run_food_transition_v2_confirmation_snellius.sbatch
)"
validation_job="$(
    sbatch --parsable \
        --dependency=afterok:"${confirmation_job}" \
        experiments/run_food_transition_v2_validation_snellius.sbatch
)"
coarse_job="$(
    sbatch --parsable \
        --dependency=afterok:"${validation_job}" \
        --export=ALL,BEES_V2_SENSITIVITY_PANEL=coarse \
        experiments/run_food_transition_v2_sensitivity_snellius.sbatch
)"
refinement_job="$(
    sbatch --parsable \
        --dependency=afterok:"${validation_job}" \
        --export=ALL,BEES_V2_SENSITIVITY_PANEL=refinement \
        experiments/run_food_transition_v2_sensitivity_snellius.sbatch
)"
interaction_job="$(
    sbatch --parsable \
        --dependency=afterok:"${validation_job}" \
        --array="${array_spec}" \
        experiments/run_evolutionary_interaction_array_snellius.sbatch
)"
interaction_finalize_job="$(
    sbatch --parsable \
        --dependency=afterok:"${interaction_job}" \
        --export=ALL,BEES_ARRAY_TASKS="${array_tasks}" \
        experiments/finalize_evolutionary_interaction_snellius.sbatch
)"

echo "submitted v2 optuna job ${optuna_job}"
echo "submitted v2 confirmation job ${confirmation_job}"
echo "submitted v2 validation job ${validation_job}"
echo "submitted v2 coarse sensitivity job ${coarse_job}"
echo "submitted v2 refinement sensitivity job ${refinement_job}"
echo "submitted v2 evolutionary interaction array job ${interaction_job}"
echo "submitted v2 evolutionary interaction finalizer job ${interaction_finalize_job}"
