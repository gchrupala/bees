#!/usr/bin/env bash
# Submit the v3 (direct_decode=unproject) food-transition pipeline on Snellius.
#
# Pipeline:
#   optuna → confirmation → validation → coarse-sensitivity  (parallel)
#                                      → interaction array → interaction finalize
#
# Skipped vs v2: sensitivity refinement, generation-budget experiments.
#
# Tunable env vars (all have sensible defaults):
#   BEES_ARRAY_TASKS          interaction array task count (default 64)
#   BEES_ARRAY_CONCURRENCY    max concurrent array tasks (default unlimited)
#   BEES_PUSH                 set to 1 to push merged CSVs after finalize
#   BEES_VENV / BEES_PYTHON   virtualenv / python binary on the cluster
set -euo pipefail

array_tasks="${BEES_ARRAY_TASKS:-64}"
concurrency="${BEES_ARRAY_CONCURRENCY:-}"

array_spec="0-$((array_tasks - 1))"
if [[ -n "${concurrency}" ]]; then
    array_spec="${array_spec}%${concurrency}"
fi

# ── v3 paths ──────────────────────────────────────────────────────────────────
export BEES_CONFIG="configs/long_vertical_transition_unproject.json"
export BEES_OPTUNA_JOURNAL="results/food_transition_v3_optuna.journal"
export BEES_OPTUNA_TRIALS_CSV="results/food_transition_v3_optuna_trials.csv"
export BEES_OPTUNA_SEED_METRICS="results/food_transition_v3_optuna_seed_metrics.csv"
export BEES_CONFIRMATION_PREFIX="results/food_transition_v3_confirmation"
export BEES_CONFIRMATION_POINTS="results/food_transition_v3_confirmation_points.csv"
export BEES_CONFIRMATION_GROUP_SUMMARY="results/food_transition_v3_confirmation_group_summary.csv"
export BEES_VALIDATION_PREFIX="results/food_transition_v3_validation"
export BEES_VALIDATION_POINTS="results/food_transition_v3_validation_points.csv"
export BEES_VALIDATION_GROUP_SUMMARY="results/food_transition_v3_validation_group_summary.csv"

# ── chain ─────────────────────────────────────────────────────────────────────
optuna_job="$(
    sbatch --parsable \
        experiments/run_food_transition_v2_optuna_snellius.sbatch
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
sensitivity_job="$(
    sbatch --parsable \
        --dependency=afterok:"${validation_job}" \
        --export=ALL,BEES_V2_SENSITIVITY_PANEL=coarse \
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

echo "submitted v3 optuna job              ${optuna_job}"
echo "submitted v3 confirmation job        ${confirmation_job}"
echo "submitted v3 validation job          ${validation_job}"
echo "submitted v3 coarse sensitivity job  ${sensitivity_job}"
echo "submitted v3 interaction array job   ${interaction_job}"
echo "submitted v3 interaction finalize    ${interaction_finalize_job}"
