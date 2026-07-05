#!/usr/bin/env bash
# Submit the full disk-ecology transition pipeline for BOTH decode variants.
#
#   flatten   : confirmation -> validation -> {sensitivity, interaction}
#               (Optuna already run; reuses results/food_transition_disk_optuna_trials.csv)
#   unproject : optuna array -> finalize -> confirmation -> validation ->
#               {sensitivity, interaction}
#   collect   : after all leaves, commit + push every disk result CSV (BEES_PUSH=1)
#
# Env knobs: BEES_DISK_ARRAY_TASKS (unproject optuna breadth, default 16),
# BEES_DISK_ARRAY_CONCURRENCY, BEES_PUSH (default 1), BEES_VENV/BEES_PYTHON.
set -euo pipefail

array_tasks="${BEES_DISK_ARRAY_TASKS:-16}"
array_spec="0-$((array_tasks - 1))"
if [[ -n "${BEES_DISK_ARRAY_CONCURRENCY:-}" ]]; then
    array_spec="${array_spec}%${BEES_DISK_ARRAY_CONCURRENCY}"
fi
push="${BEES_PUSH:-1}"

D="experiments"
CONFIRM="${D}/run_food_transition_disk_confirmation_snellius.sbatch"
VALID="${D}/run_food_transition_disk_validation_snellius.sbatch"
SENS="${D}/run_food_transition_disk_sensitivity_snellius.sbatch"
INTER="${D}/run_food_transition_disk_interaction_snellius.sbatch"

# ── flatten (Optuna already complete) ─────────────────────────────────────────
FLAT="ALL"
FLAT+=",BEES_CONFIG=configs/long_vertical_transition_disk.json"
FLAT+=",BEES_OPTUNA_TRIALS_CSV=results/food_transition_disk_optuna_trials.csv"
FLAT+=",BEES_CONFIRMATION_PREFIX=results/food_transition_disk_confirmation"
FLAT+=",BEES_VALIDATION_PREFIX=results/food_transition_disk_validation"
FLAT+=",BEES_SENSITIVITY_PREFIX=results/food_transition_disk_sensitivity"
FLAT+=",BEES_INTERACTION_OUTPUT=results/food_transition_disk_interaction.csv"

conf_f="$(sbatch --parsable --export="${FLAT}" "${CONFIRM}")"
val_f="$(sbatch --parsable --dependency=afterok:"${conf_f}" --export="${FLAT}" "${VALID}")"
sens_f="$(sbatch --parsable --dependency=afterok:"${val_f}" --export="${FLAT}" "${SENS}")"
inter_f="$(sbatch --parsable --dependency=afterok:"${val_f}" --export="${FLAT}" "${INTER}")"

# ── unproject (Optuna + downstream) ───────────────────────────────────────────
UNP="ALL"
UNP+=",BEES_CONFIG=configs/long_vertical_transition_disk_unproject.json"
UNP+=",BEES_OPTUNA_JOURNAL=results/food_transition_disk_unproject_optuna.journal"
UNP+=",BEES_OPTUNA_TRIALS_CSV=results/food_transition_disk_unproject_optuna_trials.csv"
UNP+=",BEES_OPTUNA_SEED_METRICS=results/food_transition_disk_unproject_optuna_seed_metrics.csv"
UNP+=",BEES_CONFIRMATION_PREFIX=results/food_transition_disk_unproject_confirmation"
UNP+=",BEES_VALIDATION_PREFIX=results/food_transition_disk_unproject_validation"
UNP+=",BEES_SENSITIVITY_PREFIX=results/food_transition_disk_unproject_sensitivity"
UNP+=",BEES_INTERACTION_OUTPUT=results/food_transition_disk_unproject_interaction.csv"

rm -f results/food_transition_disk_unproject_optuna.journal \
      results/food_transition_disk_unproject_optuna.journal*.lock 2>/dev/null || true

opt_u="$(sbatch --parsable --array="${array_spec}" --export="${UNP}" \
    "${D}/run_food_transition_disk_optuna_snellius.sbatch")"
fin_u="$(sbatch --parsable --dependency=afterok:"${opt_u}" --export="${UNP}" \
    "${D}/run_food_transition_disk_finalize_snellius.sbatch")"
conf_u="$(sbatch --parsable --dependency=afterok:"${fin_u}" --export="${UNP}" "${CONFIRM}")"
val_u="$(sbatch --parsable --dependency=afterok:"${conf_u}" --export="${UNP}" "${VALID}")"
sens_u="$(sbatch --parsable --dependency=afterok:"${val_u}" --export="${UNP}" "${SENS}")"
inter_u="$(sbatch --parsable --dependency=afterok:"${val_u}" --export="${UNP}" "${INTER}")"

# ── collect + push once, after every leaf ─────────────────────────────────────
collect="$(sbatch --parsable \
    --dependency=afterok:"${sens_f}":"${inter_f}":"${sens_u}":"${inter_u}" \
    --export="ALL,BEES_PUSH=${push}" \
    "${D}/run_food_transition_disk_collect_snellius.sbatch")"

echo "flatten:   confirm ${conf_f}  valid ${val_f}  sens ${sens_f}  inter ${inter_f}"
echo "unproject: optuna ${opt_u}  finalize ${fin_u}  confirm ${conf_u}  valid ${val_u}  sens ${sens_u}  inter ${inter_u}"
echo "collect:   ${collect}"
