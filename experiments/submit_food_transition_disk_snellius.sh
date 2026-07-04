#!/usr/bin/env bash
# Submit the disk-ecology vertical-transition Optuna search on Snellius.
#
# Design: a wide array of nodes all optimize one shared Optuna study (single
# JournalStorage file), followed by a finalize job that exports the merged
# trial/seed CSVs. Scale the search two ways:
#   BEES_DISK_ARRAY_TASKS         number of nodes in the array (breadth)
#   BEES_DISK_OPTUNA_TRIALS_PER_TASK  trials each node contributes (depth)
# Total trials = tasks * per-task; each node runs SLURM_CPUS_PER_TASK workers.
#
# Other knobs (passed through to the sbatch scripts):
#   BEES_DISK_ARRAY_CONCURRENCY   cap on simultaneously running array tasks
#   BEES_CONFIG                   base config (default disk transition config)
#   BEES_OPTUNA_JOURNAL           shared study journal path
#   BEES_OPTUNA_TRIALS_CSV / BEES_OPTUNA_SEED_METRICS   finalize CSV outputs
#   BEES_DISK_OPTUNA_SEEDS        per-trial seed panel (default 100-109)
#   BEES_DISK_OPTUNA_STARTUP      random TPE startup trials per worker
#   BEES_DISK_OPTUNA_GENERATIONS  override config generations
#   BEES_FRESH_JOURNAL=1          delete an existing journal before submitting
#   BEES_PUSH=1                   finalizer commits + pushes result CSVs

set -euo pipefail

array_tasks="${BEES_DISK_ARRAY_TASKS:-16}"
concurrency="${BEES_DISK_ARRAY_CONCURRENCY:-}"
journal="${BEES_OPTUNA_JOURNAL:-results/food_transition_disk_optuna.journal}"

if (( array_tasks < 1 )); then
    echo "BEES_DISK_ARRAY_TASKS must be at least 1" >&2
    exit 2
fi

if [[ "${BEES_FRESH_JOURNAL:-0}" == "1" ]]; then
    echo "removing existing journal ${journal}"
    rm -f "${journal}" "${journal}.lock" "${journal}"*.lock 2>/dev/null || true
elif [[ -e "${journal}" ]]; then
    echo "WARNING: ${journal} already exists; new trials will be appended to the" >&2
    echo "         existing study. Set BEES_FRESH_JOURNAL=1 to start clean." >&2
fi

array_spec="0-$((array_tasks - 1))"
if [[ -n "${concurrency}" ]]; then
    array_spec="${array_spec}%${concurrency}"
fi

optuna_job="$(
    sbatch --parsable \
        --array="${array_spec}" \
        experiments/run_food_transition_disk_optuna_snellius.sbatch
)"
finalize_job="$(
    sbatch --parsable \
        --dependency=afterok:"${optuna_job}" \
        experiments/run_food_transition_disk_finalize_snellius.sbatch
)"

echo "submitted disk optuna array job ${optuna_job} (${array_spec})"
echo "submitted disk optuna finalize job ${finalize_job}"
