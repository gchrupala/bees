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

array_job="$(
    sbatch --parsable \
        --array="${array_spec}" \
        experiments/run_evolutionary_interaction_array_snellius.sbatch
)"
finalize_job="$(
    sbatch --parsable \
        --dependency=afterok:"${array_job}" \
        --export=ALL,BEES_ARRAY_TASKS="${array_tasks}" \
        experiments/finalize_evolutionary_interaction_snellius.sbatch
)"

echo "submitted evolutionary interaction array job ${array_job}"
echo "submitted evolutionary interaction finalizer job ${finalize_job}"
