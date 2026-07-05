#!/usr/bin/env bash
#SBATCH --job-name=bees-disk-collect
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Commit and push all disk-transition result CSVs once, after every stage of
# both decode variants has finished. Runs only if BEES_PUSH=1.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ "${BEES_PUSH:-0}" != "1" ]]; then
    echo "BEES_PUSH != 1; leaving results uncommitted"
    exit 0
fi

git add results/food_transition_disk_*.csv results/food_transition_disk_unproject_*.csv 2>/dev/null || true
if git diff --cached --quiet; then
    echo "no result changes to commit"
    exit 0
fi
git commit -m "Add disk transition downstream results (flatten + unproject)" || true
git pull --rebase || true
git push || true
