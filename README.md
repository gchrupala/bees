# Bees

Computational models for exploring the evolution of honeybee communication.

Communication typically relies on a shared code, and any change to it must be coordinated
between senders and receivers. The honeybee waggle dance illustrates this problem: species
with horizontal combs point directly at a food source, while species with vertical combs
cannot point directly and instead reference the dance to gravity, decoded against the
position of the sun. This project models the rise of direct pointing and its
evolutionary transition to a gravity-referenced code in populations of bee-like agents,
with selection acting at the level of colonies. The full argument, setup, and results are
written up in `report/paper.tex`; this README covers the codebase and how to reproduce
every figure and table in that paper.

## Repository Structure

- `src/bees/`: the model itself (colonies, workers, foraging, mutation, evolution)
- `configs/`: reproducible experiment settings (JSON, one per experiment)
- `experiments/`: runnable experiment, analysis, plotting, and Snellius scripts
- `results/`: tracked result CSVs (large trajectory files are gzipped)
- `report/`: `paper.tex`, its bibliography and figures, and a slide deck
- `tests/`: focused tests for model behavior and experiment helpers

## Environment & Dependencies

Requires Python >=3.11. Dependencies are declared in `pyproject.toml`; add or change them
there rather than `pip install`-ing extras directly.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The `src/bees` layout needs this editable install for `import bees` and the experiment
scripts to resolve; re-run it after changing dependencies.

Run the tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

## Reproducing the Paper

Every figure and table in `report/paper.tex` traces back to a config in `configs/`, a
tracked CSV in `results/`, and a script in `experiments/`. Two ways to reproduce them:

- **From tracked results** (seconds, exact): re-run the plotting/table scripts below
  against the CSVs already committed in `results/`.
- **From scratch** (hours to days, matches procedure not necessarily bit-for-bit): re-run
  the simulation scripts that produced those CSVs. The confirmation, validation,
  sensitivity, and interaction runs are deterministic given a seed range; only the Optuna
  search itself is order-sensitive under parallel workers, so a from-scratch search will
  differ from the committed trials in its particulars while following the same
  procedure.

All commands assume the repo root as the working directory and the editable install
above. `report/figures/apis_phylogeny.pdf` (`fig:phylo`) is a hand-authored TikZ figure,
not a Python script; it is a standalone document, not `\input` by `paper.tex`, so it is
not rebuilt by the main LaTeX pass. Regenerate it separately with
`cd report/figures && pdflatex apis_phylogeny.tex` (no shell-escape needed).

### Quick build: every figure, then the PDF

Regenerates all six raster figures from the tracked results (fast, no simulation),
then builds the paper (`fig:phylo` is compiled as part of this LaTeX step, see above):

```sh
python -u experiments/illustrate_waggle_dance.py && \
python -u experiments/illustrate_direct_decode.py && \
python -u experiments/visualize_disk_food_samples.py && \
python -u experiments/plot_food_distribution_disk_grid.py && \
python -u experiments/plot_food_transition_disk_sensitivity_ridge.py && \
python -u experiments/plot_food_transition_disk_interaction_heatmap.py && \
(cd report && latexmk -pdf paper.tex)
```

The sections below cover each figure/table individually, including regenerating the
underlying result CSVs from scratch.

### Building the paper itself

```sh
cd report && latexmk -pdf paper.tex
```

### Schematic figures (no simulation data)

```sh
python -u experiments/illustrate_waggle_dance.py   # fig:waggle
python -u experiments/illustrate_direct_decode.py  # fig:direct-decode
```

### Sampled food geometry (fig:environment)

```sh
python -u experiments/visualize_disk_food_samples.py
```

### Horizontal-comb food-distribution grid (fig:food-grid)

```sh
# regenerate results/food_distribution_disk_{events,points,group_summary}.csv
python -u experiments/run_food_distribution_disk.py \
  --config configs/food_distribution_disk.json \
  --seeds 400-449 \
  --output-prefix results/food_distribution_disk \
  --max-workers 16

# regenerate the figure from the group summary
python -u experiments/plot_food_distribution_disk_grid.py
```

### Transition-stage pipeline (both decodes)

The paper runs this same pipeline once per decode, over
`configs/long_vertical_transition_disk.json` (flatten) and
`configs/long_vertical_transition_disk_unproject.json` (unproject). Substitute the
config, `--output-prefix`, `--output`, and result filenames accordingly; the flatten
paths are shown below.

**1. Optuna search** (1024 trials over 10 seeds each; backs `tab:optuna-stability` and
`tab:optuna-region`):

```sh
python -u experiments/optimize_food_transition_disk.py \
  --config configs/long_vertical_transition_disk.json \
  --workers 16 \
  --n-trials 1024 \
  --startup-trials 64 \
  --seeds 100-109 \
  --export \
  --journal-output results/food_transition_disk_optuna.journal \
  --trials-output results/food_transition_disk_optuna_trials.csv \
  --seed-output results/food_transition_disk_optuna_seed_metrics.csv
```

`tab:optuna-stability` and `tab:optuna-region` are computed directly from
`*_optuna_seed_metrics.csv` / `*_optuna_trials.csv` with pandas, not by a dedicated
script, e.g.:

```python
import pandas as pd
trials = pd.read_csv("results/food_transition_disk_optuna_trials.csv")
trials["stable_count"].value_counts().sort_index()
```

**2. Confirmation** (top 20 distinct Optuna candidates, 40 held-out seeds):

```sh
python -u experiments/run_food_transition_disk_panel.py \
  --source trials \
  --config configs/long_vertical_transition_disk.json \
  --trials results/food_transition_disk_optuna_trials.csv \
  --max-candidates 20 \
  --seeds 110-149 \
  --output-prefix results/food_transition_disk_confirmation \
  --max-workers 16
```

**3. Validation** (top 5 confirmed candidates, 100 held-out seeds; backs
`tab:validation`):

```sh
python -u experiments/run_food_transition_disk_panel.py \
  --source panel \
  --config configs/long_vertical_transition_disk.json \
  --points results/food_transition_disk_confirmation_points.csv \
  --group-summary results/food_transition_disk_confirmation_group_summary.csv \
  --max-candidates 5 \
  --seeds 200-299 \
  --output-prefix results/food_transition_disk_validation \
  --max-workers 16
```

**4. One-parameter sensitivity** (around the strongest validated candidate; backs
`fig:sensitivity-flatten` / `fig:sensitivity-unproject`):

```sh
python -u experiments/run_food_transition_disk_sensitivity.py \
  --config configs/long_vertical_transition_disk.json \
  --baseline-points results/food_transition_disk_validation_points.csv \
  --baseline-summary results/food_transition_disk_validation_group_summary.csv \
  --seeds 200-299 \
  --output-prefix results/food_transition_disk_sensitivity \
  --max-workers 16

# regenerate both decode figures at once, from tracked results
python -u experiments/plot_food_transition_disk_sensitivity_ridge.py
```

**5. Evolutionary-parameter interaction grid** (8x8 over $B$, $\sigma_m$, $\rho$; backs
`fig:interaction`):

```sh
python -u experiments/run_food_transition_disk_interaction.py \
  --config configs/long_vertical_transition_disk.json \
  --baseline-points results/food_transition_disk_validation_points.csv \
  --baseline-summary results/food_transition_disk_validation_group_summary.csv \
  --seeds 200-299 \
  --output results/food_transition_disk_interaction.csv \
  --max-workers 16

# regenerate the figure from both decodes' tracked interaction CSVs
python -u experiments/plot_food_transition_disk_interaction_heatmap.py
```

Static tables (`tab:routes`, `tab:traits`, `tab:fixed-params`, `tab:varied-params`,
`tab:search-space`) describe the model and search space directly and are not generated
from result files.

## Snellius

`<user>` and `<path-to-checkout>` below are placeholders; see `AGENTS.local.md`
(gitignored, not in this public repo) for the real values. Sync your checkout before
launching jobs:

```sh
ssh <user>@snellius.surf.nl
cd <path-to-checkout>
git pull --rebase
```

Submit the full disk-ecology pipeline (both decodes) from the remote checkout:

```sh
BEES_PUSH=1 ./experiments/submit_food_transition_disk_pipeline_snellius.sh
```

This runs each decode's Optuna search as a Slurm array, then confirmation, validation,
sensitivity, and the interaction grid for both decodes, committing and pushing the merged
result CSVs at the end. Submit the horizontal-stage food-distribution grid separately:

```sh
sbatch experiments/run_food_distribution_disk_snellius.sbatch
```

Monitor with `squeue -u <user>`, and inspect `logs/slurm-*.out` / `logs/slurm-*.err`.
Both checkouts must stay in sync via git: `git pull --rebase` before new work, and commit
result CSVs (never `scp`/`rsync` them) after a run. See `AGENTS.md` for the full Snellius
and environment-variable conventions (`BEES_VENV`, `BEES_PYTHON`, `BEES_MODULE_LOAD`, and
why job-side variables need `--export=ALL,VAR=value`).
