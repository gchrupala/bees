# Bees

Computational models for exploring the evolution of honeybee communication.

The current work asks when populations that start with horizontal direct-pointing
communication can evolve both a vertical comb and a gravity-referenced sender-receiver
code. The codebase is intentionally small: model rules live in `src/bees/`, experiment
scripts live in `experiments/`, and scientific claims should stay grounded in tracked
configuration, seeds, and result CSVs.

## Repository Structure

- `src/bees/`: Python package for model code
- `configs/`: reproducible experiment settings
- `experiments/`: runnable experiment, analysis, plotting, and Snellius scripts
- `results/`: tracked result CSVs and local/generated experiment outputs
- `report/`: lightweight Markdown/HTML report and figures
- `tests/`: focused tests for model behavior and experiment helpers

## Current V2 Model

Colonies are the reproducing entities. Workers are behavioral samples from heritable
colony means. A colony has heritable means for directional bias, receiver attention,
sender transposition, receiver transposition, search limit, comb tilt, and comb
orientation.

Each foraging episode samples food sites with direction, angular width, distance, value,
and capacity. Workers act sequentially. A worker may follow an available dance according
to its receiver-attention trait; otherwise it searches in a random direction. Every
successful worker produces a dance for the food site and pays the dance cost, including
workers that were themselves recruited by a dance.

Comb geometry supplies two possible directional mappings. Direct pointing projects the
horizontal food direction into the comb plane. Gravity-referenced signaling uses the
projection of gravity into the comb plane together with the episode sun azimuth. The
gravity cue is unavailable on a horizontal comb and strengthens as the comb becomes
vertical. The v2 transition runs use axial comb orientation, so orientations separated
by 180 degrees are treated as the same comb plane.

The transition experiments start from horizontal combs and use a linear multiplicative
vertical-comb benefit:

```text
episode payoff = foraging payoff * (1 + vertical_comb_benefit * comb_tilt)
```

This benefit scales viable foraging performance; it does not rescue a colony whose
foraging payoff has collapsed. All heritable traits, including comb tilt and comb
orientation, mutate with the shared `mutation_sd`. Sender and receiver transposition
mutations can be coupled with `transposition_mutation_correlation`.

A seed is counted as a stable vertical gravity-code outcome when final mean comb tilt is
at least `0.80` and both final mean sender and receiver transposition are at least
`0.50`. A seed is counted as collapsed if mean foraging success reaches `0.02` or below.

## Current Report

The working report is v2-only:

- source: `report/report.md`
- rendered HTML: `report/report.html`
- figures: `report/figures/`
- primary result inputs: `results/food_transition_v2_*`

Render the report:

```sh
python -u experiments/render_report_html.py
```

Regenerate the figures used by the report:

```sh
python -u experiments/plot_oat_sensitivity_effects.py \
  --points results/food_transition_v2_oat_sensitivity_points.csv \
  --events results/food_transition_v2_oat_sensitivity_events.csv \
  --output report/figures/oat_sensitivity_stable_delta

python -u experiments/plot_evolutionary_interaction_heatmap.py \
  --group-summary results/food_transition_v2_evolutionary_interaction_group_summary.csv \
  --output report/figures/evolutionary_interaction_stable_heatmap

python -u experiments/plot_evolutionary_interaction_seed_outcomes.py \
  --events results/food_transition_v2_evolutionary_interaction_events.csv \
  --output report/figures/evolutionary_interaction_seed_outcomes_binary
```

`report/paper.tex` is only a LaTeX scaffold. The primary maintained report artifact is
the Markdown/HTML workflow above.

## Development

Install dependencies:

```sh
python -m pip install -e .
```

Run tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

Run a small local v2 Optuna search:

```sh
python -u experiments/optimize_food_transition.py \
  --workers 4 \
  --n-trials 32 \
  --seeds 100-102
```

Run a small candidate panel from Optuna trials:

```sh
python -u experiments/run_food_transition_v2_candidate_panel.py \
  --source trials \
  --trials results/food_transition_v2_optuna_trials.csv \
  --max-candidates 5 \
  --seeds 110-119 \
  --max-workers 4 \
  --output-prefix results/food_transition_v2_confirmation
```

Run a small v2 sensitivity panel around the current validated baseline:

```sh
python -u experiments/run_food_transition_oat_sensitivity.py \
  --panel coarse \
  --baseline-points results/food_transition_v2_validation_points.csv \
  --baseline-group-summary results/food_transition_v2_validation_group_summary.csv \
  --seeds 300-309 \
  --max-workers 4
```

Run one local shard of the evolutionary interaction grid:

```sh
python -u experiments/run_evolutionary_interaction_array.py \
  --task-id 0 \
  --task-count 4 \
  --baseline-points results/food_transition_v2_validation_points.csv \
  --baseline-group-summary results/food_transition_v2_validation_group_summary.csv \
  --max-workers 4 \
  --max-seeds 10
```

## Snellius

The Snellius checkout is `/gpfs/home2/gchrupala1/bees`. Sync it before launching jobs:

```sh
ssh gchrupala1@snellius.surf.nl
cd /gpfs/home2/gchrupala1/bees
git pull --rebase
```

Submit the full v2 pipeline from the Snellius checkout:

```sh
BEES_ARRAY_TASKS=64 BEES_ARRAY_CONCURRENCY=4 \
./experiments/submit_food_transition_v2_snellius.sh
```

The helper submits, in order, the v2 Optuna search, confirmation panel, validation
panel, coarse and refined sensitivity panels, evolutionary interaction array, and
interaction finalizer. Set `BEES_PUSH=1` when the finalizer should commit and push the
merged evolutionary-interaction result CSVs after the array succeeds. The submit helper
and Slurm scripts also accept `BEES_VENV`, `BEES_PYTHON`, and `BEES_MODULE_LOAD` for
environment control.

Monitor jobs and logs with:

```sh
squeue -u gchrupala1
ls -lt slurm-*.out slurm-*.err
```
