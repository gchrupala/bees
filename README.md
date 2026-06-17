# Bees

Computational models for exploring the evolution of honeybee communication.

This repository is intentionally starting small. We will add models only once
the biological question and modeling assumptions are clear.

## Repository Structure

- `src/bees/`: Python package for model code
- `configs/`: reproducible experiment settings
- `experiments/`: runnable experiment scripts
- `report/`: lightweight HTML scientific report and legacy LaTeX snapshot
- `tests/`: tests for model behavior
- `results/`: local experiment outputs

## Current Model

The direction model represents food sites explicitly with direction, angular
width, distance, value, and capacity. In each episode, workers act
sequentially. A worker follows an existing dance with probability set by its
receiver-attention trait; if no dance is available, or if it does not attend, it
searches independently in a random direction. Independent searchers that find
food add a dance to the episode. Workers search along their chosen direction up
to a worker-specific limit; this limit varies within colonies and its colony
mean is heritable.

Comb tilt and orientation are colony-level heritable traits. World food
directions are projected onto the comb plane for direct pointing, while
sun-relative directions can be encoded against the gravity projection on the
comb. The sun azimuth is sampled once per episode from a configurable daytime
arc. Direct pointing is strongest when the food direction projects well into
the comb plane; gravity-referenced mapping is unavailable on a horizontal comb
and becomes stronger as the comb becomes vertical. Sender and receiver
transposition mutations can be correlated with
`transposition_mutation_correlation`, making coupling strength an explicit
experimental parameter. Comb tilt can also use its own mutation scale via
`comb_tilt_mutation_sd`, which allows constrained probes where tilt is supplied
rather than evolved. Setting `comb_orientation_axial` treats orientations
separated by 180 degrees as the same comb plane. Dance production can include
both a baseline per-dance cost and a precision-dependent cost tied to the
dancer's directional-bias trait. The optional non-communication advantage of
vertical combs is proportional to colony performance: payoff is multiplied by
`1 + vertical_comb_benefit * f(t)`, with `vertical_comb_modifier` selecting
`linear` (`f(t)=t`) or `threshold_0.8` (`f(t)=1` only when tilt is at least
0.8). This keeps verticality from rescuing colonies whose foraging payoff has
collapsed.

## Development

Install the project and experiment dependencies:

```sh
python -m pip install -e .
```

Run the horizontal direction experiment:

```sh
python experiments/run_horizontal_direction.py
```

Generate a simple SVG demo plot:

```sh
python experiments/demo_horizontal_direction.py
```

Stream a one-parameter sweep:

```sh
python experiments/sweep_horizontal_direction.py --seeds 101,102,103
```

Run the report conditions:

```sh
python experiments/run_report_conditions.py
```

Run a comb-tilt geometry sanity grid:

```sh
python experiments/run_tilt_geometry_sanity.py --seeds 101,102,103,104,105
```

Run a comb-orientation sanity grid:

```sh
python experiments/run_orientation_sanity.py
```

Run the constrained near-vertical coupling probe:

```sh
python -u experiments/run_vertical_coupling_probe.py
```

Run the longer axial-orientation vertical transition experiment:

```sh
python -u experiments/run_long_vertical_transition.py
```

Run an Optuna search over food-transition parameters:

```sh
python -u experiments/optimize_food_transition.py --workers 4 --n-trials 32
```

Run the extensive overnight Optuna search over ecology, vertical advantage,
mutation scale, tilt-mutation scale, and sender-receiver mutation coupling:

```sh
python -u experiments/optimize_food_transition.py \
  --journal-output results/food_transition_optuna_extensive.journal \
  --trials-output results/food_transition_optuna_extensive_trials.csv \
  --seed-output results/food_transition_optuna_extensive_seeds.csv \
  --study-name food_transition_optuna_extensive \
  --n-trials 144 \
  --workers 4 \
  --seeds 100,101,102,103,104 \
  --generations 120 \
  --startup-trials 32 \
  --food-site-count-min 3 \
  --food-site-count-max 8 \
  --food-site-width-min 0.20 \
  --food-site-width-max 0.40 \
  --food-site-capacity-min 6 \
  --food-site-capacity-max 16 \
  --food-value-min 0.8 \
  --food-value-max 1.3 \
  --food-site-max-distance-min 5.0 \
  --food-site-max-distance-max 9.0 \
  --travel-cost-min 0.01 \
  --travel-cost-max 0.05 \
  --vertical-comb-benefit-min 0.15 \
  --vertical-comb-benefit-max 0.45 \
  --mutation-sd-min 0.04 \
  --mutation-sd-max 0.11 \
  --comb-tilt-mutation-sd-min 0.04 \
  --comb-tilt-mutation-sd-max 0.11 \
  --transposition-mutation-correlation-min 0.3 \
  --transposition-mutation-correlation-max 0.9
```

Compare successful and unsuccessful trajectories in the best Optuna pocket:

```sh
python -u experiments/analyze_optuna_best_trajectories.py --seeds 96-120 --max-workers 4
```

Validate the best extensive Optuna candidates on held-out seeds:

```sh
python -u experiments/validate_extensive_optuna_candidates.py \
  --seeds 96-195 \
  --exclude-seeds 100-104 \
  --max-workers 4
```

Run and finalize the refined one-parameter sensitivity panel on Snellius:

```sh
sbatch -A ubsr112721 experiments/run_sensitivity_refinement_snellius.sbatch
```

The batch wrapper requests 16 CPUs by default and uses `SLURM_CPUS_PER_TASK`
worker processes. It streams raw event and trajectory rows under
`results/food_transition_sensitivity_refinement_*.csv`, updates
`report/report.md`, renders `report/report.html` if Pandoc is available, and
commits the generated artifacts when it finishes. Set `BEES_MODULE_LOAD`,
`BEES_VENV`, or `BEES_PYTHON` at submission time to override the default
`2025 Python/3.13.5-GCCcore-14.3.0` module stack, and set `BEES_PUSH=1` to push
the result commit after finalization.

Export no-threshold long-transition trajectories and event timings:

```sh
python -u experiments/analyze_long_transition_trajectories.py
```

Render the working report:

```sh
python -u experiments/render_report_html.py
```

Regenerate the tracked report result files and legacy LaTeX report artifacts:

```sh
python -u experiments/run_report_artifacts.py all
```

List the command associated with each legacy generated report artifact:

```sh
python -u experiments/run_report_artifacts.py list
```

Run tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```
