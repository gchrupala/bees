# Bees

Computational models for exploring the evolution of honeybee communication.

This repository is intentionally starting small. We will add models only once
the biological question and modeling assumptions are clear.

## Repository Structure

- `src/bees/`: Python package for model code
- `configs/`: reproducible experiment settings
- `experiments/`: runnable experiment scripts
- `report/`: LaTeX scientific report draft
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

Export no-threshold long-transition trajectories and event timings:

```sh
python -u experiments/analyze_long_transition_trajectories.py
```

Regenerate the tracked report result files and generated LaTeX report artifacts:

```sh
python -u experiments/run_report_artifacts.py all
```

List the command associated with each generated report artifact:

```sh
python -u experiments/run_report_artifacts.py list
```

Run tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```
