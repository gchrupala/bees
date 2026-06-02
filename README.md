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

The horizontal-comb model represents food sites explicitly with direction,
angular width, distance, value, and capacity. In each episode, workers act
sequentially. A worker follows an existing dance with probability set by its
receiver-attention trait; if no dance is available, or if it does not attend,
it searches independently in a random direction. Independent searchers that find
food add a dance to the episode. Workers search along their chosen direction up
to a worker-specific limit; this limit varies within colonies and its colony
mean is heritable. The model also includes an optional comb-tilt parameter and
continuous sender and receiver transposition traits for exploring the transition
from direct horizontal pointing to gravity-referenced mapping.

## Development

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

Run tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```
