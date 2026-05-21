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

The horizontal-comb model represents food sites explicitly. In each episode,
a scout discovers one site and dances for it, while recruits can either follow
the dance or search independently and find any available food site in the world.

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

Run tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```
