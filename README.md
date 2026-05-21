# Bees

Computational models for exploring the evolution of honeybee communication.

This repository is intentionally starting small. We will add models only once
the biological question and modeling assumptions are clear.

## Repository Structure

- `src/bees/`: Python package for model code
- `configs/`: reproducible experiment settings
- `experiments/`: runnable experiment scripts
- `tests/`: tests for model behavior
- `results/`: local experiment outputs

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
