# Bees

Computational models for studying the evolution of bee communication.

The project starts intentionally small: a simple simulation core, one basic
experiment script, and tests for the model rules.

## Structure

- `src/bees/`: simulation code
- `configs/`: reproducible experiment settings
- `experiments/`: scripts that run simulations
- `tests/`: model tests
- `results/`: local experiment outputs

## Quick Start

Run the basic experiment:

```sh
python experiments/run_basic.py
```

Run tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```
