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

Run tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

