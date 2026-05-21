# Bees

Computational models for exploring the evolution of honeybee communication.

This repository is intentionally starting small. We will add models only once
the biological question and modeling assumptions are clear.

## Goals

- Study how communication behaviors can evolve from simpler cues.
- Keep assumptions explicit and easy to revise.
- Build simple, inspectable simulations before adding complexity.
- Make experiments reproducible from configs and random seeds.

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

## Notes

Specific models are not yet implemented. Early work will focus on clarifying
which evolutionary pathway we want to simulate.
