# Bees

Computational models for studying the evolution of bee communication.

The project is currently a lightweight scaffold. Specific models, experiments,
and tests should be added only after the modeling question is explicit.

## Structure

- `src/bees/`: simulation code
- `configs/`: reproducible experiment settings
- `experiments/`: scripts that run simulations
- `tests/`: model tests
- `results/`: local experiment outputs

## Development

Run tests once tests exist:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```
