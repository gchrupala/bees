from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Population:
    signalers: int
    nonsignalers: int

    @property
    def size(self) -> int:
        return self.signalers + self.nonsignalers

    @property
    def signal_frequency(self) -> float:
        if self.size == 0:
            return 0.0
        return self.signalers / self.size


@dataclass(frozen=True)
class Parameters:
    signaler_payoff: float
    nonsignaler_payoff: float
    communication_cost: float

    @property
    def net_signaler_payoff(self) -> float:
        return max(0.0, self.signaler_payoff - self.communication_cost)


def step(population: Population, parameters: Parameters, rng: Random) -> Population:
    """Create the next generation using payoff-weighted sampling."""
    if population.size == 0:
        return population

    signaler_weight = population.signalers * parameters.net_signaler_payoff
    nonsignaler_weight = population.nonsignalers * parameters.nonsignaler_payoff
    total_weight = signaler_weight + nonsignaler_weight

    if total_weight <= 0:
        return population

    signaler_probability = signaler_weight / total_weight
    next_signalers = sum(
        1 for _ in range(population.size) if rng.random() < signaler_probability
    )

    return Population(
        signalers=next_signalers,
        nonsignalers=population.size - next_signalers,
    )


def simulate(
    population: Population,
    parameters: Parameters,
    generations: int,
    seed: int,
) -> list[Population]:
    rng = Random(seed)
    history = [population]

    for _ in range(generations):
        population = step(population, parameters, rng)
        history.append(population)

    return history
