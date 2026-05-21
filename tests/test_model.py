from __future__ import annotations

import unittest

from bees.model import Parameters, Population, simulate


class ModelTests(unittest.TestCase):
    def test_population_frequency(self) -> None:
        population = Population(signalers=25, nonsignalers=75)

        self.assertEqual(population.size, 100)
        self.assertEqual(population.signal_frequency, 0.25)

    def test_simulation_is_reproducible(self) -> None:
        population = Population(signalers=50, nonsignalers=50)
        parameters = Parameters(
            signaler_payoff=1.15,
            nonsignaler_payoff=1.0,
            communication_cost=0.05,
        )

        first = simulate(population, parameters, generations=10, seed=7)
        second = simulate(population, parameters, generations=10, seed=7)

        self.assertEqual(first, second)

    def test_history_includes_initial_population(self) -> None:
        population = Population(signalers=1, nonsignalers=1)
        parameters = Parameters(
            signaler_payoff=1.0,
            nonsignaler_payoff=1.0,
            communication_cost=0.0,
        )

        history = simulate(population, parameters, generations=3, seed=1)

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0], population)


if __name__ == "__main__":
    unittest.main()
