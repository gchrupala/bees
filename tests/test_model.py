from __future__ import annotations

import unittest
from math import tau
from random import Random

from bees.model import (
    ColonyTraits,
    DirectionSettings,
    angular_distance,
    create_colony,
    evaluate_colony,
    simulate,
)


class DirectionModelTests(unittest.TestCase):
    def test_angular_distance_wraps_around_zero(self) -> None:
        self.assertAlmostEqual(angular_distance(0.05, tau - 0.05), 0.1)

    def test_worker_variation_is_sampled_inside_colonies(self) -> None:
        settings = _settings(stable_worker_sd=0.1)
        colony = create_colony(
            ColonyTraits(directional_bias=0.5, receiver_attention=0.5),
            settings,
            Random(1),
        )
        directional_biases = {
            round(worker.directional_bias, 2) for worker in colony.workers
        }

        self.assertGreater(len(directional_biases), 1)
        self.assertTrue(
            all(0.0 <= worker.directional_bias <= 1.0 for worker in colony.workers)
        )
        self.assertTrue(
            all(0.0 <= worker.receiver_attention <= 1.0 for worker in colony.workers)
        )

    def test_directional_signal_beats_random_search(self) -> None:
        settings = _settings(
            episodes_per_colony=300,
            recruits_per_episode=8,
            stable_worker_sd=0.0,
            dance_noise_sd=0.0,
            interpretation_noise_sd=0.0,
            cue_cost=0.0,
            attention_cost=0.0,
        )
        signaled = create_colony(
            ColonyTraits(directional_bias=1.0, receiver_attention=1.0),
            settings,
            Random(2),
        )
        random = create_colony(
            ColonyTraits(directional_bias=0.0, receiver_attention=0.0),
            settings,
            Random(2),
        )

        signaled_evaluation = evaluate_colony(signaled, settings, Random(3))
        random_evaluation = evaluate_colony(random, settings, Random(3))

        self.assertGreater(
            signaled_evaluation.success_rate,
            random_evaluation.success_rate + 0.4,
        )

    def test_simulation_is_reproducible(self) -> None:
        settings = _settings(
            colony_count=8,
            workers_per_colony=12,
            generations=3,
            episodes_per_colony=8,
            recruits_per_episode=3,
        )

        first = simulate(settings, seed=4)
        second = simulate(settings, seed=4)

        self.assertEqual(first, second)


def _settings(**overrides: float | int) -> DirectionSettings:
    values = {
        "colony_count": 4,
        "workers_per_colony": 20,
        "generations": 1,
        "episodes_per_colony": 20,
        "recruits_per_episode": 4,
        "mutation_sd": 0.03,
        "stable_worker_sd": 0.05,
        "max_signal_concentration": 20.0,
        "dance_noise_sd": 0.08,
        "interpretation_noise_sd": 0.08,
        "success_angle": 0.35,
        "food_value": 1.0,
        "cue_cost": 0.01,
        "attention_cost": 0.01,
    }
    values.update(overrides)
    return DirectionSettings(**values)


if __name__ == "__main__":
    unittest.main()
