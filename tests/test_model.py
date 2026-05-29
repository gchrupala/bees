from __future__ import annotations

import unittest
from math import tau
from random import Random

from bees.model import (
    ColonyTraits,
    DirectionSettings,
    FoodSite,
    Worker,
    angular_distance,
    circular_interpolate,
    create_colony,
    encode_dance_direction,
    evaluate_colony,
    find_food_site,
    interpret_signal,
    simulate,
)


class DirectionModelTests(unittest.TestCase):
    def test_angular_distance_wraps_around_zero(self) -> None:
        self.assertAlmostEqual(angular_distance(0.05, tau - 0.05), 0.1)

    def test_circular_interpolation_takes_short_path(self) -> None:
        self.assertAlmostEqual(circular_interpolate(tau - 0.1, 0.1, 0.5), 0.0)

    def test_worker_variation_is_sampled_inside_colonies(self) -> None:
        settings = _settings(stable_worker_sd=0.1)
        colony = create_colony(
            ColonyTraits(
                directional_bias=0.5,
                receiver_attention=0.5,
                sender_transposition=0.5,
                receiver_transposition=0.5,
            ),
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
        self.assertTrue(
            all(0.0 <= worker.sender_transposition <= 1.0 for worker in colony.workers)
        )
        self.assertTrue(
            all(0.0 <= worker.receiver_transposition <= 1.0 for worker in colony.workers)
        )

    def test_random_search_can_find_any_available_food_site(self) -> None:
        sites = (
            FoodSite(direction=0.0, width=0.1, value=1.0, capacity=1),
            FoodSite(direction=tau / 2, width=0.1, value=1.0, capacity=1),
        )
        remaining_capacity = [1, 1]

        self.assertEqual(
            find_food_site(tau / 2 + 0.01, sites, remaining_capacity),
            1,
        )

    def test_depleted_food_sites_cannot_be_found(self) -> None:
        sites = (FoodSite(direction=0.0, width=0.1, value=1.0, capacity=1),)

        self.assertIsNone(find_food_site(0.0, sites, [0]))

    def test_horizontal_comb_preserves_direct_mapping(self) -> None:
        settings = _settings(comb_tilt=0.0, interpretation_noise_sd=0.0)
        worker = Worker(
            directional_bias=1.0,
            receiver_attention=1.0,
            sender_transposition=0.0,
            receiver_transposition=0.0,
        )
        food_direction = 1.2
        signal = encode_dance_direction(food_direction, worker, settings, Random(1))
        decoded = interpret_signal(signal, worker, settings, Random(1))

        self.assertAlmostEqual(signal, food_direction)
        self.assertAlmostEqual(decoded, food_direction)

    def test_vertical_comb_requires_transposition_mapping(self) -> None:
        settings = _settings(
            comb_tilt=1.0,
            interpretation_noise_sd=0.0,
        )
        gravity_worker = Worker(
            directional_bias=1.0,
            receiver_attention=1.0,
            sender_transposition=1.0,
            receiver_transposition=1.0,
        )
        food_direction = 1.2
        signal = encode_dance_direction(
            food_direction,
            gravity_worker,
            settings,
            Random(1),
        )
        decoded = interpret_signal(signal, gravity_worker, settings, Random(1))

        self.assertAlmostEqual(signal, food_direction)
        self.assertAlmostEqual(decoded, food_direction)

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
            ColonyTraits(
                directional_bias=1.0,
                receiver_attention=1.0,
                sender_transposition=0.0,
                receiver_transposition=0.0,
            ),
            settings,
            Random(2),
        )
        random = create_colony(
            ColonyTraits(
                directional_bias=0.0,
                receiver_attention=0.0,
                sender_transposition=0.0,
                receiver_transposition=0.0,
            ),
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
        "comb_tilt": 0.0,
        "food_site_count": 1,
        "food_site_width": 0.35,
        "food_site_capacity": 8,
        "food_value": 1.0,
        "cue_cost": 0.01,
        "attention_cost": 0.01,
    }
    values.update(overrides)
    return DirectionSettings(**values)


if __name__ == "__main__":
    unittest.main()
