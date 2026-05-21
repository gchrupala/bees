from __future__ import annotations

import unittest
from random import Random

from bees.distance_signal import (
    DistanceSettings,
    DistanceTraits,
    distance_success,
    evaluate_lineage,
    simulate_distance_signal,
)


class DistanceSignalTests(unittest.TestCase):
    def test_search_success_declines_with_distance_error(self) -> None:
        self.assertGreater(
            distance_success(error=0.0, search_width=0.2),
            distance_success(error=0.4, search_width=0.2),
        )

    def test_clear_distance_signal_beats_random_search(self) -> None:
        settings = DistanceSettings(
            lineages=4,
            generations=1,
            episodes_per_lineage=200,
            mutation_sd=0.0,
            max_distance=1.0,
            cue_noise=0.03,
            search_width=0.15,
            random_success=0.2,
            cue_cost=0.0,
            attention_cost=0.0,
        )
        signaled = DistanceTraits(cue_weight=1.0, receiver_scale=1.0, attention=1.0)
        random = DistanceTraits(cue_weight=0.0, receiver_scale=0.0, attention=0.0)

        self.assertGreater(
            evaluate_lineage(signaled, settings, Random(1)),
            evaluate_lineage(random, settings, Random(1)),
        )

    def test_simulation_is_reproducible(self) -> None:
        settings = DistanceSettings(
            lineages=8,
            generations=4,
            episodes_per_lineage=10,
            mutation_sd=0.03,
            max_distance=1.0,
            cue_noise=0.08,
            search_width=0.2,
            random_success=0.2,
            cue_cost=0.02,
            attention_cost=0.01,
        )

        first = simulate_distance_signal(settings, seed=5)
        second = simulate_distance_signal(settings, seed=5)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
