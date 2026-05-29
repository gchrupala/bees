from __future__ import annotations

import unittest
from random import Random

from bees.spatial_model import (
    SpatialSettings,
    simulate_spatial,
)


class SpatialModelTests(unittest.TestCase):
    def test_simulation_reproducible(self) -> None:
        settings = SpatialSettings(
            colony_count=4,
            workers_per_colony=8,
            generations=2,
            episodes_per_colony=6,
            recruits_per_episode=3,
            mutation_sd=0.03,
            stable_worker_sd=0.02,
            cue_cost=0.01,
            attention_cost=0.01,
            world_size=50.0,
            forager_speed=1.0,
            travel_cost_per_unit=0.01,
            perception_radius=5.0,
            food_site_count=2,
            food_site_radius=2.0,
            food_site_capacity=4,
            food_value=1.0,
            signal_noise_sd=1.0,
        )

        first = simulate_spatial(settings, seed=7)
        second = simulate_spatial(settings, seed=7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
