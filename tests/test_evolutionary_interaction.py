from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evolutionary_interaction import (
    MUTATION_SD_VALUES,
    TRANSPOSITION_MUTATION_CORRELATION_VALUES,
    VERTICAL_COMB_BENEFIT_VALUES,
    expected_point_count,
)
from run_generation_sensitivity import (
    build_generation_points,
    build_jobs as build_generation_jobs,
    parse_generations,
)
from run_evolutionary_interaction_array import build_interaction_points, build_jobs
from run_food_transition_oat_sensitivity import BASELINE_VALUES


class EvolutionaryInteractionTests(unittest.TestCase):
    def test_grid_uses_shared_mutation_scale_and_baseline_ecology(self) -> None:
        points = build_interaction_points()

        self.assertEqual(len(points), expected_point_count())
        self.assertEqual(expected_point_count(), 48)
        self.assertEqual(
            {point.values["vertical_comb_benefit"] for point in points},
            set(VERTICAL_COMB_BENEFIT_VALUES),
        )
        self.assertEqual(
            {point.values["mutation_sd"] for point in points},
            set(MUTATION_SD_VALUES),
        )
        self.assertEqual(
            {
                point.values["transposition_mutation_correlation"]
                for point in points
            },
            set(TRANSPOSITION_MUTATION_CORRELATION_VALUES),
        )
        self.assertEqual(len({point.point for point in points}), len(points))

        ecology_parameters = (
            "food_site_count",
            "food_site_width",
            "food_site_capacity",
            "food_value",
            "food_site_max_distance",
            "travel_cost_per_distance",
        )
        for point in points:
            for parameter in ecology_parameters:
                self.assertEqual(point.values[parameter], BASELINE_VALUES[parameter])

    def test_shard_assignment_covers_each_job_once(self) -> None:
        points = build_interaction_points(max_points=3)
        jobs = build_jobs(points, seeds=[96, 97])

        assigned = [
            job.index
            for shard_index in range(4)
            for job in jobs
            if job.index % 4 == shard_index
        ]

        self.assertEqual(sorted(assigned), list(range(len(jobs))))

    def test_generation_sensitivity_varies_only_generation_budget(self) -> None:
        points = build_generation_points(
            generations=[240, 480],
            baseline_values=BASELINE_VALUES,
        )

        self.assertEqual(parse_generations("240, 480"), [240, 480])
        self.assertEqual([point.generations for point in points], [240, 480])
        self.assertEqual([point.is_baseline for point in points], [False, False])
        for point in points:
            self.assertEqual(point.values["vertical_comb_benefit"], 0.10)
            self.assertEqual(point.values["mutation_sd"], 0.045)
            self.assertEqual(
                point.values["transposition_mutation_correlation"],
                0.30,
            )
            self.assertEqual(
                point.values["food_site_count"],
                BASELINE_VALUES["food_site_count"],
            )

    def test_generation_sensitivity_jobs_cover_each_point_seed_once(self) -> None:
        points = build_generation_points(generations=[240, 480])
        jobs = build_generation_jobs(points, seeds=[300, 301])

        self.assertEqual(len(jobs), 4)
        self.assertEqual(
            {(job.point.generations, job.seed) for job in jobs},
            {(240, 300), (240, 301), (480, 300), (480, 301)},
        )
        self.assertEqual([job.index for job in jobs], list(range(4)))


if __name__ == "__main__":
    unittest.main()
