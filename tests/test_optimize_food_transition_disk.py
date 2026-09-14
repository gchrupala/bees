"""Tests for the disk-ecology food-transition Optuna optimizer."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import optimize_food_transition_disk as opt  # noqa: E402

CONFIG = ROOT / "configs" / "long_vertical_transition_disk.json"


def _default_args():
    argv = sys.argv
    sys.argv = ["optimize_food_transition_disk.py"]
    try:
        return opt.parse_args()
    finally:
        sys.argv = argv


def _search_space() -> opt.DiskSearchSpace:
    return opt.build_search_space(_default_args())


class DiskOptimizerTests(unittest.TestCase):
    def test_config_is_disk_transition(self) -> None:
        settings = opt.load_settings(CONFIG)
        self.assertEqual(settings.food_geometry, "disk")
        self.assertEqual(settings.foray_distribution, "gamma")
        self.assertEqual(settings.food_capacity_scaling, "area")
        self.assertEqual(settings.food_site_count_distribution, "poisson")
        # Transition scenario: comb starts horizontal but is allowed to evolve.
        self.assertEqual(settings.initial_comb_tilt, 0.0)
        self.assertTrue(settings.evolve_comb_tilt)
        self.assertGreater(settings.vertical_comb_benefit, 0.0)

    def test_sample_settings_couples_geometry_and_range(self) -> None:
        base = opt.load_settings(CONFIG)
        space = _search_space()
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=1), direction="maximize"
        )
        sample = opt.sample_settings(study.ask(), base, space)
        s = sample.settings

        # Geometry and ecology inherited from the base config.
        self.assertEqual(s.food_geometry, "disk")
        self.assertEqual(s.foray_distribution, "gamma")
        self.assertEqual(s.food_capacity_scaling, "area")
        # Transition overrides always applied.
        self.assertEqual(s.initial_comb_tilt, 0.0)
        self.assertEqual(s.vertical_comb_modifier, "linear")
        # Foray range must be able to reach the farthest sampled patch.
        self.assertEqual(s.max_search_distance, s.food_site_max_distance)
        # Sampled values land inside the requested bounds.
        self.assertGreaterEqual(s.food_site_radius, space.food_site_radius_min)
        self.assertLessEqual(s.food_site_radius, space.food_site_radius_max)
        self.assertGreaterEqual(
            s.food_site_max_distance, space.food_site_max_distance_min
        )
        self.assertLessEqual(
            s.food_site_max_distance, space.food_site_max_distance_max
        )
        self.assertGreaterEqual(s.travel_cost_per_distance, space.travel_cost_min)
        self.assertLessEqual(s.travel_cost_per_distance, space.travel_cost_max)
        self.assertGreaterEqual(s.food_site_count, space.food_site_count_min)
        self.assertLessEqual(s.food_site_count, space.food_site_count_max)

    def test_sampler_seed_offset_decorrelates_tasks(self) -> None:
        """Two array tasks with different seed offsets must not re-draw the same
        startup parameters from a fresh (empty) study."""
        base = opt.load_settings(CONFIG)
        space = _search_space()

        def first_sample(offset: int) -> dict:
            sampler = optuna.samplers.TPESampler(
                seed=2026 + offset, n_startup_trials=64
            )
            study = optuna.create_study(sampler=sampler, direction="maximize")
            return dict(opt.sample_settings(study.ask(), base, space).values)

        self.assertNotEqual(first_sample(0), first_sample(32))

    def test_evaluate_seed_reports_dance_propensity_and_flags(self) -> None:
        base = opt.load_settings(CONFIG)
        tiny = replace(
            base,
            generations=3,
            colony_count=8,
            workers_per_colony=20,
            episodes_per_colony=10,
        )
        thresholds = opt.Thresholds(gravity=0.5, vertical=0.8, collapse_success=0.02)
        metrics = opt.evaluate_seed(tiny, seed=100, thresholds=thresholds)

        self.assertGreaterEqual(metrics["final_dance_propensity"], 0.0)
        self.assertLessEqual(metrics["final_dance_propensity"], 1.0)
        self.assertIsInstance(metrics["stable"], bool)
        self.assertIsInstance(metrics["collapsed"], bool)
        self.assertGreaterEqual(metrics["progress"], 0.0)
        self.assertLessEqual(metrics["progress"], 1.0)

    def test_travel_cost_formatting_survives_small_values(self) -> None:
        base = opt.load_settings(CONFIG)
        space = _search_space()
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=7), direction="maximize"
        )
        trial = study.ask()
        opt.sample_settings(trial, base, space)
        study.tell(trial, 0.0)
        frozen = study.trials[0]

        cell = opt.format_travel_cost(frozen)
        self.assertNotEqual(cell, "")
        # Round-trips to the sampled ~1e-5 value rather than rounding to zero.
        self.assertAlmostEqual(
            float(cell), frozen.params["travel_cost_per_distance"], places=10
        )
        self.assertGreater(float(cell), 0.0)

    def test_export_only_requires_no_simulation(self) -> None:
        """A zero-trial export pass must read an existing journal and write CSVs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            journal = tmp_path / "study.journal"
            trials_csv = tmp_path / "trials.csv"
            seed_csv = tmp_path / "seed.csv"

            argv = sys.argv
            sys.argv = [
                "optimize_food_transition_disk.py",
                "--generations", "3",
                "--n-trials", "2",
                "--workers", "1",
                "--startup-trials", "1",
                "--seeds", "100",
                "--journal-output", str(journal),
                "--trials-output", str(trials_csv),
                "--seed-output", str(seed_csv),
            ]
            try:
                opt.main()
                sys.argv = [
                    "optimize_food_transition_disk.py",
                    "--n-trials", "0",
                    "--journal-output", str(journal),
                    "--trials-output", str(trials_csv),
                    "--seed-output", str(seed_csv),
                ]
                opt.main()
            finally:
                sys.argv = argv

            trial_lines = trials_csv.read_text().strip().splitlines()
            self.assertTrue(trial_lines[0].startswith("number,state,value"))
            self.assertEqual(len(trial_lines), 3)  # header + 2 trials


if __name__ == "__main__":
    unittest.main()
