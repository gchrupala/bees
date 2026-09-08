"""Tests for the assumption-sensitivity knobs.

Each knob defaults to the behavior of the published model; these tests pin both
the default and the alternative, so a sensitivity run cannot silently differ
from the baseline for the wrong reason.
"""

from __future__ import annotations

import unittest
from math import tau
from random import Random

from bees.model import (
    MIN_COLONY_PAYOFF,
    ColonyEvaluation,
    DirectionSettings,
    Worker,
    create_colony,
    evaluate_colony,
    simulate,
    _choose_parent,
    _initial_traits,
    _scout_dances,
    _site_capacity,
)


class DanceCapacityBasisTests(unittest.TestCase):
    def _worker(self, propensity: float) -> Worker:
        return Worker(
            directional_bias=0.5,
            receiver_attention=0.5,
            sender_transposition=0.0,
            receiver_transposition=0.0,
            search_limit=3.0,
            dance_propensity=propensity,
        )

    def test_exhausted_patch_never_dances_by_default(self) -> None:
        settings = _settings(evolve_dance_propensity=True)
        self.assertFalse(
            _scout_dances(self._worker(1.0), 0, settings, Random(0)),
            "a patch with nothing left must not be advertised",
        )

    def test_pre_visit_basis_lets_a_one_load_patch_dance(self) -> None:
        settings = _settings(
            evolve_dance_propensity=True, dance_capacity_basis="pre_visit"
        )
        self.assertTrue(
            _scout_dances(self._worker(1.0), 0, settings, Random(0)),
            "under pre_visit the finder advertises the patch she just emptied",
        )

    def test_pre_visit_basis_shifts_probability_by_one_load(self) -> None:
        # Remaining capacity r under "pre_visit" must behave like r + 1 under
        # "remaining": same probability, hence same decision for a shared rng.
        remaining = _settings(evolve_dance_propensity=True)
        pre_visit = _settings(
            evolve_dance_propensity=True, dance_capacity_basis="pre_visit"
        )
        worker = self._worker(0.3)
        for draw_seed in range(20):
            self.assertEqual(
                _scout_dances(worker, 2, pre_visit, Random(draw_seed)),
                _scout_dances(worker, 3, remaining, Random(draw_seed)),
            )

    def test_unknown_basis_raises(self) -> None:
        settings = _settings(evolve_dance_propensity=True, dance_capacity_basis="nope")
        with self.assertRaises(ValueError):
            _scout_dances(self._worker(0.5), 1, settings, Random(0))


class CapacityFloorTests(unittest.TestCase):
    def _area_settings(self, **overrides: object) -> DirectionSettings:
        return _settings(
            food_geometry="disk",
            food_capacity_scaling="area",
            food_capacity_reference_radius=150.0,
            food_site_capacity=6,
            **overrides,
        )

    def test_default_floor_gives_small_patches_one_load(self) -> None:
        settings = self._area_settings()
        self.assertEqual(_site_capacity(settings, 15.0), 1)
        self.assertEqual(_site_capacity(settings, 74.0), 1)

    def test_zero_floor_lets_small_patches_hold_nothing(self) -> None:
        settings = self._area_settings(food_capacity_floor=0)
        self.assertEqual(_site_capacity(settings, 15.0), 0)

    def test_floor_does_not_touch_patches_above_the_threshold(self) -> None:
        default = self._area_settings()
        zero = self._area_settings(food_capacity_floor=0)
        for radius in (75.0, 150.0, 300.0, 600.0):
            self.assertEqual(
                _site_capacity(default, radius), _site_capacity(zero, radius)
            )


class CommonEpisodeDrawTests(unittest.TestCase):
    def _payoff(self, behavior_seed: int, env_seed: int) -> float:
        # Food-rich enough that payoffs sit above the floor and can differ.
        settings = _settings(
            food_site_count=4,
            food_site_width=1.0,
            food_site_capacity=50,
            foraging_attempts_per_episode=10,
        )
        colony = create_colony(_initial_traits(settings, Random(3)), settings, Random(3))
        return evaluate_colony(
            colony, settings, Random(behavior_seed), env_rng=Random(env_seed)
        ).payoff

    def test_environments_come_from_env_rng(self) -> None:
        # Same colony and same behavior draws, different environments.
        self.assertNotEqual(self._payoff(11, 99), self._payoff(11, 100))

    def test_behavior_still_comes_from_the_main_rng(self) -> None:
        # Same environments, different behavior draws.
        self.assertNotEqual(self._payoff(11, 99), self._payoff(12, 99))

    def test_evaluation_is_reproducible_given_both_seeds(self) -> None:
        self.assertEqual(self._payoff(11, 99), self._payoff(11, 99))

    def test_shared_draws_are_reproducible_and_differ_from_default(self) -> None:
        base = _settings(generations=3, colony_count=6)
        shared = _settings(generations=3, colony_count=6, common_episode_draws=True)
        self.assertEqual(
            [s.average_payoff for s in simulate(shared, seed=5)],
            [s.average_payoff for s in simulate(shared, seed=5)],
        )
        self.assertNotEqual(
            [s.average_payoff for s in simulate(base, seed=5)],
            [s.average_payoff for s in simulate(shared, seed=5)],
        )


class SelectionTests(unittest.TestCase):
    def _evaluations(self, payoffs: list[float]) -> list[ColonyEvaluation]:
        return [
            ColonyEvaluation(payoff=payoff, success_rate=0.0)
            for payoff in payoffs
        ]

    def _colonies(self, count: int) -> list:
        settings = _settings()
        return [
            create_colony(_initial_traits(settings, Random(i)), settings, Random(i))
            for i in range(count)
        ]

    def test_tournament_prefers_the_higher_payoff(self) -> None:
        settings = _settings(selection="tournament", tournament_size=3)
        colonies = self._colonies(6)
        payoffs = [MIN_COLONY_PAYOFF] * 5 + [4.0]
        evaluations = self._evaluations(payoffs)
        rng = Random(0)
        picks = [
            _choose_parent(colonies, evaluations, settings, rng) for _ in range(400)
        ]
        share = sum(1 for c in picks if c is colonies[5]) / len(picks)
        # A size-3 tournament picks the single best colony whenever it is drawn.
        self.assertGreater(share, 0.35)
        self.assertLess(share, 0.65)

    def test_tournament_separates_colonies_that_proportional_cannot(self) -> None:
        # Payoffs below the floor are clipped to it, so proportional selection
        # sees a flat population; tournament still orders any that differ.
        settings = _settings(selection="tournament", tournament_size=2)
        colonies = self._colonies(4)
        evaluations = self._evaluations(
            [MIN_COLONY_PAYOFF, MIN_COLONY_PAYOFF, MIN_COLONY_PAYOFF, 0.002]
        )
        rng = Random(1)
        picks = [
            _choose_parent(colonies, evaluations, settings, rng) for _ in range(400)
        ]
        share = sum(1 for c in picks if c is colonies[3]) / len(picks)
        self.assertGreater(share, 0.3)

    def test_unknown_selection_raises(self) -> None:
        settings = _settings(selection="nope")
        colonies = self._colonies(3)
        with self.assertRaises(ValueError):
            _choose_parent(colonies, self._evaluations([1.0, 1.0, 1.0]), settings, Random(0))


class FloorFractionTests(unittest.TestCase):
    def test_all_colonies_on_the_floor_when_there_is_no_food(self) -> None:
        settings = _settings(generations=1, food_site_count=0)
        for state in simulate(settings, seed=2):
            self.assertEqual(state.floor_fraction, 1.0)

    def test_some_colonies_clear_the_floor_when_food_is_plentiful(self) -> None:
        settings = _settings(
            generations=1,
            food_site_count=6,
            food_site_width=1.2,
            food_site_capacity=50,
        )
        self.assertLess(simulate(settings, seed=2)[-1].floor_fraction, 1.0)


def _settings(**overrides: object) -> DirectionSettings:
    values: dict[str, object] = {
        "colony_count": 4,
        "workers_per_colony": 20,
        "generations": 1,
        "episodes_per_colony": 20,
        "foraging_attempts_per_episode": 4,
        "mutation_sd": 0.03,
        "transposition_mutation_correlation": 0.6,
        "stable_worker_sd": 0.05,
        "max_signal_concentration": 20.0,
        "dance_noise_sd": 0.08,
        "interpretation_noise_sd": 0.08,
        "initial_comb_tilt": 0.0,
        "vertical_comb_benefit": 0.0,
        "sun_azimuth_center": tau / 2,
        "sun_azimuth_width": tau / 2,
        "food_site_count": 1,
        "food_site_width": 0.35,
        "food_site_min_distance": 1.0,
        "food_site_max_distance": 5.0,
        "max_search_distance": 5.0,
        "food_site_capacity": 8,
        "food_value": 1.0,
        "travel_cost_per_distance": 0.0,
        "base_dance_cost": 0.0,
        "cue_cost": 0.01,
        "attention_cost": 0.01,
    }
    values.update(overrides)
    return DirectionSettings(**values)


if __name__ == "__main__":
    unittest.main()
