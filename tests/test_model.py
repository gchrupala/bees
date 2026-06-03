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
    direct_projection_strength,
    encode_dance_direction,
    evaluate_colony,
    find_food_site,
    generate_food_sites,
    gravity_reference_strength,
    interpret_signal,
    sample_sun_azimuth,
    simulate,
    _mutate_traits,
    _orientation_mean_and_alignment,
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
                search_limit=3.0,
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
        self.assertTrue(
            all(
                0.0 <= worker.search_limit <= settings.max_search_distance
                for worker in colony.workers
            )
        )

    def test_zero_transposition_mutation_correlation_uses_independent_draws(
        self,
    ) -> None:
        settings = _settings(
            mutation_sd=0.04,
            transposition_mutation_correlation=0.0,
        )
        traits = ColonyTraits(
            directional_bias=0.5,
            receiver_attention=0.5,
            sender_transposition=0.5,
            receiver_transposition=0.5,
            search_limit=3.0,
        )
        seed = 7
        expected_rng = Random(seed)
        expected_rng.gauss(0.0, settings.mutation_sd)
        expected_rng.gauss(0.0, settings.mutation_sd)
        expected_sender = traits.sender_transposition + expected_rng.gauss(
            0.0,
            settings.mutation_sd,
        )
        expected_receiver = traits.receiver_transposition + expected_rng.gauss(
            0.0,
            settings.mutation_sd,
        )

        mutated = _mutate_traits(traits, settings, Random(seed))

        self.assertAlmostEqual(mutated.sender_transposition, expected_sender)
        self.assertAlmostEqual(mutated.receiver_transposition, expected_receiver)

    def test_full_transposition_mutation_correlation_couples_increments(
        self,
    ) -> None:
        settings = _settings(
            mutation_sd=0.04,
            transposition_mutation_correlation=1.0,
        )
        traits = ColonyTraits(
            directional_bias=0.5,
            receiver_attention=0.5,
            sender_transposition=0.5,
            receiver_transposition=0.5,
            search_limit=3.0,
        )

        mutated = _mutate_traits(traits, settings, Random(8))

        self.assertAlmostEqual(
            mutated.sender_transposition,
            mutated.receiver_transposition,
        )

    def test_transposition_mutation_correlation_is_clamped(self) -> None:
        traits = ColonyTraits(
            directional_bias=0.5,
            receiver_attention=0.5,
            sender_transposition=0.5,
            receiver_transposition=0.5,
            search_limit=3.0,
        )

        bounded = _mutate_traits(
            traits,
            _settings(
                mutation_sd=0.04,
                transposition_mutation_correlation=1.0,
            ),
            Random(9),
        )
        overbounded = _mutate_traits(
            traits,
            _settings(
                mutation_sd=0.04,
                transposition_mutation_correlation=2.0,
            ),
            Random(9),
        )

        self.assertEqual(overbounded, bounded)

    def test_comb_tilt_mutation_scale_can_freeze_tilt(self) -> None:
        settings = _settings(
            mutation_sd=0.04,
            comb_tilt_mutation_sd=0.0,
        )
        traits = ColonyTraits(
            directional_bias=0.5,
            receiver_attention=0.5,
            sender_transposition=0.5,
            receiver_transposition=0.5,
            search_limit=3.0,
            comb_tilt=0.95,
            comb_orientation=0.2,
        )

        mutated = _mutate_traits(traits, settings, Random(11))

        self.assertAlmostEqual(mutated.comb_tilt, traits.comb_tilt)

    def test_axial_orientation_treats_opposite_normals_as_aligned(self) -> None:
        _, circular_alignment = _orientation_mean_and_alignment(
            [0.0, tau / 2],
            axial=False,
        )
        axial_mean, axial_alignment = _orientation_mean_and_alignment(
            [0.0, tau / 2],
            axial=True,
        )

        self.assertAlmostEqual(circular_alignment, 0.0)
        self.assertAlmostEqual(axial_alignment, 1.0)
        self.assertAlmostEqual(axial_mean, 0.0)

    def test_axial_orientation_mutation_wraps_to_half_turn_period(self) -> None:
        settings = _settings(
            comb_orientation_axial=True,
            comb_orientation_mutation_sd=0.0,
        )
        traits = ColonyTraits(
            directional_bias=0.5,
            receiver_attention=0.5,
            sender_transposition=0.5,
            receiver_transposition=0.5,
            search_limit=3.0,
            comb_tilt=0.95,
            comb_orientation=0.75 * tau,
        )

        mutated = _mutate_traits(traits, settings, Random(12))

        self.assertAlmostEqual(mutated.comb_orientation, 0.25 * tau)

    def test_random_search_can_find_any_available_food_site(self) -> None:
        sites = (
            FoodSite(direction=0.0, distance=2.0, width=0.1, value=1.0, capacity=1),
            FoodSite(
                direction=tau / 2,
                distance=2.0,
                width=0.1,
                value=1.0,
                capacity=1,
            ),
        )
        remaining_capacity = [1, 1]

        self.assertEqual(
            find_food_site(tau / 2 + 0.01, 3.0, sites, remaining_capacity),
            1,
        )

    def test_depleted_food_sites_cannot_be_found(self) -> None:
        sites = (
            FoodSite(direction=0.0, distance=2.0, width=0.1, value=1.0, capacity=1),
        )

        self.assertIsNone(find_food_site(0.0, 3.0, sites, [0]))

    def test_food_site_must_be_within_search_limit(self) -> None:
        sites = (
            FoodSite(direction=0.0, distance=4.0, width=0.1, value=1.0, capacity=1),
        )

        self.assertIsNone(find_food_site(0.0, 3.99, sites, [1]))
        self.assertEqual(find_food_site(0.0, 4.0, sites, [1]), 0)

    def test_closest_matching_food_site_is_found_first(self) -> None:
        sites = (
            FoodSite(direction=0.0, distance=4.0, width=0.1, value=1.0, capacity=1),
            FoodSite(direction=0.0, distance=2.0, width=0.1, value=1.0, capacity=1),
        )

        self.assertEqual(find_food_site(0.0, 5.0, sites, [1, 1]), 1)

    def test_generated_food_sites_have_distances_in_range(self) -> None:
        settings = _settings(
            food_site_count=20,
            food_site_min_distance=2.0,
            food_site_max_distance=4.0,
        )
        sites = generate_food_sites(settings, Random(1))

        self.assertTrue(all(2.0 <= site.distance <= 4.0 for site in sites))

    def test_horizontal_comb_has_direct_but_no_gravity_reference(self) -> None:
        self.assertAlmostEqual(direct_projection_strength(1.2, 0.0, 0.0), 1.0)
        self.assertAlmostEqual(gravity_reference_strength(0.0, 0.0), 0.0)

    def test_vertical_direct_projection_depends_on_comb_orientation(self) -> None:
        self.assertAlmostEqual(direct_projection_strength(0.0, 1.0, 0.0), 0.0)
        self.assertAlmostEqual(direct_projection_strength(tau / 4, 1.0, 0.0), 1.0)

    def test_daytime_sun_sampling_stays_within_configured_arc(self) -> None:
        settings = _settings(
            sun_azimuth_center=tau / 4,
            sun_azimuth_width=tau / 2,
        )
        samples = [sample_sun_azimuth(settings, Random(seed)) for seed in range(20)]

        self.assertTrue(all(0.0 <= sample <= tau / 2 for sample in samples))

    def test_horizontal_comb_preserves_direct_mapping(self) -> None:
        settings = _settings(initial_comb_tilt=0.0, interpretation_noise_sd=0.0)
        traits = ColonyTraits(
            directional_bias=1.0,
            receiver_attention=1.0,
            sender_transposition=0.0,
            receiver_transposition=0.0,
            search_limit=5.0,
            comb_tilt=0.0,
            comb_orientation=0.0,
        )
        worker = Worker(
            directional_bias=1.0,
            receiver_attention=1.0,
            sender_transposition=0.0,
            receiver_transposition=0.0,
            search_limit=5.0,
        )
        food_direction = 1.2
        signal = encode_dance_direction(
            food_direction,
            worker,
            traits,
            settings,
            0.0,
            Random(1),
        )
        decoded = interpret_signal(signal, worker, traits, settings, 0.0, Random(1))

        self.assertAlmostEqual(signal, food_direction)
        self.assertAlmostEqual(decoded, food_direction)

    def test_vertical_comb_supports_sun_gravity_transposition_mapping(self) -> None:
        settings = _settings(
            initial_comb_tilt=1.0,
            interpretation_noise_sd=0.0,
        )
        traits = ColonyTraits(
            directional_bias=1.0,
            receiver_attention=1.0,
            sender_transposition=1.0,
            receiver_transposition=1.0,
            search_limit=5.0,
            comb_tilt=1.0,
            comb_orientation=0.0,
        )
        gravity_worker = Worker(
            directional_bias=1.0,
            receiver_attention=1.0,
            sender_transposition=1.0,
            receiver_transposition=1.0,
            search_limit=5.0,
        )
        food_direction = 1.2
        sun_azimuth = 0.3
        signal = encode_dance_direction(
            food_direction,
            gravity_worker,
            traits,
            settings,
            sun_azimuth,
            Random(1),
        )
        decoded = interpret_signal(
            signal,
            gravity_worker,
            traits,
            settings,
            sun_azimuth,
            Random(1),
        )

        self.assertAlmostEqual(decoded, food_direction)

    def test_horizontal_comb_cannot_use_gravity_transposition_mapping(self) -> None:
        settings = _settings(
            initial_comb_tilt=0.0,
            interpretation_noise_sd=0.0,
        )
        traits = ColonyTraits(
            directional_bias=1.0,
            receiver_attention=1.0,
            sender_transposition=1.0,
            receiver_transposition=1.0,
            search_limit=5.0,
            comb_tilt=0.0,
            comb_orientation=0.0,
        )
        gravity_worker = Worker(
            directional_bias=1.0,
            receiver_attention=1.0,
            sender_transposition=1.0,
            receiver_transposition=1.0,
            search_limit=5.0,
        )
        food_direction = 1.2
        sun_azimuth = 0.3
        signal = encode_dance_direction(
            food_direction,
            gravity_worker,
            traits,
            settings,
            sun_azimuth,
            Random(1),
        )
        decoded = interpret_signal(
            signal,
            gravity_worker,
            traits,
            settings,
            sun_azimuth,
            Random(1),
        )

        self.assertGreater(angular_distance(decoded, food_direction), 0.1)

    def test_dance_following_amplifies_independent_discovery(self) -> None:
        settings = _settings(
            episodes_per_colony=400,
            foraging_attempts_per_episode=50,
            stable_worker_sd=0.0,
            max_signal_concentration=50.0,
            dance_noise_sd=0.0,
            interpretation_noise_sd=0.0,
            food_site_width=0.03,
            food_site_min_distance=2.0,
            food_site_max_distance=2.0,
            max_search_distance=5.0,
            food_site_capacity=50,
            travel_cost_per_distance=0.0,
            cue_cost=0.0,
            attention_cost=0.0,
        )
        attentive = create_colony(
            ColonyTraits(
                directional_bias=1.0,
                receiver_attention=1.0,
                sender_transposition=0.0,
                receiver_transposition=0.0,
                search_limit=5.0,
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
                search_limit=5.0,
            ),
            settings,
            Random(2),
        )

        attentive_evaluation = evaluate_colony(attentive, settings, Random(3))
        random_evaluation = evaluate_colony(random, settings, Random(3))

        self.assertGreater(
            attentive_evaluation.success_rate,
            random_evaluation.success_rate + 0.01,
        )

    def test_distance_cost_reduces_payoff_for_far_food(self) -> None:
        common = {
            "episodes_per_colony": 80,
            "foraging_attempts_per_episode": 4,
            "stable_worker_sd": 0.0,
            "food_site_width": tau,
            "food_site_capacity": 8,
            "max_search_distance": 10.0,
            "travel_cost_per_distance": 0.05,
            "cue_cost": 0.0,
            "attention_cost": 0.0,
        }
        near_settings = _settings(
            food_site_min_distance=1.0,
            food_site_max_distance=1.0,
            **common,
        )
        far_settings = _settings(
            food_site_min_distance=5.0,
            food_site_max_distance=5.0,
            **common,
        )
        traits = ColonyTraits(
            directional_bias=0.0,
            receiver_attention=0.0,
            sender_transposition=0.0,
            receiver_transposition=0.0,
            search_limit=10.0,
        )
        near_colony = create_colony(traits, near_settings, Random(2))
        far_colony = create_colony(traits, far_settings, Random(2))

        near_evaluation = evaluate_colony(near_colony, near_settings, Random(3))
        far_evaluation = evaluate_colony(far_colony, far_settings, Random(3))

        self.assertGreater(near_evaluation.payoff, far_evaluation.payoff)

    def test_base_dance_cost_applies_independent_of_directional_bias(self) -> None:
        settings = _settings(
            episodes_per_colony=10,
            foraging_attempts_per_episode=3,
            stable_worker_sd=0.0,
            food_site_width=tau,
            food_site_min_distance=1.0,
            food_site_max_distance=1.0,
            food_site_capacity=3,
            food_value=2.0,
            travel_cost_per_distance=0.0,
            base_dance_cost=0.5,
            cue_cost=0.0,
            attention_cost=0.0,
        )
        colony = create_colony(
            ColonyTraits(
                directional_bias=0.0,
                receiver_attention=0.0,
                sender_transposition=0.0,
                receiver_transposition=0.0,
                search_limit=5.0,
            ),
            settings,
            Random(2),
        )

        evaluation = evaluate_colony(colony, settings, Random(3))

        self.assertAlmostEqual(evaluation.payoff, 4.5)

    def test_vertical_comb_benefit_scales_colony_payoff(self) -> None:
        traits = ColonyTraits(
            directional_bias=0.0,
            receiver_attention=0.0,
            sender_transposition=0.0,
            receiver_transposition=0.0,
            search_limit=5.0,
            comb_tilt=1.0,
            comb_orientation=0.0,
        )
        common = {
            "episodes_per_colony": 20,
            "foraging_attempts_per_episode": 1,
            "stable_worker_sd": 0.0,
            "food_site_width": tau,
            "food_value": 1.0,
            "travel_cost_per_distance": 0.0,
            "base_dance_cost": 0.0,
            "cue_cost": 0.0,
            "attention_cost": 0.0,
        }
        no_benefit_settings = _settings(vertical_comb_benefit=0.0, **common)
        benefit_settings = _settings(vertical_comb_benefit=0.05, **common)
        no_benefit_colony = create_colony(traits, no_benefit_settings, Random(2))
        benefit_colony = create_colony(traits, benefit_settings, Random(2))

        no_benefit = evaluate_colony(
            no_benefit_colony,
            no_benefit_settings,
            Random(3),
        )
        with_benefit = evaluate_colony(benefit_colony, benefit_settings, Random(3))

        self.assertAlmostEqual(with_benefit.payoff, no_benefit.payoff * 1.05)

    def test_vertical_comb_benefit_does_not_rescue_zero_foraging_payoff(self) -> None:
        traits = ColonyTraits(
            directional_bias=0.0,
            receiver_attention=0.0,
            sender_transposition=0.0,
            receiver_transposition=0.0,
            search_limit=0.0,
            comb_tilt=1.0,
            comb_orientation=0.0,
        )
        settings = _settings(
            episodes_per_colony=20,
            foraging_attempts_per_episode=1,
            stable_worker_sd=0.0,
            food_site_width=tau,
            food_site_min_distance=1.0,
            food_site_max_distance=1.0,
            food_value=1.0,
            travel_cost_per_distance=0.0,
            base_dance_cost=0.0,
            cue_cost=0.0,
            attention_cost=0.0,
            vertical_comb_benefit=0.25,
        )
        colony = create_colony(traits, settings, Random(2))

        evaluation = evaluate_colony(colony, settings, Random(3))

        self.assertAlmostEqual(evaluation.payoff, 0.001)

    def test_threshold_vertical_comb_modifier_activates_at_threshold(self) -> None:
        common = {
            "episodes_per_colony": 20,
            "foraging_attempts_per_episode": 1,
            "stable_worker_sd": 0.0,
            "food_site_width": tau,
            "food_site_min_distance": 1.0,
            "food_site_max_distance": 1.0,
            "food_value": 2.0,
            "travel_cost_per_distance": 0.0,
            "base_dance_cost": 0.0,
            "cue_cost": 0.0,
            "attention_cost": 0.0,
            "vertical_comb_benefit": 0.25,
            "vertical_comb_modifier": "threshold_0.8",
        }
        below_traits = ColonyTraits(
            directional_bias=0.0,
            receiver_attention=0.0,
            sender_transposition=0.0,
            receiver_transposition=0.0,
            search_limit=5.0,
            comb_tilt=0.79,
            comb_orientation=0.0,
        )
        threshold_traits = ColonyTraits(
            directional_bias=0.0,
            receiver_attention=0.0,
            sender_transposition=0.0,
            receiver_transposition=0.0,
            search_limit=5.0,
            comb_tilt=0.8,
            comb_orientation=0.0,
        )
        settings = _settings(**common)
        below_colony = create_colony(below_traits, settings, Random(2))
        threshold_colony = create_colony(threshold_traits, settings, Random(2))

        below = evaluate_colony(below_colony, settings, Random(3))
        threshold = evaluate_colony(threshold_colony, settings, Random(3))

        self.assertAlmostEqual(below.payoff, 2.0)
        self.assertAlmostEqual(threshold.payoff, 2.5)

    def test_simulation_is_reproducible(self) -> None:
        settings = _settings(
            colony_count=8,
            workers_per_colony=12,
            generations=3,
            episodes_per_colony=8,
            foraging_attempts_per_episode=3,
        )

        first = simulate(settings, seed=4)
        second = simulate(settings, seed=4)

        self.assertEqual(first, second)


def _settings(**overrides: float | int | bool | str | None) -> DirectionSettings:
    values = {
        "colony_count": 4,
        "workers_per_colony": 20,
        "generations": 1,
        "episodes_per_colony": 20,
        "foraging_attempts_per_episode": 4,
        "mutation_sd": 0.03,
        "transposition_mutation_correlation": 0.6,
        "comb_orientation_mutation_sd": 0.15,
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
