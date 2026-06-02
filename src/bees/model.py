from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin, sqrt, tau
from random import Random

EPSILON = 1e-9
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class ColonyTraits:
    directional_bias: float
    receiver_attention: float
    sender_transposition: float
    receiver_transposition: float
    search_limit: float
    comb_tilt: float = 0.0
    comb_orientation: float = 0.0


@dataclass(frozen=True)
class Worker:
    directional_bias: float
    receiver_attention: float
    sender_transposition: float
    receiver_transposition: float
    search_limit: float


@dataclass(frozen=True)
class Colony:
    traits: ColonyTraits
    workers: tuple[Worker, ...]


@dataclass(frozen=True)
class FoodSite:
    direction: float
    distance: float
    width: float
    value: float
    capacity: int


@dataclass(frozen=True)
class Dance:
    signal: float


@dataclass(frozen=True)
class CombBasis:
    first_axis: Vector3
    second_axis: Vector3
    normal: Vector3


@dataclass(frozen=True)
class DirectionSettings:
    colony_count: int
    workers_per_colony: int
    generations: int
    episodes_per_colony: int
    foraging_attempts_per_episode: int
    mutation_sd: float
    transposition_mutation_correlation: float
    comb_orientation_mutation_sd: float
    stable_worker_sd: float
    max_signal_concentration: float
    dance_noise_sd: float
    interpretation_noise_sd: float
    initial_comb_tilt: float
    vertical_comb_benefit: float
    sun_azimuth_center: float
    sun_azimuth_width: float
    food_site_count: int
    food_site_width: float
    food_site_min_distance: float
    food_site_max_distance: float
    max_search_distance: float
    food_site_capacity: int
    food_value: float
    travel_cost_per_distance: float
    base_dance_cost: float
    cue_cost: float
    attention_cost: float


@dataclass(frozen=True)
class ColonyEvaluation:
    payoff: float
    success_rate: float


@dataclass(frozen=True)
class GenerationSummary:
    generation: int
    average_directional_bias: float
    average_receiver_attention: float
    average_sender_transposition: float
    average_receiver_transposition: float
    average_comb_tilt: float
    average_comb_orientation: float
    comb_orientation_alignment: float
    average_search_limit: float
    average_success_rate: float
    average_payoff: float


def angular_distance(first: float, second: float) -> float:
    return abs((first - second + pi) % tau - pi)


def circular_interpolate(first: float, second: float, weight: float) -> float:
    step = (second - first + pi) % tau - pi
    return (first + weight * step) % tau


def direct_projection_strength(
    direction: float,
    comb_tilt: float,
    comb_orientation: float,
) -> float:
    basis = _comb_basis(comb_tilt, comb_orientation)
    return _length(_project_onto_plane(_world_direction_vector(direction), basis.normal))


def gravity_reference_strength(comb_tilt: float, comb_orientation: float) -> float:
    basis = _comb_basis(comb_tilt, comb_orientation)
    return _length(_project_onto_plane((0.0, 0.0, 1.0), basis.normal))


def create_colony(
    traits: ColonyTraits,
    settings: DirectionSettings,
    rng: Random,
) -> Colony:
    workers = tuple(
        Worker(
            directional_bias=_clamp(
                traits.directional_bias + rng.gauss(0.0, settings.stable_worker_sd),
                0.0,
                1.0,
            ),
            receiver_attention=_clamp(
                traits.receiver_attention + rng.gauss(0.0, settings.stable_worker_sd),
                0.0,
                1.0,
            ),
            sender_transposition=_clamp(
                traits.sender_transposition
                + rng.gauss(0.0, settings.stable_worker_sd),
                0.0,
                1.0,
            ),
            receiver_transposition=_clamp(
                traits.receiver_transposition
                + rng.gauss(0.0, settings.stable_worker_sd),
                0.0,
                1.0,
            ),
            search_limit=_clamp(
                traits.search_limit
                + rng.gauss(
                    0.0,
                    settings.stable_worker_sd * settings.max_search_distance,
                ),
                0.0,
                settings.max_search_distance,
            ),
        )
        for _ in range(settings.workers_per_colony)
    )

    return Colony(traits=traits, workers=workers)


def encode_dance_direction(
    food_direction: float,
    worker: Worker,
    colony_traits: ColonyTraits,
    settings: DirectionSettings,
    sun_azimuth: float,
    rng: Random,
) -> float:
    direct_direction, direct_strength = _direct_signal_angle(
        food_direction,
        colony_traits,
        rng,
    )
    gravity_direction, gravity_strength = _gravity_signal_angle(
        food_direction,
        colony_traits,
        sun_azimuth,
        rng,
    )

    sender_transposition = _clamp(worker.sender_transposition, 0.0, 1.0)
    return _weighted_circular_mean(
        (
            (direct_direction, (1.0 - sender_transposition) * direct_strength),
            (gravity_direction, sender_transposition * gravity_strength),
        ),
        rng,
    )


def produce_signal(
    food_direction: float,
    worker: Worker,
    colony_traits: ColonyTraits,
    settings: DirectionSettings,
    sun_azimuth: float,
    rng: Random,
) -> float:
    dance_direction = encode_dance_direction(
        food_direction,
        worker,
        colony_traits,
        settings,
        sun_azimuth,
        rng,
    )
    concentration = worker.directional_bias * settings.max_signal_concentration
    signal = rng.vonmisesvariate(dance_direction, concentration)

    return (signal + rng.gauss(0.0, settings.dance_noise_sd)) % tau


def interpret_signal(
    signal: float,
    worker: Worker,
    colony_traits: ColonyTraits,
    settings: DirectionSettings,
    sun_azimuth: float,
    rng: Random,
) -> float:
    direct_direction, direct_strength = _direct_world_direction_from_signal(
        signal,
        colony_traits,
        rng,
    )
    gravity_direction, gravity_strength = _gravity_world_direction_from_signal(
        signal,
        colony_traits,
        sun_azimuth,
        rng,
    )
    receiver_transposition = _clamp(worker.receiver_transposition, 0.0, 1.0)
    interpreted = _weighted_circular_mean(
        (
            (direct_direction, (1.0 - receiver_transposition) * direct_strength),
            (gravity_direction, receiver_transposition * gravity_strength),
        ),
        rng,
    )

    return (interpreted + rng.gauss(0.0, settings.interpretation_noise_sd)) % tau


def sample_sun_azimuth(settings: DirectionSettings, rng: Random) -> float:
    sun_width = _clamp(settings.sun_azimuth_width, 0.0, tau)
    return (
        settings.sun_azimuth_center
        + rng.uniform(-0.5 * sun_width, 0.5 * sun_width)
    ) % tau


def generate_food_sites(settings: DirectionSettings, rng: Random) -> tuple[FoodSite, ...]:
    return tuple(
        FoodSite(
            direction=rng.random() * tau,
            distance=rng.uniform(
                settings.food_site_min_distance,
                settings.food_site_max_distance,
            ),
            width=settings.food_site_width,
            value=settings.food_value,
            capacity=settings.food_site_capacity,
        )
        for _ in range(settings.food_site_count)
    )


def find_food_site(
    search_direction: float,
    search_limit: float,
    sites: tuple[FoodSite, ...],
    remaining_capacity: list[int],
) -> int | None:
    available_sites = [
        (site.distance, angular_distance(search_direction, site.direction), index)
        for index, site in enumerate(sites)
        if remaining_capacity[index] > 0
        and angular_distance(search_direction, site.direction) <= site.width
        and site.distance <= search_limit
    ]

    if not available_sites:
        return None

    return min(available_sites)[2]


def evaluate_colony(
    colony: Colony,
    settings: DirectionSettings,
    rng: Random,
) -> ColonyEvaluation:
    total_payoff = 0.0
    total_successes = 0
    total_attempts = (
        settings.episodes_per_colony * settings.foraging_attempts_per_episode
    )

    for _ in range(settings.episodes_per_colony):
        sites = generate_food_sites(settings, rng)
        sun_azimuth = sample_sun_azimuth(settings, rng)
        remaining_capacity = [site.capacity for site in sites]
        dances: list[Dance] = []
        attention_count = 0
        dance_cost = 0.0
        success_count = 0
        food_payoff = 0.0

        for _ in range(settings.foraging_attempts_per_episode):
            worker = rng.choice(colony.workers)
            follows_dance = bool(dances) and rng.random() < worker.receiver_attention

            if follows_dance:
                dance = rng.choice(dances)
                search_direction = interpret_signal(
                    dance.signal,
                    worker,
                    colony.traits,
                    settings,
                    sun_azimuth,
                    rng,
                )
                attention_count += 1
            else:
                search_direction = rng.random() * tau

            site_index = find_food_site(
                search_direction,
                worker.search_limit,
                sites,
                remaining_capacity,
            )
            if site_index is not None:
                remaining_capacity[site_index] -= 1
                success_count += 1
                food_payoff -= (
                    sites[site_index].distance * settings.travel_cost_per_distance
                )
                food_payoff += sites[site_index].value
                if not follows_dance:
                    dances.append(
                        Dance(
                            signal=produce_signal(
                                sites[site_index].direction,
                                worker,
                                colony.traits,
                                settings,
                                sun_azimuth,
                                rng,
                            )
                        )
                    )
                    dance_cost += (
                        settings.base_dance_cost
                        + settings.cue_cost * worker.directional_bias
                    )
            else:
                food_payoff -= worker.search_limit * settings.travel_cost_per_distance

        total_successes += success_count
        total_payoff += (
            food_payoff
            - dance_cost
            - settings.attention_cost * attention_count
            + settings.vertical_comb_benefit * colony.traits.comb_tilt
        )

    return ColonyEvaluation(
        payoff=max(0.001, total_payoff / settings.episodes_per_colony),
        success_rate=total_successes / total_attempts,
    )


def simulate(
    settings: DirectionSettings,
    seed: int,
) -> list[GenerationSummary]:
    rng = Random(seed)
    colonies = [
        create_colony(_initial_traits(settings, rng), settings, rng)
        for _ in range(settings.colony_count)
    ]
    history = []

    for generation in range(settings.generations + 1):
        evaluations = [evaluate_colony(colony, settings, rng) for colony in colonies]
        history.append(_summarize(generation, colonies, evaluations))

        if generation < settings.generations:
            colonies = [
                create_colony(
                    _mutate_traits(
                        _choose_parent(colonies, evaluations, rng).traits,
                        settings,
                        rng,
                    ),
                    settings,
                    rng,
                )
                for _ in colonies
            ]

    return history


def _initial_traits(settings: DirectionSettings, rng: Random) -> ColonyTraits:
    return ColonyTraits(
        directional_bias=rng.uniform(0.0, 0.15),
        receiver_attention=rng.uniform(0.0, 0.25),
        sender_transposition=0.0,
        receiver_transposition=0.0,
        comb_tilt=_clamp(settings.initial_comb_tilt, 0.0, 1.0),
        comb_orientation=rng.random() * tau,
        search_limit=rng.uniform(
            0.15 * settings.max_search_distance,
            0.45 * settings.max_search_distance,
        ),
    )


def _mutate_traits(
    traits: ColonyTraits,
    settings: DirectionSettings,
    rng: Random,
) -> ColonyTraits:
    directional_bias_change = rng.gauss(0.0, settings.mutation_sd)
    receiver_attention_change = rng.gauss(0.0, settings.mutation_sd)
    sender_transposition_change, receiver_transposition_change = (
        _correlated_gaussian_pair(
            settings.mutation_sd,
            settings.transposition_mutation_correlation,
            rng,
        )
    )
    comb_tilt_change = rng.gauss(0.0, settings.mutation_sd)
    comb_orientation_change = rng.gauss(0.0, settings.comb_orientation_mutation_sd)
    search_limit_change = rng.gauss(
        0.0,
        settings.mutation_sd * settings.max_search_distance,
    )

    return ColonyTraits(
        directional_bias=_clamp(
            traits.directional_bias + directional_bias_change,
            0.0,
            1.0,
        ),
        receiver_attention=_clamp(
            traits.receiver_attention + receiver_attention_change,
            0.0,
            1.0,
        ),
        sender_transposition=_clamp(
            traits.sender_transposition + sender_transposition_change,
            0.0,
            1.0,
        ),
        receiver_transposition=_clamp(
            traits.receiver_transposition + receiver_transposition_change,
            0.0,
            1.0,
        ),
        comb_tilt=_clamp(
            traits.comb_tilt + comb_tilt_change,
            0.0,
            1.0,
        ),
        comb_orientation=(traits.comb_orientation + comb_orientation_change) % tau,
        search_limit=_clamp(
            traits.search_limit + search_limit_change,
            0.0,
            settings.max_search_distance,
        ),
    )


def _choose_parent(
    colonies: list[Colony],
    evaluations: list[ColonyEvaluation],
    rng: Random,
) -> Colony:
    total_payoff = sum(evaluation.payoff for evaluation in evaluations)
    threshold = rng.random() * total_payoff
    cumulative = 0.0

    for colony, evaluation in zip(colonies, evaluations):
        cumulative += evaluation.payoff
        if cumulative >= threshold:
            return colony

    return colonies[-1]


def _summarize(
    generation: int,
    colonies: list[Colony],
    evaluations: list[ColonyEvaluation],
) -> GenerationSummary:
    count = len(colonies)
    orientation_x = sum(cos(colony.traits.comb_orientation) for colony in colonies)
    orientation_y = sum(sin(colony.traits.comb_orientation) for colony in colonies)
    orientation_alignment = hypot(orientation_x, orientation_y) / count
    average_comb_orientation = (
        atan2(orientation_y, orientation_x) % tau
        if orientation_alignment > EPSILON
        else 0.0
    )

    return GenerationSummary(
        generation=generation,
        average_directional_bias=sum(
            colony.traits.directional_bias for colony in colonies
        )
        / count,
        average_receiver_attention=sum(
            colony.traits.receiver_attention for colony in colonies
        )
        / count,
        average_sender_transposition=sum(
            colony.traits.sender_transposition for colony in colonies
        )
        / count,
        average_receiver_transposition=sum(
            colony.traits.receiver_transposition for colony in colonies
        )
        / count,
        average_comb_tilt=sum(colony.traits.comb_tilt for colony in colonies) / count,
        average_comb_orientation=average_comb_orientation,
        comb_orientation_alignment=orientation_alignment,
        average_search_limit=sum(colony.traits.search_limit for colony in colonies)
        / count,
        average_success_rate=sum(
            evaluation.success_rate for evaluation in evaluations
        )
        / count,
        average_payoff=sum(evaluation.payoff for evaluation in evaluations) / count,
    )


def _direct_signal_angle(
    food_direction: float,
    colony_traits: ColonyTraits,
    rng: Random,
) -> tuple[float, float]:
    basis = _comb_basis(colony_traits.comb_tilt, colony_traits.comb_orientation)
    return _projected_angle_and_strength(
        _world_direction_vector(food_direction),
        basis,
        rng,
    )


def _gravity_signal_angle(
    food_direction: float,
    colony_traits: ColonyTraits,
    sun_azimuth: float,
    rng: Random,
) -> tuple[float, float]:
    reference_angle, reference_strength = _gravity_reference_angle(
        colony_traits,
        rng,
    )
    return (reference_angle + food_direction - sun_azimuth) % tau, reference_strength


def _direct_world_direction_from_signal(
    signal: float,
    colony_traits: ColonyTraits,
    rng: Random,
) -> tuple[float, float]:
    basis = _comb_basis(colony_traits.comb_tilt, colony_traits.comb_orientation)
    signal_vector = _comb_vector(signal, basis)
    return _horizontal_angle_and_strength(signal_vector, rng)


def _gravity_world_direction_from_signal(
    signal: float,
    colony_traits: ColonyTraits,
    sun_azimuth: float,
    rng: Random,
) -> tuple[float, float]:
    reference_angle, reference_strength = _gravity_reference_angle(
        colony_traits,
        rng,
    )
    return (sun_azimuth + signal - reference_angle) % tau, reference_strength


def _gravity_reference_angle(
    colony_traits: ColonyTraits,
    rng: Random,
) -> tuple[float, float]:
    basis = _comb_basis(colony_traits.comb_tilt, colony_traits.comb_orientation)
    return _projected_angle_and_strength((0.0, 0.0, 1.0), basis, rng)


def _projected_angle_and_strength(
    vector: Vector3,
    basis: CombBasis,
    rng: Random,
) -> tuple[float, float]:
    projected = _project_onto_plane(vector, basis.normal)
    strength = _length(projected)

    if strength <= EPSILON:
        return rng.random() * tau, 0.0

    first_component = _dot(projected, basis.first_axis)
    second_component = _dot(projected, basis.second_axis)
    angle = atan2(second_component, first_component) % tau
    return _degrade_angle_by_strength(angle, strength, rng), strength


def _horizontal_angle_and_strength(
    vector: Vector3,
    rng: Random,
) -> tuple[float, float]:
    strength = hypot(vector[0], vector[1])

    if strength <= EPSILON:
        return rng.random() * tau, 0.0

    angle = atan2(vector[1], vector[0]) % tau
    return _degrade_angle_by_strength(angle, strength, rng), strength


def _degrade_angle_by_strength(angle: float, strength: float, rng: Random) -> float:
    bounded_strength = _clamp(strength, 0.0, 1.0)

    if bounded_strength >= 1.0:
        return angle

    return circular_interpolate(rng.random() * tau, angle, bounded_strength)


def _weighted_circular_mean(
    angles_and_weights: tuple[tuple[float, float], ...],
    rng: Random,
) -> float:
    x_component = 0.0
    y_component = 0.0

    for angle, weight in angles_and_weights:
        bounded_weight = max(0.0, weight)
        x_component += bounded_weight * cos(angle)
        y_component += bounded_weight * sin(angle)

    if hypot(x_component, y_component) <= EPSILON:
        return rng.random() * tau

    return atan2(y_component, x_component) % tau


def _comb_basis(comb_tilt: float, comb_orientation: float) -> CombBasis:
    normal = _comb_normal(comb_tilt, comb_orientation)
    first_axis = _project_onto_plane((1.0, 0.0, 0.0), normal)

    if _length(first_axis) <= EPSILON:
        first_axis = _project_onto_plane((0.0, 1.0, 0.0), normal)

    first_axis = _normalize(first_axis)
    second_axis = _normalize(_cross(normal, first_axis))

    return CombBasis(
        first_axis=first_axis,
        second_axis=second_axis,
        normal=normal,
    )


def _comb_normal(comb_tilt: float, comb_orientation: float) -> Vector3:
    tilt_angle = _clamp(comb_tilt, 0.0, 1.0) * pi / 2.0

    return (
        sin(tilt_angle) * cos(comb_orientation),
        sin(tilt_angle) * sin(comb_orientation),
        cos(tilt_angle),
    )


def _comb_vector(angle: float, basis: CombBasis) -> Vector3:
    return _add(
        _scale(basis.first_axis, cos(angle)),
        _scale(basis.second_axis, sin(angle)),
    )


def _world_direction_vector(direction: float) -> Vector3:
    return (cos(direction), sin(direction), 0.0)


def _project_onto_plane(vector: Vector3, normal: Vector3) -> Vector3:
    return _subtract(vector, _scale(normal, _dot(vector, normal)))


def _normalize(vector: Vector3) -> Vector3:
    length = _length(vector)

    if length <= EPSILON:
        return (1.0, 0.0, 0.0)

    return _scale(vector, 1.0 / length)


def _length(vector: Vector3) -> float:
    return sqrt(_dot(vector, vector))


def _dot(first: Vector3, second: Vector3) -> float:
    return (
        first[0] * second[0]
        + first[1] * second[1]
        + first[2] * second[2]
    )


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _add(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    )


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (scalar * vector[0], scalar * vector[1], scalar * vector[2])


def _correlated_gaussian_pair(
    standard_deviation: float,
    correlation: float,
    rng: Random,
) -> tuple[float, float]:
    bounded_correlation = _clamp(correlation, 0.0, 1.0)
    first = rng.gauss(0.0, standard_deviation)
    independent_second = rng.gauss(0.0, standard_deviation)

    # Preserve the marginal mutation scale while changing only the correlation.
    second = (
        bounded_correlation * first
        + sqrt(1.0 - bounded_correlation**2) * independent_second
    )

    return first, second


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
