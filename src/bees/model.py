from __future__ import annotations

from dataclasses import dataclass
from math import pi, tau
from random import Random


@dataclass(frozen=True)
class ColonyTraits:
    directional_bias: float
    receiver_attention: float
    sender_transposition: float
    receiver_transposition: float
    search_limit: float


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
class DirectionSettings:
    colony_count: int
    workers_per_colony: int
    generations: int
    episodes_per_colony: int
    foraging_attempts_per_episode: int
    mutation_sd: float
    stable_worker_sd: float
    max_signal_concentration: float
    dance_noise_sd: float
    interpretation_noise_sd: float
    comb_tilt: float
    food_site_count: int
    food_site_width: float
    food_site_min_distance: float
    food_site_max_distance: float
    max_search_distance: float
    food_site_capacity: int
    food_value: float
    travel_cost_per_distance: float
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
    average_search_limit: float
    average_success_rate: float
    average_payoff: float


def angular_distance(first: float, second: float) -> float:
    return abs((first - second + pi) % tau - pi)


def circular_interpolate(first: float, second: float, weight: float) -> float:
    step = (second - first + pi) % tau - pi
    return (first + weight * step) % tau


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
    settings: DirectionSettings,
    rng: Random,
) -> float:
    direct_direction = _direct_mapping_direction(food_direction, settings, rng)
    gravity_direction = food_direction

    return circular_interpolate(
        direct_direction,
        gravity_direction,
        worker.sender_transposition,
    )


def produce_signal(
    food_direction: float,
    worker: Worker,
    settings: DirectionSettings,
    rng: Random,
) -> float:
    dance_direction = encode_dance_direction(food_direction, worker, settings, rng)
    concentration = worker.directional_bias * settings.max_signal_concentration
    signal = rng.vonmisesvariate(dance_direction, concentration)

    return (signal + rng.gauss(0.0, settings.dance_noise_sd)) % tau


def interpret_signal(
    signal: float,
    worker: Worker,
    settings: DirectionSettings,
    rng: Random,
) -> float:
    direct_direction = _direct_mapping_direction(signal, settings, rng)
    gravity_direction = signal
    interpreted = circular_interpolate(
        direct_direction,
        gravity_direction,
        worker.receiver_transposition,
    )

    return (interpreted + rng.gauss(0.0, settings.interpretation_noise_sd)) % tau


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
                    settings,
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
                                settings,
                                rng,
                            )
                        )
                    )
                    dance_cost += settings.cue_cost * worker.directional_bias
            else:
                food_payoff -= worker.search_limit * settings.travel_cost_per_distance

        total_successes += success_count
        total_payoff += (
            food_payoff
            - dance_cost
            - settings.attention_cost * attention_count
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
        sender_transposition=rng.uniform(0.0, 0.10),
        receiver_transposition=rng.uniform(0.0, 0.10),
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
    return ColonyTraits(
        directional_bias=_clamp(
            traits.directional_bias + rng.gauss(0.0, settings.mutation_sd),
            0.0,
            1.0,
        ),
        receiver_attention=_clamp(
            traits.receiver_attention + rng.gauss(0.0, settings.mutation_sd),
            0.0,
            1.0,
        ),
        sender_transposition=_clamp(
            traits.sender_transposition + rng.gauss(0.0, settings.mutation_sd),
            0.0,
            1.0,
        ),
        receiver_transposition=_clamp(
            traits.receiver_transposition + rng.gauss(0.0, settings.mutation_sd),
            0.0,
            1.0,
        ),
        search_limit=_clamp(
            traits.search_limit
            + rng.gauss(0.0, settings.mutation_sd * settings.max_search_distance),
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
        average_search_limit=sum(colony.traits.search_limit for colony in colonies)
        / count,
        average_success_rate=sum(
            evaluation.success_rate for evaluation in evaluations
        )
        / count,
        average_payoff=sum(evaluation.payoff for evaluation in evaluations) / count,
    )


def _direct_mapping_direction(
    direction: float,
    settings: DirectionSettings,
    rng: Random,
) -> float:
    direct_quality = 1.0 - _clamp(settings.comb_tilt, 0.0, 1.0)
    random_direction = rng.random() * tau

    return circular_interpolate(random_direction, direction, direct_quality)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
