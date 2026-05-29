from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from random import Random
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class ColonyTraits:
    directional_bias: float
    receiver_attention: float
    sender_transposition: float
    receiver_transposition: float


@dataclass(frozen=True)
class Worker:
    directional_bias: float
    receiver_attention: float
    sender_transposition: float
    receiver_transposition: float


@dataclass(frozen=True)
class Colony:
    traits: ColonyTraits
    workers: tuple[Worker, ...]


@dataclass(frozen=True)
class SpatialFoodSite:
    x: float
    y: float
    radius: float
    value: float
    capacity: int


@dataclass(frozen=True)
class SpatialSettings:
    colony_count: int
    workers_per_colony: int
    generations: int
    episodes_per_colony: int
    recruits_per_episode: int
    mutation_sd: float
    stable_worker_sd: float
    cue_cost: float
    attention_cost: float

    # spatial params
    world_size: float
    forager_speed: float
    travel_cost_per_unit: float
    perception_radius: float
    food_site_count: int
    food_site_radius: float
    food_site_capacity: int
    food_value: float

    # signaling noise (higher means noisier)
    signal_noise_sd: float


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
    average_success_rate: float
    average_payoff: float


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _initial_traits(rng: Random) -> ColonyTraits:
    return ColonyTraits(
        directional_bias=rng.uniform(0.0, 0.15),
        receiver_attention=rng.uniform(0.0, 0.25),
        sender_transposition=rng.uniform(0.0, 0.10),
        receiver_transposition=rng.uniform(0.0, 0.10),
    )


def create_colony(traits: ColonyTraits, settings: SpatialSettings, rng: Random) -> Colony:
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
                traits.sender_transposition + rng.gauss(0.0, settings.stable_worker_sd),
                0.0,
                1.0,
            ),
            receiver_transposition=_clamp(
                traits.receiver_transposition + rng.gauss(0.0, settings.stable_worker_sd),
                0.0,
                1.0,
            ),
        )
        for _ in range(settings.workers_per_colony)
    )

    return Colony(traits=traits, workers=workers)


def generate_food_sites(settings: SpatialSettings, rng: Random) -> Tuple[SpatialFoodSite, ...]:
    sites: List[SpatialFoodSite] = []
    for _ in range(settings.food_site_count):
        x = rng.random() * settings.world_size
        y = rng.random() * settings.world_size
        sites.append(
            SpatialFoodSite(
                x=x,
                y=y,
                radius=settings.food_site_radius,
                value=settings.food_value,
                capacity=settings.food_site_capacity,
            )
        )
    return tuple(sites)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _produce_signal(
    site: SpatialFoodSite, worker: Worker, settings: SpatialSettings, rng: Random
) -> Tuple[float, float]:
    """Return signaled coordinates (x,y) with noise depending on sender accuracy."""
    # sender_transposition in [0,1], higher -> more accurate (less noise)
    noise_scale = (1.0 - worker.sender_transposition) * settings.signal_noise_sd
    x = site.x + rng.gauss(0.0, noise_scale)
    y = site.y + rng.gauss(0.0, noise_scale)
    return x, y


def _interpret_signal(
    signal_x: float, signal_y: float, worker: Worker, settings: SpatialSettings, rng: Random
) -> Tuple[float, float]:
    """Return interpreted coordinates (x,y) with receiver noise."""
    noise_scale = (1.0 - worker.receiver_transposition) * settings.signal_noise_sd
    x = signal_x + rng.gauss(0.0, noise_scale)
    y = signal_y + rng.gauss(0.0, noise_scale)
    return x, y


def _find_site_by_point(
    point: Tuple[float, float], sites: Tuple[SpatialFoodSite, ...], remaining_capacity: List[int]
) -> int | None:
    # Return index of a site whose circle contains the point and has capacity.
    for idx, site in enumerate(sites):
        if remaining_capacity[idx] <= 0:
            continue
        if _distance((site.x, site.y), point) <= site.radius:
            return idx
    return None


def evaluate_colony(
    colony: Colony, settings: SpatialSettings, rng: Random
) -> ColonyEvaluation:
    hive = (settings.world_size / 2.0, settings.world_size / 2.0)

    total_payoff = 0.0
    total_successes = 0
    total_recruits = settings.episodes_per_colony * settings.recruits_per_episode

    for _ in range(settings.episodes_per_colony):
        sites = generate_food_sites(settings, rng)
        remaining_capacity = [site.capacity for site in sites]

        scout_site = rng.choice(sites)
        scout = rng.choice(colony.workers)
        signal_x, signal_y = _produce_signal(scout_site, scout, settings, rng)

        # For each recruit, determine their target point and travel time/cost
        arrivals: List[Tuple[float, int, float, int]] = []
        # list of tuples (travel_time, site_index_guess, travel_distance, attention_flag)
        attention_count = 0

        for _ in range(settings.recruits_per_episode):
            recruit = rng.choice(colony.workers)
            if rng.random() < recruit.receiver_attention:
                # attends to the dance
                attention_count += 1
                tx, ty = _interpret_signal(signal_x, signal_y, recruit, settings, rng)
                # find which site this point corresponds to (if any)
                guess = _find_site_by_point((tx, ty), sites, remaining_capacity)
                # if no site contains the interpreted point, treat forager as going to the point
                # but we'll still compute distance to nearest site for success checks
                # travel target point is (tx,ty)
                target_point = (tx, ty)
                # approximate travel distance from hive to target point
                dist = _distance(hive, target_point)
            else:
                # random search: pick a random point in world
                rx = rng.random() * settings.world_size
                ry = rng.random() * settings.world_size
                target_point = (rx, ry)
                # see if this random point is inside any site
                guess = _find_site_by_point(target_point, sites, remaining_capacity)
                dist = _distance(hive, target_point)

            travel_time = dist / max(1e-9, settings.forager_speed)
            travel_cost = dist * settings.travel_cost_per_unit
            # store arrival with associated details; site index guess is used to resolve success
            arrivals.append((travel_time, guess if guess is not None else -1, dist, 1 if guess is not None else 0))

        # process arrivals in order of travel time to resolve capacity
        arrivals.sort(key=lambda t: t[0])
        episode_payoff = 0.0
        episode_successes = 0
        # consume capacity as arrivals happen
        for travel_time, guessed_idx, dist, guessed_flag in arrivals:
            # if guessed_idx == -1, the forager searched a point not directly matching any site
            if guessed_idx == -1:
                # attempt to find nearest site within perception radius
                # compute nearest site distance
                nearest_idx = None
                nearest_dist = float("inf")
                for idx, site in enumerate(sites):
                    if remaining_capacity[idx] <= 0:
                        continue
                    d = _distance((site.x, site.y), hive)  # distance from hive to site center
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_idx = idx
                # if nearest site within perception radius of target point, count as success
                if nearest_idx is not None and nearest_dist <= settings.perception_radius + sites[nearest_idx].radius:
                    site_idx = nearest_idx
                else:
                    site_idx = None
            else:
                site_idx = guessed_idx if remaining_capacity[guessed_idx] > 0 else None

            if site_idx is not None:
                # success
                remaining_capacity[site_idx] -= 1
                episode_successes += 1
                episode_payoff += sites[site_idx].value

            # subtract travel cost regardless of success
            episode_payoff -= dist * settings.travel_cost_per_unit

        total_successes += episode_successes
        total_payoff += episode_payoff - settings.cue_cost * scout.directional_bias - settings.attention_cost * attention_count

    # average per episode
    return ColonyEvaluation(payoff=max(0.001, total_payoff / settings.episodes_per_colony), success_rate=total_successes / total_recruits)


def _mutate_traits(traits: ColonyTraits, settings: SpatialSettings, rng: Random) -> ColonyTraits:
    return ColonyTraits(
        directional_bias=_clamp(traits.directional_bias + rng.gauss(0.0, settings.mutation_sd), 0.0, 1.0),
        receiver_attention=_clamp(traits.receiver_attention + rng.gauss(0.0, settings.mutation_sd), 0.0, 1.0),
        sender_transposition=_clamp(traits.sender_transposition + rng.gauss(0.0, settings.mutation_sd), 0.0, 1.0),
        receiver_transposition=_clamp(traits.receiver_transposition + rng.gauss(0.0, settings.mutation_sd), 0.0, 1.0),
    )


def _choose_parent(colonies: List[Colony], evaluations: List[ColonyEvaluation], rng: Random) -> Colony:
    total_payoff = sum(evaluation.payoff for evaluation in evaluations)
    threshold = rng.random() * total_payoff
    cumulative = 0.0
    for colony, evaluation in zip(colonies, evaluations):
        cumulative += evaluation.payoff
        if cumulative >= threshold:
            return colony
    return colonies[-1]


def _summarize(generation: int, colonies: List[Colony], evaluations: List[ColonyEvaluation]) -> GenerationSummary:
    count = len(colonies)
    return GenerationSummary(
        generation=generation,
        average_directional_bias=sum(colony.traits.directional_bias for colony in colonies) / count,
        average_receiver_attention=sum(colony.traits.receiver_attention for colony in colonies) / count,
        average_sender_transposition=sum(colony.traits.sender_transposition for colony in colonies) / count,
        average_receiver_transposition=sum(colony.traits.receiver_transposition for colony in colonies) / count,
        average_success_rate=sum(evaluation.success_rate for evaluation in evaluations) / count,
        average_payoff=sum(evaluation.payoff for evaluation in evaluations) / count,
    )


def simulate_spatial(settings: SpatialSettings, seed: int) -> List[GenerationSummary]:
    rng = Random(seed)
    colonies = [create_colony(_initial_traits(rng), settings, rng) for _ in range(settings.colony_count)]
    history: List[GenerationSummary] = []
    for generation in range(settings.generations + 1):
        evaluations = [evaluate_colony(colony, settings, rng) for colony in colonies]
        history.append(_summarize(generation, colonies, evaluations))
        if generation < settings.generations:
            colonies = [create_colony(_mutate_traits(_choose_parent(colonies, evaluations, rng).traits, settings, rng), settings, rng) for _ in colonies]
    return history
