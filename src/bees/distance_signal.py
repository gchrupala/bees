from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random


@dataclass(frozen=True)
class DistanceTraits:
    cue_weight: float
    receiver_scale: float
    attention: float


@dataclass(frozen=True)
class DistanceSettings:
    lineages: int
    generations: int
    episodes_per_lineage: int
    mutation_sd: float
    max_distance: float
    cue_noise: float
    search_width: float
    random_success: float
    cue_cost: float
    attention_cost: float


@dataclass(frozen=True)
class DistanceGeneration:
    generation: int
    average_cue_weight: float
    average_receiver_scale: float
    average_attention: float
    average_payoff: float


def distance_success(error: float, search_width: float) -> float:
    if search_width <= 0:
        raise ValueError("search_width must be positive")

    return exp(-0.5 * (error / search_width) ** 2)


def evaluate_lineage(
    traits: DistanceTraits,
    settings: DistanceSettings,
    rng: Random,
) -> float:
    total_success = 0.0

    for _ in range(settings.episodes_per_lineage):
        food_distance = rng.random() * settings.max_distance
        cue = traits.cue_weight * food_distance + rng.gauss(0.0, settings.cue_noise)
        estimated_distance = max(0.0, traits.receiver_scale * cue)
        error = abs(food_distance - estimated_distance)
        signaled_success = distance_success(error, settings.search_width)

        total_success += (
            traits.attention * signaled_success
            + (1.0 - traits.attention) * settings.random_success
        )

    average_success = total_success / settings.episodes_per_lineage
    cost = (
        settings.cue_cost * traits.cue_weight**2
        + settings.attention_cost * traits.attention
    )

    return max(0.001, average_success - cost)


def simulate_distance_signal(
    settings: DistanceSettings,
    seed: int,
) -> list[DistanceGeneration]:
    rng = Random(seed)
    lineages = _initial_lineages(settings.lineages, rng)
    history = []

    for generation in range(settings.generations + 1):
        payoffs = [evaluate_lineage(lineage, settings, rng) for lineage in lineages]
        history.append(_summarize(generation, lineages, payoffs))

        if generation < settings.generations:
            lineages = [
                _mutate(_choose_parent(lineages, payoffs, rng), settings, rng)
                for _ in lineages
            ]

    return history


def _initial_lineages(count: int, rng: Random) -> list[DistanceTraits]:
    return [
        DistanceTraits(
            cue_weight=rng.uniform(0.0, 0.12),
            receiver_scale=rng.uniform(0.0, 2.0),
            attention=rng.uniform(0.0, 0.25),
        )
        for _ in range(count)
    ]


def _choose_parent(
    lineages: list[DistanceTraits],
    payoffs: list[float],
    rng: Random,
) -> DistanceTraits:
    total_payoff = sum(payoffs)
    threshold = rng.random() * total_payoff
    cumulative = 0.0

    for lineage, payoff in zip(lineages, payoffs):
        cumulative += payoff
        if cumulative >= threshold:
            return lineage

    return lineages[-1]


def _mutate(
    traits: DistanceTraits,
    settings: DistanceSettings,
    rng: Random,
) -> DistanceTraits:
    return DistanceTraits(
        cue_weight=_clamp(
            traits.cue_weight + rng.gauss(0.0, settings.mutation_sd),
            0.0,
            2.0,
        ),
        receiver_scale=_clamp(
            traits.receiver_scale + rng.gauss(0.0, settings.mutation_sd),
            0.0,
            4.0,
        ),
        attention=_clamp(
            traits.attention + rng.gauss(0.0, settings.mutation_sd),
            0.0,
            1.0,
        ),
    )


def _summarize(
    generation: int,
    lineages: list[DistanceTraits],
    payoffs: list[float],
) -> DistanceGeneration:
    count = len(lineages)

    return DistanceGeneration(
        generation=generation,
        average_cue_weight=sum(lineage.cue_weight for lineage in lineages) / count,
        average_receiver_scale=sum(lineage.receiver_scale for lineage in lineages)
        / count,
        average_attention=sum(lineage.attention for lineage in lineages) / count,
        average_payoff=sum(payoffs) / count,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
