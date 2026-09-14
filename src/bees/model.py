from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, hypot, log, pi, sin, sqrt, tau
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
    dance_propensity: float = 1.0


@dataclass(frozen=True)
class Worker:
    directional_bias: float
    receiver_attention: float
    sender_transposition: float
    receiver_transposition: float
    search_limit: float
    dance_propensity: float = 1.0


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
    radius: float = 0.0


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
    comb_orientation_axial: bool = False
    vertical_comb_modifier: str = "linear"
    direct_decode: str = "flatten"
    evolve_comb_tilt: bool = True
    # Food-patch geometry. "angular" is the legacy point-site model where a
    # forager captures a site whose bearing falls within ``food_site_width``
    # radians of its heading. "disk" gives each site a physical radius (drawn
    # per site, independent of distance) and captures it when the straight
    # outbound foray path intersects the disk within range.
    food_geometry: str = "angular"
    # Lognormal patch radius (disk geometry only): ``food_site_radius`` is the
    # median (scale) and ``food_site_radius_log_sd`` the log-scale spread; a
    # spread of 0 makes every patch exactly ``food_site_radius``.
    food_site_radius: float = 0.0
    food_site_radius_log_sd: float = 0.0
    # Foray length. "fixed" keeps the legacy behavior where ``search_limit`` is
    # a hard per-worker range cutoff. "gamma" makes each foraging attempt draw
    # its outbound distance from a Gamma with evolved mean ``search_limit`` and
    # fixed shape ``foray_shape`` (clamped to ``max_search_distance``).
    foray_distribution: str = "fixed"
    foray_shape: float = 2.0
    # Patch value scaling. "fixed" gives every site ``food_site_capacity`` loads.
    # "area" makes total patch resource scale with disk area: a site's capacity
    # is ``food_site_capacity * (radius / food_capacity_reference_radius) ** 2``
    # (floored at 1), so larger patches feed proportionally more foragers before
    # depletion while per-visit value stays fixed. Requires disk geometry.
    food_capacity_scaling: str = "fixed"
    food_capacity_reference_radius: float = 0.0
    # Number of food sites per episode. "fixed" places exactly ``food_site_count``
    # sites; "poisson" treats ``food_site_count`` as a mean and draws the count
    # per episode from a Poisson (so some episodes have more, fewer, or no food).
    food_site_count_distribution: str = "fixed"
    # Capacity-conditional recruitment. When True, a successful scout dances with
    # probability 1 - (1 - dance_propensity) ** remaining_capacity, so a patch
    # with nothing left never seeds a dance and richer patches recruit more; the
    # dance_propensity trait then evolves. When False, every successful scout
    # dances unconditionally (legacy behavior) and the trait draws no randomness.
    evolve_dance_propensity: bool = False


@dataclass(frozen=True)
class ColonyEvaluation:
    payoff: float
    success_rate: float
    follower_attempts: int = 0
    follower_successes: int = 0
    matched_searcher_attempts: int = 0
    matched_searcher_successes: int = 0


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
    average_dance_propensity: float
    average_success_rate: float
    average_payoff: float
    follower_success_rate: float
    matched_searcher_success_rate: float
    recruitment_advantage: float
    dance_follow_share: float


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


def direct_world_to_signal(
    direction: float,
    comb_tilt: float,
    comb_orientation: float,
) -> tuple[float, float]:
    """Encode a world food heading as a direct-code in-plane signal angle (and its
    cue strength) by projecting the heading onto the comb surface. This is the
    noise-free forward map inverted by :func:`direct_signal_to_world`.
    """
    basis = _comb_basis(comb_tilt, comb_orientation)
    projected = _project_onto_plane(_world_direction_vector(direction), basis.normal)
    strength = _length(projected)
    if strength <= EPSILON:
        return 0.0, 0.0

    first_component = _dot(projected, basis.first_axis)
    second_component = _dot(projected, basis.second_axis)
    return atan2(second_component, first_component) % tau, strength


def direct_signal_to_world(
    signal: float,
    comb_tilt: float,
    comb_orientation: float,
) -> tuple[float, float]:
    """Recover the world food heading (and its cue strength) from a direct-code
    in-plane signal angle, by inverting the projection used to encode it.

    Encoding maps a world heading ``(cos d, sin d)`` to in-plane coordinates
    ``(a, b) = M (cos d, sin d)``, where the rows of ``M`` are the horizontal
    parts of the two comb-plane axes, and reports the angle ``atan2(b, a)``.
    Recovering ``d`` therefore needs the matrix inverse ``M^-1``. The determinant
    of ``M`` equals ``cos(theta)`` (``theta`` the comb's tilt angle), so the map
    is invertible for any non-vertical comb; a vertical comb leaves the heading
    unrecoverable and yields zero strength.
    """
    basis = _comb_basis(comb_tilt, comb_orientation)
    first = basis.first_axis
    second = basis.second_axis

    determinant = first[0] * second[1] - first[1] * second[0]
    if abs(determinant) <= EPSILON:
        return 0.0, 0.0

    cos_signal = cos(signal)
    sin_signal = sin(signal)
    world_x = (second[1] * cos_signal - first[1] * sin_signal) / determinant
    world_y = (first[0] * sin_signal - second[0] * cos_signal) / determinant

    length = hypot(world_x, world_y)
    if length <= EPSILON:
        return 0.0, 0.0

    # |M^-1 u| is the reciprocal of the projected cue strength at the recovered
    # heading, so the strength is 1 / |M^-1 u|, lying in [cos(theta), 1].
    return atan2(world_y, world_x) % tau, 1.0 / length


def direct_signal_to_world_flatten(
    signal: float,
    comb_tilt: float,
    comb_orientation: float,
) -> tuple[float, float]:
    """Recover a world heading from a direct-code signal by the naive flatten
    (Decode A) rule: drop the vertical component of the in-surface gesture vector
    and read off the resulting horizontal angle.

    This is an uncorrected readout — it does not invert the encoding projection,
    so it produces a systematic angular bias on any tilted comb (zero only along
    the slope and contour axes). See :func:`direct_signal_to_world` for the
    unbiased inverse. The returned strength is the horizontal length of the
    in-surface gesture, lying in [cos(tilt), 1].
    """
    basis = _comb_basis(comb_tilt, comb_orientation)
    first = basis.first_axis
    second = basis.second_axis

    world_x = cos(signal) * first[0] + sin(signal) * second[0]
    world_y = cos(signal) * first[1] + sin(signal) * second[1]

    length = hypot(world_x, world_y)
    if length <= EPSILON:
        return 0.0, 0.0

    return atan2(world_y, world_x) % tau, length


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
                traits.search_limit + _search_limit_jitter(settings, rng),
                0.0,
                settings.max_search_distance,
            ),
            dance_propensity=_clamp(
                traits.dance_propensity + _dance_propensity_jitter(settings, rng),
                0.0,
                1.0,
            ),
        )
        for _ in range(settings.workers_per_colony)
    )

    return Colony(traits=traits, workers=workers)


def _search_limit_jitter(settings: DirectionSettings, rng: Random) -> float:
    """Per-worker deviation of the range trait. Suppressed under gamma forays,
    where within-colony variation comes from the per-foray draw instead."""
    if settings.foray_distribution == "gamma":
        return 0.0
    return rng.gauss(0.0, settings.stable_worker_sd * settings.max_search_distance)


def _dance_propensity_jitter(settings: DirectionSettings, rng: Random) -> float:
    """Per-worker deviation of the recruitment-propensity trait. Draws no
    randomness unless the trait is under selection, so legacy runs are
    unaffected."""
    if not settings.evolve_dance_propensity:
        return 0.0
    return rng.gauss(0.0, settings.stable_worker_sd)


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
        settings,
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
    sites = []
    for _ in range(_sample_site_count(settings, rng)):
        # RNG call order (direction, distance, radius) is fixed for
        # reproducibility; capacity is derived from the drawn radius and draws
        # no randomness.
        direction = rng.random() * tau
        distance = _sample_site_distance(settings, rng)
        radius = _sample_patch_radius(settings, rng)
        sites.append(
            FoodSite(
                direction=direction,
                distance=distance,
                width=settings.food_site_width,
                value=settings.food_value,
                capacity=_site_capacity(settings, radius),
                radius=radius,
            )
        )
    return tuple(sites)


def _sample_site_distance(settings: DirectionSettings, rng: Random) -> float:
    """Draw a site's distance from the colony so that sites are uniformly
    dense per unit *area* over the annulus [min_distance, max_distance], not
    uniformly dense per unit distance. Sampling distance itself uniformly
    concentrates sites near the inner edge, since equal-width distance bins
    near the colony cover less area than equal-width bins farther out."""
    d_min = settings.food_site_min_distance
    d_max = settings.food_site_max_distance
    return sqrt(rng.random() * (d_max * d_max - d_min * d_min) + d_min * d_min)


def _sample_site_count(settings: DirectionSettings, rng: Random) -> int:
    """Number of food sites in an episode. Exactly ``food_site_count`` by
    default; under "poisson" that value is a mean and the count is drawn per
    episode (drawing no randomness in the fixed case keeps legacy runs intact)."""
    if settings.food_site_count_distribution == "fixed":
        return settings.food_site_count
    if settings.food_site_count_distribution != "poisson":
        raise ValueError(
            f"unknown food site count distribution: "
            f"{settings.food_site_count_distribution!r}"
        )
    return _poisson(settings.food_site_count, rng)


def _poisson(mean: float, rng: Random) -> int:
    """Draw a Poisson count with the given mean using Knuth's algorithm."""
    if mean <= 0.0:
        return 0
    threshold = exp(-mean)
    count = 0
    product = 1.0
    while True:
        count += 1
        product *= rng.random()
        if product <= threshold:
            return count - 1


def _site_capacity(settings: DirectionSettings, radius: float) -> int:
    """Number of forager-loads a patch holds. Fixed by default; under "area"
    scaling it grows with disk area so total patch resource scales with size."""
    if settings.food_capacity_scaling == "fixed":
        return settings.food_site_capacity
    if settings.food_capacity_scaling != "area":
        raise ValueError(
            f"unknown food capacity scaling: {settings.food_capacity_scaling!r}"
        )
    reference = settings.food_capacity_reference_radius
    if reference <= 0.0:
        return settings.food_site_capacity
    scaled = settings.food_site_capacity * (radius / reference) ** 2
    return max(1, round(scaled))


def _sample_patch_radius(settings: DirectionSettings, rng: Random) -> float:
    """Draw a physical patch radius, independent of the site's distance. Zero in
    angular geometry (radius is unused there); lognormal in disk geometry."""
    if settings.food_geometry != "disk":
        return 0.0
    median = settings.food_site_radius
    log_sd = settings.food_site_radius_log_sd
    if median <= 0.0 or log_sd <= 0.0:
        return max(0.0, median)
    return rng.lognormvariate(log(median), log_sd)


def find_food_site(
    search_direction: float,
    search_limit: float,
    sites: tuple[FoodSite, ...],
    remaining_capacity: list[int],
    geometry: str = "angular",
) -> int | None:
    if geometry == "disk":
        return _find_food_site_disk(
            search_direction, search_limit, sites, remaining_capacity
        )
    if geometry != "angular":
        raise ValueError(f"unknown food geometry: {geometry!r}")

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


def _find_food_site_disk(
    search_direction: float,
    search_limit: float,
    sites: tuple[FoodSite, ...],
    remaining_capacity: list[int],
) -> int | None:
    """Return the first patch a straight outbound foray enters, ordered by
    ray-entry distance among sites with remaining capacity."""
    reached = []
    for index, site in enumerate(sites):
        if remaining_capacity[index] <= 0:
            continue
        entry = _segment_disk_entry(
            search_direction, search_limit, site.distance, site.direction, site.radius
        )
        if entry is not None:
            reached.append((entry, index))

    if not reached:
        return None

    return min(reached)[1]


def _segment_disk_entry(
    search_direction: float,
    search_limit: float,
    center_distance: float,
    center_bearing: float,
    radius: float,
) -> float | None:
    """Along-ray distance at which a forager leaving the nest along
    ``search_direction`` first enters a disk of ``radius`` centered at polar
    ``(center_distance, center_bearing)``, or ``None`` if the straight foray of
    length ``search_limit`` never intersects the disk.
    """
    offset = center_bearing - search_direction
    along = center_distance * cos(offset)       # projection onto the ray
    perpendicular = center_distance * sin(offset)
    gap = radius * radius - perpendicular * perpendicular
    if gap < 0.0:
        return None                             # ray line misses the disk
    half_chord = sqrt(gap)
    entry = along - half_chord
    exit_point = along + half_chord
    if exit_point < 0.0:
        return None                             # disk lies entirely behind the nest
    if entry > search_limit:
        return None                             # disk beyond the foray length
    return max(0.0, entry)                       # clamp when the nest is inside the disk


def _draw_foray_length(
    mean_length: float,
    settings: DirectionSettings,
    rng: Random,
) -> float:
    """Outbound distance of a single foray. In the legacy "fixed" model this is
    the worker's hard range cutoff; in the "gamma" model it is a per-foray draw
    from a Gamma with the evolved mean ``mean_length`` and fixed shape, clamped
    to ``max_search_distance``."""
    if settings.foray_distribution == "fixed":
        return mean_length
    if settings.foray_distribution != "gamma":
        raise ValueError(
            f"unknown foray distribution: {settings.foray_distribution!r}"
        )
    if mean_length <= 0.0 or settings.foray_shape <= 0.0:
        return 0.0
    length = rng.gammavariate(settings.foray_shape, mean_length / settings.foray_shape)
    return _clamp(length, 0.0, settings.max_search_distance)


def _scout_dances(
    worker: Worker,
    remaining_capacity: int,
    settings: DirectionSettings,
    rng: Random,
) -> bool:
    """Whether a successful scout produces a dance. Unconditional in the legacy
    model; otherwise the scout dances with probability
    ``1 - (1 - dance_propensity) ** remaining_capacity`` -- zero once the patch
    is exhausted, rising with the forage still left for recruits."""
    if not settings.evolve_dance_propensity:
        return True
    if remaining_capacity <= 0:
        return False
    probability = 1.0 - (1.0 - worker.dance_propensity) ** remaining_capacity
    return rng.random() < probability


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
    follower_attempts = 0
    follower_successes = 0
    matched_searcher_attempts = 0
    matched_searcher_successes = 0

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
            dance_available = bool(dances)
            follows_dance = dance_available and rng.random() < worker.receiver_attention

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

            foray_length = _draw_foray_length(worker.search_limit, settings, rng)
            site_index = find_food_site(
                search_direction,
                foray_length,
                sites,
                remaining_capacity,
                settings.food_geometry,
            )
            succeeded = site_index is not None
            if follows_dance:
                follower_attempts += 1
                follower_successes += int(succeeded)
            elif dance_available:
                matched_searcher_attempts += 1
                matched_searcher_successes += int(succeeded)
            if site_index is not None:
                remaining_capacity[site_index] -= 1
                success_count += 1
                food_payoff -= (
                    sites[site_index].distance * settings.travel_cost_per_distance
                )
                food_payoff += sites[site_index].value
                if _scout_dances(
                    worker, remaining_capacity[site_index], settings, rng
                ):
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
                food_payoff -= foray_length * settings.travel_cost_per_distance

        total_successes += success_count
        episode_payoff = (
            food_payoff
            - dance_cost
            - settings.attention_cost * attention_count
        )
        vertical_modifier = 1.0 + (
            settings.vertical_comb_benefit
            * _vertical_comb_modifier_value(colony.traits.comb_tilt, settings)
        )
        total_payoff += episode_payoff * vertical_modifier

    return ColonyEvaluation(
        payoff=max(0.001, total_payoff / settings.episodes_per_colony),
        success_rate=total_successes / total_attempts,
        follower_attempts=follower_attempts,
        follower_successes=follower_successes,
        matched_searcher_attempts=matched_searcher_attempts,
        matched_searcher_successes=matched_searcher_successes,
    )


def _vertical_comb_modifier_value(
    comb_tilt: float,
    settings: DirectionSettings,
) -> float:
    if settings.vertical_comb_modifier == "linear":
        return _clamp(comb_tilt, 0.0, 1.0)

    if settings.vertical_comb_modifier == "threshold_0.8":
        return 1.0 if comb_tilt >= 0.8 else 0.0

    raise ValueError(f"unknown vertical comb modifier: {settings.vertical_comb_modifier}")


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
        history.append(_summarize(generation, colonies, evaluations, settings))

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
        comb_orientation=rng.random() * _comb_orientation_period(settings),
        search_limit=rng.uniform(
            0.15 * settings.max_search_distance,
            0.45 * settings.max_search_distance,
        ),
        dance_propensity=_initial_dance_propensity(settings, rng),
    )


def _initial_dance_propensity(settings: DirectionSettings, rng: Random) -> float:
    """Starting recruitment propensity. Fixed at 1.0 (always dance) when the
    trait is inert; otherwise seeded high with spread so recruitment stays
    bootstrapped and selection can push it down where dancing does not pay."""
    if not settings.evolve_dance_propensity:
        return 1.0
    return rng.uniform(0.8, 1.0)


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
    comb_tilt_change = (
        rng.gauss(0.0, settings.mutation_sd) if settings.evolve_comb_tilt else 0.0
    )
    comb_orientation_change = rng.gauss(
        0.0,
        settings.mutation_sd * _comb_orientation_period(settings),
    )
    search_limit_change = rng.gauss(
        0.0,
        settings.mutation_sd * settings.max_search_distance,
    )
    dance_propensity_change = (
        rng.gauss(0.0, settings.mutation_sd)
        if settings.evolve_dance_propensity
        else 0.0
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
        comb_orientation=(
            traits.comb_orientation + comb_orientation_change
        )
        % _comb_orientation_period(settings),
        search_limit=_clamp(
            traits.search_limit + search_limit_change,
            0.0,
            settings.max_search_distance,
        ),
        dance_propensity=_clamp(
            traits.dance_propensity + dance_propensity_change,
            0.0,
            1.0,
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
    settings: DirectionSettings,
) -> GenerationSummary:
    count = len(colonies)
    average_comb_orientation, orientation_alignment = _orientation_mean_and_alignment(
        [colony.traits.comb_orientation for colony in colonies],
        axial=settings.comb_orientation_axial,
    )

    follower_attempts = sum(
        evaluation.follower_attempts for evaluation in evaluations
    )
    follower_successes = sum(
        evaluation.follower_successes for evaluation in evaluations
    )
    matched_searcher_attempts = sum(
        evaluation.matched_searcher_attempts for evaluation in evaluations
    )
    matched_searcher_successes = sum(
        evaluation.matched_searcher_successes for evaluation in evaluations
    )
    follower_success_rate = _safe_ratio(follower_successes, follower_attempts)
    matched_searcher_success_rate = _safe_ratio(
        matched_searcher_successes,
        matched_searcher_attempts,
    )
    decision_attempts = follower_attempts + matched_searcher_attempts

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
        average_dance_propensity=sum(
            colony.traits.dance_propensity for colony in colonies
        )
        / count,
        average_success_rate=sum(
            evaluation.success_rate for evaluation in evaluations
        )
        / count,
        average_payoff=sum(evaluation.payoff for evaluation in evaluations) / count,
        follower_success_rate=follower_success_rate,
        matched_searcher_success_rate=matched_searcher_success_rate,
        recruitment_advantage=follower_success_rate - matched_searcher_success_rate,
        dance_follow_share=_safe_ratio(follower_attempts, decision_attempts),
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
    settings: DirectionSettings,
    rng: Random,
) -> tuple[float, float]:
    if settings.direct_decode == "unproject":
        angle, strength = direct_signal_to_world(
            signal,
            colony_traits.comb_tilt,
            colony_traits.comb_orientation,
        )
    elif settings.direct_decode == "flatten":
        angle, strength = direct_signal_to_world_flatten(
            signal,
            colony_traits.comb_tilt,
            colony_traits.comb_orientation,
        )
    else:
        raise ValueError(f"unknown direct_decode: {settings.direct_decode!r}")
    if strength <= EPSILON:
        return rng.random() * tau, 0.0
    return _degrade_angle_by_strength(angle, strength, rng), strength


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


def _comb_orientation_period(settings: DirectionSettings) -> float:
    if settings.comb_orientation_axial:
        return pi

    return tau


def _orientation_mean_and_alignment(
    orientations: list[float],
    axial: bool,
) -> tuple[float, float]:
    if not orientations:
        return 0.0, 0.0

    multiplier = 2.0 if axial else 1.0
    period = pi if axial else tau
    x_component = sum(cos(multiplier * orientation) for orientation in orientations)
    y_component = sum(sin(multiplier * orientation) for orientation in orientations)
    alignment = hypot(x_component, y_component) / len(orientations)

    if alignment <= EPSILON:
        return 0.0, 0.0

    mean_orientation = (atan2(y_component, x_component) / multiplier) % period
    if period - mean_orientation <= EPSILON:
        mean_orientation = 0.0

    return mean_orientation, alignment


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator
