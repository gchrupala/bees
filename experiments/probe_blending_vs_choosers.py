"""Compare blended codes against binary code-choosers as a static landscape probe.

The model lets each worker weight the direct and gravity-referenced codes
continuously (the sender/receiver transposition traits). An obvious alternative
is a binary trait: each worker either points directly or refers to gravity, with
no blending. This probe asks whether the two setups are interchangeable in
aggregate, by measuring communication accuracy directly rather than by running
evolution.

For each comb tilt we sweep a single mixing parameter x and compare:

  blender  every worker blends both codes with sender = receiver transposition x
  chooser  a fraction x of workers use the gravity code purely and the rest the
           direct code purely, with dancer and follower drawn independently

Success is a hard threshold on angular error, mirroring the angular food
geometry: a follower succeeds when the direction it recovers falls within
SUCCESS_TOLERANCE of the true food direction. Dances are noiseless (the von
Mises and Gaussian noise terms are switched off) so that the comparison isolates
the coding geometry from signal noise.

Writes two tidy CSVs to results/ and streams rows as they are computed.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import (
    ColonyTraits,
    DirectionSettings,
    Worker,
    encode_dance_direction,
    interpret_signal,
)

SEED = 20260717
TRIALS = 20000
SUCCESS_TOLERANCE = math.radians(15.0)
COMB_TILTS = (0.0, 0.25, 0.5, 0.75, 1.0)
COMB_ORIENTATION = 0.7
MIXING_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
PURE_DIRECT = 0.0
PURE_GRAVITY = 1.0
DIRECT_DECODE = "unproject"


def make_settings(direct_decode: str) -> DirectionSettings:
    """Settings for a noiseless single-dance probe.

    Only the noise terms and the decode variant matter here: the population,
    ecology, and cost fields are never read, because the probe calls the
    encoding and decoding functions directly rather than running a simulation.
    """
    return DirectionSettings(
        colony_count=0,
        workers_per_colony=0,
        generations=0,
        episodes_per_colony=0,
        foraging_attempts_per_episode=0,
        mutation_sd=0.0,
        transposition_mutation_correlation=0.0,
        stable_worker_sd=0.0,
        max_signal_concentration=0.0,
        dance_noise_sd=0.0,
        interpretation_noise_sd=0.0,
        initial_comb_tilt=0.0,
        vertical_comb_benefit=0.0,
        sun_azimuth_center=0.0,
        sun_azimuth_width=0.0,
        food_site_count=0,
        food_site_width=0.0,
        food_site_min_distance=0.0,
        food_site_max_distance=0.0,
        max_search_distance=0.0,
        food_site_capacity=0,
        food_value=0.0,
        travel_cost_per_distance=0.0,
        base_dance_cost=0.0,
        cue_cost=0.0,
        attention_cost=0.0,
        direct_decode=direct_decode,
    )


def make_worker(transposition: float) -> Worker:
    return Worker(
        directional_bias=1.0,
        receiver_attention=1.0,
        sender_transposition=transposition,
        receiver_transposition=transposition,
        search_limit=1.0,
    )


def make_traits(comb_tilt: float) -> ColonyTraits:
    return ColonyTraits(
        directional_bias=1.0,
        receiver_attention=1.0,
        sender_transposition=0.0,
        receiver_transposition=0.0,
        search_limit=1.0,
        comb_tilt=comb_tilt,
        comb_orientation=COMB_ORIENTATION,
    )


def angular_error(recovered: float, true_direction: float) -> float:
    difference = (recovered - true_direction + math.pi) % math.tau - math.pi
    return abs(difference)


def success_rate(
    sender_transposition: float,
    receiver_transposition: float,
    traits: ColonyTraits,
    settings: DirectionSettings,
    rng: Random,
    trials: int = TRIALS,
) -> float:
    """Fraction of dancer-follower pairs whose follower lands within tolerance."""
    successes = 0
    for _ in range(trials):
        food_direction = rng.random() * math.tau
        sun_azimuth = rng.random() * math.tau
        # Bypass produce_signal: the dance angle is the sender's intended one,
        # so no von Mises spread is added on top of the coding geometry.
        dance = encode_dance_direction(
            food_direction,
            make_worker(sender_transposition),
            traits,
            settings,
            sun_azimuth,
            rng,
        )
        recovered = interpret_signal(
            dance,
            make_worker(receiver_transposition),
            traits,
            settings,
            sun_azimuth,
            rng,
        )
        successes += angular_error(recovered, food_direction) < SUCCESS_TOLERANCE
    return successes / trials


def chooser_success_rate(
    gravity_frequency: float,
    traits: ColonyTraits,
    settings: DirectionSettings,
    rng: Random,
    trials: int = TRIALS,
) -> float:
    """As above, but sender and receiver each pick a pure code independently."""
    successes = 0
    for _ in range(trials):
        food_direction = rng.random() * math.tau
        sun_azimuth = rng.random() * math.tau
        sender = PURE_GRAVITY if rng.random() < gravity_frequency else PURE_DIRECT
        receiver = PURE_GRAVITY if rng.random() < gravity_frequency else PURE_DIRECT
        dance = encode_dance_direction(
            food_direction,
            make_worker(sender),
            traits,
            settings,
            sun_azimuth,
            rng,
        )
        recovered = interpret_signal(
            dance,
            make_worker(receiver),
            traits,
            settings,
            sun_azimuth,
            rng,
        )
        successes += angular_error(recovered, food_direction) < SUCCESS_TOLERANCE
    return successes / trials


def write_sweep(path: Path, settings: DirectionSettings) -> None:
    rng = Random(SEED)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comb_tilt", "mixing", "population", "success_rate"])
        for comb_tilt in COMB_TILTS:
            traits = make_traits(comb_tilt)
            for mixing in MIXING_VALUES:
                blender = success_rate(mixing, mixing, traits, settings, rng)
                chooser = chooser_success_rate(mixing, traits, settings, rng)
                for population, rate in (("blender", blender), ("chooser", chooser)):
                    writer.writerow([comb_tilt, mixing, population, f"{rate:.4f}"])
                handle.flush()
                print(
                    f"tilt={comb_tilt:.2f} x={mixing:.2f} "
                    f"blender={blender:.3f} chooser={chooser:.3f}",
                    flush=True,
                )


def write_pair_types(path: Path, settings: DirectionSettings) -> None:
    rng = Random(SEED + 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["comb_tilt", "sender_code", "receiver_code", "matched", "success_rate"]
        )
        for comb_tilt in COMB_TILTS:
            traits = make_traits(comb_tilt)
            for sender in (PURE_DIRECT, PURE_GRAVITY):
                for receiver in (PURE_DIRECT, PURE_GRAVITY):
                    rate = success_rate(sender, receiver, traits, settings, rng)
                    writer.writerow(
                        [
                            comb_tilt,
                            "gravity" if sender == PURE_GRAVITY else "direct",
                            "gravity" if receiver == PURE_GRAVITY else "direct",
                            sender == receiver,
                            f"{rate:.4f}",
                        ]
                    )
                    handle.flush()
                    print(
                        f"tilt={comb_tilt:.2f} sender={sender:.0f} "
                        f"receiver={receiver:.0f} success={rate:.3f}",
                        flush=True,
                    )


def main() -> None:
    settings = make_settings(DIRECT_DECODE)
    chance = 2 * SUCCESS_TOLERANCE / math.tau
    print(
        f"seed={SEED} trials={TRIALS} decode={DIRECT_DECODE} "
        f"tolerance={math.degrees(SUCCESS_TOLERANCE):.0f}deg chance={chance:.3f}",
        flush=True,
    )
    results = ROOT / "results"
    write_sweep(results / "blending_vs_choosers_sweep.csv", settings)
    write_pair_types(results / "blending_vs_choosers_pair_types.csv", settings)
    print("Wrote results/blending_vs_choosers_{sweep,pair_types}.csv", flush=True)


if __name__ == "__main__":
    main()
