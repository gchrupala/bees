from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.distance_signal import DistanceSettings, simulate_distance_signal


def main() -> None:
    config_path = ROOT / "configs" / "distance_signal.json"
    config = json.loads(config_path.read_text())
    seed = config.pop("seed")

    history = simulate_distance_signal(
        settings=DistanceSettings(**config),
        seed=seed,
    )

    for state in history:
        print(
            f"{state.generation:02d} "
            f"cue_weight={state.average_cue_weight:.2f} "
            f"receiver_scale={state.average_receiver_scale:.2f} "
            f"attention={state.average_attention:.2f} "
            f"payoff={state.average_payoff:.2f}"
        )


if __name__ == "__main__":
    main()
