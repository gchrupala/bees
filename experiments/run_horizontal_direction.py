from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, simulate


def main() -> None:
    config_path = ROOT / "configs" / "horizontal_direction.json"
    config = json.loads(config_path.read_text())
    seed = config.pop("seed")

    history = simulate(settings=DirectionSettings(**config), seed=seed)

    for state in history:
        print(
            f"{state.generation:02d} "
            f"bias={state.average_directional_bias:.2f} "
            f"attention={state.average_receiver_attention:.2f} "
            f"sender_t={state.average_sender_transposition:.2f} "
            f"receiver_t={state.average_receiver_transposition:.2f} "
            f"search_limit={state.average_search_limit:.2f} "
            f"success={state.average_success_rate:.2f} "
            f"payoff={state.average_payoff:.2f}"
        )


if __name__ == "__main__":
    main()
