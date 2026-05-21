from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import Parameters, Population, simulate


def main() -> None:
    config_path = ROOT / "configs" / "basic.json"
    config = json.loads(config_path.read_text())

    population = Population(
        signalers=config["initial_signalers"],
        nonsignalers=config["initial_nonsignalers"],
    )
    parameters = Parameters(
        signaler_payoff=config["signaler_payoff"],
        nonsignaler_payoff=config["nonsignaler_payoff"],
        communication_cost=config["communication_cost"],
    )

    history = simulate(
        population=population,
        parameters=parameters,
        generations=config["generations"],
        seed=config["seed"],
    )

    for generation, state in enumerate(history):
        print(
            f"{generation:02d} "
            f"signalers={state.signalers:3d} "
            f"nonsignalers={state.nonsignalers:3d} "
            f"signal_frequency={state.signal_frequency:.2f}"
        )


if __name__ == "__main__":
    main()
