from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.spatial_model import SpatialSettings, GenerationSummary, simulate_spatial


def main() -> None:
    config_path = ROOT / "configs" / "spatial_direction.json"
    config = json.loads(config_path.read_text())
    seed = config.pop("seed")

    history = simulate_spatial(settings=SpatialSettings(**config), seed=seed)
    output_path = ROOT / "results" / "spatial_direction_demo.json"
    output_path.write_text(json.dumps([s.__dict__ for s in history], default=str), encoding="utf-8")

    first = history[0]
    last = history[-1]
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(
        "directional_bias: "
        f"{first.average_directional_bias:.2f} -> {last.average_directional_bias:.2f}"
    )
    print(
        "receiver_attention: "
        f"{first.average_receiver_attention:.2f} -> {last.average_receiver_attention:.2f}"
    )
    print(
        "success_rate: "
        f"{first.average_success_rate:.2f} -> {last.average_success_rate:.2f}"
    )


if __name__ == "__main__":
    main()
