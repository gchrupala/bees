from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bees.model import DirectionSettings, GenerationSummary, simulate

WIDTH = 760
HEIGHT = 420
MARGIN_LEFT = 64
MARGIN_RIGHT = 28
MARGIN_TOP = 42
MARGIN_BOTTOM = 58


def main() -> None:
    config_path = ROOT / "configs" / "horizontal_direction.json"
    config = json.loads(config_path.read_text())
    seed = config.pop("seed")

    history = simulate(settings=DirectionSettings(**config), seed=seed)
    output_path = ROOT / "results" / "horizontal_direction_demo.svg"
    output_path.write_text(render_svg(history), encoding="utf-8")

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
        "sender_transposition: "
        f"{first.average_sender_transposition:.2f} -> {last.average_sender_transposition:.2f}"
    )
    print(
        "receiver_transposition: "
        f"{first.average_receiver_transposition:.2f} -> {last.average_receiver_transposition:.2f}"
    )
    print(
        "success_rate: "
        f"{first.average_success_rate:.2f} -> {last.average_success_rate:.2f}"
    )


def render_svg(history: list[GenerationSummary]) -> str:
    lines = [
        _polyline(
            history,
            "average_directional_bias",
            "#2f6f9f",
        ),
        _polyline(
            history,
            "average_receiver_attention",
            "#b44b5a",
        ),
        _polyline(
            history,
            "average_success_rate",
            "#5d8a3d",
        ),
    ]

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{MARGIN_LEFT}" y="24" font-family="sans-serif" font-size="18" fill="#222222">Horizontal-comb direction signal evolution</text>
  {_axis()}
  {''.join(lines)}
  {_legend()}
</svg>
"""


def _axis() -> str:
    left = MARGIN_LEFT
    right = WIDTH - MARGIN_RIGHT
    top = MARGIN_TOP
    bottom = HEIGHT - MARGIN_BOTTOM
    parts = [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333333"/>',
    ]

    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = _y(value)
        label = f"{value:.2f}".rstrip("0").rstrip(".")
        parts.extend(
            [
                f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333333"/>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#eeeeee"/>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#555555">{label}</text>',
            ]
        )

    for value in (0, 10, 20, 30):
        x = _x(value, 30)
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 5}" stroke="#333333"/>',
                f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555555">{value}</text>',
            ]
        )

    parts.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{HEIGHT - 14}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#333333">generation</text>',
            f'<text x="18" y="{(top + bottom) / 2:.1f}" transform="rotate(-90 18 {(top + bottom) / 2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#333333">mean value</text>',
        ]
    )
    return "\n  ".join(parts)


def _polyline(
    history: list[GenerationSummary],
    field: str,
    color: str,
) -> str:
    max_generation = history[-1].generation
    points = " ".join(
        f"{_x(state.generation, max_generation):.1f},{_y(getattr(state, field)):.1f}"
        for state in history
    )
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="3" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
    )


def _legend() -> str:
    entries = [
        ("#2f6f9f", "directional bias"),
        ("#b44b5a", "receiver attention"),
        ("#5d8a3d", "search success"),
    ]
    x = WIDTH - 218
    y = 54
    parts = [
        f'<rect x="{x - 12}" y="{y - 22}" width="206" height="86" fill="#ffffff" stroke="#dddddd"/>'
    ]

    for index, (color, label) in enumerate(entries):
        row_y = y + index * 24
        parts.extend(
            [
                f'<line x1="{x}" y1="{row_y}" x2="{x + 28}" y2="{row_y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
                f'<text x="{x + 38}" y="{row_y + 4}" font-family="sans-serif" font-size="13" fill="#333333">{label}</text>',
            ]
        )

    return "\n  ".join(parts)


def _x(generation: int, max_generation: int) -> float:
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    return MARGIN_LEFT + plot_width * generation / max_generation


def _y(value: float) -> float:
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    return MARGIN_TOP + plot_height * (1.0 - value)


if __name__ == "__main__":
    main()
