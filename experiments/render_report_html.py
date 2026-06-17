from __future__ import annotations

import csv
import html
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
RESULTS = ROOT / "results"


def main() -> None:
    source = REPORT / "report.md"
    output = REPORT / "report.html"
    css = REPORT / "report.css"
    expanded = expand_directives(source.read_text())
    render_with_pandoc(expanded, output, css)
    print(f"wrote {relative(output)}", flush=True)


def expand_directives(markdown: str) -> str:
    return markdown.replace("{{ long_transition_heatmap }}", long_transition_heatmap())


def long_transition_heatmap() -> str:
    rows = read_csv(RESULTS / "long_vertical_transition_summary.csv")
    modifiers = sorted(
        {row["vertical_comb_modifier"] for row in rows},
        key=modifier_sort_key,
    )
    blocks = [
        '<figure id="fig:long-transition-heatmap" class="figure heatmap-figure">',
        "<figcaption>",
        (
            "Long axial-orientation transition experiment. Rows are initial comb "
            "tilt, columns are proportional vertical-comb advantage alpha, and "
            "modifier headings give f(t). Count panels report seeds out of ten; "
            "success panels report final-generation means. Source: "
            "<code>results/long_vertical_transition_summary.csv</code>."
        ),
        "</figcaption>",
    ]
    for modifier in modifiers:
        modifier_rows = [
            row for row in rows if row["vertical_comb_modifier"] == modifier
        ]
        blocks.append(
            f'<section class="heatmap-modifier" aria-label="{html_attr(modifier)}">'
        )
        blocks.append(f"<h4>{html.escape(modifier_label(modifier))}</h4>")
        blocks.append('<div class="heatmap-grid">')
        for title, field, kind, color in (
            ("Gravity", "reached_gravity_fraction", "count", "#3b82f6"),
            ("Vertical", "retained_vertical_fraction", "count", "#14b8a6"),
            ("Collapse", "collapse_fraction", "count", "#ef4444"),
            (
                "Recovery",
                "recovered_from_collapse_fraction",
                "count",
                "#8b5cf6",
            ),
            ("Success", "mean_final_success", "unit", "#22c55e"),
        ):
            blocks.append(
                heatmap_panel(
                    title=title,
                    field=field,
                    kind=kind,
                    color=color,
                    rows=modifier_rows,
                )
            )
        blocks.append("</div>")
        blocks.append("</section>")
    blocks.append("</figure>")
    return "\n".join(blocks)


def heatmap_panel(
    title: str,
    field: str,
    kind: str,
    color: str,
    rows: list[dict[str, str]],
) -> str:
    initial_tilts = sorted(
        {float(row["initial_comb_tilt"]) for row in rows},
        reverse=True,
    )
    benefits = sorted({float(row["vertical_comb_benefit"]) for row in rows})
    by_condition = {
        (float(row["initial_comb_tilt"]), float(row["vertical_comb_benefit"])): row
        for row in rows
    }
    parts = [
        '<table class="heatmap">',
        f"<caption>{html.escape(title)}</caption>",
        "<thead><tr><th>t0 / alpha</th>",
    ]
    parts.extend(f"<th>{benefit:.2f}</th>" for benefit in benefits)
    parts.append("</tr></thead><tbody>")
    for tilt in initial_tilts:
        parts.append(f"<tr><th>{tilt:.1f}</th>")
        for benefit in benefits:
            row = by_condition[(tilt, benefit)]
            text, fraction = cell_value(row, field, kind)
            parts.append(
                (
                    '<td style="background-color: '
                    f'{blend(color, fraction)}" title="{html_attr(title)}: '
                    f'{html_attr(text)}">{html.escape(text)}</td>'
                )
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def cell_value(row: dict[str, str], field: str, kind: str) -> tuple[str, float]:
    if kind == "count":
        fraction = float(row[field])
        seeds = int(row["seeds"])
        count = round(fraction * seeds)
        return f"{count}/{seeds}", fraction

    if kind == "unit":
        value = float(row[field])
        return f"{value:.2f}", min(value / 0.2, 1.0)

    raise ValueError(f"unknown heatmap kind: {kind}")


def blend(hex_color: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    red, green, blue = (
        int(hex_color[index : index + 2], 16)
        for index in (1, 3, 5)
    )
    mixed = [
        round(255 - (255 - channel) * fraction)
        for channel in (red, green, blue)
    ]
    return f"rgb({mixed[0]}, {mixed[1]}, {mixed[2]})"


def modifier_sort_key(modifier: str) -> tuple[int, str]:
    order = {"linear": 0, "threshold_0.8": 1}
    return (order.get(modifier, len(order)), modifier)


def modifier_label(modifier: str) -> str:
    if modifier == "linear":
        return "Modifier: f(t)=t"
    if modifier == "threshold_0.8":
        return "Modifier: f(t)=1[t >= 0.8]"
    return f"Modifier: {modifier}"


def render_with_pandoc(markdown: str, output: Path, css: Path) -> None:
    pandoc = find_pandoc()
    if pandoc is None:
        raise SystemExit(
            "pandoc is required to render the HTML report. "
            "Install pandoc or pypandoc-binary, or edit report/report.md without "
            "rebuilding."
        )

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".md",
        delete=False,
    ) as handle:
        handle.write(markdown)
        expanded_source = Path(handle.name)

    try:
        subprocess.run(
            [
                pandoc,
                "--standalone",
                "--toc",
                "--toc-depth=2",
                "--from",
                (
                    "markdown+citations+fenced_divs+markdown_in_html_blocks"
                    "+pipe_tables+raw_html+tex_math_dollars"
                ),
                "--to",
                "html5",
                "--mathml",
                "--citeproc",
                "--bibliography",
                "references.bib",
                "--css",
                css.name,
                "--output",
                output.name,
                str(expanded_source),
            ],
            cwd=REPORT,
            check=True,
        )
    finally:
        expanded_source.unlink(missing_ok=True)


def find_pandoc() -> str | None:
    pandoc = shutil.which("pandoc")
    if pandoc is not None:
        return pandoc

    try:
        import pypandoc
    except ImportError:
        return None

    try:
        return pypandoc.get_pandoc_path()
    except OSError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def html_attr(value: str) -> str:
    return html.escape(value, quote=True)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
