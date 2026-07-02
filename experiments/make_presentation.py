"""Generate a short, sparse slide deck summarizing the bee-communication project.

Plain style, serif (Lora) font. Most content lives in the speaker notes; slide
bodies stay minimal. Run with:

    python -u experiments/make_presentation.py

Writes report/presentation.pptx. Figures are read from report/figures/.
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"
FIG_DIR = REPORT_DIR / "figures"
OUT_PATH = REPORT_DIR / "presentation.pptx"

FONT = "Lora"
INK = RGBColor(0x1A, 0x1A, 0x1A)       # near-black body text
GREY = RGBColor(0x55, 0x55, 0x55)      # secondary text

SW, SH = Inches(13.333), Inches(7.5)   # 16:9


def _set(run, size, color, bold=False):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = FONT


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def blank_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    return slide


def textbox(slide, left, top, width, height, lines, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, space_after=10):
    """lines: list of (text, size, color, bold) tuples; each is a paragraph."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, spec in enumerate(lines):
        text, size, color, bold = spec
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = text
        _set(r, size, color, bold)
    return tb


def title_only(slide, title):
    textbox(slide, Inches(0.9), Inches(0.7), Inches(11.5), Inches(1.0),
            [(title, 30, INK, True)])


def fitted_image(slide, path, box_top, box_h, box_w=Inches(11.6)):
    """Place an image scaled to fit within (box_w, box_h), centered."""
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(box_w / iw, box_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(str(path), int((SW - w) / 2), box_top,
                                    width=w, height=h)


def build():
    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH

    # ---- Slide 1: Title ----------------------------------------------------
    s = blank_slide(prs)
    textbox(s, Inches(0.9), Inches(2.7), Inches(11.5), Inches(2.0), [
        ("The evolution of a referential code", 38, INK, True),
        ("in populations of bee-like agents", 38, INK, True),
    ], space_after=4)
    textbox(s, Inches(0.95), Inches(4.5), Inches(11.5), Inches(0.6),
            [("From an iconic pointing dance to an abstract gravity-referenced code",
              19, GREY, False)])
    textbox(s, Inches(0.95), Inches(5.5), Inches(11.5), Inches(0.5),
            [("Grzegorz Chrupała", 16, INK, False)])
    add_notes(s, """
A computational model of how honeybee communication evolves. A shared code needs
sender-receiver alignment, so any change to it poses a coordination problem. The
waggle dance is a rare real-world case of a referential code shifting from an
iconic to a more abstract form.

Two questions drive the work: when is a direct-pointing dance worth maintaining,
and under what conditions can a population transition to the gravity-referenced
version without communication breaking down? The model is deliberately minimal
and reproducible from seeds.
""")

    # ---- Slide 2: The biological puzzle -----------------------------------
    s = blank_slide(prs)
    title_only(s, "Two ways to point to food")
    fitted_image(s, FIG_DIR / "waggle.png", Inches(1.95), Inches(5.2), Inches(9.8))
    add_notes(s, """
The waggle dance encodes food direction, in two variants. On horizontal, exposed
combs (Apis florea, dorsata) the waggle points directly at the food — iconic, the
signal's form mirrors the referent. On vertical combs direct pointing is
impossible, so bees use gravity as reference, mapping the food's sun-relative
azimuth onto the angle from vertical — abstract.

Phylogeny suggests horizontal + direct pointing is ancestral and vertical +
gravity is derived, so a transition happened. It is hard because the code is
shared: a change on one side not matched by the other disrupts communication.
That coordination problem is the heart of the project.
""")

    # ---- Slide 3: The model -----------------------------------------------
    s = blank_slide(prs)
    title_only(s, "A minimal evolutionary model")
    bullets = [
        "Colonies reproduce in proportion to foraging payoff",
        "Workers are noisy samples of heritable colony traits",
        "Dances blend a direct and a gravity-referenced code",
        "Senders and receivers must stay aligned",
        "An extrinsic benefit rewards tilting the comb vertical",
    ]
    lines = [(b, 22, INK, False) for b in bullets]
    textbox(s, Inches(1.0), Inches(2.4), Inches(11.3), Inches(4.2), lines,
            space_after=16)
    add_notes(s, """
Selection acts at the colony level; traits are inherited with Gaussian mutation,
and workers express them with individual noise. Key traits: directional bias
(dance precision), receiver attention, and sender/receiver transposition (weight
on the gravity vs. direct code), plus comb tilt and orientation.

As the comb tilts, the direct signal degrades and the gravity reference becomes
available, so maintaining success requires moving both senders and receivers to
the gravity code together — the coordination problem made concrete. Their
mutations can be coupled. A separate extrinsic benefit rewards vertical combs and
drives the tilt up. Everything is a parameter.
""")

    # ---- Slide 4: Finding 1 -----------------------------------------------
    s = blank_slide(prs)
    title_only(s, "The basic iconic code evolves readily")
    fitted_image(s, FIG_DIR / "food_distribution_recruitment_advantage.png",
                 Inches(2.1), Inches(4.7), Inches(11.0))
    add_notes(s, """
On a flat comb, only direct pointing is in play. We measure the in-run
recruitment advantage: dance-followers' success minus that of matched random
searchers in the same episodes. Because following is randomized, this cleanly
captures what the dance buys and does not mistake neutral drift for real
communication.

Communication is most valuable when food is directionally concentrated but hard
to find by chance, and is suppressed at both extremes. Adding sites makes
independent discovery easy and erodes the dance's value (left); wider patches
tolerate noisy decoding, so advantage rises (right). The basic code evolves
readily given a favorable resource distribution — the easy half.
""")

    # ---- Slide 5: Finding 2 -----------------------------------------------
    s = blank_slide(prs)
    title_only(s, "The transition to the abstract code is conditional")
    fitted_image(s, FIG_DIR / "evolutionary_interaction_stable_heatmap.png",
                 Inches(2.1), Inches(4.7), Inches(11.4))
    add_notes(s, """
Now let comb tilt evolve under the vertical-comb benefit. Horizontal-start
colonies can evolve both a vertical comb and a coordinated gravity code — the
best validated parameters are stable in 99 of 100 held-out seeds — but the
corridor is narrow.

Three factors govern it: the vertical-comb benefit, the mutation scale, and
sender-receiver coupling. At weak benefit (left panel) the transition is nearly
absent and cannot be rescued by mutation or coupling — a genuine barrier, not a
slow approach (it stays closed even over 960 generations). Too few food sites or
too small a mutation scale leaves the population in a productive but flat
direct-pointing code.
""")

    # ---- Slide 6: Conclusion ----------------------------------------------
    s = blank_slide(prs)
    title_only(s, "A reproducible, parameter-dependent corridor")
    concl = [
        "Iconic direct pointing evolves easily",
        "The gravity-code transition needs a narrow confluence",
        "Robust under both decode geometries",
        "Next: make the benefit and ecology more biological",
    ]
    lines = [(c, 22, INK, False) for c in concl]
    textbox(s, Inches(1.0), Inches(2.5), Inches(11.3), Inches(4.0), lines,
            space_after=18)
    add_notes(s, """
Mirroring the biology: the iconic direct-pointing code evolves readily when the
ecology makes food moderately hard to find, while the shift to the abstract
gravity code happens only under a rare confluence — substantial vertical-comb
benefit, enough mutation, and coupling that moves senders and receivers together.

The result is not an artifact of one decode choice: it holds under both the
approximate 'flatten' and the exact 'unproject' geometry (best unproject
candidate 96/100 seeds). The claim is modest — a reproducible but conditional
corridor. Next, replace the abstract benefit and ecology with explicit biological
constraints and test whether it survives.
""")

    prs.save(OUT_PATH)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    build()
