---
name: no-latex-in-chat
description: User's terminal chat can't render LaTeX math; use plain Unicode in replies
metadata:
  type: feedback
---

In chat replies, do not use LaTeX math delimiters (`$...$`, `\mathrm{}`, etc.) —
the Claude Code terminal/IDE renders messages as markdown with no math support,
so LaTeX shows as raw source. Use plain Unicode instead: θ, π, γ, φ, δ, and
underscore subscripts like s_dir, s_grav, δ_dir, w_dir, t_s, f_d, n, g.

**Why:** Unrendered LaTeX is hard to read in the user's environment.

**How to apply:** Only affects chat prose. LaTeX inside the paper
(`report/paper.tex`) stays correct LaTeX for the compiled PDF — do not strip it
there.
