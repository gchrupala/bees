---
name: direct-code-tilt-bias
description: Model's direct-code decode was biased on a tilt (used M^T not M^-1); fixed to the proper inverse (commit 79fb4cb). See [[direct-decode-two-ways]].
metadata:
  type: project
---

The direct (iconic) code shows a **directional bias** on a tilted comb in the
current model, but this is a **decode modeling choice, not inherent**.

Reduce to the horizontal 2x2 map. Since e1, e2 lie in the comb plane, the
in-plane coords of the projected food equal f.e1, f.e2, so
`[a;b] = M [cos d; sin d]` with `M = [[e1x,e1y],[e2x,e2y]]` and
`delta_dir = atan2(b,a)`.

- The model's decode rebuilds `v = cos(delta) e1 + sin(delta) e2` and reads its
  horizontal heading = `M^T u` (u = unit vector at delta). The correct inverse is
  `M^-1 u`. M is non-orthogonal under tilt (here `M M^T = [[.70,-.245],[-.245,.80]]`),
  so M^T != M^-1 and the result is biased (up to ~18 deg at theta=pi/4).
- The proper inverse `M^-1 u` recovers d **exactly** for all d (verified). The
  encode d -> delta_dir is a bijection for theta < pi/2 (projected circle is a
  non-degenerate ellipse), so it is invertible; the receiver shares the comb and
  thus knows the tilt, so M^-1 is implementable.
- Genuinely inherent: only (1) reliability loss — foreshortened directions are
  noise-sensitive, which is what s_dir should measure; and (2) true
  non-invertibility at theta -> pi/2 where the ellipse collapses (s_dir -> 0).

The gravity code is symbolic (food azimuth relative to the sun), so with a shared
sun it decodes back to d exactly at any tilt. Round-trip probe (matched t_s=t_r;
theta=pi/4, phi=pi/6, food d=57.3deg) recovered world dir: pure direct 75.9
(18.6deg err), t=0.5 64.8 (7.5), pure gravity 57.3 (0.0 exact).

**Resolved:** chose (B) — fixed the decode to M^-1 in `model.py` (commit 79fb4cb),
adding noise-free `direct_world_to_signal` / `direct_signal_to_world` helpers and
intermediate-tilt round-trip tests. Pre-fix state tagged `v2`; the fixed model is
"v3" but **v3 is not tagged yet** (user asked to hold). The naive-vs-idealized
decode choice and its biological reading are detailed in [[direct-decode-two-ways]].

**Why:** Determines whether "direct is unreliable on a tilt" is a modeling
artifact or a defended property; affects model.py, results, and paper text.

**How to apply:** Judge communication accuracy on *decoded world directions* vs
the true food bearing, not comb-surface angles. The current blend figure
([[comb-codes-figure]]) misleads by placing the food marker at delta_dir (the
biased projected bearing). Verify with `src/bees/model.py`
(`_direct_signal_angle`, `_direct_world_direction_from_signal`,
`_gravity_signal_angle`, `_gravity_world_direction_from_signal`).
