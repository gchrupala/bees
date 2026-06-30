---
name: direct-decode-two-ways
description: Crystal-clear account of the two ways to decode the direct dance (naive "flatten" vs exact "un-project"), with the biological caveat and open question
metadata:
  type: project
---

Two ways to decode the direct (iconic) dance on a tilted comb. The fix
([[direct-code-tilt-bias]]) switched from the first to the second.

**Setup.** The waggle run's *direction* is a 2-D angle lying IN the comb surface.
On a tilted comb that surface is a sloped plane, so an arrow drawn in it is a 3-D
arrow that rises up out of horizontal.

**Encoding (what the dancer does; both decodes assume this same step).** The food
is at a horizontal compass bearing d. The dancer takes the horizontal "to the
food" arrow and presses it flat onto the sloped comb — an orthogonal projection
straight onto the plane. This foreshortens the arrow along the steepest-slope
(downhill) direction, so the angle of the arrow *as it lies in the surface* is
NOT equal to d. That in-surface angle is the dance signal.

Given the dance arrow in the surface, there are two different ways to turn it back
into a compass bearing:

**Decode A — "flatten the gesture" (the OLD, biased decode).** The dance arrow
lies in the sloped surface, so it tilts up out of horizontal. Take its
straight-down shadow on the ground (project the in-surface arrow vertically down
onto the horizontal plane) and fly that way. Cheap and reference-free: the
follower just goes where the gesture points and lets gravity flatten it; it needs
to know nothing about how the comb is tilted or oriented. BUT it is biased:
encoding pressed a horizontal arrow UP onto the slope, and this decode presses the
result back DOWN to horizontal. Up-then-down is not a round trip — each pressing
shortens the across-slope component, so doing both shortens it twice, rotating the
recovered bearing toward the slope's tilt axis. At a 45° tilt: up to ~18° off.

**Decode B — "un-project" (the NEW, exact decode).** Instead of flattening, the
follower reasons: "Which horizontal bearing, if pressed flat onto THIS particular
sloped comb, would produce exactly the arrow I see?" and solves for it. That
exactly inverts the encode and recovers d with no bias. BUT it is not
reference-free: the inversion needs the comb's full geometry — its tilt steepness
(a bee can feel this from gravity) AND which compass direction the slope faces
(needs an external compass, i.e. the sun) — plus the inversion trig.

**One-line math.** As 2-D vectors in the horizontal plane: encoding multiplies the
unit heading by a 2x2 matrix M (rows = horizontal parts of the two in-comb axes),
so dance ∝ M·heading. Decode A computes M^T·(dance); decode B computes
M^-1·(dance). M^T = M^-1 only when M is a pure rotation, i.e. only on a flat comb.
On a tilt M stretches, so M^T M ≠ identity — that leftover M^T M *is* decode A's
bias, and decode B's M^-1 cancels the encode's M outright. det M = cos(tilt
angle), so B is solvable for any non-vertical comb and breaks down (no unique
heading) exactly at vertical.

**Biological reading.** Decode A is a simple, sun-free "point and follow" rule that
genuinely misdirects on a slope — a plausible failure mode of naive iconic
pointing. Decode B is geometrically perfect but secretly needs the sun (for comb
orientation) and trig, so on a tilt the "direct" code is no cheaper or more
reference-free than the gravity code it was meant to undercut. Real bees sidestep
the problem by switching to the gravity code on tilted/vertical combs, where the
reference (downhill) is read straight from gravity with no trig. The model still
down-weights direct as the comb tilts via s_dir (highest near horizontal, where
the decode is trivial; ~0 near vertical), so the reliability penalty survives even
with the unbiased decode.

**Open question.** What should the "direct" code represent — a naive reference-free
pointer (cheap, accurate only near horizontal: decode A's spirit) or an idealized
geometric channel (exact everywhere, but implicitly sun+trig dependent: decode B /
the fix)? v3 adopts B because it matches the documented intent (tilt = reliability
loss, not bias) and avoids an undocumented ~18° handicap, but this is a
modeling-interpretation choice to revisit. The paper should not silently imply
bees perform exact inverse projection on tilted combs.

**Decision.**

We'll make Decode A (Flatten) vs Decode B (Unproject) a switchable configuration option. So current experiments use Flatten. We'll need to run equivalent experiments under Unproject to check what effect this has. 
