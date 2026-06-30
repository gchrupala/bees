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
is at a horizontal compass bearing d. The horizontal "to the food" arrow is
projected orthogonally onto the sloped comb plane — along the comb's own normal,
not vertically. This foreshortens the arrow's component along the steepest-slope
(downhill/uphill) direction; the component along the contour (level) direction is
unchanged. So the angle of the arrow *as it lies in the surface* is NOT equal to
d. That in-surface angle is the dance signal.

Given the dance arrow in the surface, there are two different ways to turn it back
into a compass bearing:

**Decode A — "flatten the gesture" (the OLD, biased decode).** Take the
in-surface gesture vector and read off its horizontal projection (drop the
vertical component) as the compass bearing. This names a naive, uncorrected
readout of the gesture's apparent direction — a property of the operation, not a
claim about how a bee senses the dance (tactile, vibrational, and gravity-related
cues are all plausible inputs to performing it). It is biased: encoding already
shortened the slope-axis (downhill/uphill) component once; this decode shortens
it again instead of restoring it. Two shortenings is not a round trip — the
contour component is untouched both times, but the slope-axis component is hit
twice — and the result is a rotation of the recovered bearing toward the comb's
tilt axis. At 45° tilt the bias is zero for directions exactly along the slope
axis or the contour, and reaches up to ~19.5° in between (verified by direct
computation).

**Decode B — "un-project" (the NEW, exact decode).** The follower asks: "which
horizontal bearing, projected the same way, would produce exactly the arrow I
see?" and solves for it. That exactly inverts the encode and recovers d with no
bias. Decode A and decode B are defined over the same two geometric variables —
comb tilt and orientation (see the math below) — but they use tilt differently:
decode B requires the receiver to use tilt *quantitatively*, computing a
1/cos(tilt) correction; decode A's tilt-dependence falls out of the projection
itself and needs no such computation.

**One-line math.** As 2-D vectors in the horizontal plane: encoding multiplies the
unit heading by a 2x2 matrix M (rows = horizontal parts of the two in-comb axes),
so dance ∝ M·heading. Decode A computes M^T·(dance); decode B computes
M^-1·(dance). M^T = M^-1 only when M is orthogonal (M^T M = I); here det M =
cos(tilt) > 0 away from vertical, so orthogonality is reached only at zero tilt,
where M is a pure rotation.

M factors as M = U Σ Vᵀ, with Σ = diag(1, cos(tilt)) and U, V rotations fixed by
tilt and orientation. Then M^T = V Σ Uᵀ and M^-1 = V Σ⁻¹ Uᵀ — **the same U and V
for both decodes**; they differ only in Σ vs Σ⁻¹. So decode A and decode B make
identical use of comb orientation; the entire difference is shrinking the
foreshortened component by cos(tilt) (A) versus stretching it by 1/cos(tilt) (B).
det M = cos(tilt angle), so B is solvable for any non-vertical comb and breaks
down (no unique heading) exactly at vertical.

**Biological reading.** Decode A and decode B are defined over the same
geometric variables (per the math above), but decode B requires the receiver to
use tilt quantitatively — recovering a foreshortened length needs holding tilt as
a value and computing the correction. Decode A's tilt-dependence requires no such
computation; it is whatever a naive, uncorrected readout of the gesture already
produces. This describes the two operations, not an assumed sensory mechanism —
the note takes no position on whether a bee performing decode A relies on
tactile, vibrational, or gravity-related cues. Real bees sidestep the problem by
switching to the gravity code on
tilted/vertical combs, where the reference (downhill) is read straight from
gravity with no trig. The model still down-weights direct as the comb tilts via
s_dir (highest near horizontal, where the decode is trivial; ~0 near vertical),
so the reliability penalty survives even with the unbiased decode.

**Open question.** What should the "direct" code represent — a naive pointer
(cheap, passive, accurate only near horizontal: decode A's spirit) or an
idealized geometric channel (exact everywhere, but requiring active trigonometric
computation rather than passive projection: decode B / the fix)? v3 adopts B
because it matches the documented intent (tilt = reliability loss, not bias) and
avoids an undocumented ~19.5° handicap, but this is a modeling-interpretation
choice to revisit. The paper should not silently imply bees perform exact inverse
projection on tilted combs.

**Decision.**

We'll make Decode A (Flatten) vs Decode B (Unproject) a switchable configuration option. So current experiments use Flatten. We'll need to run equivalent experiments under Unproject to check what effect this has. 
