---
title: Evolution of communication in populations of honeybee-like agents
author: Grzegorz Chrupala
bibliography: references.bib
link-citations: true
---

# Introduction

Honeybee waggle dances provide spatial information about resources away from the nest.
One ecological hypothesis is that such communication is favored when food is difficult
to discover independently but remains useful after a worker has found it. Empirical and
theoretical studies suggest that the value of dance information and recruitment depends
on the spatiotemporal distribution of food resources, including patch density, patch
size, reward variability, and habitat [@sherman_visscher_2002;
@dornhaus_chittka_2004; @dornhaus_etal_2006; @beekman_lew_2008;
@donaldson_matasci_dornhaus_2012; @schurch_gruter_2014;
@price_gruter_2015]. The model is deliberately minimal: it asks which evolutionary
pressures are needed before adding richer social, developmental, or landscape detail.
We start from direct directional communication on an exposed horizontal comb, where
dance angle can point in the food direction, and then ask what additional conditions
make a gravity-referenced vertical-comb code reachable.

The simulations evaluate four evolutionary hypotheses.

1.  **Resource-distribution hypothesis.** Costly directional communication should be
    favored when independent discovery is difficult enough that recruitment is useful,
    but not so difficult that few dances are seeded. It should be weakly favored when
    food is very easy to find without dances.

2.  **Architectural-precondition hypothesis.** Direct pointing should work well on a
    horizontal comb, whereas gravity-referenced communication should become useful only
    when the comb geometry supplies a reliable gravity cue. A non-communication benefit
    of vertical combs may therefore be needed to make the gravity channel selectively
    available.

3.  **Sender&ndash;receiver coordination hypothesis.** Sender and receiver transposition
    must evolve together. Positive coupling between sender and receiver mutations should
    reduce the time spent in mismatched codes.

4.  **Transition-risk hypothesis.** Selection for vertical combs can outpace adaptation
    of the communication code. Successful transitions should therefore require overlap
    among vertical geometry, directional precision, receiver attention, and matched
    sender&ndash;receiver transposition; otherwise foraging may collapse before the new
    code becomes useful.

# Model

## Conceptual model

Colonies are the reproducing entities, while workers are behavioral agents sampled from
colony-level trait means. A colony has seven heritable mean traits: directional
precision investment, receiver attention, sender transposition, receiver transposition,
search limit, comb tilt, and comb orientation. Directional precision investment is a
costly investment in producing directionally informative dances. Receiver attention
controls whether workers use available dances. Sender and receiver transposition
interpolate between two communication modes: direct pointing in the comb plane and
sun/gravity-referenced encoding. Comb tilt and orientation define the dance plane in the
world, and search limit determines how far a worker searches along its chosen direction.

Each foraging episode samples food sites with direction, distance, angular width, value,
and capacity. Workers act sequentially. If dances are available, a worker may follow
one; otherwise it searches independently in a random direction. Independent searchers
that find food add a noisy dance for that site. There are therefore no fixed scouts in
the model. Background discovery and recruitment both emerge from the food distribution,
receiver attention, search limits, and dance precision. This follows ecological
recruitment models in which communication is valuable only when resources are
discoverable and worth recruiting to [@dornhaus_etal_2006; @beekman_lew_2008].

Comb geometry determines which directional cue is available. Direct pointing projects
the horizontal food direction onto the comb plane, so its strength depends on the food
direction, comb tilt, and comb orientation. Gravity-referenced mapping uses the
projection of gravity onto the comb plane together with the episode&rsquo;s sun azimuth.
Gravity provides no in-plane reference on a horizontal comb and becomes a stronger cue
as the comb becomes vertical. Comb orientation can be treated as circular, or as axial
when orientations $\phi$ and $\phi + 180^\circ$ represent the same comb plane.

## Mathematical implementation

Let a colony&rsquo;s mean strategy be $$\theta = (b, a, s, r, \ell, t, \phi),$$ where
$b \in [0,1]$ is directional precision investment, $a \in [0,1]$ is receiver attention,
$s,r \in [0,1]$ are sender and receiver transposition, $\ell \geq 0$ is search limit,
$t \in [0,1]$ is comb tilt, and $\phi$ is comb orientation. A worker $i$ receives stable
within-colony deviations from the colony means:
$$\theta_i = \operatorname{clip}(\theta + \epsilon_i).$$ These deviations represent
worker heterogeneity but are not themselves heritable.

For a food direction $\alpha$ and sun azimuth $\psi$, let $D(\alpha;t,\phi)$ be the
direct projected dance angle and $G(\alpha,\psi;t,\phi)$ be the sun/gravity-referenced
dance angle. Let $q_D$ and $q_G$ be their corresponding cue strengths in the comb plane.
A sender with transposition $s_i$ encodes the dance mean angle as a weighted circular
mean, $$\mu_i =
\operatorname{cmean}\left(
  (D, (1-s_i)q_D),\,
  (G, s_i q_G)
\right).$$ The produced dance signal is $$x_i =
\operatorname{VM}(\mu_i, \kappa_{\max} b_i) + \eta_d
\pmod{2\pi},$$ where $\operatorname{VM}$ is the von Mises distribution,
$\kappa_{\max}=14$ in the reported simulations, and $\eta_d$ is transient
dance-production noise. A receiver with transposition $r_j$ applies the analogous
weighted decoding rule and adds interpretation noise $\eta_r$ to obtain a search
direction $\hat{\alpha}_j$.

A worker succeeds at food site $k$ only if $$\Delta(\hat{\alpha}_j,\alpha_k) \leq w_k
\quad\text{and}\quad
d_k \leq \ell_j,$$ where $\Delta$ is circular angular distance, $w_k$ is site angular
width, and $d_k$ is site distance. If multiple available sites satisfy this condition,
the nearest site is used.

Let the net foraging payoff for an episode be $$F_e =
\sum_{j \in \mathcal{F}_e} v_{k(j)}
- c_T\left(\sum_{j \in \mathcal{F}_e} d_{k(j)}
  + \sum_{j \in \mathcal{U}_e} \ell_j\right)
- \sum_{i \in \mathcal{D}_e} (c_0 + c_b b_i)
- c_a A_e .$$ Here $\mathcal{F}_e$ is the set of successful workers, $\mathcal{U}_e$ the
unsuccessful workers, $\mathcal{D}_e$ the independent finders that produce dances, and
$A_e$ the number of dance-following attempts. The terms are food value, travel/search
cost, per-dance communication cost, and receiver-attention cost. A vertical comb can
then have an optional non-communication advantage that scales, rather than adds to, net
performance: $$P_e = F_e \left(1 + \alpha f(t)\right).$$ The default modifier is
$f(t)=t$, so $\alpha=0.05$ means that a fully vertical comb has a 5% advantage over a
horizontal comb with the same foraging performance. The long-transition robustness check
also uses $f(t)=\mathbf{1}[t \geq 0.8]$. Because the advantage is multiplicative,
verticality cannot rescue a colony whose foraging payoff has collapsed. Mean colony
payoff is averaged over episodes and floored at 0.001 so that low-payoff colonies remain
selectable but rare.

Daughter colonies are sampled in proportion to payoff. Only colony means are inherited.
Mutations are Gaussian and clipped to the allowed ranges; sender and receiver
transposition mutations can have correlation $\rho$, which is the coupling parameter
used in the vertical-transition probes.

## Simulation parameters and reproducibility

Unless otherwise stated, reported runs use 60 colonies, 80 workers per colony, 50
foraging episodes per colony per generation, 12 foraging attempts per episode, ten
random seeds, within-colony standard deviation 0.08 for unit-interval traits, mutation
standard deviation 0.07, dance-production noise standard deviation 0.18,
interpretation-noise standard deviation 0.12, maximum search distance 8.0, food value
1.0, food-site capacity 6, travel cost 0.02 per distance unit, precision-dependent
dance-cost coefficient $c_b=0.02$, baseline dance cost $c_0=0$, and attention cost
$c_a=0.01$. Search-limit noise and mutation use the same scales multiplied by the
maximum search distance. Initial directional precision is sampled from $[0,0.15]$,
initial receiver attention from $[0,0.25]$, initial transposition traits are 0, and
initial search limit is sampled from 15&ndash;45% of the maximum search distance.

This working report is edited as `report/report.md` and rendered to
`report/report.html` with:

```sh
python -u experiments/render_report_html.py
```

The rendered HTML is the lightweight report artifact. The legacy LaTeX source and PDF
are kept as a paper-style snapshot for later polishing. Tables and generated figures
remain grounded in tracked result CSVs under `results/`; the legacy provenance manifest
is `report/artifact_sources.csv`.

# Results

## Horizontal comb

We first compared three simple food distributions over ten random seeds (101&ndash;110).
The &ldquo;hard&rdquo; condition had one narrow food site per episode, the baseline
condition had two moderately wide food sites, and the &ldquo;easy&rdquo; condition had
many broad food sites.
Table&nbsp;<a href="#tab:food-distribution" data-reference-type="ref"
data-reference="tab:food-distribution">1</a> reports the generation at which mean
directional precision investment first reached 0.30, plus final-generation trait,
success, and payoff values.

Each run used 60 colonies, 80 workers per colony, 30 generations, 50 foraging episodes
per colony per generation, 12 foraging attempts per episode, and horizontal-comb tilt
$t=0$. Food-site distances were sampled uniformly from 1.0 to 8.0, maximum search
distance was 8.0, travel cost was 0.02 per distance unit, baseline dance cost was 0, the
precision-dependent dance-cost coefficient was 0.02, and food-site value was 1.0. All
three conditions used food-site capacity 6. The conditions varied only the number and
angular width of food sites: hard used one site of width 0.08, baseline used two sites
of width 0.20, and easy used eight sites of width 0.50. The hard condition therefore
makes independent discovery unlikely, while the easy condition makes independent
discovery likely even without using the dance.

<div id="tab:food-distribution">

| Condition | Reached | Reach gen. | Prec. | Attention | Search | Success | Payoff |
|:----------|--------:|-----------:|------:|----------:|-------:|--------:|-------:|
| Hard      |    4/10 |     22.750 | 0.264 |     0.313 |  2.789 |   0.008 |  0.001 |
| Baseline  |   10/10 |      7.700 | 0.844 |     0.868 |  7.163 |   0.194 |  0.733 |
| Easy      |    8/10 |     20.250 | 0.343 |     0.366 |  7.045 |   0.709 |  7.416 |

Food-distribution experiment under the current dynamic recruitment model.
&ldquo;Reached&rdquo; is the number of seeds in which mean directional precision
investment reached 0.30 by generation 30. &ldquo;Reach gen.&rdquo; is averaged over
reached seeds only. Prec. is final mean directional precision investment; attention,
search limit, success, and payoff values are also final-generation means over all ten
seeds.

</div>

The result supports the resource-distribution hypothesis. In the hard environment, food
is so rare and narrow that independent discoveries seldom seed dances; final success and
payoff remain close to zero, and communication crosses the directional-precision
threshold in only 4 of 10 seeds. In the baseline environment, independent discovery is
sufficient to seed recruitment cascades, and directional precision, receiver attention,
and search limit all rise strongly. In the easy environment, success and payoff are
high, but directional precision and receiver attention remain much lower than in the
baseline case because food is often found without precise communication.

## Vertical comb

We also compared colonies initialized with horizontal, tilted, and vertical combs while
holding the baseline food distribution fixed. Initial comb tilt was set to $t=0$,
$t=0.5$, or $t=1.0$, with ten seeds per condition. Comb tilt was still heritable and
mutable in these runs.
Table&nbsp;<a href="#tab:comb-tilt" data-reference-type="ref" data-reference="tab:comb-tilt">2</a>
summarizes the result after 30 generations.

<div id="tab:comb-tilt">

| Comb       | Reached | Reach gen. | Prec. | Attention | Sender | Receiver | Success | Payoff |
|:-----------|--------:|-----------:|------:|----------:|-------:|---------:|--------:|-------:|
| Horizontal |   10/10 |      7.700 | 0.844 |     0.868 |  0.187 |    0.176 |   0.194 |  0.733 |
| Tilted     |   10/10 |      8.100 | 0.829 |     0.886 |  0.171 |    0.196 |   0.195 |  0.727 |
| Vertical   |   10/10 |     16.500 | 0.606 |     0.794 |  0.384 |    0.362 |   0.165 |  0.414 |

Comb-tilt comparison using the baseline food distribution and ten seeds. Sender and
receiver denote final mean sender and receiver transposition traits.

</div>

With mutable tilt, all three initial-tilt conditions reach the directional-precision
threshold in all ten seeds under the current baseline settings. Tilted and vertical
starts retain substantial foraging success over 30 generations, but they also evolve
higher sender and receiver transposition than horizontal starts. The initially vertical
condition still has lower payoff than the horizontal condition, and its final mean tilt
is reduced in the underlying result file, indicating that the short-run comparison does
not by itself establish a stable vertical-comb transition.

## Tilt-geometry calibration

We next ran a calibration grid over initial comb tilt and the proportional vertical-comb
advantage $\alpha$ using the default modifier $f(t)=t$. These runs used the baseline
food distribution, the daytime sun arc from $0^\circ$ to $180^\circ$ centered at
$90^\circ$, and ten seeds (101&ndash;110). Success is the fraction of foraging attempts
that found food. Payoff is the final mean colony payoff after food rewards,
search/travel costs, dance costs, attention costs, and the multiplicative vertical-comb
modifier. Table&nbsp; <a href="#tab:tilt-benefit-calibration" data-reference-type="ref"
data-reference="tab:tilt-benefit-calibration">3</a> shows that proportional advantages
of 15&ndash;25% do not by themselves make horizontal or tilted starts retain verticality
within 30 generations, but vertical starts can retain more tilt.

<div id="tab:tilt-benefit-calibration">

| Initial tilt | $\alpha$ | Final tilt | Sender | Receiver | Success | Payoff | Reached |
|-------------:|---------:|-----------:|-------:|---------:|--------:|-------:|--------:|
|        0.000 |    0.150 |      0.129 |  0.186 |    0.177 |   0.197 |  0.757 |   10/10 |
|        0.000 |    0.200 |      0.136 |  0.188 |    0.205 |   0.193 |  0.731 |   10/10 |
|        0.000 |    0.250 |      0.149 |  0.188 |    0.196 |   0.196 |  0.753 |   10/10 |
|        0.500 |    0.150 |      0.154 |  0.183 |    0.196 |   0.192 |  0.708 |   10/10 |
|        0.500 |    0.200 |      0.126 |  0.232 |    0.204 |   0.190 |  0.692 |   10/10 |
|        0.500 |    0.250 |      0.237 |  0.211 |    0.250 |   0.187 |  0.672 |   10/10 |
|        1.000 |    0.150 |      0.568 |  0.403 |    0.424 |   0.164 |  0.450 |    9/10 |
|        1.000 |    0.200 |      0.292 |  0.248 |    0.261 |   0.176 |  0.580 |   10/10 |
|        1.000 |    0.250 |      0.636 |  0.440 |    0.432 |   0.160 |  0.449 |    9/10 |

Ten-seed calibration of the proportional vertical-comb advantage under the explicit
tilt-geometry model. Sender and receiver are final mean transposition traits. Reached is
the number of seeds in which mean directional precision investment reached 0.30 by
generation 30.

</div>

The calibration suggests that the transition is strongly path-dependent. Starting from a
horizontal or tilted comb, the tested advantages leave final mean tilt below 0.25 while
preserving high foraging success. Starting from a vertical comb, final tilt is higher,
especially at $\alpha=0.15$ and $\alpha=0.25$, but success and payoff are lower than in
flatter conditions. The proportional rule therefore removes the artificial life-support
effect of an additive bonus: verticality is retained only when enough foraging
performance remains to be scaled. The relevant open question is whether longer runs can
turn supplied verticality into a coordinated gravity-referenced code before selection
drives colonies back toward flatter combs.

## Tilt orientation

We also examined the evolved orientation of the comb tilt. The current sun arc is
centered at $90^\circ$, so
Table&nbsp;<a href="#tab:orientation-calibration" data-reference-type="ref"
data-reference="tab:orientation-calibration">4</a> reports the weighted mean orientation
in degrees and its offset from this sun-arc center. The within-alignment column measures
how strongly colonies within a run converge on an orientation. The across-alignment
column measures whether different seeds converge on the same absolute orientation,
weighting each seed by its within-run orientation alignment.

<div id="tab:orientation-calibration">

| Initial tilt | $\alpha$ | Final tilt | Success | Within align. | Mean orient. | Sun offset | Across align. |
|-------------:|---------:|-----------:|--------:|--------------:|-------------:|-----------:|--------------:|
|        0.000 |    0.150 |      0.129 |   0.197 |         0.480 |      352.790 |      -97.2 |         0.182 |
|        0.000 |    0.200 |      0.136 |   0.193 |         0.396 |      174.322 |       84.3 |         0.092 |
|        0.000 |    0.250 |      0.149 |   0.196 |         0.461 |      263.759 |      173.8 |         0.431 |
|        0.500 |    0.150 |      0.154 |   0.192 |         0.430 |       95.069 |        5.1 |         0.289 |
|        0.500 |    0.200 |      0.126 |   0.190 |         0.506 |      337.558 |     -112.4 |         0.136 |
|        0.500 |    0.250 |      0.237 |   0.187 |         0.610 |      268.532 |      178.5 |         0.255 |
|        1.000 |    0.150 |      0.568 |   0.164 |         0.653 |      285.222 |     -164.8 |         0.490 |
|        1.000 |    0.200 |      0.292 |   0.176 |         0.591 |      305.511 |     -144.5 |         0.390 |
|        1.000 |    0.250 |      0.636 |   0.160 |         0.768 |      238.536 |      148.5 |         0.423 |

Orientation sanity check under the explicit tilt-geometry model. Mean orientation and
sun offset are measured in degrees. Across alignment is the weighted across-seed
orientation alignment; values near 0 indicate that different seeds choose different
absolute orientations.

</div>

Orientation sometimes aligns within a run, but it does not show a clean, robust
relationship to the daytime sun arc across seeds under the circular orientation
representation. Some conditions point near the sun-arc center, some near the opposite
direction, and several have weak across-seed alignment. This indicates weak or
path-dependent selection on orientation in these probes and motivates the axial
representation used in the longer transition runs, where $\phi$ and $\phi + 180^\circ$
are treated as the same comb plane.

## Constrained vertical coupling

The tilt-calibration results left open whether gravity-referenced communication is
difficult because the sender&ndash;receiver code itself is hard to reach, or because
colonies must evolve the vertical geometry and the sender&ndash;receiver code at the
same time. To separate these possibilities, we ran a constrained probe with the
geometric precondition supplied. Initial comb tilt was fixed near vertical at $t=0.95$,
comb-tilt mutation was set to 0, comb orientation was treated axially, the vertical-comb
advantage was set to 0, and the baseline food distribution was otherwise retained. We
then varied only the sender&ndash;receiver transposition mutation correlation and ran
each condition for 120 generations over five seeds (101&ndash;105). A run was counted as
reaching gravity-referenced communication when both mean sender and receiver
transposition reached 0.50.

<div id="tab:vertical-coupling-probe">

| Corr. | Reached | Reach gen. | Prec. | Sender | Receiver |   Gap | Success | Payoff |
|------:|--------:|-----------:|------:|-------:|---------:|------:|--------:|-------:|
| 0.000 |     5/5 |     42.600 | 0.878 |  0.864 |    0.881 | 0.033 |   0.192 |  0.680 |
| 0.300 |     5/5 |     46.200 | 0.847 |  0.913 |    0.898 | 0.022 |   0.192 |  0.694 |
| 0.600 |     5/5 |     21.000 | 0.882 |  0.890 |    0.897 | 0.010 |   0.199 |  0.754 |
| 0.900 |     5/5 |     22.400 | 0.863 |  0.907 |    0.903 | 0.005 |   0.194 |  0.723 |
| 1.000 |     5/5 |     23.400 | 0.836 |  0.901 |    0.901 | 0.000 |   0.191 |  0.677 |

Constrained near-vertical coupling probe. Corr. is the correlation between sender and
receiver transposition mutation increments. &ldquo;Reached&rdquo; counts seeds in which
both mean sender and receiver transposition reached 0.50 by generation 120. Reach
generation is averaged over reached seeds only. Gap is the final mean absolute
sender&ndash;receiver transposition difference.

</div>

Table&nbsp;<a href="#tab:vertical-coupling-probe" data-reference-type="ref"
data-reference="tab:vertical-coupling-probe">5</a> separates reachability from
transition speed. With near-vertical geometry supplied, every correlation condition
reaches gravity-referenced communication in all five seeds, so the gravity channel is
not intrinsically unreachable. The main difference is how quickly the matched
sender&ndash;receiver code appears. Independent or weakly coupled mutations ($\rho=0$ or
$0.3$) reach the threshold only after about 43&ndash;46 generations on average. Moderate
to strong coupling ($\rho=0.6$&ndash;$1.0$) reaches it by about generation 21&ndash;23.
The final sender&ndash;receiver gap also falls as coupling increases: from 0.033 without
coupling to 0.010 at $\rho=0.6$ and 0 at $\rho=1.0$. Payoff peaks at $\rho=0.6$, not at
perfect coupling, suggesting that coupling helps align the code but complete lockstep is
not necessarily the highest-payoff mutation structure in this finite-seed probe. The
result supports the sender&ndash;receiver coordination hypothesis and implies that the
harder problem is the joint transition in which comb architecture, orientation
representation, and the sender&ndash;receiver code must all become useful together.

## Long transition runs

We therefore ran longer transition experiments using the axial orientation
representation. These runs used the baseline food distribution, 120 generations, ten
seeds (101&ndash;110), sender&ndash;receiver transposition mutation correlation 0.6,
three initial comb tilts ($0$, $0.5$, and $1.0$), and proportional vertical-comb
advantages $\alpha \in \{0.05,0.10,0.15,0.20,0.25\}$. The default runs used $f(t)=t$,
and a robustness check used $f(t)=\mathbf{1}[t \geq 0.8]$. Unlike the constrained
near-vertical probe, comb tilt was free to mutate. A seed was counted as reaching the
gravity channel when both mean sender and receiver transposition reached 0.50. It was
counted as retaining verticality when final mean comb tilt was at least 0.80. It was
counted as collapsed when mean foraging success fell to 0.02 or below at any generation,
and as recovered when a collapsed seed ended with final success at least 0.10.

{{ long_transition_heatmap }}

The proportional rule changes the interpretation of the long runs
(Figure&nbsp;<a href="#fig:long-transition-heatmap" data-reference-type="ref"
data-reference="fig:long-transition-heatmap">1</a>). From horizontal starts, no tested
advantage reaches the gravity channel or retains verticality under either modifier, and
final success remains high. From tilted starts, no tested advantage reaches the gravity
channel or retains verticality either; a few seeds transiently collapse but recover by
the final generation. Thus proportional verticality pressure up to 25% is not sufficient
to bootstrap the architecture from horizontal or intermediate tilt under the current
mutation and ecological settings.

From vertical starts, by contrast, verticality and gravity-referenced communication
sometimes coevolve without lasting foraging failure. With $f(t)=t$, the number of seeds
retaining verticality and reaching the gravity channel is 2/10 at $\alpha=0.05$, 2/10 at
$\alpha=0.10$, 4/10 at $\alpha=0.15$, 0/10 at $\alpha=0.20$, and 3/10 at $\alpha=0.25$.
With the threshold modifier, the corresponding counts are 2/10 for
$\alpha=0.05$&ndash;$0.20$ and 3/10 at $\alpha=0.25$. Collapse is limited to one seed in
each vertical-start condition and those seeds recover by generation 120. The transition
is therefore possible when vertical architecture is already supplied, but it remains
stochastic and non-monotonic rather than robust.

## Food-distribution transition probe

The long transition runs showed that the baseline food distribution does not bootstrap
verticality from horizontal or tilted starts. We next asked whether more forgiving food
distributions could create a viable transition corridor from horizontal starts without
changing the inheritance or mutation rules. All runs in this probe started at $t=0$,
used axial orientation, retained the sender&ndash;receiver mutation correlation
$\rho=0.6$, used the proportional vertical-comb modifier $f(t)=t$ with $\alpha=0.25$,
and ran for 120 generations over five seeds (101&ndash;105). We varied only the food
distribution: angular width, site count, capacity, and value.

<div id="tab:food-transition">

| Cond.           | Sites | Width | Cap. | Value | Stable | Grav. | Vert. | $m_f$ | $t_f$ | Succ. |
|:----------------|------:|------:|-----:|------:|-------:|------:|------:|------:|------:|------:|
| Baseline        |     2 | 0.200 |    6 | 1.000 |    0/5 |   0/5 |   0/5 | 0.195 | 0.134 | 0.194 |
| Broad           |     2 | 0.350 |    6 | 1.000 |    0/5 |   0/5 |   0/5 | 0.189 | 0.196 | 0.317 |
| Broad cap.      |     2 | 0.350 |   12 | 1.000 |    0/5 |   0/5 |   0/5 | 0.230 | 0.115 | 0.391 |
| Broad rich      |     2 | 0.350 |    6 | 1.500 |    0/5 |   0/5 |   0/5 | 0.195 | 0.197 | 0.316 |
| Broad rich cap. |     2 | 0.350 |   12 | 1.500 |    0/5 |   0/5 |   0/5 | 0.181 | 0.202 | 0.379 |
| Dense           |     4 | 0.300 |   10 | 1.000 |    1/5 |   1/5 |   1/5 | 0.301 | 0.297 | 0.470 |

Horizontal-start food-distribution transition probe. All rows use five seeds
(101&ndash;105), axial orientation, 120 generations, sender&ndash;receiver mutation
correlation $\rho=0.6$, and proportional vertical-comb advantage $\alpha=0.25$. Stable
counts seeds that end with final comb tilt at least 0.80 and both sender and receiver
transposition at least 0.50. Gravity counts seeds that reached the transposition
threshold at any generation; vertical counts seeds retaining final comb tilt at least
0.80. Final $m$ is the final mean of the lower of sender and receiver transposition.

</div>

The result is mixed but informative. Broader, richer, or higher-capacity patches
substantially improve foraging success relative to the baseline, but they do not by
themselves make the gravity code or stable vertical combs evolve from horizontal starts.
For example, broad high-capacity patches raise final success to 0.391, but no seed
reaches the gravity threshold or retains verticality, and mean final tilt remains 0.115.

The only successful horizontal-start transition in this probe occurs in the
moderate-dense condition with four sites, angular width 0.30, capacity 10, and unit
value. One of five seeds reaches the gravity-channel threshold at generation 75, reaches
verticality at generation 108, and ends with comb tilt 0.911 and sender and receiver
transposition 0.863 and 0.862. The same condition also has the highest final success
(0.470). The other four seeds in that condition, however, stay far from vertical and do
not reach the gravity threshold. Thus this ecology can open a transition path from
horizontal starts, but the path is still rare rather than reliably selected under the
current settings.

## Local food-transition grid

To resolve the neighborhood around the seed-101 transition, we ran a local grid from
horizontal starts while keeping the model rules fixed. The grid used seed 101, axial
orientation, $\rho=0.6$, food value 1.0, $f(t)=t$, 120 generations, and four worker
processes. It varied the food-site count $n \in \{3,4,5\}$, angular width
$w \in \{0.26,0.30,0.34\}$, capacity $K \in \{8,10,12\}$, and vertical-comb advantage
$\alpha \in \{0.20,0.25,0.30\}$, for 81 conditions. Table&nbsp;
<a href="#tab:food-transition-local-grid" data-reference-type="ref"
data-reference="tab:food-transition-local-grid">7</a> lists the only conditions that
ended in a stable vertical gravity-code state.

<div id="tab:food-transition-local-grid">

| Sites | Width | Cap. | $\alpha$ | Grav. gen. | Vert. gen. | $m_f$ | $t_f$ | Succ. |
|------:|------:|-----:|---------:|-----------:|-----------:|------:|------:|------:|
|     4 | 0.300 |   10 |    0.250 |         75 |        108 | 0.862 | 0.911 | 0.485 |
|     5 | 0.260 |    8 |    0.300 |         52 |         55 | 0.844 | 0.896 | 0.445 |
|     5 | 0.300 |   10 |    0.300 |         50 |         82 | 0.788 | 0.861 | 0.466 |
|     5 | 0.300 |   12 |    0.300 |         50 |         56 | 0.894 | 0.855 | 0.494 |

Stable outcomes in the 81-condition seed-101 local ecology grid. All runs start
horizontal, use axial orientation, $\rho=0.6$, value 1.0, the proportional modifier
$f(t)=t$, and 120 generations. Stable means final comb tilt at least 0.80 and both
sender and receiver transposition at least 0.50.

</div>

The local grid confirms that the earlier transition is reproducible for the same seed
and parameter setting, but it also shows that the transition is not a smooth monotone
response to easier foraging or stronger verticality pressure. Only 4 of 81 seed-101
conditions are stable. The original pocket ($n=4,w=0.30,K=10,\alpha=0.25$) remains
stable, while its immediate $\alpha=0.20$ and $\alpha=0.30$ neighbors do not. Three
additional stable conditions occur with five food sites and $\alpha=0.30$. In contrast,
many high-success conditions at width 0.34 remain flat and do not reach the gravity
threshold. The transition therefore appears to require a fairly specific balance: enough
discovery and recruitment opportunity to keep foraging viable, but not so much
ecological tolerance that vertical architecture and gravity-referenced transposition are
weakly selected.

We then reran the four stable pockets over seeds 96&ndash;106 for 160 generations. This
seed panel is a reproducibility check rather than a statistical neighborhood in the
random-number sequence. Table&nbsp;
<a href="#tab:food-transition-local-robustness" data-reference-type="ref"
data-reference="tab:food-transition-local-robustness">8</a> shows that the pockets
remain rare.

<div id="tab:food-transition-local-robustness">

| Sites | Width | Cap. | $\alpha$ | Stable | Seeds   | Mean $m_f$ | Mean $t_f$ | Mean succ. |
|------:|------:|-----:|---------:|-------:|:--------|-----------:|-----------:|-----------:|
|     4 | 0.300 |   10 |    0.250 |   1/11 | 101     |      0.238 |      0.244 |      0.457 |
|     5 | 0.260 |    8 |    0.300 |   2/11 | 100,101 |      0.300 |      0.369 |      0.424 |
|     5 | 0.300 |   10 |    0.300 |   1/11 | 101     |      0.264 |      0.229 |      0.512 |
|     5 | 0.300 |   12 |    0.300 |   1/11 | 101     |      0.258 |      0.245 |      0.506 |

Robustness panel for the four stable pockets from the seed-101 local grid. Each pocket
is rerun for seeds 96&ndash;106 and 160 generations. Stable counts final vertical
gravity-code outcomes; the seed list gives which seeds were stable.

</div>

All four seed-101 stable states remain stable when extended to generation 160, so they
are not transient generation-120 crossings. Across the 44 robustness runs, however, only
five stable outcomes occur. The strongest pocket is $n=5,w=0.26,K=8,\alpha=0.30$, which
succeeds in seeds 100 and 101. The other three pockets succeed only in seed 101. None of
the robustness runs collapses, and final success remains high even in non-transition
runs, so the limiting factor is not colony viability alone. Most failures keep foraging
performance while either tilt stays low, transposition stays low, or both. This supports
the view that the ecology can make the transition possible, but the
architecture-and-code lock-in remains stochastic under the current mutation and
selection regime.

## Adaptive food-transition search

Manual grids become inefficient once several ecological parameters plausibly interact.
We therefore added an Optuna-based adaptive search over the same scientific quantities,
without introducing new biological mechanisms. The search starts from the
long-transition configuration and keeps horizontal initial combs, axial orientation,
$\rho=0.6$, food value 1.0, $f(t)=t$, and 120 generations. It varies food-site count
(3&ndash;6), angular width (0.22&ndash;0.36), capacity (6&ndash;14), vertical-comb
advantage (0.15&ndash;0.35), maximum food distance (6.0&ndash;10.0), and travel cost
(0.01&ndash;0.04). For each seed, let $t_f$ be final comb tilt and let $s_f,r_f$ be
final sender and receiver transposition. A trial is stable when $t_f \geq 0.80$,
$s_f \geq 0.50$, and $r_f \geq 0.50$. Its continuous near-miss score is
$$p = \min\left(1, \frac{t_f}{0.80}, \frac{s_f}{0.50}, \frac{r_f}{0.50}\right).$$ The
optimization objective is stable seed count plus mean $p$, minus the number of seeds
whose success falls to 0.02 or below. Thus an additional stable vertical gravity-code
seed always dominates a merely better near miss, while the continuous term still lets
the optimizer rank incomplete transitions.

As a pilot, we ran eight seed-101 trials with four workers. One trial reached the target
state: six food sites, width 0.35, capacity 13, $\alpha=0.33$, maximum food distance
8.0, and travel cost 0.01 ended with $t_f=0.914$, $\min(s_f,r_f)=0.855$, and final
success 0.608. The other seven trials did not reach stability; their progress scores
ranged from 0.129 to 0.319. This first adaptive pass therefore finds another viable
seed-101 ecology, somewhat outside the earlier three-to-five-site local grid, but it
should be treated as a search result rather than a robustness result until the best
regions are rerun over multiple seeds.

We then continued the same Optuna study for 24 total trials over seeds 100&ndash;102,
for 72 full seed simulations. No trial was stable in all three seeds, but the adaptive
search did move beyond the single-seed result. The best trial was stable in two of three
seeds, with six food sites, width 0.32, capacity 9, $\alpha=0.34$, maximum food distance
6.0, and travel cost 0.03. It had objective 2.761, mean progress 0.761, mean final
success 0.589, mean final comb tilt 0.669, and mean final minimum sender&ndash;receiver
transposition 0.596. Three additional trials were stable in one of three seeds: the
seed-101 pilot setting; a nearby six-site setting with width 0.32, capacity 9,
$\alpha=0.35$, maximum distance 6.0, and travel cost 0.03; and a lower-capacity six-site
setting with width 0.28, capacity 7, $\alpha=0.25$, maximum distance 6.0, and travel
cost 0.025. None of the 24 trials collapsed. This suggests a more specific corridor than
&ldquo;easier food&rdquo; alone: relatively many moderate patches, short food distances,
and non-negligible travel costs can make vertical gravity-code transitions repeat across
more than one seed, but the transition is still not reliable across the small seed
panel.

## Successful versus unsuccessful trajectories

To compare trajectories without confounding outcome with ecology, we reran the best
Optuna pocket over a wider seed panel, seeds 96&ndash;120, and exported generation-level
traces. The setting was six food sites, width 0.32, capacity 9, $\alpha=0.34$, maximum
food distance 6.0, travel cost 0.03, horizontal initial combs, axial orientation,
$\rho=0.6$, $f(t)=t$, and 120 generations. An exact rerun of one successful seed and one
failed seed reproduced the saved event rows, confirming that identical seeds and
identical configurations are deterministic in this implementation.

Five of 25 seeds ended in stable vertical gravity-code states: seeds 100, 102, 103, 108,
and 114. The other 20 seeds did not collapse; their final foraging success was slightly
higher on average than in the successful group (0.592 versus 0.585), so the dominant
failure mode is not loss of viability. Instead, unsuccessful seeds mostly remain in a
productive low-tilt, low-transposition basin. Successful seeds reached partial tilt
($t \geq 0.5$) by generation 39.0 on average, reached verticality by generation 65.4,
and reached gravity-code transposition by generation 73.0. Their final mean tilt and
minimum sender&ndash;receiver transposition were 0.884 and 0.829. Unsuccessful seeds had
final means of only 0.278 and 0.207, although their final precision investment and
receiver attention were higher than in the successful seeds.

The generation-level trajectories show the divergence. At generation 40, successful
seeds already averaged tilt 0.553 and minimum transposition 0.250, whereas unsuccessful
seeds averaged 0.332 and 0.186. By generation 80, the successful group had moved to tilt
0.760 and minimum transposition 0.641, while the unsuccessful group remained near tilt
0.288 and minimum transposition 0.183. Only one unsuccessful seed reached the gravity
threshold at all; it did so late, at generation 104, crossed verticality at generation
116, and ended below the vertical threshold at $t_f=0.721$. Thus the successful
trajectory is not simply higher communication effort or higher foraging success. It is a
coordinated path in which moderate verticality appears early enough for
gravity-referenced transposition to become useful before selection settles into the flat
direct-pointing regime.

## Extensive adaptive search

We then ran a broader overnight Optuna search using the same stable-state objective, but
allowing both ecological and mutation parameters to vary. The run used horizontal
initial combs, axial orientation, $f(t)=t$, 120 generations, four worker processes, 144
trials, and five seeds per trial (100&ndash;104). It varied food-site count (3&ndash;8),
angular width (0.20&ndash;0.40), capacity (6&ndash;16), food value (0.8&ndash;1.3),
vertical-comb advantage (0.15&ndash;0.45), maximum food distance (5.0&ndash;9.0), travel
cost (0.01&ndash;0.05), ordinary mutation scale (0.04&ndash;0.11), comb-tilt mutation
scale (0.04&ndash;0.11), and sender&ndash;receiver transposition mutation correlation
(0.3&ndash;0.9). The complete trial-level and seed-level records are saved as the two
`food_transition_optuna_extensive_*` CSV files under `results/`.

All 144 trials completed, for 720 seed simulations, and no seed collapsed under the
success threshold. The search found 29 trials that reached stable vertical gravity-code
states in all five seeds. The distribution of stable seed counts per trial was 21 trials
with 0 of 5 stable seeds, 15 with 1 of 5, 14 with 2 of 5, 30 with 3 of 5, 35 with 4 of
5, and 29 with 5 of 5. The adaptive search also improved as trials accumulated: the
first 32 startup trials contained one 5-of-5 trial and two trials with at least four
stable seeds, whereas the final 48 trials contained 18 5-of-5 trials and 32 trials with
at least four stable seeds.

The 5-of-5 trials concentrate in a more specific regime than the earlier six-site
pocket. Among the 29 fully stable trials, 28 used seven food sites and one used eight;
their mean food-site width was 0.357, mean capacity 8.4, mean food value 0.95, mean
vertical-comb advantage 0.428, mean maximum food distance 5.03, mean travel cost 0.035,
mean ordinary mutation scale 0.091, mean comb-tilt mutation scale 0.076, and mean
sender&ndash;receiver mutation correlation 0.841. Thus the strongest optimization-panel
regime combines many moderately broad nearby patches, a substantial vertical-comb
advantage, non-negligible travel costs, relatively large mutations, and fairly strong
correlated changes to sender and receiver transposition.

The highest-success fully stable trial used eight food sites, width 0.39, capacity 10,
food value 1.0, vertical-comb advantage 0.41, maximum food distance 5.0, travel cost
0.035, ordinary mutation scale 0.08, comb-tilt mutation scale 0.07, and
sender&ndash;receiver mutation correlation 0.9. Across seeds 100&ndash;104 it had mean
final success 0.729, mean payoff 9.695, mean comb tilt 0.872, and mean minimum
sender&ndash;receiver transposition 0.764. The TPE sampler also repeatedly returned a
nearby seven-site setting: width 0.37, capacity 10, food value 0.9, vertical-comb
advantage 0.44, maximum distance 5.0, travel cost 0.035, ordinary mutation scale 0.09,
comb-tilt mutation scale 0.08, and correlation 0.9. This setting appeared in trials 114,
116, and 117, and in each repeat all five seeds were stable, with mean final success
0.670, mean comb tilt 0.848, and mean minimum transposition 0.780.

These results change the status of the transition search. The earlier Optuna pocket
showed that ecological structure could make the transition possible but not yet reliable
across seeds. The broader search shows that, once mutation scale and
sender&ndash;receiver coupling are also allowed to vary within the same model, the
optimizer can find regimes where horizontal combs repeatedly evolve toward vertical
combs while communication becomes gravity-referenced. However, the five seeds used here
are also the optimization panel, so the result should be treated as a candidate regime
rather than an independent robustness estimate. The next check is to rerun the best
high-success trial and the repeated seven-site cluster over a wider seed panel.

# Conclusion

The resource-distribution experiments support the idea that costly directional
communication is favored most strongly in an intermediate ecological regime: food must
be hard enough to make recruitment valuable, but discoverable enough for independent
finders to seed dances. The geometry experiments support the architectural-precondition
hypothesis in a more conditional way. A proportional vertical-comb advantage can
sometimes preserve supplied verticality, and supplied near-vertical geometry allows
gravity-referenced sender&ndash;receiver transposition to evolve, but the full
architecture-and-code transition is not yet robust.

The constrained coupling probe supports the sender&ndash;receiver coordination
hypothesis: with near-vertical geometry supplied, gravity-referenced communication
evolves in every condition, and correlated sender&ndash;receiver mutations shorten the
transition and reduce final mismatch. The long transition runs support the
transition-risk hypothesis under a stricter viability-scaled fitness rule. Proportional
vertical-comb advantages up to 25% do not bootstrap verticality from horizontal or
tilted starts under the baseline ecology, and vertical starts produce gravity-referenced
vertical outcomes only in a minority of seeds. The food-distribution probes suggest that
ecology can matter: a moderate-dense environment produced one stable
horizontal-to-vertical gravity-code transition in five seeds, and a local grid around
that result found four seed-101 parameter pockets that sustain the transition. But
simply making patches broader, richer, or higher-capacity mostly increases foraging
success without making the architecture-and-code transition robust. The robustness panel
reinforces this point: the best local pocket succeeds in only 2 of 11 seeds, and the
other pockets succeed only in seed 101. The first Optuna search found a stronger
six-site short-distance pocket, but a wider seed panel still succeeds in only 5 of 25
seeds. Trajectory comparison shows that unsuccessful seeds usually retain high foraging
success while remaining flat and direct-pointing. A subsequent extensive Optuna search,
which also varied mutation scales and sender&ndash;receiver coupling, found a stronger
nearby-patch regime in which 29 of 144 trials reached vertical gravity-code states in
all five optimization seeds. This suggests that reliable transitions may require
ecological opportunity and mutation structure to align: food distributions must make
recruitment valuable near vertical combs, while coupled sender&ndash;receiver changes
must keep the gravity code from lagging behind comb tilt. The next modeling priority is
independent validation of the best extensive-search settings over a wider seed panel,
followed by trajectory comparison against the earlier six-site pocket.

# References
