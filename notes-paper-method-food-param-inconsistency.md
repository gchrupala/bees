# Latent inconsistency: Method section food parameters

The Method section of `report/paper.tex` (the `\subsection{Foraging environment}`,
around line 264) states that food angular width ($0.2$), value ($v=1$), and
capacity ($6$) are "held fixed across our experiments."

This is inconsistent with what the experiments actually do. **Only food value
($v = 1$) is truly fixed.** Food-site count, angular width, capacity, maximum
distance, and travel cost are all varied:

- searched by the Optuna pipeline (`experiments/optimize_food_transition.py`),
- swept in the food-distribution experiment (`experiments/run_food_distribution_v2.py`),
- perturbed in the one-parameter sensitivity experiment.

The drafted `report/experimental_setup.tex` correctly treats width/capacity as
varied, so the Method sentence contradicts the setup section.

## Fix

When next editing the paper's Method section, narrow that sentence so only
$v = 1$ is described as fixed, and note that the other food parameters are varied
per experiment. Per the paper-editing rules, confirm the textual change with the
author before editing.
