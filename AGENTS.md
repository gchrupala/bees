# Project Instructions

We are modeling the evolution of bee communication.

## Goals

- Build computational models of how communication strategies evolve.
- Prefer simple, inspectable simulations before complex abstractions.
- Keep scientific assumptions explicit and easy to revise.
- Make experiments reproducible from configuration and random seeds.

## Code Style

- Use Python unless the project later establishes a different stack.
- Keep simulation logic separate from plotting, notebooks, and experiment scripts.
- Prefer clear data structures and named parameters over opaque numeric constants.
- Add focused tests for model rules, evolutionary updates, and edge cases.
- Write simple module code, with small functions. Avoid overengineering and gratuuitous complexity.

## Scientific Assumptions

- Document assumptions in code, configs, or experiment notes when they affect outcomes.
- Avoid hard-coding conclusions into the model.
- Treat communication strategies, costs, benefits, and environmental conditions as parameters where practical.
- Start with minimal models, then add biological realism deliberately.

## Workflow

- Do not implement only based on discussing high level ideas. Wait for an explicit request to generate code.
- Before large changes, inspect the existing structure and follow established patterns.
- Consult relevant git history before substantive changes, scoped to affected files when possible.
- Keep changes scoped to the current modeling question or tooling need.
- Run the relevant tests or checks after implementation when available.
- Update documentation when model behavior, experiment parameters, or usage changes.
- Regularly commit changes to the git repo to keep track of the evolving project.
- For long-running experiment probes, stream progress/results as they are
  produced: use unbuffered Python (`python -u`), explicit `print(...,
  flush=True)`, or scripts that flush CSV rows. Avoid long buffered one-liners
  that hide all output until the command exits.
- Long-running experiment scripts that collect per-seed or per-condition results
  must write raw result rows incrementally and flush file handles after each
  completed unit of work. Final summary files can still be generated at the end,
  but partial raw CSV outputs should survive session termination or worker
  crashes.
- Launch unattended long jobs in a way that is not susceptible to terminal or
  tool-session termination. Use a persistent process manager such as a user
  `systemd-run` service when available; do not rely on an interactive tool
  session for work that needs to continue after the current turn.

## Snellius

- Use `ssh gchrupala1@snellius.surf.nl` for Snellius access. Do not record
  passwords, private keys, or other secrets in this repo.
- The project checkout on Snellius is `/gpfs/home2/gchrupala1/bees`.
- Before launching Snellius jobs, sync the remote checkout with GitHub, e.g.
  `cd /gpfs/home2/gchrupala1/bees && git pull --rebase`.
- Snellius uses Slurm. Submit the evolutionary interaction array from the
  remote checkout with `./experiments/submit_evolutionary_interaction_snellius.sh`;
  monitor with `squeue`, and inspect `slurm-*.out` / `slurm-*.err` logs in the
  checkout.
- The submit helper accepts `BEES_ARRAY_TASKS`, `BEES_ARRAY_CONCURRENCY`,
  `BEES_VENV`, `BEES_PYTHON`, and `BEES_PUSH`. Set `BEES_PUSH=1` when the
  finalizer should commit and push merged result CSVs after the array succeeds.

## Reports

- The working report is `report/report.md`, rendered to `report/report.html`
  with `python -u experiments/render_report_html.py`.
- When editing the working report or report rendering code, rebuild
  `report/report.html` before finishing if Pandoc is available. If Pandoc is
  unavailable, say so clearly.
- Keep the HTML workflow lightweight: prefer Markdown, tracked CSV-backed
  summaries, and simple generated HTML over heavyweight notebook or PDF
  pipelines.
- For static report figures from tabular experiment results, prefer `plotnine`
  over raw `matplotlib` unless lower-level plotting control is needed.
- `paper.tex` is an empty LaTeX paper scaffold with section structure only.
  Do not compile or expand it during ordinary report maintenance unless the
  user explicitly asks for a LaTeX/PDF paper artifact.
- The old LaTeX table fragments under `report/tables/` and
  `report/figures/` may still be regenerated with
  `python -u experiments/run_report_artifacts.py artifacts` when needed for
  the legacy paper path, but they are not the primary working report output.

## Token-Conservative Workflow

- Start with a narrow discovery pass: read only directly relevant files, nearby tests, and scoped git history.
- Use targeted commands such as `rg`, `rg --files`, and line ranges instead of dumping large files.
- Run focused tests/checks first; use full suites only for broad or shared-behavior changes.
- Summarize long command output unless raw output is explicitly requested.
