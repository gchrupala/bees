# Project Instructions

We are modeling the evolution of bee communication.

## Modeling Principles

- Start from minimal, inspectable models; add biological realism deliberately
  rather than reaching for complex abstractions.
- Treat communication strategies, costs, benefits, and environmental conditions
  as parameters where practical; avoid hard-coding conclusions into the model.
- Keep scientific assumptions explicit and easy to revise, and document them (in
  code, configs, or experiment notes) when they affect outcomes.
- Make experiments reproducible from configuration and random seeds.

## Code Style

- Use Python unless the project later establishes a different stack.
- Keep simulation logic separate from plotting, notebooks, and experiment scripts.
- Prefer clear data structures, named parameters, and small functions over opaque
  numeric constants and large monolithic code.
- Add focused tests for model rules, evolutionary updates, and edge cases.

## Environment & Dependencies

- Dependencies are declared in `pyproject.toml` (there is no `requirements.txt`).
  Add or change dependencies there, not with ad-hoc `pip install` of extra
  packages.
- Always work inside a virtual environment; the repo convention is a
  project-local `.venv`. Requires Python >=3.11.
- The code uses a `src/bees` layout, so an **editable install is required** for
  `import bees` and the console scripts to resolve. From the repo root:
  `python -m pip install -e .`. Re-run it after changing dependencies. A missing
  editable install is the usual cause of `ModuleNotFoundError: bees`.
- On Snellius, do not install into the system Python: `module load` a Python
  toolchain, then create/activate a venv and `pip install -e .` in it. The
  `.sbatch` scripts activate `.venv` by default; override with `BEES_VENV` (venv
  path), `BEES_PYTHON` (interpreter), and `BEES_MODULE_LOAD` (module(s) to load)
  when the setup differs.

## Collaboration and candor

- Give honest, direct assessments. Push back when there is a good reason —
  flawed ideas, weak results, bad designs — including when the idea is the user's
  or your own. State the problem plainly and say what you would do instead.
- Do not optimize for agreement, especially during brainstorming. The user wants
  the best answer, not validation; agreeing to be agreeable wastes their time.
- Do not hedge or over-diplomatically soften a negative verdict. If something is
  bad, say it is bad and why. Reserve enthusiasm for things that earn it, so it
  stays informative.

## Workflow

- Wait for an explicit request before generating code; do not implement based
  only on high-level discussion.
- Consult relevant git history, scoped to the affected files, before substantive
  changes.
- Regularly commit changes to keep the project history current.
- Update documentation when model behavior, experiment parameters, or usage changes.
- Stream progress from long-running probes: use unbuffered Python (`python -u`),
  explicit `print(..., flush=True)`, and write/flush raw result rows incrementally
  so partial CSV outputs survive a crash or session termination. Summary files can
  still be generated at the end.
- Launch unattended long jobs so they survive terminal/tool-session termination —
  e.g. a persistent `systemd-run` user service — rather than relying on an
  interactive session.

## Snellius

- Use `ssh gchrupala1@snellius.surf.nl` for Snellius access. Do not record
  passwords, private keys, or other secrets in this repo.
- The project checkout on Snellius is `/gpfs/home2/gchrupala1/bees`.
- Snellius uses Slurm. Submit the evolutionary interaction array from the
  remote checkout with `./experiments/submit_evolutionary_interaction_snellius.sh`;
  monitor with `squeue`, and inspect the `logs/slurm-*.out` / `logs/slurm-*.err`
  logs (the `.sbatch` files write there; `logs/` is gitignored except `.gitkeep`).
- The submit helper accepts `BEES_ARRAY_TASKS`, `BEES_ARRAY_CONCURRENCY`,
  `BEES_VENV`, `BEES_PYTHON`, and `BEES_PUSH`. Set `BEES_PUSH=1` when the
  finalizer should commit and push merged result CSVs after the array succeeds.
- **Slurm does not inherit the submitting shell's environment on this cluster.**
  A plain `sbatch job.sbatch` runs with a clean environment, so any `BEES_*`
  variable the *job* reads (`BEES_PUSH`, per-task trial counts, output-path
  overrides, `BEES_CONFIG`) silently falls back to its default unless you pass
  it explicitly. Always submit job-side variables with
  `sbatch --export=ALL,VAR=value,... job.sbatch` (or bake them into the
  `.sbatch`); variables consumed only by the submit *script* itself (e.g.
  `BEES_ARRAY_TASKS`, `BEES_FRESH_JOURNAL`) do not need this. A launch that ran
  the default 64 instead of the requested 96 trials/task and skipped the result
  push traced to exactly this — the disk optuna submit helper now hard-codes
  `--export=ALL`.

### Keep results in sync via git

Both checkouts have drifted before, leaving result CSVs on only one side. Treat
git as the single source of truth for results:

- Commit and push run outputs from the Snellius checkout after each run (or set
  `BEES_PUSH=1`); never leave regenerated results uncommitted on the cluster.
- `git pull --rebase` before new work on either side; never `scp`/`rsync` results
  between machines instead of committing them.
- GitHub rejects blobs >100MB — store oversized trajectory outputs gzipped
  (`results/*.csv.gz`, whitelisted in `.gitignore`).
- If a pull is blocked by uncommitted or untracked results, reconcile via git
  (commit, or confirm byte-identical to what is tracked before removing) rather
  than deleting data blindly.

## Reports

- The working report is `report/report.md`, rendered to `report/report.html`
  with `python -u experiments/render_report_html.py`. Rebuild the HTML before
  finishing if Pandoc is available; if it is unavailable, say so clearly.
- Keep the workflow lightweight: prefer Markdown, tracked CSV-backed summaries,
  and simple generated HTML over heavyweight notebook or PDF pipelines.
- For static report figures from tabular results, prefer `plotnine` over raw
  `matplotlib` unless lower-level plotting control is needed.
- Do not compile or expand `report/paper.tex` during ordinary report maintenance;
  it is the publication paper (see Paper) and is touched only on explicit request.

## Paper

- The publication paper is `report/paper.tex` and its related files. Only touch
  it when explicitly asked.
- When editing paper text, maintain the existing writing style and the structure
  of the argument; confirm important textual changes before editing.
- The writing in the paper should focus on the final version of the model and experimental setup. There should be no references to previous versions or superseded results.
- The paper should use present tense, unless there is a good reason not to.
- Avoid excessive em-dashes (`---`) in the paper or report prose. Recast with
  commas, colons, semicolons, or parentheses instead. (En-dashes, `--`, for
  ranges and compounds such as `$80$--$91$` or `sender--receiver` are fine.)
- While writing, make sure that the paper writing style and personality matches the description in @style.md


### Data and visualizations
- For the display of quantitative data (when it's useful to show spread), prefer figures to tables
- Use clean, uncluttered design for figures. Explanatory text should be in the caption, not figure title or embedded in the figure. Labels for key elements in a figure and legends are OK. 
- If there is text in a figure, it should be in a large and readable font.


## Token-Conservative Workflow

- Start with a narrow discovery pass: read only directly relevant files, nearby
  tests, and scoped git history.
- Use targeted commands such as `rg`, `rg --files`, and line ranges instead of
  dumping large files.
- Run focused tests/checks first; use full suites only for broad or
  shared-behavior changes.
- Summarize long command output unless raw output is explicitly requested.
