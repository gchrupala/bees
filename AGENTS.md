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

## Token-Conservative Workflow

- Start with a narrow discovery pass: read only directly relevant files, nearby tests, and scoped git history.
- Use targeted commands such as `rg`, `rg --files`, and line ranges instead of dumping large files.
- Run focused tests/checks first; use full suites only for broad or shared-behavior changes.
- Summarize long command output unless raw output is explicitly requested.
