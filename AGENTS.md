# Agent instructions

## Git

- Stage finished changes with `git add`, but do not commit unless the user asks for a commit at that moment. A plan that lists commits is not that request.
- Do not push. The user pushes.

## Baseline branch

- Legacy code behavior is frozen (docs/adr/0001-legacy-behavior-frozen.md). Changes to legacy code are limited to formatting, imports, and annotations; `tests/test_legacy.py` pins its current behavior.
- Use the terms in GLOSSARY.md.
- A change is done when `bash bins/ci_tasks/perform_ci.sh` passes.

## Layout

- `scripts/` holds tooling that sits outside the package: CI checks, git hooks, and setup scripts.
- `bins/ci_tasks/` holds the shell entry points you run by hand or from CI: the CI check, data preparation, the smoke test, and Modal training.
- `experiment_runs/` holds training output and the local MLflow store. It is git-ignored; never track anything in it.
- Code that is part of the project workflow (data preparation, training, evaluation, the deployment gate) lives in `src/` and runs from the repository root with `uv run python -m src.<package>.<module>`.
- Exception: the legacy `scripts/news-classifier.py` stays where it is, because legacy code is frozen (ADR 0001) and `tests/test_legacy.py` loads it by path.

## Code

- Inline logic used once; extract a function when it has a second caller or is independently testable.
- Let exceptions propagate. Catch only a specific exception you can handle at that point.
- Type everything precisely; use `typing.Any` only where a library forces it.
- Validate unknown values once, at the boundary that owns them, then pass typed values inward. Trust guarantees from schemas, constructors, and types without rechecking them.
- Build the simplest design that fully meets current requirements and will last: no speculative abstractions or configuration, and no stopgaps meant to be replaced.
    - Before adding complexity for an unlikely race or security edge case, explain the failure mode, likelihood, and complexity cost to the user and get their buy-in.
- Grow the system in layers: start from the smallest version that works end to end, and add each capability on top of a working product.
- Prefer established, well-maintained libraries over reimplementing common functionality.
- When printing to the terminal, use rich.print or logging.info whenever appropriate.
