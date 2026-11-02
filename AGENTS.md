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
- `bins/` holds the shell entry points, one folder per purpose: `ci_tasks/` only for what CI runs (the CI check), `data/` for data preparation, `train/` for the smoke test and Modal training, and `api/` for serving and querying the inference API. Scripts you run by hand never go in `ci_tasks/`.
- `zero-shot/` is a separate uv project for the Modal image of `n24/eval/zero_shot.py`. vLLM pins torch and transformers below the root `uv.lock`, so it has its own lock. It holds no code.
- `experiment_runs/` holds training output and the local MLflow store. It is git-ignored; never track anything in it.
- Code that is part of the project workflow (data preparation, training, evaluation, the deployment gate) lives in `n24/` and runs from the repository root with `uv run python -m n24.<package>.<module>`.
- Exception: the legacy `scripts/news-classifier.py` stays where it is, because legacy code is frozen (ADR 0001) and `tests/test_legacy.py` loads it by path.

## Code

- Write a failing test first, then the code that makes it pass, unless told otherwise. Exception: `tests/test_legacy.py` pins existing legacy behavior, so its tests describe code that already exists.
- Build the simplest design that fully meets current requirements and will last: no speculative abstractions or configuration, and no stopgaps meant to be replaced.
    - Before adding complexity for an unlikely race or security edge case, explain the failure mode, likelihood, and complexity cost to the user and get their buy-in.
- Before adding a script, command, or workflow, inspect existing entry points, their callers, options, and side effects. Extend the existing workflow when the purpose is the same. Smoke/full runs, CPU/GPU selection, artifact downloads, and staging are usually options on that workflow.
- Add a new entry point only for a distinct responsibility; explain why the existing entry point cannot reasonably support it. Keep execution logic in one place and have orchestration call that implementation. Preserve existing defaults when adding options.
- Inline logic used once; extract a function when it has a second caller or is independently testable.
- Let exceptions propagate. Catch only a specific exception you can handle at that point.
- Delete dead code in the code you change. Ask before deleting code outside the task, and never delete legacy code.
- Name things descriptively, for clarity over brevity: `wait_until_observation_is_saved=True`, not `wait=True`. Don't rename existing API endpoints.
- Type everything precisely; use `typing.Any` only where a library forces it.
- Validate unknown values once, at the boundary that owns them, then pass typed values inward. Trust guarantees from schemas, constructors, and types without rechecking them.
- Grow the system in layers: start from the smallest version that works end to end, and add each capability on top of a working product.
- Prefer established, well-maintained libraries over reimplementing common functionality.
- When printing to the terminal, use rich.print or logging.info whenever appropriate.
