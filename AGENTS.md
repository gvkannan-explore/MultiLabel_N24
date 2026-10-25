# Agent instructions

## Git

- Stage finished changes with `git add`, but do not commit unless the user asks for a commit at that moment. A plan that lists commits is not that request.
- Do not push. The user pushes.

## Baseline branch

- Legacy code behavior is frozen (docs/adr/0001-legacy-behavior-frozen.md). Changes to legacy code are limited to formatting, imports, and annotations; `tests/test_legacy.py` pins its current behavior.
- Use the terms in GLOSSARY.md.
- A change is done when `bash runs/perform_ci.sh` passes.

## Code

- Inline logic used once; extract a function when it has a second caller or is independently testable.
- Let exceptions propagate. Catch only a specific exception you can handle at that point.
- Type everything precisely; use `typing.Any` only where a library forces it.
- Validate unknown values once, at the boundary that owns them, then pass typed values inward. Trust guarantees from schemas, constructors, and types without rechecking them.
- Build the simplest design that fully meets current requirements and will last: no speculative abstractions or configuration, and no stopgaps meant to be replaced.
    - Before adding complexity for an unlikely race or security edge case, explain the failure mode, likelihood, and complexity cost to the user and get their buy-in.
- Grow the system in layers: start from the smallest version that works end to end, and add each capability on top of a working product.
- Prefer established, well-maintained libraries over reimplementing common functionality.
