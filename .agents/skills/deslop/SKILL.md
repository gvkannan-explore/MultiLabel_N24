---
name: deslop
description: Remove AI-generated code slop from the current branch's diff and match the surrounding style. Use before committing or opening a PR. For prose, use unslop.
---

# Remove AI code slop

Diff the branch against its base. Use `origin/HEAD` if it exists, otherwise the repository's default branch. If the branch is the base branch, diff against the last commit or the staged changes instead. Remove slop the branch introduced.

## What to remove

- Comments that restate the code or clash with the file's comment style. `// increment i` above `i++` goes.
- Defensive checks and try/catch blocks on trusted internal paths. Keep validation at real boundaries such as user input, network, and files.
- Casts to `any` added only to silence the type checker. Fix the type instead.
- Nesting that early returns would flatten.
- Names, helpers, or patterns that differ from the rest of the file.

## Guardrails

- Keep behavior unchanged unless fixing a clear bug.
- Prefer small edits over rewrites.
- Finish with a 1 to 3 sentence summary of what you removed.
