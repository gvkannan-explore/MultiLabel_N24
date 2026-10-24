# CI and privacy checks

The checks run on CPU. They never train models, download pretrained weights, or
contact remote services other than the pinned Gitleaks download.

## Setup

Run these commands once in every clone and every Git worktree. The hooks
directory, the identifier profile, and the Gitleaks binary are local to each
worktree, so setting up one worktree does not cover another.

```bash
uv sync --locked --group lint
bash scripts/install-gitleaks.sh
bash scripts/setup-hooks.sh
bash runs/perform_ci.sh
```

The Gitleaks installer pins a release and checks its published checksum. CI pins
its GitHub actions to commit SHAs, has read-only repository permissions, and uses
the locked Python environment.

`setup-hooks.sh` saves the local Git identity and machine username in the ignored
`.privacy-identifiers.json` file with owner-only permissions. It then sets a
repository-local anonymous author and committer, `Contributor
<contributor@users.noreply.github.com>`. Global Git settings are unchanged.
Pre-commit blocks environment variables that override this identity.

## Gates

- Pre-commit checks the author and committer identity, scans the complete staged
  blobs for privacy findings and secrets, and limits staged maintained files to
  20,000 bytes. Lockfiles and generated or vendored files are exempt.
- Commit-msg checks the commit message for personal information. This includes
  email addresses in trailers, so a `Co-Authored-By` line with a real address is
  rejected.
- Pre-push checks every outgoing ref's reachable history, then runs local CI.
  Deleting a remote ref skips the checks.
- CI checks working and staged files, full history, hook regressions, Ruff lint
  and format, Pyright, offline unit tests, and shell syntax. Every check runs
  even if an earlier one fails, and any failure fails the run.

The privacy scanner rejects personal home-directory paths, non-example emails,
known local identity strings, and private file locations. Findings print rule
names, short commit IDs, and file-path hashes instead of the matched data.
Gitleaks prints redacted results. The privacy hook rejects force-staged datasets,
run outputs, audit logs, environment files, and private keys.

The only allowed emails are the anonymous contributor address and the reserved
example domains used in test fixtures. Names are matched against locally known
identifiers, not detected in general. CI cannot read the ignored identifier
file, so names in prose still need review there.

## Scope of the quality gates

Ruff and Pyright cover `src/`, `scripts/*.py`, and `tests/`, including the legacy
code. Lint fixes to legacy code may change style and types but not logic (see
[ADR 0001](adr/0001-legacy-behavior-frozen.md)). Where Pyright would need a logic
change, the line carries a narrow `# pyright: ignore[rule]` with a reason.

Notebooks are outside Ruff and Pyright. Reformatting a notebook restages the
whole file, and several notebooks are above the 20,000-byte limit. Notebooks are
still covered by the privacy and secret scans.

## History

History was rewritten before guardrails were added (see
[ADR 0002](adr/0002-history-rewritten-before-baseline.md)). Every commit uses the
anonymous identity, and the full-history privacy scan has no allowlist. Hooks
protect new work but can be bypassed, so make the CI check required in branch
protection.

## Merging

Merge by fast-forwarding `main` from your machine, not with GitHub's merge,
squash, or rebase buttons. Those buttons create commits whose author is your
GitHub account, which the history scan rejects on every later run.

```bash
git switch main
git merge --ff-only <branch>
git push origin main
```

The pre-push hook runs the history scans and local CI before the push. GitHub
marks the pull request as merged once `main` contains its commits. Pull request
CI checks out the branch's own head commit for the same reason: GitHub's
default checkout for pull requests is a temporary merge commit with your account
as author.

The `main` ruleset should require the `checks` status and linear history, block
force pushes and deletions, and leave "Require a pull request before merging"
off. A pushed commit passes the status rule when CI already succeeded on that
commit, which is true after the branch's own push run.
