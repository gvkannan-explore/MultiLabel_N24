# N24News classification

A learning project that classifies N24News articles into news sections, starting from a guarded copy of its legacy code.

## Code history

**Baseline branch**:
The legacy code plus guardrails, used as the starting point for later work.
_Avoid_: baseline, brownfield branch

**Baseline model**:
The first classifier that later models are compared against.
_Avoid_: baseline

**Legacy code**:
The code as it stood at the last commit before guardrails were added.
_Avoid_: old code, brownfield code

**Section**:
One of the 24 N24News news sections an article belongs to.
_Avoid_: category, class, label, topic

## Guardrails

**Guardrail**:
An automated check that blocks a commit, push, or CI run.
_Avoid_: gate, policy

**Privacy finding**:
A personal path, real email, known local identity, or private file in Git content or metadata.
_Avoid_: PII hit, leak

**Anonymous identity**:
The shared `Contributor` author and committer used for every commit.
_Avoid_: bot user, placeholder author

**History rewrite**:
Replacing past commits to remove privacy findings.
_Avoid_: history cleanup, scrub
