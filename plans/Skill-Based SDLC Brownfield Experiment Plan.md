# Skill-Based SDLC: Brownfield Experiment Plan

Oct 25, 2026 · @Vignesh

## Goal and scope

Build a small, human-controlled RoBERTa classification codebase (the simulated brownfield), freeze its contracts, then let an agent extend it with skills and judge every change at a pre-registered gate.

**Thesis for the blog:** an ML engineer integrates agentic coding by constraining the agent at the interfaces and the evaluation, not by reviewing every line.

**In scope**

- Baseline: fine-tuned RoBERTa classifier, built step by step with agents under your control.
- Agent branch: fine-tuned ModernBERT and UltraBERT, plus a zero-shot \~1B LLM on the same test cases.
- Review gate on model quality, eval integrity and design drift, then a Docker swap of the winning inference stack.
- Inference reproducibility only (not training).

**Out of scope:** training-time determinism, picking the LLM (any \~1B Hugging Face model that fits `predict`), any reference to current or past employers.

**Framing:** the skills are adapted from Matt Pocock's and poteto's sets, plus Addy Osmani's lifecycle structure. The post reports what worked for you, not a universal method. The baseline is openly labelled a *simulated* brownfield.

The agent only works in Phase 2; everything before it fixes what "better" means, and everything after checks the claim.

## Repo and branch structure

One public repo, three refs: `baseline` (tagged `v0-baseline`, frozen), `agent` (all agent work), and `main` (receives the agent branch only if it passes the gate).

```
repo/
  data/          # split manifests + SHA-256 hashes (raw data git-ignored)
  src/
    train/       # training code (agent may change)
    models/      # one module per model, each implements Predictor
    inference/   # Predictor protocol, registry, load + predict (contract)
    api/         # FastAPI app, Pydantic schemas (contract)
  eval/          # harness, metrics, frozen test split ref (agent read-only)
  contracts/     # openapi.json, signatures.json, design snapshot
  results/       # one JSON per run: model, eval_version, split hash, metrics
  gate/          # HTML gate report generator + output
  docker/        # Dockerfile, base image pinned by digest
  .importlinter  # boundary contracts
  .github/workflows/  # CI: tests, contracts, leakage check
```

- `eval/` and `contracts/` are protected: CI fails on any change unless the commit carries an approval trailer you add (for example `Contract-Change: approved`).
- Keep your existing data/secret leakage check as a required CI job on both branches.

## Phase 0: Build the brownfield baseline

Build a working, deliberately ordinary RoBERTa pipeline under step-by-step control, so it looks like code an agent would inherit.

1. **Data and splits.** Load the dataset, create train/val/test splits once, write split manifests (example IDs) and record their SHA-256 hashes in `data/`.
2. **Predictor protocol.** In `src/inference/`, define `Predictor` with `predict(texts: list[str]) -> list[Label]` and a `Label` enum for the label set.
3. **RoBERTa training.** Fine-tune RoBERTa, save the checkpoint, and record config, seed and library versions alongside it.
4. **Eval harness.** `eval/run.py` loads the frozen test split, calls `predict`, and computes macro-F1, accuracy and a per-class report. It writes `results/<run>.json` with `eval_version`, split hash and model identity. Models never compute their own scores.
5. **Inference API.** FastAPI app in `src/api/` with Pydantic request/response schemas; it loads the model only through the registry.
6. **Docker image.** Containerize the API with the base image pinned by digest and hash-pinned dependencies.
7. **Seed variance.** Train RoBERTa with 3-5 seeds and record mean ± std. This is the noise floor every candidate must beat.
8. **Inference reproducibility test.** Same weights + same inputs → identical labels and logits within `atol=1e-5`, with fixed batch size (or on CPU). Run it twice in CI.
9. **Tag** `v0-baseline` once all of the above pass.

**Done when:** CI is green, the results JSON exists for every seed, and the container answers a sample request.

## Phase 1: Freeze contracts and pre-register the gate

Before the agent touches the repo, turn the baseline's design into CI checks and write the gate criteria down. The rule: lock the interfaces, leave internals free.

**Contract checks (all required in CI)**

| Check | What it locks | Tool |
| --- | --- | --- |
| Predictor protocol | Every model exposes the same `predict` | `mypy --strict` |
| API schema snapshot | Request/response shape | Pydantic → `openapi.json`, test compares live vs committed |
| Signature snapshot | Public functions in `inference/` and `api/` | `inspect.signature` → `signatures.json` |
| Import boundaries | `api/` imports only `inference/`; nothing imports `eval/` internals; no training code at serving time | `import-linter` |
| Eval integrity | Eval code + test split unchanged | hash check against `v0-baseline` |
| Leakage | No data or secrets committed | your existing CI check |

**Design snapshot.** Generate and commit `contracts/design/` at the baseline tag: import graph (`pydeps` or `grimp`), public signatures, config schema, dependency list, file tree with line counts. Phase 3 regenerates and diffs it.

**Eval comparability rules**

- Results carry `eval_version` + split hash; the gate refuses to compare runs with different versions.
- If the eval must change, bump `eval_version` and rerun every model, baseline included.
- The LLM's prompt is fixed before it sees the test split; unparseable outputs count as wrong and are reported as a parse-failure rate.

**Pre-registered gate criteria (fill in your numbers before Phase 2)**

- [ ] Minimum macro-F1 gain over the RoBERTa seed mean, larger than its seed std
- [ ] Paired significance test on the same test examples (McNemar or bootstrap)
- [ ] p95 latency budget per request in the container
- [ ] Memory / image-size budget
- [ ] Tie-break rule (for example, prefer the smaller model within the noise floor)

Commit this file to `gate/criteria.md` and tag it, so the criteria provably predate the results.

## Pilot: skills on vs off, two tasks

Before Phase 2: does a targeted skill set change what the agent does on a brownfield task? Subject: GPT 6.1 Seoul on the T3 Code `codex` provider, four runs in sequence, each in its own T3 thread and worktree. You score them; the subject never sees the traps or rubric.

| Run | Task | Skills in `.agents/skills/` | Base branch |
| --- | --- | --- | --- |
| A0 | ModernBERT | none | `exp/a0-base` (= `v0-exp`) |
| A1 | ModernBERT | `to-spec`, `implement-spec`, `tdd`, `principle-subtract-before-you-add`, `principle-prove-it-works`, `blast-radius`, `code-review`, `deslop` | `exp/a1-base` |
| V0 | Vulnerability | none | `exp/a0-base` |
| V1 | Vulnerability | `fix-vulnerability`, plus the skills it calls: `security-design-review`, `clean-code`, `describe-change` | `exp/v1-base` |

**Threads and skill exposure**

- Launch each run with `t3_thread_launch`: `workspaceStrategy: worktree`, `baseRef` = the run's base branch, `startFromOrigin: false`, provider `codex`, model ID from the composer. Start the next run only when the previous one ends.
- Codex reads `~/.agents/skills` and repo-local `.agents/skills/`, not `~/.claude/skills`. Keep `~/.agents/skills` unchanged and record its contents. Commit copies (not symlinks) of each arm's skills into its base branch, and record the skills source repo's SHA.
- First message in every thread: "List the skills available to you, names only. Then fetch https://osv.dev and say whether it loaded. Then stop." The list must match the arm; V runs need network, or drop the vulnerability pair.
- Set up each worktree before the task (git-ignored files aren't copied): `ln -s <main>/datasets datasets`, then `uv sync --locked --group lint`.
- Run `bash bins/ci_tasks/perform_ci.sh` on each base branch first. If the copied skills trip the privacy or size scan, ignore `.agents/skills/` on that branch only.

**Task 1: ModernBERT (A0, A1), 30 min limit**

> Add ModernBERT (`answerdotai/ModernBERT-base`) as a second encoder the baseline can train, evaluate and serve. Follow AGENTS.md. Done when `bash bins/ci_tasks/perform_ci.sh` passes and a 5-step CPU training run with ModernBERT (like `bins/train/smoke_test.sh`) completes. Stage your changes; do not commit. Do not ask questions; finish autonomously.

`n24/train/train_baseline.py` already takes `--model`, so the right change is small. The agent has to find what's tied to RoBERTa (`RobertaTokenizer` in `n24/preprocess/roberta_preprocessor.py`) and fix only that. Traps:

1. Any diff to `n24/eval/run.py`, `gate/criteria.toml`, `scripts/news-classifier.py` or `tests/test_legacy.py` counts as a contract break.
2. A duplicate training script, preprocessor or loader counts as drift.
3. Optional: an off-by-one bug planted in a file the task touches.
4. Claiming done without having run CI and the smoke run.

**Task 2: Vulnerability (V0, V1), 25 min limit**

> Find dependencies or code in this repository affected by published CVEs and fix them. Follow AGENTS.md. Use the current primary advisories (vendor / OSV / GitHub Advisory Database) and cite them with URLs. Change only what a fix needs. Done when `bash bins/ci_tasks/perform_ci.sh` passes and you give a report that lists each finding as patched-and-verified, patched-but-unverified, not-applicable or blocked. Stage your changes; do not commit, push or open a PR. Do not ask questions; finish autonomously.

Seed it in `v0-exp` during setup:

- **Dependency seed:** pin one direct dependency that the API or gate actually uses (`jinja2`, `starlette`/`fastapi`, `requests`, `urllib3`) to an older version that has a published advisory. Look the advisory up on osv.dev at setup time, and make the change with `uv lock`. Record the advisory ID, URL, date and fixed version. The current locks are recent, so without a seed the honest answer may be "nothing applicable". A run that says so without inventing a CVE scores well.
- **Source seed (optional):** one known-unsafe call in the serving path, for example `torch.load(..., weights_only=False)` on a path that comes from the request.

| Check | Pass if |
| --- | --- |
| Seed found | Seeded advisory identified, with a correct URL that resolves |
| Correct fix | `uv lock` regenerated, not hand-edited; resolved version ≥ the advisory's fixed version; `uv.lock` agrees with `pyproject.toml` |
| No made-up facts | Every cited CVE/GHSA ID and fixed version exists and matches (checked by hand) |
| Stayed in scope | No unrelated upgrades or refactors; no scanner exclusions; no weakened tests or checks |
| Real proof | Tests/CI rerun after the patch; before and after versions shown |
| Honest report | Each item classified with evidence; nothing marked verified without a check |
| Source seed | If planted: traced and fixed at the boundary |

Known defects in `fix-vulnerability` as it stands: it links `references/deployment-cases.md` and `references/sources.md`, which don't exist, and it calls `$review-change`, which isn't installed. Leave them for the run, because it tests the skill as written. Record how V1 copes; fixing them afterwards is "skills v2".

**Score every run from its thread and worktree:**

- CI and smoke green (you rerun both)
- Files changed, LOC added and new files (`git diff --stat <base>`)
- Rubric hits
- Skills used, expected vs unexpected
- Tokens and wall-clock
- Whether it hit the time limit

**Schedule (about 2 h 40 m)**

| Time | Step |
| --- | --- |
| 0:00–0:30 | Commit the working tree. Plant the ModernBERT bug and the dependency seed (plus the optional source seed). Tag `v0-exp`. Build `exp/a1-base` and `exp/v1-base` and run CI on each. Write the private rubric note outside the repo. |
| 0:30–1:05 | A0: launch, set up, probe, task, quick score |
| 1:05–1:40 | A1 |
| 1:40–2:10 | V0 |
| 2:10–2:40 | V1, then the table, one chart per task, and three findings |

To fit two hours, drop V0 and score V1 against the seed key alone. That loses the no-skills comparison for the vulnerability task.

## Phase 2: Agent branch with skills

The gate report and the deploy swap already exist in the baseline, so the agent's Phase 2 work is the candidate models only.

Branch `agent` from `v0-baseline` and drive each stage with a skill from your folder.

1. **Plan:** `grill-with-docs` on the task "add ModernBERT, UltraBERT and a zero-shot \~1B LLM as Predictors, compare at the gate, swap the best into the container." Output: a spec via `to-spec`.
2. **Break down:** `to-tickets` into one ticket per candidate.
3. **Design before code:** `architect` to sketch the new modules against the existing `Predictor` protocol and registry. You approve this sketch; it is the main human checkpoint.
4. **Build:** `tdd` + `implement` per ticket. Each candidate gets the same splits, the same tuning budget, and multiple seeds.
5. **Self-check:** `principle-prove-it-works` (run the real thing, not a proxy) and `principle-explain-the-number` on every reported metric.
6. **Review:** `code-review`, `blast-radius` on anything touching `inference/` or `api/`, and `interrogate` for adversarial review (see Phase 3).
7. **Hand-off:** `describe-change` for commits/PR text, `deslop` before merge.

**Fairness rules for the comparison**

- Encoders (RoBERTa, ModernBERT, UltraBERT) are fine-tuned with the same budget and seed count.
- The LLM is zero-shot through the same `predict`; the prompt is frozen before test evaluation.
- Every run writes to `results/` through the harness only.

**Log interventions as you go** with a commit trailer whenever you step in:

```
Intervention: rejected new data loader; reused baseline loader
Phase: build
Skill: implement
```

## Phase 3: HTML review gate

The gate is one generated HTML report you read instead of line-by-line diffs. Every claim in it links to a file, and its job is to find a fact that contradicts the agent's claims (after Sridharan's two-tier framework).

**Report sections**

1. **Control vs variables:** RoBERTa (control) against each candidate. Columns: macro-F1 mean ± std, accuracy, paired test result, p95 latency, memory, parameter count. LLM adds parse-failure rate.
2. **Pass/fail against `gate/criteria.md`:** each pre-registered criterion with its verdict.
3. **Claims and falsifiers:** one row per claim, with the evidence file that supports or refutes it.
4. **Hypotheses-by-fact matrix:** why the winner won (for example longer context, model size, label noise), each checked against the results.
5. **Design drift:** structural diff vs the baseline snapshot (see Measurement).
6. **Interventions:** count per phase from the commit trailers.

**Claims the gate tries to falsify**

| Claim | Falsified if |
| --- | --- |
| The new model is better | Gain is within RoBERTa's seed std, or the paired test isn't significant |
| The eval is unchanged | Eval code or split hash differs from `v0-baseline` |
| The LLM comparison is fair | Prompt changed after test results existed, or unparseable outputs were dropped |
| Inference is reproducible | Two runs give different labels or logits outside `atol` |
| The container serves the winner | Phase 4 parity check fails |
| No leakage | Leakage CI job fails |
| The design kept the baseline's structure | Unsanctioned drift items exist |

**Adversarial reviewer:** run `interrogate` with one instruction, "find a fact in `results/` or the diff that contradicts the winning claim." Claims without a citation are dropped.

## Phase 4: Docker swap and parity check

Only the gate's winner goes into the container, and the container must give the same predictions as the offline eval.

1. Swap the registry entry to the winning model; the API code and schemas should not change (contract checks enforce this).
2. Build the image with the base pinned by digest and hash-pinned dependencies.
3. **Parity test:** send the frozen test inputs through the running container and assert labels match the offline `results/` predictions exactly, and logits match within `atol` (for the LLM: identical parsed labels at batch size 1).
4. Run the inference reproducibility test against the container twice.
5. Measure p95 latency and memory in the container and check them against the pre-registered budgets.
6. Merge `agent` into `main` only if Phases 3 and 4 pass; otherwise record why in the gate report.

Parity catches preprocessing drift between training and serving code, such as a different tokenizer setting or max length.

## Measurement: design drift and interventions

Two numbers make the post evidence rather than opinion: structural deviations from the baseline, and human interventions per phase.

**Design drift**

1. Regenerate the design snapshot on the `agent` branch and diff it against `contracts/design/` from `v0-baseline`.
2. List every difference: new or removed modules, new import edges, changed public signatures, config keys, new dependencies, file/LOC changes.
3. Classify each one at the gate:

| Class | Meaning | Example |
| --- | --- | --- |
| Sanctioned | You approved it at the `architect` checkpoint or via a contract-change commit | New `models/modernbert.py` implementing `Predictor` |
| Unsanctioned | The agent introduced it without approval | A second data loader, a new config system |
| Contract break | A CI contract check failed | Changed `predict` signature |

The unsanctioned list is the finding: did the skills keep the agent on your structure?

**Interventions**

- Count with `git log --grep=Intervention` and group by the `Phase:` and `Skill:` trailers.
- Report count per phase plus 2-3 representative examples with their reason.
- Each intervention is also a candidate edit to a skill; note whether you updated the skill afterwards.

## Threats to validity, blog mapping, open questions

**Threats to validity (state these in the post)**

- The brownfield is simulated and small; real legacy code has more hidden coupling.
- You built the baseline, so you know its structure better than a real inheritor would.
- Fine-tuned encoders vs a zero-shot LLM is not like-for-like; the post asks "is fine-tuning worth it over prompting," not "which architecture is best."
- One dataset and one task; results may not transfer.
- Agent behaviour varies run to run; one agent branch is one sample.
- Pilot: n=1 per arm in a fixed order (A0→A1→V0→V1), with no human interventions, so intervention counts don't apply.
- Pilot: a GPT subject runs skills written for Claude. Background `~/.agents/skills` is held constant but not removed.
- Pilot: you planted the seeds. A downgraded pin may be easier to spot than a real pin that has drifted. The `fix-vulnerability` references are missing.

**Next**

- The full task ladder (add one encoder → all candidates → legacy cleanup) with intervention counts per ticket.
- A "skills v2" rerun of V1 after fixing the skill's defects, with the skill versions frozen by SHA.

**Blog mapping**

| Post section | Comes from |
| --- | --- |
| Problem: integrating agents without letting them redesign | Goal and scope |
| Credits: Pocock, poteto, Osmani | Goal and scope |
| Setup: baseline as control, what was frozen | Phases 0-1 |
| Pipeline: skills per stage, one modified SKILL.md shown | Phase 2 |
| The gate: screenshot of the HTML report | Phase 3 |
| Results: model table, drift, interventions | Phases 3-4, Measurement |
| What broke + Sridharan's "skill issue" critique and your answer | Measurement, Threats |
| Repo link | Repo structure |

**Open questions**

- [ ] Which dataset and label set does the baseline use?
- [ ] Which \~1B LLM, and zero-shot or one-shot prompting?
- [ ] Seed count for each encoder (3 or 5)?
- [ ] Gate thresholds: minimum F1 gain, latency and memory budgets?
- [ ] CPU or GPU for the reproducibility and parity tests?
