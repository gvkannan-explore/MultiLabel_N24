# Skill-Based SDLC: Brownfield Experiment Plan

Oct 25, 2026 · @Vignesh

## Goal and scope

Build a small, human-controlled RoBERTa classification codebase (the simulated brownfield), freeze its contracts, then let an agent extend it with skills and judge every change at a pre-registered gate.

**Thesis for the blog:** an ML engineer integrates agentic coding by constraining the agent at the interfaces and the evaluation, not by reviewing every line.

**In scope**

- Baseline: fine-tuned RoBERTa classifier, built step by step with agents under your control.
- Agent work: fine-tuned ModernBERT and UltraBERT (built in the pilot, trained before the gate), plus a one-shot or revised prompt for the baseline's \~1B zero-shot LLM (Phase 2), all on the same test cases.
- Review gate on model quality, eval integrity and design drift, then a Docker swap of the winning inference stack.
- Inference reproducibility only (not training).

**Out of scope:** training-time determinism, picking the LLM (any \~1B Hugging Face model that fits `predict`), any reference to current or past employers.

**Framing:** the skills are adapted from Matt Pocock's and poteto's sets, plus Addy Osmani's lifecycle structure. The post reports what worked for you, not a universal method. The baseline is openly labelled a *simulated* brownfield.

The agent only works in the pilot (encoders) and Phase 2 (LLM prompt); everything before them fixes what "better" means, and everything after checks the claim.

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

## Pilot and full encoder training

See the [pilot plan](<Skill-Based SDLC Brownfield Pilot.md>) for A0/A1, V0/V1, scoring, the schedule, and full encoder training before Phase 2.

## Phase 2: Agent branch with skills

The gate report, the deploy swap and a zero-shot LLM path already exist in the baseline. The encoders arrive from the pilot (see above). The agent's Phase 2 work is improving the LLM candidate: a one-shot prompt and/or a revised prompt.

This task needs planning, unlike adding an encoder. The baseline's `n24/eval/zero_shot.py` scores Qwen3-0.6B and SmolLM2-360M-Instruct on the **test split only** and records a `prompt_sha256`. So nothing yet lets the agent tune a prompt without touching test. The open questions the skills must settle:

- Where the one-shot example comes from (train split only), and whether it's fixed or chosen per section.
- How prompt iteration is scored on val, which needs a val path that doesn't exist yet.
- How a prompt version is recorded so the gate refuses to compare across versions (`prompt_sha256` alone, or an `eval_version` bump).
- How parse failures are handled and reported for each prompt.
- Whether zero-shot and one-shot are compared as two gate candidates, or one replaces the other.

Branch `agent` from `v0-baseline` (with the merged encoder change) and drive each stage with a skill from your folder.

1. **Plan:** `grill-with-docs` on the task "add a one-shot prompt (and a revised zero-shot prompt if val supports it) for the baseline's LLMs, tune on val only, freeze the prompt, compare at the gate against the encoders, swap the best into the container." Output: a spec via `to-spec`.
2. **Break down:** `to-tickets` into tickets such as: val scoring path, prompt versioning, one-shot example selection, the prompt change, gate wiring.
3. **Design before code:** `architect` to sketch the changes against the existing `zero_shot.py`, the eval harness and the `Predictor` protocol. You approve this sketch; it is the main human checkpoint.
4. **Build:** `tdd` + `implement` per ticket. Prompt iteration runs on val only; test runs once per frozen prompt.
5. **Self-check:** `principle-prove-it-works` (run the real thing, not a proxy) and `principle-explain-the-number` on every reported metric.
6. **Review:** `code-review`, `blast-radius` on anything touching `inference/` or `api/`, and `interrogate` for adversarial review (see Phase 3).
7. **Hand-off:** `describe-change` for commits/PR text, `deslop` before merge.

**Fairness rules for the comparison**

- Encoders (RoBERTa, ModernBERT, UltraBERT) are fine-tuned with the same budget and seed count (done before Phase 2).
- One-shot examples come from the train split only. Prompt choices are made on val; the test split is scored once per frozen prompt.
- Every prompt version that reaches the gate is pre-registered in `gate/criteria.toml` before its test run, with its `prompt_sha256`.
- The LLM goes through the same `predict`; unparseable outputs count as wrong and are reported as a parse-failure rate per prompt.
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
| The LLM comparison is fair | Prompt changed after test results existed, the one-shot example came from outside train, a prompt reached test without being pre-registered, or unparseable outputs were dropped |
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
- Fine-tuned encoders vs a zero-shot or one-shot LLM is not like-for-like; the post asks "is fine-tuning worth it over prompting," not "which architecture is best."
- The encoders were built hands-off in the pilot and the LLM prompt with supervision in Phase 2, so skill effects on the two candidate types aren't directly comparable. The encoder code that reaches the gate is a human-reviewed pick from one pilot branch.
- One dataset and one task; results may not transfer.
- Agent behaviour varies run to run; one agent branch is one sample.
- Pilot: n=1 per arm in a fixed order (A0→A1→V0→V1), with no human interventions, so intervention counts don't apply.
- Pilot: a GPT subject runs skills written for Claude. Background `~/.agents/skills` is held constant but not removed.
- Pilot: step 1b runs in the same thread as 1a, so it measures generalisation of the agent's own fix, not a fresh start. A failed 1a leaves 1b unscored.
- Pilot: you planted the seeds. A downgraded pin may be easier to spot than a real pin that has drifted. The `fix-vulnerability` references are missing.

**Next**

- The full task ladder (add one encoder → add a second → LLM prompt work → legacy cleanup) with intervention counts per ticket.
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
- [ ] Which baseline LLM(s) Phase 2 targets: Qwen3-0.6B, SmolLM2-360M-Instruct, or both?
- [ ] Zero-shot vs one-shot: two gate candidates, or one-shot replaces zero-shot?
- [ ] Which UltraBERT checkpoint (verified Hugging Face ID)?
- [ ] Seed count for each encoder (3 or 5)?
- [ ] Gate thresholds: minimum F1 gain, latency and memory budgets?
- [ ] CPU or GPU for the reproducibility and parity tests?
