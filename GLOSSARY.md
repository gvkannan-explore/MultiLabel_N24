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

## Data

**Body**:
The main text of an article.
_Avoid_: article (the dataset's field name), content

**Model input**:
The part of an article the classifier reads. For the baseline model, the abstract.
_Avoid_: features, text

**Prepared dataset**:
The cleaned train, dev, and test splits with section labels that the trainer reads.
_Avoid_: processed data, features

**Dev split**:
The split used to choose the best epoch during training.
_Avoid_: validation set, val

**Test split**:
The held-out split that only the eval harness reads, after training is finished.
_Avoid_: holdout, eval set

## Evaluation

**Eval harness**:
The code that scores a trained run on the test split and writes its results file. Models never score themselves.
_Avoid_: evaluator, scorer

**Results file**:
The JSON the eval harness writes for one run: eval version, test split hash, model identity, test metrics, repeatability numbers, and per-article predictions.
_Avoid_: report, scores

**Eval version**:
An integer that changes whenever the eval harness or test split changes. Results with different eval versions are never compared.
_Avoid_: harness version, schema version

**Deployment gate**:
The post-training check that reads a results file and pre-registered criteria and decides whether the run may be deployed.
_Avoid_: gate (that word alone means a guardrail), release check

**Repeatability**:
Two prediction passes over the test split, each with a fresh model load, give the same sections and logits within a tolerance.
_Avoid_: determinism, stability

**Reproducibility**:
A run can be rebuilt from what it recorded: its weights and dataset hashes still match, and it ran from a clean git commit.
_Avoid_: provenance, traceability

## Guardrails

**Guardrail**:
An automated check that blocks a commit, push, or CI run.
_Avoid_: gate, policy

**Privacy finding**:
A personal path, real email, known local identity, or private file in Git content or metadata.
_Avoid_: PII hit, leak

**Repository identity**:
The owner's GitHub login and private noreply address, used as author and committer for every commit.
_Avoid_: anonymous identity, bot user, placeholder author

**History rewrite**:
Replacing past commits to remove privacy findings.
_Avoid_: history cleanup, scrub
