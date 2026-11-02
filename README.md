# MultiLabel_N24

A learning project that classifies N24News articles into their 24 news
sections with PyTorch and transformers. This baseline branch holds the frozen
legacy code, the guardrails added around it, and the pipeline that trains and
judges the baseline model: the first classifier that every later model is
compared against. Terms in this README follow the [glossary](GLOSSARY.md).

The baseline model scores 0.786 macro F1 and 0.799 accuracy on the test split
and passes all seven checks of the deployment gate
([gate report](gate/reports/baseline-roberta-epoch3-Oct312026.html)). A small
LLM with no training (Qwen3-0.6B) scores 0.071 macro F1 on the same test
split.

## How it fits together

```mermaid
flowchart LR
    subgraph s1["1 · Prepare data"]
        direction TB
        p1["Clean and split<br/>the news articles"] --> a1[/"Prepared dataset<br/>24 sections"/]
    end
    subgraph s2["2 · Train model"]
        direction TB
        p2["Fine-tune RoBERTa<br/>on a Modal GPU"] --> a2[/"Trained model<br/>and training report"/]
    end
    subgraph s3["3 · Deployment gate"]
        direction TB
        c3[/"Criteria, committed<br/>before any test score"/] --> p3{"All seven checks<br/>pass?"} --> a3[/"HTML report<br/>of each check"/]
    end
    subgraph s4["4 · Deployment"]
        direction TB
        p4["Deploy a frozen copy<br/>of the approved model"] --> a4[/"Inference API<br/>behind proxy auth"/]
    end
    blocked["Blocked.<br/>Nothing deploys."]

    s1 --> s2 --> s3
    s3 -- "yes" --> s4
    s3 -- "no" --> blocked

    classDef step fill:#dbeafe,stroke:#1d4ed8,color:#172554
    classDef artifact fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef input fill:#fef9c3,stroke:#a16207,color:#422006
    classDef stop fill:#fee2e2,stroke:#b91c1c,color:#450a0a
    class p1,p2,p3,p4 step
    class a1,a2,a3,a4 artifact
    class c3 input
    class blocked stop
```

Blue boxes are the steps, and green boxes are what each step writes. The
yellow box is the one input you write by hand: the criteria in
`gate/criteria.toml`, which set the minimum test scores. You commit them before
you score any model on the test split, so the git history shows that nobody
lowered the bar after seeing a score.

The table lists every stage, including the ones the diagram leaves out.

| Stage | Command | Reads | Writes | Lives in |
|---|---|---|---|---|
| Data preparation | `n24.data.prepare_data` | `datasets/news/nytimes_{train,dev,test}.json` | `datasets/prepared/` | local and git-ignored, with a copy on the `n24-data` volume |
| Training | `n24.train.train_baseline`, or `n24.train.modal_train` on a GPU | prepared train and dev splits | `<run>/best_model/`, `<run>/metrics.json`, `<run>/training_report.pdf` | the `n24-runs` volume, or `experiment_runs/` for a local run |
| Tracking | `n24.track.import_run` | `metrics.json` or `metrics/<run>.json` | `experiment_runs/mlflow.db` | a local cache, rebuilt from `metrics/` in git |
| Eval harness | `n24.eval.run` | `<run>/`, prepared test split | `results/nlp/<run>.json` | git |
| Zero-shot LLMs | `n24.eval.zero_shot` | prepared test split | `results/llm/zs-<model>.json` | git |
| Deployment gate | `n24.gate.report` | results file, `gate/criteria.toml` | `gate/reports/<run>.html`, exit code | git, and the `gate-report` artifact of the deploy job |
| Deployment | `bins/ci_tasks/deploy.sh` in GitHub Actions | `results/nlp/<run>.json`, `<run>/` on the `n24-runs` volume | the `n24-api` Modal app | Modal, as a frozen image |

The [system diagram](docs/system-diagram.md) draws every command, artifact,
and guardrail, and each of the deployment gate's seven checks.

Every stage is a module run from the repository root with
`uv run python -m n24.<package>.<module>`. Stages talk to each other only
through files on disk, so each one can be rerun, tested, and inspected alone.

One rule holds the design together: **the trainer never reads the test split.**
Training chooses its best epoch on the dev split. Only the eval harness reads
the test split, after training is finished, so models never score themselves.
The deployment gate then judges the harness's results file against criteria
committed before any test score existed.

## Setup

Run this once in each clone or worktree. It installs the locked environment,
the pinned Gitleaks binary, and the git hooks, then runs the CI check:

```bash
uv sync --locked --group lint
bash scripts/install-gitleaks.sh
bash scripts/setup-hooks.sh
bash bins/ci_tasks/perform_ci.sh
```

The N24News split files (`nytimes_train.json`, `nytimes_dev.json`,
`nytimes_test.json`) are not in the repository. Put them, or a link to the
folder that holds them, at `datasets/news/`. The `datasets/` folder is
git-ignored.

## Data preparation

`n24/data/prepare_data.py` turns the three split files into the prepared
dataset that the trainer and the eval harness read.

```bash
uv run python -m n24.data.prepare_data --input-dir datasets/news --output-dir datasets/prepared
# or, deleting any earlier datasets/prepared first:
bash bins/data/data_prep.sh
```

- The sections are the sorted set of sections in the train split. There must
  be exactly 24, or preparation stops.
- A dev or test record whose section is missing from train, or any record with
  an empty abstract, stops preparation with the split and record index.
- Text cleaning only collapses whitespace. Accents, numbers, and punctuation
  stay, because the tokenizer handles them.
- Each row keeps `headline`, `abstract`, and `body`, plus a `label` column of
  type `ClassLabel` whose names are the sections. The section names travel
  with the data, so no later stage keeps its own copy of the mapping.
- The output is a Hugging Face `DatasetDict` saved with `save_to_disk`. The
  command prints row counts and per-section counts for each split.

The baseline model reads only the abstract, but the headline and body are kept
so later models can use them without preparing the data again
([ADR 0004](docs/adr/0004-baseline-model-reads-abstract-alone.md)).

## Training the baseline model

`n24/train/train_baseline.py` fine-tunes `FacebookAI/roberta-base` for
sequence classification. The model input is the abstract alone.

```bash
uv run python -m n24.train.train_baseline --data-dir datasets/prepared --output-dir experiment_runs/roberta-baseline
```

Pass `--model answerdotai/ModernBERT-base` to train ModernBERT through the same
trainer. Its saved run uses the existing evaluation and inference API commands,
and the deployment image accepts its encoder type. A five-step CPU smoke run:

```bash
uv run python -m n24.train.train_baseline --model answerdotai/ModernBERT-base --data-dir datasets/prepared --output-dir experiment_runs/modernbert-smoke --max-steps 5 --batch-size 4 --max-length 64 --use-cpu
```

Use a GPU for the full run. How a run works:

1. It creates the output directory and refuses to continue if it already
   exists, so an earlier run is never overwritten.
2. It tokenizes the train and dev abstracts, truncated to `--max-length`
   tokens. Padding happens per batch in the data collator.
3. The Hugging Face `Trainer` trains with linear learning-rate decay, 6%
   warmup, weight decay 0.01, and bf16 when the GPU supports it. It scores the
   dev split after every epoch and keeps the epoch with the best macro F1.
4. It predicts the dev split once more with the best epoch to get F1 for each
   section, then saves the weights and tokenizer to `best_model/` and deletes
   the checkpoints.
5. It writes `metrics.json`.

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `datasets/prepared` | Prepared dataset |
| `--output-dir` | required | Run directory; must not exist |
| `--model` | `FacebookAI/roberta-base` | Pretrained model name |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-5` | Peak learning rate |
| `--batch-size` | `32` | Train batch size; dev uses twice this |
| `--max-length` | `128` | Tokens kept per abstract |
| `--seed` | `42` | Random seed |
| `--max-steps` | `-1` | Stop after N steps for a smoke run |
| `--use-cpu` | off | Force training on CPU |

`metrics.json` is the run's record. Tracking and the eval harness both read it:

- `dev_loss`, `dev_accuracy`, `dev_f1_macro`, and `dev_f1_per_section`
- `args`: every flag, as strings
- `git_commit`: the output of `git describe --always --dirty`, so a run from
  uncommitted code ends in `-dirty`
- `gpu`: the CUDA device's name, such as `NVIDIA L4`, or null without CUDA
- `dataset_sha256`: a hash of every file in the prepared dataset except
  `state.json`, which the datasets library rewrites with a random fingerprint
  on each save, and the `cache-*.arrow` files that each training run's
  tokenization adds next to the data
- `weights_sha256`: the hash of `best_model/model.safetensors`
- `log_history`: the trainer's loss and dev scores at each logging step and
  evaluation, which become the training curves

The recipe differs from the N24News paper's, which reports about 78% macro F1
from abstracts alone; see ADR 0004 before you change it.

**Smoke runs.** `--max-steps 5` stops after five steps, scoring and saving at
that step instead of at each epoch. `bash bins/train/smoke_test.sh` does
this into `experiment_runs/smoke-1` to check that the trainer runs end to end.

## Training on Modal

`n24/train/modal_train.py` runs the same `train` function on a Modal GPU
with a two-hour timeout. It takes an L4, or an A10G when no L4 is free; both
support bf16, and `metrics.json` records which one ran.

- The image installs the dependencies from `uv.lock` and has the
  `FacebookAI/roberta-base` weights and tokenizer built in, so a run does not
  download them again.
- The prepared dataset comes from the `n24-data` volume. Each run is written
  to `/news24-runs/<name>` on the `n24-runs` volume.
- The trainer flags are parsed on your machine before the GPU starts, so a
  typo fails right away. The git commit is also read locally, because the
  container has no git checkout.

One-time setup, from the repository root:

```bash
uv run modal setup
uv run modal volume create n24-data
uv run modal volume put n24-data datasets/prepared /prepared
```

A real experiment goes through `bins/train/train_modal.sh`, which trains
and publishes the run in one step:

```bash
bash bins/train/train_modal.sh roberta-lr3e-5 "--lr 3e-5"
```

It trains, copies the run's `metrics.json` to `metrics/<name>.json`, imports
that file into the local MLflow store, and stages it for you to commit. Smoke
runs use plain `modal run`, so `metrics/` holds only real experiments:

```bash
uv run modal run -m n24.train.modal_train --name smoke-1 --args "--max-steps 5"
```

The script fetches only `metrics.json`. Download the whole run before you
score it with the eval harness:

```bash
mkdir -p runs_modal
uv run modal volume get n24-runs roberta-lr3e-5 runs_modal/
```

## Experiment tracking

`n24/track/import_run.py` logs runs into a local MLflow store
([ADR 0006](docs/adr/0006-mlflow-imports-metrics-json.md)). Training never
talks to MLflow. The import reads a finished run's `metrics.json` and logs it
as one MLflow run in the `baseline-model` experiment, named after the run's
output directory:

- parameters: the trainer flags
- metrics: the `log_history` curves by step, the final dev scores, and
  per-section dev F1 as `dev_f1/<section>` (MLflow rejects `&`, so
  "Art & Design" becomes "Art and Design")
- tags: `git_commit`, `gpu`, `dataset_sha256`, `weights_sha256`

A run whose name is already in the store is skipped, so imports are safe to
repeat. The store `experiment_runs/mlflow.db` is a git-ignored cache; the
`metrics/*.json` files in git are the source of truth. Rebuild the store on
any clone, add a local run, and browse:

```bash
uv run python -m n24.track.import_run metrics/*.json
uv run python -m n24.track.import_run experiment_runs/smoke-1/metrics.json
uv run mlflow ui --backend-store-uri sqlite:///experiment_runs/mlflow.db
```

## Evaluation

`n24/eval/run.py` (the eval harness) scores a trained run on the test split
and writes `results/nlp/<run>.json`. `n24/eval/zero_shot.py` scores an
instruction-tuned LLM on a Modal GPU and writes `results/llm/`.
`n24/eval/probes.py` scores a model on the external probes and writes
`results/probes/`. [Evaluation](docs/evaluation.md) has the commands, what
each results file holds, and when to bump the eval version
([ADR 0005](docs/adr/0005-eval-comparability.md)).

## Deployment gate

`n24/gate/report.py` decides whether a scored run may be deployed. It reads a
results file and the criteria in `gate/criteria.toml`.

**Pre-register the criteria first.** Fill in `gate/criteria.toml` and commit it
before you score any model on the test split, so the git history proves the
thresholds came before the results. Pin the test split hash there:

```bash
uv run python -c "from pathlib import Path; from n24.train.train_baseline import dataset_sha256; print(dataset_sha256(Path('datasets/prepared/test')))"
```

The criteria file holds `eval_version`, `test_split_sha256`, the
`[accuracy]` minimums, and the `[repeatability]` tolerance `logits_atol`.
Unknown keys or an unpinned hash are rejected. Then judge a run:

```bash
uv run python -m n24.gate.report results/nlp/roberta-baseline.json
```

The deployment gate first refuses, with exit code 2, a results file whose eval
version or test split hash differs from the criteria. Otherwise it runs seven
checks; the run is deployable only if all of them pass.

| Group | Check | Fails if |
|---|---|---|
| Accuracy | Test macro F1 | below `min_test_f1_macro` |
| Accuracy | Test accuracy | below `min_test_accuracy` |
| Accuracy | Every section | the worst section's F1 is below `min_section_f1` |
| Repeatability | Two passes agree | any section differs, or any logit differs by more than `logits_atol` |
| Reproducibility | Weights | the recomputed weights hash differs from `metrics.json` |
| Reproducibility | Dataset | the recomputed prepared dataset hash differs from `metrics.json` |
| Reproducibility | Clean commit | no commit is recorded, or it ends in `-dirty` |

It prints the checks as a table and writes `gate/reports/<run>.html`, whose
rows link to their evidence: the results file or the run's `metrics.json`.
`--criteria` and `--reports-dir` override the default paths.

| Exit code | Meaning |
|---|---|
| 0 | Deployable |
| 1 | Not deployable: at least one check failed |
| 2 | Refused: other eval version or test split |

## Deploying to Modal

A push to `main` that adds or changes one file under `results/nlp/` judges
it with the deployment gate and, on exit 0, deploys the run to Modal from a
frozen image behind proxy auth. Zero-shot results in `results/llm/` never
deploy. [Deploying to Modal](docs/deployment.md) has the steps, the
one-time setup, and how to call the API.

## Guardrails and CI

Guardrails are automated checks that block a commit, a push, or a CI run.
[CI and privacy checks](docs/ci-and-privacy.md) has the full details.

- **pre-commit** checks the commit identity, scans the staged files for
  privacy findings and secrets, and blocks staged files over 20,000 bytes.
- **commit-msg** scans the message for personal information.
- **pre-push** scans the history being pushed, then runs the CI check.
- **`bins/ci_tasks/perform_ci.sh`** runs on every push and pull request in
  GitHub Actions. It scans working, staged, and historical content for privacy
  findings and secrets, runs the hook regression tests, Ruff lint and format,
  Pyright, the offline tests, and a syntax check of every shell script. Every
  check runs even after a failure, and any failure fails the run. A change is
  done when this script passes.

Commits use the owner's GitHub noreply identity
([ADR 0003](docs/adr/0003-github-noreply-identity.md)), and `main` is updated
only by fast-forward merges from a local machine; see the merging section of
the CI document.

## Legacy code

The legacy code's behavior is frozen
([ADR 0001](docs/adr/0001-legacy-behavior-frozen.md)). It covers
`scripts/news-classifier.py`, `n24/utils/data.py`,
`n24/preprocess/roberta_preprocessor.py`, and the notebooks. Some of it does
not work: `news-classifier.py` expects an `n24news.csv` that does not exist,
and `load_datajson` reads an `article` key. That is the brownfield starting
point, and `tests/test_legacy.py` pins it. Changes to legacy code are limited
to formatting, imports, and annotations.

## Repository layout

```text
n24/
  data/          data preparation
  train/         trainer and its Modal wrapper
  track/         MLflow import
  eval/          eval harness
  gate/          deployment gate and its HTML template
  deploy/        frozen Modal image that serves a gated run
  preprocess/, utils/   legacy code
bins/            shell entry points, by purpose:
  ci_tasks/      the CI check, run locally and by GitHub Actions, and the deploy
  data/          data preparation
  train/         local smoke test and Modal training
  api/           serve, query, smoke-test, and check the inference API
scripts/         guardrail tooling, setup scripts, and the legacy news-classifier.py
gate/            criteria.toml, and reports/ written by the deployment gate
metrics/         metrics.json of each published Modal run
results/         results files: nlp/ from the eval harness, llm/ from zero-shot scoring, probes/ from external probes
tests/           offline unit tests
docs/            CI, evaluation, and deployment documents, and ADRs
notebooks/       exploration notebooks (legacy)
plans/           planning notes
datasets/        raw and prepared data (git-ignored)
experiment_runs/ local runs and the MLflow store (git-ignored)
runs_modal/      runs downloaded from Modal (git-ignored)
```

## Tests

[Tests](docs/tests.md) lists what each test file covers and the command that
runs them offline.

## Decisions

The [decision records](docs/adr/README.md) list each ADR. The
[glossary](GLOSSARY.md) defines the project's terms.
