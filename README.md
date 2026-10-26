# MultiLabel_N24
Classification using PyTorch and transformers
## Guardrails and docs

This branch keeps the legacy code's behavior unchanged and adds privacy, secret,
file-size, lint, type, and test checks. Run the setup once in each clone or
worktree:

```bash
uv sync --locked --group lint
bash scripts/install-gitleaks.sh
bash scripts/setup-hooks.sh
bash bins/ci_tasks/perform_ci.sh
```

- [CI and privacy checks](docs/ci-and-privacy.md)
- [ADR 0001: Legacy behavior is frozen](docs/adr/0001-legacy-behavior-frozen.md)
- [ADR 0002: History was rewritten before the baseline branch](docs/adr/0002-history-rewritten-before-baseline.md)
- [ADR 0003: Commits use the owner's GitHub noreply identity](docs/adr/0003-github-noreply-identity.md)
- [ADR 0004: The baseline model reads the abstract alone](docs/adr/0004-baseline-model-reads-abstract-alone.md)
- [ADR 0005: Results are compared only within one eval version and test split](docs/adr/0005-eval-comparability.md)
- [ADR 0006: MLflow is a local view imported from each run's metrics.json](docs/adr/0006-mlflow-imports-metrics-json.md)
- [Glossary](GLOSSARY.md)

## Layout

`scripts/` holds tooling outside the package (CI checks, git hooks, setup).
`bins/ci_tasks/` holds the shell entry points (CI, data prep, smoke test, Modal
training). Training output goes to the git-ignored `experiment_runs/`.
Workflow code lives in `src/` and runs with `uv run python -m src.…` from the
repository root.

## Baseline model

The N24News split files (`nytimes_train.json`, `nytimes_dev.json`,
`nytimes_test.json`) are not in the repository. Put them, or a link to the
folder that holds them, at `datasets/news/`. The `datasets/` folder is
git-ignored.

```bash
# 1. Prepare the data: headline, abstract, and body, with section labels.
uv run python -m src.data.prepare_data --input-dir datasets/news --output-dir datasets/prepared

# 2. Fine-tune roberta-base on the abstracts. Use a GPU for the full run.
uv run python -m src.train.train_baseline --data-dir datasets/prepared --output-dir experiment_runs/roberta-baseline
```

Add `--max-steps 5` for a quick smoke run. The trainer keeps the epoch with the
best macro F1 on the dev split. It writes `best_model/` and `metrics.json`,
which records the dev metrics, per-section F1, flags, git commit, and sha256
hashes of the prepared dataset and the model weights. It refuses to overwrite
an existing output directory. The trainer never reads the test split.

To train on a Modal L4 GPU instead, follow the setup and run commands at the
top of `src/train/modal_train.py`. Run them from the repository root.

## Experiment tracking

`bins/ci_tasks/train_modal.sh` trains on Modal and publishes the run in one step. It
copies the run's `metrics.json` to `metrics/<run>.json`, imports it into the
local MLflow store, and stages the file for you to commit. Smoke runs use the
plain `modal run` command instead, so `metrics/` holds only real experiments.

```bash
bash bins/ci_tasks/train_modal.sh roberta-lr3e-5 "--lr 3e-5"
uv run mlflow ui --backend-store-uri sqlite:///experiment_runs/mlflow.db
```

The store `experiment_runs/mlflow.db` is git-ignored. Rebuild it on any clone, or add a
local run, with the import command; it skips runs already in the store
([ADR 0006](docs/adr/0006-mlflow-imports-metrics-json.md)):

```bash
uv run python -m src.track.import_run metrics/*.json
uv run python -m src.track.import_run experiment_runs/smoke-1/metrics.json
```

## Deployment gate

The deployment gate decides whether a trained run may be deployed. Fill in
`gate/criteria.toml` and commit it before scoring any model on the test split.
Pin the test split hash there first:

```bash
uv run python -c "from pathlib import Path; from src.train.train_baseline import dataset_sha256; print(dataset_sha256(Path('datasets/prepared/test')))"
```

Then score the run and judge it:

```bash
# 1. Eval harness: predicts the test split twice on CPU and writes results/<run>.json.
uv run python -m src.eval.run --run-dir runs_modal/roberta-baseline

# 2. Deployment gate: writes gate/reports/<run>.html. Exit 0 deployable,
#    1 not deployable, 2 refused (other eval version or test split).
uv run python -m src.gate.report results/roberta-baseline.json
```

The gate checks accuracy (test macro F1, accuracy, and the worst section's F1
against the minimums), repeatability (two prediction passes give the same
sections and logits within `logits_atol`), and reproducibility (the weights and
dataset hashes match `metrics.json`, and the run came from a clean commit).
