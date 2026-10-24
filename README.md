# MultiLabel_N24
Multi-Label Multi-Modal Classification using lightning, PyTorch, transformers and timm
## Guardrails and docs

This branch keeps the legacy code's behavior unchanged and adds privacy, secret,
file-size, lint, type, and test checks. Run the setup once in each clone or
worktree:

```bash
uv sync --locked --group lint
bash scripts/install-gitleaks.sh
bash scripts/setup-hooks.sh
bash runs/perform_ci.sh
```

- [CI and privacy checks](docs/ci-and-privacy.md)
- [ADR 0001: Legacy behavior is frozen](docs/adr/0001-legacy-behavior-frozen.md)
- [ADR 0002: History was rewritten before the baseline branch](docs/adr/0002-history-rewritten-before-baseline.md)
- [ADR 0003: Commits use the owner's GitHub noreply identity](docs/adr/0003-github-noreply-identity.md)
- [ADR 0004: The baseline model reads the abstract alone](docs/adr/0004-baseline-model-reads-abstract-alone.md)
- [Glossary](GLOSSARY.md)

## Baseline model

The N24News split files (`nytimes_train.json`, `nytimes_dev.json`,
`nytimes_test.json`) are not in the repository. Put them, or a link to the
folder that holds them, at `datasets/news/`. The `datasets/` folder is
git-ignored.

```bash
# 1. Prepare the data: headline, abstract, and body, with section labels.
uv run python scripts/prepare_data.py --input-dir datasets/news --output-dir datasets/prepared

# 2. Fine-tune roberta-base on the abstracts. Use a GPU for the full run.
uv run python scripts/train_baseline.py --data-dir datasets/prepared --output-dir runs/roberta-baseline
```

Add `--max-steps 5` for a quick smoke run. The trainer keeps the epoch with the
best macro F1 on the dev split. It writes `best_model/` and `metrics.json`,
which records the dev metrics, per-section F1, flags, git commit, and sha256
hashes of the prepared dataset and the model weights. It refuses to overwrite
an existing output directory. The test split is not used yet.
