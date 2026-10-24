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
- [Glossary](GLOSSARY.md)
