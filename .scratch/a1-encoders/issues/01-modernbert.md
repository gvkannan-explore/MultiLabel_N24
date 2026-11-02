# modernbert

Status: blocked
Blocked by: none

Train, evaluate, freeze, and serve ModernBERT through existing interfaces. Acceptance: offline regression, full CI, and real five-step CPU smoke pass.

## Evidence
The offline train/evaluate/freeze/serve regression passes. The real ModernBERT five-step CPU run produced its checkpoint, metrics, and training report in `experiment_runs/a1-modernbert-smoke/`.

## Blockers
Full CI fails on 32 author-metadata history findings in the agreed base. The copied TDD example previously triggered a false positive; its generic API route was renamed before commit. Guardrails remain unchanged. Logs and skill exposure hashes are saved in `.audit/`.
