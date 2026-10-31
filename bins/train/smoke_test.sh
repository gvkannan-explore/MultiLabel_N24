#!/usr/bin/env bash
# Train for five steps on datasets/prepared to check the trainer runs end to end.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
rm -rf experiment_runs/smoke-1
uv run python -m src.train.train_baseline --data-dir datasets/prepared --output-dir experiment_runs/smoke-1 --max-steps 5
