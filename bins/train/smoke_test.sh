#!/usr/bin/env bash
# Train for five steps on datasets/prepared, then freeze the run the way the
# deployment image does, to check the model trains and deploys end to end.
# Extra arguments go to the trainer, for example:
#   bash bins/train/smoke_test.sh --model answerdotai/ModernBERT-base --use-cpu
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
rm -rf experiment_runs/smoke-1
uv run python -m n24.train.train_baseline --data-dir datasets/prepared --output-dir experiment_runs/smoke-1 --max-steps 5 "$@"
uv run python - experiment_runs/smoke-1 <<'PY'
import json
import sys
import tempfile
from pathlib import Path

from n24.deploy.freeze import freeze_run

run_dir = Path(sys.argv[1])
weights_sha256 = json.loads((run_dir / "metrics.json").read_text())["weights_sha256"]
with tempfile.TemporaryDirectory() as dest:
    freeze_run(run_dir, weights_sha256, Path(dest))
print(f"{run_dir} trained and froze for deployment")
PY
