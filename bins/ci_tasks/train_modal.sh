#!/usr/bin/env bash
# Train on a Modal GPU, then publish the run's metrics.json to metrics/<name>.json,
# import it into the local MLflow store, and stage it. Smoke runs skip this
# script and use the plain modal run command in src/train/modal_train.py.
#
# Usage: bash bins/ci_tasks/train_modal.sh <name> ["extra trainer flags"]
#   bash bins/ci_tasks/train_modal.sh roberta-lr3e-5 "--lr 3e-5"
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
    echo "Usage: bash bins/ci_tasks/train_modal.sh <name> [\"extra trainer flags\"]" >&2
    exit 2
fi
name="$1"
args="${2:-}"
cd "$(git rev-parse --show-toplevel)"
uv run modal run -m src.train.modal_train --name "$name" --args "$args"
mkdir -p metrics
uv run modal volume get n24-runs "$name/metrics.json" - > "metrics/$name.json"
uv run python -m src.track.import_run "metrics/$name.json"
git add "metrics/$name.json"
