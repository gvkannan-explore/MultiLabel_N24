#!/usr/bin/env bash
# Serve a trained run with the inference API until Ctrl-C.
#   bash bins/api/serve_api.sh [run-dir] [port]
# Then predict with bins/api/predict_api.sh, or open http://127.0.0.1:<port>/docs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
run_dir=${1:-experiment_runs/smoke-1}
port=${2:-8000}
uv run python -m n24.api.app --run-dir "$run_dir" --port "$port"
