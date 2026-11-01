#!/usr/bin/env bash
# Start the inference API on a trained run, check its responses, and stop it.
# Exits non-zero on the first failed check.
#   bash bins/api/api_smoke_test.sh [run-dir] [port]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
run_dir=${1:-experiment_runs/smoke-1}
port=${2:-8765}
url="http://127.0.0.1:$port"

uv run python -m n24.api.app --run-dir "$run_dir" --port "$port" &
server=$!
trap 'kill "$server" 2>/dev/null' EXIT

# Loading the model takes a few seconds; FastAPI serves /docs once it is up.
for _ in $(seq 120); do
    curl -sf "$url/docs" >/dev/null && break
    kill -0 "$server" 2>/dev/null || { echo "❌ The API exited during startup" >&2; exit 1; }
    sleep 1
done
curl -sf "$url/docs" >/dev/null || { echo "❌ The API did not start within 120 s" >&2; exit 1; }

# The local server has no proxy auth, so check it without the headers.
MODAL_PROXY_KEY='' bash bins/api/check_api.sh "$url" "$(basename "$run_dir")"
