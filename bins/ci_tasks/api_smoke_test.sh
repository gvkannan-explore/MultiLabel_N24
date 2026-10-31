#!/usr/bin/env bash
# Start the inference API on a trained run, check its responses, and stop it.
# Exits non-zero on the first failed check.
#   bash bins/ci_tasks/api_smoke_test.sh [run-dir] [port]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
run_dir=${1:-experiment_runs/smoke-1}
port=${2:-8765}
url="http://127.0.0.1:$port"

uv run python -m src.api.app --run-dir "$run_dir" --port "$port" &
server=$!
trap 'kill "$server" 2>/dev/null' EXIT

# Loading the model takes a few seconds; FastAPI serves /docs once it is up.
for _ in $(seq 120); do
    curl -sf "$url/docs" >/dev/null && break
    kill -0 "$server" 2>/dev/null || { echo "❌ The API exited during startup" >&2; exit 1; }
    sleep 1
done
curl -sf "$url/docs" >/dev/null || { echo "❌ The API did not start within 120 s" >&2; exit 1; }

curl -sS --fail-with-body -X POST "$url/predict" \
    -H 'content-type: application/json' \
    -d '{"abstracts": [
        "The team clinched the championship with a late goal in extra time.",
        "The chef opened a new restaurant serving seasonal small plates."
    ]}' |
    uv run python -c '
import json, sys
body = json.load(sys.stdin)
run, sections, predictions = body["run"], body["sections"], body["predictions"]
assert run == sys.argv[1], f"run is {run}, expected {sys.argv[1]}"
assert len(predictions) == 2, f"{len(predictions)} predictions for 2 abstracts"
for prediction in predictions:
    section, logits = prediction["section"], prediction["logits"]
    assert len(logits) == len(sections), "one logit per section"
    best = sections[logits.index(max(logits))]
    assert section == best, f"{section} is not the argmax {best}"
    print("  ", section)
' "$(basename "$run_dir")"
echo "✅ POST /predict returns one section per abstract, each the argmax of its logits"

for bad in '{"abstracts": []}' '{"abstracts": [""]}' '{}'; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$url/predict" \
        -H 'content-type: application/json' -d "$bad")
    [[ $code == 422 ]] || { echo "❌ $bad returned $code, expected 422" >&2; exit 1; }
    echo "✅ $bad is rejected with 422"
done
