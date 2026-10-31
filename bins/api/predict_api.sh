#!/usr/bin/env bash
# Send abstracts to a running inference API and print a section for each.
#   bash bins/api/predict_api.sh ["abstract" ...]
#   PORT=8765 bash bins/api/predict_api.sh "The team won the final."
# Start the API first with bins/api/serve_api.sh.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
url="http://127.0.0.1:${PORT:-8000}/predict"
if (($# == 0)); then
    set -- \
        "The team clinched the championship with a late goal in extra time." \
        "The chef opened a new restaurant serving seasonal small plates."
fi

# Build the JSON body in Python so quotes inside an abstract stay valid JSON.
body=$(uv run python -c 'import json, sys; print(json.dumps({"abstracts": sys.argv[1:]}))' "$@")
curl -sS --fail-with-body -X POST "$url" -H 'content-type: application/json' -d "$body" |
    uv run python -c '
import json, sys
body = json.load(sys.stdin)
print("run:", body["run"])
for prediction in body["predictions"]:
    print("section:", prediction["section"])'
