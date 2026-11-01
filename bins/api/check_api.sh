#!/usr/bin/env bash
# Check a running inference API's responses. Exits non-zero on the first failed check.
#   bash bins/api/check_api.sh <url> <run>
# With MODAL_PROXY_KEY and MODAL_PROXY_SECRET set, sends them as Modal proxy
# auth headers and checks that a request without them is refused.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if (($# != 2)); then
    echo "Usage: bash bins/api/check_api.sh <url> <run>" >&2
    exit 2
fi
url=$1
run=$2
auth=()
if [[ -n ${MODAL_PROXY_KEY:-} ]]; then
    auth=(-H "Modal-Key: $MODAL_PROXY_KEY" -H "Modal-Secret: ${MODAL_PROXY_SECRET:?}")
fi

# A cold start loads the model before the first response.
curl -sS --fail-with-body --retry 3 --retry-all-errors --max-time 180 \
    ${auth[@]+"${auth[@]}"} -X POST "$url/predict" \
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
' "$run"
echo "✅ POST /predict returns one section per abstract, each the argmax of its logits"

for bad in '{"abstracts": []}' '{"abstracts": [""]}' '{}'; do
    code=$(curl -s -o /dev/null -w '%{http_code}' ${auth[@]+"${auth[@]}"} -X POST "$url/predict" \
        -H 'content-type: application/json' -d "$bad")
    [[ $code == 422 ]] || { echo "❌ $bad returned $code, expected 422" >&2; exit 1; }
    echo "✅ $bad is rejected with 422"
done

if ((${#auth[@]})); then
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$url/predict" \
        -H 'content-type: application/json' -d '{"abstracts": ["No key."]}')
    [[ $code == 401 ]] || { echo "❌ A request without proxy auth returned $code, expected 401" >&2; exit 1; }
    echo "✅ A request without proxy auth is rejected with 401"
fi
