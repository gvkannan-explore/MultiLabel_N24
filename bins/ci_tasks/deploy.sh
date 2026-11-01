#!/usr/bin/env bash
# Judge the one results/nlp/ file a push added or changed with the deployment
# gate, deploy its run to Modal from a frozen image, and check the live API.
# Run by .github/workflows/deploy.yml; see ADR 0008.
#   bash bins/ci_tasks/deploy.sh <before-sha> <after-sha>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if (($# != 2)); then
    echo "Usage: bash bins/ci_tasks/deploy.sh <before-sha> <after-sha>" >&2
    exit 2
fi

changed=()
while IFS= read -r path; do
    changed+=("$path")
done < <(git diff --name-only --diff-filter=AM "$1" "$2" -- 'results/nlp/*.json')
if ((${#changed[@]} != 1)); then
    echo "❌ Expected one added or changed results/nlp/ file, found ${#changed[@]}:" \
        ${changed[@]+"${changed[@]}"} >&2
    echo "   One deployment serves one run; push each results file separately." >&2
    exit 1
fi
results=${changed[0]}

# Exit 1 (not deployable) or 2 (refused) stops here, before anything deploys.
uv run python -m n24.gate.report "$results"

# The gate just confirmed the recomputed weights hash equals the recorded one.
fields=$(uv run python -c '
import sys
from pathlib import Path
from n24.eval.run import Results
results = Results.model_validate_json(Path(sys.argv[1]).read_text())
print(results.run, results.weights_sha256.recomputed)
' "$results")
read -r N24_RUN N24_WEIGHTS_SHA256 <<< "$fields"
export N24_RUN N24_WEIGHTS_SHA256

uv run modal deploy -m n24.deploy.modal_app
url=$(uv run python -c 'import modal; print(modal.Function.from_name("n24-api", "web").get_web_url())')
echo "Deployed $N24_RUN -> $url"
bash bins/api/check_api.sh "$url" "$N24_RUN"
