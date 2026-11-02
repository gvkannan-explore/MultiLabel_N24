#!/usr/bin/env bash
# Enforce the pilot CI gate. --modal selects the billable GPU variant.
set -euo pipefail
backend=local
if [[ "${1:-}" == --modal ]]; then
    backend=modal
    shift
fi
usage() {
    echo "Usage: bash bins/experiments/run_a1.sh 1a OUTPUT_DIR"
    echo "       bash bins/experiments/run_a1.sh 1b OUTPUT_DIR MODERNBERT_RUN"
    echo "Modal GPU variant: add --modal; OUTPUT_DIR becomes a fresh remote RUN_NAME."
}
if [[ $# -eq 1 && "$1" == --help ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi
case "$1:$#" in
    1a:2) model=answerdotai/ModernBERT-base ;;
    1b:3) model=dmis-lab/UltraBERT-150M ;;
    *) usage >&2; exit 2 ;;
esac
repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$repo_root"
output_dir="$2"
if [[ "$backend" == modal ]]; then
    output_dir="runs_modal/$2"
fi
if [[ -e "$output_dir" ]]; then
    echo "Output already exists: $output_dir. Choose a new run directory." >&2
    exit 1
fi
mkdir -p .audit
evidence=$(mktemp -d ".audit/a1-$1.XXXXXX")
echo "Evidence: $evidence"
bash bins/ci_tasks/perform_ci.sh 2>&1 | tee "$evidence/ci-before.log"
if [[ "$1" == 1b ]]; then
    .venv/bin/python - "$3" "$backend" <<'PY'
import json
import sys
from pathlib import Path

from n24.deploy.freeze import ENCODER_MODEL_TYPES
from n24.train.train_baseline import sha256_file

run = Path(sys.argv[1])
metrics = json.loads((run / "metrics.json").read_text())
args = metrics["args"]
expected_cpu = sys.argv[2] == "local"
if (
    args["model"] != "answerdotai/ModernBERT-base"
    or args["max_steps"] != "5"
    or args["use_cpu"] != str(expected_cpu)
    or (metrics["gpu"] is None) != expected_cpu
    or metrics["log_history"][-1]["step"] != 5
    or sha256_file(run / "best_model/model.safetensors") != metrics["weights_sha256"]
):
    raise SystemExit("Step 1b requires a verified five-step ModernBERT run on the selected backend.")
if "ultrabert" not in ENCODER_MODEL_TYPES:
    raise SystemExit("Step 1b implementation is pending: UltraBERT cannot yet freeze.")
PY
    git diff --cached --binary > "$evidence/step1a.patch"
fi
if [[ "$backend" == modal ]]; then
    bash bins/train/encoder_smoke_modal.sh "$model" "$2" 2>&1 | tee "$evidence/smoke.log"
else
    bash bins/train/encoder_smoke.sh "$model" "$2" 2>&1 | tee "$evidence/smoke.log"
fi
bash bins/ci_tasks/perform_ci.sh 2>&1 | tee "$evidence/ci-after.log"
echo "$backend smoke and CI passed. Run: $output_dir. Evidence: $evidence"
