#!/usr/bin/env bash
# Submit five GPU steps to the existing Modal trainer, then retrieve the run.
set -euo pipefail
usage() {
    echo "Usage: bash bins/train/encoder_smoke_modal.sh MODEL_ID RUN_NAME"
}
if [[ $# -eq 1 && "$1" == --help ]]; then
    usage
    exit 0
fi
if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi
if [[ ! "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9._/-]*$ ]]; then
    echo "MODEL_ID must be a single Hugging Face model identifier." >&2
    exit 2
fi
if [[ ! "$2" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "RUN_NAME must start with a letter or number and contain only letters, numbers, dots, underscores, or hyphens." >&2
    exit 2
fi
repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$repo_root"
if [[ -e "runs_modal/$2" ]]; then
    echo "Output already exists: runs_modal/$2. Choose a new run name." >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "Prepare the environment with: uv sync --locked --group lint" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv before running this script." >&2
    exit 1
fi
echo "Submitting five training steps to Modal's configured GPU (L4/A10G)."
uv run --locked --group lint modal run -m n24.train.modal_train \
    --name "$2" \
    --args "--model $1 --max-steps 5 --batch-size 2 --max-length 128 --seed 42"
mkdir -p runs_modal
uv run --locked --group lint modal volume get n24-runs "$2" runs_modal/
echo "Downloaded run: runs_modal/$2"
.venv/bin/python -m json.tool "runs_modal/$2/metrics.json"
