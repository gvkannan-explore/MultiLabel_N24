#!/usr/bin/env bash
# Five local CPU training steps. Existing output is never removed.
set -euo pipefail
if [[ $# -eq 1 && "$1" == --help ]]; then
    echo "Usage: bash bins/train/encoder_smoke.sh MODEL_ID OUTPUT_DIR"
    exit 0
fi
if [[ $# -ne 2 ]]; then
    echo "Usage: bash bins/train/encoder_smoke.sh MODEL_ID OUTPUT_DIR" >&2
    exit 2
fi
repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$repo_root"
if [[ -e "$2" ]]; then
    echo "Output already exists: $2. Choose a new run directory." >&2
    exit 1
fi
if [[ ! -x .venv/bin/python || ! -d datasets/prepared ]]; then
    echo "Prepare datasets/prepared and run: uv sync --locked --group lint" >&2
    exit 1
fi
exec env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" .venv/bin/python \
    -m n24.train.train_baseline --model "$1" --output-dir "$2" \
    --data-dir datasets/prepared --max-steps 5 --batch-size 2 \
    --max-length 128 --seed 42 --use-cpu
