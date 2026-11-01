#!/usr/bin/env bash
# Score Qwen3-0.6B and SmolLM2-360M-Instruct zero-shot on the test split, each
# on a Modal GPU (n24/eval/zero_shot.py). Results go to results/llm/.
#
# Usage: bash bins/eval/zero_shot.sh [model ...]
#   bash bins/eval/zero_shot.sh
#   bash bins/eval/zero_shot.sh Qwen/Qwen3-0.6B
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
models=("$@")
if (( ${#models[@]} == 0 )); then
    models=(Qwen/Qwen3-0.6B HuggingFaceTB/SmolLM2-360M-Instruct)
fi
for model in "${models[@]}"; do
    uv run modal run -m n24.eval.zero_shot --model "$model"
done
