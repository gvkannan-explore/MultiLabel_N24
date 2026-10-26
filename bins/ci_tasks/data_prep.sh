#!/usr/bin/env bash
# Rebuild datasets/prepared from the raw news files in datasets/news.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
rm -rf datasets/prepared
uv run python -m src.data.prepare_data --input-dir datasets/news --output-dir datasets/prepared
