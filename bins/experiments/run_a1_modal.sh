#!/usr/bin/env bash
# Modal GPU variant of the A1 pilot driver, with the same CI gates.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_a1.sh" --modal "$@"
