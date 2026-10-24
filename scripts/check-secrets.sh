#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
scanner="$repo_root/.tools/bin/gitleaks"
if [[ ! -x "$scanner" ]]; then
    if command -v gitleaks >/dev/null 2>&1; then
        scanner=$(command -v gitleaks)
    else
        echo 'Install Gitleaks with bash scripts/install-gitleaks.sh.' >&2
        exit 1
    fi
fi
cd "$repo_root"
if [[ $# -eq 0 ]]; then
    # Scan what this repository can publish. --all would also cover remote-tracking
    # refs and app checkpoint refs (refs/t3/*) that this repository never pushes.
    set -- '--log-opts=--branches --tags HEAD'
fi
exec "$scanner" git --redact --verbose --no-banner "$@" .
