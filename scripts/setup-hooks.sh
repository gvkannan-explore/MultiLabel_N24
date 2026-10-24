#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
python3 - <<'PY'
import json
import os
from pathlib import Path
from scripts.check_privacy import local_identities

root = Path.cwd()
identities = local_identities(root)
profile = root / '.privacy-identifiers.json'
fd = os.open(profile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, 'w') as stream:
    json.dump(identities, stream)
os.chmod(profile, 0o600)
PY
git config --local core.hooksPath .githooks
git config --local user.name "$(python3 -c 'from scripts.check_privacy import REPO_NAME; print(REPO_NAME)')"
git config --local user.email "$(python3 -c 'from scripts.check_privacy import REPO_EMAIL; print(REPO_EMAIL)')"
printf 'Hooks enabled. Future commits use the repository-local GitHub noreply identity. Global Git settings are unchanged.\n'
