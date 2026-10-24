#!/usr/bin/env bash
# Regression: in a linked worktree, Git exports GIT_DIR to hooks. The hook
# regression scripts build fixture repositories and must not commit into the
# repository named by an inherited GIT_DIR. Run them against a throwaway clone.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
unset $(git rev-parse --local-env-vars)
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
git clone --quiet --no-local "$repo_root" "$tmp_dir/target"
before=$(git -C "$tmp_dir/target" rev-parse HEAD)
cd "$repo_root"
for script in scripts/test-secret-hooks.sh scripts/test-staged-file-size.sh; do
    GIT_DIR="$tmp_dir/target/.git" bash "$script" > "$tmp_dir/log" 2>&1 || {
        cat "$tmp_dir/log" >&2
        echo "FAIL: $script failed with an inherited GIT_DIR" >&2
        exit 1
    }
    after=$(git -C "$tmp_dir/target" rev-parse HEAD)
    if [[ "$after" != "$before" ]]; then
        echo "FAIL: $script committed into the repository named by GIT_DIR" >&2
        exit 1
    fi
done
echo 'PASS: hook regression scripts ignore an inherited GIT_DIR'
