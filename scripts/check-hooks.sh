#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

hooks_path="$(git config core.hooksPath || true)"
commit_template="$(git config commit.template || true)"

ok=1

if [[ "$hooks_path" != ".githooks" ]]; then
  echo "WARNING: Git hooks are not enabled for this repo."
  echo "  core.hooksPath is not set to .githooks"
  ok=0
fi

if [[ "$commit_template" != ".gitmessage" ]]; then
  echo "WARNING: Commit template is not enabled for this repo."
  echo "  commit.template is not set to .gitmessage"
  ok=0
fi

if [[ "$ok" -eq 1 ]]; then
  exit 0
fi

echo "  Run: ./scripts/setup-hooks.sh"
exit 1
