#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

hooks_path="$(git config core.hooksPath || true)"

if [[ "$hooks_path" == ".githooks" ]]; then
  exit 0
fi

echo "WARNING: Git hooks are not enabled for this repo."
echo "  core.hooksPath is not set to .githooks"
echo "  Run: ./scripts/setup-hooks.sh"
exit 1
