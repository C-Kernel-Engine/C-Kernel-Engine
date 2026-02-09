#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer hooksPath; fall back to symlink if user doesn't want global config change.
git config core.hooksPath .githooks

echo "✓ Set core.hooksPath to .githooks"
echo "  (Pre-push checks will now run on git push)"
