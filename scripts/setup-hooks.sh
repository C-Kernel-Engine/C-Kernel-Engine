#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer hooksPath; fall back to symlink if user doesn't want global config change.
git config core.hooksPath .githooks
git config commit.template .gitmessage

echo "✓ Set core.hooksPath to .githooks"
echo "✓ Set commit.template to .gitmessage"
echo "  (Pre-push checks and commit message template are now active)"
