#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  exec "$ROOT/.venv/bin/python" "$ROOT/version/v7/scripts/v7_doctor.py" "$@"
fi

exec python3 "$ROOT/version/v7/scripts/v7_doctor.py" "$@"
