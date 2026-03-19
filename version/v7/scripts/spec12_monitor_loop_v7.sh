#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$MODEL_CACHE_ROOT/reports/overnight_monitor/spec12}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"

mkdir -p "$SNAPSHOT_DIR"

while true; do
  SNAPSHOT_DIR="$SNAPSHOT_DIR" \
  bash "$ROOT/version/v7/scripts/spec12_monitor_tick_v7.sh" || true

  ACTIVE_SUFFIX="$(sed -n 's/^session_name=ck-v7-spec12-//p' "$SNAPSHOT_DIR/latest.log" | head -n 1)"
  CURRENT_SESSION="ck-v7-spec12-${ACTIVE_SUFFIX:-r3}"

  if command -v tmux >/dev/null 2>&1; then
    if ! tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; then
      break
    fi
  else
    break
  fi
  sleep "$INTERVAL_SECONDS"
done
