#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$MODEL_CACHE_ROOT/reports/overnight_monitor/spec12}"
TARGET_PROBE="${TARGET_PROBE:-$MODEL_CACHE_ROOT/train/spec11_keyed_scene_dsl_l3_d192_h384_ctx512_r2/spec11_probe_report.json}"

detect_active_rung() {
  local suffix
  for suffix in r5 r4 r3; do
    local candidate="$MODEL_CACHE_ROOT/train/spec12_scene_dsl_l3_d192_h384_ctx768_${suffix}"
    local ledger="$candidate/run_ledger.jsonl"
    if [[ ! -f "$ledger" ]]; then
      continue
    fi
    local status
    status="$(tail -n 1 "$ledger" | jq -r '.status // empty' 2>/dev/null || true)"
    if [[ "$status" == "running" ]]; then
      printf '%s\n' "$suffix"
      return 0
    fi
  done
  for suffix in r5 r4 r3; do
    if [[ -d "$MODEL_CACHE_ROOT/train/spec12_scene_dsl_l3_d192_h384_ctx768_${suffix}" ]]; then
      printf '%s\n' "$suffix"
      return 0
    fi
  done
  printf 'r3\n'
}

prev_rung() {
  case "$1" in
    r5) printf 'r4\n' ;;
    r4) printf 'r3\n' ;;
    *) printf 'r2\n' ;;
  esac
}

ACTIVE_SUFFIX="${ACTIVE_SUFFIX:-$(detect_active_rung)}"
PREV_SUFFIX="${PREV_SUFFIX:-$(prev_rung "$ACTIVE_SUFFIX")}"
SESSION_NAME="${SESSION_NAME:-ck-v7-spec12-${ACTIVE_SUFFIX}}"
MAIN_LOG="${MAIN_LOG:-$MODEL_CACHE_ROOT/reports/spec12_progression/spec12_${ACTIVE_SUFFIX}.log}"
RUN1="${RUN1:-$MODEL_CACHE_ROOT/train/spec12_scene_dsl_l3_d192_h384_ctx768_${ACTIVE_SUFFIX}}"
RUN2="${RUN2:-$MODEL_CACHE_ROOT/train/spec12_scene_dsl_l3_d192_h384_ctx768_${PREV_SUFFIX}}"
BASELINE_PROBE="${BASELINE_PROBE:-$RUN2/spec12_probe_report.json}"

SESSION_NAME="$SESSION_NAME" \
RUN1="$RUN1" \
RUN2="$RUN2" \
MAIN_LOG="$MAIN_LOG" \
SNAPSHOT_DIR="$SNAPSHOT_DIR" \
BASELINE_PROBE="$BASELINE_PROBE" \
TARGET_PROBE="$TARGET_PROBE" \
bash "$ROOT/version/v7/tools/monitor_overnight_runs_v7.sh"
