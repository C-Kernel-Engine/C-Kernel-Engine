#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
SESSION_NAME="${SESSION_NAME:-ck-v7-spec10-overnight}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$MODEL_CACHE_ROOT/reports/overnight_monitor/spec10}"
MAIN_LOG="${MAIN_LOG:-$MODEL_CACHE_ROOT/reports/spec10_progression/spec10_progression_overnight.log}"
BASELINE_PROBE="${BASELINE_PROBE:-$MODEL_CACHE_ROOT/train/spec10_asset_scene_dsl_l3_d192_h384_ctx512_r1/spec10_probe_report.json}"
TARGET_PROBE="${TARGET_PROBE:-}"
RUN1="${RUN1:-$MODEL_CACHE_ROOT/train/spec10_asset_scene_dsl_l3_d192_h384_ctx512_r4}"
RUN2="${RUN2:-$MODEL_CACHE_ROOT/train/spec10_asset_scene_dsl_l3_d192_h384_ctx512_r5}"

SESSION_NAME="$SESSION_NAME" \
RUN1="$RUN1" \
RUN2="$RUN2" \
MAIN_LOG="$MAIN_LOG" \
SNAPSHOT_DIR="$SNAPSHOT_DIR" \
BASELINE_PROBE="$BASELINE_PROBE" \
TARGET_PROBE="$TARGET_PROBE" \
bash "$ROOT/version/v7/tools/monitor_overnight_runs_v7.sh"
