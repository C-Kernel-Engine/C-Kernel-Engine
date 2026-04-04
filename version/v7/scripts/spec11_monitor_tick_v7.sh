#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
SESSION_NAME="${SESSION_NAME:-ck-v7-spec11-r2}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$MODEL_CACHE_ROOT/reports/overnight_monitor/spec11}"
MAIN_LOG="${MAIN_LOG:-$MODEL_CACHE_ROOT/reports/spec11/spec11_r2_train.log}"
BASELINE_PROBE="${BASELINE_PROBE:-$MODEL_CACHE_ROOT/train/spec11_keyed_scene_dsl_l3_d192_h384_ctx512_r1/spec11_probe_report.json}"
TARGET_PROBE="${TARGET_PROBE:-$MODEL_CACHE_ROOT/train/spec10_asset_scene_dsl_l3_d192_h384_ctx512_r4/spec10_probe_report.json}"
RUN1="${RUN1:-$MODEL_CACHE_ROOT/train/spec11_keyed_scene_dsl_l3_d192_h384_ctx512_r2}"
RUN2="${RUN2:-}"

SESSION_NAME="$SESSION_NAME" \
RUN1="$RUN1" \
RUN2="$RUN2" \
MAIN_LOG="$MAIN_LOG" \
SNAPSHOT_DIR="$SNAPSHOT_DIR" \
BASELINE_PROBE="$BASELINE_PROBE" \
TARGET_PROBE="$TARGET_PROBE" \
bash "$ROOT/version/v7/tools/monitor_overnight_runs_v7.sh"
