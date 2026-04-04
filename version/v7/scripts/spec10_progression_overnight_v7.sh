#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
RUN_ROOT="${RUN_ROOT:-$MODEL_CACHE_ROOT/train}"
LOG_DIR="${LOG_DIR:-$MODEL_CACHE_ROOT/reports/spec10_progression}"
mkdir -p "$RUN_ROOT" "$LOG_DIR"
MASTER_LOG="${MASTER_LOG:-$LOG_DIR/spec10_progression_overnight.log}"
START_RUNG="${START_RUNG:-r3}"

SUCCESS_EXACT_RATE="${SUCCESS_EXACT_RATE:-0.33}"
SUCCESS_RENDERABLE_RATE="${SUCCESS_RENDERABLE_RATE:-0.90}"
SUCCESS_MATERIALIZED_RATE="${SUCCESS_MATERIALIZED_RATE:-0.33}"

write_probe_status() {
  local run_dir="$1"
  local status="$2"
  local trusted="$3"
  local note="$4"
  python3 - "$run_dir" "$status" "$trusted" "$note" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir, status, trusted_raw, note = sys.argv[1:]
payload = {
    "schema": "ck.probe_status.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_dir": run_dir,
    "status": status,
    "trusted": trusted_raw.lower() == "true",
    "note": note,
}
Path(run_dir, "spec10_probe_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_variant() {
  local suffix="$1"
  shift
  local run_dir="$RUN_ROOT/spec10_asset_scene_dsl_l3_d192_h384_ctx512_${suffix}"
  local log_path="$LOG_DIR/spec10_${suffix}.log"
  echo "[spec10-progress] starting ${suffix} -> $run_dir"
  set +e
  (
    cd "$ROOT"
    env RUN_PARITY=0 \
      RUN_STAGE_EVAL=0 \
      PROBE_PER_SPLIT=12 \
      CANARY_PER_SPLIT=4 \
      "$@" \
      bash version/v7/scripts/spec10_pretrain_midtrain_v7.sh "$run_dir"
  ) | tee "$log_path" | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[spec10-progress] ${suffix} run script exited with rc=$rc" | tee -a "$MASTER_LOG"
  fi
  return $rc
}

ensure_probe_artifacts() {
  local suffix="$1"
  local run_dir="$RUN_ROOT/spec10_asset_scene_dsl_l3_d192_h384_ctx512_${suffix}"
  local probe_json="$run_dir/spec10_probe_report.json"
  if [[ -f "$probe_json" ]]; then
    return 0
  fi
  if [[ ! -d "$run_dir" ]]; then
    return 1
  fi
  echo "[spec10-progress] backfilling missing probe artifacts for ${suffix}" | tee -a "$MASTER_LOG"
  set +e
  (
    cd "$ROOT"
    python3 version/v7/scripts/build_spec10_probe_contract_v7.py \
      --run "$run_dir" \
      --prefix spec10_asset_scene_dsl \
      --output "$run_dir/spec10_probe_contract.json" \
      --per-split 12
    .venv/bin/python version/v7/scripts/ck_run_v7.py run "$run_dir" \
      --generate-only \
      --context-len 512
    python3 version/v7/scripts/build_probe_report_v7.py \
      --run "$run_dir" \
      --contract "$run_dir/spec10_probe_contract.json" \
      --output "$run_dir/spec10_probe_report.html" \
      --json-out "$run_dir/spec10_probe_report.json"
    python3 version/v7/scripts/build_structured_scene_tested_prompts_doc_v7.py \
      --probe-report "$run_dir/spec10_probe_report.json" \
      --output-html "$run_dir/spec10_tested_prompts_report.html" \
      --output-md "$run_dir/spec10_tested_prompts_report.md"
  ) | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  return $rc
}

probe_ok() {
  local probe_json="$1"
  python3 - "$probe_json" "$SUCCESS_EXACT_RATE" "$SUCCESS_RENDERABLE_RATE" "$SUCCESS_MATERIALIZED_RATE" <<'PY'
import json
import sys

probe_path, exact_s, renderable_s, materialized_s = sys.argv[1:]
with open(probe_path, encoding="utf-8") as f:
    data = json.load(f)
totals = data.get("totals") or {}
exact = float(totals.get("exact_rate") or 0.0)
renderable = float(totals.get("renderable_rate") or 0.0)
materialized = float(totals.get("materialized_exact_rate") or 0.0)
ok = (
    exact >= float(exact_s)
    and renderable >= float(renderable_s)
    and materialized >= float(materialized_s)
)
print(json.dumps({
    "exact_rate": exact,
    "renderable_rate": renderable,
    "materialized_exact_rate": materialized,
    "pass": ok,
}, indent=2))
raise SystemExit(0 if ok else 1)
PY
}

should_run_rung() {
  local suffix="$1"
  case "$START_RUNG" in
    r3) return 0 ;;
    r4) [[ "$suffix" != "r3" ]] ;;
    r5) [[ "$suffix" == "r5" ]] ;;
    *) return 0 ;;
  esac
}

run_and_gate() {
  local suffix="$1"
  shift
  local run_dir="$RUN_ROOT/spec10_asset_scene_dsl_l3_d192_h384_ctx512_${suffix}"
  local run_rc=0
  if ! run_variant "$suffix" "$@"; then
    run_rc=1
    echo "[spec10-progress] ${suffix} will still attempt probe backfill after run failure" | tee -a "$MASTER_LOG"
  fi
  ensure_probe_artifacts "$suffix" || true
  local probe_json="$run_dir/spec10_probe_report.json"
  if [[ ! -f "$probe_json" ]]; then
    echo "[spec10-progress] missing probe report for ${suffix}: $probe_json" >&2
    return 1
  fi
  if [[ $run_rc -ne 0 ]]; then
    write_probe_status "$run_dir" "diagnostic_backfill" "false" \
      "Probe artifacts were backfilled after the run script failed. Treat this probe as diagnostic only, not as a canonical completed-run result."
  else
    write_probe_status "$run_dir" "primary_run" "true" \
      "Probe artifacts were produced from the primary run path and can be used as the canonical run result."
  fi
  if probe_ok "$probe_json"; then
    echo "[spec10-progress] ${suffix} met success thresholds"
    return 0
  fi
  echo "[spec10-progress] ${suffix} below success thresholds"
  return 1
}

if should_run_rung r3; then
  if run_and_gate r3 \
    TRAIN_REPEATS=4 \
    HOLDOUT_REPEATS=1 \
    PRETRAIN_EPOCHS=3 \
    MIDTRAIN_EPOCHS=3 \
    MIDTRAIN_DIRECT_REPEAT=4 \
    MIDTRAIN_EDIT_REPEAT=2 \
    MIDTRAIN_CLOSE_REPEAT=6; then
    exit 0
  fi
fi

if should_run_rung r4; then
  if run_and_gate r4 \
    TRAIN_REPEATS=4 \
    HOLDOUT_REPEATS=1 \
    PRETRAIN_EPOCHS=3 \
    MIDTRAIN_EPOCHS=4 \
    MIDTRAIN_DIRECT_REPEAT=5 \
    MIDTRAIN_EDIT_REPEAT=3 \
    MIDTRAIN_CLOSE_REPEAT=8; then
    exit 0
  fi
fi

if should_run_rung r5; then
  run_and_gate r5 \
    TRAIN_REPEATS=5 \
    HOLDOUT_REPEATS=1 \
    PRETRAIN_EPOCHS=4 \
    MIDTRAIN_EPOCHS=5 \
    MIDTRAIN_DIRECT_REPEAT=6 \
    MIDTRAIN_EDIT_REPEAT=4 \
    MIDTRAIN_CLOSE_REPEAT=10
fi
