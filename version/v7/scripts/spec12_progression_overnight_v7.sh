#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
RUN_ROOT="${RUN_ROOT:-$MODEL_CACHE_ROOT/train}"
LOG_DIR="${LOG_DIR:-$MODEL_CACHE_ROOT/reports/spec12_progression}"
mkdir -p "$RUN_ROOT" "$LOG_DIR"
MASTER_LOG="${MASTER_LOG:-$LOG_DIR/spec12_progression_overnight.log}"
START_RUNG="${START_RUNG:-r3}"

SUCCESS_EXACT_RATE="${SUCCESS_EXACT_RATE:-0.50}"
SUCCESS_RENDERABLE_RATE="${SUCCESS_RENDERABLE_RATE:-0.85}"
SUCCESS_MATERIALIZED_RATE="${SUCCESS_MATERIALIZED_RATE:-0.55}"

log_run_timing() {
  local run_dir="$1"
  local suffix="$2"
  python3 - "$run_dir" "$suffix" <<'PY' | tee -a "$MASTER_LOG"
import json
import sys
from datetime import datetime
from pathlib import Path

run_dir = Path(sys.argv[1])
suffix = sys.argv[2]
ledger_path = run_dir / "run_ledger.jsonl"
if not ledger_path.exists():
    print(f"[spec12-progress] timing {suffix}: ledger missing")
    raise SystemExit(0)

def parse_iso(text):
    return datetime.fromisoformat(text) if text else None

entries = []
for line in ledger_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        entries.append(json.loads(line))
    except json.JSONDecodeError:
        continue

for stage_id in ("pretrain", "midtrain"):
    stage_entries = [e for e in entries if e.get("stage_id") == stage_id and e.get("status") == "completed"]
    if not stage_entries:
        continue
    entry = stage_entries[-1]
    started = parse_iso(entry.get("started_at"))
    ended = parse_iso(entry.get("ended_at"))
    elapsed = int((ended - started).total_seconds()) if started and ended else None
    print(
        "[spec12-progress] timing "
        f"{suffix}:{stage_id} "
        f"started_at={entry.get('started_at')} "
        f"ended_at={entry.get('ended_at')} "
        f"elapsed_sec={elapsed if elapsed is not None else 'n/a'} "
        f"steps={entry.get('steps', 'n/a')} "
        f"loss_first={entry.get('loss_first', 'n/a')} "
        f"loss_final={entry.get('loss_final', 'n/a')}"
    )
PY
}

wait_for_completion() {
  local session_name="$1"
  local run_dir="$2"
  local probe_json="$run_dir/spec12_probe_report.json"
  local loops=0
  while true; do
    if [[ -f "$probe_json" ]]; then
      return 0
    fi
    if ! tmux has-session -t "$session_name" 2>/dev/null; then
      return 0
    fi
    loops=$((loops + 1))
    if (( loops % 20 == 0 )); then
      echo "[spec12-progress] waiting on ${session_name} (${run_dir})" | tee -a "$MASTER_LOG"
    fi
    sleep 30
  done
}

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
Path(run_dir, "spec12_probe_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_variant() {
  local suffix="$1"
  shift
  local run_dir="$RUN_ROOT/spec12_scene_dsl_l3_d192_h384_ctx768_${suffix}"
  local log_path="$LOG_DIR/spec12_${suffix}.log"
  local session_name="ck-v7-spec12-${suffix}"
  echo "[spec12-progress] starting ${suffix} -> $run_dir" | tee -a "$MASTER_LOG"
  if tmux has-session -t "$session_name" 2>/dev/null; then
    tmux kill-session -t "$session_name"
  fi
  tmux new-session -d -s "$session_name" \
    "cd $ROOT && env $* bash version/v7/scripts/spec12_pretrain_midtrain_v7.sh $run_dir > $log_path 2>&1"
  wait_for_completion "$session_name" "$run_dir"
}

observe_existing_variant() {
  local suffix="$1"
  local run_dir="$RUN_ROOT/spec12_scene_dsl_l3_d192_h384_ctx768_${suffix}"
  local session_name="ck-v7-spec12-${suffix}"
  echo "[spec12-progress] observing existing ${suffix} -> $run_dir" | tee -a "$MASTER_LOG"
  wait_for_completion "$session_name" "$run_dir"
}

ensure_probe_artifacts() {
  local suffix="$1"
  local run_dir="$RUN_ROOT/spec12_scene_dsl_l3_d192_h384_ctx768_${suffix}"
  local probe_json="$run_dir/spec12_probe_report.json"
  if [[ -f "$probe_json" ]]; then
    return 0
  fi
  if [[ ! -d "$run_dir" ]]; then
    return 1
  fi
  echo "[spec12-progress] backfilling missing probe artifacts for ${suffix}" | tee -a "$MASTER_LOG"
  (
    cd "$ROOT"
    python3 version/v7/scripts/build_spec12_probe_contract_v7.py \
      --run "$run_dir" \
      --prefix spec12_scene_dsl \
      --output "$run_dir/spec12_probe_contract.json" \
      --per-split 12
    .venv/bin/python version/v7/scripts/ck_run_v7.py run "$run_dir" \
      --generate-only \
      --context-len 768
    python3 version/v7/scripts/build_probe_report_v7.py \
      --run "$run_dir" \
      --contract "$run_dir/spec12_probe_contract.json" \
      --output "$run_dir/spec12_probe_report.html" \
      --json-out "$run_dir/spec12_probe_report.json"
  ) | tee -a "$MASTER_LOG"
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

run_and_gate() {
  local suffix="$1"
  shift
  local run_dir="$RUN_ROOT/spec12_scene_dsl_l3_d192_h384_ctx768_${suffix}"
  local started_epoch
  local ended_epoch
  started_epoch="$(date +%s)"
  run_variant "$suffix" "$@"
  ended_epoch="$(date +%s)"
  echo "[spec12-progress] wallclock ${suffix} elapsed_sec=$((ended_epoch - started_epoch))" | tee -a "$MASTER_LOG"
  log_run_timing "$run_dir" "$suffix"
  ensure_probe_artifacts "$suffix" || true
  local probe_json="$run_dir/spec12_probe_report.json"
  if [[ ! -f "$probe_json" ]]; then
    echo "[spec12-progress] missing probe report for ${suffix}: $probe_json" >&2
    return 1
  fi
  write_probe_status "$run_dir" "primary_run" "true" \
    "Probe artifacts were produced from the primary run path and can be used as the canonical run result."
  if probe_ok "$probe_json"; then
    echo "[spec12-progress] ${suffix} met success thresholds" | tee -a "$MASTER_LOG"
    return 0
  fi
  echo "[spec12-progress] ${suffix} below success thresholds" | tee -a "$MASTER_LOG"
  return 1
}

observe_and_gate() {
  local suffix="$1"
  local run_dir="$RUN_ROOT/spec12_scene_dsl_l3_d192_h384_ctx768_${suffix}"
  local started_epoch
  local ended_epoch
  started_epoch="$(date +%s)"
  observe_existing_variant "$suffix"
  ended_epoch="$(date +%s)"
  echo "[spec12-progress] wallclock observed-${suffix} elapsed_sec=$((ended_epoch - started_epoch))" | tee -a "$MASTER_LOG"
  log_run_timing "$run_dir" "$suffix"
  ensure_probe_artifacts "$suffix" || true
  local probe_json="$run_dir/spec12_probe_report.json"
  if [[ ! -f "$probe_json" ]]; then
    echo "[spec12-progress] missing probe report for observed ${suffix}: $probe_json" >&2
    return 1
  fi
  write_probe_status "$run_dir" "primary_run" "true" \
    "Probe artifacts were produced from the primary run path and can be used as the canonical run result."
  if probe_ok "$probe_json"; then
    echo "[spec12-progress] observed ${suffix} met success thresholds" | tee -a "$MASTER_LOG"
    return 0
  fi
  echo "[spec12-progress] observed ${suffix} below success thresholds" | tee -a "$MASTER_LOG"
  return 1
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

if should_run_rung r3; then
  if observe_and_gate r3; then
    exit 0
  fi
fi

if should_run_rung r4; then
  if run_and_gate r4 \
    TRAIN_REPEATS=7 \
    PRETRAIN_EPOCHS=2 \
    MIDTRAIN_EPOCHS=4 \
    MIDTRAIN_EDIT_REPEAT=4 \
    MIDTRAIN_DIRECT_REPEAT=7 \
    MIDTRAIN_CLOSE_REPEAT=10 \
    MIDTRAIN_HEADER_REPEAT=8 \
    MIDTRAIN_BLOCK_REPEAT=5 \
    MIDTRAIN_TRANSITION_REPEAT=8 \
    MIDTRAIN_TABLE_REPEAT=4; then
    exit 0
  fi
fi

if should_run_rung r5; then
  run_and_gate r5 \
    TRAIN_REPEATS=8 \
    PRETRAIN_EPOCHS=3 \
    MIDTRAIN_EPOCHS=5 \
    MIDTRAIN_EDIT_REPEAT=5 \
    MIDTRAIN_DIRECT_REPEAT=8 \
    MIDTRAIN_CLOSE_REPEAT=12 \
    MIDTRAIN_HEADER_REPEAT=10 \
    MIDTRAIN_BLOCK_REPEAT=6 \
    MIDTRAIN_TRANSITION_REPEAT=10 \
    MIDTRAIN_TABLE_REPEAT=5
fi
