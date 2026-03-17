#!/usr/bin/env bash
set -euo pipefail

PATH="/usr/bin:/bin:/usr/local/bin"

ROOT="/home/antshiv/Workspace/C-Kernel-Engine"
SESSION_NAME="${SESSION_NAME:-ck-v7-overnight}"
RUN1="${RUN1:-}"
RUN2="${RUN2:-}"
MAIN_LOG="${MAIN_LOG:-}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$HOME/.cache/ck-engine-v7/models/reports/overnight_monitor}"
HISTORY_LOG="${HISTORY_LOG:-$SNAPSHOT_DIR/history.log}"

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
timestamp_slug="$(date -u +"%Y%m%dT%H%M%SZ")"
mkdir -p "$SNAPSHOT_DIR"
snapshot_path="$SNAPSHOT_DIR/$timestamp_slug.log"

print_probe_summary() {
  local probe_json="$1"
  if [[ ! -f "$probe_json" ]]; then
    return 0
  fi
  jq -r '
    "probe.count=" + ((.totals.count // 0) | tostring),
    "probe.exact_rate=" + ((.totals.exact_rate // 0) | tostring),
    "probe.renderable_rate=" + ((.totals.renderable_rate // 0) | tostring)
  ' "$probe_json" 2>/dev/null || true
}

print_stage_summary() {
  local run_dir="$1"
  local stage_json=""
  if [[ -f "$run_dir/train_spec07_stage_b.json" ]]; then
    stage_json="$run_dir/train_spec07_stage_b.json"
  elif [[ -f "$run_dir/train_spec06_stage_b.json" ]]; then
    stage_json="$run_dir/train_spec06_stage_b.json"
  elif [[ -f "$run_dir/train_spec07_stage_a.json" ]]; then
    stage_json="$run_dir/train_spec07_stage_a.json"
  elif [[ -f "$run_dir/train_spec06_stage_a.json" ]]; then
    stage_json="$run_dir/train_spec06_stage_a.json"
  fi
  if [[ -z "$stage_json" ]]; then
    return 0
  fi
  jq -r '
    "stage.json=" + input_filename,
    "stage.final_loss=" + ((.ck_loss.final // .final_loss // .ck_loss_final // .loss_final // "n/a") | tostring),
    "stage.min_loss=" + ((.ck_loss.min // .min_loss // .ck_loss_min // "n/a") | tostring)
  ' "$stage_json" 2>/dev/null || true
}

print_training_plan() {
  local plan_json="$1"
  if [[ ! -f "$plan_json" ]]; then
    return 0
  fi
  jq -r '
    "plan.active_stage=" + ((.active_stage // "unknown") | tostring),
    "plan.stage_count=" + ((.stages // []) | length | tostring)
  ' "$plan_json" 2>/dev/null || true
}

print_run_snapshot() {
  local run_dir="$1"
  echo "== run: $run_dir =="
  if [[ ! -d "$run_dir" ]]; then
    echo "status=missing"
    return 0
  fi

  print_training_plan "$run_dir/training_plan.json"
  print_stage_summary "$run_dir"

  if [[ -f "$run_dir/spec07_probe_report.json" ]]; then
    print_probe_summary "$run_dir/spec07_probe_report.json"
  elif [[ -f "$run_dir/spec06_probe_report.json" ]]; then
    print_probe_summary "$run_dir/spec06_probe_report.json"
  else
    echo "probe.status=not_ready"
  fi

}

{
  echo "timestamp_utc=$timestamp_utc"
  echo "session_name=$SESSION_NAME"
  if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "tmux_session=present"
    else
      echo "tmux_session=absent_or_inaccessible"
    fi
  else
    echo "tmux_session=tmux_not_installed"
  fi

  if [[ -n "$MAIN_LOG" && -f "$MAIN_LOG" ]]; then
    echo "-- main_log_tail --"
    tail -n 40 "$MAIN_LOG" || true
  fi

  if [[ -n "$RUN1" ]]; then
    print_run_snapshot "$RUN1"
  fi
  if [[ -n "$RUN2" ]]; then
    print_run_snapshot "$RUN2"
  fi
} > "$snapshot_path"

ln -sfn "$snapshot_path" "$SNAPSHOT_DIR/latest.log"
cat "$snapshot_path" >> "$HISTORY_LOG"
