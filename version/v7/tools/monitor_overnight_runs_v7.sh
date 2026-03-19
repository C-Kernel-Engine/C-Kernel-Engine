#!/usr/bin/env bash
set -euo pipefail

PATH="/usr/bin:/bin:/usr/local/bin"
shopt -s nullglob

ROOT="/home/antshiv/Workspace/C-Kernel-Engine"
SESSION_NAME="${SESSION_NAME:-ck-v7-overnight}"
RUN1="${RUN1:-}"
RUN2="${RUN2:-}"
MAIN_LOG="${MAIN_LOG:-}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$HOME/.cache/ck-engine-v7/models/reports/overnight_monitor}"
HISTORY_LOG="${HISTORY_LOG:-$SNAPSHOT_DIR/history.log}"
BASELINE_PROBE="${BASELINE_PROBE:-}"
TARGET_PROBE="${TARGET_PROBE:-}"

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
    "probe.path=" + input_filename,
    "probe.count=" + (((.totals.count // .summary.count) // 0) | tostring),
    "probe.exact_rate=" + (((.totals.exact_rate // .summary.exact_rate) // 0) | tostring),
    "probe.renderable_rate=" + (((.totals.renderable_rate // .summary.renderable_rate) // 0) | tostring),
    "probe.materialized_exact_rate=" + (((.totals.materialized_exact_rate // .summary.materialized_exact_rate) // 0) | tostring)
  ' "$probe_json" 2>/dev/null || true
}

print_stage_summary() {
  local run_dir="$1"
  local stage_json=""
  local candidates=(
    "$run_dir"/train_spec*_stage_b.json
    "$run_dir"/train_spec*_stage_a.json
  )
  if (( ${#candidates[@]} > 0 )); then
    stage_json="${candidates[0]}"
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

print_run_ledger() {
  local ledger_jsonl="$1"
  if [[ ! -f "$ledger_jsonl" ]]; then
    return 0
  fi
  jq -sr '
    if length == 0 then empty else
      .[-1] as $row |
      "ledger.stage_id=" + (($row.stage_id // "unknown") | tostring),
      "ledger.phase_label=" + (($row.phase_label // "unknown") | tostring),
      "ledger.status=" + (($row.status // "unknown") | tostring),
      "ledger.steps=" + (($row.steps // "n/a") | tostring),
      "ledger.total_tokens=" + (($row.total_tokens // "n/a") | tostring),
      "ledger.loss_first=" + (($row.loss_first // "n/a") | tostring),
      "ledger.loss_final=" + (($row.loss_final // "n/a") | tostring),
      "ledger.loss_min=" + (($row.loss_min // "n/a") | tostring)
    end
  ' "$ledger_jsonl" 2>/dev/null || true
}

print_probe_delta() {
  local label="$1"
  local ref_json="$2"
  local current_json="$3"
  if [[ ! -f "$ref_json" || ! -f "$current_json" ]]; then
    return 0
  fi
  jq -nr --arg label "$label" --arg ref_path "$ref_json" --arg cur_path "$current_json" \
    --slurpfile ref_doc "$ref_json" --slurpfile cur_doc "$current_json" '
    def totals(doc): (doc.totals // doc.summary // {});
    def num(v): if (v == null) then 0 else v end;
    (totals($ref_doc[0])) as $ref_totals |
    (totals($cur_doc[0])) as $cur_totals |
    $label + ".path=" + $ref_path,
    $label + ".exact_rate=" + ((num($ref_totals.exact_rate)) | tostring),
    $label + ".renderable_rate=" + ((num($ref_totals.renderable_rate)) | tostring),
    $label + ".materialized_exact_rate=" + ((num($ref_totals.materialized_exact_rate)) | tostring),
    $label + ".delta_exact_rate=" + ((num($cur_totals.exact_rate) - num($ref_totals.exact_rate)) | tostring),
    $label + ".delta_renderable_rate=" + ((num($cur_totals.renderable_rate) - num($ref_totals.renderable_rate)) | tostring),
    $label + ".delta_materialized_exact_rate=" + ((num($cur_totals.materialized_exact_rate) - num($ref_totals.materialized_exact_rate)) | tostring)
  ' 2>/dev/null || true
}

print_run_snapshot() {
  local run_dir="$1"
  local probe_json=""
  echo "== run: $run_dir =="
  if [[ ! -d "$run_dir" ]]; then
    echo "status=missing"
    return 0
  fi

  print_training_plan "$run_dir/training_plan.json"
  print_run_ledger "$run_dir/run_ledger.jsonl"
  print_stage_summary "$run_dir"

  local probe_candidates=( "$run_dir"/spec*_probe_report.json )
  if (( ${#probe_candidates[@]} > 0 )); then
    probe_json="${probe_candidates[0]}"
    print_probe_summary "$probe_json"
  else
    echo "probe.status=not_ready"
  fi

  if [[ -n "$probe_json" ]]; then
    if [[ -n "$BASELINE_PROBE" ]]; then
      print_probe_delta "baseline" "$BASELINE_PROBE" "$probe_json"
    fi
    if [[ -n "$TARGET_PROBE" ]]; then
      print_probe_delta "target" "$TARGET_PROBE" "$probe_json"
    fi
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
