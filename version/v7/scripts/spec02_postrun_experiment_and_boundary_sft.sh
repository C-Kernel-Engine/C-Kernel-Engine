#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/antshiv/Workspace/C-Kernel-Engine}"
RUN="${RUN:-$HOME/.cache/ck-engine-v7/models/train/svg_l16_d128_h512_v1024_ctx512_spec02}"
MODEL_DIR="${MODEL_DIR:-$RUN/.ck_build}"
SFT_DATA="${SFT_DATA:-}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

resolve_sft_data() {
  local configured=""
  local pipeline="$RUN/training_pipeline_latest.json"
  if [[ -f "$pipeline" ]]; then
    configured="$(jq -r '
      (
        .pipeline.stages[]?
        | select((.stage // .stage_id // "") == "sft")
        | .datasets[]?
        | .path // empty
      ) | select(length>0) | . ' "$pipeline" 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -n "$configured" && -f "$configured" ]]; then
    printf '%s\n' "$configured"
    return 0
  fi
  if [[ -f "$RUN/data/svg_l16_d128_h512_v1024_ctx512_spec02_sft_mix_overnight_v1.txt" ]]; then
    printf '%s\n' "$RUN/data/svg_l16_d128_h512_v1024_ctx512_spec02_sft_mix_overnight_v1.txt"
    return 0
  fi
  if [[ -f "$RUN/data/svg_l16_d128_h512_v1024_ctx512_spec02_stage_b_syn_instruction_train.txt" ]]; then
    printf '%s\n' "$RUN/data/svg_l16_d128_h512_v1024_ctx512_spec02_stage_b_syn_instruction_train.txt"
    return 0
  fi
  return 1
}

wait_for_current_train() {
  local pattern="ck_run_v7.py train --run $RUN"
  while pgrep -af "$pattern" >/dev/null 2>&1; do
    log "current SFT still running; waiting 30s"
    sleep 30
  done
}

run_probe_suite() {
  local out_dir="$1"
  local prompts_file="$out_dir/prompts.txt"
  cat > "$prompts_file" <<'PROMPTS'
[circle][palette:cool][style:minimal]<svg
[rect][palette:warm][style:filled][layout:center]<svg
[bar-chart][bars:5][ascending][palette:warm][axes][style:filled]<svg
[infographic][card][palette:dark][style:outlined][complexity:rich][labeled]<svg
PROMPTS

  local i=0
  while IFS= read -r prompt; do
    [[ -z "$prompt" ]] && continue
    i="$((i+1))"
    local file="$out_dir/probe_$(printf '%02d' "$i").log"
    log "probe $i prompt: $prompt"
    (
      printf 'prompt=%s\n\n' "$prompt"
      "$ROOT/.venv/bin/python" "$ROOT/scripts/ck_chat.py" \
        --model-dir "$MODEL_DIR" \
        --python-tokenizer \
        --chat-template none \
        --prompt "$prompt" \
        --max-tokens 128 \
        --temperature 0.0 \
        --top-p 1.0 \
        --repeat-penalty 1.05 \
        --repeat-last-n 256
    ) > "$file" 2>&1
  done < "$prompts_file"

  local summary="$out_dir/probe_summary.md"
  {
    echo "# Post-Run Probe Summary"
    echo
    echo "- run: \`$RUN\`"
    echo "- model_dir: \`$MODEL_DIR\`"
    echo "- generated_at_utc: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
    echo
    for f in "$out_dir"/probe_*.log; do
      [[ -f "$f" ]] || continue
      local name
      name="$(basename "$f")"
      local response
      response="$(sed -n '/^Response:/,/^prompt eval:/p' "$f" | sed '$d' | sed '1s/^Response: //')"
      echo "## $name"
      echo
      echo '```text'
      if [[ -n "$response" ]]; then
        printf '%s\n' "$response"
      else
        echo "(response not parsed; see raw log)"
      fi
      echo '```'
      echo
    done
  } > "$summary"
}

main() {
  if [[ ! -d "$RUN" ]]; then
    echo "run dir missing: $RUN" >&2
    exit 2
  fi
  if [[ ! -d "$MODEL_DIR/tokenizer_bin" && ! -d "$RUN/tokenizer_bin" ]]; then
    echo "tokenizer_bin missing in $MODEL_DIR or $RUN" >&2
    exit 2
  fi

  local sft_data
  if [[ -n "$SFT_DATA" ]]; then
    if [[ ! -f "$SFT_DATA" ]]; then
      echo "SFT_DATA set but missing: $SFT_DATA" >&2
      exit 2
    fi
    sft_data="$SFT_DATA"
  else
    if ! sft_data="$(resolve_sft_data)"; then
      echo "unable to resolve SFT dataset under $RUN/data or training_pipeline_latest.json" >&2
      exit 2
    fi
  fi

  log "resolved sft_data=$sft_data"
  wait_for_current_train
  log "current SFT ended; starting probe capture"

  local stamp out_dir
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  out_dir="$RUN/experiments/postrun_${stamp}"
  mkdir -p "$out_dir"
  run_probe_suite "$out_dir"
  log "probe capture done: $out_dir"

  log "starting boundary-packed SFT (pack-mode sample, grad_accum=2)"
  "$ROOT/.venv/bin/python" "$ROOT/version/v7/scripts/train_data_pipeline_v7.py" \
    --run "$RUN" \
    --curriculum-stage sft \
    --tokenizer ascii_bpe \
    --reuse-run-tokenizer \
    --strict-data-gates \
    --pack-mode sample \
    --data "$sft_data" \
    --layers 16 \
    --embed-dim 128 \
    --hidden-dim 512 \
    --num-heads 8 \
    --num-kv-heads 4 \
    --context-len 512 \
    --optimizer adamw \
    --epochs 1 \
    --seq-len 512 \
    --total-tokens 16777216 \
    --grad-accum 2 \
    --lr 8e-5 \
    --no-open-visualizer \
    --json-out "$RUN/experiments/boundary_sft_report_latest.json"

  log "boundary-packed SFT completed; regenerating ir_report.html"
  python3 "$ROOT/version/v7/tools/open_ir_visualizer.py" \
    --generate \
    --run "$RUN" \
    --html-only \
    --strict-run-artifacts

  log "done"
}

main "$@"
