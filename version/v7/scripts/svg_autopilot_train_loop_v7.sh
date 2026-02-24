#!/usr/bin/env bash
set -euo pipefail

# Nightly SVG training autopilot:
# - expands dataset each iteration
# - resumes from latest checkpoint in same run
# - records metrics in JSONL for next-day inspection

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

RUN_DEFAULT="${HOME}/.cache/ck-engine-v7/models/train/v7_svg_autopilot_seq128"
RUN="${RUN:-$RUN_DEFAULT}"
HOURS="${HOURS:-6}"
SEQ_LEN="${SEQ_LEN:-128}"
TOTAL_TOKENS="${TOTAL_TOKENS:-1200000}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR_STAGE_B="${LR_STAGE_B:-2e-4}"
LR_SFT="${LR_SFT:-1.5e-4}"
BASE_SAMPLES="${BASE_SAMPLES:-20000}"
SAMPLE_STEP="${SAMPLE_STEP:-5000}"
TARGET_VOCAB="${TARGET_VOCAB:-640}"
SEED="${SEED:-42}"

DATA_DIR="$ROOT/version/v7/data"
LOG_DIR="$RUN/autopilot"
mkdir -p "$LOG_DIR" "$DATA_DIR"

LOG_TXT="$LOG_DIR/autopilot.log"
SUMMARY_JSONL="$LOG_DIR/summary.jsonl"
STATUS_JSON="$LOG_DIR/status.json"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_TXT"
}

run_vocab() {
  if [ -f "$RUN/weights_manifest.json" ]; then
    jq -r ".config.vocab_size // ${TARGET_VOCAB}" "$RUN/weights_manifest.json"
  else
    echo "$TARGET_VOCAB"
  fi
}

resume_args() {
  if compgen -G "$RUN/checkpoints/weights_step_*.bump" >/dev/null 2>&1; then
    echo "--resume-latest-checkpoint"
  fi
}

write_status() {
  local phase="$1"
  local iter="$2"
  local note="$3"
  cat >"$STATUS_JSON" <<EOF
{
  "run": "$RUN",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "phase": "$phase",
  "iteration": $iter,
  "log": "$LOG_TXT",
  "summary_jsonl": "$SUMMARY_JSONL",
  "note": "$note"
}
EOF
}

if [ ! -s "$DATA_DIR/svg_assets_docs_ascii.txt" ]; then
  log "building docs ascii svg corpus..."
  "$PY" version/v7/scripts/build_svg_corpus_from_assets_v7.py \
    --assets-glob "$ROOT/docs/site/assets/*.svg" \
    --output "$DATA_DIR/svg_assets_docs_ascii.txt" \
    --manifest "$DATA_DIR/svg_assets_docs_ascii_manifest.json" \
    --ascii-map-common \
    --ascii-mode xml_escape \
    --no-dedupe
fi

if [ ! -s "$DATA_DIR/svg_stage_a_plus_bridge_small.txt" ]; then
  log "missing $DATA_DIR/svg_stage_a_plus_bridge_small.txt; falling back to svg_assets_train.txt"
fi

mkdir -p "$RUN"
touch "$SUMMARY_JSONL"
log "autopilot start run=$RUN hours=$HOURS seq_len=$SEQ_LEN grad_accum=$GRAD_ACCUM"
write_status "starting" 0 "autopilot boot"

START_TS="$(date +%s)"
END_TS=$((START_TS + HOURS * 3600))
iter=0

while [ "$(date +%s)" -lt "$END_TS" ]; do
  iter=$((iter + 1))
  iter_seed=$((SEED + iter))
  samples=$((BASE_SAMPLES + (iter - 1) * SAMPLE_STEP))
  prefix="svg_autopilot_${iter}"
  iter_dir="$RUN/autopilot/iter_${iter}"
  mkdir -p "$iter_dir"

  write_status "build_data" "$iter" "generating synthetic corpus"
  log "iter=$iter generating synthetic svg samples=$samples seed=$iter_seed"
  "$PY" version/v7/scripts/generate_svg_instruction_dataset_v7.py \
    --out-dir "$DATA_DIR" \
    --prefix "$prefix" \
    --num-samples "$samples" \
    --holdout-ratio 0.10 \
    --seed "$iter_seed" \
    --fill-mode mixed

  svg_sources=(
    "$DATA_DIR/svg_stage_a_plus_bridge_small.txt"
    "$DATA_DIR/svg_stage_a_bridge_small.txt"
    "$DATA_DIR/svg_stage_a_bridge_pack.txt"
    "$DATA_DIR/svg_assets_train.txt"
    "$DATA_DIR/svg_assets_docs_ascii.txt"
    "$DATA_DIR/svg_assets_ascii_all.txt"
    "$DATA_DIR/svg_assets_ascii_le4096_vocab2048.txt"
    "$DATA_DIR/svg_instruction_10k_svg_train.txt"
    "$DATA_DIR/svg_instruction_aug_svg_train.txt"
    "$DATA_DIR/svg_instruction_ui_1k_svg_train.txt"
  )
  syn_svg="$DATA_DIR/${prefix}_svg_train.txt"
  train_svg="$iter_dir/svg_train_stageb.txt"
  svg_existing=()
  for src in "${svg_sources[@]}"; do
    if [ -s "$src" ]; then
      svg_existing+=("$src")
    fi
  done
  svg_existing+=("$syn_svg")
  if [ "${#svg_existing[@]}" -eq 0 ]; then
    log "ERROR: no SVG sources available"
    break
  fi

  cat "${svg_existing[@]}" | sed '/^[[:space:]]*$/d' | awk 'length($0) <= 4096' | sort -u > "$train_svg"

  "$PY" version/v7/scripts/prepare_ascii_dataset_v7.py \
    --input "$train_svg" \
    --output "$train_svg" \
    --input-format text \
    --ascii-map-common \
    --ascii-mode xml_escape \
    --svg-only

  V="$(run_vocab)"
  report_stageb="$iter_dir/train_stageb_report.json"
  write_status "train_stageb" "$iter" "resuming checkpoint and running stage_b"
  log "iter=$iter train stage_b vocab=$V svg_sources=${#svg_existing[@]} data=$(wc -l <"$train_svg") lines"
  RESUME_FLAG="$(resume_args || true)"
  if [ -n "$RESUME_FLAG" ]; then
    log "iter=$iter resume mode: latest checkpoint"
  else
    log "iter=$iter resume mode: fresh-weights (no checkpoints yet)"
  fi

  set +e
  "$PY" version/v7/scripts/train_data_pipeline_v7.py \
    --run "$RUN" \
    --init-if-missing \
    ${RESUME_FLAG:+$RESUME_FLAG} \
    --template qwen3 \
    --curriculum-stage stage_b \
    --tokenizer ascii_bpe \
    --require-svg-rows \
    --strict-data-gates \
    --min-valid-svg-rate 0.70 \
    --roundtrip-max-lines 2048 \
    --roundtrip-sample-limit 16 \
    --data "$train_svg" \
    --vocab-size "$V" \
    --bpe-vocab-size "$V" \
    --layers 24 \
    --embed-dim 64 \
    --hidden-dim 128 \
    --epochs 1 \
    --seq-len "$SEQ_LEN" \
    --total-tokens "$TOTAL_TOKENS" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR_STAGE_B" \
    --max-grad-norm 1.0 \
    --seed "$iter_seed" \
    --train-driver ck_cli \
    --ck-cli-log-every 200 \
    --json-out "$report_stageb"
  rc_stageb=$?
  set -e

  valid_rate=0
  closure_rate=0
  final_loss=0
  if [ -f "$report_stageb" ]; then
    valid_rate="$(jq -r '.post_train_eval.valid_svg_rate // 0' "$report_stageb")"
    closure_rate="$(jq -r '.post_train_eval.closure_success_rate // 0' "$report_stageb")"
    final_loss="$(jq -r '.ck_loss.final // 0' "$report_stageb")"
  fi

  jq -nc \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg run "$RUN" \
    --arg phase "stage_b" \
    --argjson iter "$iter" \
    --argjson rc "$rc_stageb" \
    --arg data "$train_svg" \
    --arg report "$report_stageb" \
    --argjson valid_svg_rate "$valid_rate" \
    --argjson closure_rate "$closure_rate" \
    --argjson final_loss "$final_loss" \
    '{ts:$ts,run:$run,phase:$phase,iter:$iter,rc:$rc,data:$data,report:$report,valid_svg_rate:$valid_svg_rate,closure_rate:$closure_rate,final_loss:$final_loss}' \
    >>"$SUMMARY_JSONL"

  log "iter=$iter stage_b rc=$rc_stageb valid_svg_rate=$valid_rate closure_rate=$closure_rate final_loss=$final_loss"

  # Trigger SFT phase whenever output-quality gates are in a decent range.
  if "$PY" - <<PY >/dev/null 2>&1
v=float("$valid_rate")
c=float("$closure_rate")
raise SystemExit(0 if (v >= 0.70 and c >= 0.70) else 1)
PY
  then
    write_status "train_sft" "$iter" "running one sft pass"
    report_sft="$iter_dir/train_sft_report.json"
    sft_data="$iter_dir/sft_instruction_mix.txt"
    instruction_sources=(
      "$DATA_DIR/svg_instruction_10k_instruction_train.txt"
      "$DATA_DIR/svg_instruction_aug_instruction_train.txt"
      "$DATA_DIR/svg_instruction_ui_1k_instruction_train.txt"
      "$DATA_DIR/${prefix}_instruction_train.txt"
    )
    instr_existing=()
    for src in "${instruction_sources[@]}"; do
      if [ -s "$src" ]; then
        instr_existing+=("$src")
      fi
    done
    if [ "${#instr_existing[@]}" -eq 0 ]; then
      log "iter=$iter SFT skipped: no instruction sources found"
      continue
    fi
    cat "${instr_existing[@]}" | sed '/^[[:space:]]*$/d' | awk 'length($0) <= 4096' | sort -u > "$sft_data"
    "$PY" version/v7/scripts/prepare_ascii_dataset_v7.py \
      --input "$sft_data" \
      --output "$sft_data" \
      --input-format text \
      --ascii-map-common \
      --ascii-mode xml_escape

    log "iter=$iter stage_b passed gate; running SFT pass on $sft_data (sources=${#instr_existing[@]})"
    RESUME_FLAG_SFT="$(resume_args || true)"

    set +e
    "$PY" version/v7/scripts/train_data_pipeline_v7.py \
      --run "$RUN" \
      ${RESUME_FLAG_SFT:+$RESUME_FLAG_SFT} \
      --template qwen3 \
      --curriculum-stage stage_b \
      --tokenizer ascii_bpe \
      --strict-data-gates \
      --no-post-train-eval \
      --roundtrip-max-lines 2048 \
      --roundtrip-sample-limit 16 \
      --data "$sft_data" \
      --vocab-size "$V" \
      --bpe-vocab-size "$V" \
      --layers 24 \
      --embed-dim 64 \
      --hidden-dim 128 \
      --epochs 1 \
      --seq-len "$SEQ_LEN" \
      --total-tokens 800000 \
      --grad-accum "$GRAD_ACCUM" \
      --lr "$LR_SFT" \
      --max-grad-norm 1.0 \
      --seed "$iter_seed" \
      --train-driver ck_cli \
      --ck-cli-log-every 200 \
      --json-out "$report_sft"
    rc_sft=$?
    set -e

    sft_loss=0
    if [ -f "$report_sft" ]; then
      sft_loss="$(jq -r '.ck_loss.final // 0' "$report_sft")"
    fi
    jq -nc \
      --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --arg run "$RUN" \
      --arg phase "sft" \
      --argjson iter "$iter" \
      --argjson rc "$rc_sft" \
      --arg data "$sft_data" \
      --arg report "$report_sft" \
      --argjson final_loss "$sft_loss" \
      '{ts:$ts,run:$run,phase:$phase,iter:$iter,rc:$rc,data:$data,report:$report,final_loss:$final_loss}' \
      >>"$SUMMARY_JSONL"
    log "iter=$iter sft rc=$rc_sft final_loss=$sft_loss"
  fi
done

write_status "completed" "$iter" "time window reached"
log "autopilot done iterations=$iter summary=$SUMMARY_JSONL"
