#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
RUN="${1:-$MODEL_CACHE_ROOT/train/spec15a_scene_dsl_l3_d192_h384_ctx768_r1}"
DATASET_PREFIX="${DATASET_PREFIX:-spec15a_scene_dsl}"
WORKSPACE="$RUN/dataset"

ensure_oneapi_runtime_libs() {
  local candidates=(
    "/opt/intel/oneapi/compiler/latest/lib"
    "/opt/intel/oneapi/compiler/2025.3/lib"
    "/opt/intel/oneapi/2025.3/lib"
    "$HOME/intel/oneapi/compiler/latest/lib"
    "$HOME/intel/oneapi/compiler/2025.3/lib"
    "$HOME/intel/oneapi/2025.3/lib"
  )
  local found=""
  local path_entry=""
  for path_entry in "${candidates[@]}"; do
    if [ -f "$path_entry/libimf.so" ]; then
      found="$path_entry"
      break
    fi
  done
  if [ -n "$found" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$found:"*) ;;
      *) export LD_LIBRARY_PATH="$found${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    printf '[spec15a-env] oneapi_runtime_lib=%s\n' "$found"
  fi
}

clear_stale_postrun_artifacts() {
  local run_dir="$1"
  rm -f \
    "$run_dir/spec15a_probe_contract.json" \
    "$run_dir/spec15a_probe_report.json" \
    "$run_dir/spec15a_probe_report.html" \
    "$run_dir/spec15a_tested_prompts_report.html" \
    "$run_dir/spec15a_tested_prompts_report.md" \
    "$run_dir/ir_report.html"
}

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_timed() {
  local label="$1"
  shift
  local started_epoch
  local ended_epoch
  local rc
  started_epoch="$(date +%s)"
  printf '[spec15a-timing] stage=%s event=start started_at=%s\n' "$label" "$(timestamp_utc)"
  "$@"
  rc=$?
  ended_epoch="$(date +%s)"
  printf '[spec15a-timing] stage=%s event=end ended_at=%s elapsed_sec=%s\n' \
    "$label" "$(timestamp_utc)" "$((ended_epoch - started_epoch))"
  return "$rc"
}

build_probe_artifacts() {
  local run_dir="$1"
  local probe_contract="$2"
  local dataset_prefix="$3"
  local context_len="$4"
  local probe_per_split="$5"
  local hidden_probe_per_split="$6"

  python3 version/v7/scripts/build_spec15a_probe_contract_v7.py \
    --run "$run_dir" \
    --prefix "$dataset_prefix" \
    --output "$probe_contract" \
    --per-split "$probe_per_split" \
    --hidden-per-split "$hidden_probe_per_split"

  .venv/bin/python version/v7/scripts/ck_run_v7.py run "$run_dir" \
    --generate-only \
    --context-len "$context_len"

  python3 version/v7/scripts/build_probe_report_v7.py \
    --run "$run_dir" \
    --contract "$probe_contract" \
    --output "$run_dir/spec15a_probe_report.html" \
    --json-out "$run_dir/spec15a_probe_report.json"

  python3 version/v7/scripts/build_structured_scene_tested_prompts_doc_v7.py \
    --probe-report "$run_dir/spec15a_probe_report.json" \
    --output-html "$run_dir/spec15a_tested_prompts_report.html" \
    --output-md "$run_dir/spec15a_tested_prompts_report.md"

  python3 version/v7/tools/open_ir_visualizer.py \
    --generate \
    --run "$run_dir" \
    --html-only \
    --strict-run-artifacts \
    --output "$run_dir/ir_report.html"
}

LAYERS="${LAYERS:-3}"
EMBED_DIM="${EMBED_DIM:-192}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
CONTEXT_LEN="${CONTEXT_LEN:-768}"

PRETRAIN_TOTAL_TOKENS="${PRETRAIN_TOTAL_TOKENS:-}"
MIDTRAIN_TOTAL_TOKENS="${MIDTRAIN_TOTAL_TOKENS:-}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2}"
MIDTRAIN_EPOCHS="${MIDTRAIN_EPOCHS:-2}"
PRETRAIN_CANONICAL_REPEAT="${PRETRAIN_CANONICAL_REPEAT:-20}"
PRETRAIN_BRIDGE_REPEAT="${PRETRAIN_BRIDGE_REPEAT:-6}"
MIDTRAIN_CANONICAL_REPEAT="${MIDTRAIN_CANONICAL_REPEAT:-16}"
MIDTRAIN_BRIDGE_REPEAT="${MIDTRAIN_BRIDGE_REPEAT:-6}"
RUN_SCOPE_SPEC="${RUN_SCOPE_SPEC:-}"
RUN_SCOPE_RUNG="${RUN_SCOPE_RUNG:-}"
RUN_SCOPE_FAMILY="${RUN_SCOPE_FAMILY:-memory_map}"
RUN_SCOPE_TITLE="${RUN_SCOPE_TITLE:-}"
RUN_SCOPE_OBJECTIVE="${RUN_SCOPE_OBJECTIVE:-}"
RUN_SCOPE_HYPOTHESIS="${RUN_SCOPE_HYPOTHESIS:-}"
RUN_SCOPE_PROMPT_CONTRACT="${RUN_SCOPE_PROMPT_CONTRACT:-}"
RUN_SCOPE_OUTPUT_CONTRACT="${RUN_SCOPE_OUTPUT_CONTRACT:-}"
RUN_SCOPE_NOTES="${RUN_SCOPE_NOTES:-}"
RUN_SCOPE_NOTES_FILE="${RUN_SCOPE_NOTES_FILE:-}"
RUN_SCOPE_READ_FIRST_LIST="${RUN_SCOPE_READ_FIRST_LIST:-}"
RUN_SCOPE_CONTEXT_FILE_LIST="${RUN_SCOPE_CONTEXT_FILE_LIST:-}"

PROBE_PER_SPLIT="${PROBE_PER_SPLIT:-12}"
HIDDEN_PROBE_PER_SPLIT="${HIDDEN_PROBE_PER_SPLIT:-6}"
RUN_PARITY="${RUN_PARITY:-1}"
RUN_STAGE_EVAL="${RUN_STAGE_EVAL:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_STRICT="${PREFLIGHT_STRICT:-1}"
CANARY_PER_SPLIT="${CANARY_PER_SPLIT:-4}"
TRAIN_DRIVER="${TRAIN_DRIVER:-ck_cli}"
CK_CLI_LOG_EVERY="${CK_CLI_LOG_EVERY:-100}"
ANALYSIS_CHECKPOINTS="${ANALYSIS_CHECKPOINTS:-log}"
TRAIN_SAVE_EVERY="${TRAIN_SAVE_EVERY:-0}"
TRAIN_SAVE_FINAL="${TRAIN_SAVE_FINAL:-1}"

TRAIN_SAVE_FINAL_FLAG=()
if [ "${TRAIN_SAVE_FINAL}" != "1" ]; then
  TRAIN_SAVE_FINAL_FLAG+=(--no-train-save-final)
fi

TOKENIZER_CORPUS="$WORKSPACE/tokenizer/${DATASET_PREFIX}_tokenizer_corpus.txt"
PRETRAIN_DATA="$WORKSPACE/pretrain/train/${DATASET_PREFIX}_pretrain_train.txt"
MIDTRAIN_DATA="$WORKSPACE/midtrain/train/${DATASET_PREFIX}_midtrain_train.txt"
TOKENIZER_JSON="$WORKSPACE/tokenizer/tokenizer.json"
TOKENIZER_BIN="$WORKSPACE/tokenizer/tokenizer_bin"
PROBE_CONTRACT="$RUN/spec15a_probe_contract.json"
PREFLIGHT_JSON="$RUN/spec15a_preflight.json"

cd "$ROOT"
ensure_oneapi_runtime_libs
mkdir -p "$RUN"
clear_stale_postrun_artifacts "$RUN"

PLAN_ACTIVE_STAGE="pretrain"
PRETRAIN_STAGE_STATUS="active"
PRETRAIN_STAGE_ENABLED="true"
MIDTRAIN_STAGE_STATUS="planned"

MATERIALIZE_ARGS=(
  python3
  version/v7/scripts/dataset/materialize_spec15a_scene_dsl_v7.py
  --workspace "$WORKSPACE"
  --prefix "$DATASET_PREFIX"
  --pretrain-canonical-repeat "$PRETRAIN_CANONICAL_REPEAT"
  --pretrain-bridge-repeat "$PRETRAIN_BRIDGE_REPEAT"
  --midtrain-canonical-repeat "$MIDTRAIN_CANONICAL_REPEAT"
  --midtrain-bridge-repeat "$MIDTRAIN_BRIDGE_REPEAT"
  --force
)

run_timed materialize_workspace "${MATERIALIZE_ARGS[@]}"

if [ "$RUN_PREFLIGHT" = "1" ]; then
  PREFLIGHT_ARGS=(
    python3
    version/v7/scripts/spec15a_preflight_v7.py
    --run "$RUN"
    --prefix "$DATASET_PREFIX"
    --tokenizer-json "$TOKENIZER_JSON"
    --tokenizer-bin "$TOKENIZER_BIN"
    --seq-len "$CONTEXT_LEN"
    --pretrain-epochs "$PRETRAIN_EPOCHS"
    --midtrain-epochs "$MIDTRAIN_EPOCHS"
    --canary-per-split "$CANARY_PER_SPLIT"
    --json-out "$PREFLIGHT_JSON"
  )
  if [ -n "$PRETRAIN_TOTAL_TOKENS" ]; then
    PREFLIGHT_ARGS+=(--current-pretrain-total-tokens "$PRETRAIN_TOTAL_TOKENS")
  fi
  if [ -n "$MIDTRAIN_TOTAL_TOKENS" ]; then
    PREFLIGHT_ARGS+=(--current-midtrain-total-tokens "$MIDTRAIN_TOTAL_TOKENS")
  fi
  if [ "$PREFLIGHT_STRICT" = "1" ]; then
    PREFLIGHT_ARGS+=(--strict)
  fi
  run_timed preflight "${PREFLIGHT_ARGS[@]}"
  if [ -z "$PRETRAIN_TOTAL_TOKENS" ]; then
    PRETRAIN_TOTAL_TOKENS="$(jq -r '.stages.pretrain.recommended_total_tokens // 0' "$PREFLIGHT_JSON")"
  fi
  if [ -z "$MIDTRAIN_TOTAL_TOKENS" ]; then
    MIDTRAIN_TOTAL_TOKENS="$(jq -r '.stages.midtrain.recommended_total_tokens // 0' "$PREFLIGHT_JSON")"
  fi
fi

if [ -z "$PRETRAIN_TOTAL_TOKENS" ] || [ "$PRETRAIN_TOTAL_TOKENS" = "0" ] || [ "$PRETRAIN_TOTAL_TOKENS" = "null" ]; then
  PRETRAIN_TOTAL_TOKENS="131072"
fi
if [ -z "$MIDTRAIN_TOTAL_TOKENS" ] || [ "$MIDTRAIN_TOTAL_TOKENS" = "0" ] || [ "$MIDTRAIN_TOTAL_TOKENS" = "null" ]; then
  MIDTRAIN_TOTAL_TOKENS="131072"
fi

printf '[spec15a-budget] pretrain_total_tokens=%s midtrain_total_tokens=%s\n' "$PRETRAIN_TOTAL_TOKENS" "$MIDTRAIN_TOTAL_TOKENS"
printf '[spec15a-telemetry] train_driver=%s ck_cli_log_every=%s analysis_checkpoints=%s train_save_every=%s train_save_final=%s\n' \
  "$TRAIN_DRIVER" "$CK_CLI_LOG_EVERY" "$ANALYSIS_CHECKPOINTS" "$TRAIN_SAVE_EVERY" "$TRAIN_SAVE_FINAL"
printf '[spec15a-policy] dataset_prefix=%s family=memory_map\n' "$DATASET_PREFIX"
printf '[spec15a-policy] oracle=pytorch parity_regimen=python ck_runtime_bridge=ck_run production_train=ck_cli\n'
printf '[spec15a-policy] stage_a_mix=canonical:%s bridge:%s stage_b_mix=canonical:%s bridge:%s\n' \
  "$PRETRAIN_CANONICAL_REPEAT" "$PRETRAIN_BRIDGE_REPEAT" "$MIDTRAIN_CANONICAL_REPEAT" "$MIDTRAIN_BRIDGE_REPEAT"

VOCAB_SIZE="$(jq -r '.vocab_size // 0' "$TOKENIZER_BIN/tokenizer_meta.json")"
if [ -z "$VOCAB_SIZE" ] || [ "$VOCAB_SIZE" = "0" ] || [ "$VOCAB_SIZE" = "null" ]; then
  echo "failed to read tokenizer vocab size from $TOKENIZER_BIN/tokenizer_meta.json" >&2
  exit 1
fi

cat > "$RUN/training_plan.json" <<JSON
{
  "schema": "ck.training_plan.v1",
  "created_by": "spec15a_pretrain_midtrain_v7.sh",
  "active_stage": "$PLAN_ACTIVE_STAGE",
  "stage_order": ["pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"],
  "tokenizer": {
    "type": "ascii_bpe",
    "vocab_size": $VOCAB_SIZE,
    "tokenizer_corpora": [
      {
        "name": "$(basename "$TOKENIZER_CORPUS")",
        "path": "$TOKENIZER_CORPUS"
      }
    ]
  },
  "stages": [
    {
      "stage": "pretrain",
      "seq": 1,
      "status": "$PRETRAIN_STAGE_STATUS",
      "enabled": $PRETRAIN_STAGE_ENABLED,
      "datasets": [
        {
          "name": "$(basename "$PRETRAIN_DATA")",
          "path": "$PRETRAIN_DATA",
          "kind": "generated_dataset"
        }
      ],
      "runs": []
    },
    {
      "stage": "midtrain",
      "seq": 2,
      "status": "$MIDTRAIN_STAGE_STATUS",
      "enabled": true,
      "datasets": [
        {
          "name": "$(basename "$MIDTRAIN_DATA")",
          "path": "$MIDTRAIN_DATA",
          "kind": "generated_dataset"
        }
      ],
      "runs": []
    },
    {"stage": "sft", "seq": 3, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "dpo", "seq": 4, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "grpo", "seq": 5, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "ppo", "seq": 6, "status": "planned", "enabled": false, "datasets": [], "runs": []}
  ]
}
JSON

RUN_SCOPE_ARGS=(
  python3
  version/v7/scripts/init_run_scope_v7.py
  --run "$RUN"
  --family "$RUN_SCOPE_FAMILY"
)
if [ -n "$RUN_SCOPE_SPEC" ]; then
  RUN_SCOPE_ARGS+=(--spec "$RUN_SCOPE_SPEC")
fi
if [ -n "$RUN_SCOPE_RUNG" ]; then
  RUN_SCOPE_ARGS+=(--rung "$RUN_SCOPE_RUNG")
fi
if [ -n "$RUN_SCOPE_TITLE" ]; then
  RUN_SCOPE_ARGS+=(--title "$RUN_SCOPE_TITLE")
fi
if [ -n "$RUN_SCOPE_OBJECTIVE" ]; then
  RUN_SCOPE_ARGS+=(--objective "$RUN_SCOPE_OBJECTIVE")
fi
if [ -n "$RUN_SCOPE_HYPOTHESIS" ]; then
  RUN_SCOPE_ARGS+=(--hypothesis "$RUN_SCOPE_HYPOTHESIS")
fi
if [ -n "$RUN_SCOPE_PROMPT_CONTRACT" ]; then
  RUN_SCOPE_ARGS+=(--prompt-contract "$RUN_SCOPE_PROMPT_CONTRACT")
fi
if [ -n "$RUN_SCOPE_OUTPUT_CONTRACT" ]; then
  RUN_SCOPE_ARGS+=(--output-contract "$RUN_SCOPE_OUTPUT_CONTRACT")
fi
if [ -n "$RUN_SCOPE_NOTES" ]; then
  RUN_SCOPE_ARGS+=(--notes "$RUN_SCOPE_NOTES")
fi
if [ -n "$RUN_SCOPE_NOTES_FILE" ]; then
  RUN_SCOPE_ARGS+=(--notes-file "$RUN_SCOPE_NOTES_FILE")
fi
if [ -n "$RUN_SCOPE_READ_FIRST_LIST" ]; then
  IFS='::' read -r -a _run_scope_read_first_items <<< "$RUN_SCOPE_READ_FIRST_LIST"
  for _item in "${_run_scope_read_first_items[@]}"; do
    if [ -n "$_item" ]; then
      RUN_SCOPE_ARGS+=(--read-first "$_item")
    fi
  done
fi
if [ -n "$RUN_SCOPE_CONTEXT_FILE_LIST" ]; then
  IFS='::' read -r -a _run_scope_context_file_items <<< "$RUN_SCOPE_CONTEXT_FILE_LIST"
  for _item in "${_run_scope_context_file_items[@]}"; do
    if [ -n "$_item" ]; then
      RUN_SCOPE_ARGS+=(--context-file "$_item")
    fi
  done
fi

run_timed init_run_scope "${RUN_SCOPE_ARGS[@]}"

run_timed init_model \
  .venv/bin/python version/v7/scripts/init_tiny_train_model_v7.py \
    --output-dir "$RUN" \
    --seed 42 \
    --init xavier_uniform \
    --template qwen3 \
    --vocab-size "$VOCAB_SIZE" \
    --layers "$LAYERS" \
    --embed-dim "$EMBED_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-heads "$NUM_HEADS" \
    --num-kv-heads "$NUM_KV_HEADS" \
    --context-len "$CONTEXT_LEN" \
    --kernel-policy fp32_reference_first

cp "$RUN/train_init_config.json" "$RUN/config.json"
cp "$RUN/weights.bump" "$RUN/weights_init.bump"

run_timed stage_tokenizer \
  python3 version/v7/scripts/stage_custom_tokenizer_to_run_v7.py \
    --run "$RUN" \
    --tokenizer-json "$TOKENIZER_JSON" \
    --tokenizer-bin "$TOKENIZER_BIN"

if [ "$RUN_PARITY" = "1" ]; then
  set +e
  run_timed parity_regimen \
    .venv/bin/python version/v7/scripts/run_training_parity_regimen_v7.py \
      --run-dir "$RUN" \
      --no-stop-on-fail \
      --json-out "$RUN/training_parity_regimen_latest.json" \
      --md-out "$RUN/training_parity_regimen_latest.md"
  PARITY_RC=$?
  set -e
  PARITY_EVAL="$(python3 - "$RUN/training_parity_regimen_latest.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    doc = json.load(f)

stages = {
    str(s.get("id")): str(s.get("status"))
    for s in (doc.get("stages") or [])
    if isinstance(s, dict) and s.get("id")
}
required = ["A1", "A4", "D1", "D2", "E1", "F1"]
missing = [sid for sid in required if stages.get(sid) != "PASS"]
failed = [str(x) for x in ((doc.get("summary") or {}).get("failed_stage_ids") or [])]
tolerated = {"B2", "B4", "B8", "C1", "C2", "C3"}
unexpected = [sid for sid in failed if sid not in tolerated]
ok = (not missing) and (not unexpected)
print(json.dumps({
    "ok": ok,
    "required_missing": missing,
    "failed_stage_ids": failed,
    "unexpected_failed_stage_ids": unexpected,
    "tolerated_failed_stage_ids": [sid for sid in failed if sid in tolerated],
}, separators=(",", ":")))
raise SystemExit(0 if ok else 1)
PY
  )"
  PARITY_GATE_RC=$?
  if [ "$PARITY_GATE_RC" -ne 0 ]; then
    echo "[parity] blocking failure: $PARITY_EVAL" >&2
    exit 1
  fi
  if [ "$PARITY_RC" -ne 0 ]; then
    echo "[parity] continuing after tolerated non-critical failures: $PARITY_EVAL"
  fi
fi

PRETRAIN_CMD=(
  .venv/bin/python version/v7/scripts/train_data_pipeline_v7.py
  --run "$RUN"
  --curriculum-stage stage_a
  --tokenizer ascii_bpe
  --reuse-run-tokenizer
  --pack-mode sample
  --no-pack-total-tokens-from-windows
  --strict-data-gates
  --data "$PRETRAIN_DATA"
  --layers "$LAYERS"
  --embed-dim "$EMBED_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --num-heads "$NUM_HEADS"
  --num-kv-heads "$NUM_KV_HEADS"
  --context-len "$CONTEXT_LEN"
  --optimizer adamw
  --train-driver "$TRAIN_DRIVER"
  --ck-cli-log-every "$CK_CLI_LOG_EVERY"
  --analysis-checkpoints "$ANALYSIS_CHECKPOINTS"
  --train-save-every "$TRAIN_SAVE_EVERY"
  --epochs "$PRETRAIN_EPOCHS"
  --seq-len "$CONTEXT_LEN"
  --total-tokens "$PRETRAIN_TOTAL_TOKENS"
  --lr 3e-4
  --no-post-train-eval
  --json-out "$RUN/train_spec15a_stage_a.json"
)
if [ "${#TRAIN_SAVE_FINAL_FLAG[@]}" -gt 0 ]; then
  PRETRAIN_CMD+=("${TRAIN_SAVE_FINAL_FLAG[@]}")
fi

run_timed pretrain "${PRETRAIN_CMD[@]}"

run_timed promote_pretrain_checkpoint \
  python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run "$RUN" --stage pretrain

MIDTRAIN_CMD=(
  .venv/bin/python version/v7/scripts/train_data_pipeline_v7.py
  --run "$RUN"
  --curriculum-stage stage_b
  --tokenizer ascii_bpe
  --reuse-run-tokenizer
  --pack-mode sample
  --no-pack-total-tokens-from-windows
  --strict-data-gates
  --data "$MIDTRAIN_DATA"
  --layers "$LAYERS"
  --embed-dim "$EMBED_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --num-heads "$NUM_HEADS"
  --num-kv-heads "$NUM_KV_HEADS"
  --context-len "$CONTEXT_LEN"
  --optimizer adamw
  --train-driver "$TRAIN_DRIVER"
  --ck-cli-log-every "$CK_CLI_LOG_EVERY"
  --analysis-checkpoints "$ANALYSIS_CHECKPOINTS"
  --train-save-every "$TRAIN_SAVE_EVERY"
  --epochs "$MIDTRAIN_EPOCHS"
  --seq-len "$CONTEXT_LEN"
  --total-tokens "$MIDTRAIN_TOTAL_TOKENS"
  --lr 2.5e-4
  --no-post-train-eval
  --json-out "$RUN/train_spec15a_stage_b.json"
)
if [ "${#TRAIN_SAVE_FINAL_FLAG[@]}" -gt 0 ]; then
  MIDTRAIN_CMD+=("${TRAIN_SAVE_FINAL_FLAG[@]}")
fi

run_timed midtrain "${MIDTRAIN_CMD[@]}"

if [ "$RUN_STAGE_EVAL" = "1" ]; then
  .venv/bin/python version/v7/scripts/eval_stage_v7.py \
    --run "$RUN" \
    --all-stages \
    --n-samples 5
else
  echo "[stage-eval] skipped builtin default probes; use spec15a_probe_report.json/html for memory-map evaluation."
fi

build_probe_artifacts "$RUN" "$PROBE_CONTRACT" "$DATASET_PREFIX" "$CONTEXT_LEN" "$PROBE_PER_SPLIT" "$HIDDEN_PROBE_PER_SPLIT"
printf 'spec15a training finished for %s\n' "$RUN"
