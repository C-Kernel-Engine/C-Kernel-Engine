#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
RUN="${1:-$MODEL_CACHE_ROOT/train/spec12_scene_dsl_l3_d192_h384_ctx768}"
DATASET_PREFIX="${DATASET_PREFIX:-spec12_scene_dsl}"
WORKSPACE="$RUN/dataset"

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_timed() {
  local label="$1"
  shift
  local started_epoch
  local ended_epoch
  started_epoch="$(date +%s)"
  printf '[spec12-timing] stage=%s event=start started_at=%s\n' "$label" "$(timestamp_utc)"
  "$@"
  ended_epoch="$(date +%s)"
  printf '[spec12-timing] stage=%s event=end ended_at=%s elapsed_sec=%s\n' \
    "$label" "$(timestamp_utc)" "$((ended_epoch - started_epoch))"
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
MIDTRAIN_EPOCHS="${MIDTRAIN_EPOCHS:-3}"
MIDTRAIN_EDIT_REPEAT="${MIDTRAIN_EDIT_REPEAT:-3}"
MIDTRAIN_DIRECT_REPEAT="${MIDTRAIN_DIRECT_REPEAT:-6}"
MIDTRAIN_CLOSE_REPEAT="${MIDTRAIN_CLOSE_REPEAT:-8}"
MIDTRAIN_HEADER_REPEAT="${MIDTRAIN_HEADER_REPEAT:-6}"
MIDTRAIN_BLOCK_REPEAT="${MIDTRAIN_BLOCK_REPEAT:-4}"
MIDTRAIN_TRANSITION_REPEAT="${MIDTRAIN_TRANSITION_REPEAT:-6}"
MIDTRAIN_TABLE_REPEAT="${MIDTRAIN_TABLE_REPEAT:-3}"
TRAIN_REPEATS="${TRAIN_REPEATS:-6}"
HOLDOUT_REPEATS="${HOLDOUT_REPEATS:-1}"
PROBE_PER_SPLIT="${PROBE_PER_SPLIT:-12}"
RUN_PARITY="${RUN_PARITY:-1}"
RUN_STAGE_EVAL="${RUN_STAGE_EVAL:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_STRICT="${PREFLIGHT_STRICT:-1}"
CANARY_PER_SPLIT="${CANARY_PER_SPLIT:-4}"

TOKENIZER_CORPUS="$WORKSPACE/tokenizer/${DATASET_PREFIX}_tokenizer_corpus.txt"
PRETRAIN_DATA="$WORKSPACE/pretrain/train/${DATASET_PREFIX}_pretrain_train.txt"
MIDTRAIN_DATA="$WORKSPACE/midtrain/train/${DATASET_PREFIX}_midtrain_train.txt"
TOKENIZER_JSON="$WORKSPACE/tokenizer/tokenizer.json"
TOKENIZER_BIN="$WORKSPACE/tokenizer/tokenizer_bin"
PROBE_CONTRACT="$RUN/spec12_probe_contract.json"
PREFLIGHT_JSON="$RUN/spec12_preflight.json"

cd "$ROOT"
mkdir -p "$RUN"

run_timed materialize_workspace \
  python3 version/v7/scripts/dataset/materialize_spec12_scene_dsl_v7.py \
    --workspace "$WORKSPACE" \
    --prefix "$DATASET_PREFIX" \
    --train-repeats "$TRAIN_REPEATS" \
    --holdout-repeats "$HOLDOUT_REPEATS" \
    --midtrain-edit-repeat "$MIDTRAIN_EDIT_REPEAT" \
    --midtrain-direct-repeat "$MIDTRAIN_DIRECT_REPEAT" \
    --midtrain-close-repeat "$MIDTRAIN_CLOSE_REPEAT" \
    --midtrain-header-repeat "$MIDTRAIN_HEADER_REPEAT" \
    --midtrain-block-repeat "$MIDTRAIN_BLOCK_REPEAT" \
    --midtrain-transition-repeat "$MIDTRAIN_TRANSITION_REPEAT" \
    --midtrain-table-repeat "$MIDTRAIN_TABLE_REPEAT" \
    --force

if [ "$RUN_PREFLIGHT" = "1" ]; then
  PREFLIGHT_ARGS=(
    python3
    version/v7/scripts/spec12_preflight_v7.py
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
  PRETRAIN_TOTAL_TOKENS="1048576"
fi
if [ -z "$MIDTRAIN_TOTAL_TOKENS" ] || [ "$MIDTRAIN_TOTAL_TOKENS" = "0" ] || [ "$MIDTRAIN_TOTAL_TOKENS" = "null" ]; then
  MIDTRAIN_TOTAL_TOKENS="1572864"
fi

printf '[spec12-budget] pretrain_total_tokens=%s midtrain_total_tokens=%s\n' "$PRETRAIN_TOTAL_TOKENS" "$MIDTRAIN_TOTAL_TOKENS"

VOCAB_SIZE="$(jq -r '.vocab_size // 0' "$TOKENIZER_BIN/tokenizer_meta.json")"
if [ -z "$VOCAB_SIZE" ] || [ "$VOCAB_SIZE" = "0" ] || [ "$VOCAB_SIZE" = "null" ]; then
  echo "failed to read tokenizer vocab size from $TOKENIZER_BIN/tokenizer_meta.json" >&2
  exit 1
fi

cat > "$RUN/training_plan.json" <<JSON
{
  "schema": "ck.training_plan.v1",
  "created_by": "spec12_pretrain_midtrain_v7.sh",
  "active_stage": "pretrain",
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
      "status": "active",
      "enabled": true,
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
      "status": "planned",
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
    {
      "stage": "sft",
      "seq": 3,
      "status": "planned",
      "enabled": false,
      "datasets": [],
      "runs": []
    },
    {"stage": "dpo", "seq": 4, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "grpo", "seq": 5, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "ppo", "seq": 6, "status": "planned", "enabled": false, "datasets": [], "runs": []}
  ]
}
JSON

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
      --json-out "$RUN/training_parity_regimen_latest.json" \
      --md-out "$RUN/training_parity_regimen_latest.md"
  PARITY_RC=$?
  set -e
  if [ "$PARITY_RC" -ne 0 ]; then
    FAILED_IDS="$(jq -cr '.summary.failed_stage_ids // []' "$RUN/training_parity_regimen_latest.json")"
    if [ "$FAILED_IDS" = "[\"B8\"]" ]; then
      echo "[parity] continuing: only B8 failed (known g=8 AdamW drift at lr=1e-3); stage training uses lower learning rates."
    else
      echo "[parity] blocking failure: failed_stage_ids=$FAILED_IDS" >&2
      exit "$PARITY_RC"
    fi
  fi
fi

run_timed pretrain \
  .venv/bin/python version/v7/scripts/train_data_pipeline_v7.py \
    --run "$RUN" \
    --curriculum-stage stage_a \
    --tokenizer ascii_bpe \
    --reuse-run-tokenizer \
    --pack-mode sample \
    --no-pack-total-tokens-from-windows \
    --strict-data-gates \
    --data "$PRETRAIN_DATA" \
    --layers "$LAYERS" \
    --embed-dim "$EMBED_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-heads "$NUM_HEADS" \
    --num-kv-heads "$NUM_KV_HEADS" \
    --context-len "$CONTEXT_LEN" \
    --optimizer adamw \
    --epochs "$PRETRAIN_EPOCHS" \
    --seq-len "$CONTEXT_LEN" \
    --total-tokens "$PRETRAIN_TOTAL_TOKENS" \
    --lr 3e-4 \
    --no-post-train-eval \
    --json-out "$RUN/train_spec12_stage_a.json"

run_timed promote_pretrain_checkpoint \
  python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run "$RUN" --stage pretrain

run_timed midtrain \
  .venv/bin/python version/v7/scripts/train_data_pipeline_v7.py \
    --run "$RUN" \
    --curriculum-stage stage_b \
    --tokenizer ascii_bpe \
    --reuse-run-tokenizer \
    --pack-mode sample \
    --no-pack-total-tokens-from-windows \
    --strict-data-gates \
    --data "$MIDTRAIN_DATA" \
    --layers "$LAYERS" \
    --embed-dim "$EMBED_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-heads "$NUM_HEADS" \
    --num-kv-heads "$NUM_KV_HEADS" \
    --context-len "$CONTEXT_LEN" \
    --optimizer adamw \
    --epochs "$MIDTRAIN_EPOCHS" \
    --seq-len "$CONTEXT_LEN" \
    --total-tokens "$MIDTRAIN_TOTAL_TOKENS" \
    --lr 2.5e-4 \
    --no-post-train-eval \
    --json-out "$RUN/train_spec12_stage_b.json"

if [ "$RUN_STAGE_EVAL" = "1" ]; then
  .venv/bin/python version/v7/scripts/eval_stage_v7.py \
    --run "$RUN" \
    --all-stages \
    --n-samples 5
else
  echo "[stage-eval] skipped builtin default probes; use spec12_probe_report.json/html for scene evaluation."
fi

python3 version/v7/scripts/build_spec12_probe_contract_v7.py \
  --run "$RUN" \
  --prefix "$DATASET_PREFIX" \
  --output "$PROBE_CONTRACT" \
  --per-split "$PROBE_PER_SPLIT"

.venv/bin/python version/v7/scripts/ck_run_v7.py run "$RUN" \
  --generate-only \
  --context-len "$CONTEXT_LEN"

python3 version/v7/scripts/build_probe_report_v7.py \
  --run "$RUN" \
  --contract "$PROBE_CONTRACT" \
  --output "$RUN/spec12_probe_report.html" \
  --json-out "$RUN/spec12_probe_report.json"

python3 version/v7/scripts/build_structured_scene_tested_prompts_doc_v7.py \
  --probe-report "$RUN/spec12_probe_report.json" \
  --output-html "$RUN/spec12_tested_prompts_report.html" \
  --output-md "$RUN/spec12_tested_prompts_report.md"

python3 version/v7/tools/open_ir_visualizer.py \
  --generate \
  --run "$RUN" \
  --html-only \
  --strict-run-artifacts \
  --output "$RUN/ir_report.html"

printf 'spec12 pretrain+midtrain finished for %s\n' "$RUN"
