#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

NAME="svg_l16_d128_h512_v1024_ctx512"
TEMPLATE="qwen3"
INIT="xavier_uniform"
TOKENIZER="ascii_bpe"
OPTIMIZER="adamw"
LAYERS=16
EMBED_DIM=128
HIDDEN_DIM=512
CONTEXT_LEN=512
VOCAB_SIZE=1024
NUM_HEADS=8
NUM_KV_HEADS=4
SEQ_LEN=512
TOTAL_TOKENS=1048576
STAGE_A_SAMPLES=27000
STAGE_B_SAMPLES=63000
SFT_SAMPLES=55000
LR_PRETRAIN="3e-4"
LR_MIDTRAIN="3e-4"
LR_SFT="1e-4"
ROOT="$ROOT_DEFAULT"
MODE="both" # env|commands|both

usage() {
  cat <<'USAGE'
Usage:
  bash version/v7/scripts/v7_train_init_preset.sh [options]

Options:
  --name NAME
  --template qwen3|qwen2|gemma3|llama
  --init normal_0p02|xavier_uniform|xavier_normal|kaiming_uniform|zeros
  --tokenizer byte|bpe|ascii_bpe
  --optimizer adamw|sgd
  --layers N
  --embed-dim N
  --hidden-dim N
  --context-len N
  --vocab-size N
  --num-heads N
  --num-kv-heads N
  --seq-len N
  --total-tokens N
  --stage-a-samples N
  --stage-b-samples N
  --sft-samples N
  --lr-pretrain FLOAT
  --lr-midtrain FLOAT
  --lr-sft FLOAT
  --root PATH
  --mode env|commands|both
  -h, --help

Examples:
  # 1) Print exports + command block
  bash version/v7/scripts/v7_train_init_preset.sh \
    --name svg_l16_d128_h512_v1024_ctx512_clean02 \
    --template qwen3 --init xavier_uniform \
    --tokenizer ascii_bpe --optimizer adamw \
    --layers 16 --embed-dim 128 --context-len 512 --vocab-size 1024

  # 2) Apply env vars in current shell only
  eval "$(bash version/v7/scripts/v7_train_init_preset.sh --mode env --name svg_demo_ctx512)"
USAGE
}

is_pos_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="${2:?}"; shift 2 ;;
    --template) TEMPLATE="${2:?}"; shift 2 ;;
    --init) INIT="${2:?}"; shift 2 ;;
    --tokenizer) TOKENIZER="${2:?}"; shift 2 ;;
    --optimizer) OPTIMIZER="${2:?}"; shift 2 ;;
    --layers) LAYERS="${2:?}"; shift 2 ;;
    --embed-dim) EMBED_DIM="${2:?}"; shift 2 ;;
    --hidden-dim) HIDDEN_DIM="${2:?}"; shift 2 ;;
    --context-len) CONTEXT_LEN="${2:?}"; shift 2 ;;
    --vocab-size) VOCAB_SIZE="${2:?}"; shift 2 ;;
    --num-heads) NUM_HEADS="${2:?}"; shift 2 ;;
    --num-kv-heads) NUM_KV_HEADS="${2:?}"; shift 2 ;;
    --seq-len) SEQ_LEN="${2:?}"; shift 2 ;;
    --total-tokens) TOTAL_TOKENS="${2:?}"; shift 2 ;;
    --stage-a-samples) STAGE_A_SAMPLES="${2:?}"; shift 2 ;;
    --stage-b-samples) STAGE_B_SAMPLES="${2:?}"; shift 2 ;;
    --sft-samples) SFT_SAMPLES="${2:?}"; shift 2 ;;
    --lr-pretrain) LR_PRETRAIN="${2:?}"; shift 2 ;;
    --lr-midtrain) LR_MIDTRAIN="${2:?}"; shift 2 ;;
    --lr-sft) LR_SFT="${2:?}"; shift 2 ;;
    --root) ROOT="${2:?}"; shift 2 ;;
    --mode) MODE="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

for v in LAYERS EMBED_DIM HIDDEN_DIM CONTEXT_LEN VOCAB_SIZE NUM_HEADS NUM_KV_HEADS SEQ_LEN TOTAL_TOKENS STAGE_A_SAMPLES STAGE_B_SAMPLES SFT_SAMPLES; do
  if ! is_pos_int "${!v}"; then
    echo "[ERROR] $v must be a positive integer, got '${!v}'" >&2
    exit 2
  fi
done

if [ "$NUM_KV_HEADS" -gt "$NUM_HEADS" ]; then
  echo "[ERROR] NUM_KV_HEADS ($NUM_KV_HEADS) cannot exceed NUM_HEADS ($NUM_HEADS)" >&2
  exit 2
fi

if [[ "$TOKENIZER" != "byte" && "$TOKENIZER" != "bpe" && "$TOKENIZER" != "ascii_bpe" ]]; then
  echo "[ERROR] --tokenizer must be one of: byte, bpe, ascii_bpe" >&2
  exit 2
fi

if [[ "$OPTIMIZER" != "adamw" && "$OPTIMIZER" != "sgd" ]]; then
  echo "[ERROR] --optimizer must be one of: adamw, sgd" >&2
  exit 2
fi

if [[ "$MODE" != "env" && "$MODE" != "commands" && "$MODE" != "both" ]]; then
  echo "[ERROR] --mode must be one of: env, commands, both" >&2
  exit 2
fi

emit_env() {
  cat <<EOF
export ROOT="$ROOT"
export CK_NAME="$NAME"
export CK_TEMPLATE="$TEMPLATE"
export CK_INIT="$INIT"
export CK_TOKENIZER="$TOKENIZER"
export CK_OPTIMIZER="$OPTIMIZER"
export CK_LAYERS=$LAYERS
export CK_EMBED_DIM=$EMBED_DIM
export CK_HIDDEN_DIM=$HIDDEN_DIM
export CK_CONTEXT_LEN=$CONTEXT_LEN
export CK_VOCAB_SIZE=$VOCAB_SIZE
export CK_NUM_HEADS=$NUM_HEADS
export CK_NUM_KV_HEADS=$NUM_KV_HEADS
export CK_SEQ_LEN=$SEQ_LEN
export CK_TOTAL_TOKENS=$TOTAL_TOKENS
export CK_STAGE_A_SAMPLES=$STAGE_A_SAMPLES
export CK_STAGE_B_SAMPLES=$STAGE_B_SAMPLES
export CK_SFT_SAMPLES=$SFT_SAMPLES
export CK_LR_PRETRAIN="$LR_PRETRAIN"
export CK_LR_MIDTRAIN="$LR_MIDTRAIN"
export CK_LR_SFT="$LR_SFT"

export RUN="\$HOME/.cache/ck-engine-v7/models/train/\$CK_NAME"
export DATA_DIR="\$ROOT/version/v7/data"
export GEN_DIR="\$DATA_DIR/generated"
export CK_PREFIX="\$CK_NAME"
export TOKENIZER_CORPUS="\$GEN_DIR/\${CK_PREFIX}_stage_a_plus_bridge.txt"
export PRETRAIN_DATA="\$GEN_DIR/\${CK_PREFIX}_stage_a_plus_bridge.txt"
export MIDTRAIN_DATA="\$GEN_DIR/\${CK_PREFIX}_stage_b.txt"
export SFT_DATA="\$GEN_DIR/\${CK_PREFIX}_instruction_train.txt"
EOF
}

emit_commands() {
  cat <<'EOF'
# Build staged corpora and instruction data
mkdir -p "$GEN_DIR" "$RUN"

# tokenizer arguments for bpe/ascii_bpe only
BPE_ARGS=""
REUSE_TOK_ARG=""
if [ "$CK_TOKENIZER" = "ascii_bpe" ] || [ "$CK_TOKENIZER" = "bpe" ]; then
  BPE_ARGS="--vocab-size $CK_VOCAB_SIZE --bpe-vocab-size $CK_VOCAB_SIZE"
  REUSE_TOK_ARG="--reuse-run-tokenizer"
fi

python3 version/v7/scripts/build_svg_pretrain_corpus_v7.py \
  --out-dir "$GEN_DIR" \
  --prefix "$CK_PREFIX" \
  --assets-glob "$ROOT/docs/site/assets/*.svg" \
  --stage-a-samples "$CK_STAGE_A_SAMPLES" \
  --stage-b-samples "$CK_STAGE_B_SAMPLES"

python3 version/v7/scripts/generate_svg_instruction_dataset_v7.py \
  --out-dir "$GEN_DIR" \
  --prefix "$CK_PREFIX" \
  --num-samples "$CK_SFT_SAMPLES" \
  --jsonl

# Bootstrap run + tokenizer (no training yet)
.venv/bin/python version/v7/scripts/train_data_pipeline_v7.py \
  --run "$RUN" \
  --init-if-missing \
  --init "$CK_INIT" \
  --template "$CK_TEMPLATE" \
  --curriculum-stage stage_a \
  --tokenizer "$CK_TOKENIZER" \
  --require-svg-rows \
  --strict-data-gates \
  --data "$TOKENIZER_CORPUS" \
  $BPE_ARGS \
  --layers "$CK_LAYERS" --embed-dim "$CK_EMBED_DIM" --hidden-dim "$CK_HIDDEN_DIM" \
  --num-heads "$CK_NUM_HEADS" --num-kv-heads "$CK_NUM_KV_HEADS" \
  --context-len "$CK_CONTEXT_LEN" \
  --optimizer "$CK_OPTIMIZER" \
  --seq-len "$CK_SEQ_LEN" --total-tokens "$CK_TOTAL_TOKENS" \
  --prepare-only \
  --json-out "$RUN/train_prepare_stage_a.json"

# Stage A pretrain
.venv/bin/python version/v7/scripts/train_data_pipeline_v7.py \
  --run "$RUN" \
  --curriculum-stage stage_a \
  --tokenizer "$CK_TOKENIZER" \
  --require-svg-rows \
  --strict-data-gates \
  --data "$PRETRAIN_DATA" \
  $BPE_ARGS \
  --layers "$CK_LAYERS" --embed-dim "$CK_EMBED_DIM" --hidden-dim "$CK_HIDDEN_DIM" \
  --num-heads "$CK_NUM_HEADS" --num-kv-heads "$CK_NUM_KV_HEADS" \
  --context-len "$CK_CONTEXT_LEN" \
  --optimizer "$CK_OPTIMIZER" \
  --epochs 1 --seq-len "$CK_SEQ_LEN" --total-tokens "$CK_TOTAL_TOKENS" \
  --lr "$CK_LR_PRETRAIN"

# REQUIRED gate before long/full training:
#   1) Run Step 3.1 canary parity gate from v7-runbook.html#step-3-5-parity-gate
#   2) Run Step 3.2 regimen below and confirm PASS in JSON/Markdown
python3 version/v7/scripts/run_training_parity_regimen_v7.py \
  --run-dir "$RUN" \
  --json-out "$RUN/training_parity_regimen_latest.json" \
  --md-out "$RUN/training_parity_regimen_latest.md"

# Stage B midtrain (reuse tokenizer)
.venv/bin/python version/v7/scripts/train_data_pipeline_v7.py \
  --run "$RUN" \
  --curriculum-stage stage_b \
  --tokenizer "$CK_TOKENIZER" \
  --require-svg-rows \
  --strict-data-gates \
  --data "$MIDTRAIN_DATA" \
  --layers "$CK_LAYERS" --embed-dim "$CK_EMBED_DIM" --hidden-dim "$CK_HIDDEN_DIM" \
  --num-heads "$CK_NUM_HEADS" --num-kv-heads "$CK_NUM_KV_HEADS" \
  --context-len "$CK_CONTEXT_LEN" \
  --optimizer "$CK_OPTIMIZER" \
  --epochs 1 --seq-len "$CK_SEQ_LEN" --total-tokens "$CK_TOTAL_TOKENS" \
  --lr "$CK_LR_MIDTRAIN" $REUSE_TOK_ARG

# SFT (reuse tokenizer)
.venv/bin/python version/v7/scripts/train_data_pipeline_v7.py \
  --run "$RUN" \
  --curriculum-stage sft \
  --tokenizer "$CK_TOKENIZER" \
  --strict-data-gates \
  --data "$SFT_DATA" \
  --layers "$CK_LAYERS" --embed-dim "$CK_EMBED_DIM" --hidden-dim "$CK_HIDDEN_DIM" \
  --num-heads "$CK_NUM_HEADS" --num-kv-heads "$CK_NUM_KV_HEADS" \
  --context-len "$CK_CONTEXT_LEN" \
  --optimizer "$CK_OPTIMIZER" \
  --epochs 1 --seq-len "$CK_SEQ_LEN" --total-tokens "$CK_TOTAL_TOKENS" \
  --lr "$CK_LR_SFT" $REUSE_TOK_ARG

# Generate operator views
python3 version/v7/tools/open_ir_visualizer.py --generate --run "$RUN" --html-only --strict-run-artifacts
python3 version/v7/tools/open_ir_hub.py --open
EOF
}

case "$MODE" in
  env) emit_env ;;
  commands) emit_commands ;;
  both)
    emit_env
    printf '\n'
    emit_commands
    ;;
esac
