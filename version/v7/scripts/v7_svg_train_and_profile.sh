#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

RUN="/tmp/v7_svg_assets_l24"
DATA_TXT="/tmp/v7_svg_assets_train.txt"
USER_DATA_FILE=""
PIPELINE_REPORT="$ROOT/version/v7/reports/v7_svg_assets_bpe_pipeline_l24_e10_latest.json"

VOCAB_SIZE=320
BPE_VOCAB_SIZE=320
LAYERS=24
EMBED_DIM=64
HIDDEN_DIM=128
EPOCHS=10
SEQ_LEN=32
TOTAL_TOKENS=65536
GRAD_ACCUM=1
LR="5e-4"
MAX_GRAD_NORM="1.0"
SEED=42
THREADS=8

WITH_TORCH_REF=1
RUN_VTUNE=1
RUN_ADVISOR=1
RUN_VIS=1
WITH_INFERENCE=0
INFER_MODEL="Qwen/Qwen3-0.6B"
INFER_PROMPT="Generate a tiny SVG icon."
INFER_MAX_TOKENS=64
CLI_MODEL="qwen"

usage() {
  cat <<'EOF'
Usage:
  version/v7/scripts/v7_svg_train_and_profile.sh [options]

Options:
  --run DIR                  Run directory (default: /tmp/v7_svg_assets_l24)
  --data-txt PATH            Generated/working corpus text file path
  --data-file PATH           Use existing UTF-8 corpus file; skip SVG folder concat
  --report PATH              Pipeline report JSON path
  --epochs N                 Training epochs (default: 10)
  --seq-len N                Sequence length (default: 32)
  --total-tokens N           Total tokens (default: 65536)
  --threads N                CK_NUM_THREADS for profiling runs (default: 8)
  --no-torch-ref             Skip torch reference pass in pipeline
  --skip-vtune               Skip VTune profile pass
  --skip-advisor             Skip Advisor profile pass
  --no-visualizer            Skip IR visualizer HTML generation
  --with-inference           Run inference smoke at end (ck_run_v7 + ck-cli-v7)
  --infer-model ID           Model ID for ck_run_v7.py run (default: Qwen/Qwen3-0.6B)
  --infer-prompt TEXT        Prompt for inference smoke
  --infer-max-tokens N       Max tokens for inference smoke
  --cli-model NAME           --model value passed to ck-cli-v7 (default: qwen)
  -h, --help                 Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    --data-txt) DATA_TXT="$2"; shift 2 ;;
    --data-file) USER_DATA_FILE="$2"; shift 2 ;;
    --report) PIPELINE_REPORT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --total-tokens) TOTAL_TOKENS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --no-torch-ref) WITH_TORCH_REF=0; shift ;;
    --skip-vtune) RUN_VTUNE=0; shift ;;
    --skip-advisor) RUN_ADVISOR=0; shift ;;
    --no-visualizer) RUN_VIS=0; shift ;;
    --with-inference) WITH_INFERENCE=1; shift ;;
    --infer-model) INFER_MODEL="$2"; shift 2 ;;
    --infer-prompt) INFER_PROMPT="$2"; shift 2 ;;
    --infer-max-tokens) INFER_MAX_TOKENS="$2"; shift 2 ;;
    --cli-model) CLI_MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required (used to read token file from pipeline report)." >&2
  exit 2
fi

RUN="$(realpath -m "$RUN")"
DATA_TXT="$(realpath -m "$DATA_TXT")"
PIPELINE_REPORT="$(realpath -m "$PIPELINE_REPORT")"

mkdir -p "$(dirname "$DATA_TXT")"
mkdir -p "$(dirname "$PIPELINE_REPORT")"

if [[ -n "$USER_DATA_FILE" ]]; then
  USER_DATA_FILE="$(realpath -m "$USER_DATA_FILE")"
  if [[ ! -f "$USER_DATA_FILE" ]]; then
    echo "ERROR: --data-file not found: $USER_DATA_FILE" >&2
    exit 2
  fi
  TRAIN_DATA="$USER_DATA_FILE"
  echo "[data] using provided corpus file: $TRAIN_DATA"
else
  {
    if [[ -d "$ROOT/docs/site/assets" ]]; then
      find "$ROOT/docs/site/assets" -maxdepth 1 -type f -name '*.svg' -print0
    fi
    if [[ -d "$ROOT/2docs/assets" ]]; then
      find "$ROOT/2docs/assets" -maxdepth 1 -type f -name '*.svg' -print0
    fi
  } | xargs -0 -r cat > "$DATA_TXT"
  if [[ ! -s "$DATA_TXT" ]]; then
    echo "ERROR: no SVG data found in docs/site/assets or 2docs/assets; use --data-file." >&2
    exit 2
  fi
  TRAIN_DATA="$DATA_TXT"
  echo "[data] built corpus file: $TRAIN_DATA"
fi

PIPELINE_CMD=(
  "$PY" "$ROOT/version/v7/scripts/train_data_pipeline_v7.py"
  --run "$RUN"
  --init-if-missing
  --init xavier_uniform
  --tokenizer bpe
  --data "$TRAIN_DATA"
  --vocab-size "$VOCAB_SIZE" --bpe-vocab-size "$BPE_VOCAB_SIZE"
  --layers "$LAYERS" --embed-dim "$EMBED_DIM" --hidden-dim "$HIDDEN_DIM"
  --epochs "$EPOCHS" --seq-len "$SEQ_LEN" --total-tokens "$TOTAL_TOKENS"
  --grad-accum "$GRAD_ACCUM" --lr "$LR" --max-grad-norm "$MAX_GRAD_NORM" --seed "$SEED"
  --json-out "$PIPELINE_REPORT"
)
if [[ "$WITH_TORCH_REF" -eq 1 ]]; then
  PIPELINE_CMD+=(--with-torch-ref)
fi

echo "[run] ${PIPELINE_CMD[*]}"
"${PIPELINE_CMD[@]}"

TOKEN_FILE="$(jq -r '.artifacts.bpe.token_file // empty' "$PIPELINE_REPORT")"
if [[ -z "$TOKEN_FILE" || ! -f "$TOKEN_FILE" ]]; then
  echo "ERROR: could not resolve token file from report: $PIPELINE_REPORT" >&2
  exit 2
fi
echo "[token] $TOKEN_FILE"

run_profile() {
  local mode="$1"
  local out_json="$2"
  CK_NUM_THREADS="$THREADS" "$PY" "$ROOT/version/v7/scripts/ck_run_v7.py" train \
    --run "$RUN" \
    --backend ck \
    --train-token-file "$TOKEN_FILE" \
    --train-epochs "$EPOCHS" \
    --train-seq-len "$SEQ_LEN" \
    --train-total-tokens "$TOTAL_TOKENS" \
    --train-grad-accum "$GRAD_ACCUM" \
    --train-lr "$LR" \
    --train-max-grad-norm "$MAX_GRAD_NORM" \
    --profile-train "$mode" \
    --train-profile-dir "$RUN/profile_train_latest" \
    --train-json-out "$out_json"
}

if [[ "$RUN_VTUNE" -eq 1 ]]; then
  echo "[profile] vtune"
  run_profile vtune "$RUN/train_e2e_profile_vtune.json"
fi

if [[ "$RUN_ADVISOR" -eq 1 ]]; then
  echo "[profile] advisor"
  run_profile advisor "$RUN/train_e2e_profile_advisor.json"
fi

if [[ "$RUN_VIS" -eq 1 ]]; then
  echo "[viz] generate IR visualizer HTML"
  "$PY" "$ROOT/version/v7/tools/open_ir_visualizer.py" \
    --generate \
    --run "$RUN" \
    --html-only
fi

if [[ "$WITH_INFERENCE" -eq 1 ]]; then
  echo "[infer] compile/cache model via ck_run_v7.py run"
  "$PY" "$ROOT/version/v7/scripts/ck_run_v7.py" run "$INFER_MODEL" \
    --prompt "$INFER_PROMPT" \
    --max-tokens "$INFER_MAX_TOKENS" \
    --chat-template none

  if [[ -x "$ROOT/build/ck-cli-v7" ]]; then
    echo "[infer] ck-cli-v7 smoke"
    "$ROOT/build/ck-cli-v7" --model "$CLI_MODEL" \
      --prompt "$INFER_PROMPT" \
      --max-tokens "$INFER_MAX_TOKENS" \
      --temperature 0.0 \
      --timing || true
  fi
fi

echo
echo "[done] run-dir: $RUN"
echo "[done] pipeline report: $PIPELINE_REPORT"
echo "[done] profile dir: $RUN/profile_train_latest"
echo "[done] vtune summary: $RUN/vtune_summary.json"
echo "[done] advisor summary: $RUN/advisor_summary.json"
echo "[done] visualizer html: $RUN/ir_report.html"
