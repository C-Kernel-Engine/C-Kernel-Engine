#!/usr/bin/env bash
set -euo pipefail

MODEL="hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
PROMPT="Hello"
CTX_LEN=256
MAX_TOKENS=1
PHASE="both" # embed|layer0|both
LLAMA_STOP_AFTER=25
LLAMA_TIMEOUT=0
WORK_DIR=""
FORCE_COMPILE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --model <hf://...gguf|/path/model.gguf>   Model input (default: Gemma-3-270M GGUF)
  --prompt <text>                          Prompt string (default: "Hello")
  --context-len <n>                         Context length (default: 256)
  --max-tokens <n>                          Max tokens to generate (default: 1)
  --phase <embed|layer0|both>               Which parity phase to run (default: both)
  --llama-stop-after <n>                    Max number of llama dumps (default: 25)
  --llama-timeout <sec>                     llama.cpp timeout (0 disables, default: 0)
  --work-dir <path>                         Override ck_build dir (auto if not set)
  --force-compile                           Force rebuild of C code
  -h|--help                                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --prompt) PROMPT="$2"; shift 2;;
    --context-len) CTX_LEN="$2"; shift 2;;
    --max-tokens) MAX_TOKENS="$2"; shift 2;;
    --phase) PHASE="$2"; shift 2;;
    --llama-stop-after) LLAMA_STOP_AFTER="$2"; shift 2;;
    --llama-timeout) LLAMA_TIMEOUT="$2"; shift 2;;
    --work-dir) WORK_DIR="$2"; shift 2;;
    --force-compile) FORCE_COMPILE=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

if [[ -z "$WORK_DIR" ]]; then
  if [[ "$MODEL" == hf://* && "$MODEL" == *.gguf ]]; then
    rest="${MODEL#hf://}"
    repo_id="$(printf '%s' "$rest" | cut -d/ -f1-2)"
    WORK_DIR="$HOME/.cache/ck-engine-v6.6/models/${repo_id//\//--}/ck_build"
  elif [[ "$MODEL" == *.gguf ]]; then
    WORK_DIR="$(cd "$(dirname "$MODEL")" && pwd)/ck_build"
  else
    echo "Could not infer work dir. Pass --work-dir."
    exit 2
  fi
fi

CK_DUMP="$WORK_DIR/ck_parity_dumps/dump.bin"
LLAMA_DUMP="$WORK_DIR/llama_parity_dumps/dump.bin"

run_phase() {
  local phase_name="$1"
  local layer="$2"
  local filter="$3"
  local include_global="$4"

  echo "==> Phase: $phase_name"
  rm -f "$CK_DUMP" "$LLAMA_DUMP"

  cmd=(
    python version/v6.6/scripts/ck_run_v6_6.py run
    "$MODEL"
    --context-len "$CTX_LEN"
    --max-tokens "$MAX_TOKENS"
    --prompt "$PROMPT"
    --detailed-llamacpp-parity
    --llama-layer "$layer"
    --llama-filter "$filter"
    --llama-stop-after "$LLAMA_STOP_AFTER"
    --llama-timeout "$LLAMA_TIMEOUT"
  )
  if [[ "$include_global" == "1" ]]; then
    cmd+=(--llama-include-global)
  fi
  if [[ "$FORCE_COMPILE" == "1" ]]; then
    cmd+=(--force-compile)
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"

  if [[ ! -s "$CK_DUMP" ]]; then
    echo "CK dump missing or empty: $CK_DUMP"
    exit 1
  fi
  if [[ ! -s "$LLAMA_DUMP" ]]; then
    echo "llama.cpp dump missing or empty: $LLAMA_DUMP"
    exit 1
  fi

  python version/v6.6/scripts/parity_test.py \
    --ck-dump "$CK_DUMP" \
    --ref-dump "$LLAMA_DUMP" \
    --model gemma --pass prefill
}

case "$PHASE" in
  embed)
    run_phase "embedding" "-1" "inp_embd,token_embd" "1"
    ;;
  layer0)
    run_phase "layer0" "0" "attn_norm,Qcur,Kcur,Vcur,attn_out" "0"
    ;;
  both)
    run_phase "embedding" "-1" "inp_embd,token_embd" "1"
    run_phase "layer0" "0" "attn_norm,Qcur,Kcur,Vcur,attn_out" "0"
    ;;
  *)
    echo "Invalid --phase: $PHASE (expected embed|layer0|both)"
    exit 2
    ;;
esac
