#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT_DIR/version/v6.6/scripts/ck_run_v6_6.py"

if [ ! -f "$RUNNER" ]; then
  echo "ERROR: v6.6 runner not found at $RUNNER"
  exit 1
fi

CONTEXT_LEN="${CONTEXT_LEN:-1024}"
MAX_TOKENS="${MAX_TOKENS:-64}"
FORCE_COMPILE="${FORCE_COMPILE:-0}"

MODELS=(
  "hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
  "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
  "hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"
)

PROMPTS=(
  "Hello! Reply with a single short sentence."
  "Give a tiny C function that returns 42."
  "Give a tiny Python function that returns 42."
  "Give a tiny SQL query that selects the number 42."
)

COMMON_ARGS=(
  "--context-len" "$CONTEXT_LEN"
  "--max-tokens" "$MAX_TOKENS"
)

if [ "$FORCE_COMPILE" = "1" ]; then
  COMMON_ARGS+=("--force-compile")
fi

echo "Manual E2E v6.6"
echo "Runner: $RUNNER"
echo "Context: $CONTEXT_LEN | Max tokens: $MAX_TOKENS | Force compile: $FORCE_COMPILE"
echo ""

for M in "${MODELS[@]}"; do
  echo "============================================================"
  echo "Model: $M"
  echo "============================================================"
  for P in "${PROMPTS[@]}"; do
    echo ""
    echo "Prompt: $P"
    python3 "$RUNNER" run "$M" \
      --prompt "$P" \
      "${COMMON_ARGS[@]}"
    echo ""
  done
done
