#!/usr/bin/env bash
set -euo pipefail

MODEL_URI="${MODEL_URI:-hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf}"
PROMPT="${PROMPT:-Hello}"
CONTEXT_LEN="${CONTEXT_LEN:-256}"
MAX_TOKENS="${MAX_TOKENS:-1}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

python "${ROOT}/version/v6.6/scripts/detailed_parity_analysis.py" \
  --model-uri "${MODEL_URI}" \
  --family gemma \
  --prompt "${PROMPT}" \
  --context-len "${CONTEXT_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  "$@"

