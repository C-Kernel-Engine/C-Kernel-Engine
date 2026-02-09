#!/usr/bin/env bash
set -euo pipefail

MODEL_URI="${MODEL_URI:-hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf}"
PROMPT="${PROMPT:-Hello}"
MAX_TOKENS="${MAX_TOKENS:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-256}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${HOME}/.cache/ck-engine-v6.6/gemma_patch_runs/gemma_${TS}/ck_build"

echo "[gemma-patch] session output_dir=${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "[gemma-patch] step 1/4: generate artifacts (no run)"
python "${ROOT}/version/v6.6/scripts/ck_run_v6_6.py" run "${MODEL_URI}" \
  --output-dir "${OUT_DIR}" \
  --context-len "${CONTEXT_LEN}" \
  --force-compile \
  --generate-only

echo "[gemma-patch] step 2/4: patch generated C (inp_scaled)"
python "${ROOT}/version/v6.6/scripts/patch_generated_gemma3.py" \
  --model-dir "${OUT_DIR}"

echo "[gemma-patch] step 3/4: compile patched libmodel.so"
bash "${ROOT}/version/v6.6/scripts/compile_model_v6_6.sh" "${OUT_DIR}"

echo "[gemma-patch] step 4/5: run CK parity dump from patched lib (no regenerate)"
mkdir -p "${OUT_DIR}/ck_parity_dumps"
CK_PARITY_DUMP=1 \
CK_PARITY_DIR="${OUT_DIR}/ck_parity_dumps" \
LD_LIBRARY_PATH="${ROOT}/build:${LD_LIBRARY_PATH:-}" \
"${ROOT}/build/ck-cli-v6.6" \
  --lib "${OUT_DIR}/libmodel.so" \
  --weights "${OUT_DIR}/weights.bump" \
  --prompt "${PROMPT}" \
  --max-tokens "${MAX_TOKENS}" \
  --context "${CONTEXT_LEN}" \
  --temperature 0 \
  --top-p 1.0 \
  --no-chat-template

echo "[gemma-patch] step 5/5: run detailed parity analysis (reuses patched ck dump)"
python "${ROOT}/version/v6.6/scripts/detailed_parity_analysis.py" \
  --model-uri "${MODEL_URI}" \
  --family gemma \
  --output-dir "${OUT_DIR}" \
  --prompt "${PROMPT}" \
  --context-len "${CONTEXT_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  --skip-ck-run \
  "$@"

echo "[gemma-patch] done"
echo "[gemma-patch] ck_dump=${OUT_DIR}/ck_parity_dumps/dump.bin"
echo "[gemma-patch] llama_dump=${OUT_DIR}/llama_parity_dumps/dump.bin"
echo "[gemma-patch] report=${OUT_DIR}/detailed_parity_analysis.md"
