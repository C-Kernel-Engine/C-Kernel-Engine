#!/usr/bin/env bash
set -euo pipefail

# Repeatable parity run for Gemma-3-270M:
# - Dumps a broad layer-0 tensor set from llama.cpp + CKE
# - Runs parity_test.py to find the first divergence with meaningful coverage
#
# Usage:
#   bash version/v7/scripts/parity/run_gemma_first_divergence.sh \
#     [MODEL_URI] [PROMPT] [CONTEXT_LEN] [MAX_TOKENS]
#
# Defaults:
#   MODEL_URI   = local cached Gemma GGUF when available, otherwise hf://...
#   PROMPT      = Hello
#   CONTEXT_LEN = 256
#   MAX_TOKENS  = 1
#
# Optional env knobs for focused layer-0 qkv contract check:
#   RUN_QKV_CONTRACT=1      (default: run checker)
#   QKV_DECODE_TOKEN=9259
#   QKV_PREFILL_TOKENS=2,9259
#   QKV_TOL=1e-3
#   QKV_NO_FAIL_FAST=0      (set 1 to collect all mismatches)

DEFAULT_HF_URI="hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
DEFAULT_LOCAL_GGUF="$HOME/.cache/ck-engine-v7/models/unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
if [[ $# -ge 1 && -n "${1}" ]]; then
  MODEL_URI="$1"
else
  if [[ -f "${DEFAULT_LOCAL_GGUF}" ]]; then
    MODEL_URI="${DEFAULT_LOCAL_GGUF}"
  else
    MODEL_URI="${DEFAULT_HF_URI}"
  fi
fi
PROMPT="${2:-Hello}"
CONTEXT_LEN="${3:-256}"
MAX_TOKENS="${4:-1}"
RUN_QKV_CONTRACT="${RUN_QKV_CONTRACT:-1}"
QKV_DECODE_TOKEN="${QKV_DECODE_TOKEN:-9259}"
QKV_PREFILL_TOKENS="${QKV_PREFILL_TOKENS:-2,9259}"
QKV_TOL="${QKV_TOL:-1e-3}"
QKV_NO_FAIL_FAST="${QKV_NO_FAIL_FAST:-0}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V7_SCRIPTS_DIR="${ROOT}/version/v7/scripts"
PARITY_DIR="${V7_SCRIPTS_DIR}/parity"

# Infer work dir following ck_run_v7.py cache behavior.
WORK_DIR=""
if [[ "$MODEL_URI" == hf://* ]]; then
  repo_path="${MODEL_URI#hf://}"
  repo_path="${repo_path%/*}" # drop filename
  repo_key="${repo_path//\//--}"
  WORK_DIR="$HOME/.cache/ck-engine-v7/models/${repo_key}"
elif [[ "$MODEL_URI" == *.gguf ]]; then
  stem="$(basename "${MODEL_URI}" .gguf)"
  WORK_DIR="$HOME/.cache/ck-engine-v7/models/${stem}"
fi
if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$HOME/.cache/ck-engine-v7/models/gemma-3-270m-it-Q5_K_M"
fi

echo "[1/4] Clean parity dumps"
rm -rf "${WORK_DIR}/ck_parity_dumps" "${WORK_DIR}/llama_parity_dumps"

LLAMA_FILTER_BASE="inp_embd,inp_scaled,token_embd,attn_norm,Qcur,Kcur,Vcur,Qcur_normed,Kcur_normed,attn_out,kqv_out,attn_post_norm,sa_out,ffn_norm,ffn_gate,ffn_up,ffn_down,ffn_post_norm,final_norm,result_norm,ln_final,logits,result_output"
LLAMA_STOP_AFTER_BASE="${LLAMA_STOP_AFTER_BASE:-96}"
LLAMA_TIMEOUT_BASE="${LLAMA_TIMEOUT_BASE:-120}"

echo "[2/4] Run ck_run_v7.py (broad llama+CK dumps)"
python "${V7_SCRIPTS_DIR}/ck_run_v7.py" run \
  "${MODEL_URI}" \
  --context-len "${CONTEXT_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  --force-compile \
  --detailed-llamacpp-parity \
  --prompt "${PROMPT}" \
  --llama-layer 0 \
  --llama-include-global \
  --llama-filter "${LLAMA_FILTER_BASE}" \
  --llama-stop-after "${LLAMA_STOP_AFTER_BASE}" \
  --llama-timeout "${LLAMA_TIMEOUT_BASE}"

INDEX_JSON="${WORK_DIR}/llama_parity_dumps/index.json"
if [[ -f "${INDEX_JSON}" ]]; then
  echo "[info] llama.cpp index.json:"
  cat "${INDEX_JSON}"
  echo
fi

if [[ -f "${INDEX_JSON}" ]]; then
  DUMP_COUNT="$(python - <<'PY' "${INDEX_JSON}"
import json, sys
p = sys.argv[1]
try:
    data = json.load(open(p, "r"))
    print(len(data) if isinstance(data, list) else 0)
except Exception:
    print(0)
PY
)"
  if [[ "${DUMP_COUNT}" -lt 12 ]]; then
    echo "[warn] llama dump count is low (${DUMP_COUNT}); running broader discovery pass"
    rm -rf "${WORK_DIR}/llama_parity_dumps"
    python "${V7_SCRIPTS_DIR}/ck_run_v7.py" run \
      "${MODEL_URI}" \
      --context-len "${CONTEXT_LEN}" \
      --max-tokens "${MAX_TOKENS}" \
      --force-compile \
      --detailed-llamacpp-parity \
      --prompt "${PROMPT}" \
      --llama-layer 0 \
      --llama-include-global \
      --llama-filter "attn_norm,Qcur,Kcur,Vcur,Qcur_normed,Kcur_normed,attn_out,kqv_out,sa_out,ffn_norm,ffn_gate,ffn_up,ffn_down,ffn_post_norm,embd,embed,token,tok,final_norm,result_norm,logits" \
      --llama-stop-after 160 \
      --llama-timeout "${LLAMA_TIMEOUT_BASE}"
    if [[ -f "${INDEX_JSON}" ]]; then
      echo "[info] discovery index.json:"
      cat "${INDEX_JSON}"
      echo
    fi
  fi
fi

GGUF_PATH="$(ls "${WORK_DIR}"/*.gguf "${WORK_DIR}"/../*.gguf 2>/dev/null | head -n1 || true)"
LLAMA_TOKENIZE="${ROOT}/llama.cpp/build/bin/llama-tokenize"
if [[ -n "${GGUF_PATH}" && -x "${LLAMA_TOKENIZE}" ]]; then
  echo "[info] llama-tokenize ids (default):"
  "${LLAMA_TOKENIZE}" -m "${GGUF_PATH}" -p "${PROMPT}" --ids --log-disable || true
  echo "[info] llama-tokenize ids (--no-bos):"
  "${LLAMA_TOKENIZE}" -m "${GGUF_PATH}" -p "${PROMPT}" --ids --no-bos --log-disable || true
  echo
fi

echo "[info] ck tokenizer ids:"
python "${PARITY_DIR}/ck_dump_tokens.py" --model-dir "${WORK_DIR}" --prompt "${PROMPT}" || true
echo

if [[ -f "${INDEX_JSON}" ]] && ! grep -Eq '"name":"[^"]*embd[^"]*"|"name":"token_embedding"' "${INDEX_JSON}"; then
  echo "[warn] token embedding not found in llama dump; running discovery pass"
  rm -rf "${WORK_DIR}/llama_parity_dumps"
  python "${V7_SCRIPTS_DIR}/ck_run_v7.py" run \
    "${MODEL_URI}" \
    --context-len "${CONTEXT_LEN}" \
    --max-tokens "${MAX_TOKENS}" \
    --force-compile \
    --detailed-llamacpp-parity \
    --prompt "${PROMPT}" \
    --llama-layer 0 \
    --llama-include-global \
    --llama-filter inp_embd,embd,embed,token,tok \
    --llama-stop-after 4 \
    --llama-timeout "${LLAMA_TIMEOUT_BASE}"
  if [[ -f "${INDEX_JSON}" ]]; then
    echo "[info] discovery index.json:"
    cat "${INDEX_JSON}"
    echo
  fi
  echo "[stop] Update --llama-filter based on names above and re-run."
  exit 1
fi

echo "[3/4] Run parity_test.py"
CK_DUMP="${WORK_DIR}/ck_parity_dumps/dump.bin"
REF_DUMP="${WORK_DIR}/llama_parity_dumps/dump.bin"
if [[ -f "${REF_DUMP}" ]]; then
  python "${V7_SCRIPTS_DIR}/parity_test.py" \
    --ck-dump "${CK_DUMP}" \
    --ref-dump "${REF_DUMP}"
else
  echo "[warn] reference dump missing (${REF_DUMP}); running CK-only parity checks"
  python "${V7_SCRIPTS_DIR}/parity_test.py" \
    --ck-dump "${CK_DUMP}"
fi

if [[ "${RUN_QKV_CONTRACT}" == "1" ]]; then
  echo "[4/4] Run layer-0 qkv contract check (pre-rope, pre-attention)"
  QKV_ARGS=(
    --model-dir "${WORK_DIR}"
    --decode-token "${QKV_DECODE_TOKEN}"
    --prefill-tokens "${QKV_PREFILL_TOKENS}"
    --tol "${QKV_TOL}"
  )
  if [[ "${QKV_NO_FAIL_FAST}" == "1" ]]; then
    QKV_ARGS+=(--no-fail-fast)
  fi
  python "${PARITY_DIR}/check_layer0_qkv_contract.py" "${QKV_ARGS[@]}"
else
  echo "[4/4] Skip layer-0 qkv contract check (RUN_QKV_CONTRACT=${RUN_QKV_CONTRACT})"
fi

echo "[done] If token_embd is missing in the llama index above, re-run with a discovery filter:"
echo "  --llama-filter embd,emb,token,tok"
