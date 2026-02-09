#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  bash version/v6.6/scripts/run_model_autocheck_session.sh <model_uri> <family> [extra args...]

Examples:
  bash version/v6.6/scripts/run_model_autocheck_session.sh \
    hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf gemma --force-compile

  bash version/v6.6/scripts/run_model_autocheck_session.sh \
    hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf qwen2 --skip-ck-run
EOF
  exit 1
fi

MODEL_URI="$1"
FAMILY="$2"
shift 2

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_BASE="${HOME}/.cache/ck-engine-v6.6/autocheck_runs/${FAMILY}_${TS}"
OUT_DIR="${RUN_BASE}/ck_build"

mkdir -p "${OUT_DIR}"
echo "[autocheck-session] output_dir=${OUT_DIR}"

python "${ROOT}/version/v6.6/scripts/model_autocheck.py" \
  --model-uri "${MODEL_URI}" \
  --family "${FAMILY}" \
  --output-dir "${OUT_DIR}" \
  "$@"

echo "[autocheck-session] report_json=${OUT_DIR}/model_autocheck.json"
echo "[autocheck-session] report_md=${OUT_DIR}/model_autocheck.md"
