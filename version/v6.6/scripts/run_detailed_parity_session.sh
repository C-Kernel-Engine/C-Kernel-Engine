#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  bash version/v6.6/scripts/run_detailed_parity_session.sh <model_uri> <family> [extra args...]

Runs detailed parity analysis into a timestamped output dir so existing ck_build
artifacts are not overwritten.
EOF
  exit 1
fi

MODEL_URI="$1"
FAMILY="$2"
shift 2

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_BASE="${HOME}/.cache/ck-engine-v6.6/parity_runs/${FAMILY}_${TS}"
OUT_DIR="${RUN_BASE}/ck_build"

mkdir -p "${OUT_DIR}"
echo "[parity-session] output_dir=${OUT_DIR}"

python "${ROOT}/version/v6.6/scripts/detailed_parity_analysis.py" \
  --model-uri "${MODEL_URI}" \
  --family "${FAMILY}" \
  --output-dir "${OUT_DIR}" \
  "$@"

echo "[parity-session] report_json=${OUT_DIR}/detailed_parity_analysis.json"
echo "[parity-session] report_md=${OUT_DIR}/detailed_parity_analysis.md"
