#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

RUN_DIR="${RUN_DIR:-/tmp/v7_roadmap_check}"
SKIP_INFERENCE_SMOKE=0
FORCE_CLEAN=0
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-8}"
TRAIN_TOTAL_TOKENS="${TRAIN_TOTAL_TOKENS:-64}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-2}"
PROMPT_TEXT="${PROMPT_TEXT:-hello}"

usage() {
  cat <<USAGE
Usage: scripts/v7_roadmap_check.sh [options]

Options:
  --run-dir <path>             Run directory (default: /tmp/v7_roadmap_check)
  --python <path>              Python executable (default: .venv/bin/python or python3)
  --skip-inference-smoke       Skip 'make v7-inference-smoke'
  --force-clean                Delete run dir before running
  --train-epochs <n>           Training epochs for backend checks (default: 1)
  --train-seq-len <n>          Sequence length (default: 8)
  --train-total-tokens <n>     Total tokens (default: 64)
  --train-grad-accum <n>       Grad accumulation steps (default: 2)
  --prompt <text>              Prompt text for train checks (default: hello)
  --help                       Show this help

Notes:
  - This script does not touch git state.
  - Logs are written under: <run-dir>/roadmap_check_logs/
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-inference-smoke)
      SKIP_INFERENCE_SMOKE=1
      shift
      ;;
    --force-clean)
      FORCE_CLEAN=1
      shift
      ;;
    --train-epochs)
      TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --train-seq-len)
      TRAIN_SEQ_LEN="$2"
      shift 2
      ;;
    --train-total-tokens)
      TRAIN_TOTAL_TOKENS="$2"
      shift 2
      ;;
    --train-grad-accum)
      TRAIN_GRAD_ACCUM="$2"
      shift 2
      ;;
    --prompt)
      PROMPT_TEXT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$FORCE_CLEAN" == "1" ]]; then
  rm -rf "$RUN_DIR"
fi

LOG_DIR="$RUN_DIR/roadmap_check_logs"
mkdir -p "$LOG_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python not found: $PYTHON_BIN" >&2
  exit 2
fi

declare -A STATUS
declare -A LOG_PATH
declare -A DUR_SEC

step_run() {
  local key="$1"
  local desc="$2"
  shift 2
  local log="$LOG_DIR/${key}.log"
  local t0 t1 rc
  t0="$(date +%s)"

  echo "[RUN] $key :: $desc"
  if "$@" >"$log" 2>&1; then
    rc=0
    STATUS["$key"]="PASS"
  else
    rc=$?
    STATUS["$key"]="FAIL($rc)"
  fi

  t1="$(date +%s)"
  DUR_SEC["$key"]="$((t1 - t0))"
  LOG_PATH["$key"]="$log"
  echo "[${STATUS[$key]}] $key (${DUR_SEC[$key]}s)"
}

step_skip() {
  local key="$1"
  local desc="$2"
  STATUS["$key"]="SKIP"
  DUR_SEC["$key"]="0"
  LOG_PATH["$key"]="-"
  echo "[SKIP] $key :: $desc"
}

run_py_cmd() {
  "$PYTHON_BIN" "$@"
}

# PR1 checks
step_run "01_contracts" "make v7-validate-contracts" make --no-print-directory v7-validate-contracts
if [[ "$SKIP_INFERENCE_SMOKE" == "1" ]]; then
  step_skip "02_inference_smoke" "make v7-inference-smoke"
else
  step_run "02_inference_smoke" "make v7-inference-smoke" make --no-print-directory v7-inference-smoke
fi

# IR/codegen/compile readiness
step_run "03_train_compile_smoke" "make v7-train-compile-smoke" make --no-print-directory v7-train-compile-smoke

# Init and train checks
step_run "04_init_runtime" "ck_run_v7.py init --generate-ir --generate-runtime --strict" \
  run_py_cmd version/v7/scripts/ck_run_v7.py init --run "$RUN_DIR" --generate-ir --generate-runtime --strict

step_run "05_train_ck" "ck_run_v7.py train --backend ck" \
  run_py_cmd version/v7/scripts/ck_run_v7.py train \
    --run "$RUN_DIR" \
    --backend ck \
    --prompt "$PROMPT_TEXT" \
    --train-epochs "$TRAIN_EPOCHS" \
    --train-seq-len "$TRAIN_SEQ_LEN" \
    --train-total-tokens "$TRAIN_TOTAL_TOKENS" \
    --train-grad-accum "$TRAIN_GRAD_ACCUM" \
    --train-json-out "$RUN_DIR/train_ck.json"

if [[ "${STATUS[05_train_ck]}" == PASS ]]; then
  step_run "06_ck_json_sanity" "validate train_ck.json has finite/non-trivial loss" \
    "$PYTHON_BIN" -c "import json,math,sys; p='$RUN_DIR/train_ck.json'; d=json.load(open(p)); c=d.get('loss_curve') or []; \
steps=int(d.get('steps',0)); \
assert steps>0, 'steps<=0'; \
assert c, 'empty loss_curve'; \
vals=[float(r.get('loss_ck', float('nan'))) for r in c]; \
assert all(math.isfinite(v) for v in vals), 'non-finite loss'; \
assert (max(vals)-min(vals))>1e-12 if len(vals)>1 else True, 'flat loss curve'; \
print('ok steps',steps,'first',vals[0],'last',vals[-1])"
else
  step_skip "06_ck_json_sanity" "validate train_ck.json has finite/non-trivial loss"
fi

step_run "07_train_both" "ck_run_v7.py train --backend both" \
  run_py_cmd version/v7/scripts/ck_run_v7.py train \
    --run "$RUN_DIR" \
    --backend both \
    --prompt "$PROMPT_TEXT" \
    --train-epochs "$TRAIN_EPOCHS" \
    --train-seq-len "$TRAIN_SEQ_LEN" \
    --train-total-tokens "$TRAIN_TOTAL_TOKENS" \
    --train-grad-accum "$TRAIN_GRAD_ACCUM" \
    --train-json-out "$RUN_DIR/train_both.json"

if [[ "${STATUS[07_train_both]}" == PASS ]]; then
  step_run "08_both_parity" "validate train_both.json pass_parity=true" \
    "$PYTHON_BIN" -c "import json,sys; d=json.load(open('$RUN_DIR/train_both.json')); \
assert bool(d.get('pass_parity', False)), 'pass_parity=false'; \
print('parity ok max_loss_abs_diff',d.get('max_loss_abs_diff'),'max_param_diff',d.get('final_param_max_abs_diff'))"
else
  step_skip "08_both_parity" "validate train_both.json pass_parity=true"
fi

pr_pass() {
  local keys=("$@")
  local k
  for k in "${keys[@]}"; do
    case "${STATUS[$k]:-MISSING}" in
      PASS|SKIP)
        ;;
      *)
        return 1
        ;;
    esac
  done
  return 0
}

echo ""
echo "============================================================"
echo "v7 ROADMAP CHECK SUMMARY"
echo "============================================================"
printf "%-24s %-10s %-8s %s\n" "CHECK" "STATUS" "SEC" "LOG"
printf "%-24s %-10s %-8s %s\n" "------------------------" "----------" "--------" "---"
for key in 01_contracts 02_inference_smoke 03_train_compile_smoke 04_init_runtime 05_train_ck 06_ck_json_sanity 07_train_both 08_both_parity; do
  printf "%-24s %-10s %-8s %s\n" "$key" "${STATUS[$key]:-MISSING}" "${DUR_SEC[$key]:-0}" "${LOG_PATH[$key]:--}"
done

if pr_pass 01_contracts 02_inference_smoke; then
  PR1_STATUS="PASS"
else
  PR1_STATUS="FAIL"
fi
if pr_pass 03_train_compile_smoke 04_init_runtime 05_train_ck; then
  PR2_STATUS="PASS"
else
  PR2_STATUS="FAIL"
fi
if pr_pass 05_train_ck 06_ck_json_sanity; then
  PR3_STATUS="PASS"
else
  PR3_STATUS="FAIL"
fi
if pr_pass 07_train_both 08_both_parity; then
  PARITY_STATUS="PASS"
else
  PARITY_STATUS="FAIL"
fi

echo ""
echo "Milestone status:"
echo "  PR1 (contracts + inference smoke): $PR1_STATUS"
echo "  PR2 (generated runtime execution): $PR2_STATUS"
echo "  PR3 (numerically real CK runtime): $PR3_STATUS"
echo "  Parity baseline (backend both):    $PARITY_STATUS"
echo ""
echo "Run dir:  $RUN_DIR"
echo "Logs dir: $LOG_DIR"

if [[ "$PR1_STATUS" == "PASS" && "$PR2_STATUS" == "PASS" && "$PR3_STATUS" == "PASS" && "$PARITY_STATUS" == "PASS" ]]; then
  exit 0
fi
exit 1
