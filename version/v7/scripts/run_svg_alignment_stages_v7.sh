#!/usr/bin/env bash
set -euo pipefail

# Run SVG alignment stages (DPO/GRPO/PPO) on top of an existing v7 run-dir.
#
# Current v7 behavior:
# - stage metadata: dpo/grpo/ppo is fully supported in pipeline/visualizer
# - objective math: these stages currently run as CE-surrogate updates through
#   train_data_pipeline_v7.py (until true objective-specific trainers land)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

RUN=""
INSTRUCTION_DATA=""
OUT_DIR=""
PREFIX="svg_alignment"
MAX_SAMPLES="50000"
SEED="42"
EPOCHS="1"
SEQ_LEN="512"
TOTAL_TOKENS="1048576"
GRAD_ACCUM="1"
MAX_GRAD_NORM="1.0"
LR_DPO="8e-5"
LR_GRPO="6e-5"
LR_PPO="5e-5"
RUN_DPO=0
RUN_GRPO=0
RUN_PPO=0
PLAN_ONLY=0
SKIP_VIS=0

usage() {
  cat <<'USAGE'
Usage:
  bash version/v7/scripts/run_svg_alignment_stages_v7.sh --run <run_dir> [options]

Required:
  --run PATH                     Existing v7 run-dir

Optional:
  --instruction-data PATH        Input instruction file (<task>...<svg>... rows)
                                 default: infer from training_plan.json stage=sft dataset path
  --out-dir PATH                 Alignment artifact dir (default: <run>/alignment)
  --prefix NAME                  Dataset prefix (default: svg_alignment)
  --max-samples N                Max alignment rows (default: 50000)
  --seed N                       Random seed (default: 42)
  --epochs N                     Epochs per stage (default: 1)
  --seq-len N                    Seq len per stage (default: 512)
  --total-tokens N               Total token budget per stage (default: 1048576)
  --grad-accum N                 Grad accumulation (default: 1)
  --max-grad-norm F              Max grad norm (default: 1.0)
  --lr-dpo F                     DPO-stage LR (default: 8e-5)
  --lr-grpo F                    GRPO-stage LR (default: 6e-5)
  --lr-ppo F                     PPO-stage LR (default: 5e-5)

Stage flags:
  --run-dpo                      Run DPO stage
  --run-grpo                     Run GRPO stage
  --run-ppo                      Run PPO stage
                                 If none are provided: run all three.

Controls:
  --plan-only                   Build alignment datasets + summary only (no training)
  --skip-visualizer-refresh      Skip open_ir_visualizer.py generate step
  -h, --help                     Show help

Example:
  bash version/v7/scripts/run_svg_alignment_stages_v7.sh \
    --run "$HOME/.cache/ck-engine-v7/models/train/svg_demo" \
    --instruction-data "$HOME/.cache/ck-engine-v7/models/train/svg_demo/data/svg_demo_instruction_train.txt" \
    --prefix svg_demo --run-dpo --run-grpo --run-ppo \
    --plan-only \
    --epochs 1 --seq-len 512 --total-tokens 1048576
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="${2:?}"; shift 2 ;;
    --instruction-data) INSTRUCTION_DATA="${2:?}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
    --prefix) PREFIX="${2:?}"; shift 2 ;;
    --max-samples) MAX_SAMPLES="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --epochs) EPOCHS="${2:?}"; shift 2 ;;
    --seq-len) SEQ_LEN="${2:?}"; shift 2 ;;
    --total-tokens) TOTAL_TOKENS="${2:?}"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="${2:?}"; shift 2 ;;
    --max-grad-norm) MAX_GRAD_NORM="${2:?}"; shift 2 ;;
    --lr-dpo) LR_DPO="${2:?}"; shift 2 ;;
    --lr-grpo) LR_GRPO="${2:?}"; shift 2 ;;
    --lr-ppo) LR_PPO="${2:?}"; shift 2 ;;
    --run-dpo) RUN_DPO=1; shift ;;
    --run-grpo) RUN_GRPO=1; shift ;;
    --run-ppo) RUN_PPO=1; shift ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    --skip-visualizer-refresh) SKIP_VIS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$RUN" ]; then
  echo "[ERROR] --run is required" >&2
  usage
  exit 2
fi

RUN="$(python3 - <<PY
from pathlib import Path
print(Path("$RUN").expanduser().resolve())
PY
)"
if [ ! -d "$RUN" ]; then
  echo "[ERROR] run dir missing: $RUN" >&2
  exit 2
fi
if [ ! -f "$RUN/weights_manifest.json" ]; then
  echo "[ERROR] run dir missing weights_manifest.json: $RUN" >&2
  exit 2
fi

if [ "$RUN_DPO" -eq 0 ] && [ "$RUN_GRPO" -eq 0 ] && [ "$RUN_PPO" -eq 0 ]; then
  RUN_DPO=1
  RUN_GRPO=1
  RUN_PPO=1
fi

if [ -z "$OUT_DIR" ]; then
  OUT_DIR="$RUN/alignment"
fi
mkdir -p "$OUT_DIR"

if [ -z "$INSTRUCTION_DATA" ]; then
  if [ -f "$RUN/training_plan.json" ]; then
    INSTRUCTION_DATA="$(jq -r '.stages[]? | select(.stage=="sft") | .datasets[]?.path // empty' "$RUN/training_plan.json" | head -n 1 || true)"
  fi
fi
if [ -z "$INSTRUCTION_DATA" ]; then
  if [ -f "$RUN/training_pipeline_latest.json" ]; then
    INSTRUCTION_DATA="$(jq -r '.pipeline.stages[]? | select((.stage // .stage_id)=="sft") | .datasets[]?.path // empty' "$RUN/training_pipeline_latest.json" | head -n 1 || true)"
  fi
fi
if [ -z "$INSTRUCTION_DATA" ]; then
  INSTRUCTION_DATA="$RUN/data/$(basename "$RUN")_instruction_train.txt"
fi
if [ ! -f "$INSTRUCTION_DATA" ]; then
  echo "[ERROR] instruction data missing: $INSTRUCTION_DATA" >&2
  echo "        pass --instruction-data PATH explicitly." >&2
  exit 2
fi

echo "[align] run=$RUN"
echo "[align] instruction_data=$INSTRUCTION_DATA"
echo "[align] out_dir=$OUT_DIR"
echo "[align] stages: dpo=$RUN_DPO grpo=$RUN_GRPO ppo=$RUN_PPO"
echo "[align] mode: $([ "$PLAN_ONLY" -eq 1 ] && echo plan-only || echo execute)"

"$PY" version/v7/scripts/build_svg_alignment_datasets_v7.py \
  --instruction-data "$INSTRUCTION_DATA" \
  --out-dir "$OUT_DIR" \
  --prefix "$PREFIX" \
  --max-samples "$MAX_SAMPLES" \
  --seed "$SEED"

DPO_DATA="$OUT_DIR/${PREFIX}_dpo_ce_train.txt"
GRPO_DATA="$OUT_DIR/${PREFIX}_grpo_ce_train.txt"
PPO_DATA="$OUT_DIR/${PREFIX}_ppo_ce_train.txt"
MANIFEST="$OUT_DIR/${PREFIX}_alignment_manifest.json"
SUMMARY="$OUT_DIR/alignment_stage_run_latest.json"

run_stage() {
  local stage="$1"
  local data="$2"
  local lr="$3"
  local report="$OUT_DIR/train_${stage}_report.json"

  if [ ! -f "$data" ]; then
    echo "[ERROR] stage=$stage missing data: $data" >&2
    return 2
  fi

  echo "[align] stage=$stage data=$data lr=$lr"
  set +e
  "$PY" version/v7/scripts/train_data_pipeline_v7.py \
    --run "$RUN" \
    --curriculum-stage "$stage" \
    --tokenizer ascii_bpe \
    --reuse-run-tokenizer \
    --strict-data-gates \
    --data "$data" \
    --epochs "$EPOCHS" \
    --seq-len "$SEQ_LEN" \
    --total-tokens "$TOTAL_TOKENS" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$lr" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --json-out "$report" \
    --no-open-visualizer
  local rc=$?
  set -e
  echo "$rc"
}

RC_DPO=-1
RC_GRPO=-1
RC_PPO=-1

if [ "$PLAN_ONLY" -eq 1 ]; then
  [ "$RUN_DPO" -eq 1 ] && RC_DPO=0
  [ "$RUN_GRPO" -eq 1 ] && RC_GRPO=0
  [ "$RUN_PPO" -eq 1 ] && RC_PPO=0
else
  if [ "$RUN_DPO" -eq 1 ]; then
    RC_DPO="$(run_stage dpo "$DPO_DATA" "$LR_DPO")"
  fi
  if [ "$RUN_GRPO" -eq 1 ]; then
    RC_GRPO="$(run_stage grpo "$GRPO_DATA" "$LR_GRPO")"
  fi
  if [ "$RUN_PPO" -eq 1 ]; then
    RC_PPO="$(run_stage ppo "$PPO_DATA" "$LR_PPO")"
  fi
fi

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

summary = {
    "schema": "ck.svg_alignment_run.v1",
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "run_dir": "$RUN",
    "objective_mode": "ce_surrogate",
    "plan_only": bool($PLAN_ONLY),
    "inputs": {
        "instruction_data": "$INSTRUCTION_DATA",
        "dataset_manifest": "$MANIFEST",
    },
    "stages": {
        "dpo": {"enabled": bool($RUN_DPO), "rc": int("$RC_DPO")},
        "grpo": {"enabled": bool($RUN_GRPO), "rc": int("$RC_GRPO")},
        "ppo": {"enabled": bool($RUN_PPO), "rc": int("$RC_PPO")},
    },
    "reports": {
        "dpo": "$OUT_DIR/train_dpo_report.json",
        "grpo": "$OUT_DIR/train_grpo_report.json",
        "ppo": "$OUT_DIR/train_ppo_report.json",
    },
}
Path("$SUMMARY").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("[align] summary:", "$SUMMARY")
PY

if [ "$SKIP_VIS" -ne 1 ]; then
  python3 version/v7/tools/open_ir_visualizer.py --generate --run "$RUN" --html-only --strict-run-artifacts
fi

echo "[align] done"
echo "  manifest: $MANIFEST"
echo "  summary:  $SUMMARY"
exit 0
