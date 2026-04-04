#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
SPEC_TAG="${SPEC_TAG:-spec17}"
RUN="${1:-$MODEL_CACHE_ROOT/train/${SPEC_TAG}_scene_bundle_l3_d192_h384_ctx768_r1}"
DATASET_PREFIX="${DATASET_PREFIX:-${SPEC_TAG}_scene_bundle}"
WORKSPACE="$RUN/dataset"

ensure_oneapi_runtime_libs() {
  local candidates=(
    "/opt/intel/oneapi/compiler/latest/lib"
    "/opt/intel/oneapi/compiler/2025.3/lib"
    "/opt/intel/oneapi/2025.3/lib"
    "$HOME/intel/oneapi/compiler/latest/lib"
    "$HOME/intel/oneapi/compiler/2025.3/lib"
    "$HOME/intel/oneapi/2025.3/lib"
  )
  local found=""
  local path_entry=""
  for path_entry in "${candidates[@]}"; do
    if [ -f "$path_entry/libimf.so" ]; then
      found="$path_entry"
      break
    fi
  done
  if [ -n "$found" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$found:"*) ;;
      *) export LD_LIBRARY_PATH="$found${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    printf '[%s-env] oneapi_runtime_lib=%s\n' "$SPEC_TAG" "$found"
  fi
}

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_timed() {
  local label="$1"
  shift
  local started_epoch
  local ended_epoch
  local rc
  started_epoch="$(date +%s)"
  printf '[%s-timing] stage=%s event=start started_at=%s\n' "$SPEC_TAG" "$label" "$(timestamp_utc)"
  "$@"
  rc=$?
  ended_epoch="$(date +%s)"
  printf '[%s-timing] stage=%s event=end ended_at=%s elapsed_sec=%s\n' \
    "$SPEC_TAG" "$label" "$(timestamp_utc)" "$((ended_epoch - started_epoch))"
  return "$rc"
}

clear_stale_postrun_artifacts() {
  local run_dir="$1"
  rm -f \
    "$run_dir/${SPEC_TAG}_probe_contract.json" \
    "$run_dir/${SPEC_TAG}_probe_report.json" \
    "$run_dir/${SPEC_TAG}_probe_report.html" \
    "$run_dir/${SPEC_TAG}_tested_prompts_report.html" \
    "$run_dir/${SPEC_TAG}_tested_prompts_report.md" \
    "$run_dir/ir_report.html"
}

refresh_ir_hub() {
  python3 version/v7/tools/open_ir_hub.py --models-root "$MODEL_CACHE_ROOT"
}

assert_disk_headroom() {
  local label="$1"
  local min_free_gb="$2"
  local path="$3"
  python3 version/v7/scripts/check_disk_headroom_v7.py \
    --path "$path" \
    --min-free-gb "$min_free_gb" \
    --label "$label"
}

resolve_run_dir() {
  local ref="$1"
  if [ -z "$ref" ]; then
    return 1
  fi
  if [[ "$ref" == /* ]]; then
    printf '%s\n' "$ref"
    return 0
  fi
  if [ -d "$ref" ]; then
    (cd "$ref" && pwd)
    return 0
  fi
  printf '%s\n' "$MODEL_CACHE_ROOT/train/$ref"
}

scale_tokens() {
  local total="$1"
  local numerator="$2"
  local denominator="$3"
  python3 - "$total" "$numerator" "$denominator" "$CONTEXT_LEN" <<'PY'
import sys
total = max(0, int(sys.argv[1]))
num = max(1, int(sys.argv[2]))
den = max(1, int(sys.argv[3]))
ctx = max(64, int(sys.argv[4]))
scaled = (total * num + den - 1) // den
print(max(ctx, scaled))
PY
}

sync_training_plan_stage() {
  local plan_path="$1"
  local active_stage="$2"
  python3 - "$plan_path" "$active_stage" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
active_stage = sys.argv[2]
if not plan_path.exists():
    raise SystemExit(0)

doc = json.loads(plan_path.read_text(encoding="utf-8"))
doc["active_stage"] = active_stage
for stage in doc.get("stages") or []:
    if not isinstance(stage, dict):
        continue
    name = str(stage.get("stage") or "")
    if active_stage == "completed":
        if name in {"pretrain", "midtrain"}:
            stage["status"] = "completed"
    elif name == active_stage:
        stage["status"] = "active"
    elif active_stage == "midtrain" and name == "pretrain":
        stage["status"] = "completed"
    elif stage.get("status") != "completed":
        stage["status"] = "planned"

plan_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY
}

build_probe_artifacts() {
  local run_dir="$1"
  local probe_contract="$2"
  local dataset_prefix="$3"
  local context_len="$4"
  local probe_per_split="$5"
  local hidden_probe_per_split="$6"

  python3 "$PROBE_CONTRACT_BUILDER" \
    --run "$run_dir" \
    --prefix "$dataset_prefix" \
    --output "$probe_contract" \
    --per-split "$probe_per_split" \
    --hidden-per-split "$hidden_probe_per_split"

  .venv/bin/python version/v7/scripts/ck_run_v7.py run "$run_dir" \
    --generate-only \
    --context-len "$context_len"

  python3 version/v7/scripts/build_probe_report_v7.py \
    --run "$run_dir" \
    --contract "$probe_contract" \
    --output "$run_dir/${SPEC_TAG}_probe_report.html" \
    --json-out "$run_dir/${SPEC_TAG}_probe_report.json"

  python3 version/v7/scripts/build_structured_scene_tested_prompts_doc_v7.py \
    --probe-report "$run_dir/${SPEC_TAG}_probe_report.json" \
    --output-html "$run_dir/${SPEC_TAG}_tested_prompts_report.html" \
    --output-md "$run_dir/${SPEC_TAG}_tested_prompts_report.md"

  python3 version/v7/tools/open_ir_visualizer.py \
    --generate \
    --run "$run_dir" \
    --html-only \
    --strict-run-artifacts \
    --output "$run_dir/ir_report.html"

  if [ -n "$PROBE_AUTOPSY_SCRIPT" ]; then
    python3 "$PROBE_AUTOPSY_SCRIPT" \
      --probe-report "$run_dir/${SPEC_TAG}_probe_report.json" \
      --json-out "$run_dir/${SPEC_TAG}_probe_autopsy.json" \
      --md-out "$run_dir/${SPEC_TAG}_probe_autopsy.md"
  fi

  python3 version/v7/scripts/build_run_artifact_index_v7.py \
    --run "$run_dir" \
    --spec-tag "$SPEC_TAG" \
    --prefix "$dataset_prefix" \
    --json-out "$run_dir/artifact_index.json" \
    --md-out "$run_dir/artifact_index.md"
}

LAYERS="${LAYERS:-3}"
EMBED_DIM="${EMBED_DIM:-192}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
CONTEXT_LEN="${CONTEXT_LEN:-768}"

PRETRAIN_TOTAL_TOKENS="${PRETRAIN_TOTAL_TOKENS:-}"
MIDTRAIN_TOTAL_TOKENS="${MIDTRAIN_TOTAL_TOKENS:-}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
MIDTRAIN_EPOCHS="${MIDTRAIN_EPOCHS:-1}"
PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-2}"
MIDTRAIN_GRAD_ACCUM="${MIDTRAIN_GRAD_ACCUM:-2}"
WEIGHT_QUANTUM="${WEIGHT_QUANTUM:-5}"

RUN_MODE="${RUN_MODE:-canary}"
ALLOW_FULL_RUNG="${ALLOW_FULL_RUNG:-0}"
CANARY_TOKEN_NUMERATOR="${CANARY_TOKEN_NUMERATOR:-1}"
CANARY_TOKEN_DENOMINATOR="${CANARY_TOKEN_DENOMINATOR:-1}"

PROBE_PER_SPLIT="${PROBE_PER_SPLIT:-12}"
HIDDEN_PROBE_PER_SPLIT="${HIDDEN_PROBE_PER_SPLIT:-6}"
RUN_PARITY="${RUN_PARITY:-1}"
RUN_STAGE_EVAL="${RUN_STAGE_EVAL:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_STRICT="${PREFLIGHT_STRICT:-1}"
CANARY_PER_SPLIT="${CANARY_PER_SPLIT:-4}"
TRAIN_DRIVER="${TRAIN_DRIVER:-ck_cli}"
CK_CLI_LOG_EVERY="${CK_CLI_LOG_EVERY:-100}"
ANALYSIS_CHECKPOINTS="${ANALYSIS_CHECKPOINTS:-log}"
TRAIN_SAVE_EVERY="${TRAIN_SAVE_EVERY:-0}"
TRAIN_SAVE_FINAL="${TRAIN_SAVE_FINAL:-1}"
DISK_HEADROOM_PRETRAIN_GB="${DISK_HEADROOM_PRETRAIN_GB:-24}"
DISK_HEADROOM_MIDTRAIN_GB="${DISK_HEADROOM_MIDTRAIN_GB:-24}"
DISK_HEADROOM_POSTRUN_GB="${DISK_HEADROOM_POSTRUN_GB:-8}"
SEED_FROM_RUN="${SEED_FROM_RUN:-$MODEL_CACHE_ROOT/train/spec16_scene_bundle_l3_d192_h384_ctx768_r9}"
FREEZE_TOKENIZER_FROM="${FREEZE_TOKENIZER_FROM:-$SEED_FROM_RUN}"
SEED_WORKSPACE="${SEED_WORKSPACE:-}"
BLUEPRINT_PATH="${BLUEPRINT_PATH:-$ROOT/version/v7/reports/${SPEC_TAG}_curriculum_blueprint.json}"
MATERIALIZE_SCRIPT="${MATERIALIZE_SCRIPT:-version/v7/scripts/dataset/materialize_${SPEC_TAG}_scene_bundle_v7.py}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-version/v7/scripts/${SPEC_TAG}_preflight_v7.py}"
PROBE_CONTRACT_BUILDER="${PROBE_CONTRACT_BUILDER:-version/v7/scripts/build_${SPEC_TAG}_probe_contract_v7.py}"
COMPILER_SMOKE_SCRIPT="${COMPILER_SMOKE_SCRIPT:-}"
PROBE_AUTOPSY_SCRIPT="${PROBE_AUTOPSY_SCRIPT:-version/v7/scripts/build_bundle_probe_autopsy_v7.py}"
MATERIALIZE_EXTRA_ARGS="${MATERIALIZE_EXTRA_ARGS:-}"
RUN_KIND="${RUN_KIND:-seeded_${SPEC_TAG}_branch}"

RUN_SCOPE_SPEC="${RUN_SCOPE_SPEC:-spec17}"
RUN_SCOPE_RUNG="${RUN_SCOPE_RUNG:-r1}"
RUN_SCOPE_FAMILY="${RUN_SCOPE_FAMILY:-visual_scene_bundle}"
RUN_SCOPE_TITLE="${RUN_SCOPE_TITLE:-Spec17 R1 Bounded Intent Canary}"
RUN_SCOPE_OBJECTIVE="${RUN_SCOPE_OBJECTIVE:-Validate that a bounded intent prompt can warm-start from frozen spec16 r9 and still emit exactly one shared [bundle] scene contract without prompt or prose leakage.}"
RUN_SCOPE_HYPOTHESIS="${RUN_SCOPE_HYPOTHESIS:-A small seeded canary over the spec17 bounded-intent curriculum should retain compiler-valid bundle emission across all three families and show nonzero exact learning before we spend full spec17 compute.}"
RUN_SCOPE_PROMPT_CONTRACT="${RUN_SCOPE_PROMPT_CONTRACT:-Bounded intent prompt in: topic, goal, audience, and limited hints only. Family/style/topology tags are not part of the external user prompt contract.}"
RUN_SCOPE_OUTPUT_CONTRACT="${RUN_SCOPE_OUTPUT_CONTRACT:-Emit exactly one shared [bundle] scene_bundle.v1 only, using the frozen spec16 renderer and canonicalizer boundary.}"
RUN_NAME="$(basename "$RUN")"
PROMOTION_TARGET_RUN_NAME="$RUN_NAME"
if [[ "$RUN_NAME" =~ ^(.*_r)([0-9]+)$ ]]; then
  PROMOTION_TARGET_RUN_NAME="${BASH_REMATCH[1]}$((BASH_REMATCH[2] + 1))"
fi

TRAIN_SAVE_FINAL_FLAG=()
if [ "${TRAIN_SAVE_FINAL}" != "1" ]; then
  TRAIN_SAVE_FINAL_FLAG+=(--no-train-save-final)
fi

TOKENIZER_CORPUS="$WORKSPACE/tokenizer/${DATASET_PREFIX}_tokenizer_corpus.txt"
PRETRAIN_DATA="$WORKSPACE/pretrain/train/${DATASET_PREFIX}_pretrain_train.txt"
MIDTRAIN_DATA="$WORKSPACE/midtrain/train/${DATASET_PREFIX}_midtrain_train.txt"
TOKENIZER_JSON="$WORKSPACE/tokenizer/tokenizer.json"
TOKENIZER_BIN="$WORKSPACE/tokenizer/tokenizer_bin"
PROBE_CONTRACT="$RUN/${SPEC_TAG}_probe_contract.json"
PREFLIGHT_JSON="$RUN/${SPEC_TAG}_preflight.json"
BLUEPRINT_AUDIT_JSON="$RUN/${SPEC_TAG}_curriculum_audit.json"
BLUEPRINT_AUDIT_MD="$RUN/${SPEC_TAG}_curriculum_audit.md"
COMPILER_SMOKE_JSON="$RUN/${SPEC_TAG}_compiler_smoke_report.json"
COMPILER_SMOKE_HTML="$RUN/${SPEC_TAG}_compiler_smoke_report.html"
COMPILER_SMOKE_OUT_DIR="$RUN/${SPEC_TAG}_compiler_smoke"
RENDER_CATALOG_JSON="$WORKSPACE/manifests/generated/structured_atoms/${DATASET_PREFIX}_render_catalog.json"

cd "$ROOT"
ensure_oneapi_runtime_libs
mkdir -p "$RUN"
clear_stale_postrun_artifacts "$RUN"

if [ "$RUN_MODE" != "canary" ] && [ "$RUN_MODE" != "full" ]; then
  echo "[$SPEC_TAG-policy] unsupported RUN_MODE=$RUN_MODE; use canary or full." >&2
  exit 1
fi
if [ "$RUN_MODE" = "full" ] && [ "$ALLOW_FULL_RUNG" != "1" ]; then
  echo "[$SPEC_TAG-policy] full $SPEC_TAG rung is blocked by default; set ALLOW_FULL_RUNG=1 to continue." >&2
  exit 1
fi
if [ ! -f "$BLUEPRINT_PATH" ]; then
  echo "[$SPEC_TAG-policy] blueprint missing at $BLUEPRINT_PATH" >&2
  exit 1
fi

SEED_RUN_DIR="$(resolve_run_dir "$SEED_FROM_RUN")"
FREEZE_TOKENIZER_RUN_DIR="$(resolve_run_dir "$FREEZE_TOKENIZER_FROM")"
if [ ! -d "$SEED_RUN_DIR" ]; then
  echo "[$SPEC_TAG-seed] missing seed run directory: $SEED_RUN_DIR" >&2
  exit 1
fi
if [ ! -f "$SEED_RUN_DIR/weights.bump" ]; then
  echo "[$SPEC_TAG-seed] missing seed weights: $SEED_RUN_DIR/weights.bump" >&2
  exit 1
fi
if [ ! -d "$FREEZE_TOKENIZER_RUN_DIR" ]; then
  echo "[$SPEC_TAG-tokenizer] missing frozen-tokenizer run directory: $FREEZE_TOKENIZER_RUN_DIR" >&2
  exit 1
fi

run_timed blueprint_audit \
  python3 version/v7/scripts/audit_curriculum_blueprint_v7.py \
    --blueprint "$BLUEPRINT_PATH" \
    --strict \
    --out-json "$BLUEPRINT_AUDIT_JSON" \
    --out-md "$BLUEPRINT_AUDIT_MD"

MATERIALIZE_ARGS=(
  python3
  "$MATERIALIZE_SCRIPT"
  --workspace "$WORKSPACE"
  --prefix "$DATASET_PREFIX"
  --freeze-tokenizer-run "$FREEZE_TOKENIZER_RUN_DIR"
  --weight-quantum "$WEIGHT_QUANTUM"
  --force
)
if [ -n "$SEED_WORKSPACE" ]; then
  MATERIALIZE_ARGS+=(--seed-workspace "$SEED_WORKSPACE")
fi
if [ -n "$MATERIALIZE_EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $MATERIALIZE_EXTRA_ARGS )
  MATERIALIZE_ARGS+=("${EXTRA_ARGS[@]}")
fi
run_timed materialize_workspace "${MATERIALIZE_ARGS[@]}"

run_timed blueprint_audit_materialized \
  python3 version/v7/scripts/audit_curriculum_blueprint_v7.py \
    --blueprint "$BLUEPRINT_PATH" \
    --render-catalog "$RENDER_CATALOG_JSON" \
    --strict \
    --out-json "$BLUEPRINT_AUDIT_JSON" \
    --out-md "$BLUEPRINT_AUDIT_MD"

if [ -n "$COMPILER_SMOKE_SCRIPT" ]; then
  run_timed compiler_smoke \
    python3 "$COMPILER_SMOKE_SCRIPT" \
      --run "$RUN" \
      --prefix "$DATASET_PREFIX" \
      --out-dir "$COMPILER_SMOKE_OUT_DIR" \
      --json-out "$COMPILER_SMOKE_JSON" \
      --html-out "$COMPILER_SMOKE_HTML"
fi

if [ "$RUN_PREFLIGHT" = "1" ]; then
  PREFLIGHT_ARGS=(
    python3
    "$PREFLIGHT_SCRIPT"
    --run "$RUN"
    --prefix "$DATASET_PREFIX"
    --tokenizer-json "$TOKENIZER_JSON"
    --tokenizer-bin "$TOKENIZER_BIN"
    --seq-len "$CONTEXT_LEN"
    --pretrain-epochs "$PRETRAIN_EPOCHS"
    --midtrain-epochs "$MIDTRAIN_EPOCHS"
    --canary-per-split "$CANARY_PER_SPLIT"
    --json-out "$PREFLIGHT_JSON"
  )
  if [ -n "$PRETRAIN_TOTAL_TOKENS" ]; then
    PREFLIGHT_ARGS+=(--current-pretrain-total-tokens "$PRETRAIN_TOTAL_TOKENS")
  fi
  if [ -n "$MIDTRAIN_TOTAL_TOKENS" ]; then
    PREFLIGHT_ARGS+=(--current-midtrain-total-tokens "$MIDTRAIN_TOTAL_TOKENS")
  fi
  if [ "$PREFLIGHT_STRICT" = "1" ]; then
    PREFLIGHT_ARGS+=(--strict)
  fi
  run_timed preflight "${PREFLIGHT_ARGS[@]}"
fi

RECOMMENDED_PRETRAIN_TOTAL_TOKENS="$(jq -r '.stages.pretrain.recommended_total_tokens // 0' "$PREFLIGHT_JSON")"
RECOMMENDED_MIDTRAIN_TOTAL_TOKENS="$(jq -r '.stages.midtrain.recommended_total_tokens // 0' "$PREFLIGHT_JSON")"

if [ -z "$PRETRAIN_TOTAL_TOKENS" ] || [ "$PRETRAIN_TOTAL_TOKENS" = "0" ] || [ "$PRETRAIN_TOTAL_TOKENS" = "null" ]; then
  if [ "$RUN_MODE" = "canary" ]; then
    PRETRAIN_TOTAL_TOKENS="$(scale_tokens "$RECOMMENDED_PRETRAIN_TOTAL_TOKENS" "$CANARY_TOKEN_NUMERATOR" "$CANARY_TOKEN_DENOMINATOR")"
  else
    PRETRAIN_TOTAL_TOKENS="$RECOMMENDED_PRETRAIN_TOTAL_TOKENS"
  fi
fi
if [ -z "$MIDTRAIN_TOTAL_TOKENS" ] || [ "$MIDTRAIN_TOTAL_TOKENS" = "0" ] || [ "$MIDTRAIN_TOTAL_TOKENS" = "null" ]; then
  if [ "$RUN_MODE" = "canary" ]; then
    MIDTRAIN_TOTAL_TOKENS="$(scale_tokens "$RECOMMENDED_MIDTRAIN_TOTAL_TOKENS" "$CANARY_TOKEN_NUMERATOR" "$CANARY_TOKEN_DENOMINATOR")"
  else
    MIDTRAIN_TOTAL_TOKENS="$RECOMMENDED_MIDTRAIN_TOTAL_TOKENS"
  fi
fi

printf '[%s-budget] run_mode=%s recommended_pretrain_total_tokens=%s selected_pretrain_total_tokens=%s recommended_midtrain_total_tokens=%s selected_midtrain_total_tokens=%s\n' \
  "$SPEC_TAG" \
  "$RUN_MODE" "$RECOMMENDED_PRETRAIN_TOTAL_TOKENS" "$PRETRAIN_TOTAL_TOKENS" "$RECOMMENDED_MIDTRAIN_TOTAL_TOKENS" "$MIDTRAIN_TOTAL_TOKENS"
printf '[%s-seed] seed_from=%s frozen_tokenizer_from=%s\n' "$SPEC_TAG" "$SEED_RUN_DIR" "$FREEZE_TOKENIZER_RUN_DIR"

VOCAB_SIZE="$(jq -r '.vocab_size // 0' "$TOKENIZER_BIN/tokenizer_meta.json")"
if [ -z "$VOCAB_SIZE" ] || [ "$VOCAB_SIZE" = "0" ] || [ "$VOCAB_SIZE" = "null" ]; then
  echo "failed to read tokenizer vocab size from $TOKENIZER_BIN/tokenizer_meta.json" >&2
  exit 1
fi

cat > "$RUN/training_plan.json" <<JSON
{
  "schema": "ck.training_plan.v1",
  "created_by": "${SPEC_TAG}_pretrain_midtrain_v7.sh",
  "active_stage": "pretrain",
  "run_policy": {
    "mode": "$RUN_MODE",
    "kind": "$RUN_KIND",
    "promotion_target": "$PROMOTION_TARGET_RUN_NAME",
    "frozen_seed_run": "$SEED_RUN_DIR"
  },
  "token_budget": {
    "recommended_pretrain_total_tokens": $RECOMMENDED_PRETRAIN_TOTAL_TOKENS,
    "selected_pretrain_total_tokens": $PRETRAIN_TOTAL_TOKENS,
    "recommended_midtrain_total_tokens": $RECOMMENDED_MIDTRAIN_TOTAL_TOKENS,
    "selected_midtrain_total_tokens": $MIDTRAIN_TOTAL_TOKENS,
    "canary_token_fraction": "$CANARY_TOKEN_NUMERATOR/$CANARY_TOKEN_DENOMINATOR"
  },
  "stage_order": ["pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"],
  "tokenizer": {
    "type": "ascii_bpe",
    "vocab_size": $VOCAB_SIZE,
    "tokenizer_corpora": [
      {
        "name": "$(basename "$TOKENIZER_CORPUS")",
        "path": "$TOKENIZER_CORPUS"
      }
    ]
  },
  "stages": [
    {
      "stage": "pretrain",
      "seq": 1,
      "status": "active",
      "enabled": true,
      "datasets": [
        {
          "name": "$(basename "$PRETRAIN_DATA")",
          "path": "$PRETRAIN_DATA",
          "kind": "generated_dataset"
        }
      ],
      "runs": []
    },
    {
      "stage": "midtrain",
      "seq": 2,
      "status": "planned",
      "enabled": true,
      "datasets": [
        {
          "name": "$(basename "$MIDTRAIN_DATA")",
          "path": "$MIDTRAIN_DATA",
          "kind": "generated_dataset"
        }
      ],
      "runs": []
    },
    {"stage": "sft", "seq": 3, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "dpo", "seq": 4, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "grpo", "seq": 5, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "ppo", "seq": 6, "status": "planned", "enabled": false, "datasets": [], "runs": []}
  ]
}
JSON

run_timed run_scope_init \
  python3 version/v7/scripts/init_run_scope_v7.py \
    --run "$RUN" \
    --spec "$RUN_SCOPE_SPEC" \
    --rung "$RUN_SCOPE_RUNG" \
    --family "$RUN_SCOPE_FAMILY" \
    --title "$RUN_SCOPE_TITLE" \
    --objective "$RUN_SCOPE_OBJECTIVE" \
    --hypothesis "$RUN_SCOPE_HYPOTHESIS" \
    --prompt-contract "$RUN_SCOPE_PROMPT_CONTRACT" \
    --output-contract "$RUN_SCOPE_OUTPUT_CONTRACT" \
    --success-gate "Canary shows nonzero exact learning across memory_map, timeline, and system_diagram." \
    --success-gate "No prompt-control leakage, raw SVG, or prose spill becomes the dominant failure mode." \
    --success-gate "Hidden prompts remain compiler-valid and renderable under the frozen spec16 bundle boundary." \
    --guardrail "Do not widen the output surface beyond exactly one shared [bundle]." \
    --guardrail "Do not introduce repair-language rows that spell out control junk." \
    --guardrail "Do not treat r1 as a promotion unless the contract is clearly learned." \
    --research-prior "Curriculum Learning (Bengio et al., 2009): front-load meaningful easier anchor surfaces, then widen to the harder bounded-intent bridge." \
    --research-prior "Chinchilla (Hoffmann et al., 2022): keep token budget explicit; do not confuse a canary with a full run." \
    --research-prior "phi-1 (Gunasekar et al., 2023): clean dense synthetic teaching data beats noisy scale for structured output." \
    --research-prior "HumanEval/Codex (Chen et al., 2021): measure correctness by executable or compilable structure, not loss alone." \
    --lesson-learned "Spec16 r9 proved the shared bundle contract can be learned at current model size." \
    --lesson-learned "Spec16 r10-r12 showed that post-hoc repair-language raw rungs regress more often than they help." \
    --lesson-learned "Spec17 is a new branch and should be judged first on clean contract retention, not on raw score chasing." \
    --read-first "$BLUEPRINT_PATH" \
    --read-first "$ROOT/version/v7/reports/SPEC16_R12_AUTOPSY_2026-03-31.md" \
    --context-file "$BLUEPRINT_PATH" \
    --context-file "$ROOT/version/v7/reports/SPEC16_R12_AUTOPSY_2026-03-31.md"

run_timed launcher_guard \
  python3 version/v7/scripts/training_launcher_guard_v7.py \
    --run "$RUN" \
    --preflight "$PREFLIGHT_JSON" \
    --blueprint "$BLUEPRINT_PATH" \
    --blueprint-audit "$BLUEPRINT_AUDIT_JSON" \
    --require-run-scope \
    --require-run-policy \
    --require-token-budget \
    --require-canary-metadata

run_timed init_model \
  .venv/bin/python version/v7/scripts/init_tiny_train_model_v7.py \
    --output-dir "$RUN" \
    --seed 42 \
    --init xavier_uniform \
    --template qwen3 \
    --vocab-size "$VOCAB_SIZE" \
    --layers "$LAYERS" \
    --embed-dim "$EMBED_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-heads "$NUM_HEADS" \
    --num-kv-heads "$NUM_KV_HEADS" \
    --context-len "$CONTEXT_LEN" \
    --kernel-policy fp32_reference_first

cp "$RUN/train_init_config.json" "$RUN/config.json"
cp "$RUN/weights.bump" "$RUN/weights_init.bump"

run_timed stage_tokenizer \
  python3 version/v7/scripts/stage_custom_tokenizer_to_run_v7.py \
    --run "$RUN" \
    --tokenizer-json "$TOKENIZER_JSON" \
    --tokenizer-bin "$TOKENIZER_BIN"

run_timed tokenizer_contract \
  python3 - "$RUN/tokenizer.json" <<'PY'
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))
from tokenizer_policy_v7 import BANNED_SPECIAL_TOKENS, visible_special_tokens  # type: ignore

path = Path(sys.argv[1])
doc = json.loads(path.read_text(encoding="utf-8"))
visible = visible_special_tokens(doc)
missing = [token for token in ("[bundle]", "[/bundle]") if token not in visible]
present_banned = [token for token in visible if token in BANNED_SPECIAL_TOKENS]
if missing or present_banned:
    raise SystemExit(
        json.dumps(
            {
                "missing_required_special_tokens": missing,
                "banned_visible_special_tokens": present_banned,
            },
            ensure_ascii=True,
        )
    )
print(
    json.dumps(
        {
            "visible_special_tokens": visible,
        },
        ensure_ascii=True,
    )
)
PY

if [ -f "$SEED_RUN_DIR/tokenizer.json" ]; then
  if ! python3 - "$SEED_RUN_DIR/tokenizer.json" "$RUN/tokenizer.json" <<'PY'
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))
from tokenizer_policy_v7 import normalize_tokenizer_for_warmstart  # type: ignore

seed_path = Path(sys.argv[1])
run_path = Path(sys.argv[2])
seed_doc = json.loads(seed_path.read_text(encoding="utf-8"))
run_doc = json.loads(run_path.read_text(encoding="utf-8"))
raise SystemExit(0 if normalize_tokenizer_for_warmstart(seed_doc) == normalize_tokenizer_for_warmstart(run_doc) else 1)
PY
  then
    echo "[$SPEC_TAG-seed] tokenizer mismatch between seed run and target run after normalization; refusing warm-start." >&2
    exit 1
  fi
fi
cp "$SEED_RUN_DIR/weights.bump" "$RUN/weights.bump"
cp "$SEED_RUN_DIR/weights.bump" "$RUN/weights_init.bump"
if [ -f "$SEED_RUN_DIR/weights_manifest.json" ]; then
  cp "$SEED_RUN_DIR/weights_manifest.json" "$RUN/weights_manifest.json"
fi
cat > "$RUN/seed_source.json" <<JSON
{
  "seed_from_run": "$SEED_RUN_DIR",
  "seed_weights": "$SEED_RUN_DIR/weights.bump",
  "copied_at": "$(timestamp_utc)",
  "run_mode": "$RUN_MODE"
}
JSON

if [ "$RUN_PARITY" = "1" ]; then
  set +e
  run_timed parity_regimen \
    .venv/bin/python version/v7/scripts/run_training_parity_regimen_v7.py \
      --run-dir "$RUN" \
      --no-stop-on-fail \
      --json-out "$RUN/training_parity_regimen_latest.json" \
      --md-out "$RUN/training_parity_regimen_latest.md"
  PARITY_RC=$?
  set -e
  PARITY_EVAL="$(python3 - "$RUN/training_parity_regimen_latest.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    doc = json.load(f)

stages = {
    str(s.get("id")): str(s.get("status"))
    for s in (doc.get("stages") or [])
    if isinstance(s, dict) and s.get("id")
}
required = ["A1", "A4", "D1", "D2", "E1", "F1"]
missing = [sid for sid in required if stages.get(sid) != "PASS"]
failed = [str(x) for x in ((doc.get("summary") or {}).get("failed_stage_ids") or [])]
tolerated = {"B2", "B4", "B8", "C1", "C2", "C3"}
unexpected = [sid for sid in failed if sid not in tolerated]
ok = (not missing) and (not unexpected)
print(json.dumps({
    "ok": ok,
    "required_missing": missing,
    "failed_stage_ids": failed,
    "unexpected_failed_stage_ids": unexpected,
}, separators=(",", ":")))
raise SystemExit(0 if ok else 1)
PY
  )"
  PARITY_GATE_RC=$?
  if [ "$PARITY_GATE_RC" -ne 0 ]; then
    echo "[parity] blocking failure: $PARITY_EVAL" >&2
    exit 1
  fi
  if [ "$PARITY_RC" -ne 0 ]; then
    echo "[parity] continuing after tolerated non-critical failures: $PARITY_EVAL"
  fi
fi

run_timed disk_headroom_pretrain \
  assert_disk_headroom "pretrain" "$DISK_HEADROOM_PRETRAIN_GB" "$RUN"

PRETRAIN_CMD=(
  .venv/bin/python version/v7/scripts/train_data_pipeline_v7.py
  --run "$RUN"
  --curriculum-stage stage_a
  --tokenizer ascii_bpe
  --reuse-run-tokenizer
  --pack-mode sample
  --no-pack-total-tokens-from-windows
  --strict-data-gates
  --data "$PRETRAIN_DATA"
  --layers "$LAYERS"
  --embed-dim "$EMBED_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --num-heads "$NUM_HEADS"
  --num-kv-heads "$NUM_KV_HEADS"
  --context-len "$CONTEXT_LEN"
  --optimizer adamw
  --train-driver "$TRAIN_DRIVER"
  --ck-cli-log-every "$CK_CLI_LOG_EVERY"
  --analysis-checkpoints "$ANALYSIS_CHECKPOINTS"
  --train-save-every "$TRAIN_SAVE_EVERY"
  --epochs "$PRETRAIN_EPOCHS"
  --seq-len "$CONTEXT_LEN"
  --total-tokens "$PRETRAIN_TOTAL_TOKENS"
  --grad-accum "$PRETRAIN_GRAD_ACCUM"
  --lr 3e-4
  --no-post-train-eval
  --json-out "$RUN/train_${SPEC_TAG}_stage_a.json"
)
if [ "${#TRAIN_SAVE_FINAL_FLAG[@]}" -gt 0 ]; then
  PRETRAIN_CMD+=("${TRAIN_SAVE_FINAL_FLAG[@]}")
fi
run_timed pretrain "${PRETRAIN_CMD[@]}"

run_timed promote_pretrain_checkpoint \
  python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run "$RUN" --stage pretrain
sync_training_plan_stage "$RUN/training_plan.json" "midtrain"

run_timed disk_headroom_midtrain \
  assert_disk_headroom "midtrain" "$DISK_HEADROOM_MIDTRAIN_GB" "$RUN"

MIDTRAIN_CMD=(
  .venv/bin/python version/v7/scripts/train_data_pipeline_v7.py
  --run "$RUN"
  --curriculum-stage stage_b
  --tokenizer ascii_bpe
  --reuse-run-tokenizer
  --pack-mode sample
  --no-pack-total-tokens-from-windows
  --strict-data-gates
  --data "$MIDTRAIN_DATA"
  --layers "$LAYERS"
  --embed-dim "$EMBED_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --num-heads "$NUM_HEADS"
  --num-kv-heads "$NUM_KV_HEADS"
  --context-len "$CONTEXT_LEN"
  --optimizer adamw
  --train-driver "$TRAIN_DRIVER"
  --ck-cli-log-every "$CK_CLI_LOG_EVERY"
  --analysis-checkpoints "$ANALYSIS_CHECKPOINTS"
  --train-save-every "$TRAIN_SAVE_EVERY"
  --epochs "$MIDTRAIN_EPOCHS"
  --seq-len "$CONTEXT_LEN"
  --total-tokens "$MIDTRAIN_TOTAL_TOKENS"
  --grad-accum "$MIDTRAIN_GRAD_ACCUM"
  --lr 2.5e-4
  --no-post-train-eval
  --json-out "$RUN/train_${SPEC_TAG}_stage_b.json"
)
if [ "${#TRAIN_SAVE_FINAL_FLAG[@]}" -gt 0 ]; then
  MIDTRAIN_CMD+=("${TRAIN_SAVE_FINAL_FLAG[@]}")
fi
run_timed midtrain "${MIDTRAIN_CMD[@]}"
sync_training_plan_stage "$RUN/training_plan.json" "completed"

if [ "$RUN_STAGE_EVAL" = "1" ]; then
  .venv/bin/python version/v7/scripts/eval_stage_v7.py --run "$RUN" --all-stages --n-samples 5
else
  echo "[stage-eval] skipped builtin default probes; use ${SPEC_TAG}_probe_report.json/html for bounded-intent evaluation."
fi

run_timed disk_headroom_postrun \
  assert_disk_headroom "postrun" "$DISK_HEADROOM_POSTRUN_GB" "$RUN"

build_probe_artifacts "$RUN" "$PROBE_CONTRACT" "$DATASET_PREFIX" "$CONTEXT_LEN" "$PROBE_PER_SPLIT" "$HIDDEN_PROBE_PER_SPLIT"
run_timed ir_hub_refresh refresh_ir_hub
printf '%s training finished for %s\n' "$SPEC_TAG" "$RUN"
