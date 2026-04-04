#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"
RUN="${1:-$MODEL_CACHE_ROOT/train/spec16_scene_bundle_l3_d192_h384_ctx768_r12}"
DATASET_PREFIX="${DATASET_PREFIX:-spec16_scene_bundle}"
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
    printf '[spec16-env] oneapi_runtime_lib=%s\n' "$found"
  fi
}

clear_stale_postrun_artifacts() {
  local run_dir="$1"
  rm -f \
    "$run_dir/spec16_probe_contract.json" \
    "$run_dir/spec16_probe_report.json" \
    "$run_dir/spec16_probe_report.html" \
    "$run_dir/spec16_tested_prompts_report.html" \
    "$run_dir/spec16_tested_prompts_report.md" \
    "$run_dir/ir_report.html"
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
  printf '[spec16-timing] stage=%s event=start started_at=%s\n' "$label" "$(timestamp_utc)"
  "$@"
  rc=$?
  ended_epoch="$(date +%s)"
  printf '[spec16-timing] stage=%s event=end ended_at=%s elapsed_sec=%s\n' \
    "$label" "$(timestamp_utc)" "$((ended_epoch - started_epoch))"
  return "$rc"
}

build_probe_artifacts() {
  local run_dir="$1"
  local probe_contract="$2"
  local dataset_prefix="$3"
  local context_len="$4"
  local probe_per_split="$5"
  local hidden_probe_per_split="$6"

  python3 version/v7/scripts/build_spec16_probe_contract_v7.py \
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
    --output "$run_dir/spec16_probe_report.html" \
    --json-out "$run_dir/spec16_probe_report.json"

  python3 version/v7/scripts/build_structured_scene_tested_prompts_doc_v7.py \
    --probe-report "$run_dir/spec16_probe_report.json" \
    --output-html "$run_dir/spec16_tested_prompts_report.html" \
    --output-md "$run_dir/spec16_tested_prompts_report.md"

  python3 version/v7/tools/open_ir_visualizer.py \
    --generate \
    --run "$run_dir" \
    --html-only \
    --strict-run-artifacts \
    --output "$run_dir/ir_report.html"
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

LAYERS="${LAYERS:-3}"
EMBED_DIM="${EMBED_DIM:-192}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
CONTEXT_LEN="${CONTEXT_LEN:-768}"

PRETRAIN_TOTAL_TOKENS="${PRETRAIN_TOTAL_TOKENS:-}"
MIDTRAIN_TOTAL_TOKENS="${MIDTRAIN_TOTAL_TOKENS:-}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-3}"
MIDTRAIN_EPOCHS="${MIDTRAIN_EPOCHS:-3}"
PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-2}"
MIDTRAIN_GRAD_ACCUM="${MIDTRAIN_GRAD_ACCUM:-2}"
PRETRAIN_CANONICAL_REPEAT="${PRETRAIN_CANONICAL_REPEAT:-2}"
PRETRAIN_BRIDGE_REPEAT="${PRETRAIN_BRIDGE_REPEAT:-1}"
MIDTRAIN_CANONICAL_REPEAT="${MIDTRAIN_CANONICAL_REPEAT:-1}"
MIDTRAIN_BRIDGE_REPEAT="${MIDTRAIN_BRIDGE_REPEAT:-1}"
RUN_MODE="${RUN_MODE:-pilot}"
ALLOW_FULL_RUNG="${ALLOW_FULL_RUNG:-0}"
ALLOW_DECISION_OVERRIDE="${ALLOW_DECISION_OVERRIDE:-0}"
PILOT_TOKEN_NUMERATOR="${PILOT_TOKEN_NUMERATOR:-1}"
PILOT_TOKEN_DENOMINATOR="${PILOT_TOKEN_DENOMINATOR:-3}"
FROZEN_BASELINE_RUN="${FROZEN_BASELINE_RUN:-$MODEL_CACHE_ROOT/train/spec16_scene_bundle_l3_d192_h384_ctx768_r9}"
DECISION_ARTIFACT="${DECISION_ARTIFACT:-$ROOT/version/v7/reports/spec16_training_decision.json}"
RUN_SCOPE_SPEC="${RUN_SCOPE_SPEC:-spec16}"
RUN_SCOPE_RUNG="${RUN_SCOPE_RUNG:-r12}"
RUN_SCOPE_FAMILY="${RUN_SCOPE_FAMILY:-visual_scene_bundle}"
RUN_SCOPE_TITLE="${RUN_SCOPE_TITLE:-Spec16 R12 System-Diagram Pilot Gate}"
RUN_SCOPE_OBJECTIVE="${RUN_SCOPE_OBJECTIVE:-Run a constrained pilot against frozen r9 to test whether targeted system_diagram cross-form repair rows can improve system_diagram exactness without regressing the other families or hidden holdouts.}"
RUN_SCOPE_HYPOTHESIS="${RUN_SCOPE_HYPOTHESIS:-A small pilot with targeted system_diagram cross-form/style frontier rows should raise system_diagram exactness while preserving the frozen r9 baseline on memory_map, timeline, and hidden splits.}"
RUN_SCOPE_PROMPT_CONTRACT="${RUN_SCOPE_PROMPT_CONTRACT:-Family-generic design/control prompts only; no topic-bearing payload text.}"
RUN_SCOPE_OUTPUT_CONTRACT="${RUN_SCOPE_OUTPUT_CONTRACT:-Compact [bundle] generalized visual DSL only, lowered through deterministic family compilers.}"
RUN_SCOPE_RESEARCH_PRIORS=(
  "Chinchilla (Hoffmann et al., 2022): do not undertrain on tokens; keep token budget proportional to model and data regime."
  "phi-1 (Gunasekar et al., 2023): prefer clean, dense, synthetic or curated data over noisy scale."
  "Deduplication of Training Data Makes Language Models Better (Lee et al., 2022): remove shortcut repetition, but replace it with new useful coverage rather than shrinking the curriculum."
  "Evaluating Large Language Models Trained on Code / HumanEval (Chen et al., 2021): score correctness by executable or compilable structure, not by loss alone."
  "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets (Power et al., 2022): low loss is not proof of generalization."
)
RUN_SCOPE_LESSONS=(
  "Spec16 r7 lesson: dedupe and cleanliness were directionally right, but the curriculum was over-shrunk. We removed repetition without replacing it with enough fresh, targeted, compiler-valid examples, so exactness collapsed while loss improved."
  "Operator rule: if a rung cuts duplicate mass, replace it with new targeted synthetic coverage before accepting a lower total token budget."
  "Spec16 r8 lesson: contrast-set recovery fixed the memory_map collapse, but exactness still trailed the best rung because style drift and clean-stop failures remained."
  "Spec16 r9 lesson: the line now solves the shared bundle contract at current model size, but the residual failures are concentrated in prompt-schema leakage on tag_canonical and a small unseen memory style frontier."
  "Spec16 r10 lesson: explicit schema-lock and prompt-echo repair surfaces taught the wrong literal control tokens and regressed system_diagram robustness. Prefer control-agnostic clean-stop repair prompts over prompts that spell out the exact junk to avoid."
  "Spec16 r11 lesson: rolling back r10 restored renderability and hidden balance, but system_diagram exactness still lagged frozen r9. Future raw changes must clear a pilot gate against r9 before any broader rung."
)

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

TRAIN_SAVE_FINAL_FLAG=()
if [ "${TRAIN_SAVE_FINAL}" != "1" ]; then
  TRAIN_SAVE_FINAL_FLAG+=(--no-train-save-final)
fi

TOKENIZER_CORPUS="$WORKSPACE/tokenizer/${DATASET_PREFIX}_tokenizer_corpus.txt"
PRETRAIN_DATA="$WORKSPACE/pretrain/train/${DATASET_PREFIX}_pretrain_train.txt"
MIDTRAIN_DATA="$WORKSPACE/midtrain/train/${DATASET_PREFIX}_midtrain_train.txt"
TOKENIZER_JSON="$WORKSPACE/tokenizer/tokenizer.json"
TOKENIZER_BIN="$WORKSPACE/tokenizer/tokenizer_bin"
PROBE_CONTRACT="$RUN/spec16_probe_contract.json"
PREFLIGHT_JSON="$RUN/spec16_preflight.json"

cd "$ROOT"
ensure_oneapi_runtime_libs
mkdir -p "$RUN"
clear_stale_postrun_artifacts "$RUN"

if [ "$RUN_MODE" != "pilot" ] && [ "$RUN_MODE" != "full" ]; then
  echo "[spec16-policy] unsupported RUN_MODE=$RUN_MODE; use pilot or full." >&2
  exit 1
fi
if [ "$RUN_MODE" = "full" ] && [ "$ALLOW_FULL_RUNG" != "1" ]; then
  echo "[spec16-policy] full rungs are blocked by default. Set ALLOW_FULL_RUNG=1 only after a pilot clears the frozen-baseline gate." >&2
  exit 1
fi
if [ ! -f "$FROZEN_BASELINE_RUN/spec16_probe_report.json" ]; then
  echo "[spec16-policy] frozen baseline probe report not found at $FROZEN_BASELINE_RUN/spec16_probe_report.json" >&2
  exit 1
fi
if [ ! -f "$DECISION_ARTIFACT" ]; then
  echo "[spec16-policy] decision artifact missing at $DECISION_ARTIFACT. Generate it with spec16_training_decision_v7.py before launching raw training." >&2
  exit 1
fi
if [ "$ALLOW_DECISION_OVERRIDE" != "1" ]; then
  python3 - "$DECISION_ARTIFACT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
doc = json.loads(path.read_text(encoding="utf-8"))
if bool(doc.get("training_allowed")):
    raise SystemExit(0)
print(
    "[spec16-policy] raw training blocked by decision artifact. "
    f"default_action={doc.get('default_action')} reasons={doc.get('reasons')}",
    file=sys.stderr,
)
raise SystemExit(1)
PY
fi

run_timed materialize_workspace \
  python3 version/v7/scripts/dataset/materialize_spec16_scene_bundle_v7.py \
    --workspace "$WORKSPACE" \
    --prefix "$DATASET_PREFIX" \
    --pretrain-canonical-repeat "$PRETRAIN_CANONICAL_REPEAT" \
    --pretrain-bridge-repeat "$PRETRAIN_BRIDGE_REPEAT" \
    --midtrain-canonical-repeat "$MIDTRAIN_CANONICAL_REPEAT" \
    --midtrain-bridge-repeat "$MIDTRAIN_BRIDGE_REPEAT" \
    --force

if [ "$RUN_PREFLIGHT" = "1" ]; then
  PREFLIGHT_ARGS=(
    python3
    version/v7/scripts/spec16_preflight_v7.py
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
  if [ -z "$PRETRAIN_TOTAL_TOKENS" ]; then
    PRETRAIN_TOTAL_TOKENS="$(jq -r '.stages.pretrain.recommended_total_tokens // 0' "$PREFLIGHT_JSON")"
  fi
  if [ -z "$MIDTRAIN_TOTAL_TOKENS" ]; then
    MIDTRAIN_TOTAL_TOKENS="$(jq -r '.stages.midtrain.recommended_total_tokens // 0' "$PREFLIGHT_JSON")"
  fi
fi

if [ -z "$PRETRAIN_TOTAL_TOKENS" ] || [ "$PRETRAIN_TOTAL_TOKENS" = "0" ] || [ "$PRETRAIN_TOTAL_TOKENS" = "null" ]; then
  PRETRAIN_TOTAL_TOKENS="131072"
fi
if [ -z "$MIDTRAIN_TOTAL_TOKENS" ] || [ "$MIDTRAIN_TOTAL_TOKENS" = "0" ] || [ "$MIDTRAIN_TOTAL_TOKENS" = "null" ]; then
  MIDTRAIN_TOTAL_TOKENS="131072"
fi

RECOMMENDED_PRETRAIN_TOTAL_TOKENS="$PRETRAIN_TOTAL_TOKENS"
RECOMMENDED_MIDTRAIN_TOTAL_TOKENS="$MIDTRAIN_TOTAL_TOKENS"
if [ "$RUN_MODE" = "pilot" ]; then
  PRETRAIN_TOTAL_TOKENS="$(( PRETRAIN_TOTAL_TOKENS * PILOT_TOKEN_NUMERATOR / PILOT_TOKEN_DENOMINATOR ))"
  MIDTRAIN_TOTAL_TOKENS="$(( MIDTRAIN_TOTAL_TOKENS * PILOT_TOKEN_NUMERATOR / PILOT_TOKEN_DENOMINATOR ))"
  if [ "$PRETRAIN_TOTAL_TOKENS" -lt "$CONTEXT_LEN" ]; then
    PRETRAIN_TOTAL_TOKENS="$CONTEXT_LEN"
  fi
  if [ "$MIDTRAIN_TOTAL_TOKENS" -lt "$CONTEXT_LEN" ]; then
    MIDTRAIN_TOTAL_TOKENS="$CONTEXT_LEN"
  fi
fi

printf '[spec16-budget] pretrain_total_tokens=%s midtrain_total_tokens=%s\n' "$PRETRAIN_TOTAL_TOKENS" "$MIDTRAIN_TOTAL_TOKENS"
printf '[spec16-optimizer] optimizer=adamw pretrain_grad_accum=%s midtrain_grad_accum=%s\n' "$PRETRAIN_GRAD_ACCUM" "$MIDTRAIN_GRAD_ACCUM"
printf '[spec16-policy] dataset_prefix=%s family=visual_scene_bundle run_mode=%s baseline=%s pilot_fraction=%s/%s\n' \
  "$DATASET_PREFIX" "$RUN_MODE" "$FROZEN_BASELINE_RUN" "$PILOT_TOKEN_NUMERATOR" "$PILOT_TOKEN_DENOMINATOR"

VOCAB_SIZE="$(jq -r '.vocab_size // 0' "$TOKENIZER_BIN/tokenizer_meta.json")"
if [ -z "$VOCAB_SIZE" ] || [ "$VOCAB_SIZE" = "0" ] || [ "$VOCAB_SIZE" = "null" ]; then
  echo "failed to read tokenizer vocab size from $TOKENIZER_BIN/tokenizer_meta.json" >&2
  exit 1
fi

cat > "$RUN/training_plan.json" <<JSON
{
  "schema": "ck.training_plan.v1",
  "created_by": "spec16_pretrain_midtrain_v7.sh",
  "active_stage": "pretrain",
  "run_policy": {
    "mode": "$RUN_MODE",
    "full_rung_allowed": $([ "$ALLOW_FULL_RUNG" = "1" ] && printf 'true' || printf 'false'),
    "decision_override": $([ "$ALLOW_DECISION_OVERRIDE" = "1" ] && printf 'true' || printf 'false'),
    "decision_artifact": "$DECISION_ARTIFACT",
    "frozen_baseline_run": "$FROZEN_BASELINE_RUN",
    "pilot_token_fraction": {
      "numerator": $PILOT_TOKEN_NUMERATOR,
      "denominator": $PILOT_TOKEN_DENOMINATOR
    },
    "promotion_gate": {
      "hidden_non_regression": true,
      "family_non_regression": true,
      "system_diagram_must_improve": true
    }
  },
  "token_budget": {
    "recommended_pretrain_total_tokens": $RECOMMENDED_PRETRAIN_TOTAL_TOKENS,
    "recommended_midtrain_total_tokens": $RECOMMENDED_MIDTRAIN_TOTAL_TOKENS,
    "selected_pretrain_total_tokens": $PRETRAIN_TOTAL_TOKENS,
    "selected_midtrain_total_tokens": $MIDTRAIN_TOTAL_TOKENS
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
      "datasets": [{ "name": "$(basename "$PRETRAIN_DATA")", "path": "$PRETRAIN_DATA", "kind": "generated_dataset" }],
      "runs": []
    },
    {
      "stage": "midtrain",
      "seq": 2,
      "status": "planned",
      "enabled": true,
      "datasets": [{ "name": "$(basename "$MIDTRAIN_DATA")", "path": "$MIDTRAIN_DATA", "kind": "generated_dataset" }],
      "runs": []
    }
  ]
}
JSON

RUN_SCOPE_ARGS=(
  python3
  version/v7/scripts/init_run_scope_v7.py
  --run "$RUN"
  --family "$RUN_SCOPE_FAMILY"
  --spec "$RUN_SCOPE_SPEC"
  --rung "$RUN_SCOPE_RUNG"
  --title "$RUN_SCOPE_TITLE"
  --objective "$RUN_SCOPE_OBJECTIVE"
  --hypothesis "$RUN_SCOPE_HYPOTHESIS"
  --prompt-contract "$RUN_SCOPE_PROMPT_CONTRACT"
  --output-contract "$RUN_SCOPE_OUTPUT_CONTRACT"
)
for _item in "${RUN_SCOPE_RESEARCH_PRIORS[@]}"; do
  if [ -n "$_item" ]; then
    RUN_SCOPE_ARGS+=(--research-prior "$_item")
  fi
done
for _item in "${RUN_SCOPE_LESSONS[@]}"; do
  if [ -n "$_item" ]; then
    RUN_SCOPE_ARGS+=(--lesson-learned "$_item")
  fi
done

run_timed init_run_scope "${RUN_SCOPE_ARGS[@]}"

LAUNCHER_GUARD_ARGS=(
  python3
  version/v7/scripts/training_launcher_guard_v7.py
  --run "$RUN"
  --preflight "$PREFLIGHT_JSON"
  --decision-artifact "$DECISION_ARTIFACT"
  --require-run-scope
  --require-run-policy
  --require-token-budget
  --require-canary-metadata
)
if [ "$ALLOW_DECISION_OVERRIDE" = "1" ]; then
  LAUNCHER_GUARD_ARGS+=(--allow-decision-override)
fi
run_timed launcher_guard "${LAUNCHER_GUARD_ARGS[@]}"

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
  --json-out "$RUN/train_spec16_stage_a.json"
)
if [ "${#TRAIN_SAVE_FINAL_FLAG[@]}" -gt 0 ]; then
  PRETRAIN_CMD+=("${TRAIN_SAVE_FINAL_FLAG[@]}")
fi
run_timed pretrain "${PRETRAIN_CMD[@]}"

run_timed promote_pretrain_checkpoint \
  python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run "$RUN" --stage pretrain
sync_training_plan_stage "$RUN/training_plan.json" "midtrain"

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
  --json-out "$RUN/train_spec16_stage_b.json"
)
if [ "${#TRAIN_SAVE_FINAL_FLAG[@]}" -gt 0 ]; then
  MIDTRAIN_CMD+=("${TRAIN_SAVE_FINAL_FLAG[@]}")
fi
run_timed midtrain "${MIDTRAIN_CMD[@]}"
sync_training_plan_stage "$RUN/training_plan.json" "completed"

if [ "$RUN_STAGE_EVAL" = "1" ]; then
  .venv/bin/python version/v7/scripts/eval_stage_v7.py --run "$RUN" --all-stages --n-samples 5
else
  echo "[stage-eval] skipped builtin default probes; use spec16_probe_report.json/html for generalized bundle evaluation."
fi

build_probe_artifacts "$RUN" "$PROBE_CONTRACT" "$DATASET_PREFIX" "$CONTEXT_LEN" "$PROBE_PER_SPLIT" "$HIDDEN_PROBE_PER_SPLIT"
if [ "$RUN_MODE" = "pilot" ]; then
  run_timed pilot_gate \
    python3 version/v7/scripts/spec16_pilot_gate_v7.py \
      --current "$RUN/spec16_probe_report.json" \
      --baseline "$FROZEN_BASELINE_RUN/spec16_probe_report.json" \
      --json-out "$RUN/spec16_pilot_gate.json" \
      --md-out "$RUN/spec16_pilot_gate.md"
else
  echo "[spec16-policy] full rung completed; pilot gate skipped."
fi
printf 'spec16 training finished for %s\n' "$RUN"
