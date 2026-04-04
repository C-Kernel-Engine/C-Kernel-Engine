#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"

SPEC_TAG="${SPEC_TAG:-spec19}"
RUN="${1:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3d_sft_instruction}"
DATASET_PREFIX="${DATASET_PREFIX:-spec19_scene_bundle}"
WORKSPACE="$RUN/dataset"

MATERIALIZE_SCRIPT="${MATERIALIZE_SCRIPT:-version/v7/scripts/dataset/materialize_spec19_sft_instruction_v7.py}"
PROBE_CONTRACT_BUILDER="${PROBE_CONTRACT_BUILDER:-version/v7/scripts/build_spec19_probe_contract_v7.py}"
PROBE_AUTOPSY_SCRIPT="${PROBE_AUTOPSY_SCRIPT:-version/v7/scripts/build_bundle_probe_autopsy_v7.py}"
BLUEPRINT_PATH="${BLUEPRINT_PATH:-$ROOT/version/v7/reports/spec19_curriculum_blueprint.json}"

BASE_POLICY_RUN="${BASE_POLICY_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3d_balanced_coverage}"
FREEZE_TOKENIZER_FROM="${FREEZE_TOKENIZER_FROM:-$BASE_POLICY_RUN}"
SPEC16_R9_RUN="${SPEC16_R9_RUN:-$MODEL_CACHE_ROOT/train/spec16_scene_bundle_l3_d192_h384_ctx768_r9}"
R2_SOURCE_RUN="${R2_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r2}"
R3B_SOURCE_RUN="${R3B_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3b_coherent_replay}"
R3C_SOURCE_RUN="${R3C_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3c_cumulative_neighbors}"
R3D_SOURCE_RUN="${R3D_SOURCE_RUN:-$BASE_POLICY_RUN}"

LAYERS="${LAYERS:-3}"
EMBED_DIM="${EMBED_DIM:-192}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
CONTEXT_LEN="${CONTEXT_LEN:-768}"

SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
SFT_TOTAL_TOKENS="${SFT_TOTAL_TOKENS:-786432}"
SFT_LR="${SFT_LR:-6e-5}"

WEIGHT_QUANTUM="${WEIGHT_QUANTUM:-5}"
TRAIN_VARIANTS="${TRAIN_VARIANTS:-3}"
EVAL_VARIANTS="${EVAL_VARIANTS:-2}"
TRAIN_DRIVER="${TRAIN_DRIVER:-ck_cli}"
CK_CLI_LOG_EVERY="${CK_CLI_LOG_EVERY:-100}"
ANALYSIS_CHECKPOINTS="${ANALYSIS_CHECKPOINTS:-log}"
TRAIN_SAVE_EVERY="${TRAIN_SAVE_EVERY:-0}"
SAVE_FINAL_WEIGHTS="${SAVE_FINAL_WEIGHTS:-1}"
PROBE_PER_SPLIT="${PROBE_PER_SPLIT:-12}"
HIDDEN_PROBE_PER_SPLIT="${HIDDEN_PROBE_PER_SPLIT:-6}"
RUN_PARITY="${RUN_PARITY:-1}"
DISK_HEADROOM_SFT_GB="${DISK_HEADROOM_SFT_GB:-8}"
DISK_HEADROOM_POSTRUN_GB="${DISK_HEADROOM_POSTRUN_GB:-4}"
FORCE="${FORCE:-0}"

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

clear_stale_postrun_artifacts() {
  local run_dir="$1"
  rm -f \
    "$run_dir/${SPEC_TAG}_probe_contract.json" \
    "$run_dir/${SPEC_TAG}_probe_report.json" \
    "$run_dir/${SPEC_TAG}_probe_report.html" \
    "$run_dir/${SPEC_TAG}_probe_autopsy.json" \
    "$run_dir/${SPEC_TAG}_probe_autopsy.md" \
    "$run_dir/${SPEC_TAG}_tested_prompts_report.html" \
    "$run_dir/${SPEC_TAG}_tested_prompts_report.md" \
    "$run_dir/artifact_index.json" \
    "$run_dir/artifact_index.md" \
    "$run_dir/ir_report.html"
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
        if name == "sft":
            stage["status"] = "completed"
    elif name == active_stage:
        stage["status"] = "active"
    elif name in {"pretrain", "midtrain"}:
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

  python3 "$PROBE_AUTOPSY_SCRIPT" \
    --probe-report "$run_dir/${SPEC_TAG}_probe_report.json" \
    --json-out "$run_dir/${SPEC_TAG}_probe_autopsy.json" \
    --md-out "$run_dir/${SPEC_TAG}_probe_autopsy.md"

  python3 version/v7/scripts/build_run_artifact_index_v7.py \
    --run "$run_dir" \
    --spec-tag "$SPEC_TAG" \
    --prefix "$DATASET_PREFIX" \
    --json-out "$run_dir/artifact_index.json" \
    --md-out "$run_dir/artifact_index.md"
}

cd "$ROOT"

if [ -e "$RUN" ]; then
  if [ "$FORCE" = "1" ]; then
    rm -rf "$RUN"
  else
    echo "run dir already exists: $RUN" >&2
    echo "set FORCE=1 to replace it" >&2
    exit 1
  fi
fi

mkdir -p "$RUN"
clear_stale_postrun_artifacts "$RUN"

run_timed materialize_dataset \
  python3 "$MATERIALIZE_SCRIPT" \
    --workspace "$WORKSPACE" \
    --seed-workspace "$ROOT/version/v7/data/spec04" \
    --prefix "$DATASET_PREFIX" \
    --freeze-tokenizer-run "$FREEZE_TOKENIZER_FROM" \
    --source-run "$R2_SOURCE_RUN" \
    --source-run "$R3B_SOURCE_RUN" \
    --source-run "$R3C_SOURCE_RUN" \
    --source-run "$R3D_SOURCE_RUN" \
    --weight-quantum "$WEIGHT_QUANTUM" \
    --train-variants "$TRAIN_VARIANTS" \
    --eval-variants "$EVAL_VARIANTS" \
    --python-exec "$(command -v python3)" \
    --force

TOKENIZER_JSON="$WORKSPACE/tokenizer/tokenizer.json"
TOKENIZER_BIN="$WORKSPACE/tokenizer/tokenizer_bin"
SFT_DATA="$WORKSPACE/sft/train/${DATASET_PREFIX}_sft_train.txt"
PROBE_CONTRACT="$RUN/${SPEC_TAG}_probe_contract.json"
VOCAB_SIZE="$(jq -r '.vocab_size // 0' "$TOKENIZER_BIN/tokenizer_meta.json")"

if [ ! -f "$BASE_POLICY_RUN/weights.bump" ] || [ ! -f "$BASE_POLICY_RUN/weights_manifest.json" ]; then
  echo "base policy run missing weights artifacts: $BASE_POLICY_RUN" >&2
  exit 1
fi

cp "$BASE_POLICY_RUN/weights.bump" "$RUN/weights.bump"
cp "$BASE_POLICY_RUN/weights.bump" "$RUN/weights_init.bump"
cp "$BASE_POLICY_RUN/weights_manifest.json" "$RUN/weights_manifest.json"
cp "$BASE_POLICY_RUN/config.json" "$RUN/config.json"
cp "$BASE_POLICY_RUN/train_init_config.json" "$RUN/train_init_config.json"
if [ -f "$BASE_POLICY_RUN/template_train.json" ]; then
  cp "$BASE_POLICY_RUN/template_train.json" "$RUN/template_train.json"
fi

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
print(json.dumps({"visible_special_tokens": visible}, ensure_ascii=True))
PY

cat > "$RUN/training_plan.json" <<JSON
{
  "schema": "ck.training_plan.v1",
  "created_by": "spec19_sft_instruction_on_r3d_v7.sh",
  "active_stage": "sft",
  "run_policy": {
    "mode": "alignment",
    "kind": "spec19_sft_on_frozen_r3d",
    "base_policy_run": "$BASE_POLICY_RUN",
    "frozen_seed_run": "$BASE_POLICY_RUN",
    "tokenizer_source_run": "$FREEZE_TOKENIZER_FROM"
  },
  "token_budget": {
    "recommended_sft_total_tokens": $SFT_TOTAL_TOKENS,
    "selected_sft_total_tokens": $SFT_TOTAL_TOKENS
  },
  "stage_order": ["pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"],
  "tokenizer": {
    "type": "ascii_bpe",
    "vocab_size": $VOCAB_SIZE,
    "tokenizer_corpora": [
      {
        "name": "$(basename "$WORKSPACE/tokenizer/${DATASET_PREFIX}_tokenizer_corpus.txt")",
        "path": "$WORKSPACE/tokenizer/${DATASET_PREFIX}_tokenizer_corpus.txt"
      }
    ]
  },
  "stages": [
    {
      "stage": "pretrain",
      "seq": 1,
      "status": "completed",
      "enabled": false,
      "datasets": [],
      "runs": [{"source_run": "$BASE_POLICY_RUN", "note": "frozen base policy source"}]
    },
    {
      "stage": "midtrain",
      "seq": 2,
      "status": "completed",
      "enabled": false,
      "datasets": [],
      "runs": [{"source_run": "$BASE_POLICY_RUN", "note": "frozen base policy source"}]
    },
    {
      "stage": "sft",
      "seq": 3,
      "status": "active",
      "enabled": true,
      "datasets": [
        {
          "name": "$(basename "$SFT_DATA")",
          "path": "$SFT_DATA",
          "kind": "instruction_dataset"
        }
      ],
      "runs": []
    },
    {"stage": "dpo", "seq": 4, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "grpo", "seq": 5, "status": "planned", "enabled": false, "datasets": [], "runs": []},
    {"stage": "ppo", "seq": 6, "status": "planned", "enabled": false, "datasets": [], "runs": []}
  ]
}
JSON

run_timed run_scope_init \
  python3 version/v7/scripts/init_run_scope_v7.py \
    --run "$RUN" \
    --spec "spec19" \
    --rung "r3d_sft" \
    --family "visual_scene_bundle" \
    --title "Spec19 R3d SFT Instruction Follow-Up" \
    --objective "Start from the frozen r3d winner and run a prompt-fidelity SFT phase on top of the unified spec19 curriculum without changing the canonical [bundle] output contract." \
    --hypothesis "A dedicated instruction-style SFT pass on top of r3d should improve bounded-intent completion fidelity more safely than another pretrain or midtrain curriculum rewrite." \
    --prompt-contract "Keep the frozen spec19 bounded-intent user prompt contract unchanged. Instruction-style rows may wrap or restate that same request, but the external probe contract stays the same." \
    --output-contract "Emit exactly one shared [bundle] scene_bundle.v1 only, using the frozen spec16/spec19 renderer and canonicalizer boundary." \
    --success-gate "Held-out exactness improves over the frozen r3d base without new visible test regressions." \
    --success-gate "The model keeps one-bundle stop behavior and does not drift into prose or duplicated bundles." \
    --guardrail "Do not widen the output surface beyond exactly one shared [bundle]." \
    --guardrail "Do not change tokenizer, compiler boundary, or probe contract in the same rung." \
    --research-prior "HumanEval/Codex (Chen et al., 2021): measure correctness by compilable structure rather than loss alone." \
    --research-prior "phi-1 (Gunasekar et al., 2023): dense clean instructional data can sharpen structured behavior more effectively than noisy scale." \
    --research-prior "CodeT5 (Wang et al., 2021): identifier-sensitive structured output benefits from more explicit supervised prompt-following pressure." \
    --lesson-learned "Spec19 r3d is the best completed base policy on this bounded-intent line." \
    --lesson-learned "Fresh unified retraining from spec16 r9 regressed, so the later winner prior still matters." \
    --lesson-learned "Continuation deltas alone are unstable; instruction-focused SFT is the next cleaner lever." \
    --read-first "$BLUEPRINT_PATH" \
    --context-file "$BLUEPRINT_PATH"

run_timed launcher_guard \
  python3 version/v7/scripts/training_launcher_guard_v7.py \
    --run "$RUN" \
    --require-run-scope \
    --require-run-policy \
    --require-token-budget

cat > "$RUN/seed_source.json" <<JSON
{
  "seed_from_run": "$BASE_POLICY_RUN",
  "seed_weights": "$BASE_POLICY_RUN/weights.bump",
  "tokenizer_from_run": "$FREEZE_TOKENIZER_FROM",
  "copied_at": "$(timestamp_utc)",
  "run_mode": "sft"
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

run_timed disk_headroom_sft \
  assert_disk_headroom "sft" "$DISK_HEADROOM_SFT_GB" "$RUN"

SFT_CMD=(
  .venv/bin/python version/v7/scripts/train_data_pipeline_v7.py
  --run "$RUN"
  --curriculum-stage sft
  --tokenizer ascii_bpe
  --reuse-run-tokenizer
  --pack-mode sample
  --no-pack-total-tokens-from-windows
  --strict-data-gates
  --data "$SFT_DATA"
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
  --epochs "$SFT_EPOCHS"
  --seq-len "$CONTEXT_LEN"
  --total-tokens "$SFT_TOTAL_TOKENS"
  --grad-accum "$SFT_GRAD_ACCUM"
  --lr "$SFT_LR"
  --no-post-train-eval
  --json-out "$RUN/train_${SPEC_TAG}_sft.json"
)
if [ "$SAVE_FINAL_WEIGHTS" != "1" ]; then
  SFT_CMD+=(--no-train-save-final)
fi
run_timed sft "${SFT_CMD[@]}"

run_timed promote_sft_checkpoint \
  python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run "$RUN" --stage sft
sync_training_plan_stage "$RUN/training_plan.json" "completed"

run_timed disk_headroom_postrun \
  assert_disk_headroom "postrun" "$DISK_HEADROOM_POSTRUN_GB" "$RUN"

build_probe_artifacts "$RUN" "$PROBE_CONTRACT" "$DATASET_PREFIX" "$CONTEXT_LEN" "$PROBE_PER_SPLIT" "$HIDDEN_PROBE_PER_SPLIT"
run_timed ir_hub_refresh refresh_ir_hub
printf 'spec19 sft finished for %s\n' "$RUN"
