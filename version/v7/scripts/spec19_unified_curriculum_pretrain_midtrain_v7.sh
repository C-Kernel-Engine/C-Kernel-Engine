#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"

export SPEC_TAG="${SPEC_TAG:-spec19r4}"
export DATASET_PREFIX="${DATASET_PREFIX:-spec19_scene_bundle}"
export BLUEPRINT_PATH="${BLUEPRINT_PATH:-$ROOT/version/v7/reports/spec19_curriculum_blueprint.json}"
export MATERIALIZE_SCRIPT="${MATERIALIZE_SCRIPT:-version/v7/scripts/dataset/materialize_spec19_unified_curriculum_v7.py}"
export PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-version/v7/scripts/spec19_preflight_v7.py}"
export PROBE_CONTRACT_BUILDER="${PROBE_CONTRACT_BUILDER:-version/v7/scripts/build_spec19_probe_contract_v7.py}"
export COMPILER_SMOKE_SCRIPT="${COMPILER_SMOKE_SCRIPT:-version/v7/scripts/build_spec19_compiler_smoke_report_v7.py}"
export RUN_KIND="${RUN_KIND:-seeded_spec19_unified_curriculum_fresh_retrain}"

export RUN_SCOPE_SPEC="${RUN_SCOPE_SPEC:-spec19}"
export RUN_SCOPE_RUNG="${RUN_SCOPE_RUNG:-r4}"
export RUN_SCOPE_FAMILY="${RUN_SCOPE_FAMILY:-visual_scene_bundle}"
export RUN_SCOPE_TITLE="${RUN_SCOPE_TITLE:-Spec19 R4 Unified Curriculum Fresh Retrain}"
export RUN_SCOPE_OBJECTIVE="${RUN_SCOPE_OBJECTIVE:-Retrain spec19 from the frozen spec16 r9 base on one unified curriculum: full cumulative winner-line replay plus the balanced generalized recovery delta, deduped into a single coherent corpus.}"
export RUN_SCOPE_HYPOTHESIS="${RUN_SCOPE_HYPOTHESIS:-A fresh retrain from the frozen base seed on the larger unified spec19 curriculum should preserve the visible routing gains while reducing hidden and syntax regressions caused by late continuation drift.}"
export RUN_SCOPE_PROMPT_CONTRACT="${RUN_SCOPE_PROMPT_CONTRACT:-Keep the frozen spec19 bounded-intent user contract and shared bundle output surface unchanged. Change only the training recipe from winner-continuation to fresh retraining on a unified cumulative curriculum.}"
export RUN_SCOPE_OUTPUT_CONTRACT="${RUN_SCOPE_OUTPUT_CONTRACT:-Emit exactly one shared [bundle] scene_bundle.v1 only, using the frozen spec16/spec19 renderer and canonicalizer boundary.}"

export SPEC16_R9_RUN="${SPEC16_R9_RUN:-$MODEL_CACHE_ROOT/train/spec16_scene_bundle_l3_d192_h384_ctx768_r9}"
export R2_SOURCE_RUN="${R2_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r2}"
export R3B_SOURCE_RUN="${R3B_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3b_coherent_replay}"
export R3C_SOURCE_RUN="${R3C_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3c_cumulative_neighbors}"
export R3D_SOURCE_RUN="${R3D_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3d_balanced_coverage}"
export SEED_FROM_RUN="${SEED_FROM_RUN:-$SPEC16_R9_RUN}"
export FREEZE_TOKENIZER_FROM="${FREEZE_TOKENIZER_FROM:-$SPEC16_R9_RUN}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2}"
export MIDTRAIN_EPOCHS="${MIDTRAIN_EPOCHS:-2}"

export MATERIALIZE_EXTRA_ARGS="${MATERIALIZE_EXTRA_ARGS:---source-run $R2_SOURCE_RUN --source-run $R3B_SOURCE_RUN --source-run $R3C_SOURCE_RUN --source-run $R3D_SOURCE_RUN}"

exec bash "$ROOT/version/v7/scripts/spec17_pretrain_midtrain_v7.sh" "$@"
