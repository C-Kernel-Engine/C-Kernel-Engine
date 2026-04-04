#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"

export SPEC_TAG="${SPEC_TAG:-spec19c}"
export DATASET_PREFIX="${DATASET_PREFIX:-spec19_scene_bundle}"
export BLUEPRINT_PATH="${BLUEPRINT_PATH:-$ROOT/version/v7/reports/spec19_curriculum_blueprint.json}"
export MATERIALIZE_SCRIPT="${MATERIALIZE_SCRIPT:-version/v7/scripts/dataset/materialize_spec19_coherent_replay_union_v7.py}"
export PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-version/v7/scripts/spec19_preflight_v7.py}"
export PROBE_CONTRACT_BUILDER="${PROBE_CONTRACT_BUILDER:-version/v7/scripts/build_spec19_probe_contract_v7.py}"
export COMPILER_SMOKE_SCRIPT="${COMPILER_SMOKE_SCRIPT:-version/v7/scripts/build_spec19_compiler_smoke_report_v7.py}"
export RUN_KIND="${RUN_KIND:-seeded_spec19_coherent_replay_union_continuation}"

export RUN_SCOPE_SPEC="${RUN_SCOPE_SPEC:-spec19}"
export RUN_SCOPE_RUNG="${RUN_SCOPE_RUNG:-r3b}"
export RUN_SCOPE_FAMILY="${RUN_SCOPE_FAMILY:-visual_scene_bundle}"
export RUN_SCOPE_TITLE="${RUN_SCOPE_TITLE:-Spec19 R3b Coherent Replay Union Continuation}"
export RUN_SCOPE_OBJECTIVE="${RUN_SCOPE_OBJECTIVE:-Train on one deduped coherent curriculum built from prior spec19 train corpora, with strong replay breadth and clean eval pools that exclude train-eval prompt collisions.}"
export RUN_SCOPE_HYPOTHESIS="${RUN_SCOPE_HYPOTHESIS:-A replay-heavy union of prior spec19 train corpora should retain r2 behavior, absorb the useful r3a lessons, and avoid the forgetting pattern caused by narrow miss-only patching.}"
export RUN_SCOPE_PROMPT_CONTRACT="${RUN_SCOPE_PROMPT_CONTRACT:-Keep the frozen spec19 bounded-intent user contract and shared bundle output surface unchanged. Change only the stage train corpora by unioning prior spec19 train rows and deduping them into one coherent replay curriculum.}"
export RUN_SCOPE_OUTPUT_CONTRACT="${RUN_SCOPE_OUTPUT_CONTRACT:-Emit exactly one shared [bundle] scene_bundle.v1 only, using the frozen spec16/spec19 renderer and canonicalizer boundary.}"

export R1_SOURCE_RUN="${R1_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r1}"
export R2_SOURCE_RUN="${R2_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r2}"
export R3A_SOURCE_RUN="${R3A_SOURCE_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r3a_delta_replay}"

export SEED_FROM_RUN="${SEED_FROM_RUN:-$R2_SOURCE_RUN}"
export FREEZE_TOKENIZER_FROM="${FREEZE_TOKENIZER_FROM:-$SEED_FROM_RUN}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2}"
export MIDTRAIN_EPOCHS="${MIDTRAIN_EPOCHS:-2}"

export MATERIALIZE_EXTRA_ARGS="${MATERIALIZE_EXTRA_ARGS:---source-run $R1_SOURCE_RUN --source-run $R2_SOURCE_RUN --source-run $R3A_SOURCE_RUN}"

exec bash "$ROOT/version/v7/scripts/spec17_pretrain_midtrain_v7.sh" "$@"
