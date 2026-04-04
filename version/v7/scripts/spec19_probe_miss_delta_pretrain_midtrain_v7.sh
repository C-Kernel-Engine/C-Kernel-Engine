#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/ck-engine-v7/models}"

export SPEC_TAG="${SPEC_TAG:-spec19d}"
export DATASET_PREFIX="${DATASET_PREFIX:-spec19_scene_bundle}"
export BLUEPRINT_PATH="${BLUEPRINT_PATH:-$ROOT/version/v7/reports/spec19_curriculum_blueprint.json}"
export MATERIALIZE_SCRIPT="${MATERIALIZE_SCRIPT:-version/v7/scripts/dataset/materialize_spec19_probe_miss_delta_v7.py}"
export PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-version/v7/scripts/spec19_preflight_v7.py}"
export PROBE_CONTRACT_BUILDER="${PROBE_CONTRACT_BUILDER:-version/v7/scripts/build_spec19_probe_contract_v7.py}"
export COMPILER_SMOKE_SCRIPT="${COMPILER_SMOKE_SCRIPT:-version/v7/scripts/build_spec19_compiler_smoke_report_v7.py}"
export RUN_KIND="${RUN_KIND:-seeded_spec19_probe_miss_delta_continuation}"

export RUN_SCOPE_SPEC="${RUN_SCOPE_SPEC:-spec19}"
export RUN_SCOPE_RUNG="${RUN_SCOPE_RUNG:-r3a}"
export RUN_SCOPE_FAMILY="${RUN_SCOPE_FAMILY:-visual_scene_bundle}"
export RUN_SCOPE_TITLE="${RUN_SCOPE_TITLE:-Spec19 R3a Replay-Anchored Delta Continuation}"
export RUN_SCOPE_OBJECTIVE="${RUN_SCOPE_OBJECTIVE:-Test whether a replay-anchored continuation on the exact spec19 r2 miss set improves those misses without causing forgetting on untouched cases.}"
export RUN_SCOPE_HYPOTHESIS="${RUN_SCOPE_HYPOTHESIS:-A replay-anchored delta continuation from spec19 r2 will preserve most untouched behavior while selectively fixing part of the miss set; if untouched cases regress sharply, the narrow-patch concern is confirmed.}"
export RUN_SCOPE_PROMPT_CONTRACT="${RUN_SCOPE_PROMPT_CONTRACT:-Keep the frozen spec19 bounded-intent user contract and shared bundle output surface unchanged. This run changes only the continuation dataset mixture.}"
export RUN_SCOPE_OUTPUT_CONTRACT="${RUN_SCOPE_OUTPUT_CONTRACT:-Emit exactly one shared [bundle] scene_bundle.v1 only, using the frozen spec16/spec19 renderer and canonicalizer boundary.}"

export SEED_FROM_RUN="${SEED_FROM_RUN:-$MODEL_CACHE_ROOT/train/spec19_scene_bundle_l3_d192_h384_ctx768_r2}"
export FREEZE_TOKENIZER_FROM="${FREEZE_TOKENIZER_FROM:-$SEED_FROM_RUN}"
export SOURCE_RUN="${SOURCE_RUN:-$SEED_FROM_RUN}"
export REPLAY_TO_DELTA_RATIO="${REPLAY_TO_DELTA_RATIO:-3}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2}"
export MIDTRAIN_EPOCHS="${MIDTRAIN_EPOCHS:-2}"

export SEED_WORKSPACE="${SEED_WORKSPACE:-}"

export MATERIALIZE_EXTRA_ARGS="${MATERIALIZE_EXTRA_ARGS:---source-run $SOURCE_RUN --replay-to-delta-ratio $REPLAY_TO_DELTA_RATIO}"

exec bash "$ROOT/version/v7/scripts/spec17_pretrain_midtrain_v7.sh" "$@"
