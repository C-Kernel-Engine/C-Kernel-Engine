#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export SPEC_TAG="${SPEC_TAG:-spec18}"
export DATASET_PREFIX="${DATASET_PREFIX:-spec18_scene_bundle}"
export BLUEPRINT_PATH="${BLUEPRINT_PATH:-$ROOT/version/v7/reports/spec18_curriculum_blueprint.json}"
export MATERIALIZE_SCRIPT="${MATERIALIZE_SCRIPT:-version/v7/scripts/dataset/materialize_spec18_scene_bundle_v7.py}"
export PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-version/v7/scripts/spec18_preflight_v7.py}"
export PROBE_CONTRACT_BUILDER="${PROBE_CONTRACT_BUILDER:-version/v7/scripts/build_spec18_probe_contract_v7.py}"
export RUN_KIND="${RUN_KIND:-seeded_spec18_routing_first_branch}"

export RUN_SCOPE_SPEC="${RUN_SCOPE_SPEC:-spec18}"
export RUN_SCOPE_RUNG="${RUN_SCOPE_RUNG:-r1}"
export RUN_SCOPE_FAMILY="${RUN_SCOPE_FAMILY:-visual_scene_bundle}"
export RUN_SCOPE_TITLE="${RUN_SCOPE_TITLE:-Spec18 R1 Routing First Canary}"
export RUN_SCOPE_OBJECTIVE="${RUN_SCOPE_OBJECTIVE:-Validate that a routing-first bounded intent curriculum can warm-start from frozen spec16 r9, preserve one clean shared [bundle], and produce nonzero held-out exactness before style and topology widening dominate the branch.}"
export RUN_SCOPE_HYPOTHESIS="${RUN_SCOPE_HYPOTHESIS:-A small seeded canary over the spec18 routing-first curriculum should learn nonzero family and form routing on held-out prompts while keeping the frozen spec16 bundle contract stable and renderable.}"
export RUN_SCOPE_PROMPT_CONTRACT="${RUN_SCOPE_PROMPT_CONTRACT:-Bounded intent prompt in: topic, goal, audience, and limited hints only. Family and form stay internal scaffold fields during bridge rows and never become part of the external user contract.}"
export RUN_SCOPE_OUTPUT_CONTRACT="${RUN_SCOPE_OUTPUT_CONTRACT:-Emit exactly one shared [bundle] scene_bundle.v1 only, using the frozen spec16 renderer and canonicalizer boundary.}"

exec bash "$ROOT/version/v7/scripts/spec17_pretrain_midtrain_v7.sh" "$@"
