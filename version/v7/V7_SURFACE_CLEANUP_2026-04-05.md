# v7 Surface Cleanup

## Goal

Keep `version/v7` focused on the active text stack:

- text runtime
- text IR lowering
- text backprop/runtime codegen
- A-F PyTorch parity gates
- family transfer from the known-good `qwen3` lane

Do **not** let the older SVG/spec/gen1 experiment surface define the active `v7` shape.

## Current Boundary

### Core `v7` text path

These are the scripts that matter for the current `v7` text work:

- `version/v7/scripts/ck_run_v7.py`
- `version/v7/scripts/build_ir_train_v7.py`
- `version/v7/scripts/lower_ir2_backward_v7.py`
- `version/v7/scripts/codegen_train_runtime_v7.py`
- `version/v7/scripts/train_parity_epochs_v7.py`
- `version/v7/scripts/run_training_parity_regimen_v7.py`
- `version/v7/scripts/check_backprop_stitch_runtime_v7.py`
- `version/v7/scripts/check_replay_determinism_v7.py`
- `version/v7/scripts/check_runtime_replay_accum_v7.py`
- `version/v7/scripts/train_data_pipeline_v7.py`
- `version/v7/scripts/init_tiny_train_model_v7.py`
- `version/v7/templates/*.json`
- `version/v7/kernel_maps/*`

### Experimental / non-core `v7` surface

These are still relevant as archived research history, but they are not the active `v7` text-backprop surface:

- `version/v7/scripts/spec*_*.py`
- `version/v7/scripts/spec*_*.sh`
- `version/v7/scripts/generate_svg_*`
- `version/v7/scripts/render_svg_structured_scene_spec*`
- `version/v7/scripts/render_svg_structured_scene_gen1*`
- `version/v7/scripts/bootstrap_gen1_*`
- `version/v7/scripts/build_gen1_*`
- `version/v7/scripts/materialize_gen1_*`
- `version/v7/scripts/gen1_*scene_dsl*`
- most SVG/spec/gen1 smoke/probe/report builders

Those files are not being deleted in this pass because moving script paths would churn imports, tests, docs, and existing references.

## What Was Archived

Generated `gen1` SVG DSL outputs were moved out of the active report surface into:

- `version/v7/artifacts/svg_dsl/gen1_archive_2026-04-05/`

This archive now holds:

- `gen1_all_asset_registry_*`
- `gen1_full_coverage_plan_*`
- `gen1_program_*`
- `gen1_full_family_packs/`
- `gen1_gold_mappings/`
- `gen1_smoke/`

## What Was Intentionally Left Alone

- `version/v7/reports/spec_broader_1_*`
  - still tracked and part of older documented history
- active text backprop/runtime scripts
- `version/v8/*`
  - out of scope for this cleanup

## Next Cleanup Pass

If we want a stricter `v7` later without breaking the known-good `qwen3` lane:

1. add a dedicated `version/v7/experiments/svg_dsl/` namespace
2. move experiment scripts there with compatibility wrappers
3. update tests/docs/imports in one controlled pass
4. keep `version/v7/scripts/` reserved for active text runtime/backprop tooling
