# Spec16 Kickoff 2026-03-29

Date: `2026-03-29`

## Status

`spec16` has started in authoring.

It is not training yet. The current work is the compiler-side foundation for a
generalized visual-DSL line above the stable family winners:

- `spec15a` `memory_map`
- `spec14b` `timeline`
- `spec15b` `system_diagram`

## New Artifacts

- Execution contract:
  [SPEC16_GENERALIZED_VISUAL_DSL_EXECUTION_CONTRACT_2026-03-29.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC16_GENERALIZED_VISUAL_DSL_EXECUTION_CONTRACT_2026-03-29.md)
- Founding-family pack:
  [spec16_asset_pack_20260329.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_asset_pack_20260329.json)
- Shared bundle schema:
  [spec16_scene_bundle_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/spec16_scene_bundle_v7.py)
- Shared bundle smoke builder:
  [build_spec16_bundle_smoke_report_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/build_spec16_bundle_smoke_report_v7.py)
- Shared bundle lowerer:
  [spec16_bundle_lowering_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/spec16_bundle_lowering_v7.py)
- Cross-family compiler smoke builder:
  [build_spec16_compiler_smoke_report_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/build_spec16_compiler_smoke_report_v7.py)

## Smoke

Shared bundle validation passed:

- JSON: [bundle_smoke_report.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/bundle_smoke_report.json)
- HTML: [bundle_smoke_report.html](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/bundle_smoke_report.html)

Result:

- `3/3` founding bundles validate
- canonical prompt-tag strings are emitted for all three families

Lowering smoke also passed:

- [memory_map.scene.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/lowered/memory_map.scene.dsl)
- [timeline.scene.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/lowered/timeline.scene.dsl)
- [system_diagram.scene.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/lowered/system_diagram.scene.dsl)

These files prove that a `scene_bundle.v1` can already lower into exact
family-specific scene DSL for the founding set.

Compiler smoke also passed:

- JSON: [compiler_smoke_report.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/compiler_smoke_report.json)
- HTML: [compiler_smoke_report.html](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/compiler_smoke_report.html)

Result:

- `3/3` founding bundles lower and compile successfully
- generated artifacts:
  - [memory_map.svg](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/gold_compiled/memory_map.svg)
  - [timeline.svg](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/gold_compiled/timeline.svg)
  - [system_diagram.svg](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_smoke/gold_compiled/system_diagram.svg)

## Decision

- `spec16` is now a real line, not just a future idea.
- Training is still blocked.
- The next authoring step is tokenizer/dataset audit and first shared-bundle
  probe design for `spec16 r1`.
