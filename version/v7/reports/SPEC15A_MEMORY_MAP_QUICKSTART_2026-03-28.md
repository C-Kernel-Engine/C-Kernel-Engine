# Spec15a Memory Map Quickstart

Date: 2026-03-28

## Read First

1. `docs/site/_pages/spec-training-method.html`
2. `version/v7/reports/SPEC15_DOMAIN_AGNOSTIC_VISUAL_DSL_KICKOFF_2026-03-27.md`
3. `version/v7/reports/VISUAL_DSL_FAMILY_PROGRAM_2026-03-27.md`
4. `version/v7/reports/SPEC15A_MEMORY_MAP_NEXT_AGENT_BRIEF_2026-03-28.md`

## Build Before Any Training

`spec15a` is a strict single-family `memory_map` line.

Current authored baseline:

1. Contract exists at `version/v7/reports/SPEC15A_EXECUTION_CONTRACT_2026-03-28.md`.
2. Current gold pack is `memory-layout-map`, `bump_allocator_quant`, and `v7-train-memory-canary`.
3. Strict compiler path exists in `spec15a_scene_canonicalizer_v7.py` and `render_svg_structured_scene_spec15a_v7.py`.
4. Gold `scene.dsl + content.json` pairs exist under `version/v7/reports/spec15a_gold_mappings/`.
5. Compiler smoke exists at `version/v7/reports/spec15a_smoke/compiler_smoke_report.json`.
6. Full-scene and compact-scene forms already compile to identical SVGs for all `3` gold cases.
7. Next work is token-budget, dataset, launcher, and probe authoring.

Training is blocked until:

- compiler smoke passes
- tokenizer coverage is complete
- longest gold output is under `80%` of context after tokenization
- dataset QC with disjoint holdouts passes
- parity regimen passes
- probe path, launcher, and rung policy exist

## First Runnable Training Step

Once infra is ready, launch only a small `spec15a r1` canary:

- about `40-80` examples total
- `3` hand-mapped gold assets
- `3-6` synthetic control variants
- `4-6` family-generic prompt surfaces per case
- default mix:
  - `40%` replay anchors
  - `40%` direct canonical rows
  - `20%` repair/contrast rows

The task is:

- family-generic control prompts in
- family-generic `memory_map` scene DSL out
- payload facts remain external

## Do Not Do

- do not broaden `spec14a`; keep it as the comparison baseline
- do not use topic-bearing prompts
- do not emit asset-specific ids or tokens
- do not put labels, numbers, prose, or domain facts in model input/output
- do not mix families before `memory_map` is nearly solved
- do not change grammar, capacity, and curriculum at once
- do not use a bigger model to hide a broken DSL/compiler/data contract
- do not launch training or autopilot before all preflight gates are green
