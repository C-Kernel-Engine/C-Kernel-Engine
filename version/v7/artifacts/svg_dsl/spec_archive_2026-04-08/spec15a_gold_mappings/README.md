# Spec15a Gold Mappings

These are the compiler-first gold targets for the first strict
domain-agnostic `memory_map` line.

They are not training rows yet.

Their purpose is:

1. prove the bounded `memory_map` family can be expressed cleanly
2. keep payload facts external in `content.json`
3. block training until the compiler contract is real

## Current Gold-Pack State

Smoke-ready today:

- `docs/site/assets/memory-layout-map.svg`
- `docs/site/assets/bump_allocator_quant.svg`
- `docs/site/assets/v7-train-memory-canary.svg`

Rejected after family review:

- `docs/site/assets/weight_memory_layout.svg`

Why rejected:

- it is a multi-section instructional poster with formulas, code, and format
  tables
- it does not fit the bounded `memory_map` board contract used by the current
  renderer/canonicalizer
- carrying it forward would silently broaden `spec15a` beyond the strict family
  we said we wanted

The old rejected candidate is kept here as research memory only.

## Required Files Per Asset

Each accepted asset should eventually have:

- `*.scene.dsl`
- `*.scene.compact.dsl`
- `*.content.json`

## Starting Reuse

Useful prior mappings already exist under `version/v7/reports/spec12_gold_mappings`
for:

- `memory-layout-map`
- `training-memory-canary`

Those should be treated as reference material, not copied blindly. `spec15a`
must keep the family while tightening the contract:

- family-generic model input only
- family-generic scene DSL only
- no topic-bearing dependence

The current `spec15a` gold workspace now contains:

- `memory-layout-map.scene.dsl`
- `memory-layout-map.scene.compact.dsl`
- `memory-layout-map.content.json`
- `bump_allocator_quant.scene.dsl`
- `bump_allocator_quant.scene.compact.dsl`
- `bump_allocator_quant.content.json`
- `v7-train-memory-canary.scene.dsl`
- `v7-train-memory-canary.scene.compact.dsl`
- `v7-train-memory-canary.content.json`
- `spec15a_gold_pack_status_20260328.json`

## Blocker Rule

Do not start `spec15a` training until all accepted assets have clean gold
compiler targets and a smoke pass exists.
