# SPEC16 Generalized Visual DSL Execution Contract 2026-03-29

Purpose: define the first multi-family successor line now that several bounded
families have stable `training winner + decode validation` paths.

---

## Bottom Line

`spec16` should not be:

- arbitrary SVG generation
- free-form topic planning
- a rollback to raw family mixing

`spec16` should be:

- the first shared visual-planning line
- one model over multiple stable infographic families
- a generalized scene-bundle IR above the family-specific compilers
- compiler-first and repair-aware from day one

Target pipeline:

`upstream request/router -> shared scene bundle -> family lowerer -> family DSL -> deterministic compiler -> SVG`

The model should learn the shared bundle, not raw SVG.

---

## Starting Preconditions

`spec16` is allowed now because these families are stable:

- `spec15a` `memory_map` via [SPEC15A_R9_DECODE_VALIDATION_2026-03-29.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC15A_R9_DECODE_VALIDATION_2026-03-29.md)
- `spec14b` `timeline` via [SPEC14B_R3_DECODE_VALIDATION_2026-03-29.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC14B_R3_DECODE_VALIDATION_2026-03-29.md)
- `spec15b` `system_diagram` via [SPEC15B_R2_DECODE_VALIDATION_2026-03-29.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC15B_R2_DECODE_VALIDATION_2026-03-29.md)

This is enough to start generalized authoring. It is not yet a license to skip
compiler-first validation.

---

## Hard Boundary

Allowed model-facing input:

- family
- form
- shared style controls
- shared topology/count controls
- generic audience/tone controls when they are reusable design signals

Disallowed model-facing input:

- topic names
- asset names
- case ids
- visible copy
- domain-specific prose or measurements

Allowed model output:

- shared `scene_bundle.v1` only

Disallowed model output:

- raw SVG
- asset-specific ids
- payload prose
- one-off family hacks that do not generalize

---

## Shared Bundle Contract

The initial shared bundle must at minimum carry:

- `family`
- `form`
- `theme`
- `tone`
- `density`
- `background`
- family topology counts

Initial founding families:

- `memory_map`
- `timeline`
- `system_diagram`

The first compiler-side implementation of this bundle is:

- [spec16_scene_bundle_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/spec16_scene_bundle_v7.py)

This bundle can already normalize canonical prompt-tag strings for the founding
families. That prompt surface is the first compatibility bridge, not the final
training target.

---

## Initial Asset Pack

Use:

- [spec16_asset_pack_20260329.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec16_asset_pack_20260329.json)

Do not add new families until the first shared bundle lowering path works
cleanly on the founding set.

Keep `comparison_board` as reserve until it has the same explicit repaired
inference path.

---

## Stage Order

### Stage 0: Shared Bundle Authoring

1. define `scene_bundle.v1`
2. define family-specific count vocabularies
3. prove bundle validation on the founding families

### Stage 1: Family Lowerers

1. lower shared bundle -> `memory_map` family DSL
2. lower shared bundle -> `timeline` family DSL
3. lower shared bundle -> `system_diagram` family DSL

The lowerers must be deterministic.

### Stage 2: Compiler Smoke

For each founding family:

1. author `3-5` shared bundles
2. lower to family DSL
3. compile to SVG
4. verify the family compiler path remains correct

### Stage 3: Tokenizer Audit

Only after the shared bundle is stable:

1. build a tokenizer corpus from the finalized shared bundle surface
2. verify context budgets
3. only then define `spec16 r1`

### Stage 4: Training

`spec16 r1` should answer one question only:

Can a small model emit a correct shared scene bundle across multiple stable
families without topic-bearing input?

It should not try to learn all SVG families at once.

---

## Immediate Next Authoring Steps

1. validate the founding shared bundle examples
2. write the first family lowerers from shared bundle -> family DSL
3. produce a compiler smoke report across the three founding families
4. only then define tokenizer and dataset work for `spec16 r1`
