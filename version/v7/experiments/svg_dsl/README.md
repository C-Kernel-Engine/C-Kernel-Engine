# v7 SVG DSL Reuse Map

This namespace is the planned landing zone for the non-core SVG/spec/gen1
surface that is still sitting in `version/v7/scripts/`.

The goal is not to throw that work away. The goal is to separate:

- reusable compiler/data primitives
- reusable catalog/program definitions
- one-off gen1 bootstrap/report generators
- generated outputs that belong in `version/v7/artifacts/`

## What Is Actually Reusable

### 1. Core primitives

These are the strongest reuse candidates and should stay as source code, not
artifacts:

- `version/v7/scripts/atomic_scene_dsl_v7.py`
  - canonical atomic scene DSL serialization/lowering
  - bridges atomic DSL and older legacy scene tokens
- `version/v7/scripts/atomic_dsl_tokenizer_v7.py`
  - fixed-vocab tokenizer surface for the atomic DSL
  - useful beyond gen1 because it encodes the contract of the DSL itself
- `version/v7/scripts/render_svg_structured_scene_gen1_v7.py`
  - family router that delegates layouts to the right renderer
  - useful as a reusable compatibility layer while layout families evolve

These three files are the real reusable substrate. If we keep any part of the
SVG/gen1 line alive, these should be promoted into a stable shared location
first.

### 2. Catalog / planning definitions

These are reusable as data/program definitions, even though they are not part
of the active text runtime:

- `version/v7/scripts/build_gen1_full_coverage_plan_v7.py`
  - owns `FULL_COVERAGE_FAMILIES`
  - effectively the family catalog for the broader asset library
- `version/v7/scripts/build_gen1_all_asset_registry_v7.py`
  - derives the full asset registry and family pack structure
  - useful as a catalog/materialization tool
- `version/v7/scripts/build_gen1_program_v7.py`
  - reusable as a program-level planning/report builder

This layer is not runtime-critical, but it is reusable if we want to run a
later `gen2`/`genN` broad scene-DSL program without rebuilding the planning
surface from scratch.

## What Is Mostly One-Off Program Logic

These files are useful history, but they are mostly program-specific wrappers or
data tables rather than general infrastructure:

- `version/v7/scripts/bootstrap_gen1_*_gold_packs_v7.py`
  - most of the value is the embedded gold-pack data per family
  - they should live under a `programs/gen1/` area, not the active scripts root
- `version/v7/scripts/build_gen1_*_smoke_report_v7.py`
  - useful as program QA wrappers, but not core runtime infrastructure
- `version/v7/scripts/gen1_*`
  - bootstrap/preflight/train shell orchestration for one specific program line
- `version/v7/scripts/generate_svg_structured_gen1*_v7.py`
  - program-facing generators, reusable only inside the SVG/gen1 experiment line
- `version/v7/scripts/render_svg_structured_scene_gen1_*_v7.py`
  - reusable within the experiment family layer, but not part of active text
    runtime/training

These should not be archived as outputs, but they also should not remain mixed
into the top-level active `version/v7/scripts/` surface forever.

## What Belongs In Artifacts

Generated outputs belong under:

- `version/v7/artifacts/svg_dsl/`

That includes:

- generated `*.json`, `*.md`, `*.html`
- smoke reports
- gold-pack status outputs
- generated `.scene.compact.dsl` snapshots
- archived family-pack snapshots

It does **not** include Python source, shell source, or tests.

## Recommended Future Layout

If we do the actual source move later, the clean split should look like:

- `version/v7/experiments/svg_dsl/core/`
  - atomic DSL serializer/lowering
  - atomic DSL tokenizer helpers
  - shared scene/router helpers
- `version/v7/experiments/svg_dsl/catalog/`
  - family catalogs
  - asset registries
  - coverage-plan builders
- `version/v7/experiments/svg_dsl/renderers/`
  - family renderers
  - gen1 router/delegation layer
- `version/v7/experiments/svg_dsl/programs/gen1/`
  - gold-pack bootstrap builders
  - smoke report builders
  - gen1 shell entrypoints and preflight wrappers
- `version/v7/artifacts/svg_dsl/`
  - generated outputs only

## Current Cleanup Decision

For the current dirty worktree:

- keep atomic DSL + tokenizer + active text/training files as source
- keep gen1/program code as source, but classify it as experiment source
- keep generated archives in `version/v7/artifacts/svg_dsl/`
- do not move experiment source into artifacts
- when we do the real move, leave compatibility wrappers in
  `version/v7/scripts/` so imports/tests/docs do not break all at once

## Practical Rule

Ask one question per file:

- if deleting the file would destroy reusable semantics or contracts, it is
  source and should stay reusable
- if deleting the file would only remove a generated snapshot, it belongs in
  artifacts
- if the file is source but only meaningful for one bootstrap line, move it to
  `experiments/svg_dsl/programs/gen1/` later
