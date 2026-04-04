# SPEC13B Generalized Scene IR Brief 2026-03-24

Purpose: define what `spec13b` must change after `spec13a` proved that the
current training stack is stable but the prompt-bridge task is stalled.

## Current Implementation Status

First compiler-side landing is now in place:

- generalized decision-graph lowering in
  `version/v7/scripts/spec13b_decision_graph_v7.py`
- generalized `decision_tree` renderer in
  `version/v7/scripts/render_svg_structured_scene_spec13b_v7.py`
- renderer registration in
  `version/v7/scripts/probe_report_adapters_v7.py`
- backward-compatible legacy smoke render in
  `version/v7/reports/spec13b_smoke/failure-decision-tree.legacy-adapter.svg`
- synthetic deeper-tree smoke render in
  `version/v7/reports/spec13b_smoke/synthetic-deeper-tree.svg`

What this means:

- `spec13b` is no longer just a design idea
- backward-compatible decision-tree lowering exists
- variable-depth decision-tree rendering exists as a smoke path
- table and memory-map generalization are still not implemented
- no `spec13b` training rung should start until the dataset/generator path is
  wired to this renderer

## Current State

`spec13a` did not fail like `spec12 r7/r8`.

What held:

- parity and preflight gates
- native `ck_cli` training
- whole-scene supervision
- frozen output DSL discipline

What stalled:

- family choice from partial intent
- robust inference of layout/style fields

Best `spec13a` rung is still `r2`:

- exact `0.596`
- renderable `1.000`
- materialized `0.596`

Latest `r6` regressed:

- exact `0.333`
- renderable `0.979`
- materialized `0.333`

Primary observed failure pattern in `r6`:

- `decision_tree -> table_matrix`
- `memory_map -> table_matrix`

This means the next step is not another prompt-mix tweak. The line is now
limited by the scene IR and renderer contract.

## What `spec12` Already Proved

`spec12 r17` proved:

- prompt -> scene DSL
- scene DSL + `content.json` -> deterministic compiler
- compiler -> final SVG
- the native CK train/infer stack is viable

`spec13a` tried to prove:

- the model can choose a good scene plan when `layout`, `theme`, `tone`, and
  `density` are omitted

`spec13a` only partially succeeded. It preserved grammar and renderability, but
it did not reliably solve family choice.

## What `spec13b` Must Prove

`spec13b` should prove a different claim:

- the system can render variable structure from a more general scene IR while
  staying deterministic and compiler-backed

This is where:

- deeper or wider decision trees
- variable-width tables
- more flexible memory-region layouts

belong.

`spec13b` is not:

- `spec13a` with harder prompts
- a civility/chat layer
- a free-form English benchmark

## Why The Current Contract Is The Bottleneck

The current renderer is family-template code.

Decision tree:

- hardcoded slots like `start`, `l0_l1`, `l2`, `l3_l4`, `finish`
- hardcoded connector routing

Table matrix:

- fixed 4-column geometry
- fixed legend/header/footer pattern
- repeated `table_block` template

Memory map:

- fixed stacked tower plus side-card pattern
- fixed segment ids and one special bracket-like span

So the parser looks more generic than the renderer actually is. The model is
still choosing among a few frozen templates rather than planning over a truly
general structure space.

## IR Direction

`spec13b` should split semantic structure from placement template.

### Decision Graph

Represent:

- nodes
- edges
- outcomes
- optional roles

Do not require:

- fixed node ids tied to renderer slots

### Table Schema

Represent:

- sections
- columns
- rows
- cells
- optional legends and notes

Do not require:

- fixed 4-column geometry

### Memory Regions

Represent:

- ordered regions
- optional spans/brackets
- optional callouts/cards
- optional lanes if needed later

Do not require:

- fixed segment ids or one special bracket rule

### Layoutable IR Stage

Add an intermediate stage:

- parse -> canonical IR -> layoutable IR -> SVG renderer

That is where deterministic placement should happen.

## Tokenizer Policy

Avoid repeating the `spec19`-style problem where new structural tokens force
scratch retraining.

Principles:

1. Keep the control vocabulary as small and stable as possible.
2. Prefer generic structure tokens over case-specific ids.
3. Keep semantic payloads external in `content.json`.
4. Only add reserved tokens when they express reusable structure, not a single
   new asset or case id.

Applied rule for graph-family work:

- `flow_graph` scenes should emit a family-generic contract
- asset identity such as `ir_pipeline_flow` or `qwen_layer_dataflow` should
  live in routing metadata, `case_id`, and external payloads
- older lines may still expose identity in prompt text, but spec15+ should not
- the model should learn graph planning, not memorize asset names in the DSL

## Architecture Policy

Do not turn `spec13b r1` into an architecture experiment.

Recommended `r1` policy:

- keep `3L / 192d / 384ff / ctx768`
- same native CK train/infer path
- same emphasis on whole-scene supervision

Reason:

- `spec13b r1` should answer an IR/renderer question first
- if it fails, we need to know whether the contract was wrong before asking
  whether the model was too small

Scale-up can come later if the generalized IR is stable.

## Milestones

### M1: Generic Decision-Tree IR

- variable node count
- variable branch count
- bounded variable depth
- deterministic layered placement

### M2: Backward-Compatible Adapter

- lower current `spec12/spec13a` decision trees into the new graph IR

### M3: Generalized Table IR

- variable column count
- variable section count
- optional legend/footer

### M4: Generalized Memory IR

- variable segment count
- explicit spans/brackets
- optional annotation cards

### M5: Unified Layoutable Scene Model

- shared layout stage before family-specific SVG paint

### M6: New Probes

Measure:

- parser correctness
- structure correctness
- materialized exactness
- renderability
- family choice when applicable

Do not rely on exact scene-string match alone once multiple valid structures
exist.

## `spec13b r1` Scope

Keep `r1` narrow.

Recommended first proof:

- implement generalized decision-tree IR + deterministic layered renderer
- provide an adapter so current `spec12/spec13a` decision trees still render
- train only enough to show the model can emit the new graph form on bounded
  cases

Do not try to solve:

- generalized tables
- generalized memory maps
- assistant/tool behavior
- open-ended prose

in the same rung.

## Acceptance Checklist

`spec13b r1` should only count as a success if:

1. Existing `spec12/spec13a` decision-tree scenes still render through the new
   path.
2. The new graph renderer supports at least one bounded variable-depth case.
3. Compiler/render determinism remains intact.
4. Materialized correctness is prioritized over raw scene-string exactness.
5. The run report clearly separates:
   - old-template compatibility
   - new generalized-graph capability

## Main Risks

- backward-compatibility regressions
- new valid-layout multiplicity weakening exact-match probes
- tokenizer drift from over-specializing new structure tokens
- opening too much structure freedom too fast and recreating attractor failures
- confusing IR generalization with assistant behavior

## Bottom Line

`spec13b` should be a renderer and IR upgrade, not just another prompt-bridge
rung.

The highest-signal first step is:

- generalized decision-tree IR
- deterministic layered renderer
- backward-compatible adapters
- unchanged architecture for `r1`

Only after that should broader structure families or larger models be tested.
