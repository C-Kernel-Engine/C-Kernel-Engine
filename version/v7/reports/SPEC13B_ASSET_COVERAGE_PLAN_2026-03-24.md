# SPEC13B Asset Coverage Plan 2026-03-24

Purpose: turn the broad `spec13b` goal into a concrete renderer-family roadmap
 tied to the actual `docs/site/assets` library.

## Bottom Line

We should not try to make one `spec13b` renderer cover every site asset in one
 jump.

The right sequence is:

1. generalized `decision_tree`
2. generalized `flow_graph`
3. generalized `comparison_board`
4. generalized `memory_map`
5. generalized `timeline`

This is the shortest path to broad practical coverage while keeping the
 training contract deterministic.

## Asset Family Scan

Current rough coverage buckets from `docs/site/assets/*.svg`:

- `comparison_board`: 25
- `flow_graph`: 16
- `technical_diagram`: 22
- `memory_map`: 8
- `decision_tree`: 2
- `timeline`: 2
- `other`: 22

Interpretation:

- many “technical” and “other” assets are still reducible to either
  `flow_graph` or `comparison_board` with the right content pack
- the biggest missing structural family after decision trees is clearly
  `flow_graph`

## Renderer Priority

### P1: Decision Graph

Status:

- partially implemented
- backward-compatible legacy lowering exists
- variable-depth smoke render exists

Purpose:

- deeper gate ladders
- wider branching diagnostic flows
- deterministic layered graph placement

Representative assets:

- `ir-v66-failure-decision-tree.svg`
- `ir-v66-gate-ladder.svg`

### P2: Flow Graph

This is the next highest-leverage family.

Why:

- covers pipeline diagrams
- covers topology/flow diagrams
- covers registry/lineage chains
- covers many architecture/dataflow diagrams

Representative assets:

- `pipeline-overview.svg`
- `ir-pipeline-flow.svg`
- `ir-lowering-pipeline.svg`
- `ir-v66-artifact-lineage.svg`
- `kernel-registry-flow.svg`
- `ir-kernel-registry-chain.svg`
- `qwen_layer_dataflow.svg`
- `forward-backward-flow.svg`
- `rdma-observer-architecture.svg`

IR requirements:

- nodes
- directed edges
- optional swimlanes or groups
- optional stage badges
- optional edge labels
- deterministic layered or left-to-right placement

Training implication:

- this is likely the single best new family for `spec13b r2` or `r3`

### P3: Comparison Board

This is the next biggest general family.

Why:

- many infographic assets are actually structured comparisons, not arbitrary
  diagrams
- a strong comparison renderer would absorb a large share of the current asset
  library

Representative assets:

- `quantization-formats.svg`
- `tokenizer-performance-comparison.svg`
- `sentencepiece-vs-bpe-wordpiece.svg`
- `rope-layouts-compared.svg`
- `compute-bandwidth-chasm.svg`
- `performance-balance.svg`
- `architecture-overview.svg`
- `comparison-diagram.svg`

IR requirements:

- sections or cards
- columns and rows
- optional highlights
- optional legends and callouts
- optional score/chip/badge blocks

Training implication:

- this should absorb many current `table_matrix` and “board-like” assets under
  a more flexible contract

### P4: Generalized Memory Map

Current memory renderer is useful but still template-bound.

Representative assets:

- `memory-layout-map.svg`
- `weight_memory_layout.svg`
- `v7-train-memory-canary.svg`
- `activation-memory-infographic.svg`
- `memory-reality-infographic.svg`

IR requirements:

- ordered regions
- optional spans/brackets
- optional side cards
- optional multiple lanes

### P5: Timeline

Small family count, but easy to isolate.

Representative assets:

- `ir-timeline-why.svg`
- `ir-v66-evolution-timeline.svg`

IR requirements:

- events
- milestones
- grouped eras
- callouts

## Tokenizer Policy

Two modes are now necessary.

### Mode A: Frozen-Tokenizer `spec13b r1`

Use when:

- proving the generalized decision-tree renderer only
- staying warm-start-friendly from `spec12/spec13a`

Policy:

- no new reserved output control tokens
- legacy warm-start mode only: semantics may stay in prompt text and
  `content.json`
- stricter spec15+ policy should remove domain-bearing semantics from
  model-facing prompt text as well
- keep `3L / 192d`

### Mode B: New Tokenizer Line

Use when:

- adding genuinely new output families such as `flow_graph`,
  `comparison_board`, or `timeline`
- introducing new reusable structural tokens

Policy:

- new tokenizer line is acceptable
- do not pretend this is warm-start-compatible with old output vocab
- keep the new reserved token surface generic, not asset-specific

## Practical Recommendation

Do not train broad `spec13b` yet.

Next engineering sequence:

1. finish generalized decision-tree path
2. implement generalized `flow_graph`
3. add 3-5 flow-graph gold assets
4. define whether this stays frozen-vocab or becomes a new tokenizer line
5. then train the first real `spec13b` rung

## What This Means

We do not need to redo the entire empirical journey from scratch.

What transfers from `spec12`:

- whole-scene supervision
- parity/preflight gating
- hidden eval
- loss skepticism
- compiler-backed scoring

What changes for `spec13b`:

- the compiler/renderer contract becomes the main moving part
- each new renderer family should be proved with smoke renders before training
- training should follow renderer maturity, not lead it
