# SPEC15B System Diagram Execution Contract 2026-03-29

Purpose: define the bounded-family execution contract for the first strict
`system_diagram` line after `spec15a` and `spec14b` established the
compiler-first family method.

---

## Why This Is The Next Step

`spec15b` is not "all SVG now."

It is the next bounded family needed to broaden coverage without collapsing the
training signal:

- `spec15a` = strict `memory_map`
- `spec14b` = strict `timeline`
- `spec15b` = strict `system_diagram`

The numbering is roadmap branch naming, not a monotonic capability ladder.

`spec16` should only begin once multiple bounded families are stable enough to
justify a generalized visual-DSL/tokenizer line.

---

## Bottom Line

`spec15b` should start as a narrow `system_diagram` family line that:

- keeps compiler-first discipline
- keeps model input and output family-generic
- reuses only the renderer/gold material that already fits one bounded family
- does **not** try to absorb every technical poster or hybrid architecture SVG

This line should prove that a small model can emit coherent `system_diagram`
scene DSL for a bounded left-to-right systems/pipeline board, with visible
payload remaining external in `content.json`.

---

## Family Scope

`spec15b` is `system_diagram` only.

In scope for `r1`:

- left-to-right stage pipelines
- architecture/dataflow boards with ordered processing stages
- one terminal result panel
- directional links between adjacent stages
- short footer interpretation band

Out of scope for `r1`:

- mixed graph + memory-map hybrids
- decision-tree branching
- timelines
- free-form posters
- multi-column comparison boards
- dense topology maps with nonlocal cross-links

---

## Hard Boundary

Allowed model-facing input:

- family name
- layout request
- reusable style controls
- count/structure controls
- generic audience/tone controls when they are reusable design signals

Disallowed model-facing input:

- topic names
- asset names
- case ids
- visible copy
- labels, measurements, or prose that belong in payload data
- domain-specific routing hints such as `kernel`, `qwen`, `compiler`, or
  `registry`

Allowed model output:

- family-generic `system_diagram` scene DSL only

Disallowed model output:

- topic-bearing ids
- asset-specific DSL tags
- embedded prose or facts
- node ids that name one topic instead of one reusable role

Target pipeline:

`upstream request/router -> design/control bundle + content.json -> scene model -> scene DSL -> deterministic compiler -> SVG`

---

## Initial Gold Pack

Primary bounded assets:

- `docs/site/assets/pipeline-overview.svg`
- `docs/site/assets/ir-pipeline-flow.svg`
- `docs/site/assets/kernel-registry-flow.svg`

Reserve bounded assets for later family expansion only:

- `docs/site/assets/qwen_layer_dataflow.svg`
- `docs/site/assets/ir-dataflow-stitching.svg`

Rejected for `r1`:

- `docs/site/assets/rdma-observer-architecture.svg`
- `docs/site/assets/architecture-overview.svg`

Reason:

- the three primary assets already form one clean bounded family:
  four ordered stages, adjacent links, one terminal panel, one footer band
- the rejected assets are more composite and should not be used to define `r1`

---

## DSL Starting Point

`spec15b` should not reuse topic-bearing `spec13b` node ids directly.

The family-generic starting surface should use reusable components such as:

- `header_band`
- `system_stage`
- `system_link`
- `terminal_panel`
- `footer_note`

The scene model should learn:

- ordered stage structure
- style bundle binding
- stage/link counts
- terminal panel placement

The scene model should **not** learn:

- graph-family branching logic
- asset-specific node names
- payload copy or terminology

---

## Data Preparation Target For `r1`

The first package should stay intentionally small:

- `3` hand-mapped gold assets
- `3-6` synthetic control variants
- `4-6` family-generic prompt surfaces per case
- roughly `40-80` unique train rows before repetition

Default staged mix:

- `40%` replay anchors
- `40%` direct canonical rows
- `20%` repair/contrast rows

---

## Gates Before Any Training

Training is blocked until all of these are true:

- the `spec15b` DSL surface exists
- the deterministic compiler/canonicalizer exists
- gold `scene.dsl + content.json` pairs exist for the chosen assets
- compiler smoke passes
- longest gold output fits within `80%` of context after tokenization
- tokenizer coverage is complete
- dataset QC passes and holdouts are disjoint
- parity regimen passes
- probe contract exists
- launcher exists

If any item is missing, the correct action is authoring, not training.

---

## Success Condition

`spec15b` is successful when:

- the model emits coherent `system_diagram` DSL for held-out prompts
- the compiler renders exact or near-exact system-diagram SVGs from that DSL
- payload facts remain external
- the family is good enough that the next move is another bounded family or the
  generalized `spec16` design phase, not endless rung churn

---

## Immediate Next Authoring Steps

1. copy the accepted three-asset gold pack into `spec15b_gold_mappings`
2. rewrite the content/schema around family-generic stage ids
3. build the `spec15b` canonicalizer and renderer
4. pass compiler smoke
5. only then author the generator/materializer and define `spec15b r1`
