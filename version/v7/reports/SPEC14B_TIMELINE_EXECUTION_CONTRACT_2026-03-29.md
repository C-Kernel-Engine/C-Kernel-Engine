# SPEC14B Timeline Execution Contract 2026-03-29

Purpose: define the bounded-family execution contract for the `timeline`
family as the next broadening step after `spec15a`.

---

## Why This Is Not "Going Backward"

`spec14b` is a family branch label, not a claim that it is "smaller than"
`spec15a` in importance.

The numbering reflects the roadmap:

- `spec14a` = `comparison_board`
- `spec14b` = `timeline`
- `spec15a` = strict domain-agnostic `memory_map`

So starting `spec14b` after `spec15a` is not a regression. It means:

- freeze the `memory_map` win
- broaden to the next bounded family
- avoid jumping too early to a mixed-family `spec16`

`spec16` should only exist once multiple family winners are stable enough to
justify a generalized successor DSL/tokenizer line.

---

## Bottom Line

`spec14b` should be the bounded timeline family line.

It should:

- stay inside one family
- keep compiler-first discipline
- keep payload text and facts outside the scene model
- reuse the strict contract lessons from `spec15a`

This line is not "all SVG now."

It is the next clean family needed to move toward broad infographic coverage
without losing determinism.

---

## Family Scope

`spec14b` is `timeline` only.

In scope:

- dated milestone sequences
- grouped eras / phases
- event markers
- short callouts bound through `content.json`
- horizontal or vertical bounded timeline layouts

Out of scope for `r1`:

- mixed graph/timeline hybrids
- free-form posters
- system diagrams
- memory maps
- comparison boards

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
- dated prose
- payload labels, numbers, measurements, or copy

Allowed model output:

- family-generic `timeline` scene DSL only

Disallowed model output:

- topic-bearing ids
- asset-specific output tokens
- embedded milestone text or facts

Target pipeline:

`upstream request/router -> design/control bundle + content.json -> scene model -> scene DSL -> deterministic compiler -> SVG`

---

## Initial Gold Pack

Primary bounded assets:

- `docs/site/assets/ir-v66-evolution-timeline.svg`
- `docs/site/assets/ir-timeline-why.svg`

Reserve candidate if a third bounded case is needed:

- `docs/site/assets/ir-pipeline-flow.svg`

Rule:

- only keep the reserve if it can be expressed as a strict timeline family
- reject it if it is actually a process-flow or hybrid graph

---

## Method

Before any `spec14b` training:

1. author the family-generic timeline DSL
2. author gold `scene.dsl + content.json` pairs
3. build the deterministic compiler/canonicalizer
4. pass compiler smoke
5. pass tokenizer/context budget checks
6. pass dataset QC and parity
7. only then launch `r1`

Default `r1` package target:

- `2-3` hand-mapped gold assets
- `3-6` synthetic control variants
- `4-6` family-generic prompt surfaces per case
- roughly `40-80` unique train rows before repetition

Default staged mix:

- `40%` replay anchors
- `40%` direct canonical rows
- `20%` repair/contrast rows

---

## Success Condition

`spec14b` is successful when:

- the model can emit coherent `timeline` DSL for held-out prompts
- the compiler renders exact or near-exact timeline SVGs from that DSL
- payload facts remain external
- the family is good enough that the next broadening step is another family,
  not endless rung churn

---

## Immediate Next Authoring Steps

1. validate the two primary timeline assets as truly one family
2. decide whether the reserve candidate belongs
3. author the first `spec14b` DSL surface
4. build compiler smoke artifacts
5. only then define `spec14b r1`
