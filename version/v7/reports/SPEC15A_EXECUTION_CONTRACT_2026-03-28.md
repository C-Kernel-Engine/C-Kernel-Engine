# SPEC15A Memory Map Execution Contract 2026-03-28

Purpose: define the explicit, agent-readable process for the first strict
domain-agnostic visual-language line.

This contract exists so future agents do not guess what `spec15a` is trying to
measure or silently fall back to topic-bearing prompt text.

---

## Bottom Line

`spec15a` should be the first line that:

- keeps the bounded-family training discipline proven in earlier specs
- freezes `spec14a` as the current `comparison_board` reference line
- starts a new `memory_map` family line
- enforces a strict family-generic boundary on both model input and model output
- keeps payload facts, labels, numbers, and prose outside the scene model

This line is not a topic-understanding benchmark.

It is a:

- family-definition line
- compiler line
- domain-agnostic prompt-contract line

---

## Hard Boundary

Allowed model-facing input:

- family name
- layout request
- reusable style controls
- composition/count controls
- generic audience/tone controls when they are reusable design signals

Disallowed model-facing input:

- topic names
- asset names
- case ids
- visible copy
- labels, measurements, numbers, or prose that belong in payload data
- domain facts such as `quantization`, `tokenizer`, or `cooking`

Allowed model output:

- family-generic `memory_map` scene DSL only

Disallowed model output:

- topic-bearing ids
- asset-specific DSL tags
- embedded facts or prose
- any output token that only exists to name one asset

Target pipeline:

`upstream request/router -> design/control bundle + content.json -> scene model -> scene DSL -> deterministic compiler -> SVG`

If an asset only works when the scene model sees topic-bearing prompt text, that
is a contract failure, not a training success.

---

## Method Baseline

Use the following as the method baseline:

- `docs/site/_pages/spec-training-method.html`
- `version/v7/reports/VISUAL_DSL_FAMILY_PROGRAM_2026-03-27.md`
- `version/v7/reports/SPEC15_DOMAIN_AGNOSTIC_VISUAL_DSL_KICKOFF_2026-03-27.md`

Use `spec14a` only as the bounded-family iteration reference.

Do not inherit `spec14a` topic-bearing prompt surfaces into `spec15a`.

---

## Family Scope

`spec15a` is `memory_map` only.

In scope:

- ordered memory regions
- address strips
- optional grouped brackets/spans
- optional side cards
- optional multi-lane memory views when they stay inside one bounded family

Out of scope for `r1`:

- mixed-family composites
- free-form posters
- timelines
- architecture graphs
- topic-conditioned prompt routing

---

## Initial Reference Assets

The initial bounded gold pack should use exactly these three assets unless a
strong compiler reason forces a substitution:

- `docs/site/assets/memory-layout-map.svg`
- `docs/site/assets/bump_allocator_quant.svg`
- `docs/site/assets/v7-train-memory-canary.svg`

Reserve candidates only if one of the above proves outside the bounded family:

- `docs/site/assets/weight_memory_layout.svg` (research memory only; rejected as
  a poster-style multi-section instructional asset)
- `docs/site/assets/memory-reality-infographic.svg`
- `docs/site/assets/activation-memory-infographic.svg`

Reasoning:

- `memory-layout-map.svg` is the clearest existing `memory_map` compiler target
- `bump_allocator_quant.svg` expands the family while staying inside one bounded
  ordered memory arena with address markers, typed regions, and side cards
- `v7-train-memory-canary.svg` adds a stronger operator-facing memory partition
  case while staying within the same family

Do not swap in poster-like assets just because they mention memory.

---

## DSL Starting Point

Start from the proven `spec12` compact `memory_map` surface conceptually, but
remove topic-bearing dependence.

Useful reusable components already seen in prior work:

- `header_band`
- `address_strip`
- `memory_segment`
- `region_bracket`
- `info_card`

`spec15a` should keep those components family-generic and reusable.

The next DSL iteration may add reusable structure only if the initial three
assets truly require it.

Do not add asset-specific bracket tokens or topic ids.

---

## Data Preparation Target For `r1`

The first package should be intentionally small:

- `3` hand-mapped gold assets
- `3-6` synthetic control variants
- `4-6` family-generic prompt surfaces per case
- roughly `40-80` unique train rows before repetition

Default staged mix:

- `40%` replay anchors
- `40%` direct canonical rows
- `20%` repair/contrast rows

The point of `r1` is not broad coverage. The point is to answer one bounded
question cleanly.

---

## Gates Before Any Training

Training is blocked until all of these are true:

- the `spec15a` DSL surface exists
- the deterministic compiler/canonicalizer exists
- gold `scene.dsl + content.json` pairs exist for the chosen assets
- compiler smoke passes
- longest gold output fits within `80%` of context after tokenization
- tokenizer coverage is complete
- dataset QC passes and holdouts are disjoint
- parity regimen passes
- seed/probe integrity path is ready
- launcher exists
- autopilot rung policy exists

If any item is missing, the correct action is authoring, not training.

---

## Work Sequence

### Phase 0: Freeze The Current Reference

- keep `spec14a` frozen as the `comparison_board` reference line
- do not broaden `spec14a`
- do not use `spec14a` prompts as the `spec15a` input contract

### Phase 1: Gold Pack Selection

1. confirm the three `memory_map` assets
2. verify they really form one bounded family
3. reject any candidate that is actually a poster or mixed-family asset

### Phase 2: DSL And Compiler First

1. author the first strict family-generic `memory_map` DSL
2. author gold `scene.dsl + content.json` pairs
3. compile all three with the real renderer
4. produce a smoke report

No training starts before this passes.

### Phase 3: Domain-Agnostic Prompt Contract

1. define family-generic model-facing prompt surfaces
2. ensure no topic-bearing text appears in those prompts
3. keep payload semantics external in `content.json`

### Phase 4: Tokenizer And Dataset

1. build tokenizer corpus from the explicit DSL surface
2. build direct canonical rows, replay anchors, and contrast/repair rows
3. create holdouts that test control binding, not domain wording

### Phase 5: `r1` Canary

Only after Phases 0-4 pass:

- run a small `r1` canary
- keep architecture fixed at `3L / 192d / 384ff / ctx768`
- keep the question narrow

### Phase 6: Autopilot

Autopilot may manage `spec15a` only after:

- launcher exists
- run brief path exists
- rung interventions are encoded
- the line remains inside the same family/tokenizer contract

---

## `r1` Question

`spec15a r1` should answer exactly one question:

Can a small model emit family-generic `memory_map` scene DSL from
family-generic control prompts, with payload data externalized, under a clean
deterministic compiler contract?

It should not try to answer:

- arbitrary topic generalization
- mixed-family planning
- open-ended infographic design
- larger-model scaling questions

---

## Promotion Gates

Promote a rung only when:

- exact and materialized exact improve materially
- renderability remains high
- the failure mode narrows cleanly
- holdout performance improves without reintroducing topic leakage

Pause the line when:

- the same failure class repeats for two clean rungs
- renderability is high but exactness stalls under the same contract
- contamination or probe-integrity issues are suspected

Change the contract when:

- the initial three assets cannot be expressed cleanly by one bounded
  `memory_map` family
- the compiler is still the bottleneck
- the DSL requires asset-specific hacks

Scale capacity only after:

- grammar, compiler, data, and probe paths are stable
- several clean canaries plateau on the same failure class

---

## Agent Handoff Checklist

The next agent should leave the repo with:

- this execution contract
- a finalized three-asset shortlist
- initial gold-pair authoring plan
- a clear family-generic prompt contract
- a clear note that topic-bearing input is disallowed

The first deliverable is not a training run.

The first deliverable is a working bounded-family compiler path plus the data
contract for `r1`.
