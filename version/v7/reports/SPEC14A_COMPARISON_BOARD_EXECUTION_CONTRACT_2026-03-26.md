# SPEC14A Comparison Board Execution Contract 2026-03-26

Purpose: define the explicit, agent-readable process for the first
post-`spec12 r17` successor line.

This contract exists so future agents do not have to infer the method from old
reports or guess when to start training.

---

## Bottom Line

`spec14a` should be the first line that:

- keeps the successful `spec12 r17` training method
- replaces the narrow `spec12` output ceiling with a broader generic DSL
- preserves external payload separation
- adds exactly one new reusable family: `comparison_board`

This line is **not** a prompt-understanding benchmark.

It is a representation and compiler line.

The model should learn:

- visual structure
- style controls
- family-specific reusable layout primitives

The model should **not** learn:

- topic facts
- asset-specific ids
- case-specific numbers or copy
- free-form English planning beyond the bounded prompt contract

---

## Hard Boundary

Keep these layers separate:

- output DSL: generic structure and visual control only
- current spec14a prompt text: user-visible request wording
- spec15+ model-facing prompt text: family-generic design/control wording only
- routing/request metadata: domain/topic identity, user-visible wording, and
  case selection
- `case_id`: dataset bookkeeping and routing metadata
- `content.json`: copy, numbers, labels, facts, and topic payload

If a future asset only works by adding its topic name to the model-side DSL,
that is a contract failure, not a training success.

This matches the boundary rule in
`version/v7/reports/SVG_FAMILY_EXECUTION_METHOD_2026-03-25.md`.

---

## Compatibility Promise

The `spec14a` line is allowed to start a new tokenizer line, but it must keep a
clear compatibility story.

Required carry-forward coverage:

- `table_matrix`
- `decision_tree`
- `memory_map`

Optional carry-forward reference:

- `flow_graph` may be brought forward later, but it is not required for the
  first `spec14a` compiler milestone

Interpretation:

- `spec14a` is a successor to the `spec12` family set
- it is not allowed to throw away the old compiler-backed families just to add
  `comparison_board`

---

## What Changes Relative To `spec12 r17`

What stays:

- whole-scene rows only
- hidden evaluation
- parity and preflight gating
- compiler-backed scoring
- small, interpretable early rungs
- fixed initial architecture: `3L / 192d / 384ff / ctx768`

What changes:

- the DSL becomes more generic
- the tokenizer is rebuilt from the new generic DSL corpus
- a new family, `comparison_board`, is added
- asset-specific semantics are pushed further out of the output contract

---

## `spec14a` Family Scope

`spec14a` is `comparison_board` only.

Scope:

- generalized comparison tables
- multi-card boards
- structured comparison layouts
- legends, chips, highlights, and grouped sections when those are reusable

Non-goals for `r1`:

- timeline
- free-form system diagrams
- arbitrary mixed-family composites
- open-ended content population

Representative first assets:

- `docs/site/assets/quantization-formats.svg`
- `docs/site/assets/tokenizer-performance-comparison.svg`
- `docs/site/assets/sentencepiece-vs-bpe-wordpiece.svg`
- `docs/site/assets/rope-layouts-compared.svg`
- `docs/site/assets/compute-bandwidth-chasm.svg`

---

## Work Sequence

### Phase 0: Freeze The Method Baseline

Use `spec12 r17` as the reference for:

- curriculum discipline
- probe discipline
- parity expectations
- promotion discipline

Do **not** reuse the `spec12` tokenizer blindly as the new ceiling.

### Phase 1: Define The Successor DSL

Author the next DSL/canonicalizer so it can express:

- the stable `spec12` family set
- the new `comparison_board` family

Rules:

- structure tokens must be reusable and family-level
- topic ids must not be required in the output DSL
- copy must remain external
- legacy `spec12` scenes should continue to canonicalize or have a deterministic
  adapter path

### Phase 2: Compiler First

Before any training:

1. author `3-5` gold `comparison_board` scenes
2. author matching `content.json`
3. compile them with the real renderer
4. generate a smoke report
5. verify the SVGs are visually strong enough to be worth learning

`spec14a` is not train-ready until this passes.

### Phase 3: Tokenizer Corpus

Only after the DSL is explicit:

1. build a tokenizer corpus from canonical DSL rows spanning:
   - legacy `spec12` families
   - new `comparison_board` gold rows
2. remove asset-specific ids from the output surface
3. keep generic visual and structural vocabulary only

Tokenizer rule:

- new line is allowed because the output grammar genuinely grows
- this must be a family-level tokenizer, not an asset-level token dump

### Phase 4: Dataset And Curriculum

`r1` should be explicit and conservative.

Stage A:

- `60%` legacy `spec12` anchor rows
- `40%` canonical `comparison_board` rows

Stage B:

- `40%` legacy anchors
- `60%` canonical `comparison_board` rows

For `r1`, avoid prompt-surface weakening.

Use:

- explicit tagged prompts
- whole-scene rows only
- hidden paraphrase probe
- hidden holdout probe

Do not start with:

- bridge-heavy prompt mixes
- topic-inference tasks
- multi-family omitted-field routing

### Phase 5: Parity And Preflight

Before the first real rung:

- materializer workspace check
- tokenizer roundtrip
- compiler smoke report
- probe contract build
- parity regimen
- preflight token-budget recommendation

No real rung starts until these pass.

### Phase 6: Training And Background Operation

Only after Phases 0-5 are complete:

- launch the run in a dedicated tmux train session
- launch a monitor session
- let autopilot poll and report automatically

Autopilot may auto-advance only when:

- the family policy entry exists
- the launcher exists
- the next rung interventions are explicitly encoded
- the line stays within the same family/tokenizer contract

Autopilot must stop when:

- the next rung would change family scope
- the tokenizer would need to change
- compiler authoring is incomplete
- the observed failure mode is not one of the encoded rung questions

---

## `r1` Question

`spec14a r1` should answer exactly one question:

Can the model reliably emit the new generic `comparison_board` DSL while
preserving the legacy `spec12` family anchors under the new tokenizer line?

It should **not** try to answer:

- whether the model can populate content from world knowledge
- whether omitted-field prompt routing works
- whether multiple new families can be added at once

---

## Promotion Gates

Promote a rung only when:

- `comparison_board` exact/materialized improves materially
- legacy `spec12` anchors do not regress badly
- hidden eval does not collapse
- compiler-valid outputs remain high
- the failure mode narrows cleanly

Pause the line when:

- renderability is high but exact/materialized stalls for two rungs
- new failures are dominated by contract ambiguity
- legacy anchors regress more than the new family improves

Change the contract when:

- the renderer is still the bottleneck
- the tokenizer keeps growing with individual assets
- family structure cannot be expressed without asset-specific DSL hacks

---

## Agent Handoff Checklist

Any agent may continue this line end-to-end only after the repo contains all of
the following:

- updated canonicalizer for the successor DSL
- `comparison_board` renderer/compiler path
- gold scenes plus `content.json`
- compiler smoke report
- materializer for the new family
- tokenizer corpus builder for legacy + `comparison_board`
- probe contract builder
- launcher script
- autopilot policy entry with encoded rung interventions

If any item is missing, the correct action is to finish authoring the missing
infrastructure first, not to launch training anyway.

---

## Start State

As of this contract:

- `spec12 r17` remains the method champion
- `spec13a` remains paused
- `spec13b` graph-family work is useful reference material, but not the
  training line to resume blindly
- `spec14a` is the explicit next family-construction line

The next concrete implementation step is:

1. author the successor DSL surface
2. implement `comparison_board` compiler/rendering
3. build the tokenizer corpus
4. then write the launcher and rung policy
5. only then start background training
