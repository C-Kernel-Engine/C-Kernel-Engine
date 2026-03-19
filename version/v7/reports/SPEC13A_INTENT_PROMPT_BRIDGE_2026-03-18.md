# SPEC13A Intent Prompt Bridge 2026-03-18

This note defines the first bridge from explicit scene-control prompts to
intent-driven scene planning.

`spec06` through `spec12` proved that the runtime can learn compiler-backed
scene DSLs under a tightly controlled contract. `spec13a` should not jump
straight to open-ended prose. The next step is a bounded prompt curriculum where
the model must choose scene structure from intent, while content and rendering
remain external and deterministic.

## Purpose

`spec13a` answers one narrow question:

- can the model infer a good scene plan when `layout`, `theme`, `tone`, and
  similar visual controls are omitted from the prompt?

It does **not** try to solve:

- open-domain English understanding
- free text generation
- world-knowledge retrieval
- arbitrary SVG synthesis

## Contract

Keep the `spec12` boundary:

- prompt -> `scene.dsl`
- external library -> `content.json`
- compiler -> final `svg`

`spec13a` changes only the prompt side. The output contract stays deterministic
and compiler-backed.

## Prompt Schema

Primary prompt form:

```text
[task:svg] [topic:<topic_id>] [goal:<goal_id>] [audience:<audience_id>] [OUT]
```

Optional fields for controlled ablations:

```text
[task:svg] [topic:<topic_id>] [goal:<goal_id>] [audience:<audience_id>] [emphasis:<hint>] [OUT]
```

or

```text
[task:svg] [topic:<topic_id>] [goal:<goal_id>] [audience:<audience_id>] [content_pack:<pack_id>] [OUT]
```

Primary rule:

- omit `layout`, `theme`, `tone`, `density`, `frame`, `background`, `rail`,
  `connector`, and other presentation choices from the prompt

Those are the fields the model should infer.

## Allowed Inferred Fields

For `spec13a`, the model may infer:

- `layout`
- `theme`
- `tone`
- `density`
- `frame`
- `background`
- `rail`
- `connector`
- `hero_align`
- `columns`
- `emphasis`
- component presence/order when the family allows multiple valid arrangements

## Fields That Must Stay Fixed

The model should **not** invent or free-write:

- literal infographic copy
- topic facts
- numeric values from world knowledge
- raw SVG coordinates
- gradients, markers, filters, or other low-level rendering details
- arbitrary new scene families outside the frozen `spec12/spec13a` vocabulary

Those stay in:

- `content.json`
- the compiler
- the scene schema

## What The Model Is Actually Learning

`spec13a` should teach:

- intent -> scene family choice
- goal -> component mix
- audience -> tone/emphasis defaults
- topic -> reusable scene planning priors

It should **not** be framed as "learn English now." This is still a bounded
ontology with explicit topic and goal anchors.

## Sample Prompts

These are the right shape for `spec13a`:

```text
[task:svg] [topic:quantization_formats] [goal:compare_options] [audience:technical] [OUT]
```

```text
[task:svg] [topic:failure_decision_tree] [goal:route_debug] [audience:operator] [OUT]
```

```text
[task:svg] [topic:memory_layout_map] [goal:explain_structure] [audience:engineer] [OUT]
```

```text
[task:svg] [topic:long_context_inference] [goal:compare_tradeoffs] [audience:technical] [OUT]
```

```text
[task:svg] [topic:governance_path] [goal:show_process] [audience:operator] [OUT]
```

```text
[task:svg] [topic:pipeline_overview] [goal:show_flow] [audience:technical] [OUT]
```

Controlled ablation examples:

```text
[task:svg] [topic:memory_layout_map] [goal:explain_structure] [audience:engineer] [emphasis:topology] [OUT]
```

```text
[task:svg] [topic:quantization_formats] [goal:compare_options] [audience:technical] [content_pack:compact] [OUT]
```

## Curriculum Mix

Do not train `spec13a` as a pure prompt jump. Use anchored curriculum mixing.

Recommended stage mix:

### Stage A

- `60%` explicit `spec12` anchor prompts
- `40%` `spec13a` prompts with omitted presentation fields

Goal:

- preserve format discipline
- start learning layout/style inference without destabilizing the scene grammar

### Stage B

- `40%` explicit `spec12` anchors
- `40%` `spec13a` primary prompts
- `20%` `spec13a` controlled ablations with `emphasis` or `content_pack`

Goal:

- shift the model burden from serialization to planning
- keep enough anchors to catch regressions immediately

## Holdout Design

`spec13a` needs harder holdouts than `spec12`, but they still must be
interpretable.

### H1: Layout-choice holdout

- prompt omits `layout`
- topic and goal are seen
- target checks whether the model chooses the correct family

### H2: Style-choice holdout

- prompt omits `theme`, `tone`, `density`, `frame`
- layout family is implied by topic/goal
- target checks whether inferred visual settings match the canonical gold scene

### H3: Topic x goal recombination

- both topic and goal are seen independently
- their combination is unseen in train
- target checks whether the model recombines known pieces instead of memorizing

### H4: Family-confusion holdout

- near-neighbor prompts that should map to different families
- example: `compare_options` vs `route_debug`
- target checks decision quality, not just syntax

### H5: Anchor replay

- exact `spec12` prompt style rows
- used only to confirm non-regression

## Success Metrics

Hard gates:

- `100%` parse success on canary prompts
- `100%` renderability on canary prompts
- zero stop-marker truncation
- no tokenizer `<unk>` fallbacks on the frozen prompt surface

Primary probe metrics:

- scene exact match on unambiguous prompts
- layout-family accuracy on `H1` and `H4`
- materialized exact match after compiler + `content.json`
- split metrics for train/dev/test
- non-regression against `spec12` anchor replay

Decision-quality metrics:

- inferred-field exactness for `layout`, `theme`, `tone`, `density`, `frame`
- component coverage for family-specific required blocks
- confusion matrix over chosen layout families

## Run Acceptance Rule

`spec13a r1` should only be considered successful if:

1. `spec12` anchor rows remain stable
2. omitted-field prompts do not collapse renderability
3. family-choice accuracy is materially above chance
4. the report shows where inferred-field errors occur, not just aggregate exact

If `spec13a` fails, the first question should be:

- did the model fail planning?

before asking:

- did the runtime fail training?

## What Comes Next

If `spec13a` works, then the next bridge is:

- `spec13b`: controlled paraphrases over the same ontology

Example:

```text
Make a technical comparison infographic about quantization formats.
```

That is the point where natural-language variation enters, but still inside a
bounded topic and family set.

## Bottom Line

`spec13a` is the first prompt-generalization spec, not the first open-language
spec.

The correct question is not:

- can the model write beautiful prose?

The correct question is:

- can the model choose a good scene plan when the prompt stops specifying the
  scene directly?
