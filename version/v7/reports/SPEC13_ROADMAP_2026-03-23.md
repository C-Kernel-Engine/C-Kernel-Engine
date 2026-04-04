# SPEC13 Roadmap 2026-03-23

Purpose: define the next line after `spec12` without collapsing prompt-bridge,
renderer generalization, and assistant behavior into one opaque experiment.

## What `spec12` Already Proved

`spec12` proved a real bounded capability:

- prompt -> scene DSL
- scene DSL + `content.json` -> deterministic compiler
- compiler -> final SVG
- native CK train/infer path is viable

The strongest checkpoints are:

- `r17`: strict visible-contract champion
- `r20`: proof that frozen-vocab semantic expansion works

Main lessons from `spec02` to `spec12`:

- loss is not the decision surface
- compiler-backed structure is the correct supervision target
- whole-scene rows are safer than fragment repair rows
- hidden eval matters
- tokenizer drift makes warm-starting fragile unless the control surface is frozen

## Overall Goal For `spec13`

Make the line more generalized as an infographic planner without pretending that
one run can solve:

- open-ended English
- arbitrary topology
- arbitrary new SVG families
- conversational assistant behavior

`spec13` should be the umbrella for that progression, not a single monolithic
run.

## Three Sub-Lines

### `spec13a` - Intent Prompt Bridge

Question:

- can the model choose a good scene plan when the prompt omits explicit
  `layout/theme/tone/density`?

Keep fixed:

- same compiler boundary
- same scene DSL output family
- external `content.json`
- deterministic render path

### `spec13b` - Generalized Scene IR / Renderer

Question:

- can the renderer and DSL support variable-width / variable-depth scenes
  rather than fixed family templates?

This is where generalized decision trees, wider tables, and more flexible
memory-map layouts belong.

### `spec13c` - Assistant / Tool Shell

Question:

- can the system act more naturally around the scene planner?

This is where:

- `hello`
- `thank you`
- `help me make...`
- compile/save/open tool actions

belong. Do not pollute the core scene-only target distribution with assistant
chatter.

## Architecture Policy

For `spec13a r1`, keep the architecture fixed to the current working line:

- family: `qwen3-like`
- layers: `3`
- embed dim: `192`
- hidden dim: `384`
- heads: `8`
- kv heads: `4`
- context: `768`

Reason:

- `r1` should answer a prompt-contract question, not an architecture question
- changing both prompt contract and model size at once weakens the evidence

Scale-up should only happen after the prompt-bridge line is stable.

Suggested later architecture checks:

- `spec13a r3/r4`: `4L/256d`
- `spec13b`: only after the generalized renderer exists

## Tokenizer Principles

For `spec13a`, warm-startability matters more than making every new prompt field
a reserved special token.

Principles:

1. Keep the output scene-control vocabulary frozen when possible.
2. Prefer a stable reserved DSL/control vocabulary.
3. Allow new prompt-side fields to ride on the ASCII base when needed.
4. Do not create new reserved tokens for every topic/case/id if the compiler
   can keep those external.

For `spec13a r1`, the right compromise is:

- freeze the tokenizer to the `spec12 r17` line
- let new prompt-side fields like `goal` / `audience` encode through the ASCII
  base instead of forcing a tokenizer reset

## `spec13a r1`

### Narrow Question

Can the model infer the correct `layout/theme/tone/density` from a bounded
intent prompt while preserving the `spec12` scene grammar?

### Prompt Schema

Primary prompt:

```text
[task:svg] [topic:<topic_id>] [goal:<goal_id>] [audience:<audience_id>] [OUT]
```

No explicit:

- `layout`
- `theme`
- `tone`
- `density`

### Training Policy

- seed from `spec12 r17`
- keep tokenizer frozen to `r17`
- keep output DSL exactly in the `spec12` family
- use whole-scene rows only
- no fragment repair rows

### Curriculum

Stage A:

- `60%` explicit `spec12` anchor prompts
- `40%` intent prompts

Stage B:

- `40%` explicit anchors
- `60%` intent prompts

### Success Rule

`spec13a r1` is successful only if:

1. `spec12` anchor rows do not regress badly
2. intent prompts remain renderable
3. inferred family choice is materially above chance
4. the report separates anchor behavior from intent behavior

## Candidate Rung Plan

### `r1`

- same `3L/192d` architecture
- frozen tokenizer from `r17`
- bounded topic/goal/audience prompt bridge
- seed from `r17`

### `r2`

- add controlled ablations:
  - `emphasis`
  - `content_pack`
- widen hidden eval

### `r3`

- add more semantic cases under the same frozen tokenizer
- only if `r1/r2` keep anchors stable

### `r4`

- optional modest architecture scale test
- only if prompt-bridge behavior is stable enough that capacity is the real
  question

## Renderer Milestones

Do not expect `spec13a` to solve arbitrary-depth trees.

Renderer sequence:

1. `spec13a`: keep current `spec12` renderer families
2. `spec13b`: introduce generalized tree/table/memory layout logic
3. only then train for variable depth/width/topology

## Assistant / Tool Milestones

The assistant/tool shell should wrap the scene planner, not corrupt it.

Milestones:

1. prompt -> scene DSL -> compile -> save/open
2. bounded tool loop in Python
3. later, trained tool-calling behavior if still needed

## Bottom Line

`spec13` should aim at more generalized infographic planning.

But `spec13a r1` should stay narrow:

- same model line
- same output contract
- new prompt-side planning burden

That is the most deterministic next experiment.
