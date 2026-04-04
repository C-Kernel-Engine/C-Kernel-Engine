# Spec15 Domain-Agnostic Visual DSL Kickoff

Date: 2026-03-27

## Decision

Starting with the `spec15` line, the visual-DSL experiment should enforce a
strict boundary on both model input and model output:

- model input must be family-generic design/control language only
- model output must be family-generic scene DSL only
- domain facts, asset identity, labels, numbers, and prose must stay outside
  the scene model

This is a scope decision, not only a data-format decision.

## Why

The purpose of the line is to measure visual planning and compiler-backed scene
structure, not domain knowledge transfer.

If model-facing prompts contain domain text such as `quantization`, `tokenizer`,
or `cooking`, then the scene model is forced to learn topic-conditioned
associations. That expands the problem from "learn visual design language" to
"learn visual design language plus broad topic generalization."

That broader task is outside the intended scope of this experiment.

## Strict Boundary

Allowed model-facing input:

- family name
- layout request
- style controls
- composition controls
- count controls
- generic audience / tone controls when they are reusable design signals

Disallowed model-facing input:

- topic names
- asset names
- case-specific ids
- domain facts
- visible copy
- labels, measurements, or prose that belong in payload data

Allowed model output:

- scene DSL with family-generic structure and reusable control tokens

Disallowed model output:

- topic-specific ids
- asset-specific tags
- embedded facts or prose
- domain-bearing labels that should come from payload data

## Required Pipeline

The target architecture for strict visual-language work is:

`upstream request/router -> design/control bundle + content.json -> scene model -> scene DSL -> deterministic compiler -> SVG`

Where:

- upstream routing owns domain interpretation and payload selection
- the scene model owns visual planning only
- the compiler owns deterministic syntax expansion and SVG emission

## Spec15 Operator Policy

- Do not train `spec15` on topic-bearing prompt text.
- Do not score `spec15` improvements by adding richer domain phrasing to model
  inputs.
- If a family only works when the model sees asset identity, treat that as a
  contract failure, not a capability success.
- Prefer generic control prompts and external payload swaps over asset-grounded
  prompt rewrites.

## Immediate Implication

`spec12 r17` remains a strong bounded-family result, but it is not the method
target for `spec15` if it relies on topic-bearing prompt surfaces.

`spec14a` taught useful family-generalization lessons, but its prompt surfaces
still expose asset identity in model input. `spec15` should not continue that
pattern.

## Success Condition

`spec15` is successful when the same model-side prompt surface can drive the
same visual family regardless of whether the external payload is about
quantization, tokenizers, cooking, memory maps, or another topic, because the
scene model never sees those domain identities in the first place.
