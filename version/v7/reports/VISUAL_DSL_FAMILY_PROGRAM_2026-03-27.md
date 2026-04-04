# Visual DSL Family Program

This file is the family-level program for the visual DSL training line.

It defines what each spec is trying to learn, how the compiler/data/model loop
should advance, and how to judge whether a new spec is better than the previous
one.

## Mission

Train the model to generate scene DSL that deterministically compiles into SVG
quality close to the target visual family.

The target is not domain knowledge.
The target is visual design language plus correct DSL generation.

Visible facts, labels, topic names, numbers, and prose must stay outside the
scene-model contract unless a spec explicitly declares otherwise.

## Core Principle

Do not ask the model to invent content and design at the same time.

The line should converge toward:

`request/router -> design control bundle + content.json -> scene model -> scene DSL -> deterministic compiler -> SVG`

The scene model is responsible for visual planning and DSL correctness.
The compiler is responsible for deterministic SVG realization.

## Methodical Family Loop

For every new spec:

1. Choose the target visual family.
   The reference family should come from a bounded SVG library such as
   `docs/site/assets/*.svg` or another explicitly declared asset set.

2. Define what this spec is trying to generate.
   Write the family goal in a run-brief or spec-brief before training begins.
   State the visual family, the DSL contract, the compiler contract, and the
   success gates.

3. Design or update the DSL first.
   Add only the control surface needed for the chosen visual family.
   Keep the DSL generic and family-oriented.

4. Update the compiler before the model.
   Prove that the deterministic compiler can generate correct SVG for the new
   DSL surface before training the model on it.

5. Test compiler parity first.
   Build smoke assets and canonical DSL examples.
   If the compiler cannot render the family correctly, training is blocked.

6. Generate model data from the working compiler contract.
   Produce narrow, family-bounded rows first.
   Start with a limited visual library so the model learns one family cleanly.

7. Train the model to emit DSL, not final SVG facts.
   The model should learn design language, structure, and DSL closure.
   It should not depend on topic-specific facts to succeed.

8. Compare to the previous spec.
   Judge whether the new spec is better than the previous one on the intended
   family, not just on loss.

9. Broaden only after the family is nearly solved.
   When one family reaches near-parity, expand to a broader library or a new
   visual family in the next spec.

10. Repeat without discarding history.
    Keep every spec and rung. Regressions and dead ends are part of the
    research memory and must remain visible.

## What “Better Than Previous Spec” Means

A new spec is better only if it improves the intended family contract.

Priority order:

1. Compiler-valid DSL generation
2. Exact visual-control binding
3. Render/materialization correctness
4. Holdout generalization within the same visual family
5. Broader family coverage without breaking prior solved families

Loss alone is not sufficient evidence.

## Narrow-Then-Broaden Policy

The intended progression is:

1. Pick one visual family
2. Build DSL and compiler support for that family
3. Train on a limited library
4. Reach near-full parity on that family
5. Add the next family as a new spec
6. If needed, expand tokenizer or pretraining only when the new family truly
   requires a broader control surface or representation reset

This means new model families should correspond to newly added visual families,
not random rung churn inside the same unresolved contract.

## Spec-Level Checklist

Before any new spec is declared ready:

- target family is named
- target asset library is named
- DSL additions are documented
- compiler updates are implemented
- compiler smoke tests pass
- dataset generation path is implemented
- run brief exists
- success gates are explicit

## Rung-Level Checklist

Before any new rung is launched:

- the rung objective is explicit
- the mutable surfaces are explicit
- the unchanged surfaces are explicit
- the previous rung result is summarized
- the expected improvement is named
- the new run writes `run_scope.json`, `agent.md`, and `training.md`

## Non-Goals

- training the model to know arbitrary external topics
- mixing domain facts into the scene-model prompt surface
- hiding failed runs
- replacing the compiler with language modeling
- declaring progress from loss reduction alone

## Operator Rule

If a spec only works because the model memorizes topic-bearing prompt text,
that is not the target method for the strict visual-language line.

The strict line should improve because the model learned the visual family and
its DSL, not because it recognized a topic phrase.
