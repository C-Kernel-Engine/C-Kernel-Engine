# Structured Domain Tokenizer Roadmap

This note records the intended path from a tiny SVG-control toy to a broader
multi-domain model that can work over:

- SVG composition
- code (`C`, shell)
- Linux / filesystem concepts
- English explanation and reasoning

## Core Principle

For small structured domains, prefer a hand-authored atom vocabulary over
letting BPE invent giant template tokens.

The goal is to train:

- control atoms
- structural atoms
- compositional rules

not:

- dataset-specific long chunks
- template retrieval disguised as reasoning

## Representation Layers

Use two layers:

1. Structured layer
   The model acts through stable domain atoms.

2. Surface layer
   The model explains and reasons in English.

The structured layer should stay crisp even when the surface layer grows.

## Stage 1: SVG Toy

Train on a fixed SVG DSL / IR vocabulary first.

Example prompt:

```text
[task:svg] [shape:circle] [color:red] [size:big] [OUT]
```

Example target:

```text
[svg] [w:128] [h:128] [circle] [cx:64] [cy:64] [r:36] [fill:red] [stroke:black] [sw:2] [/svg]
```

Why this is the right first step:

- every token has a clear job
- held-out combinations measure composition directly
- one bad token only damages one local attribute, not an entire SVG template
- vocab growth is intentional, not an artifact of merge heuristics

Do not start with raw XML for the toy if the tokenizer is allowed to learn
whole-document chunks.

## Stage 2: SVG Composition

Extend the same representation with:

- position atoms
- 2-shape compositions
- layout atoms
- text-card atoms
- simple chart atoms

Still prefer a closed structured vocabulary for the output representation.

If needed, render the structured output to actual SVG deterministically.

## Stage 3: Code / Bash / Linux

Add structured vocabularies for:

- `C` syntax atoms
- shell command atoms
- filesystem path / op atoms
- Linux concept atoms

Keep the same philosophy:

- control atoms should be explicit
- high-value structured symbols should be protected
- avoid giant corpus-specific merged chunks as the primary abstraction

## Stage 4: English + Structured Domains

Once the structured domains are stable:

- train a mixed tokenizer over a larger union corpus
- keep domain-critical atoms protected/reserved
- let natural language stay broad
- keep structure-bearing symbols stable

This is the point where a broader BPE can make sense.

## Frontier-Scale Intuition

Frontier models do usually train a tokenizer over a very large mixed corpus.
But that does not mean tokenizer design is an afterthought.

The important work happens before training:

- dedupe
- normalization
- domain balancing
- preserving critical control strings
- auditing bad merges
- validating segmentation on target tasks

The lesson is not "always use hand-authored vocab forever."

The lesson is:

- use hand-authored atoms when you need clean structure and intuition
- widen only after the structure is proven

## Practical Rule

For new structured domains:

1. define the target behavior
2. define the control atoms
3. define the structured output atoms
4. prove composition on a tiny closed dataset
5. only then widen tokenizer freedom

That is the path from toy SVG control to a mixed English + code + shell +
Linux + SVG model without losing the structure too early.
