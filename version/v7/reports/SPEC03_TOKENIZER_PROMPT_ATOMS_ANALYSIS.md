# spec03 Tokenizer Prompt-Atoms Analysis

## Scope

This note is the current tokenizer-side read on `spec03` using the live staged dataset and bootstrap run artifacts.

Primary evidence:
- `version/v7/data/spec03/tokenizer/spec03_tag_seed_rows.txt`
- `version/v7/data/spec03/tokenizer/spec03_tokenizer_corpus.txt`
- `version/v7/data/spec03/tokenizer/spec03_tokenizer_corpus_manifest.json`
- `version/v7/data/spec03/manifests/spec03_fit_audit_manifest.json`
- `/home/antshiv/.cache/ck-engine-v7/models/train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512/tokenizer_roundtrip.json`
- `/home/antshiv/.cache/ck-engine-v7/models/train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512/tokenizer_vocab_index.jsonl`

## Bottom Line

- The tokenizer is technically clean enough to proceed.
- Exact encode/decode roundtrip passes.
- The old bad cross-row merge class like `</svg>\n<svg` is gone.
- The remaining problem is not reversibility. It is **control-interface representation**.
- `spec03` is using a DSL-like prompt contract. Those control tags should be protected atoms, not left to ordinary BPE frequency competition.

## Current State

### Corpus composition

Current staged tokenizer corpus:
- `spec03_tokenizer_corpus.txt`: `781` rows
- `spec03_tag_seed_rows.txt`: `98` rows
- `small_full_rows`: `62`
- `structural_rows`: `617`
- `tag_seed_rows`: `98`

This is materially better than the older `731/48` state.

### Roundtrip status

From the live bootstrap run:
- `status = pass`
- `exact_match = true`
- `byte_match_rate = 1.0`
- `line_match_rate = 1.0`
- `input_lines = 781`
- `token_count = 146266`

So the tokenizer is reversible and staging is coherent.

### Merge quality

What is fixed:
- `bpe_max_piece_bytes = 72`
- cross-row merges such as `</svg>\n<svg` are not present in the learned vocab

What is acceptable:
- long SVG-structural tokens such as gradient, marker, font-family, and stop-color fragments

What is still weak:
- exact control tags are not present as learned pieces

## Prompt-atom coverage

Expected canonical control tags from `spec03_tag_seed_rows.txt`:
- `50`

Exact learned pieces in the tokenizer vocab matching those tags:
- `0 / 50`

That means:
- SVG/XML structure is learning well
- prompt/control atoms are still fragmented
- increasing vocab to `2048` was not enough by itself

This is the core representation issue.

## Why this matters

The prompt side is not ordinary prose. It is a DSL for controlling SVG generation.

For a DSL-like interface, the important units are:
- `[rect]`
- `[bar-chart]`
- `[palette:cool]`
- `[style:minimal]`
- `[layout:grid]`
- `[complexity:rich]`

If those remain fragmented across many subpieces, the model must learn the interface through longer token chains instead of stable control atoms.

That has three costs:
1. weaker control conditioning
2. more fragile generalization across rare tags
3. wasted token budget on the prompt side

## Fit audit summary

Using the current tokenizer:

### `small_full`
- `512`: `19 / 62` fit (`30.6%`)
- `2048`: `61 / 62` fit (`98.4%`)

### `structural`
- `512`: `595 / 617` fit (`96.4%`)
- `2048`: `617 / 617` fit (`100%`)

Practical implication:
- `512` is fine for structural pretrain
- `2048` is the first useful setting for most full small SVG rows

## Decision

This prompt interface is a DSL.

The correct next move is:
1. keep the cleaned tokenizer corpus and current fit audit
2. stop relying on BPE alone to discover the control language
3. promote the canonical control tags to reserved/protected tokens

Weighting the tag-seed rows harder is still useful, but it is secondary.

For this task, the interface atoms should be explicit.

## Operational recommendation

Proceed with the following mental model:
- tokenizer defines the atoms of the control interface and output language
- pretrain teaches the SVG/output manifold
- midtrain teaches structured transformations
- SFT teaches obedience to the prompt DSL
- later stages sharpen behavior, but they do not fix a weak interface contract

That is the representation-first view to carry into `svg`, `c`, `sql`, and similar specialist datasets.
