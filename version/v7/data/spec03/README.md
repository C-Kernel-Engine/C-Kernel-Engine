# spec03

`spec03` is the next clean SVG data workspace for v7.

Use this directory while the data contract, corpus shape, and eval rules are still evolving.
Do not move this into a separate repo or submodule yet. That would make iteration slower
without solving a real problem.

## Why `spec03` exists

`spec02` proved that the model can learn valid SVG syntax, but it did not reliably learn:

- prompt/tag adherence
- richer infographic structure
- target-asset grammar from `docs/site/assets/*.svg`

`spec03` is the clean restart point for:

- a closed-vocabulary tag contract
- cleaner dataset families
- stronger canary evaluation before long training runs

## Canonical format

All supervised rows in `spec03` should use the same tag-prefix contract:

```text
[circle][palette:cool][style:minimal][layout:center]<svg ...>...</svg><eos>
```

Rules:

- Do not mix `<task>...</task><svg ...>` rows into `spec03` corpora.
- Do not introduce free-form tag spellings or synonyms.
- Keep prompt vocabulary closed and versioned.
- Output is pure SVG plus `<eos>`.

## Folder layout

- `contracts/`
  - tag vocabulary, eval contract, and schema notes
- `raw_assets/`
  - source-of-truth imports with provenance preserved
- `normalized/`
  - normalized + placeholderized + dedupe-ready SVG assets
- `pretrain/`
  - pretrain-ready corpora
- `midtrain/`
  - transform/edit/style-conditioning corpora
- `sft/`
  - strict tag->SVG supervised corpora
- `holdout/`
  - fixed evaluation/canary assets and prompt sets
- `tokenizer/`
  - tokenizer corpus manifests and fit reports
- `manifests/`
  - derived inventory, dedupe, and coverage summaries

## Data-family plan

## Raw vs normalized

`raw_assets/` is not split by training stage.

Purpose:

- preserve the original imported files
- preserve source provenance
- allow re-normalization if policy changes

Do not copy the same file into separate raw `pretrain/`, `midtrain/`, and `sft/` buckets.

Instead:

1. import into `raw_assets/`
2. normalize into `normalized/`
3. derive stage-specific corpora from `normalized/`

`normalized/` is the cleanup layer. It is where we:

- extract the SVG root cleanly
- strip obvious editor noise/comments
- replace human prose with placeholders like `TITLE_A`, `PARA_A`, `LABEL_1`
- compute dedupe identity on the cleaned result

Only after that should we classify rows into:

- `pretrain/`
- `midtrain/`
- `sft/`
- `holdout/`

### Pretrain

Teach SVG grammar and reusable visual structure:

- small full SVGs that fit the chosen context
- panel-level chunks from larger assets
- structural groups (`<g>...</g>`)
- defs/gradient/marker/filter blocks
- repair/continuation snippets

### Midtrain

Teach controlled structural edits:

- theme swap
- add/remove labels
- enrich/simplify layout
- panel restyling

### SFT

Teach strict specialist rendering:

- input: closed-vocab tag prefix
- output: pure SVG only
- no mixed conditioning

## Text policy

Replace human prose with placeholders before training:

- `TITLE_A`
- `SUBTITLE_A`
- `LABEL_1`
- `PARA_A`
- `AXIS_X`
- `AXIS_Y`

This keeps the model focused on composition and SVG behavior instead of memorizing English.

## Asset sources

Primary target style:

- `docs/site/assets/*.svg`

Curated supplemental sources:

- `../BCGov/ai-hub-tracking/docs/assets/*.svg`
- `../BCGov/citz-imb-ai/assets/*.svg`
- `../BCGov/ai-technical-solutions-research/docs/assets/*.svg`

These should be imported deliberately with inventory + dedupe summaries, not blindly pooled.

## Context strategy

Do not assume full production infographics fit as single training rows.

Current guidance:

- mainline branch: chunked data for `ctx=512` or `ctx=2048`
- larger-context branch: test `ctx=2048` first, then `ctx=4096` only if needed

## When to move to a separate repo/submodule

Only split `spec03` out when all of these are true:

1. the tag contract is stable
2. the folder layout is stable
3. the eval contract is stable
4. asset ingestion and dedupe rules are stable
5. the dataset becomes large enough or private enough that this repo should not carry it directly

Until then, keep `spec03` in-tree so scripts, reports, and evaluation evolve together.
