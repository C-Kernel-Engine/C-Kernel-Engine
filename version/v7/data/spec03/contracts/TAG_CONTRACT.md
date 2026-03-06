# spec03 Tag Contract

This model is a specialist tag->SVG renderer.

It is not a general instruction-following LLM.

## Canonical training/inference format

```text
[circle][palette:cool][style:minimal][layout:center]<svg ...>...</svg><eos>
[bar-chart][bars:5][ascending][palette:warm][axes][trend-line]<svg ...>...</svg><eos>
[infographic][palette:dark][style:gradient][labeled][complexity:rich]<svg ...>...</svg><eos>
```

## Hard rules

- No `<task>...</task>` rows in `spec03`.
- No free-form natural-language tags.
- No synonyms for the same concept.
- No mixed conditioning formats in tokenizer or train corpora.

## Vocabulary policy

Only add a tag if all three exist:

1. generator or asset evidence
2. eval support
3. enough corpus frequency to justify tokenizer exposure

If a tag is not supported end-to-end, do not include it in the canonical contract yet.

## Current normalization policy

Prefer these families:

- shape/chart/infographic tags
- count tags
- palette tags
- style tags
- layout tags
- complexity tags

Do not rename tags casually once introduced.

Example:

- keep `style:outlined` if that is the canonical spelling
- do not silently fork to `style:outline`

## Placeholder text policy

Keep layout roles, remove human prose:

- `TITLE_A`
- `SUBTITLE_A`
- `LABEL_1`
- `PARA_A`

## Validation rule

Before tokenizer training or model training, every row in `spec03` corpora should satisfy:

```text
^\s*(?:\[[^\]]+\])+\s*<svg.*</svg><eos>\s*$
```

If not, fail the build.
