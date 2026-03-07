# spec03 Representation Worksheet

## Purpose

This worksheet turns the representation-first mental model into a concrete planning tool for `spec03`.

Use it to decide:

- which atoms matter
- what each atom should mean
- where the current setup is weak
- what tokenizer and data changes should strengthen it
- how to evaluate whether the representation generalized

Related notes:

- `version/v7/reports/SPEC03_TOKENIZER_PROMPT_ATOMS_ANALYSIS.md`
- `version/v7/reports/REPRESENTATION_FIRST_DATASET_DESIGN.md`

## Current Snapshot

Based on the current local `spec03` bootstrap artifacts:

- prompt-seed rows are only `48 / 731` corpus lines
- prompt-seed rows are only about `0.293%` of tokenizer corpus bytes
- canonical bracketed prompt atoms appear as exact learned pieces `0 / 50`
- the current tokenizer strongly learns SVG/XML chunks
- the older SVG stage-B run produced valid SVG, but weak semantic diversity and weak control

This means the current priority is not "more data" in the abstract.

The priority is:

- strengthen weak prompt/control representations
- normalize geometry better
- keep the output language expressive
- add evaluation that checks controllability, not just syntax validity

## How To Use This Worksheet

For each atom:

1. decide whether it is important for controllability or output quality
2. check whether the tokenizer exposes it well
3. check whether the training data teaches it consistently
4. decide which stage should strengthen it most
5. add a concrete evaluation check

## Atom 1: Family Tags

Examples:

- `[circle]`
- `[rect]`
- `[line]`
- `[bar-chart]`
- `[infographic]`
- `[flow]`

Desired meaning:

- specify the coarse output family
- bias the model toward the right structural template

Current weakness:

- canonical bracketed forms are not exact learned tokenizer pieces
- some unbracketed terms appear, but that is not enough for a stable closed contract

Tokenizer intervention:

- make each canonical family tag an exact piece, or explicitly protect it

Data intervention:

- atomic family rows
- family + one modifier rows
- family + full canonical prefix rows
- bridge rows where the same family appears under several layouts and palettes

Main stage:

- tokenizer
- bridge/midtrain
- SFT

Evaluation:

- held-out prompts for each family tag
- check whether the output family changes correctly when only the family tag changes

## Atom 2: Palette Tags

Examples:

- `[palette:cool]`
- `[palette:warm]`
- `[palette:neutral]`
- `[palette:dark]`

Desired meaning:

- control palette family, contrast mood, and likely color-role choices

Current weakness:

- bracketed palette atoms are not exact pieces
- current tokenizer is learning many raw SVG color contexts but not enough prompt-side palette control

Tokenizer intervention:

- each palette tag should be an exact piece

Data intervention:

- many repeated good palette examples
- same structure rendered under multiple palette tags
- contrastive rows where palette changes but layout and family stay fixed

Main stage:

- tokenizer
- bridge/midtrain
- SFT

Evaluation:

- change only palette tag, keep other tags fixed
- verify color family changes while structure remains mostly stable

## Atom 3: Style Tags

Examples:

- `[style:minimal]`
- `[style:gradient]`
- `[style:filled]`
- `[style:outlined]`

Desired meaning:

- control rendering style, not just one local token
- influence fill/stroke, gradients, noise level, decoration level

Current weakness:

- `gradient` appears in the vocab mainly because of SVG content, not because of bracketed style control
- exact bracketed style atoms are missing

Tokenizer intervention:

- promote exact bracketed style tags

Data intervention:

- paired rows that hold family/layout fixed and vary only style
- examples where the same composition appears in minimal vs gradient vs outlined form

Main stage:

- tokenizer
- bridge/midtrain
- SFT

Evaluation:

- style-only tag swaps
- verify style changes without breaking family/layout semantics

## Atom 4: Layout Tags

Examples:

- `[layout:center]`
- `[layout:grid]`
- `[layout:horizontal]`
- `[layout:stacked]`
- `[layout:flow]`

Desired meaning:

- control spatial organization and composition pattern

Current weakness:

- exact bracketed layout atoms are not learned cleanly
- geometry learning is likely noisy because raw numeric conventions vary a lot

Tokenizer intervention:

- exact layout tags

Data intervention:

- same content under several stable layouts
- normalized geometry conventions
- many contrastive examples where only layout changes

Main stage:

- tokenizer
- bridge/midtrain
- SFT

Evaluation:

- held-out layout prompts
- visual or structural checks for grid, stacked, horizontal, and flow organization

## Atom 5: Chart Modifier Tags

Examples:

- `[axes]`
- `[labeled]`
- `[trend-line]`
- `[values]`
- `[ascending]`
- `[descending]`
- `[mixed]`
- `[bars:5]`
- `[points:6]`

Desired meaning:

- control specific chart substructure
- affect counts, ordering, labels, overlays, and annotation

Current weakness:

- these are exactly the kinds of control atoms that become weak if fragmented
- current tokenizer does not promote them as exact bracketed atoms

Tokenizer intervention:

- exact pieces for chart modifiers
- keep `][` as an exact piece too

Data intervention:

- many small synthetic chart rows
- paired examples where one modifier changes at a time
- exact count supervision for bars/points

Main stage:

- tokenizer
- bridge/midtrain
- SFT

Evaluation:

- modifier-swap tests
- count accuracy tests
- ordering tests

## Atom 6: SVG Root and Closure

Examples:

- `<svg`
- `</svg>`
- `viewBox`
- `width`
- `height`
- `<eos>`

Desired meaning:

- stable document start
- stable closure
- stable canvas definition

Current weakness:

- SVG root pieces are learned well
- literal `<eos>` is not cleanly aligned with tokenizer special tokens

Tokenizer intervention:

- decide one EOS contract and keep it consistent
- either exact `<eos>` piece or tokenizer-native EOS only

Data intervention:

- canonical root formatting
- canonical closure formatting
- stable viewBox conventions

Main stage:

- tokenizer
- pretrain

Evaluation:

- parse rate
- closure rate
- EOS consistency checks

## Atom 7: Grouping and Hierarchy

Examples:

- `<g>`
- nested groups
- defs + usage structure
- marker references
- filter references

Desired meaning:

- compose multi-part structures
- share visual definitions
- support nested layouts and reusable motifs

Current weakness:

- current tokenizer learns many XML chunks, but hierarchy understanding depends heavily on data consistency
- nested composition likely remains under-taught compared with simple local primitives

Tokenizer intervention:

- no need to force too many exact large hierarchy blobs
- preserve useful smaller reusable SVG structure tokens

Data intervention:

- more normalized nested-group examples
- bridge examples for multi-panel and nested infographic structure
- keep hierarchy patterns consistent across asset families

Main stage:

- pretrain
- bridge/midtrain

Evaluation:

- held-out nested composition tasks
- structural depth checks
- visual coherence on multi-panel outputs

## Atom 8: Gradient and Filter Structure

Examples:

- `linearGradient`
- `stop`
- `stop-color`
- `filter`
- `feDropShadow`

Desired meaning:

- support richer visual polish and aesthetic identity

Current weakness:

- these are actually learned strongly already
- risk is not absence, but over-dominance relative to prompt/control atoms

Tokenizer intervention:

- keep useful reusable gradient/filter pieces
- avoid letting them consume too much merge budget at the expense of control atoms

Data intervention:

- many good examples of gradients and shadows
- style-conditioned rows so these become controllable rather than merely frequent

Main stage:

- pretrain
- bridge/midtrain

Evaluation:

- style-conditioned gradient usage
- visual richness checks without collapse into one repeated pattern

## Atom 9: Colors

Examples:

- hex colors
- gradient stop colors
- background/foreground contrast
- palette-role relationships

Desired meaning:

- aesthetically coherent color combinations
- prompt-conditioned palette behavior

Current weakness:

- color may be learnable from examples, but palette control is currently weak
- raw color strings can become sparse and noisy if not normalized

Tokenizer intervention:

- exact prompt-side palette tags
- keep raw color formatting consistent

Data intervention:

- many examples of strong palette families
- stable palette-role patterns
- same structure rendered under several curated palettes

Main stage:

- pretrain
- bridge/midtrain
- SFT

Evaluation:

- prompt-side palette swap tests
- held-out aesthetic judgments or render comparisons

## Atom 10: Numeric Geometry

Examples:

- `x`
- `y`
- `width`
- `height`
- `cx`
- `cy`
- `rx`
- `ry`
- `offset`
- `points`

Desired meaning:

- encode coherent geometry, spacing, alignment, proportion, and composition

Current weakness:

- numbers are context-dependent
- noisy raw coordinates can teach weak habits instead of reusable geometry
- over-blobbing attribute strings with numbers hides controllable structure

Tokenizer intervention:

- keep attributes and numeric content separable enough
- avoid giant blobs that fuse many geometry fields together

Data intervention:

- normalize canvas and viewBox conventions
- consistent decimal precision
- reduce unnecessary transform complexity
- prefer stable coordinate conventions
- optionally use relative or bounded numeric formats where possible

Main stage:

- pretrain
- bridge/midtrain

Evaluation:

- alignment checks
- spacing checks
- held-out geometry composition tests

## Atom 11: Text Role Placeholders

Examples:

- `TITLE_A`
- `SUBTITLE_A`
- `LABEL_1`
- `PARA_A`

Desired meaning:

- preserve layout role without forcing memorized prose

Current weakness:

- placeholders exist, which is good
- but the model still needs enough text-heavy structured examples to learn where and how those roles appear

Tokenizer intervention:

- exact placeholder pieces should exist cleanly

Data intervention:

- enough text-heavy normalized assets
- repeated role-consistent placement patterns

Main stage:

- pretrain
- bridge/midtrain

Evaluation:

- held-out text-heavy structure tests
- verify placeholder placement and role consistency

## Atom 12: Prompt Boundary Glue

Examples:

- `][`

Desired meaning:

- support compact parsing of multi-tag prefixes

Current weakness:

- `][` exists, which is good
- but the surrounding bracketed atoms do not yet exist as exact pieces

Tokenizer intervention:

- keep `][`
- add exact bracketed prompt atoms

Data intervention:

- many prompt-prefix rows with controlled variation

Main stage:

- tokenizer

Evaluation:

- prompt token count should shrink materially for canonical prefixes

## Atom 13: Multi-Panel / High-Composition Structures

Examples:

- infographic cards
- dashboards
- flow diagrams
- grouped technical diagrams

Desired meaning:

- support beautiful, coherent, nontrivial SVG beyond single primitive shapes

Current weakness:

- old runs learned valid syntax but often collapsed to simple repeated motifs
- richer composition was under-taught relative to basic syntax

Tokenizer intervention:

- not mainly a tokenizer problem
- tokenizer just must not hide structural cues

Data intervention:

- more normalized multi-panel assets
- bridge tasks that reconstruct or transform complex compositions
- holdout evaluation on family-level composition

Main stage:

- pretrain
- bridge/midtrain
- SFT

Evaluation:

- family holdout render quality
- compositional stress tests
- repetition-collapse checks

## Priority Order

### Priority 0

- fix prompt weighting in tokenizer corpus generation
- add tokenizer coverage audit for canonical prompt atoms
- clean up EOS contract

### Priority 1

- strengthen family, palette, style, layout, and chart-modifier atoms
- add prompt-prefix bridge data

### Priority 2

- normalize numeric geometry more aggressively
- strengthen multi-panel and nested structure examples

### Priority 3

- refine palette-role learning and aesthetic consistency
- add richer holdout evaluations for controllability and composition

## Minimal Next Actions

1. Patch tokenizer corpus construction so repeated prompt rows actually survive.
2. Expand prompt-seed corpus from tens of rows to low thousands of effective rows.
3. Add exact-piece audit for canonical bracketed tags.
4. Add a small bridge dataset:
- same structure under several palettes
- same structure under several layouts
- same chart with one modifier changed at a time
5. Normalize geometry more aggressively before the next serious `spec03` run.

## Final Principle

The goal is not just to expose tokens.

The goal is to make each important atom learn:

- stable local meaning
- useful interactions
- controllable output effects
- enough invariance to generalize

If an atom does not learn those properties, adding more random data will not fix the problem reliably.
