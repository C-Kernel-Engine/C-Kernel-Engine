# SVG DSL Production Runbook

This note defines the practical workflow for the infographic line after the first asset-grounded scene runs.

Canonical docs-site checklist:

- [spec-training-method.html](/home/antshiv/Workspace/C-Kernel-Engine/docs/site/_pages/spec-training-method.html)

Use the docs-site page as the operator-facing checklist for future spec/run reviews.
This repo note stays as the source-controlled companion for deeper implementation detail.

The goal is no longer "see if a small model can learn any SVG-like contract."

The goal is:

- derive a production DSL from the assets we actually want
- keep DSL and content/data separate
- make the compiler the deterministic boundary between them
- train and evaluate in a way where each run teaches us something concrete

This is the methodical path from exploratory specs to a production-oriented system.

## Core Thesis

`DSL != data`

The DSL is structure.
The data is content.
The compiler blends the two into final SVG.

That means a production infographic stack should look like:

`prompt -> scene DSL`

`content system -> content JSON`

`scene DSL + content JSON -> deterministic compiler -> SVG`

The model should primarily learn structure, composition, and style choice.
It should not memorize final prose, numbers, or one-off asset-specific sentences as part of the structure language.
For strict visual-language lines, the model-facing input should also stay
domain-agnostic: feed family/control wording only, and route topic facts
through upstream systems plus external `content.json`.

## What The Recent Runs Proved

The recent asset-grounded scene runs proved something useful:

- a richer scene/compiler path can train
- the model can learn asset-grounded scene structure
- the line can get very strong contract accuracy

Those runs also exposed the next constraint:

- some current component rows still carry literal content inside the DSL
- that is acceptable for proving the compiler boundary
- it is not the final production architecture

So the next line should shift from:

- `component token + literal prose/value payload`

to:

- `component token + semantic content key`

Example:

Bad long-term target:

```text
[section_card:Memory_Capacity|25x_more_memory_capacity|Capacity_sets_the_real_context_boundary|variant=metric|accent=green]
```

Better target:

```text
[section_card:key=memory_capacity|variant=metric|accent=green]
```

with:

```json
{
  "memory_capacity": {
    "title": "Memory Capacity",
    "value": "25x more memory capacity",
    "caption": "Capacity sets the real context boundary"
  }
}
```

That is the split we want to train toward.

## Two Different Training Problems

There are really two different systems:

### 1. DSL Model

Learns:

- scene family
- composition
- component presence/order
- theme/tone/density choice
- emphasis and hierarchy
- slot/key placement

Should not learn:

- final prose
- exact numeric facts
- one-off business claims
- raw SVG geometry

### 2. Content/Data System

Supplies:

- text
- values
- labels
- captions
- optional per-component metadata

Can come from:

- curated JSON libraries
- templates
- retrieval
- another model
- human-authored source data

The compiler is the stable meeting point between these two systems.

## Backward Workflow

This should now be the default workflow for every new spec line.

### Phase 0: Define The Production Boundary

Before any training:

1. Pick the real shipped assets we want to get close to in `docs/site/assets/`.
2. Decide which visual families matter now.
3. Decide what "good enough to train" means.

Recommended initial families:

- `poster_stack`
- `comparison_span_chart`
- `pipeline_lane`
- `dual_panel_compare`
- `dashboard_cards`
- `table_analysis`

Recommended initial production criteria:

- compiler can render all gold cases
- the result is recognizably in the same visual language
- placeholders wrap and fit
- no layout family depends on literal prose baked into the DSL

### Phase 1: Build A Dummy-Text Asset Mirror

Do not modify `docs/site/assets/` directly.

Instead:

1. mirror chosen assets into the run workspace
2. replace literal text with placeholder content
3. keep composition, shapes, grouping, and style intact

The purpose is to answer:

- what is structural?
- what is content?
- what must the compiler own?

Good placeholder conventions:

- `heading_1`
- `heading_2`
- `label_1`
- `value_1`
- `caption_1`
- `paragraph_1`
- `note_1`

For charts/tables:

- `series_a`
- `series_b`
- `row_1_label`
- `row_1_value`
- `axis_title`

Important:

- placeholder text should preserve approximate line-length classes
- if the real asset has short labels vs long captions, keep that distinction
- use multiple placeholder lengths to exercise wrapping

### Phase 2: Extract Asset Vocabulary

From the mirrored assets, extract four separate layers:

#### A. Scene Families

Examples:

- `poster_stack`
- `pipeline_lane`
- `comparison_span_chart`
- `table_analysis`

#### B. Components

Examples:

- `header_band`
- `section_card`
- `compare_bar`
- `table_header`
- `table_row`
- `compare_panel`
- `callout_card`
- `phase_divider`
- `stage_card`
- `flow_arrow`
- `curved_connector`
- `legend_row`
- `axis`
- `annotation`

#### C. Theme / Style Tokens

Examples:

- `theme:infra_dark`
- `theme:paper_editorial`
- `theme:signal_glow`
- `tone:amber|green|blue|purple|mixed`
- `density:compact|balanced|airy`
- `rail:accent|muted|none`
- `background:grid|rings|mesh|none`
- `connector:line|arrow|bracket`

#### D. Compiler-Owned Primitives

These stay below the DSL:

- `rect`
- `path`
- `line`
- `polygon`
- `defs`
- `linearGradient`
- `filter`
- `marker`
- `tspan`

This layer belongs to the compiler, not the model.

### Phase 3: Prove The Compiler First

Before training:

1. hand-map gold assets into the DSL
2. compile them with dummy content
3. compare compiled output against the mirrored dummy-text assets

Do not train until this works.

This is the key discipline change.

If the compiler cannot produce the target visual language from the DSL, training is premature.

Treat this as a hard gate, not a soft note.

Minimum standard before a serious run:

- 5 to 10 gold assets, not just 2 or 3 hero examples
- at least one table-heavy asset
- at least one pipeline/flow asset
- at least one comparison/poster asset
- a side-by-side compiled-vs-reference HTML report
- explicit operator judgment that the outputs are visibly in-family

### Phase 4: Freeze A Canonical DSL

The DSL must be:

- small
- ordered
- canonical
- parseable

Rules:

- fixed attribute order
- fixed component order where possible
- no stylistic aliases
- no raw coordinates
- no raw color hex
- no literal production prose in the structure layer

The DSL should encode:

- structure
- roles
- style family
- content keys

not literal asset copy.

After the keyed structure/content split is proven, token granularity becomes its own explicit step.

Do not leave whole component rows as the permanent atomic token surface if they still look like:

```text
[compare_bar:@compare_bar.0.label|@compare_bar.0.value|@compare_bar.0.caption|accent=amber|note=@compare_bar.0.note]
```

Move toward a smaller compositional grammar such as:

```text
[compare_bar]
[label_ref:compare_bar.0.label]
[value_ref:compare_bar.0.value]
[caption_ref:compare_bar.0.caption]
[accent:amber]
[note_ref:compare_bar.0.note]
[/compare_bar]
```

That should be a dedicated spec question, not an incidental side effect of some larger run.

### Phase 5: Build The Training Bundle

Every compiler-backed training example should be a structured bundle:

1. `scene.dsl`
2. `content.json`
3. `compiled.svg`
4. `metadata.json`

Suggested metadata:

```json
{
  "family": "poster_stack",
  "theme": "infra_dark",
  "tone": "green",
  "density": "compact",
  "asset_source": "memory-reality-infographic.svg",
  "variant_kind": "dummy_asset_reconstruction",
  "tags": ["gold", "dsl_only", "compiler_validated"]
}
```

### Phase 6: Train The DSL Line

For the DSL model:

- train on scene structure plus content keys
- do not train on literal production content rows as reserved control tokens
- keep placeholder content external
- add negative repair rows, not only positive rows

A good target looks like:

```text
[scene]
[canvas:tall]
[layout:poster_stack]
[theme:infra_dark]
[tone:green]
[density:compact]
[section_card:key=memory_capacity|variant=metric|accent=green]
[compare_bar:key=gpu_vram|accent=red]
[compare_bar:key=cpu_ram|accent=green]
[/scene]
```

not:

```text
[section_card:Memory_Capacity|25x_more_memory_capacity|Capacity_sets_the_real_context_boundary|variant=metric|accent=green]
```

The negative repair rows should include more than closing-tag fixes.

At minimum, teach:

- wrong layout -> corrected layout
- wrong component order -> corrected order
- missing required component -> corrected scene
- wrong theme/tone -> corrected scene
- parseable but semantically wrong scene -> corrected scene

### Phase 7: Add The Content System

Once the DSL line is stable:

- keep content external
- expand the content library
- optionally train or build a separate content generator
- preserve compiler determinism

This keeps the problem modular:

- structure model
- content system
- deterministic compiler

## What To Tokenize

Tokenize:

- scene grammar
- component names
- content keys
- theme/style tokens
- canonical attribute/value enums

Do not reserve entire literal component rows when those rows contain asset-specific prose or numbers.

Bad:

- one token for a full card with literal text

Good:

- token for `section_card`
- token for `variant=metric`
- token for `accent=green`
- token for `key=memory_capacity`

Best long-term target:

- smaller reserved grammar tokens
- stable enums and field refs
- fewer one-off component strings
- enough compositional reuse that similar scenes share most of their structure

## What A Real Run Must Prove

Every real spec/run should answer one explicit question.

Bad run goal:

- "try a bunch of stuff and see what happens"

Good run goal:

- "does keyed content improve generalization over literal component payloads?"
- "does the compiler reproduce all dummy gold assets for table-heavy layouts?"
- "does canonical ordering remove duplicate theme/tone drift?"
- "does smaller token granularity preserve renderability while improving recombination?"

If the run does not have a sharp question, it is too vague to teach us much.

## Run Design Template

For each spec or run, define:

### Hypothesis

One sentence.

Example:

`Replacing literal component payloads with content keys will preserve renderability while improving generalization.`

### Change Surface

What changed:

- DSL grammar
- compiler
- dataset materializer
- tokenizer policy
- budget
- evaluation

### Non-Goals

What is intentionally not being changed.

### Success Criteria

Examples:

- `100%` compiler validity on dummy gold assets
- `100%` renderability on probe
- improved exact/materialized exact without regressions in layout families
- no literal production prose inside reserved control tokens
- content-binding success on held-out payloads
- acceptable asset-parity score on gold assets

### Failure Interpretation

Predefine what each failure means:

- parse failure -> grammar or decoder hygiene
- render failure -> compiler or DSL contract problem
- exact miss but render match -> canonicalization problem
- render drift -> true semantic/layout problem

## Standard Post-Run HTML Report

Every real run should emit one HTML report into the cache reports bucket or run dir.

Suggested path:

- run-local: `~/.cache/ck-engine-v7/models/train/<run>/specXX_probe_report.html`
- cross-run: `~/.cache/ck-engine-v7/models/reports/<family>_progress_report_<date>.html`

The report should always include:

1. Run summary
   - hypothesis
   - run dir
   - commit/script version if relevant

2. Scorecard
   - exact
   - renderable
   - materialized exact
   - content-binding success
   - asset-parity / gold reconstruction status
   - delta vs prior baseline

3. Dataset / training config
   - row counts
   - packed one-epoch budgets
   - effective epochs
   - what changed from prior run

4. Layout-by-layout breakdown
   - per family exact/render/materialized

5. Example gallery
   - prompt
   - expected DSL
   - model DSL
   - content JSON or content keys
   - compiled SVG
   - expected/dummy/reference SVG

6. Failure taxonomy
   - parse failures
   - render failures
   - content-binding failures
   - exact-only misses
   - real semantic drifts

7. What worked
   - keep list of wins, not just failures

8. What did not work
   - concrete regressions

9. Decision
   - promote
   - reject
   - repair

10. Next run recipe
   - explicit targeted changes

11. Frozen surfaces
   - what must not change in the next run so the lesson stays interpretable

If a run cannot produce this report, it is not yet a disciplined run.

## Methodical Training Intuition

The goal is to make each run part of a learning sequence.

Good intuition comes from repeated cycles of:

1. hypothesize
2. isolate one change
3. run
4. inspect by slice
5. decide what the result means
6. write the next run recipe

This is how we avoid random experimentation.

A run should teach one of:

- the DSL is missing a concept
- the compiler is missing a capability
- the data separation is wrong
- the budgets are wrong
- the grammar is too loose
- the eval is wrong

If a run teaches none of these clearly, the run was poorly scoped.

## Recommended Next Specs

After `spec10`, a practical path is:

### `spec11`

Goal:

- remove literal content-bearing component payloads from the DSL line
- move to `component + content key`

Deliverables:

- keyed DSL grammar
- keyed content JSON schema
- compiler support for keyed content resolution
- dummy-text gold asset reconstructions

### `spec12`

Goal:

- train the DSL model on structure-only scenes with keyed content

Deliverables:

- `scene.dsl + content.json + compiled.svg + metadata.json`
- placeholder-only tokenizer policy
- compiler-validated canary
- cross-run HTML progress report

### `spec13`

Goal:

- bridge from explicit scene-control prompts to intent-driven scene planning
- keep content generation separate from scene planning
- for future strict visual-language lines, keep domain-bearing request text out
  of the model-facing prompt surface

Recommended first step:

- `spec13a`: semi-structured intent prompts with `topic + goal + audience`, but
  no explicit `layout`, `theme`, `tone`, or `density`
- keep `content.json` external and the compiler deterministic
- measure layout-family choice and inferred-field accuracy before introducing
  freer language

See:

- [SPEC13A_INTENT_PROMPT_BRIDGE_2026-03-18.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC13A_INTENT_PROMPT_BRIDGE_2026-03-18.md)

Deliverables:

- `spec13a`: intent-prompt bridge with bounded ontology
- `spec13b`: generalized scene IR / renderer with deterministic layout
- `spec13c`: assistant / tool shell around the scene planner
- optional separate content-generation line later
- production-style scene + content rendering tests

## Bottom Line

Yes, the proposed method makes sense, with one important correction:

- do not train on literal asset copy as if it were the final DSL
- train on structure and content keys
- keep the content system separate
- prove the compiler with dummy assets first

The workflow should now be:

1. choose target assets
2. create dummy-text mirrors
3. extract scene/component/style vocabulary
4. prove compiler fidelity on gold cases
5. freeze canonical DSL
6. tokenize only the structure vocabulary plus content keys
7. train the DSL line
8. report every run in the same way
9. use each run to answer one explicit question

That is the path from experimentation to a method.
