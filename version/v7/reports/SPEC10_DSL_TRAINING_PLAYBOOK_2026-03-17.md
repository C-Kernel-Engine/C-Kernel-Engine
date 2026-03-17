# Spec10 DSL Training Playbook

This note captures the practical training approach for the infographic line after the exploratory `spec06 -> spec08` phase.

The older line answered:

- can a small model learn a closed infographic contract at all?
- can a scene-like representation outperform flat atoms?
- can a richer compiler improve renderability and contract behavior?

The `spec09 -> spec10` line answers a different question:

- how do we train a useful production-oriented infographic planner that emits a stable scene DSL and relies on a deterministic compiler for the fragile visual work?

That is a different objective, and it requires a different workflow.

## Core Principle

Do not start from a synthetic training grid and hope it grows into production visuals.

Start from the production visuals we actually want, derive the DSL from them, prove the compiler can render them, and only then train.

The intended split is:

- model chooses scene family, component order, emphasis, theme, and content roles
- external content/data source supplies final text and values
- compiler resolves both into final SVG

In other words:

`prompt -> scene DSL`

`content system -> content JSON`

`scene DSL + content JSON -> deterministic compiler -> SVG`

## Why This Is More Practical

The older flat-atom and early-scene runs were useful, but they still trained against a target that was simpler than the site assets.

That creates two bad failure modes:

- the model gets better at producing a weak target
- we spend compute improving exact-match on a representation that still cannot produce the visuals we want to ship

The new workflow prevents that by making compiler expressivity a prerequisite for training.

## Run Placement

Every training run is part of the experiment ledger and should stay in the shared cache tree.

- Training runs belong under `~/.cache/ck-engine-v7/models/train/<run-name>`.
- Cross-run monitors, progression logs, and generated dashboards belong under `~/.cache/ck-engine-v7/models/reports/`.
- `version/v7/reports/` stays for curated, source-controlled notes only.

Do not launch one-off training jobs into bespoke repo folders like `version/v7/runs/`. If a run teaches us something, it should land where `open_ir_hub.py` and the rest of the artifact tooling can discover it later.

Why this is mandatory:

- the operator can view the whole training history in one shared cache tree
- IR hub and generated reports can discover runs without custom path knowledge
- datasets, checkpoints, probes, telemetry, and HTML stay co-located per run
- we avoid fragmented “private” experiment folders that hide what the system already learned

If automation ever backfills a missing probe report after a failed launch, treat that probe as diagnostic only.
It is useful for triage, but it is not a promotable run result and should not be compared against baselines as if the training run completed normally.

## Training Contract

The training target should no longer be "raw SVG atoms" and should not be "scene DSL with hardcoded prose only."

The target should be a structured bundle:

1. `scene.dsl`
2. `content.json`
3. `compiled.svg`
4. `metadata.json`

Where:

- `scene.dsl` is canonical and model-facing
- `content.json` contains final text, numbers, labels, and optional per-component values
- `compiled.svg` is deterministic output from the compiler
- `metadata.json` captures family/theme/tone/density/source-asset/probe tags

## What the Model Should Learn

The model should learn:

- scene family selection
- component presence and ordering
- composition hints such as `hero`, `columns`, `rail`, `background`, `connector`
- slot and role binding
- high-level style family choice

The model should not learn:

- raw SVG coordinates
- gradient stop definitions
- marker ids
- filter ids
- path control points
- font sizes in pixels
- line wrapping policy

Those stay compiler-owned.

## Data Shape

### Scene DSL

Example:

```text
[scene]
[canvas:wide]
[layout:comparison_span_chart]
[theme:infra_dark]
[tone:mixed]
[frame:card]
[density:balanced]
[inset:md]
[gap:md]
[hero:center]
[columns:3]
[emphasis:center]
[rail:none]
[background:mesh]
[connector:bracket]
[topic:live_balance]
[header_band:@title.kicker|@title.headline|@title.subtitle]
[compare_bar:@bars.gpu.label|@bars.gpu.value|@bars.gpu.caption|accent=@bars.gpu.accent|note=@bars.gpu.note]
[compare_bar:@bars.cpu.label|@bars.cpu.value|@bars.cpu.caption|accent=@bars.cpu.accent|note=@bars.cpu.note]
[thesis_box:@thesis.title|@thesis.line_1|@thesis.line_2]
[conclusion_strip:@summary.conclusion]
[footer_note:@summary.footer]
[/scene]
```

### Content JSON

Example:

```json
{
  "title": {
    "kicker": "live brief",
    "headline": "Compiler-Bound Scene + Data",
    "subtitle": "Structure and data can arrive from separate systems."
  },
  "bars": {
    "gpu": {
      "label": "GPU Path",
      "value": "67,000 GB/s eq",
      "caption": "5,360x span",
      "accent": "amber",
      "note": "HBM is fast but capacity bound."
    },
    "cpu": {
      "label": "CPU Path",
      "value": "1,800 GB/s eq",
      "caption": "144x span",
      "accent": "green",
      "note": "Fit and cost are closer to deployment reality."
    }
  },
  "thesis": {
    "title": "Compiler + content is the right split.",
    "line_1": "The model chooses structure and emphasis.",
    "line_2": "Another system can provide final data and copy."
  },
  "summary": {
    "conclusion": "The compiler renders the full infographic, not just a skeleton.",
    "footer": "This is the bridge from planning to production content."
  }
}
```

### Metadata JSON

Suggested fields:

```json
{
  "family": "comparison_span_chart",
  "theme": "infra_dark",
  "tone": "mixed",
  "density": "balanced",
  "asset_source": "performance-balance.svg",
  "variant_kind": "gold_reconstruction",
  "tags": ["gold", "compiler_validated", "asset_bridge"]
}
```

## Backward Workflow

### Phase A: Asset Grounding

1. Scan `docs/site/assets/*.svg`.
2. Extract recurring scene families, components, and style/effect patterns.
3. Build an asset library.
4. Pick a small gold set that covers multiple families.

Current scripts:

- [build_spec09_asset_library_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/build_spec09_asset_library_v7.py)
- [build_spec09_asset_alignment_report_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/build_spec09_asset_alignment_report_v7.py)
- [build_spec09_compiler_validation_report_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/build_spec09_compiler_validation_report_v7.py)
- [render_svg_structured_scene_spec09_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/render_svg_structured_scene_spec09_v7.py)

### Phase B: DSL Design

1. Write canonical scene grammar.
2. Keep field order fixed.
3. Add component vocabulary only when real assets require it.
4. Prefer semantic tokens over raw geometry.

See:

- [SPEC09_SCENE_DSL_V2_GRAMMAR_2026-03-17.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC09_SCENE_DSL_V2_GRAMMAR_2026-03-17.md)

### Phase C: Compiler Acceptance

1. Hand-map gold assets into the DSL.
2. Compile them back to SVG.
3. Inspect compiled-vs-real side by side.
4. Refine compiler until the outputs are directionally acceptable.

Do not generate training data before this phase is acceptable.

### Phase D: Variant Generation

Once the gold reconstructions are good enough, generate controlled variants:

- topic swap
- content/data swap
- tone swap
- density swap
- emphasis swap
- component-preserving composition variations

Important rule:

- keep the compiled output visually inside the same design language as the gold assets

### Phase E: Training Corpora

Build `spec10` data from:

- gold asset reconstructions
- controlled synthetic variants around those reconstructions
- compiler-validated scene/content pairs only

Every row should have:

- canonical prompt
- target `scene.dsl`
- target `content.json`
- deterministic compiled SVG
- family/style metadata

### Phase F: Preflight and Canary

Before any serious training run:

1. compute packed token budgets from the actual dataset and tokenizer
2. run full-catalog parse/compile/render checks
3. run a small balanced canary
4. reject undersized budgets in strict mode

See:

- [DETERMINISTIC_PREFLIGHT_ADAPTERS.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/DETERMINISTIC_PREFLIGHT_ADAPTERS.md)

## Suggested Spec10 Dataset Layout

One workable layout is:

```text
dataset/
  train/
    case_000001/
      prompt.txt
      scene.dsl
      content.json
      compiled.svg
      metadata.json
  dev/
  test/
```

The materializer can also emit packed JSONL sidecars for training:

```json
{
  "prompt": "Build an infographic comparing CPU vs GPU memory fit.",
  "scene": "[scene] ... [/scene]",
  "content": {"title": {"headline": "..."}, "bars": {...}},
  "metadata": {"family": "dual_panel_compare", "theme": "infra_dark"}
}
```

## Eval Gates

Training should be promoted only if it passes all of these:

### Contract Gates

- DSL parse success
- canonical scene start/stop
- compile success
- SVG/XML validity
- deterministic render success

### Capability Gates

- non-regression by family
- scene exact or canonical-equivalent match
- materialized exact where exact equivalence is appropriate
- held-out content binding success
- renderability must remain `100%`

### Product Gates

- compiled gold-asset reconstructions remain visually acceptable
- human review says outputs are in-family, not generic filler
- theme/style range remains coherent across variants

## When To Start Training

Do not start a serious `spec10` run yet.

The right answer today is:

- yes to continued compiler and dataset work
- yes to tiny smoke/canary runs after the materializer exists
- no to a real capability run until the target is a bit tighter

## What Is Still Missing

Before a serious run, we still need:

1. More gold assets with hand-mapped DSL and real content.
   Six is a good start, but it is still a small target family.
2. A real `spec10` materializer that emits `scene.dsl + content.json + compiled.svg + metadata`.
3. A stable content-binding schema, not just ad hoc JSON examples.
4. Stronger visual acceptance on the gold asset bridge.
   The compiler is directionally better now, but some families are still clearly less sophisticated than the source assets.
5. A `spec10` preflight using the real tokenizer and packed data.
6. A held-out asset/variant eval split.

## Recommendation

The next practical sequence should be:

1. keep expanding and tightening the `spec09` gold asset bridge
2. define the `spec10` materializer contract
3. emit a small real dataset with content JSON included
4. run preflight and a tiny canary
5. only then launch a serious training run

That gives us a much better chance of spending compute on the right target.
