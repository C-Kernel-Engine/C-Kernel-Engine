# SVG Capability Evolution Executive Report

This report summarizes the capability line from `spec02` through `spec06`.

The important caveat is that `spec02` and `spec03` used an older raw-SVG legacy probe, while `spec04` onward uses structured probe contracts. So the numbers are not perfectly apples-to-apples. The progression is still clear:

- `spec02` proved the engine could train a raw SVG generator that emitted valid files and partially followed prompt intent.
- `spec03` showed the bootstrap/tokenizer path could fail completely even when training still ran.
- `spec04` solved syntax and renderability with a structured DSL, but semantic control remained weak.
- `spec05` was the first real held-out semantic success on structured scenes.
- `spec06` extended that into infographic template filling with headings, bullets, stat cards, and spectrum layouts, and `r6` is now the best run in the line.

## Executive Read

- `spec02`: first practical raw-SVG success, but with fragile control and an entangled output surface.
- `spec03`: clear failure case; lower loss did not produce usable SVG behavior.
- `spec04`: representation fix worked for syntax, not for composition.
- `spec05`: curriculum plus modest scaling produced real structured-scene generalization.
- `spec06`: asset-derived infographics now work strongly on held-out prompts, with only one narrow `bullet-panel governance_path` slice still unstable.

## Capability Timeline

| Spec | Representation | Headline Result | What It Proved | Main Limitation |
| --- | --- | --- | --- | --- |
| `spec02` | raw SVG tokens | Legacy probe hit up to `87.5%` adherence on a tiny 6-probe slice; latest stable phase was `50.0%` adherence with `100%` valid SVG | The engine can train a raw SVG model end to end | Prompt control is fragile and metrics are noisy on the old probe |
| `spec03` | bootstrap + tokenizer variant | `0%` valid SVG, `0%` adherence, `0%` closure | A bad representation contract can fail cleanly and visibly | No usable generation capability emerged |
| `spec04` | structured scene atom DSL | `100%` valid/renderable, but `0%` exact on balanced `train/dev/test` structured probe | Syntax and rendering can be made deterministic and automatically scorable | Semantic binding and composition were still weak |
| `spec05` | improved structured scene DSL + richer midtrain + 3-layer model | `80.8%` overall exact, `75.0%` dev, `87.5%` test | Held-out structured visual control really emerged | A few narrow scene slices still fail |
| `spec06` | asset-derived infographic templates + slot filling | `r6` reached `91.7%` overall exact with `100%` renderability | The same approach extends to richer infographic layouts and now generalizes strongly on held-out prompts | Only three `bullet-panel governance_path` mismatches remain |

## Canonical Evidence

### Spec02

- Run: [svg_l16_d128_h512_v1024_ctx512_spec02](/home/antshiv/.cache/ck-engine-v7/models/train/svg_l16_d128_h512_v1024_ctx512_spec02)
- Training card: [svg_training_report_card.html](/home/antshiv/.cache/ck-engine-v7/models/train/svg_l16_d128_h512_v1024_ctx512_spec02/svg_training_report_card.html)
- IR report: [ir_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/svg_l16_d128_h512_v1024_ctx512_spec02/ir_report.html)
- Screenshot: [Screenshot 2026-03-10 at 09-48-28 SVG Training Report Card.png](/home/antshiv/Workspace/C-Kernel-Engine/Screenshots/Screenshot%202026-03-10%20at%2009-48-28%20SVG%20Training%20Report%20Card.png)

### Spec03

- Run: [svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512](/home/antshiv/.cache/ck-engine-v7/models/train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512)
- Training card: [svg_training_report_card.html](/home/antshiv/.cache/ck-engine-v7/models/train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512/svg_training_report_card.html)
- IR report: [ir_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512/ir_report.html)
- Screenshot: [Screenshot 2026-03-09 at 09-58-44 IR Visualizer svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512 C-Kernel-Engine.png](/home/antshiv/Workspace/C-Kernel-Engine/Screenshots/Screenshot%202026-03-09%20at%2009-58-44%20IR%20Visualizer%20svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512%20C-Kernel-Engine.png)

### Spec04

- Run: [spec04_structured_scenes_ctx512_d64_h128_v224](/home/antshiv/.cache/ck-engine-v7/models/train/spec04_structured_scenes_ctx512_d64_h128_v224)
- Capability report: [spec04_capability_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec04_structured_scenes_ctx512_d64_h128_v224/spec04_capability_report.html)
- Probe report: [spec04_probe_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec04_structured_scenes_ctx512_d64_h128_v224/spec04_probe_report.html)
- Screenshot: [Screenshot 2026-03-10 at 14-15-12 Spec04 Capability Report.png](/home/antshiv/Workspace/C-Kernel-Engine/Screenshots/Screenshot%202026-03-10%20at%2014-15-12%20Spec04%20Capability%20Report.png)

### Spec05

- Canonical best run: [spec05_structured_scenes_l3_d192_h384_ctx128_r2](/home/antshiv/.cache/ck-engine-v7/models/train/spec05_structured_scenes_l3_d192_h384_ctx128_r2)
- Probe report: [spec05_probe_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec05_structured_scenes_l3_d192_h384_ctx128_r2/spec05_probe_report.html)
- Tested prompts: [spec05_tested_prompts_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec05_structured_scenes_l3_d192_h384_ctx128_r2/spec05_tested_prompts_report.html)
- Screenshot: [Screenshot 2026-03-11 at 09-35-18 spec05_structured_svg_atoms Structured Scenes Probe Report.png](/home/antshiv/Workspace/C-Kernel-Engine/Screenshots/Screenshot%202026-03-11%20at%2009-35-18%20spec05_structured_svg_atoms%20Structured%20Scenes%20Probe%20Report.png)

### Spec06

- Best run: [spec06_structured_infographics_l3_d192_h384_ctx512_r6](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6)
- Prior balanced milestone: [spec06_structured_infographics_l3_d192_h384_ctx512_r2](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r2)
- Progression report: [spec06_progression_report_20260311.html](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec06_progression_report_20260311.html)
- Probe report `r6`: [spec06_probe_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/spec06_probe_report.html)
- Tested prompts `r6`: [spec06_tested_prompts_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/spec06_tested_prompts_report.html)

## Why The Line Matters

- `spec02` proved training infrastructure and raw SVG generation worked at all.
- `spec03` proved representation mistakes show up as hard capability failures.
- `spec04` established the structured DSL and automatic scoring path.
- `spec05` showed the model could generalize on held-out structured scene prompts.
- `spec06` showed the same method can move beyond shapes into templated infographics with headings, bullets, comparison panels, stat cards, and spectrum layouts.

## Current Best Message For The Team

This work is not about shipping SVG art as the end product.

It is about proving that local governed models can:

- follow a structured visual specification
- preserve deterministic rendering contracts
- generalize on held-out prompts
- be measured automatically with repeatable probes

The progression from `spec02` through `spec06` shows that this is no longer a toy syntax exercise. The strongest lines now demonstrate real structured-generation capability, and the remaining errors are narrow enough to repair surgically instead of redesigning the whole stack.

## Live Status

As of March 12, 2026:

- best current run: [spec06_structured_infographics_l3_d192_h384_ctx512_r6](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6)
- ledger: [run_ledger.jsonl](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/run_ledger.jsonl)
- status: pretrain and midtrain completed; probe report generated

## Best Companion Reports

- [spec04_spec05_iteration_report_20260311.html](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec04_spec05_iteration_report_20260311.html)
- [spec06_progression_report_20260311.html](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec06_progression_report_20260311.html)
- [ir_hub.html](/home/antshiv/.cache/ck-engine-v7/models/ir_hub.html)
