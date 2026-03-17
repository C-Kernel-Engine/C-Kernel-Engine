# Spec04 and Spec05 Iteration Report

This report summarizes the structured-scene line from the original `spec04` baseline through the stronger `spec05` run.

It focuses on the runs that materially changed curriculum or architecture, using the canonical probe reports where available and the stage-eval matrix for older `spec04` diagnostics.

## Executive Read

- `spec04` learned SVG syntax and renderability first, not semantic control.
- curriculum-only tweaks inside the old flat atom target helped a little, but did not generalize on `dev/test`
- `spec05` is where the line became a real structured-generation capability
- the best `spec05` run moved held-out exactness from `0%` in `spec04` to `75%` on `dev` and `87.5%` on `test`

## Run Summary

| Line | Run | Kind | Overall Exact | Train | Dev | Test | Renderable | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `spec04` | [canonical baseline](/home/antshiv/.cache/ck-engine-v7/models/train/spec04_structured_scenes_ctx512_d64_h128_v224) | baseline structured scenes | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `100.0%` | Valid SVG, weak semantic binding |
| `spec04` | [/tmp/spec04_structured_scenes_comp_midtrain_eval2](/tmp/spec04_structured_scenes_comp_midtrain_eval2) | composition-only midtrain | `5.0%` | `10.0%` | `0.0%` | `0.0%` | `100.0%` | Narrow train improvement only |
| `spec04` | [/tmp/spec04_structured_scenes_blended_midtrain_r2](/tmp/spec04_structured_scenes_blended_midtrain_r2) | blended midtrain | `10.0%` | `20.0%` | `0.0%` | `0.0%` | `95.0%` | Best `spec04` exact score, but still no holdout generalization |
| `spec04` | [/tmp/spec04_structured_scenes_blended_midtrain_r3](/tmp/spec04_structured_scenes_blended_midtrain_r3) | lighter blended midtrain | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `100.0%` | Reverted to generic valid SVGs |
| `spec05` | [/tmp/spec05_structured_scenes_l3_d192_h384_ctx128](/tmp/spec05_structured_scenes_l3_d192_h384_ctx128) | first scaled structured-scenes run | `69.2%` | `100.0%` | `50.0%` | `50.0%` | `100.0%` | First run with real held-out semantic control |
| `spec05` | [best canonical run](/home/antshiv/.cache/ck-engine-v7/models/train/spec05_structured_scenes_l3_d192_h384_ctx128_r2) | targeted midtrain repair | `80.8%` | `80.0%` | `75.0%` | `87.5%` | `92.3%` | Best overall run in this line |

## Stage-Level Spec04 Diagnostics

These older stage metrics are useful because they show what the model was learning before exact-match probe scores moved.

| Run | Valid SVG | Closure | OOD | Adherence | Tag Adherence | Prefix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [spec04 baseline](/home/antshiv/.cache/ck-engine-v7/models/train/spec04_structured_scenes_ctx512_d64_h128_v224/stage_eval_matrix.json) | `100.0%` | `100.0%` | `100.0%` | `12.5%` | `18.8%` | `0.0%` |
| [/tmp/spec04_structured_scenes_comp_midtrain_eval2/stage_eval_matrix.json](/tmp/spec04_structured_scenes_comp_midtrain_eval2/stage_eval_matrix.json) | `100.0%` | `100.0%` | `100.0%` | `25.0%` | `25.0%` | `0.0%` |
| [/tmp/spec04_structured_scenes_blended_midtrain_r2/stage_eval_matrix.json](/tmp/spec04_structured_scenes_blended_midtrain_r2/stage_eval_matrix.json) | `87.5%` | `87.5%` | `50.0%` | `25.0%` | `31.3%` | `0.0%` |
| [/tmp/spec04_structured_scenes_blended_midtrain_r3/stage_eval_matrix.json](/tmp/spec04_structured_scenes_blended_midtrain_r3/stage_eval_matrix.json) | `100.0%` | `100.0%` | `100.0%` | `0.0%` | `25.0%` | `0.0%` |

Interpretation:

- `spec04` became very good at emitting renderable SVG shells
- `prefix_integrity` stayed at `0.0%` across all `spec04` variants
- better adherence did not translate into holdout exactness until the line changed more substantially in `spec05`

## What Changed Across Iterations

### 1. Baseline Spec04

Run:

- [spec04_structured_scenes_ctx512_d64_h128_v224](/home/antshiv/.cache/ck-engine-v7/models/train/spec04_structured_scenes_ctx512_d64_h128_v224)

Observed behavior:

- outputs were usually renderable
- outputs did not bind the requested controls reliably
- balanced probe exact match was `0.0%`

Main lesson:

- flat next-token learning over the atom DSL was enough for grammar, not enough for compositional control

### 2. Spec04 Composition-Only Midtrain

Run:

- [/tmp/spec04_structured_scenes_comp_midtrain_eval2](/tmp/spec04_structured_scenes_comp_midtrain_eval2)

Observed behavior:

- train exact moved slightly
- `dev/test` exact remained `0.0%`

Main lesson:

- composition belongs in `midtrain`, but composition-only data caused forgetting of the broader scene contract

### 3. Spec04 Blended Midtrain

Runs:

- [/tmp/spec04_structured_scenes_blended_midtrain_r2](/tmp/spec04_structured_scenes_blended_midtrain_r2)
- [/tmp/spec04_structured_scenes_blended_midtrain_r3](/tmp/spec04_structured_scenes_blended_midtrain_r3)

Observed behavior:

- `r2` was the best `spec04` variant on exact match
- `r3` kept valid SVG but lost semantic control again

Main lesson:

- curriculum balance mattered, but the flat target representation was still the ceiling

### 4. First Spec05 Run

Run:

- [/tmp/spec05_structured_scenes_l3_d192_h384_ctx128](/tmp/spec05_structured_scenes_l3_d192_h384_ctx128)

Observed behavior:

- first run with strong train exact and real held-out generalization
- `dev` and `test` exact both reached `50.0%`

Main lesson:

- richer composition controls plus a larger model moved the task from “syntax benchmark” toward “real capability”

### 5. Best Spec05 Run

Run:

- [spec05_structured_scenes_l3_d192_h384_ctx128_r2](/home/antshiv/.cache/ck-engine-v7/models/train/spec05_structured_scenes_l3_d192_h384_ctx128_r2)

Observed behavior:

- `80.8%` overall exact
- `75.0%` `dev` exact
- `87.5%` `test` exact

Main lesson:

- narrow midtrain repair on the right failure slice was more valuable than immediately scaling again

## Best Current Artifacts

If someone wants the strongest concrete evidence from this line, use:

- `spec04` baseline truth-telling report:
  [spec04_probe_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec04_structured_scenes_ctx512_d64_h128_v224/spec04_probe_report.html)
- `spec05` best run:
  [spec05_probe_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec05_structured_scenes_l3_d192_h384_ctx128_r2/spec05_probe_report.html)
- `spec05` prompt-by-prompt report:
  [spec05_tested_prompts_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec05_structured_scenes_l3_d192_h384_ctx128_r2/spec05_tested_prompts_report.html)
- `spec05` narrative analysis:
  [spec05_midtrain_analysis_20260311.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec05_midtrain_analysis_20260311.md)

## Bottom Line

`spec04` established the evaluation problem and showed that renderability alone was not enough.

`spec05` is the run family that demonstrated the real transition from:

- valid SVG syntax

to:

- held-out structured control following

That is the important story in this line.
