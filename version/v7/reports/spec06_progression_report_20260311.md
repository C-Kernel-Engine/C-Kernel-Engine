# Spec06 Progression Report

This report summarizes the `spec06` structured infographic line from the first template-filling run through the current `r6` result.

## Executive Read

- `r1` proved the infographic template path worked mechanically, but semantic binding was weak.
- `r2` was the first real capability jump and remains the best balanced run so far.
- `r3` overfit a narrow topic slice and regressed badly.
- `r4` fixed topic-slot binding for the target slice, but layout binding regressed.
- `r5` recovered most of that damage and is the current best post-`r2` repair.
- `r6` closed the `spectrum-band` issue and pushed the line to the strongest result so far.

## Run Summary

| Run | Overall Exact | Train | Dev | Test | Renderable | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [r1](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r1) | `25.0%` | `20.0%` | `28.6%` | `28.6%` | `100.0%` | Template/slot mechanics worked, topic binding weak |
| [r2](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r2) | `75.0%` | `80.0%` | `85.7%` | `57.1%` | `100.0%` | Best balanced run so far |
| [r3](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r3) | `29.2%` | `40.0%` | `42.9%` | `0.0%` | `100.0%` | Narrow overweight caused regression |
| [r4](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r4) | `38.9%` | `50.0%` | `41.7%` | `25.0%` | `100.0%` | Generic slots fixed topic/slot mismatch but hurt layout binding |
| [r5](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r5) | `69.4%` | `91.7%` | `75.0%` | `41.7%` | `100.0%` | Strong recovery from `r4`, still weak on two slices |
| [r6](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6) | `91.7%` | `100.0%` | `91.7%` | `83.3%` | `100.0%` | Best run so far; only three `bullet-panel governance_path` misses remain |

## Midtrain Loss Trend

| Run | CK Loss First | CK Loss Final | CK Loss Min |
| --- | ---: | ---: | ---: |
| [r1](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r1/train_spec06_stage_b.json) | `2.4213` | `0.0230` | `0.0142` |
| [r2](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r2/train_spec06_stage_b.json) | `3.0498` | `0.0180` | `0.0131` |
| [r3](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r3/train_spec06_stage_b.json) | `1.9101` | `0.0192` | `0.0145` |
| [r4](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r4/train_spec06_stage_b.json) | `2.3358` | `0.0242` | `0.0142` |
| [r5](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r5/train_spec06_stage_b.json) | `1.8456` | `0.0430` | `0.0144` |
| [r6](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/train_spec06_stage_b.json) | `1.5005` | `0.0208` | `0.0141` |

Interpretation:

- training remained numerically healthy across the line
- regressions came from curriculum/schema choices, not from instability

## Layout Progression

| Run | Bullet | Compare | Stats | Spectrum | Flow |
| --- | --- | --- | --- | --- | --- |
| `r1` | `2/4` | `2/6` | `0/4` | `0/6` | `2/4` |
| `r2` | `2/4` | `5/6` | `4/4` | `4/6` | `3/4` |
| `r3` | `2/4` | `0/6` | `0/4` | `3/6` | `2/4` |
| `r4` | `6/9` | `3/9` | `0/6` | `0/6` | `5/6` |
| `r5` | `4/9` | `8/9` | `6/6` | `2/6` | `5/6` |
| `r6` | `6/9` | `9/9` | `6/6` | `6/6` | `6/6` |

## What Changed by Run

### R1

- asset-derived infographic templates
- topic-specific slot families like `[slot:governance_path__title]`

Result:

- topic drift toward dominant templates
- `stat-cards` and `spectrum-band` were especially weak

### R2

- topic-binding midtrain patch
- more topic edits, fewer layout edits

Result:

- large exact-match jump
- strongest balanced run in the line

### R3

- narrow overweighting around the `governance_path` bullet-panel slice

Result:

- learned to copy topic tags better
- but regressed badly on other layouts

### R4

- switched to generic slots like `[slot:title]`
- renderer resolves slot text using `[topic:...]`

Result:

- fixed the "right topic tag, wrong slot family" problem
- but `stat-cards` and `spectrum-band` collapsed into neighboring layouts

### R5

- removed redundant `source` from the learnable token surface
- boosted layout-separation edits around `compare-panels`, `stat-cards`, and `spectrum-band`

Result:

- `stat-cards` recovered completely
- `compare-panels` became strong
- `spectrum-band` improved but is still the weakest layout

### R6

- narrow repair for `bullet-panel governance_path` under `slate`
- stronger contrast between `spectrum-band` and neighboring layouts for `platform_rollout` and `structured_outputs`

Result:

- `spectrum-band` is fully fixed on the probe
- overall exact rose to `91.7%`
- only three misses remain, all inside the same `bullet-panel governance_path` family

## Current Best Read

`r6` is now the best `spec06` run.

It:

- preserved `100%` renderability
- fixed the `spectrum-band` collapse
- kept every non-bullet layout exact on the current probe
- reduced the failure surface to one narrow slice

## Remaining Failure Slices After R5

From [spec06_probe_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/spec06_probe_report.html):

- `bullet-panel governance_path` on `slate` with `frame:card`
- specific misses are:
  - `dev_02`: `green / slate / card / airy`
  - `test_02`: `green / slate / card / compact`
  - `test_03`: `orange / slate / card / compact`

## Current Best Run

Best run:

- [r6](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6)

Probe artifacts:

- [spec06_probe_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/spec06_probe_report.html)
- [spec06_tested_prompts_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/spec06_tested_prompts_report.html)
- [run_ledger.jsonl](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/run_ledger.jsonl)

## Best Artifacts To Review

- [r2 probe report](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r2/spec06_probe_report.html)
- [r5 probe report](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r5/spec06_probe_report.html)
- [r6 probe report](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/spec06_probe_report.html)
- [r6 tested prompts](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r6/spec06_tested_prompts_report.html)
- [IR Hub](/home/antshiv/.cache/ck-engine-v7/models/ir_hub.html)

## Bottom Line

`spec06` is no longer a toy template demo.

By `r6`, it has strong held-out infographic capability with `91.7%` overall exact and `100%` renderability.

It is not fully closed yet, but the remaining error surface is now a single narrow `bullet-panel governance_path` slice rather than a broad layout or topic failure.
