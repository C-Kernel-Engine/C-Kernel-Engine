# Spec06 R3 Narrow Topic Patch Note

This note captures the `r3` experiment for the structured infographic line after narrowing the midtrain curriculum toward the remaining `bullet-panel governance_path` failure slice.

Run:

- [spec06_structured_infographics_l3_d192_h384_ctx512_r3](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r3)

Baseline comparator:

- [spec06_structured_infographics_l3_d192_h384_ctx512_r2](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r2)

## What Changed

The midtrain materializer was patched in [materialize_spec06_structured_atoms_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/dataset/materialize_spec06_structured_atoms_v7.py) to overweight the narrow train slice:

- `layout=bullet-panel`
- `bg=slate`
- `frame=card`
- `topic in {governance_path, eval_discipline}`

The direct rows for that slice were repeated more heavily, and `topic` edit rows for the same slice were boosted further.

## Operational Result

The launcher patch worked as intended:

- `.ck_build` was materialized automatically
- the canonical run compiled successfully
- probe report generation completed without manual recovery

Artifacts:

- [train_spec06_stage_a.json](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r3/train_spec06_stage_a.json)
- [train_spec06_stage_b.json](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r3/train_spec06_stage_b.json)
- [spec06_probe_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r3/spec06_probe_report.html)
- [spec06_tested_prompts_report.html](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r3/spec06_tested_prompts_report.html)

## Metrics

| Run | Overall Exact | Train | Dev | Test | Renderable |
| --- | ---: | ---: | ---: | ---: | ---: |
| [r2](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r2/spec06_probe_report.json) | `75.0%` | `80.0%` | `85.7%` | `57.1%` | `100.0%` |
| [r3](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r3/spec06_probe_report.json) | `29.2%` | `40.0%` | `42.9%` | `0.0%` | `100.0%` |

Delta from `r2` to `r3`:

- overall exact: `-45.8 pts`
- dev exact: `-42.9 pts`
- test exact: `-57.1 pts`

## What Improved

The targeted bullet-panel failure changed in one narrow way:

- in `r2`, the `governance_path` bullet-panel failures collapsed both the outer topic token and the slot content to `eval_discipline`
- in `r3`, the outer `[topic:governance_path]` token is preserved on those prompts

## What Still Failed

The content binding did not actually repair.

For the targeted `bullet-panel governance_path` prompts:

- the response now keeps `[topic:governance_path]`
- but the slots still come from `eval_discipline`, e.g. `[slot:eval_discipline__title]`

So the patch taught the model to copy the topic tag more often, but not to rebind the layout content slots.

The regression also spread beyond the target slice:

- `compare-panels`: `5/6 -> 0/6`
- `stat-cards`: `4/4 -> 0/4`
- `spectrum-band`: `4/6 -> 3/6`

## Training Summary

Midtrain optimization itself was numerically fine:

- `r2` CK loss: `3.0498 -> 0.0180`, min `0.0131`
- `r3` CK loss: `1.9101 -> 0.0192`, min `0.0145`

So this is not a numerical failure. It is a curriculum failure caused by overweighting one narrow semantic slice.

## Conclusion

`r3` should be kept as a recorded negative result, not promoted as the best `spec06` line.

Best current judgment:

- keep [r2](/home/antshiv/.cache/ck-engine-v7/models/train/spec06_structured_infographics_l3_d192_h384_ctx512_r2) as the strongest `spec06` run
- keep the launcher patch
- revert the narrow `r3` overweighting strategy
- if we repair the `governance_path` bullet-panel slice again, do it with a balanced same-layout minimal-pair set rather than a heavy local repeat boost
