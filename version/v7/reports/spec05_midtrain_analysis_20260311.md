# Spec05 Midtrain Analysis

Run: `/tmp/spec05_structured_scenes_l3_d192_h384_ctx128_r2`

Artifacts:
- Probe report HTML: `/tmp/spec05_structured_scenes_l3_d192_h384_ctx128_r2/spec05_probe_report.html`
- Probe report JSON: `/tmp/spec05_structured_scenes_l3_d192_h384_ctx128_r2/spec05_probe_report.json`
- Tested prompts HTML: `/tmp/spec05_structured_scenes_l3_d192_h384_ctx128_r2/spec05_tested_prompts_report.html`
- Tested prompts Markdown: `/tmp/spec05_structured_scenes_l3_d192_h384_ctx128_r2/spec05_tested_prompts_report.md`
- Stage A train report: `/tmp/spec05_structured_scenes_l3_d192_h384_ctx128_r2/train_spec05_stage_a.json`
- Stage B train report: `/tmp/spec05_structured_scenes_l3_d192_h384_ctx128_r2/train_spec05_stage_b.json`

## Summary

- Model: `3L / d192 / h384 / ctx128`
- Pretrain loss: `5.7081 -> 0.0391`
- Midtrain loss: `4.3155 -> 0.0806`
- Probe cases: `26`
- Overall exact: `80.8%`
- Overall renderable: `92.3%`

## Split Results

| Split | Count | Exact | Renderable | SVG Exact |
| --- | ---: | ---: | ---: | ---: |
| `train` | `10` | `80.0%` | `80.0%` | `80.0%` |
| `dev` | `8` | `75.0%` | `100.0%` | `75.0%` |
| `test` | `8` | `87.5%` | `100.0%` | `87.5%` |

## What Improved

- The earlier `badge label:map -> GO` failure appears fixed.
- Held-out behavior improved materially over the previous `spec05` run.
- The targeted curriculum changes moved dev/test more than train, which is the right direction.

## Remaining Failures

There are 5 failing probe cases.

1. `Train Label #8`
   - Prompt: label-card, `label:ai`, `bg:paper`, `frame:card`, `density:compact`
   - Failure: malformed prefix; output drops `[svg]` and duplicates `[bg:paper]`

2. `Train Badge #10`
   - Prompt: badge, `shape:circle`, `label:ai`, `bg:paper`, `frame:card`, `density:compact`
   - Failure: collapses into malformed label-card-like structure instead of badge structure

3. `Dev Pair-V #3`
   - Prompt: pair-v, `circle + triangle`, `gold + blue`, `bg:paper`, `frame:card`, `density:airy`
   - Failure: collapses to `layout:single`

4. `Dev Pair-V #4`
   - Prompt: pair-v, `circle + triangle`, `gold + blue`, `bg:slate`, `frame:card`, `density:airy`
   - Failure: keeps `layout:pair-v` but corrupts the top object geometry tokens

5. `Test Pair-V #3`
   - Prompt: pair-v, `circle + triangle`, `gold + blue`, `bg:mint`, `frame:card`, `density:airy`
   - Failure: same geometry-token corruption on the top object

## Midtrain Curriculum Snapshot

From the generated workspace manifest:

- Total midtrain rows: `37812`
- Direct rows: `15472`
- Edit rows: `22340`
- Layout counts:
  - `single`: `8040`
  - `pair-h`: `3096`
  - `pair-v`: `5028`
  - `label-card`: `5984`
  - `badge`: `15664`
- Edit counts:
  - `color2`: `324`
  - `label`: `8960`
  - `compose`: `2960`
  - `density`: `4576`
  - `frame`: `1632`
  - `simplify`: `3768`

## Read

The curriculum patch worked, but it worked unevenly:

- `label` binding improved enough to fix the `MAP` badge problem.
- `pair-v` still remains the brittle slice.
- The failure mode is now more geometric/prefix corruption than semantic color substitution.

That means the next patch should focus on `pair-v airy card` stability, not more generic label work.

## Recommended Next Patch

1. Add more exact minimal pairs for `pair-v` with fixed `shape2=triangle`.
2. Add targeted `pair-v` geometry repair rows, especially `circle top + triangle bottom`.
3. Add a small number of train probes with the exact `gold + blue + airy + card` pattern.
4. Add a prefix-integrity check into the structured probe summary so malformed-but-close outputs are easier to isolate.
