# Gen1 Program

## Goal

Move from narrow diagnostic specs to a broad-contract scene-DSL line that tests compositional generalization over a much larger asset-backed distribution.

## Baseline

- Site assets: `97`
- Current gold-covered assets: `13`
- Bootstrap run: `/home/antshiv/.cache/ck-engine-v7/models/train/spec_broader_1_scene_dsl_l3_d192_h384_ctx512_r1`
- Bootstrap seed cases: `11`
- Bootstrap generated scenes: `396`
- Bootstrap pretrain rows: `1011`
- Bootstrap midtrain rows: `4718`
- Bootstrap exact/renderable: `0.0238` / `0.5476`

## Principles

- Broaden coverage aggressively, but only with coherent gold DSL/compiler supervision.
- Freeze tokenizer, DSL, compiler, canonicalizer, and eval contract for each gen1 training phase.
- Use held-out recombinations as the primary generalization target rather than only held-out prompt variants.
- Treat narrow spec/rung work as diagnosis; treat gen1 as the broad-contract scaling experiment.

## Phase Plan

### `bootstrap_baseline`

- Status: `completed`
- Purpose: Prove the broader training path and establish a real baseline before coverage expansion.
### `coverage_sprint`

- Status: `next`
- Purpose: Expand gold DSL/compiler coverage rapidly across the first-wave families before the first true gen1-scale training run.
- Target gold assets after sprint: `36`
- Families: `comparison_span_chart, table_matrix, architecture_map, poster_stack, dashboard_cards, timeline_flow`
### `gen1_full_mix`

- Status: `planned`
- Purpose: Train one broad-contract model on the full mix instead of patching narrow failure slices.
- Gold assets target: `40-50`
- Generated scenes target: `5000-10000`
- Holdout policy: `novel_recombinations`
- Capacity canary: `6L d384 h768`

## Eval Contract

- `compiler_valid_rate`
- `exact_rate`
- `renderable_rate`
- `recombination_exact_rate`

Failure taxonomy:
- `family`
- `form`
- `style`
- `topology`
- `syntax`

## Gates

- Do not start gen1_full_mix until the wider tokenizer/DSL/compiler contract is frozen.
- Do not start gen1_full_mix until the first-wave families have gold DSL seeds and compiler smoke passing.
- Do not compare capacity until the data contract is fixed and the same broad dataset is used for both models.

## Next Steps

- Promote the finished spec_broader_1 run to the documented gen1 bootstrap baseline.
- Expand gold DSL/compiler coverage quickly across the first-wave family plan.
- Generate recombination-heavy holdouts, not only prompt paraphrase holdouts.
- Launch the first true gen1 full-mix run only after the broader contract is frozen.

