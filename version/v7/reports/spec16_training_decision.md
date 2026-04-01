# Spec16 Training Decision

- Verdict: **BLOCK RAW TRAINING**
- Frozen raw winner: `/home/antshiv/Workspace/C-Kernel-Engine/version/v7/runs/spec16_scene_bundle_l3_d192_h384_ctx768_r9`
- Default action: `decode_repair`
- Suggested next training branch: `capacity_branch`

## Reasons

- frozen raw baseline already demonstrates the spec16 bundle capability
- multiple post-baseline rungs failed to beat or even preserve the frozen winner
- the latest pilot failed the frozen-baseline gate and should not promote into another raw rung

## Unblock Requirements

- complete an autopsy of the latest blocked rung before changing training data again
- verify pilot budget integrity so selected tokens match actual processed tokens
- prove the next training idea is a new branch, not another narrow raw-repair rung
- if more raw margin is still needed after decode/repair, branch to a capacity test on the frozen r9 recipe

## Training Re-Enable Conditions

- No new raw spec16 rung may be a narrow repair-prose continuation. The next allowed training branch must be a clean capacity branch.
- A capacity branch must reuse the frozen r9 recipe and contract as the baseline surface. Do not reuse the r10-r12 repair-heavy prompt surfaces as the default raw recipe.
- Before launch, record both selected stage budgets and expected actual processed tokens, and verify the pilot fraction against the frozen baseline on actual processed tokens.
- A pilot may proceed to a full rung only if it improves system_diagram while showing zero family regression and zero hidden regression relative to frozen r9.
- If a candidate branch reuses the same unique row set as the previous blocked rung, treat it as a compute/order experiment only. It does not count as a new training idea.
- If deterministic repair on the frozen baseline is still the main source of gains, keep the line on decode/repair. Do not reopen raw training just because loss can still move.

## Banned Training Patterns

- Do not launch another raw rung whose main change is more warning-language rows about stopping, singleton tags, wrapper junk, or control markers.
- Do not treat the r11/r12 repair-heavy row surface as a safe pilot substrate for reduced-compute experiments.
- Do not approve a pilot that only changes row ordering or token budget on an otherwise unchanged brittle repair curriculum.
- Do not use low loss or improved renderability by itself as evidence that a raw branch is promotable.

## Paper Guidance

- Chinchilla (Hoffmann et al., 2022): Do not trust a pilot budget unless selected tokens and actual processed tokens match. Suggestion: Block new raw training until token-budget accounting is verified end to end.
- phi-1 (Gunasekar et al., 2023): Prefer clean, compiler-validated coverage over more repair prose. Suggestion: Only add new training rows when they add validated unique coverage, not just new warning language.
- Deduplication of Training Data Makes Language Models Better (Lee et al., 2022): Do not remove shortcut repetition unless it is replaced with new useful coverage. Suggestion: Any dedup or curriculum shrink must show replacement coverage before training is allowed.
- HumanEval / Codex (Chen et al., 2021): When outputs are mostly renderable or mechanically repairable, prefer decode/repair over more CE training. Suggestion: Use executable or compilable correctness as the route selector, not train loss.
- Grokking (Power et al., 2022): If loss improves while exactness regresses, stop repeating the same objective and data style. Suggestion: Treat that pattern as a block on more same-family repair rungs.

## Descendant Runs

- `spec16_scene_bundle_l3_d192_h384_ctx768_r1`: exact `0.3542`, renderable `0.7500`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r2`: exact `0.3542`, renderable `0.8958`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r3`: exact `0.0833`, renderable `0.6458`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r5`: exact `0.0000`, renderable `0.4167`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r6`: exact `0.7500`, renderable `0.8958`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r7`: exact `0.3333`, renderable `0.7500`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r8`: exact `0.6250`, renderable `0.9167`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r10`: exact `0.7083`, renderable `0.8125`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r11`: exact `0.8125`, renderable `0.9583`, beats frozen `False`, pilot gate `None`
- `spec16_scene_bundle_l3_d192_h384_ctx768_r12`: exact `0.0000`, renderable `0.4375`, beats frozen `False`, pilot gate `False`
