# Gemma Narrow Workflow (No Qwen Regressions)

## Goal
Add Gemma support without changing default behavior for Qwen2/Qwen3.

## Allowed Files (Default)
- `src/kernels/attention_kernels_sliding.c`
- `src/kernels/geglu_kernels.c`
- `src/kernels/gemm_kernels_q5_k.c`
- `src/kernels/gemm_kernels_q5_1_q8_1.c`
- `src/kernels/gemm_kernels_q8_0_q8_0_contract.c`
- `src/tokenizer/tokenizer_spm.c` (plus minimal tokenizer wiring only)
- `version/v6.6/kernel_maps/<new gemma map files>.json`
- `version/v6.6/kernel_maps/KERNEL_REGISTRY.json`
- `version/v6.6/templates/gemma3.json`
- Minimal, Gemma-flag-gated edits in:
  - `version/v6.6/scripts/build_ir_v6_6.py`
  - `version/v6.6/scripts/codegen_prefill_v6_6.py`
  - `version/v6.6/scripts/memory_planner_v6_6.py`

## Not Allowed (Unless Explicitly Required)
- Broad/global behavior changes for all models.
- Default routing changes for Qwen paths.
- Quant fallback policy changes that affect all families.
- Large Makefile refactors unrelated to new Gemma sources.

## Hard Rule
Any logic change in shared scripts must be gated:
- by model family (`gemma`) or
- by explicit template flag (example: `prefer_q8_0_contract`).

## Stash Cherry-Pick Workflow
1. Save everything:
```bash
git stash push -u -m "wip gemma"
```

2. Inspect stash content:
```bash
git stash list
git stash show --name-only --include-untracked stash@{0}
```

3. Restore only selected paths from stash:
```bash
git checkout stash@{0} -- <path1> <path2> <path3>
```
Alternative:
```bash
git restore --source=stash@{0} -- <path1> <path2> <path3>
```

4. Drop stash only after verified:
```bash
git stash drop stash@{0}
```

## Commit Slices (Small and Safe)
1. Kernels only.
2. Kernel maps + registry.
3. Template.
4. Minimal IR stitching (Gemma-gated).
5. Tokenizer SPM wiring.

## Regression Gates (Run After Every Slice)
1. Qwen2 hello + multi-token prompt.
2. Qwen3 hello + multi-token prompt.
3. Gemma hello smoke.

If Qwen regresses, revert last slice immediately and re-apply smaller.

## Quick Scope Check Before Commit
```bash
git diff --name-only
git diff --name-only --cached
```

If unrelated files appear:
```bash
git restore --staged <path>
git restore <path>
```
