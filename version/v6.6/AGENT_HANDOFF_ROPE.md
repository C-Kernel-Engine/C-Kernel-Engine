# v6.6 RoPE Handoff

## Scope
`version/v6.6` pipeline, with required shared RoPE API compile fixes.

## What is already fixed

1. RoPE API/signature alignment (C)
- Updated RoPE declarations to include `rotary_dim`, `scaling_type`, `scaling_factor`.
- Updated callsites still using old `rope_precompute_cache` signature.
- Added missing forward declarations in `src/kernels/rope_kernels.c` to avoid C99 implicit declaration failures.
- Files:
  - `include/ckernel_engine.h`
  - `src/kernels/rope_kernels.c`
  - `src/ck_parity_api.c`
  - `src/ckernel_codegen.c`
  - `src/v6/ckernel_codegen_v6.c`
  - `src/v6.5/ckernel_codegen_v6.5.c`
  - `src/v6.6/ckernel_codegen_v6.6.c`
  - `version/v6.6/src/ckernel_engine.h`
  - `version/v6.6/src/ckernel_codegen_v6.6.c`

2. v6.6 converter/codegen runtime blockers
- Fixed `head_dim` unbound path in converter.
- Fixed `rope_theta` unbound path in codegen logging path.
- Files:
  - `version/v6.6/scripts/convert_gguf_to_bump_v6_6.py`
  - `version/v6.6/scripts/codegen_v6_6.py`

3. RoPE metadata extraction + normalization cleanup
- Converter now reads multiple GGUF RoPE key variants (including structured scaling dicts).
- Removed old hardcoded `rope_theta=1000000.0` write fallback.
- Added canonical config normalization in IR builder/codegen (`embed_dim`, `num_heads`, `head_dim`, `rope_*` aliases).
- Files:
  - `version/v6.6/scripts/convert_gguf_to_bump_v6_6.py`
  - `version/v6.6/scripts/build_ir_v6_6.py`
  - `version/v6.6/scripts/codegen_v6_6.py`

4. Validation done
- `make test-libs` passes.
- End-to-end runs succeed for:
  - Qwen2
  - Qwen3
  - Gemma

## What still needs to be fixed (remaining debt)

1. Complete RoPE scaling implementations in kernel
- `src/kernels/rope_kernels.c` currently effectively supports linear scaling only.
- Need real `dynamic` and `yarn` behavior, not just metadata passthrough.

2. RoPE layout contract
- Pipeline still assumes contiguous cos/sin layout (`sin = cos + MAX_SEQ_LEN * ROTARY_DIM / 2`).
- Need explicit `rope_layout` in metadata + layout-aware init arg generation.

3. Canonical RoPE schema in BUMP metadata (strict pass-through)
- Ensure converter stores and forwards:
  - `rope_theta`, `rotary_dim`, `rope_scaling_type`, `rope_scaling_factor`
  - `rope_layout`, `rope_original_context_length`
  - `rope_beta_fast`, `rope_beta_slow`, `rope_attn_factor` (if present)
- Ensure manifest/init IR/codegen preserve this block without loss.

4. Strict validation mode
- Add `--strict-rope` to converter:
  - fail when required fields for chosen scaling type are missing/inconsistent
  - keep legacy fallback only for non-strict mode

5. Update stale debt notes
- In `version/v6.6/scripts/codegen_v6_6.py` header:
  - "RoPE scaling type" note is partially outdated (linear now wired).
  - "RoPE memory layout" note is partially fixed (uses `ROTARY_DIM`) but still incomplete.

6. Minor cleanup
- Fix shebang typo in `version/v6.6/scripts/codegen_v6_6.py:1` (`#!nusr/bin/env python3`).

## Suggested execution order for next agent

1. Define/lock canonical RoPE schema in converter metadata.
2. Add strict validation (`--strict-rope`).
3. Propagate schema unchanged through manifest -> init IR -> codegen.
4. Implement kernel `dynamic` + `yarn`.
5. Add parity tests (metadata pass-through + runtime checks on scaled-RoPE model).
6. Update debt tracker text to reflect current state.
