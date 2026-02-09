# Gemma Debug Handoff (2026-02-08)

This file captures all work completed in this session to resume quickly next time.

## Goal
Make Gemma-3-270M compatible with CK v6.6 without regressing Qwen2/Qwen3.

## Files changed in this session

1. `src/kernels/gemm_kernels_q8_0.c`
2. `version/v6.6/src/ck_parallel_decode.c`
3. `scripts/test_kernels_vs_llamacpp.py`

No other files were edited by me in this session.

---

## What was changed

### 1) Enforced Q8_0 contract in public GEMV path
**File:** `src/kernels/gemm_kernels_q8_0.c`

- Updated `gemv_q8_0(...)` to act as FP32 adapter:
  - quantize input activations to Q8_0
  - call `gemv_q8_0_q8_0(...)`
- Added bounded fallback to old FP32 dispatch only when:
  - `K % QK8_0 != 0`, or
  - block count exceeds stack budget
- Added helper/fallback:
  - `gemv_q8_0_fp32_dispatch(...)`
  - `CK_Q80_STACK_Q8_BLOCKS`
- Removed dead/unreachable fallback code in `gemm_nt_q8_0(...)` and clarified via comments that it now inherits Q8_0xQ8_0 contract through `gemv_q8_0(...)`.

### 2) Kept decode threadpool path numerically consistent
**File:** `version/v6.6/src/ck_parallel_decode.c`

- Updated `gemv_q8_0_parallel_dispatch(...)`:
  - quantize FP32 input once to Q8_0
  - dispatch `work_gemv_q8_0_q8_0`
  - single-thread fallback now uses `gemv_q8_0_q8_0(...)`
- Removed use of `gemv_q8_0_parallel_simd(...)` path for this dispatch.
- Added stack bound define:
  - `DECODE_Q8_STACK_MAX_BLOCKS`

### 3) Added explicit q8 parity coverage
**File:** `scripts/test_kernels_vs_llamacpp.py`

- Added constants:
  - `QK8_0 = 32`
  - `BLOCK_Q8_0_SIZE = 34`
- Added generators:
  - `random_q8_0_block()`
  - `random_q8_0_weights()`
- Added ctypes signatures:
  - llama: `test_gemv_q8_0`, `test_gemm_q8_0`
  - CK: `ck_test_gemv_q8_0`, `ck_test_gemm_q8_0`
- Added Q8_0 contract reference helpers:
  - `_q8_0_quantize_reference(...)`
  - `_q8_0_q8_0_reference_gemv(...)`
  - `_q8_0_q8_0_reference_gemm(...)`
- Added tests:
  - `test_gemv_q8_0(...)`
  - `test_gemm_q8_0(...)`
- Included q8 tests in `run_all(...)` sequence and updated banner text.

---

## Build and test commands run

### Build
- `make -j4 libck_parity.so` -> success
- `python -m py_compile scripts/test_kernels_vs_llamacpp.py` -> success
- `make -j4` -> success

### Kernel parity (new q8 tests)
- `python scripts/test_kernels_vs_llamacpp.py --kernel gemv_q8_0 --tol 1e-3`
  - **FAIL**: `max_diff=3.81e-03`, `mean=3.81e-03`
- `python scripts/test_kernels_vs_llamacpp.py --kernel gemm_q8_0 --tol 1e-3`
  - **FAIL**: `max_diff=1.44e-01`, `mean=3.25e-02`

### End-to-end runs
- Gemma:
  - `python version/v6.6/scripts/ck_run_v6_6.py run ...gemma... --prompt "Hello" --max-tokens 16 --context-len 100 --force-compile`
  - Output still garbled.
- Qwen2:
  - `python version/v6.6/scripts/ck_run_v6_6.py run ...qwen2... --prompt "Hello" --max-tokens 8 --context-len 100`
  - Output coherent.
- Qwen3:
  - `python version/v6.6/scripts/ck_run_v6_6.py run ...qwen3... --prompt "Hello" --max-tokens 8 --context-len 100`
  - Output coherent.

---

## Fresh divergence run results

Command run:
- `bash scripts/run_gemma_first_divergence.sh hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf Hello 100 1`
  (moved to `bash version/v6.6/scripts/parity/run_gemma_first_divergence.sh hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf Hello 100 1`)

Observed:
- llama parity path had timeout/partial dump coverage issues.
- `parity_test.py` still produced layer-0 first-failure set.

Key layer-0 failures reported:
- `attn_proj`: `max_diff ~ 4.99e-01`, `mean ~ 6.12e-02`
- `q_proj`: `max_diff ~ 2.87e-01`, `mean ~ 1.28e-02`
- `k_proj`: `max_diff ~ 1.16e-01`, `mean ~ 1.49e-02`
- `v_proj`: `max_diff ~ 6.10e-02`, `mean ~ 7.99e-03`
- `qk_norm`: large in parity output (`~9.50e+00`) but likely still partly dump/layout-comparison artifact

Conclusion from this run:
- First hard divergence remains in **layer-0 projection path**, not a late-layer-only issue.

---

## Current status

1. Qwen2/Qwen3 behavior appears intact.
2. Gemma remains incompatible (garbled generation).
3. We improved Q8_0 contract consistency in runtime code, but not enough yet.
4. Remaining issue is likely still projection contract exactness and/or dump interpretation mismatch.

---

## Highest-priority next step

Add a focused test to isolate layer-0 qkv numerics directly (before rope/attention):

### Proposed new script
- `version/v6.6/scripts/parity/check_layer0_qkv_contract.py`

### What it should do
- Use real Gemma layer-0 weights + real layer-0 `attn_norm` activation (`[2, 9259]` tokens).
- Compute CK outputs for `q_proj`, `k_proj`, `v_proj` directly.
- Compute llama refs using patched parity helpers:
  - `test_gemv_q5_1`
  - `test_gemv_q8_0`
  - and prefill GEMM equivalents for token batch where applicable.
- Compare tensors with explicit shape/order checks before rope and before attention.
- Fail fast on first mismatch with max/mean diff and first index.

Why this is the fastest path:
- Removes attention/MLP cascade noise.
- Gives immediate proof whether root cause is qkv projection contract vs later stages.

---

## Notes

- `parity_test.py` is currently noisy when llama dumps are incomplete/timeout; use it as broad signal but not final root-cause proof.
- For Gemma, keep context length pinned (`100`) during debugging to minimize variance and speed runs.
