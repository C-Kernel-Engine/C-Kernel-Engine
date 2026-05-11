# Qwen3.5 CK vs llama.cpp Iteration Log

Date: 2026-05-07

## Model

- GGUF: `Qwen3.5-0.8B-Q4_K_M.gguf`
- CK run dir: `/opt/app-root/src/.cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF`
- llama.cpp: `/opt/app-root/src/Software/llama.cpp`, build `b9048-5207d120e`
- Prompt: `Give me an example of C, sql and python code?`
- Context: `1034`
- Threads: `24`

## CK Trace Result

`CK_TRACE_POS=1` showed `pos` and `rope_pos` advancing monotonically from prefill through generation. The simple stale-KV-slot theory is not supported by this trace: the generated API is not repeatedly writing or reading the same KV position.

## CK Smoke Test Before No-Repeat N-Gram

Command:

```bash
.venv/bin/python version/v8/scripts/ck_run_v8.py run \
  hf://unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "Give me an example of C, sql and python code?" \
  --max-tokens 256 \
  --thinking-mode suppressed \
  --context-len 1034
```

Result:

- Prompt eval: `28.74 tok/s`
- Decode: `27.98 tok/s`
- Stop: `token 4-gram repetition`
- Behavior: starts coherent, then falls into numeric phrase repetition like `2.2.3.4.5...`

## llama.cpp Comparison

Command:

```bash
/opt/app-root/src/Software/llama.cpp/build/bin/llama-cli \
  -m /opt/app-root/src/.cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  -p "Give me an example of C, sql and python code?" \
  -n 256 -c 1034 -t 24 \
  --temp 0.7 --top-k 40 --top-p 1.0 \
  --repeat-penalty 1.12 --repeat-last-n 96 \
  --reasoning off --single-turn --no-display-prompt --no-warmup
```

Result:

- Prompt: `96.0 tok/s`
- Generation: `45.9 tok/s`
- Behavior: content is imperfect, but does not hit the CK phrase loop.

## Deterministic Multi-Token Parity

Prompt tokens:

```text
248045,846,198,32002,5673,741,198,33963,728,449,3010,314,351,11,5526,321,9834,1970,30,248046,198,248045,74455,198,248068,271,248069,271
```

Result:

- First divergence: step `11`, prefix length `39`
- CK top-1: `29292`
- llama.cpp top-1: `34`
- Cosine: `0.989376`
- RMSE: `0.408371`
- Top-k overlap: `17/20`

Interpretation: CK and llama.cpp logits are close, but top-rank swaps occur early enough to steer CK into lower-quality continuations.

## Debug A/B Before Recurrent Conv Fix

- `CK_V7_DEBUG_MLP_DOWN_FP32=1`: did not improve the first divergence.
- `CK_V7_DEBUG_OUTPROJ_FP32=0`: made the first divergence slightly worse.

This conclusion was superseded by the recurrent SSM conv alias fix below. The
FP32 out-proj path was masking the real bug and became worse once the recurrent
conv buffer corruption was removed.

## Focused Recurrent / DeltaNet Kernel Tests

Command:

```bash
make --no-print-directory \
  test-deltanet \
  test-recurrent-split-qkv \
  test-split-q-gate \
  test-recurrent-dt-gate \
  test-recurrent-conv-state-update \
  test-recurrent-silu \
  test-recurrent-split-conv-qkv \
  test-recurrent-qk-l2-norm \
  test-recurrent-norm-gate \
  test-attn-gate-sigmoid-mul
```

Result: all listed tests passed, including Qwen3.5-like recurrent cases.

Interpretation: an isolated scalar kernel bug in DeltaNet/recurrent split/gate/state update is less likely. The remaining correctness suspect is graph-level composition: operation ordering, state carry across recurrent/full-attention transitions, attention-gate composition, or quantized projection drift that only shows up in the full model.

## Sampler Iteration

Added:

- `--no-repeat-ngram-size`
- model-level sampler default for Qwen3.5: `no_repeat_ngram_size=4`
- token n-gram loop stop guard as a final safety net

Test with `--no-repeat-ngram-size 4`:

- Prompt eval: `27.19 tok/s`
- Decode: `28.93 tok/s`
- Stop: `max_tokens reached`
- Behavior: hard repetition is avoided, but answer quality is still weaker than llama.cpp and can drift off-task.

## Recurrent SSM Conv Alias Fix

Found a generated-code correctness bug in the Qwen3.5 recurrent block:

- `recurrent_conv_state_update_forward` wrote `conv_x` into `recurrent_packed`.
- `ssm_conv1d_forward` then used `recurrent_packed` for both input and output.
- That is unsafe because `conv_x` is laid out as `[channel, history + tokens]`, while the conv output is `[token, channel]`; writing output element `out[ch]` can overwrite later input samples that the same convolution still has to read.

Fix:

- Add distinct activation buffers:
  - `recurrent_conv_input`
  - `recurrent_conv_qkv_raw`
  - `recurrent_conv_qkv`
- Regenerated Qwen3.5 now emits:
  - conv state update output -> `A_RECURRENT_CONV_INPUT`
  - SSM conv output -> `A_RECURRENT_CONV_QKV_RAW`
  - SiLU output -> `A_RECURRENT_CONV_QKV`

Verification:

- `python -m py_compile version/v8/scripts/build_ir_v8.py version/v7/scripts/build_ir_v7.py`: pass
- `python -m unittest tests.test_qwen35_template_guard -v`: pass, `20` tests
- `make test-ssm-conv`: pass
- Qwen3.5 short smoke after regeneration:
  - Prompt: `Hello!`
  - Output: `Hello! How can I help you today`
  - Decode: `27.70 tok/s`

Post-fix deterministic CK-vs-llama parity:

- First divergence: step `5`, prefix length `33`
- CK top-1: `314`
- llama.cpp top-1: `42726`
- Cosine: `0.988085`
- RMSE: `0.546749`
- Top-k overlap: `17/20`

Post-fix user-facing smoke with `--no-repeat-ngram-size 4`:

- Prompt eval: `30.57 tok/s`
- Decode: `29.30 tok/s`
- Stop: EOS token `248046`
- Behavior: catastrophic phrase looping is gone in this smoke, but the answer is still weaker/off-task compared with llama.cpp.

## Out-Projection Policy Cleanup

After the recurrent conv alias fix, forcing the full-attention out-projection
back to the normal Q8 activation path with `CK_V7_DEBUG_OUTPROJ_FP32=0` made
the deterministic parity pass for the first `8` generated tokens. The old
Qwen3.5 converter default `out_proj_input_policy=fp32` caused the regenerated
default path to diverge at step `5`.

Fix:

- Remove the Qwen3.5 `out_proj_input_policy=fp32` converter default.
- Keep the debug environment switch available for A/B testing, but do not make
  FP32 out-proj the Qwen3.5 default.

Verification after regeneration:

- `out_proj_input_policy` is absent from Qwen3.5 `config.json` and call IR.
- Default deterministic parity now matches llama.cpp through generated step
  `10`, then diverges at step `11`, prefix length `39`.
- First remaining divergence:
  - CK top-1: `29292`
  - llama.cpp top-1: `34`
  - Cosine: `0.989543`
  - RMSE: `0.405320`
  - Top-k overlap: `17/20`
- User-facing smoke:
  - Prompt eval: `32.39 tok/s`
  - Decode: `28.95 tok/s`
  - Stop: EOS token `248046`
  - Behavior: still off-task, but no hard loop or max-token collapse.

## Q5_K Fallback Audit

While chasing the remaining quantized-projection divergence, found a dormant
Q5_K FP32 fallback dequantization bug:

- The fallback treated `qh` as if high bits were grouped by subblock byte range.
- llama.cpp Q5_K uses `qh[i] & (1 << subblock)` for the high bit of value `i`
  in each subblock.

Fix:

- Add a shared `q5_k_quant_value()` helper for the Q5_K fallback GEMV/GEMM
  paths.
- Update the v7/v8 memory planner defaults so recurrent conv input/raw/silu
  buffers map to their distinct physical buffers instead of stale
  `A_RECURRENT_PACKED` aliases.

Verification:

- `make --no-print-directory build/libckernel_engine.so`: pass
- `make --no-print-directory test-head-major-q5-outproj-quick`: pass
- `.venv/bin/python -m py_compile unittest/test_head_major_q5_outproj.py`: pass
- `.venv/bin/python -m py_compile version/v8/scripts/memory_planner_v8.py version/v7/scripts/memory_planner_v7.py version/v8/scripts/build_ir_v8.py version/v7/scripts/build_ir_v7.py version/v8/scripts/compare_multitoken_logits_v8.py`: pass

Interpretation: this is real Q5_K correctness hardening, but it probably does
not explain the current Qwen3.5 step-11 divergence because the active
Q4_K_M run is not using this FP32 fallback on its decode hot path.

## Current Next Work

The next correctness work should chase the remaining step-11 parity divergence
into the quantized recurrent/full-model decode path rather than the sampler.
Highest-priority suspects:

- Q4_K/Q8_K matvec path used by Qwen3.5 decode
- Q5_K recurrent QKV/out projection accumulation compared with llama.cpp
- quantized head-major / output projection ordering and accumulation
- attention or recurrent-layer parity for the Qwen3.5 hybrid layout

The next performance work remains the quantized Q4_K/Q8_K/Q5 decode and head-major paths; llama.cpp is about `1.6x` faster on generation in this test.

## Negative A/Bs After Step-11 Divergence

Q4_K/Q8_K and Q5_K accumulation-order A/B:

- Changed local dot accumulation toward llama.cpp-style lane accumulation.
- Focused Q4_K/Q8_K and Q5 out-projection tests still passed.
- Qwen3.5 deterministic parity did not improve; step `11` still diverged and
  cosine/RMSE were slightly worse.
- Reverted this A/B.

FP32 logits A/B:

- Forced Qwen3.5 final logits from `gemv_q6_k_q8_k` to `gemv_q6_k`.
- Step `11` still diverged:
  - CK top-1: `29292`
  - llama.cpp top-1: `34`
  - Cosine: `0.989615`
  - RMSE: `0.403913`
  - Top-k overlap: `17/20`
- User-facing speed regressed badly: decode dropped to about `5.8 tok/s`
  because the final vocab projection is very large.
- Reverted this A/B and regenerated the cache back to the fast Q8 activation
  logits path.

Additional diagnostics:

- Extended the deterministic parity tooling to record per-id CK/llama logit
  values for the union of CK and llama top-k ids.
- At step `11`, the swapped ids are not a tiny tie:
  - token `34`: CK `16.8447`, llama `18.6407`
  - token `29292`: CK `18.2401`, llama `17.3127`
- Interpretation: the remaining issue is upstream hidden-state drift, not just
  final logits rounding.

Verification:

- `.venv/bin/python -m py_compile version/v8/scripts/compare_first_token_logits_v8.py version/v8/scripts/compare_multitoken_logits_v8.py version/v8/scripts/convert_gguf_to_bump_v8.py version/v8/scripts/build_ir_v8.py unittest/test_q6k_q8k_parity.py`: pass
- `unittest/test_q6k_q8k_parity.py`: harness now links all required objects;
  vec-dot passes, GEMV/GEMM fail only under the current very tight `1e-4`
  scalar-reference tolerance with max diff around `2e-4`.
- Qwen3.5 cache regenerated back to normal logits path:
  `prefer_q8=True`, `prefer_fp32_logits=False`, `logits -> gemv_q6_k_q8_k`.
- Quick smoke after restore: prompt eval `33.05 tok/s`, decode `33.85 tok/s`.

## Qwen3.5 RoPE Dimension Sections Fix

Root cause found:

- The GGUF carries Qwen3.5-specific RoPE metadata:
  - `qwen35.rope.dimension_count = 64`
  - `qwen35.rope.dimension_sections = [11, 11, 10, 0]`
- The converter knew the Qwen3.5 keys existed, but the active MRoPE section
  extraction only read `qwen3vl.rope.dimension_sections`.
- As a result, Qwen3.5 was lowered as plain split RoPE with `rotary_dim=256`
  instead of llama.cpp-style text multi-section RoPE with `rotary_dim=64`.

Fix:

- Add `extract_mrope_sections_for_arch()` and route Qwen3.5 to
  `qwen35.rope.dimension_sections`.
- Preserve Qwen3.5 `rotary_dim`, RoPE scaling fields, `rope_layout`, and
  `mrope_sections` in the Qwen3.5 config.
- Regenerate Qwen3.5 with `--force-convert --force-compile`.

Verification:

- Regenerated IR now reports `rope_init: theta=10000000.0, rotary=64`.
- Full-attention layers now lower `rope_qk -> mrope_qk_text`.
- Deterministic CK-vs-llama parity:
  - Before: first divergence at step `11`, cosine `0.989543`, RMSE `0.405320`,
    top-k overlap `17/20`.
  - After: `24/24` greedy generated steps pass.
  - Longer run: first divergence moved to step `39`, cosine `0.998983`,
    RMSE `0.197718`, top-k overlap `19/20`.
- User-facing CK smoke prompt no longer showed the early repeated-section
  collapse; decode was about `29.49 tok/s` for the sampled 180-token run.

Interpretation: the hard Qwen3.5 corruption was mostly a RoPE lowering bug,
not KV-cache position drift. Remaining drift is now a much smaller numerical
ranking issue after a longer prefix.
