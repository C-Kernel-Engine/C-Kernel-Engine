# Gemma3 Baseline Parity Investigation

Status: development candidate, not full-model certified. Start with standard
Q4_K_M before testing the five Q5_K substitutions in UD-Q4_K_XL.

## Reproduction

- Baseline CKE: `28b321bb5`, isolated worktree; shared developer worktree untouched.
- Oracle llama.cpp: `f3e182816421c648188b5eab269853bf1531d950`.
- Artifact: `unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q4_K_M.gguf`.
- HF revision: `c90975dbd40c0c7b275fefaae758c3415c906238`.
- SHA256: `b1baabd6b729e4041822220d3e648e00d99cac5df86b10dffb77bcccf0688e39`.
- Ryzen evidence root: `/data/cke/ud-certification`.
- Matched numerical run: 26 identical prefix token IDs, 256 teacher-forced
  reference output rows, 1024 capacity, FP32 KV, oracle flash attention disabled.
  Capacity is not consumed context. This does not test long-context attention.

## Proven Findings

1. Local RoPE metadata: main used the global 1,000,000 base on local layers.
   Identical-input replay against llama Q/K captures gives RMSE 0.7173038 with
   that base and 0.0000003869 with the local 10,000 base. The candidate declares
   the per-layer RoPE schedule and uses the existing direct provider. No new
   quantization arithmetic is involved. Global/local attention masks still need
   separate validation beyond the 512-token sliding window.
2. Thread-pool publication race: lagging inactive workers could combine an old
   dispatch epoch with the next dispatch's active width. A 16/2/2-width stress
   failed at iteration 3 on the original pool. A model run crashed at function
   pointer 0x25. Publishing/snapshotting epoch and descriptor under the existing
   mutex passes 200,000 transitions. The existing thread-pool test now includes
   20,000 changing-width dispatches and passes on P3 and Ryzen (308,713 checks).
   This change is shared; broader model regression remains required.
3. Tokenizer metadata override: codegen forced Gemma's declared `llama` SPM
   algorithm to unigram. The C prompt became 37 tokens, including split role
   labels. Honoring the declared algorithm produces 23 tokens and readable
   C-oriented output. Exact token-ID comparison is required, not length alone.
   Subsequent comparison confirms all 23 IDs exactly match llama.cpp for the
   same rendered prompt. Unit coverage preserves both declared modes
   across Gemma3, Gemma4 and Llama.
4. Attention-input RMSNorm: identical captured input and converted gamma give
   15,421/16,640 exact values using the generic provider; the existing
   `rmsnorm_forward_llama_production` gives 16,640/16,640 exact values. The
   candidate binds only Gemma3 `attn_norm` to that provider for trajectory
   evaluation. Gemma4 and the shared default provider are unchanged.
5. Q/K normalization: identical-input replay of the existing
   `qk_norm_forward_llama_production` provider is exact for both tensors, while
   the generic provider has RMSE about 0.000000053. The Gemma3 candidate now
   selects the matched provider. No shared arithmetic implementation changed.

## Numerical Evidence Before RMSNorm Routing Change

| Candidate | Exact logit rows | Matching top-1 rows | Initial logit RMSE |
| --- | ---: | ---: | ---: |
| Main, 1 thread | 0/256 | 172/256 | 6.7481 |
| RoPE and pool correction, 16 threads | 0/256 | 248/256 | 0.8808197 |
| Plus matched attention-input RMSNorm | 0/256 | 246/256 | 1.7165851 |
| Plus matched Q/K normalization | 0/256 | 245/256 | 1.2230454 |

Main's 16-thread results were not repeatable and cannot certify parity. The
corrected run passed baseline repeatability and capture-neutrality checks.
Improved top-1 agreement is not full-logit acceptance.

After the RoPE correction, layer-0 Q-normalization and RoPE RMSE are both about
0.0003868: RoPE now preserves upstream error instead of introducing the earlier
large discrepancy. Q projection RMSE is 0.0031040 and attention-input RMSNorm
RMSE is 0.0000005508. Replay the first discrepancy with identical inputs before
changing projection kernels.

With the matched attention-input RMSNorm, layer-0 input, attention norm, Q
projection and K projection are all bit-exact. Post-attention norm RMSE falls
to 0.0000005450; layer output RMSE falls from 0.05849 to 0.00656. The full-model
score does not improve monotonically: correcting one boundary does not remove
downstream arithmetic differences. Keep these changes as a development
candidate until the entire trajectory passes.

## Reused Test Gates

- Scaffold and direct-RoPE suite: 27 passed, 10 subtests passed before the final
  binding assertion; six targeted metadata/binding tests pass afterward.
- `make test-rmsnorm-llama-production`: four bit-exact oracle cases at each
  thread setting 1, 16, 20 and 24 on P3. This includes Q/K decode and prefill.
- The isolated worktree initially lacked llama headers. The existing pinned
  oracle checkout was linked locally, then the target passed; no download.
- `git diff --check` passes.

## Readable Generation Is Separate

With RoPE, pool and tokenizer corrections, the C task produces readable output
but invents `stdio++.h`. It is not a code-quality pass. The short run reports
23 input tokens, 160 output runs, 1033.4 prefill tok/s and 127.4 decode tok/s.
These are short-run observations, not sustained-performance certification.

## Evidence Files On Ryzen

- `reports/rope-pool-fixed-matched.json`: trajectory and neutrality.
- `reports/corrected-layer0-boundaries.json`: aligned layer-0 boundaries.
- `reports/identical-input-norm-replay.json`: isolated RMSNorm replay.
- `reports/identical-input-qk-norm-replay.json`: isolated Q/K normalization.
- `reports/rope-pool-tokenizer-qknorm-matched.json`: latest full-logit replay.
- `reports/corrected-tokenizer-parity.json`: exact prompt-token comparison.
- `reports/gemma3-three-fixes-generation.log`: readable candidate output.
- `reports/gemma3-rope-control-gdb.log`: original pool crash.
- `reports/threadpool-fixed-test.log`: fixed thread-pool regression.
- `runs/gemma3-rope-control/ir_report.html`: generate/refresh after final routing.

## Continued Boundary Isolation

The next candidate adds a distinct `rope_forward_qk_split_llama` provider:
recursive FP32 frequency stepping, system math, explicit FMA ordering and
parallel independent token ranges. Existing direct-frequency RoPE is unchanged.
The existing llama-style cache plus split rotation matched captured Q/K exactly;
the new direct variant matches that cache on P3 for decode, prefill and nonzero
offsets. Full-capture layer-0 Q/K normalization and RoPE now match exactly.

Identical-input FFN normalization and FFN-post-normalization replay also match
exactly using the existing llama production provider. GeGLU's existing FP32
tanh path has RMSE 0.00039958 (max 0.06681824); native FP16-table GELU times the
captured up tensor is bit-exact. The candidate adds `geglu_forward_ggml_native`
and binds the Gemma3 norm operations explicitly. This native path builds its
own table and does not load llama.cpp. `make test-geglu-ggml-native` compares
against the real ggml graph for 1/26 rows, separate/in-place buffers. All four
cases pass on P3. It is connected to the existing optional llama-oracle section
of the numerical-contract target, not an unreferenced standalone test.

Latest full-model replay (`native-gelu-norm-matched.json`): 0/256 exact rows,
246/256 matching top-1, initial RMSE 3.06194. Both baseline repeatability and
capture neutrality pass. This is NOT a full-model improvement claim. Layer-0
MLP-down is now bit-exact; remaining layer-0 output RMSE is 0.0000005434.

The first remaining nonexact stage is attention. Last-query attention output
RMSE is 0.0000008508 (max 0.0000152588). A diagnostic replay of the existing
ggml-style dot helper differs from the oracle's Q*K scores. The helper uses
four eight-lane accumulators even in an AVX-512 build. A separate diagnostic
four-sixteen-lane replay matches all four heads' Q*K scores bit-exactly.
The oracle's exported dot function also matches the graph exactly, confirming
the source of that score discrepancy. No production attention provider has
been changed based on this diagnostic yet.

Additional attention detail: the oracle pads its KV dimension to 256 for this
26-token prompt. Preserve that padded reduction length when comparing softmax
and probability-times-value arithmetic; comparing only live tokens can alter
reduction ordering even when padded probabilities are zero.

Performance debt: native GeGLU currently uses ascending serial compaction and
per-element table initialization checks. Optimize only after full parity;
disjoint-row parallelism must not race with in-place compacted input. RoPE's
new provider shares trigonometric work across heads and parallelizes tokens.

Diagnostic files on Ryzen: `tools/gemma-attention-probe.c`,
`tools/gemma-attention-replay.py`, `reports/identical-input-downstream-replay.json`,
`reports/native-corrected-layer0-boundaries.json`. The probe is not a production
provider and loads oracle symbols solely for comparison.

With the corrected AVX-512 scores and the padded 256-position softmax row,
the native softmax helper matches all four heads exactly. Remaining PV error
is max 0.0000038147. The oracle's own standalone dot function produces the same
PV discrepancy against its graph, while a serial-FMA replay is also nonexact.
Therefore inspect the batched graph's actual V*probability GEMM path; simply
substituting a supposedly oracle-matched dot helper is insufficient. This is
still a diagnostic conclusion, not full-attention certification.

Final local scaffold/RoPE rerun: 29 tests and 13 subtests pass. Native GeGLU
graph oracle: four cases pass. No UD certification or model-support promotion
was made during this stage.

## Certification Work Still Required

Complete standard Q4_K_M full-logit replay, then identical-settings UD-Q4_K_XL.
Inspect each artifact's real tensor inventory; UD labels are not kernel dtypes.
Do not claim IQ4_XS or other IQ formats are certified by this model: this UD XL
artifact does not contain IQ4_XS. Validate Gemma4 and other shared-pool consumers
before promotion. Extend consumed context beyond the local window only after
the global/local mask contract is corrected and tested.

## Standard Q4_K_M Baseline Accepted

The remaining probability-times-value replay was exact only when softmax retained
llama.cpp's 256-position KV padding. The first generated provider incorrectly
bounded that reduction extent by the compact prefill K/V stride (`26` for the
diagnostic prompt). The corrected provider derives reduction capacity from its
declared scratch buffers while retaining the compact K/V stride for addressing.
Layer-0 attention then became bit-exact in the generated runtime.

An all-layer boundary scan subsequently matched every layer output and the final
RMSNorm output bit-exactly. The only remaining discrepancy was the tied Q8_0
vocabulary projection. Replaying the exact final hidden row proved that the old
FP32-activation `gemv_q8_0` route differed in 262,142 of 262,144 logits, while
the existing Q8_0-activation contract matched all logits exactly. Gemma3 now
selects that contract, and the prefill map explicitly preserves its GEMM provider
for last-token dispatch instead of deriving an unregistered row function.

Acceptance result on Ryzen, 16 threads, identical 26-token input and forced
256-token trajectory:

- 256/256 complete vocabulary rows bit-exact to llama.cpp.
- 256/256 top-1 choices identical.
- Baseline repeatability bit-exact.
- Aggregate X-Ray capture neutrality bit-exact.
- Layer 0 through layer 17 and final normalized hidden state bit-exact at the
  captured first-output boundary.

Evidence: `/data/cke/ud-certification/reports/native-gelu-norm-matched.json`,
`/data/cke/ud-certification/reports/all-layer-boundaries.json`, and
`/data/cke/ud-certification/runs/native-gelu-norm-ck-capture`.

This accepts the standard Gemma3 270M Q4_K_M numerical baseline.

## UD-Q4_K_XL Mixed-Format Accepted

The Unsloth `gemma-3-270m-it-UD-Q4_K_XL.gguf` artifact contains no IQ
tensors. Compared with the standard recipe, five FFN-down tensors use Q5_K;
the remaining tensors use existing Q4_K, Q5_0, Q6_K, and Q8_0 providers.

Using the same 26-token input, FP32 KV cache, regular attention, sequential
decode, and llama.cpp-generated teacher-forced continuation:

- 256/256 full-vocabulary logit rows are bit-exact.
- 256/256 top-1 choices match.
- Repeated uncaptured execution is bit-exact.
- Aggregate X-Ray capture is neutral and bit-exact.

Evidence: `/data/cke/ud-certification/reports/ud-q4-k-xl-f32-matched.json`.
This certifies this specific mixed-format recipe. It does not imply support for
unrelated UD recipes containing IQ formats without registered providers and
oracle coverage.

## Extended Boundary And Blast-Radius Checks

A 520-token UD prefill capture remains exact through layer-0 input, Q/K
normalization, RoPE, and MLP-down. Five output rows differ from llama.cpp:
positions 152, 380, 430, 509, and 514. The layer-output RMSE is approximately
`1.09e-7`, with maximum error `3.0518e-5`. At position 152, query head 1 has
eleven one-ULP Q*K score differences; heads 0, 2, and 3 are exact. Four-way
AVX-512 accumulation and explicit FP16 materialization did not improve the
result and were reverted. This does not invalidate the accepted 256-row
trajectory, but longer-context bit-exact certification remains open.

The Gemma-specific numerical providers are selected only by Gemma3's
`fp32_llama_regular` contract. Shared-risk validation covers the thread pool
and generic `sliding_window=0` lowering. Current branch results:

- Dense Qwen3.8 contracts: 79 tests and 11 subtests passed.
- Qwen3.8 Flash contracts: 128 tests, 2 skipped, and 2 subtests passed.
- Official Qwen3.6 configuration/inventory contracts: 9 tests passed.
- Xeon/Gemma4 compiler and circuit contracts passed.
- Thread-pool regression: 308,713 assertions passed.
- Fresh real-model generation is coherent for Qwen2 0.5B, Qwen3 0.6B,
  Qwen3.5 0.8B, Nanbeige 4.1 3B, and Qwen3.8 dense 27B.

The exact `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` artifact is not certified. Current
main fails lowering because its MoE scratch plan is 8,192 bytes short. This
branch fixes that planner bound, but fresh generation then emits a repeating
punctuation sequence. Treat that as a separate full-model numerical failure;
passing family contracts does not override it.

Gemma4 E4B initially exposed a provider-selection collision: the new
llama-matched GeGLU implementation was globally marked `production`, making it
compete with Gemma4's established exact provider. The matched implementation
is now a `candidate` selected explicitly by Gemma3. A fresh forced Gemma4
conversion/build then generated coherent text at 6.02 prompt tok/s and 9.02
decode tok/s on the P3. Tests assert that Gemma4 cannot inherit the Gemma3
normalization or GeGLU providers.

Fresh large-artifact checks on Ryzen also generated coherent text for Qwen3.5
35B-A3B (14.69 prompt tok/s, 14.71 decode tok/s) and dense Qwen3.8 27B (1.86
prompt tok/s, 2.43 decode tok/s). These are blast-radius coherence checks, not
new full-logit certifications. A forced Qwen3.8 Flash conversion found a
missing standard-library `re` import in the Qwen4 quantization-summary path;
that converter defect is fixed and the real 120B rebuild remains in progress.

The complete architecture-contract target passes after regenerating the kernel
registry. Its v7-seed comparison now excludes v8 numerical-provider and
per-layer RoPE schedule declarations, while retaining topology comparison;
those excluded declarations remain covered by dedicated provider, ABI, and
Gemma contract tests.

## Cross-Family Fresh-Rebuild Findings

The first complete fast-regression pass exposed a physical-layout naming fault
in the new Gemma3 regular-attention map. Its K/V inputs are compact local
`[KV,T,D]` tensors produced by the existing token-to-head transpose, whose
registered layout is `head_major_contiguous`; the map had introduced the
unregistered synonym `head_major_compact`. The map now uses the established
layout vocabulary. A route test covers Q5_1 projections into both K and V, and
fresh Gemma3 build, coherent generation, repeatability, and runtime-contract
checks all pass.

The real 120B Qwen3.8 Flash rebuild exposed an independent terminal-row planner
fault. At context 512 the circuit retains its default 4096-token prefill chunk,
but memory is correctly allocated for only 512 execution rows. Terminal-row
validation incorrectly used the larger default chunk as the live extent and
rejected a correctly sized buffer. The validated extent is now
`min(context_length, prefill_chunk_length)`, with an explicit regression test
and detailed buffer/shape diagnostics on failure. The same cached 119 GB model
then completed IR lowering, compilation, loading, and coherent generation. The
short 23-token prompt measured 6.08 prompt tok/s and 6.88 decode tok/s for 32
generated tokens on the 16-thread Ryzen route. This is a fresh execution check,
not long-context or BF16 certification.

The complete fresh P3 fast-regression matrix subsequently passed for Gemma3,
Qwen2, Qwen3, Qwen3.5, and Nanbeige. Every row rebuilt from its GGUF artifact,
executed smoke prompts, and passed its runtime contracts. Nanbeige retained its
known non-blocking strict-coherence warning while producing structured text.
The report is
`~/.cache/ck-engine-v8/regression/reports/20260906_010953/summary.json`.

An exact-artifact tokenizer-free check of
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` confirms the separately tracked numerical
failure at the first complete vocabulary row: CKE selected token 248312 while
llama.cpp selected token 11, cosine similarity was 0, RMSE was 1.909259, and
the top-20 sets had no overlap. The planner correction exposes this failure but
does not cause or solve it. Qwen3.6 35B-A3B must remain uncertified pending its
own X-Ray diagnosis.
