# Qwen3.8 Flash Next Bring-up

Date: 2026-09-04

This note records the Qwen3.8 Flash Next text-runtime bring-up in v8. It is
an implementation and numerical-parity record, not a long-context quality or
performance certification.

## Model and oracle

- Model family: Qwen3.8 Flash Next / GGUF architecture `qwen4exp`
- Quantized artifact: Q4_K_M, approximately 120 GiB resident runtime image
- Hardware: AMD Ryzen 9 9950X3D, 160 GB DDR5
- Oracle: a local llama.cpp build with the matching Qwen architecture support
- Runtime topology: 48 layers, recurrent DeltaNet layers interleaved with 12
  sparse full-attention layers, PLE, four hyper streams, 512 routed experts,
  ten selected experts per token, and one shared expert

## Bring-up path

1. Added explicit safetensors and GGUF architecture contracts. The importers
   preserve layer kinds, PLE ownership, hyper-stream dimensions, sparse
   attention placement, expert topology, RoPE metadata, and tensor storage.
2. Added a dedicated `qwen4_exp` circuit. Model topology lives in the circuit
   and model maps; provider implementations remain reusable kernel-map entries.
3. Added provider maps and C kernels for PLE, hyper-stream mixing/injection,
   QSA attention, recurrent gates and normalization, and packed/shared MoE.
4. Extended generated prefill and decode stitching for the declared composite
   operations. The compiler resolves concrete providers from shape, storage,
   phase, and numerical contracts.
5. Added ChatML/thinking protocol handling from model metadata so end-to-end
   generation uses the checkpoint's embedded conversation contract.

## General defects exposed

### Hierarchical weight names in generated C

Per-layer offset members were originally derived from weight suffixes without
the same identifier normalization used for header offsets. A valid name such
as `attn_hyper.mix_down` therefore produced an invalid C member. Code generation
now sanitizes these names and resolves collisions deterministically. This is a
general generated-C fix rather than a model-name branch.

### Ambiguous provider fallback

Early lowering could classify an absent selected provider as a KV-cache case.
Provider resolution now distinguishes a missing registry entry from a real
cache operation and fails closed when the declared operation cannot be
implemented by the selected map.

### Runtime provenance during X-Ray

Model-sized debugging initially allowed a regenerated model library to be
paired with stale engine/tokenizer libraries. X-Ray now verifies the runtime
bundle stamp, hashes all three libraries, and rejects mixed binaries before
capturing tensors. It also records the selected prefill/decode schedules.

### Hybrid prefill capture mode

The first-token comparison harness did not consistently propagate the requested
CK batched/hybrid prefill mode into its child process. The mode is now explicit
and covered by tests, so a sequential replay cannot be mislabeled as batched
prefill evidence.

### Quantized sigmoid and SiLU order

The quantized recurrent and hyper paths needed llama.cpp-compatible evaluation
and storage order around sigmoid and SiLU. Dedicated numerical providers now
carry those contracts instead of placing model-specific arithmetic in lowering.

### Shared-expert prefill reduction order

The final layer-0 mismatch appeared only in prompt row zero. Router logits,
routing weights, routed experts, and the shared scalar gate were exact. The
shared Q4 gate/up projections differed by approximately `1e-6`; that small
difference crossed a later Q8_0 quantization boundary and produced a
`0.0015434325` shared-expert output difference and a `0.0007214546` combined
output difference.

The cause was scheduling, not weight conversion: CKE called the projection as
one-row GEMV operations while llama.cpp grouped four prompt rows in its GEMM
path. Calling the same CKE projection with `M=4` reproduced the oracle exactly.
The shared-expert provider now quantizes and projects complete four-row groups,
then applies split SwiGLU, down projection, shared gate, and routed addition per
row. A dedicated workspace-size function and independent staged regression test
guard this schedule.

## Numerical evidence

- Layer 0 recurrent X-Ray: 641 comparable checkpoints exact across 40 real chat
  prompt rows; no value divergence.
- Historical layer 3 attention X-Ray: 600 comparable checkpoints exact in a
  one-token-prefix replay; this was not a 40-token prompt certification.
- Greedy trajectory: 256/256 steps bit-exact across all 248,320 vocabulary
  logits, with identical top-5 sets and no first divergence.
- Real-chat trajectory: a 40-token batched prompt followed by 64 persistent
  decode steps matched all 64 greedy choices, but zero steps were bit-exact.
  The first maximum absolute logit difference was approximately 0.928.
  This remains an unresolved numerical gate; the harness's top-1 pass status
  must not be interpreted as full-logit parity.
- Post-contract runtime rebuild: a one-token prefix followed by 64 persistent
  decode comparisons was bit-exact across all 248,320 vocabulary logits.
  The configured context capacity was 8192; this did not exercise 8192 tokens.
- Focused compiler, X-Ray, chat, codegen, and numerical suite: 200 tests plus
  five subtests passed.

Eight hyper-connection boundaries are not yet emitted by the optimized batched
prefill capture route. They are reported as missing diagnostics and are not
counted as exact. Selected internal captures and the one-token-prefix decode
trajectory are exact. Batched chat prefill still needs full-logit investigation;
adding the missing exports remains observability work.

## Remaining certification work

0. Resolve the remaining batched-prompt logit drift after the explicit rebuild.
   The post-contract runtime had no compiled batched-prefill capability. X-Ray
   rejected it, while the trajectory runner incorrectly labeled its execution
   hybrid. The trajectory runner now checks that capability before model loading
   and offers `--require-bit-exact` to fail on full-logit differences even when
   greedy token choices match. Historical mismatched-schedule results above are
   retained as debugging evidence, not certification.
1. Measure short, medium, and 128K/256K context performance, peak memory,
   provider selection, and sustained CPU occupancy against llama.cpp.
2. Generate long-form C and SVG artifacts from a repository-grounded prompt,
   retain raw responses, and validate extracted C/SVG syntax separately.
3. Repeat the quality prompts to distinguish model behavior from one sampled
   response. Human review remains the quality gate.
4. Extend optimized-prefill hidden exports so all declared hyper boundaries can
   be compared directly by X-Ray.

## Integration and regression gates

Integrated main at `e87b90d0f`, including explicit audio projection input contracts
and validated circuit-owned MLA providers. Flash Next now declares projection
inputs in its own circuit. The shared compiler keeps upstream's fail-closed
behavior; legacy MLA provider-name fallbacks were not reintroduced.

`make test-v8-qwen38-flash-contracts` is registered as a nightly parity row. On
the P3 native AVX2/AVX-VNNI build its first run passed 111 tests and two subtests;
two AVX-512-only sparse BF16 checks were skipped. The merged compiler, Flash,
MLA, numerical contract and audio suites passed 139 tests and 145 subtests.
The kernel-map gate passed 167 tests and 1199 subtests. These are component
gates, not full BF16 model certification.

The oneDNN 3.7 and 3.12 providers share an allocating implementation. Both maps
declare its allocation helper so the audit retains that debt. There remain
51 production allocator call sites; three mapped provider entries now expose
allocation debt, including the new versioned provider. No allocation reduction
is claimed from extracting that shared implementation.

## Completed all-layer capture (September 4)

The native Ryzen runtime was rebuilt with generated batched prefill. The
matched-schedule 40-token prompt plus 64 decode comparisons still matched all
64 greedy choices but none of the full-vocabulary rows bit-exactly. Step zero
had maximum absolute logit difference 0.951632738 and RMSE 0.174499848.
Fixing schedule validation was necessary, but did not resolve numerical drift.

One CKE capture and one llama.cpp capture then covered requested boundaries
across all 48 layers, using the same 40-token prompt and two decode tokens.
Context capacity was 8192, not 8192 consumed tokens. Runtime binary provenance
was verified before capture.

- Layers 0-5: all comparable checkpoint rows exact; missing/incompatible
  checkpoints remain untested and are not counted as passes.
- Earliest observed mismatch by layer: layer 6, logical token 10,
  `moe_combined_output`, maximum absolute difference 1.932727173e-6.
- Layer 7: additional combined-MoE differences and a material `q_proj`
  mismatch at logical token 10. Later layers show accumulated drift.
- This localizes the next investigation to the layer-6 expert combination
  and its inputs; it does not yet prove the underlying cause.

Artifacts on Ryzen: `xray-chat40-all-layers-main-e87/summary.json` and
`layer-00.json` through `layer-47.json` under
`/home/antshiv/.cache/cke-nvme/qwen38-flash-next/`.
Capture wall times were 134.37 seconds for CKE and 19.98 seconds for the oracle.
These include loading and diagnostic dumps, so are not matched inference
throughput measurements. Capture is complete; no background certification run
is implied by these results.

Commit hooks completed v7 and v8 fast regression targets. The v8 report's
overall PASS still contains a Nanbeige coherence FAIL; it must not be described
as an all-model coherence pass.

## Shared-expert and contract follow-up

Layer 6 uses Q4_K gate/up and Q5_0 down weights. Its scalar gate projection
and sigmoid matched the oracle, as did routed experts. Replaying the captured
MLP input through the shared expert reproduced the row-10 discrepancy alone.
The Q5_0-down implementation still invoked the Q4_K projections one row at a
time; only the Q8_0-down variant had received the four-row prefill schedule.

Both variants now use one shared implementation, selecting only the down
projection function. Kernel maps declare the matching four-row workspace and
internal output-parallel dispatch. No model-name branch was added to lowering.
The isolated 40-by-2560 shared output now matches all 102400 values exactly,
both gated and ungated. This is kernel-level evidence; the full trajectory is
being rerun separately. The new staged test covers both down formats and rejects
an undersized workspace. Eight mixed-MoE tests passed; the complete Flash
component target passed 113 tests, two subtests, and two P3 ISA skips.

The scaffold test now separately checks the exact projection input declarations
before comparing v8 graph topology with its v7 seed. This retains both checks
instead of treating explicit v8 metadata as a topology regression. The combined
scaffold/audio tests passed 33 tests and 59 subtests; the complete
`v8-kernel-map-contracts` target passed, including 168 map tests and 1199 subtests.

The earlier Gemma 270M timeout did not reproduce in two direct attempts or six
additional fresh-process smoke runs alternating the two pre-push prompts.
Those used capacity 128, eight CKE threads, OMP=1, and five output tokens.
The first repeated run included a rebuild (60.35 seconds); subsequent runs
took approximately 2.17 seconds each. Its original cause is not established,
so this is successful revalidation, not proof of a repaired timeout bug.

## Strict replay follow-up (2026-09-05)

After the shared-expert fix, the real 40-token prompt followed by 64 full-logit
comparisons has 63 exact rows. Only the initial prefill logit row differs:
maximum absolute error 2.86102294921875e-6, RMSE 3.523921538602865e-7.
All 64 greedy choices match, but the strict harness correctly returns FAIL.
Capacity is 8192; this is not an 8K consumed-context test.

The first follow-up all-layer capture contained 30116 exact comparisons,
15840 missing/incompatible comparisons, and no reported value differences.
That does NOT certify every boundary: prefill lacked the hyper exports present
in decode, and the row loader cannot interpret the oracle's final-layer
last-row-only tensors as full batches. Prefill now emits corresponding hyper
checkpoints, with full and last-row widths checked by codegen tests.

Direct final-layer replay bypassing that diagnostic shape limitation isolates
the remaining first mismatch to the FP32 router: its input is exact, but 267
of 512 router logits differ by at most 1.9073486328125e-6. All ten routing
weights differ slightly. The routed output differs by at most
4.470348358154297e-8 and the combined expert output by 5.960464477539063e-8.
The hyper-stream input and injection weights match; reconstructing injection
with the captured expert output reproduces CKE's final hyper stream exactly.

The oracle prunes to its requested last row after final attention; CKE retains
the 40-row final MLP batch. An isolated replay with identical input and weights
confirms that M=1 matches all 512 router logits exactly, while M=40 reproduces
all 267 differences. Matching this row-dependent reduction schedule and
teaching X-Ray about explicit output-row selection remain open. Do not relax
the exact gate or label missing comparisons as passes. Full BF16 and long
context certification remain separate work.
