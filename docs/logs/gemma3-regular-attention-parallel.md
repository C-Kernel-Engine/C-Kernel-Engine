# Gemma3 regular-attention parallelism

Date: 2026-09-09

## Scope

This change parallelizes independent head/query jobs in Gemma3's llama.cpp-
compatible FP32 regular-attention prefill provider. Each worker retains the
existing Q dot K, softmax, and probability dot V reduction order. The planner
provides 16 score/probability rows and one read-only transposed-value bank per
KV head. Decode and Gemma4 providers are unchanged, and old direct callers
with a single workspace row retain the serial path.

The provider performs no allocation. Compact K/V head stride remains separate
from the padded 256-token reduction capacity.

The first parallel candidate assigned jobs cyclically by worker index. On the
P3's hybrid CPU, E-core workers took about 2.2 times as long as P-core workers
for equal partitions. The final implementation instead lets each worker claim
the next independent head/query job from a per-call atomic counter. Worker IDs
still select disjoint score/probability scratch rows, and the dispatch completion
join remains required.

## Numerical evidence

- Direct kernel tests compare one and eight threads bit-for-bit with two
  observably distinct KV heads at 67 tokens and with global attention at 520
  tokens.
- A fresh Gemma3 270M Q5_K_M candidate runtime matched its parent runtime's
  complete first-logit FNV-1a hash and generated token IDs for the identical
  1,024-token profiler input.
- A separate 676-token literal prompt matched the parent first-logit hash
  `488ef2a50b72916f` and all 64 generated token IDs.
- Candidate runs at 1, 4, 8, and 16 threads retained that same hash and token
  trajectory.
- Dynamic scheduling is bit-exact to the static parallel implementation for
  local and global attention, multiple KV heads, repeated calls, and token
  lengths 67, 255, 256, 257, and 520.

## P3 performance evidence

Hardware: Intel Core i7-14700T, AVX2/FMA, physical-core CPU policy, 16 CKE
workers. Parent commit: `e9fb5d243`.

| Workload | Parent | Candidate | Improvement |
|---|---:|---:|---:|
| 1,024-token profiled prefill | 4,931.1 ms | 1,541.0 ms | 3.20x |
| Regular attention within that prefill | 3,963.1 ms | 566.6 ms | 6.99x |
| 676-token literal-prompt prefill | 2,603.6 ms | 1,088.1 ms | 2.39x |

Regular-attention occupancy increased from 1.01 to 10.76 core equivalents in
the 1,024-token profile. The candidate's 676-token thread sweep measured
6,357.9, 1,736.7, 1,090.2, and 1,012.9 ms at 1, 4, 8, and 16 threads.

Raw reports and traces are retained under
`/data/cke/profiles/utilization-current-main-p3` and
`/data/cke/profiles/utilization-gemma3-candidate-p3` on P3.

### Dynamic scheduling follow-up

For the isolated 4-head, 1-KV-head, 1,024-token, 256-dimension provider shape,
15-call medians were:

| Placement | Static | Dynamic | Change |
|---|---:|---:|---:|
| 8 P cores | 28.32 ms | 28.08 ms | 0.9% faster |
| 8 P cores with SMT, 16 workers | 24.44 ms | 24.46 ms | unchanged |
| 8 P + 8 E cores, 16 workers | 30.92 ms | 19.94 ms | 35.5% faster |

Mixed-core completion wait fell from 16.55 to 0.11 ms per provider call. All
six output hashes were identical. Raw JSON is retained under
`/data/cke/profiles/gemma3-attention-scheduler-p3`.

Three interleaved generated-runtime runs at 1,024 prompt tokens measured a
static median of 1,551.8 ms and dynamic median of 1,291.7 ms, a 16.8% prefill
improvement. Completion-wait medians fell from 454.4 to 152.2 ms. Decode
medians were 20.1 and 19.8 ms respectively. Every run retained first-logit hash
`e930f3e243039e4c` and token IDs `[140437, 236896]`. These corrected runs are
under `gemma3-attention-scheduler-p3/e2e-corrected`.

The first attempted A/B was discarded because the profiler prepended the
current worktree's `build/` directory ahead of the selected runtime directory
in `LD_LIBRARY_PATH`. That caused the static runtime to load the dynamic engine
despite its retained local library. The profiler now gives the runtime bundle
precedence, with a regression test covering that provenance requirement.

### Native Ryzen guard

The same commit was compiled natively with AVX-512/VNNI/BF16 on the 16-core,
32-thread Ryzen node. With one hardware thread per physical core (`0-15`), five
interleaved 25-call provider runs produced median-of-medians of 9.25 ms for
static scheduling and 5.66 ms for dynamic scheduling. All ten runs produced
the same output hash. Four dynamic medians were 5.10-6.46 ms; one complete
dynamic run was a 23.87 ms host-noise outlier and remains visible in the raw
evidence rather than being discarded. Static medians were 8.16-11.45 ms.

A separate dynamic width sweep measured 64.59, 9.51, 5.02, and 5.62 ms at 1,
8, 16, and 32 workers. SMT therefore did not improve this shape beyond the 16
physical cores. Raw native results are retained under
`/data/cke/profiles/gemma3-attention-dynamic-ryzen` and
`/data/cke/profiles/gemma3-attention-scheduler-ryzen-pinned` on Ryzen.

## Validation boundary

This establishes exact equivalence to the previously certified generated CKE
runtime for the tested trajectories; it does not constitute new 128K or
Gemma4 certification. One unrelated current-main Qwen3.5 template guard fails
because `post_attention_norm` is absent from the v7 template operation list.
