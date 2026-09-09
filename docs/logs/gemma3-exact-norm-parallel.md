# Gemma3 Exact Normalization Parallelism

Status: development candidate. This change preserves Gemma3's validated
llama.cpp-compatible arithmetic while distributing independent prefill rows
through CKE's persistent thread pool. It does not change decode arithmetic.

## Diagnosis

Gemma3 selected serial exact RMSNorm and Q/K normalization providers in
prefill. The shared exact parallel RMSNorm provider already existed, but the
Gemma3 circuit did not declare the numerical contracts needed for phase-aware
provider selection. Q/K normalization had no exact parallel prefill provider.

The circuit now declares exact RMSNorm and Q/K normalization contracts for
prefill and decode. Prefill selects independent-row parallel providers; decode
retains the serial providers. The new Q/K dispatcher delegates each Q and K
head-token row to the existing exact RMSNorm dispatcher, preserving the FP64
sum and every row's arithmetic order.

## Correctness Evidence

- Artifact: `unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf`
- SHA256: `0d29b1a23a4cb1f14fda28c9e63563fa4bbefb6f34db4f7e873bb72edd932b00`
- Prompt: 1,024 repeated token-100 IDs; SHA256
  `66ba880e75f680d093120fb7c5b722d76c58c7f51b0d76830f812164faf679cb`
- Hardware lane: P3, 16 physical CPUs selected by affinity.
- All six alternating baseline/candidate runs produced full-logit FNV-1a hash
  `e930f3e243039e4c` and token IDs `140437, 236896`.
- Generated prefill C selects the exact parallel RMSNorm and Q/K providers.
  Generated decode C retains the exact serial providers.
- Exact in-place Q/K tests cover multiple heads, KV heads, tokens and feature
  dimensions, with repeated multi-thread executions.

Focused validation passed 113 tests and 1,072 subtests. The broader family
sweep passed 186 tests and 44 subtests except for a pre-existing Qwen3.5 v7
template-guard assertion that assumes `post_attention_norm` is a string rather
than the current conditional object. Kernel-map tests passed 172 tests and
1,236 subtests before registry freshness was checked.

The complete `v8-regression-fast` production-path matrix then rebuilt Gemma3,
Qwen2, Qwen3, Qwen3.5 and Nanbeige. Build, smoke and contracts passed for all
five; Gemma3, Qwen2, Qwen3 and Qwen3.5 also passed coherence. Nanbeige retained
the known contradictory result `Coherence FAIL` with aggregate `PASS`; this is
an evidence-reporting defect and is not counted as clean Nanbeige validation.
The Qwen3.8 Flash contract gate passes 130 tests, two subtests and two declared
skips; it now asserts exact parallel FP32 Q/K normalization in prefill and the
serial provider in decode, while BF16 retains its existing provider in both
phases.

## P3 Alternating A/B

The measurements use the same artifact, input, runtime settings, CPU affinity
and two-token trajectory. Times are wall-clock milliseconds.

| Run | Prefill baseline | Prefill candidate | Q/K baseline | Q/K candidate | Four RMSNorm baseline | Four RMSNorm candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4925.1 | 4734.6 | 58.8986 | 7.4289 | 114.8277 | 20.2923 |
| 2 | 4931.8 | 4785.7 | 59.8876 | 8.2119 | 117.3420 | 20.3785 |
| 3 | 4918.5 | 4760.1 | 58.2476 | 8.1443 | 114.1962 | 20.5931 |

Median prefill improves from 4925.1 ms to 4760.1 ms, or 3.35%. Median Q/K
normalization improves by 7.23x. The four RMSNorm groups improve by 5.63x.

The profiler summary currently omits `attn_norm` and `ffn_norm` after they move
to the parallel provider. The table therefore sums all four operation labels
from the retained raw CSV events instead of relying on the grouped summary.

Evidence root:
`/data/cke/profiles/gemma3-norm-parallel-p3/ab`.

## Scope And Next Measurement

This branch is based on main before the separate Gemma3 dynamic-attention
change in PR #484. The gains must not be added arithmetically. After that
change lands, rebase and measure the combined generated runtime on P3 and
native Ryzen. Other circuits are affected only when they explicitly request
the same exact numerical contract; generic FP32 normalization providers remain
unchanged.
