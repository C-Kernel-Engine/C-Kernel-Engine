# Qwen recurrent state preparation: independent-channel scheduling

Date: 2026-09-09

## Scope

`recurrent_conv_state_update_forward` copies recurrent convolution history,
converts token-major Q/K/V rows to channel-major convolution input, and extracts
the next history window. The operation previously advertised sequence-only
parallelism. Normal single-request inference has one sequence, so Qwen3.5 and
Qwen3.8 prefill executed this work serially.

The candidate dynamically partitions independent `(sequence, channel)` rows
through the persistent CKE thread pool. Each worker retains the original
token-major traversal within its channel tile. Decode, small calls, and nested
worker calls remain serial. No arithmetic or reduction order changes.

An initial channel-major source traversal was rejected: it preserved bytes but
made the production shape approximately 2.9 times slower due to strided Q/K/V
reads.

## Correctness

- PyTorch forward and backward tests pass.
- Serial and four-thread outputs are byte-identical across eight repetitions.
- Exact `state_out == state_in` aliasing is covered.
- Batched multi-token execution matches repeated one-token state continuation.
- Qwen3.8 dense contracts: 79 passed, including 11 subtests.
- Qwen3.8 Flash contracts: 130 passed, 2 skipped, including 2 subtests.
- Shared Qwen/template/numerical suite: 117 passed, 1 pre-existing v7 template
  failure, including 248 subtests. The failure is unchanged by this patch and
  concerns conditional `post_attention_norm` representation.

A 1,088-token C-source prompt on Qwen3.5 0.8B produced the same full-vocabulary
first-logit FNV-1a hash (`0596d81bddb3e8f2`) and the same four generated token
IDs (`713, 2153, 348, 74`) in baseline and candidate runs.

## P3 measurements

Host: Intel Core i7-14700T, 20 physical / 28 logical CPUs, AVX2+FMA. Runs used
16 physical-core representatives. Baseline source was current main at
`f0e4b01a2f269c5405c1095bf7a202c9532dcdaa`. The Qwen3.5 0.8B Q4_K_M bump
artifact SHA-256 was
`9297be150ac361e95e27387f5bd4b17a55a24c5428467e64bb304734f6be097d`.

Isolated production shape (`H=3, S=1, T=1024, Q=K=V=2048`), median of 15:

| Threads | Time |
|---:|---:|
| Baseline | 8.01 ms |
| Candidate 1 | 7.99 ms |
| Candidate 4 | 1.56 ms |
| Candidate 8 | 1.22 ms |
| Candidate 16 | 0.64 ms |

Interleaved full-model 1K prefill, three runs per side:

| Runtime | Samples | Median |
|---|---|---:|
| Current main | 2432.2, 2425.8, 2431.1 ms | 2431.1 ms |
| Candidate | 2213.4, 2212.5, 2214.2 ms | 2213.4 ms |

This is an 8.95% median prefill reduction. Every run retained first-logit hash
`7fedeeccedd89648` and token IDs `100, 100`.

The repeated-token fixture is suitable for controlled execution timing and
exact baseline/candidate comparison, not language-quality evaluation. The raw
operation profiler dropped early prefill rows at its fixed capacity, so the
model-level timing is the performance acceptance evidence; no candidate
per-operation total is inferred from the incomplete trace.

Raw evidence is retained under
`/data/cke/profiles/qwen-recurrent-prep-p3/` on the P3 node.
