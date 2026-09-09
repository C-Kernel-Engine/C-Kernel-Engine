# Qwen recurrent SiLU parallelization

## Scope

This change parallelizes independent rows in the validated
`recurrent_silu_forward_ggml` provider. It does not change the ggml-compatible
AVX2, AVX-512, or scalar-tail arithmetic within a row. Small, decode, and
nested thread-pool calls remain inline.

## Diagnosis

The retained Qwen3.5 0.8B Q4_K_M 1K prefill profile attributed about 47 ms
across 18 recurrent layers to this provider, at effectively one-core
occupancy. The kernel map advertised row parallelism, but the C provider was
serial.

## Numerical evidence

- The native unit test compares one-thread and four-thread execution over
  irregular dimensions for eight repetitions, both out-of-place and in-place.
  Every float is bit-identical.
- The independent llama.cpp production oracle passed four shapes at each of
  1, 16, 20, and 24 threads: 16/16 cases exact.
- Three baseline and three candidate full-model runs produced the same
  first-logit FNV-1a hash, `7fedeeccedd89648`, and token IDs `[100, 100]`.

## P3 measurements

Host: Intel Core i7-14700T, AVX2/FMA, 16 physical-core representatives.
Artifact: Qwen3.5 0.8B Q4_K_M, bump SHA-256
`9297be150ac361e95e27387f5bd4b17a55a24c5428467e64bb304734f6be097d`.
Input: 1,024 repeated token IDs, two generated IDs, context capacity 2,048.

Isolated production shape (`rows=1024`, `dim=6144`):

| Runtime | Median |
| --- | ---: |
| Baseline, 1 thread | 2.768 ms |
| Candidate, 1 thread | 2.738 ms |
| Candidate, 4 threads | 0.719 ms |
| Candidate, 8 threads | 0.647 ms |
| Candidate, 16 threads | 0.263 ms |

Interleaved full-model runs:

| Runtime | Prefill runs | Median | Recurrent SiLU median |
| --- | --- | ---: | ---: |
| Baseline | 2418.264, 2377.349, 2407.160 ms | 2407.160 ms | 47.167 ms |
| Candidate | 2335.634, 2346.533, 2368.050 ms | 2346.533 ms | 10.174 ms |

The candidate reduced recurrent SiLU wall time by about 4.6x and median
end-to-end prefill by 2.5%. Raw logs and token traces are retained under
`/data/cke/profiles/qwen-recurrent-silu-p3/`.

An initial A/B attempt was rejected because `LD_LIBRARY_PATH` caused both
runs to load the candidate engine. The accepted comparison gives each model
runtime its own baseline or candidate `libckernel_engine.so`; their hashes
are different while model, weights, generated C, CLI, affinity, and inputs
remain fixed.
