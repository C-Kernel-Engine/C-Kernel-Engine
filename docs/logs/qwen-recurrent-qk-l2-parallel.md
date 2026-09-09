# Qwen recurrent Q/K L2 normalization parallelization

## Scope

This prototype partitions complete `(Q-or-K, row, head)` normalization jobs
through the existing CK thread pool. Each head retains the original scalar
FP64 accumulation, FP32 square-root, epsilon clamp, and element update order.
No reduction is split across workers. Small, decode, nested, and overlapping
Q/K ranges use the original serial loops.

## Diagnosis

The retained Qwen3.5 0.8B Q4_K_M 1K profile attributed about 39 ms across 18
recurrent layers to `recurrent_qk_l2_norm_forward`, at one-core occupancy.
The provider map described independent rows and heads but the implementation
was serial.

## Correctness evidence

- Native tests compare one-thread and four-thread results over 257 rows,
  unequal Q/K head counts, and eight repetitions. Every float is identical.
- The llama.cpp production oracle includes a 65-row case above the dispatch
  threshold. Five shapes pass exactly at 1, 16, 20, and 24 threads: 20/20.
- Three baseline and three candidate full-model runs retain first-logit hash
  `7fedeeccedd89648` and token IDs `[100, 100]`.

## P3 measurements

Host: Intel Core i7-14700T, AVX2/FMA, 16 physical-core representatives.
Artifact: Qwen3.5 0.8B Q4_K_M, bump SHA-256
`9297be150ac361e95e27387f5bd4b17a55a24c5428467e64bb304734f6be097d`.
Input: 1,024 repeated token IDs, two generated IDs, context capacity 2,048.

Isolated production shape (`rows=1024`, `q_dim=k_dim=2048`,
`head_dim=128`):

| Runtime | Median |
| --- | ---: |
| Baseline, 1 thread | 2.142 ms |
| Candidate, 1 thread | 2.109 ms |
| Candidate, 4 threads | 0.975 ms |
| Candidate, 8 threads | 0.757 ms |
| Candidate, 16 threads | 0.404 ms |

The first prototype mapped serial work through the unified head-job loop and
regressed one-thread execution by about 8%. That version was rejected. The
accepted prototype retains the original nested serial loops whenever it does
not dispatch multiple workers.

Initial isolated-change full-model runs:

| Runtime | Prefill runs | Median | Q/K L2 median |
| --- | --- | ---: | ---: |
| Baseline | 2460.691, 2388.368, 2368.618 ms | 2388.368 ms | 38.307 ms |
| Candidate | 2360.501, 2355.096, 2358.118 ms | 2358.118 ms | 8.948 ms |

The candidate reduced Q/K normalization wall time by about 4.3x and median
end-to-end prefill by 1.3%. Raw logs and token traces are retained under
`/data/cke/profiles/qwen-recurrent-qk-norm-p3/`.

After recurrent-state preparation and recurrent SiLU merged, the candidate
was rebased onto exact main `c1702bb56`. Five interleaved runs per side gave:

| Runtime | Prefill samples | Median | Q/K L2 median |
| --- | --- | ---: | ---: |
| Merged main | 2160.170, 2202.574, 2171.823, 2149.715, 2157.043 ms | 2160.170 ms | 40.537 ms |
| Combined candidate | 2140.459, 2117.058, 2097.513, 2113.447, 2128.990 ms | 2117.058 ms | 9.002 ms |

The Q/K provider is about 4.5x faster in this combined stack and median
end-to-end prefill improves 2.0%. Baseline engine SHA-256 is
`d865e043cf826603b1a21267d01b695e435346fe5b69c4dfd306796d1ce1e5b6`;
candidate engine SHA-256 is
`824c41b3c5716a70bb5a29d38ffd250583c67c2cbcc5b7ac4a7359f64c2227be`.
Raw combined evidence is under `/data/cke/profiles/qwen-recurrent-combined-p3/`.

Ryzen and long-context measurements remain open.
