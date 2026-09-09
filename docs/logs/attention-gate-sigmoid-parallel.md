# Attention gate sigmoid parallelization

## Scope

The llama.cpp-compatible FP32 attention gate now schedules complete token
rows through the persistent CK thread pool for sufficiently large prefill
calls. Each output retains the existing scalar `libm.so.6` `expf` call and
multiplication order. Decode, small calls, nested dispatch, and unsupported
overlap cases retain serial execution. The backward and PyTorch BF16-storage
providers are unchanged.

The forward provider is consumed by Qwen3.5, Qwen3.8 dense, and Instella MoE
circuits. This change does not affect the separately selected Qwen3.8 Flash
PyTorch BF16-storage gate provider.

## Correctness evidence

- The native test compares one-thread and four-thread output bit-for-bit over
  257 rows, 16 heads, and 128 values per head. Separate and exact in-place
  output are repeated eight times.
- A production-sized 65 by 2,048 output comparison is bit-exact to the pinned
  llama.cpp kernel oracle at 16 threads.
- Five baseline and five candidate Qwen3.5 full-model runs retain first-logit
  hash `7fedeeccedd89648` and token IDs `[100, 100]`.

## P3 measurements

Host: Intel Core i7-14700T, AVX2/FMA, 16 physical-core representatives.
Artifact: Qwen3.5 0.8B Q4_K_M, bump SHA-256
`9297be150ac361e95e27387f5bd4b17a55a24c5428467e64bb304734f6be097d`.
Input: 1,024 repeated token IDs, two generated IDs, context capacity 2,048.

An isolated 1,024 by 2,048 call measured 7.391 ms on the serial merged-main
provider and 2.677 ms at 16 threads. The candidate's one-thread median was
7.440 ms. Output SHA-256 prefixes were identical.

Back-to-back full-model runs against merged main `c1702bb56` gave:

| Runtime | Prefill samples | Median | Gate median |
| --- | --- | ---: | ---: |
| Merged main | 2171.237, 2167.665, 2151.194, 2136.443, 2165.770 ms | 2165.770 ms | 46.844 ms |
| Candidate | 2114.964, 2145.552, 2088.240, 2108.758, 2092.054 ms | 2108.758 ms | 4.933 ms |

The gate region is about 9.5x faster and median end-to-end prefill improves
2.6%. Baseline engine SHA-256 is
`d865e043cf826603b1a21267d01b695e435346fe5b69c4dfd306796d1ce1e5b6`;
candidate engine SHA-256 is
`fe4c1af0514d5120038d6393363f7fa7c78c6ec2abca5196790ff4db67653cd4`.
Raw traces are retained under
`/data/cke/profiles/attn-gate-sigmoid-p3/`.

The Qwen3.5 template guard has one pre-existing failure on merged main because
the recurrent operation list lacks `post_attention_norm`; this change does not
alter templates. Instella generated-IR coverage passes when its temporary
files are placed on `/data`; `/tmp` exceeded the P3 user quota during the first
run, and that infrastructure failure is retained separately.

Ryzen, long-context, Instella full-model performance, and matched llama.cpp
end-to-end performance remain open.
