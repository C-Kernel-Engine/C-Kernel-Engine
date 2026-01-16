# v6.6 Parallelism Plan: Matching llama.cpp Performance

## Problem Statement

**Current performance gap:**
- llama.cpp: ~30 tok/sec (prefill AND decode)
- CK-Engine: ~8 tok/sec
- **Gap: ~4x slower on same hardware**

**Root cause:** Our quantized GEMV kernels run single-threaded while llama.cpp parallelizes at orchestration level.

## llama.cpp Parallelization Pattern

From `ggml/src/ggml-cpu/ops.cpp`:

```c
const int ith = params->ith; // thread index (0 to nth-1)
const int nth = params->nth; // number of threads

// Work splitting pattern:
const int nr = total_rows;
const int dr = (nr + nth - 1) / nth;  // rows per thread (ceiling division)
const int ir0 = dr * ith;              // start row for this thread
const int ir1 = MIN(ir0 + dr, nr);     // end row for this thread

// Each thread processes only its rows
for (int row = ir0; row < ir1; row++) {
    // kernel work
}
```

**Key insight:** Parallelism is at orchestration level, not inside kernels!

## Our Architecture (Correct Approach)

```
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION LAYER (OpenMP here)                               │
│                                                                  │
│  #pragma omp parallel num_threads(nth)                          │
│  {                                                               │
│      int ith = omp_get_thread_num();                            │
│      int nth = omp_get_num_threads();                           │
│                                                                  │
│      // Each kernel call receives ith, nth                       │
│      gemv_q4k(W, x, out, M, K, ith, nth);                       │
│      rmsnorm(in, weight, out, dim, ith, nth);                   │
│      ...                                                         │
│  }                                                               │
│  #pragma omp barrier                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  KERNEL LAYER (NO OpenMP - single-threaded per call)            │
│                                                                  │
│  void gemv_q4k(..., int ith, int nth) {                         │
│      int dr = (M + nth - 1) / nth;                              │
│      int r0 = dr * ith;                                         │
│      int r1 = MIN(r0 + dr, M);                                  │
│                                                                  │
│      // Only process rows [r0, r1)                               │
│      for (int row = r0; row < r1; row++) {                      │
│          // SIMD kernel work                                     │
│      }                                                           │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Add ith/nth to Kernel Signatures (Non-Breaking)

Add overloaded versions with thread parameters:

```c
// Existing (keep for compatibility)
void gemv_q4k(const void *W, const float *x, float *out, int M, int K);

// New threaded version
void gemv_q4k_threaded(const void *W, const float *x, float *out,
                       int M, int K, int ith, int nth);
```

**Files to modify:**
- `src/kernels/gemm_kernels_q4k.c`
- `src/kernels/gemm_kernels_q4k_q8k.c`
- `src/kernels/gemm_kernels_q6k.c`
- `src/kernels/gemm_kernels_q8_0.c`
- `src/kernels/rmsnorm_kernels.c`
- `src/kernels/rope_kernels.c`
- `src/kernels/softmax_kernels.c`
- `src/kernels/attention_kernels.c`

### Phase 2: Parallel Orchestration

Modify `ckernel_orchestration.c` to use thread pool:

```c
// Per-token decode with parallelism
void ck_decode_token_parallel(CKContext *ctx, int token_idx) {
    const int nth = omp_get_max_threads();

    #pragma omp parallel num_threads(nth)
    {
        const int ith = omp_get_thread_num();

        // Each layer
        for (int layer = 0; layer < n_layers; layer++) {
            // RMSNorm (parallel across hidden_dim)
            rmsnorm_threaded(hidden, rms_w, normed, hidden_dim, eps, ith, nth);
            #pragma omp barrier

            // QKV projection (parallel across output rows)
            gemv_q4k_threaded(W_q, normed, q, q_dim, hidden_dim, ith, nth);
            gemv_q4k_threaded(W_k, normed, k, kv_dim, hidden_dim, ith, nth);
            gemv_q4k_threaded(W_v, normed, v, kv_dim, hidden_dim, ith, nth);
            #pragma omp barrier

            // RoPE (parallel across heads)
            rope_threaded(q, k, pos, head_dim, n_heads, n_kv_heads, ith, nth);
            #pragma omp barrier

            // Attention (parallel across heads)
            attention_decode_threaded(q, k_cache, v_cache, out,
                                      n_heads, seq_len, head_dim, ith, nth);
            #pragma omp barrier

            // Output projection
            gemv_q4k_threaded(W_o, attn_out, hidden_add, hidden_dim, q_dim, ith, nth);
            #pragma omp barrier

            // MLP (parallel across intermediate_dim)
            gemv_q4k_threaded(W_gate, normed2, gate, inter_dim, hidden_dim, ith, nth);
            gemv_q4k_threaded(W_up, normed2, up, inter_dim, hidden_dim, ith, nth);
            #pragma omp barrier

            swiglu_threaded(gate, up, swiglu_out, inter_dim, ith, nth);
            #pragma omp barrier

            gemv_q4k_threaded(W_down, swiglu_out, hidden_add, hidden_dim, inter_dim, ith, nth);
            #pragma omp barrier
        }
    }
}
```

### Phase 3: Prefill Parallelism (Biggest Win)

For prefill, parallelize across TOKENS (not just within operations):

```c
void ck_prefill_parallel(CKContext *ctx, int *tokens, int n_tokens) {
    // Process multiple tokens in parallel
    #pragma omp parallel for
    for (int t = 0; t < n_tokens; t++) {
        // Each thread processes one token through embedding
        embed_token(tokens[t], hidden[t]);
    }

    // Then parallel layer processing
    for (int layer = 0; layer < n_layers; layer++) {
        // Parallel across tokens AND hidden_dim
        ...
    }
}
```

## Work Splitting Strategy by Operation

| Operation | Split Dimension | Notes |
|-----------|-----------------|-------|
| GEMV (M×K @ K) | Output rows (M) | M = q_dim, kv_dim, inter_dim |
| RMSNorm | Hidden dimension | Reduction requires final sum |
| RoPE | Heads | Each head independent |
| Attention | Heads (GQA-aware) | n_heads=14, n_kv=2 for Qwen2-0.5B |
| SwiGLU | Intermediate dim | Element-wise, easy split |
| Softmax | Sequence length | Reduction for normalization |

## Benchmark Results (2026-01-16)

### GEMV Q4_K_Q8_K Performance - Full Comparison

| Configuration | Q proj (896x1024) | K proj (128x1024) | Gate (4864x1024) | Down (896x5120) |
|---------------|-------------------|-------------------|------------------|-----------------|
| **SIMD (1T)** | 141 us | 31 us | 1100 us | 945 us |
| Reference (1T) | 293 us | 68 us | 1807 us | 1497 us |
| Parallel ref (4T) | 93 us | 19 us | 572 us | 684 us |
| **Parallel SIMD (4T)** | **58 us** | **10 us** | **380 us** | **332 us** |

### Speedup vs Single-Threaded SIMD

| Operation | Parallel SIMD (4T) Speedup |
|-----------|---------------------------|
| Q projection (896×1024) | **2.44x** |
| K projection (128×1024) | **3.11x** |
| Gate projection (4864×1024) | **2.89x** |
| Down projection (896×5120) | **2.85x** |

**Key Findings:**
1. **Parallel SIMD (4T) is 2.4-3.1x faster than single-threaded SIMD**
2. **Parallel SIMD is 1.5-2x faster than parallel scalar**
3. **4 threads optimal** - 8 threads hits memory bandwidth limits
4. **Prefetching helps** - Row-ahead prefetch hides ~50-70ns latency

### Projected Performance with Full Parallel Orchestration

With all operations parallelized using `gemv_q4_k_q8_k_parallel_simd`:

| Phase | Current | Projected | Improvement |
|-------|---------|-----------|-------------|
| Decode | 8 tok/s | **20-25 tok/s** | 2.5-3x |
| Prefill | 8 tok/s | **50-100 tok/s** | 6-12x |

**Target: Match llama.cpp at ~30 tok/sec** (within 70% now)

## Files to Create/Modify

### New Files
- `src/v6.6/parallel_orchestration.c` - Thread pool orchestration
- `include/parallel_kernels.h` - Threaded kernel declarations

### Modified Files
- `src/kernels/gemm_kernels_q4k.c` - Add `_threaded` versions
- `src/kernels/gemm_kernels_q4k_q8k.c` - Add `_threaded` versions
- `src/kernels/rmsnorm_kernels.c` - Add `_threaded` versions
- `src/kernels/rope_kernels.c` - Add `_threaded` versions
- `src/kernels/attention_kernels.c` - Add `_threaded` versions
- `src/ckernel_orchestration.c` - Add parallel decode path

## Testing Strategy

1. **Correctness**: Compare output against single-threaded version
2. **Performance**: Benchmark with 1, 2, 4, 8 threads
3. **Scaling**: Measure speedup vs thread count
4. **Comparison**: Match llama.cpp tok/sec

## Priority Order

1. **GEMV Q4_K** - Biggest time consumer
2. **Attention decode** - Second biggest
3. **RMSNorm** - Quick win
4. **MLP (SwiGLU + down proj)** - Third biggest
5. **Prefill** - Massive win for long prompts

## Risks

1. **Overhead**: Thread spawn/barrier overhead for small operations
2. **Memory bandwidth**: May hit memory limits before compute
3. **False sharing**: Cache line conflicts between threads
4. **Load imbalance**: Uneven work distribution

## Mitigation

- Use persistent thread pool (not per-operation spawn)
- Align output buffers to cache lines (64 bytes)
- Dynamic scheduling for uneven workloads
- Minimum work threshold before parallelizing
