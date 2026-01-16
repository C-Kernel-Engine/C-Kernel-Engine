# Memory Bandwidth Analysis: Why Parallelism Has Limits

## Your System Specs

| Component | Spec | Bandwidth |
|-----------|------|-----------|
| CPU | i7-3630QM (Ivy Bridge) | 4 cores, 8 threads |
| Memory | DDR3-1600 dual channel | ~25 GB/s theoretical |
| L3 Cache | 6 MB shared | ~100 GB/s |
| L2 Cache | 256 KB per core | ~200 GB/s |
| L1D Cache | 32 KB per core | ~500 GB/s |

## Qwen2-0.5B Weight Sizes

```
Per layer (24 layers):
  Q projection:    896 × 896  × 0.5625 B/elem (Q4_K) =  452 KB
  K projection:    128 × 896  × 0.5625 B/elem       =   64 KB
  V projection:    128 × 896  × 0.5625 B/elem       =   64 KB
  O projection:    896 × 896  × 0.5625 B/elem       =  452 KB
  Gate proj:      4864 × 896  × 0.5625 B/elem       = 2.45 MB
  Up proj:        4864 × 896  × 0.5625 B/elem       = 2.45 MB
  Down proj:       896 × 4864 × 0.5625 B/elem       = 2.45 MB
  RMSNorm (×2):    896 × 4 bytes × 2               =  7.2 KB
  ─────────────────────────────────────────────────────────────
  Total per layer:                                  ≈ 8.4 MB

Total model weights: 24 × 8.4 MB ≈ 200 MB (Q4_K compressed)
```

## Single Token Decode: Memory Traffic

For **each token generated**, we must:

```
Load ALL weights once:  200 MB
Load KV cache:          seq_len × 1024 bytes (grows with context)
Load activations:       Negligible (fits in L2)
Write activations:      Negligible (fits in L2)
─────────────────────────────────────────────────────────────────
Total per token:        ~200 MB + KV cache reads
```

## Theoretical Performance Limit

```
Memory bandwidth:     25 GB/s
Weights per token:    200 MB
Minimum time/token:   200 MB / 25 GB/s = 8 ms

Theoretical max:      1000 ms / 8 ms = 125 tok/s

But you're getting:   8 tok/s
Efficiency:           8 / 125 = 6.4%
```

**Where's the other 93.6%?**

## The Latency Problem

DDR3 memory access isn't just about bandwidth - there's **latency**:

```
DDR3 latency:         ~50-70 ns per random access
Cache line:           64 bytes
Sequential read:      64 B / 50 ns = 1.28 GB/s per stream

To saturate 25 GB/s:  Need ~20 concurrent memory streams
```

**Problem:** Our GEMV kernel does sequential row access, but each row starts at a different address. The CPU's hardware prefetcher can't predict the pattern well.

## Why Parallelism Helps (a little)

With 4 threads, each thread:
- Issues its own memory requests
- Has its own L1/L2 prefetch streams
- Together they can better saturate memory bandwidth

```
1 thread:   ~6 GB/s  effective (24% of peak)
4 threads:  ~15 GB/s effective (60% of peak)
8 threads:  ~18 GB/s effective (72% of peak, diminishing returns)
```

This explains our benchmark results:
- 4 threads: ~2x speedup (1 thread was only using 24% bandwidth)
- 8 threads: No improvement (bandwidth saturated)

## Software Prefetching Strategy

### The Idea

```
Current flow (BAD):
┌────────────────────────────────────────────────────────────┐
│ Compute row 0 ──► STALL (wait for row 1) ──► Compute row 1 │
│                   ~50ns latency                            │
└────────────────────────────────────────────────────────────┘

With prefetch (GOOD):
┌────────────────────────────────────────────────────────────┐
│ Prefetch row 1 ──► Compute row 0 ──► row 1 ready! ──► ...  │
│ (no stall - data arrives during computation)               │
└────────────────────────────────────────────────────────────┘
```

### Prefetch Distance Calculation

```
Computation per row:     K × 2 FMAs = 1024 × 2 = 2048 ops
CPU throughput:          ~20 GFLOPS (single thread, Ivy Bridge)
Time per row:            2048 / 20e9 = 0.1 us = 100 ns

Memory latency:          50-70 ns
Prefetch distance:       1-2 rows ahead

Bytes per row:           K / 256 × 144 bytes (Q4_K) = 576 bytes
Cache lines per row:     576 / 64 = 9 cache lines
```

### Implementation

```c
void gemv_q4_k_prefetch(float *y, const void *W, const void *x,
                        int M, int K, int ith, int nth)
{
    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = MIN(r0 + dr, M);

    const int bytes_per_row = (K / QK_K) * sizeof(block_q4_K);
    const char *W_bytes = (const char *)W;

    // Prefetch first few rows
    for (int p = 0; p < 4 && r0 + p < r1; p++) {
        _mm_prefetch(W_bytes + (r0 + p) * bytes_per_row, _MM_HINT_T0);
    }

    for (int row = r0; row < r1; row++) {
        // Prefetch 4 rows ahead
        if (row + 4 < r1) {
            _mm_prefetch(W_bytes + (row + 4) * bytes_per_row, _MM_HINT_T0);
        }

        // Compute current row (data should be in L2/L1 now)
        y[row] = dot_q4_k_q8_k_simd(...);
    }
}
```

## Prefetching the Full Decode Path

Your idea of prefetching across operations is even better:

```
Timeline (per layer):
─────────────────────────────────────────────────────────────────────────
Time:   0    1    2    3    4    5    6    7    8    9   10   11   12
─────────────────────────────────────────────────────────────────────────
Prefetch: [Q weights] [K weights] [V weights] [Gate]  [Up]   [Down]
Compute:       [RMSNorm] [Q proj]  [K proj]   [V proj] [Gate] [Up] [Down]
─────────────────────────────────────────────────────────────────────────
```

### Contiguous Arena Advantage

Since your memory is in one contiguous arena:

```c
// Weight layout in arena (simplified)
struct LayerWeights {
    block_q4_K W_q[Q_DIM * EMBED / QK_K];     // Offset 0
    block_q4_K W_k[KV_DIM * EMBED / QK_K];    // Offset Q_SIZE
    block_q4_K W_v[KV_DIM * EMBED / QK_K];    // Offset Q_SIZE + K_SIZE
    block_q4_K W_o[EMBED * Q_DIM / QK_K];     // ...
    block_q4_K W_gate[INTER * EMBED / QK_K];
    block_q4_K W_up[INTER * EMBED / QK_K];
    block_q4_K W_down[EMBED * INTER / QK_K];
};

// Prefetch next operation's weights while computing current
void decode_layer_prefetch(LayerWeights *L, ...) {
    // Prefetch K weights while computing Q
    prefetch_async(L->W_k, sizeof(L->W_k));
    compute_q_projection(L->W_q, ...);

    // Prefetch V weights while computing K
    prefetch_async(L->W_v, sizeof(L->W_v));
    compute_k_projection(L->W_k, ...);

    // ... etc
}
```

## System Comparison: Why Better Hardware Helps

| System | DDR | Channels | Bandwidth | L3 | Weights in L3 |
|--------|-----|----------|-----------|-----|---------------|
| Your i7-3630QM | DDR3-1600 | 2 | 25 GB/s | 6 MB | 3% |
| Ryzen 9 7950X | DDR5-5600 | 2 | 90 GB/s | 64 MB | 32% |
| Xeon w9-3595X | DDR5-4800 | 8 | 307 GB/s | 105 MB | 52% |
| Apple M3 Ultra | LPDDR5 | 16 | 800 GB/s | 144 MB | 72% |

**On better systems:**
1. More weights fit in L3 (less DRAM trips)
2. Higher bandwidth = less time waiting
3. More memory channels = better prefetch efficiency
4. Prefetching becomes MUCH more effective

## Recommended Strategy

### For Your i7-3630QM (DDR3, 25 GB/s)

1. **Single-token decode:**
   - Use SIMD kernels (2x faster than scalar)
   - 4 threads max (bandwidth saturated)
   - Software prefetching: ~10-20% additional gain
   - **Expected: 12-15 tok/s** (up from 8)

2. **Prefill (prompt processing):**
   - Parallelize across tokens (not within operations)
   - Much better cache utilization
   - **Expected: 30-50 tok/s** for short prompts

3. **Long context:**
   - KV cache becomes bottleneck
   - FP16 KV cache helps (2x less memory)

### For Modern Systems (DDR5, 100+ GB/s)

1. **Single-token decode:**
   - Parallel SIMD with 8-16 threads
   - Aggressive prefetching
   - **Expected: 50-100 tok/s**

2. **Prefill:**
   - Full parallelization
   - **Expected: 200-500 tok/s**

## Implementation Priority

1. **Software prefetching in GEMV** (Quick win, 10-20%)
2. **Parallel SIMD kernels** (2-3x improvement with good bandwidth)
3. **Cross-operation prefetching** (Additional 10-20%)
4. **Prefill parallelization** (Massive win for prompt processing)

## Why llama.cpp is 4x Faster

llama.cpp likely achieves ~30 tok/s on your system through:

1. ✅ Parallel SIMD in all kernels (we have SIMD, not parallel SIMD)
2. ✅ Software prefetching in GEMV loops
3. ✅ Optimized memory layout for sequential access
4. ✅ Careful OpenMP scheduling to avoid false sharing
5. ✅ Thread-local accumulators to reduce contention

**Our gap:** We have SIMD OR parallelism, not both together.
