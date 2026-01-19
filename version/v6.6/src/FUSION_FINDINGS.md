# Kernel Fusion Benchmark Findings

## Summary

After implementing and benchmarking multiple variations of fused kernels for Block 1 (RMSNorm + QKV) and Block 2 (MLP), we discovered that **fusion does NOT provide speedup at Qwen2-0.5B scale** because intermediate buffers are too small relative to weight matrices.

## Benchmark Results

### Block 1: RMSNorm + QKV (Qwen2-0.5B dimensions)

| Kernel | Time (µs) | Speedup vs Separate |
|--------|-----------|---------------------|
| Separate (scalar) | 250 | 1.00x |
| Fused v1 (per-output normed) | 323 | 0.77x (SLOWER) |
| Fused v2 (8-at-a-time) | 254 | 0.98x |
| Fused v3 (Q,K,V simultaneous) | 299 | 0.83x (SLOWER) |

**Key Finding**: All fused versions are slower or equal to separate!

## Root Cause Analysis

### Why Fusion Doesn't Help for Block 1

1. **Small intermediate buffer**: `normed` is only 3.5 KB (embed_dim × 4 bytes)
2. **Fits in L1 cache**: The "extra" memory traffic from separate is actually free
3. **Fused versions do MORE work**: They recompute `normed` for each output row

**Computation analysis**:
```
Separate: 1 × N (normed) + Q×N + K×N + V×N FMAs = 1.03M
Fused v1: (Q + K + V) × N × 2 FMAs = 2.06M (2x MORE!)
Fused v3: min(Q,K,V) × N × 4 + (Q-min) × N × 2 = 1.69M (1.6x MORE!)
```

### When Fusion DOES Help

Fusion provides benefit when:
1. **Intermediate buffer exceeds L1** (>32 KB) causing cache spills
2. **OR** the buffer competes with other hot data for cache space
3. **AND** the fused kernel uses optimized (SIMD) compute

## Block 2: Where Fusion CAN Help

For MLP (intermediate_dim = 4864):
- `gate_out`: 19.5 KB
- `up_out`: 19.5 KB
- Combined: **39 KB** (exceeds 32KB L1D!)

This is where fusion SHOULD provide benefit by:
1. Computing gate + up together (one pass through normed)
2. Fusing SwiGLU + down projection (no intermediate store)

**BUT**: Current `attention_mlp_fused_fp32` uses scalar loops = 0.60x SLOWER!

## Recommendations

### Short Term
1. **Skip Block 1 fusion** for embed_dim < 2048
2. **Fix Block 2** to use optimized GEMV kernels (MKL or SIMD)
3. **Skip Block 3 fusion** for same reason as Block 1

### Long Term
1. For larger models (2B+), Block 1 fusion may help:
   - embed_dim=4096 → normed=16KB (still L1)
   - embed_dim=8192 → normed=32KB (L1 pressure!)

2. The REAL optimization is in the **full decode path**:
   - Weight quantization (Q4_K reduces memory BW)
   - Batched operations (process multiple tokens)
   - KV cache optimization (FP16 to double context in L3)

## Code Changes

Files modified:
- `src/kernels/fused/rmsnorm_qkv.c` - Added v2, v3 variations
- `unittest/test_fusion_benchmark.py` - Comprehensive benchmark
- `src/v6.6/TRUE_SIMD_FUSION_PLAN.md` - Updated with findings

## Conclusion

**Kernel fusion is NOT a silver bullet.** It only helps when:
1. Intermediate buffers are large enough to spill cache
2. The fused implementation uses equally optimized compute

For Qwen2-0.5B specifically:
- Block 1: No fusion benefit (normed too small)
- Block 2: Potential benefit if we fix the scalar loops
- Block 3: No fusion benefit (normed too small)

## Block 2: MLP Fusion Results

| Model Size | v2 (gate+up fused) | v3 (sequential SIMD) | Scalar |
|------------|-------------------|----------------------|--------|
| Small (256×1024) | **2.65x** | **2.65x** | 1.0x |
| Medium (512×2048) | 0.90x | 0.84x | 1.0x |
| Qwen2-0.5B (896×4864) | 0.92x | 0.76x | 1.0x |
| 1B-class (1024×4096) | 0.95x | 0.74x | 1.0x |
| 2B-class (2048×8192) | 0.91x | 0.74x | 1.0x |

### Key Finding: SIMD Horizontal Sum Overhead

For small models, our SIMD GEMV is 2.65x faster. For larger models, compiler-auto-vectorized scalar code wins!

**Root cause**: Per-row SIMD requires horizontal sum (~6-8 instructions) per output element.
- Qwen2-0.5B: ~10,000 horizontal sums per MLP call
- This overhead dominates for larger dimensions

### Why Fusion Doesn't Help

1. **Weight matrices dominate**: 52MB of weights vs 38KB intermediate buffers
2. **Intermediate traffic is negligible**: 38KB / 52MB = 0.07% of memory traffic
3. **No cache benefit**: Even if we eliminate intermediate writes, weight loading still dominates

## Final Conclusions

**Kernel fusion is NOT beneficial at Qwen2-0.5B scale** because:
- Intermediate buffers fit in L1/L2 cache
- Weight matrices are 1000x larger than intermediates
- The "saved" memory traffic is negligible

**Where fusion WOULD help**:
- Very large models where intermediates exceed L2 (>256KB)
- Batched inference where activations scale with batch size
- Operations where intermediate buffers are comparable to weights

The biggest performance gains will come from:
1. Using quantized weights (Q4_K) - already done
2. Optimizing GEMV kernels (tiled, not per-row) - future work
3. KV cache optimization (FP16) - planned
