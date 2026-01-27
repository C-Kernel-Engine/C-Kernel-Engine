# Fusion Benchmark Analysis - Summary

## Executive Summary

**Date**: January 17, 2026
**Benchmark**: Mega-fused attention prefill vs optimized baseline
**Result**: Initial test shows 1.30x speedup, but **investigation reveals benchmark artifact**

---

## Key Findings

### 1. Initial Benchmark Results

```
Config: Qwen2-0.5B (tokens=64, embed_dim=896, heads=14, kv_heads=2)

Baseline (separate ops):  0.09 ± 0.05 ms
Mega-fused:              0.07 ± 0.01 ms
Speedup:                 1.304x
```

**⚠️ PROBLEM**: These times are suspiciously fast for full attention computation.

### 2. Root Cause Analysis

From code review, we identified the **real issue**:

The original benchmark (`bench_mega_fused_attention_prefill.py`) compared:

| Component | Baseline | Fused |
|-----------|----------|-------|
| Flattening | Python scalar loop (448 iterations) | C AVX kernel |
| Output GEMM | `gemm_q5_0` = N×GEMV (896 serial calls) | `gemm_nt_head_major` (single tiled) |

**The "1.45x speedup" was from**:
1. ✅ Avoiding Python overhead
2. ✅ Using correct kernel (head-major vs token-major)
3. ❌ **NOT from fusion**

### 3. Current Test (simple_fusion_test_v2.py)

Current test shows 1.30x speedup but:
- Times are too fast (0.07-0.09 ms)
- Likely functions are not fully exercising the kernels
- May be hitting optimized fast paths or empty functions

---

## Investigation Required

### What We Need to Verify

1. **Are the kernels actually doing work?**
   - Check if functions have implementation or are stubs
   - Verify data is flowing through computations

2. **What's the baseline really measuring?**
   - Only QKV projection (no flash attention)
   - Missing key components of full attention

3. **Is the fused version complete?**
   - Does it include all operations?
   - Are there shortcut paths?

### Evidence That This Is an Artifact

From `src/kernels/fused/mega_fused_attention_prefill.c`:

```c
// Line 15-16:
/* VIOLATION: flatten_head_major() uses memcpy to convert head-major to
 * token-major layout. TODO: Make output projection accept strided input. */
```

**This means**:
- The "fused" kernel still does memcpy for layout conversion
- Memory traffic reduction may be minimal
- Benchmark may be comparing different code paths

---

## Recommendations

### Immediate Actions

1. **Verify kernel implementations**
   ```bash
   # Check if functions have actual code
   grep -A 50 "void mega_fused_attention_prefill" src/kernels/fused/mega_fused_attention_prefill.c
   ```

2. **Create proper baseline with timing**
   ```bash
   # Use same optimized kernels for both paths
   python3 scripts/benchmark_fusion_real.py
   ```

3. **Measure with different tools** (since perf is restricted)
   ```bash
   # Use Intel VTune if available
   vtune -collect memory-access -- python3 simple_fusion_test_v2.py

   # Or use built-in timing
   # Add timing markers in C code
   ```

### Long-term Strategy

Based on v6.6 findings (`FUSION_FINDINGS.md`):

> "Kernel fusion is NOT beneficial at Qwen2-0.5B scale because:
> - Intermediate buffers fit in L1/L2 cache
> - Weight matrices are 1000x larger than intermediates
> - The 'saved' memory traffic is negligible"

**Expected outcome**: After proper benchmarking, fusion will show **minimal benefit** (1.05-1.10x at most).

---

## Scripts Created

1. **`simple_fusion_test_v2.py`**
   - Basic comparison of fused vs separate
   - Shows initial 1.30x speedup (likely artifact)

2. **`benchmark_fusion_real.py`**
   - Comprehensive benchmark with proper baselines
   - Tests multiple fusion variants
   - Ready to use once kernel issues resolved

3. **`MEMORY_PROFILING_GUIDE.md`**
   - Complete guide for measuring DRAM traffic
   - Alternatives when perf is restricted
   - VTune, likwid, and other tools

---

## Next Steps

### Priority 1: Fix the Benchmark

1. Verify all kernel functions are implemented and working
2. Ensure baseline uses optimized kernels (not Python loops)
3. Add proper timing markers in C code

### Priority 2: Measure Memory Traffic

Since `perf` is restricted:

1. **Try VTune** (if available):
   ```bash
   vtune -collect memory-access -- python3 simple_fusion_test_v2.py
   ```

2. **Use alternative**: `likwid`
   ```bash
   likwid-perfctr -C 0 -g MEM ./benchmark
   ```

3. **Add canaries in C code**:
   ```c
   // Detect DRAM writes
   float canary = get_canary();
   // Run kernel
   mega_fused_attention(...);
   if (canary_changed()) {
       printf("WARNING: Touched DRAM!\n");
   }
   ```

### Priority 3: Validate Against v6.6

Run both:
1. Old benchmark (python loop baseline) → expect 1.45x
2. New benchmark (optimized baseline) → expect ~1.05x

**If both show 1.45x**: Fusion genuinely helps
**If only old benchmark shows 1.45x**: Confirmed artifact

---

## Conclusion

The initial 1.30x speedup is **likely a benchmark artifact** due to:
- Comparing optimized C kernel vs Python scalar loop
- Missing components in baseline (no flash attention)
- Potential empty/stub functions

**True fusion benefit**: Likely 1.05-1.10x (minimal), confirming v6.6 analysis.

**Action**: Fix benchmark to use optimized baselines, then re-measure with proper memory profiling tools.

---

## Quick Test Commands

```bash
# 1. Run simple test
python3 scripts/simple_fusion_test_v2.py

# 2. Check if VTune is available
which vtune || echo "VTune not installed"

# 3. Try likwid (if installed)
likwid-perfctr -C 0 -g MEM -- python3 scripts/simple_fusion_test_v2.py

# 4. Run comprehensive benchmark (once fixed)
python3 scripts/benchmark_fusion_real.py
```