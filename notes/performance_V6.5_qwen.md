# V6.5 Performance Analysis Report - Qwen2-0.5B-Instruct

**Date:** 2026-01-13
**Model:** Qwen2-0.5B-Instruct (Q5_0 quantization)
**CPU:** Intel i7-3630QM (Ivy Bridge) - AVX but NOT AVX2/FMA
**Platform:** Linux 5.15.0-130-generic

## Baseline Performance (Before AVX Dispatch Fix)

### CK-Engine v6.5

| Metric | Value |
|--------|-------|
| Total Inference Time | 6.91s |
| Threads | 1 |
| Prefill Speed | 2.2 tok/s |
| Decode Speed | 4.36 tok/s |

**VTune Hotspots Analysis:**

| Function | CPU Time | % of Total |
|----------|----------|------------|
| `gemv_q5_0_avx` | 2.30s | 33.3% |
| `gemv_q6_k` | 1.88s | 27.2% |
| `vec_dot_q5_0_q8_0_sse` | 0.92s | 13.3% |
| `gemv_q8_0_avx` | 0.83s | 12.0% |
| `gemv_q4_k_avx` | 0.31s | 4.5% |

**Key Issue:** The INT8 dot product dispatch (`vec_dot_q5_0_q8_0`) is using SSE instead of AVX because the dispatch logic requires `AVX2 && FMA`, but Ivy Bridge only has AVX.

### llama.cpp (Reference)

| Metric | Value |
|--------|-------|
| Total Inference Time | 3.13s |
| Threads | 4 (OpenMP) |
| Prefill Speed | 57-62 tok/s |
| Decode Speed | 27-30 tok/s |

**VTune Hotspots Analysis:**

| Function | CPU Time | % of Total |
|----------|----------|------------|
| `ggml_vec_dot_q5_0_q8_0` | 0.94s | 30.0% |
| `llamafile_sgemm` | 0.68s | 21.7% |
| `ggml_graph_compute_thread` | 0.52s | 16.6% |
| `quantize_row_q8_0` | 0.31s | 9.9% |

## Performance Gap Analysis

### Overall Throughput Comparison

| Engine | Prefill | Decode |
|--------|---------|--------|
| llama.cpp | 62 tok/s | 30 tok/s |
| CK-Engine | 2.2 tok/s | 4.4 tok/s |
| **Gap** | **28x slower** | **7x slower** |

### Root Causes Identified

1. **Threading Gap (4x potential speedup)**
   - llama.cpp: 4 threads with OpenMP
   - CK-Engine: Single-threaded

2. **SSE vs AVX Dispatch (Est. 1.3x speedup)**
   - INT8 dot product (`vec_dot_q5_0_q8_0`) dispatches to SSE
   - AVX implementation exists but dispatch requires AVX2+FMA
   - **Fix:** Change dispatch to `#elif defined(__AVX__)`

3. **FP32 Prefill MLP (Est. 1.5x speedup)**
   - Prefill W1 uses `gemm_nt_q5_0` (FP32 dequantize)
   - Should use `gemm_nt_q5_0_q8_0` (INT8 batch)
   - ~60% of CK-Engine time in FP32 dequantization paths

4. **Per-Head Attention Loops (Est. 3-10x speedup)**
   - CK-Engine: ~40 kernel calls per layer (14 heads × 3 ops)
   - llama.cpp: ~6-8 batched operations per layer (tinyBLAS)
   - ~711 total kernel calls vs ~160 in llama.cpp

### Kernel Parity (Isolated Benchmarks)

Individual kernel performance is competitive with llama.cpp:

| Kernel | CK-Engine | llama.cpp | Status |
|--------|-----------|-----------|--------|
| Q4_K dot | ~same | baseline | PASS |
| Q5_0 dot | ~same | baseline | PASS |
| Q6_K dot | ~same | baseline | PASS |
| Q8_0 dot | ~same | baseline | PASS |

**Conclusion:** Kernel implementations are efficient; overhead comes from architecture.

## Fix Applied

### Change 1: AVX Dispatch Fix

**File:** `src/kernels/gemm_kernels_q5_0.c:1022`

```c
// BEFORE:
#elif defined(__AVX2__) && defined(__FMA__)
    vec_dot_q5_0_q8_0_avx(n, s, vx, vy);

// AFTER:
#elif defined(__AVX__)
    vec_dot_q5_0_q8_0_avx(n, s, vx, vy);
```

## Post-Fix Performance

**Result:** AVX dispatch fix applied and verified with VTune.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total Time | 6.91s | 7.30s | ~0% (no change) |
| Prefill | 2.2 tok/s | 2.14 tok/s | ~0% |
| Decode | 4.36 tok/s | 4.36 tok/s | ~0% |

**VTune Hotspots After Fix:**

| Function | CPU Time | % of Total |
|----------|----------|------------|
| `gemv_q5_0_avx` | 2.53s | 34.6% |
| `gemv_q6_k` | 1.93s | 26.5% |
| `vec_dot_q5_0_q8_0_avx` | 0.95s | 13.0% |
| `gemv_q8_0_avx` | 0.84s | 11.5% |
| `gemv_q4_k_avx` | 0.37s | 5.0% |

**Analysis:**
- The AVX dispatch IS working - `vec_dot_q5_0_q8_0_avx` is now called instead of `vec_dot_q5_0_q8_0_sse`
- Performance is unchanged because:
  1. AVX integer operations on Ivy Bridge (without AVX2) don't significantly outperform SSE
  2. The `vec_dot_q5_0_q8_0` function only accounts for ~13% of total time
  3. The main bottlenecks are `gemv_q5_0_avx` (35%) and `gemv_q6_k` (27%) which are FP32 dequantization paths

**Conclusion:** The AVX dispatch fix is correct but doesn't improve performance on this CPU. The real bottlenecks are:
1. Single-threaded execution (vs llama.cpp's 4 threads)
2. Per-head attention loops (40 calls/layer vs 6-8 for llama.cpp)
3. FP32 dequantization in prefill MLP (60% of time)

---

## Fix 2: Q6_K x Q8_K Kernel Implementation

**Date:** 2026-01-14

### Problem
The Q6_K down projection (W2) layers were using FP32 dequantization path (`gemm_nt_q6_k`) instead of INT8 quantized path.

### Solution
Created new Q6_K x Q8_K kernels:
- `vec_dot_q6_k_q8_k` - INT8 dot product
- `gemv_q6_k_q8_k` - GEMV for decode
- `gemm_nt_q6_k_q8_k` - GEMM for prefill (batch)

**Files Changed:**
- `src/kernels/gemm_kernels_q6k_q8k.c` (NEW - 550 lines)
- `include/ckernel_engine.h` - Added declarations
- `scripts/v6.5/codegen_v6_5.py` - Added Q6_K x Q8_K to kernel mappings
- `unittest/test_q6k_q8k_parity.py` (NEW - Unit tests)

### Performance Results

| Metric | Before Q6_K x Q8_K | After Q6_K x Q8_K | Improvement |
|--------|-------------------|-------------------|-------------|
| Decode Speed | 4.36 tok/s | **7.46 tok/s** | **+71%** |
| Decode Latency | 229.42 ms/token | 133.97 ms/token | **-42%** |
| Prefill Speed | 2.14 tok/s | 2.26 tok/s | +6% |

### Unit Test Results
```
vec_dot_q6_k_q8_k: PASS (rel error: 8.6e-7)
gemv_q6_k_q8_k: PASS (max diff: 7.6e-5)
gemm_nt_q6_k_q8_k: PASS (max diff: 6.1e-5)
vs_fp32_accuracy: PASS (correlation: 0.999967)
```

### Analysis
The 71% decode speedup came from converting the Q6_K down projection from FP32 dequantization to INT8 quantized multiplication. Previously, `gemv_q6_k` was one of the top hotspots at 27% CPU time.

---

## Updated Performance Summary

| Engine | Prefill | Decode |
|--------|---------|--------|
| llama.cpp | 62 tok/s | 30 tok/s |
| CK-Engine (before) | 2.2 tok/s | 4.4 tok/s |
| CK-Engine (after Q6_K x Q8_K) | 2.3 tok/s | **7.5 tok/s** |

### Remaining Gap
- Prefill: 27x slower than llama.cpp
- Decode: **4x slower** (improved from 7x)

### Next Optimizations
1. **High Impact: Add OpenMP Threading** (4x expected)
2. **High Impact: Batch Head Projections** (3-5x expected for attention)
3. **Medium Impact: INT8 Prefill W1** (1.5x expected)

## Recommendations for Future Optimization

1. **High Impact: Add OpenMP Threading**
   - Parallelize GEMM/GEMV across rows
   - Expected: 3-4x speedup

2. **High Impact: Batch Head Projections**
   - Replace per-head loops with batched GEMM
   - Expected: 3-5x speedup for attention

3. **Medium Impact: INT8 Prefill MLP**
   - Use `gemm_nt_q5_0_q8_0` for W1 in prefill
   - Expected: 1.5x speedup

4. **Low Impact: Memory Layout Optimization**
   - Improve cache locality for KV cache
   - Expected: 10-20% speedup
