# Q5_0 x Q8_0 Vectorized Kernel Implementation Plan

## CRITICAL FINDING: Parity Test Bug

**The parity test is MISLEADING!** It reports "896×896" but only tests M=1:

```python
# Bug in test_gemv_kernels_comprehensive.py:
test_rows = 1 if has_llama_ref else M  # <-- Only tests 1 row!
```

| Test | M | Time |
|------|---|------|
| What parity test does | 1 | 0.004 ms |
| What inference needs | 896 | 1.59 ms |
| **Hidden slowdown** | | **384x** |

This means the kernel benchmarks were showing M=1 performance, not M=896!

---

## Problem Statement

FlameGraph profiling revealed that 62% of inference time is spent in Q5_0 kernels:
- `vec_dot_q5_0_q8_0_avx`: 32.3% - **SCALAR despite "avx" name**
- `gemv_q5_0_avx`: 29.8% - **Mostly scalar with SSE helper**

The "avx" variants are placeholders with scalar implementations. Only AVX512 has real SIMD.

### Current CPU: AVX (no AVX2/FMA)
```
Flags: avx, sse, sse2, sse4_1, sse4_2, ssse3
Missing: avx2, fma, avx512f
```

### Dispatcher Path
```c
// gemv_q5_0 dispatcher (line 377-388):
#if defined(__AVX512F__)
    gemv_q5_0_avx512(...)      // Real SIMD
#elif defined(__AVX2__) && defined(__FMA__)
    gemv_q5_0_avx2(...)        // MISSING IMPLEMENTATION!
#elif defined(__AVX__)
    gemv_q5_0_avx(...)         // Scalar placeholder
#elif defined(__SSE4_1__)
    gemv_q5_0_sse_v2(...)      // SSE version exists
```

## Root Cause

1. `gemv_q5_0_avx` uses scalar loops to extract 5-bit weights
2. `vec_dot_q5_0_q8_0_avx` is 100% scalar (NO intrinsics!)
3. `gemv_q5_0_avx2` is referenced but NOT implemented
4. Only AVX512 has proper SIMD implementation

## Solution: Implement Vectorized SSE/AVX Kernels

Since the target CPU has SSE4.1 but not AVX2, we need to fix the SSE path.

### Phase 1: Fix vec_dot_q5_0_q8_0 for SSE4.1

**File:** `src/kernels/gemm_kernels_q5_0.c`

**Target function:** `vec_dot_q5_0_q8_0` dispatcher + new `vec_dot_q5_0_q8_0_sse` implementation

**Algorithm (SSE4.1):**
```c
// Process 16 elements at a time using SSE4.1
// 1. Load Q5_0 block (16 packed nibbles + 4 high bits)
// 2. Load Q8_0 block (32 int8 values)
// 3. Unpack Q5_0 to int8 using bit manipulation
// 4. Use _mm_maddubs_epi16 for int8 x int8 -> int16
// 5. Horizontal sum and scale
```

**Key intrinsics needed:**
- `_mm_loadu_si128` - load 128 bits
- `_mm_maddubs_epi16` - multiply-add unsigned x signed bytes (SSSE3)
- `_mm_madd_epi16` - pairwise multiply-add int16 (SSE2)
- `_mm_hadd_epi32` - horizontal add (SSSE3)
- `_mm_cvtsi128_si32` - extract int32

### Phase 2: Fix gemv_q5_0 for SSE4.1

Ensure `gemv_q5_0_sse_v2` is fully vectorized (currently calls per-block).

### Phase 3: Add gemv_q5_0_q8_0 (INT8 activation GEMV)

**New function:** `gemv_q5_0_q8_0(float *y, const void *W_q5_0, const void *x_q8_0, int M, int K)`

This is used for decode path with INT8 activations.

---

## Implementation Steps

### Step 1: Unit Tests First (TDD)

Create `unittest/test_q5_0_q8_0_kernel.py`:

```python
# Test cases:
# 1. Numerical parity with reference implementation
# 2. Numerical parity with llama.cpp's ggml_vec_dot_q5_0_q8_0
# 3. Edge cases: K=32, K=64, K=896, K=4096
# 4. Performance comparison vs llama.cpp
```

### Step 2: Reference Implementation

```c
// Pure scalar reference for testing
void vec_dot_q5_0_q8_0_ref(int n, float *s, const void *vx, const void *vy);
```

### Step 3: SSE4.1 Implementation

```c
void vec_dot_q5_0_q8_0_sse(int n, float *s, const void *vx, const void *vy)
{
    const int qk = QK5_0;  // 32
    const int nb = n / qk;

    const block_q5_0 *x = (const block_q5_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;

    __m128 acc = _mm_setzero_ps();

    for (int ib = 0; ib < nb; ib++) {
        const float d = CK_FP16_TO_FP32(x[ib].d) * CK_FP16_TO_FP32(y[ib].d);

        // Load Q5_0 nibbles (16 bytes = 32 nibbles packed)
        __m128i qs = _mm_loadu_si128((const __m128i *)x[ib].qs);

        // Load high bits
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        // Unpack to 32 x int8 values
        // ... vectorized bit manipulation ...

        // Load Q8_0 values (32 int8)
        __m128i q8_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        __m128i q8_1 = _mm_loadu_si128((const __m128i *)(y[ib].qs + 16));

        // Dot product: _mm_maddubs_epi16 + _mm_madd_epi16
        // ...

        acc = _mm_add_ps(acc, _mm_set1_ps(d * sum));
    }

    *s = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(acc, acc), acc));
}
```

### Step 4: Update Dispatcher

```c
void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy)
{
#ifdef __AVX512F__
    vec_dot_q5_0_q8_0_avx512(n, s, vx, vy);
#elif defined(__AVX2__) && defined(__FMA__)
    vec_dot_q5_0_q8_0_avx2(n, s, vx, vy);  // TODO: implement
#elif defined(__SSSE3__)
    vec_dot_q5_0_q8_0_sse(n, s, vx, vy);   // NEW!
#else
    vec_dot_q5_0_q8_0_ref(n, s, vx, vy);
#endif
}
```

### Step 5: GEMV Wrapper

```c
void gemv_q5_0_q8_0(float *y, const void *W, const void *x, int M, int K)
{
    const int blocks_per_row = K / QK5_0;
    const block_q5_0 *W_blocks = (const block_q5_0 *)W;
    const block_q8_0 *x_blocks = (const block_q8_0 *)x;

    for (int row = 0; row < M; row++) {
        vec_dot_q5_0_q8_0(K, &y[row],
                          &W_blocks[row * blocks_per_row],
                          x_blocks);
    }
}
```

---

## Testing Requirements

### 1. Numerical Parity Tests

```bash
# Compare against Python reference
python unittest/test_q5_0_q8_0_kernel.py --numerical

# Compare against llama.cpp
python unittest/test_q5_0_q8_0_kernel.py --llama-parity
```

**Tolerance:** `rtol=1e-4, atol=1e-5` (quantization introduces small errors)

### 2. Performance Tests

```bash
python unittest/test_q5_0_q8_0_kernel.py --perf
```

**Metrics:**
- GFLOPS achieved
- Comparison vs llama.cpp (target: >= 0.9x, ideal: > 1.0x)
- Comparison vs scalar reference (target: > 4x speedup)

### 3. Integration Test

```bash
# Run inference with new kernel
python scripts/v6.5/profile_inference.py

# Target: > 5 tok/s (currently 1.38 tok/s)
```

### 4. Parity Regression Test

```bash
make llamacpp-parity-full
# All tests must still pass
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/kernels/gemm_kernels_q5_0.c` | Add `vec_dot_q5_0_q8_0_sse`, fix dispatcher |
| `include/ckernel_quant.h` | Add `gemv_q5_0_q8_0` declaration |
| `src/ck_parity_api.c` | Add test wrapper |
| `include/ck_parity_api.h` | Add test wrapper declaration |
| `unittest/test_q5_0_q8_0_kernel.py` | New comprehensive test file |

---

## Expected Performance Improvement

| Metric | Before | After (Target) |
|--------|--------|----------------|
| Decode throughput | 1.38 tok/s | > 5 tok/s |
| vec_dot time % | 32.3% | < 15% |
| gemv_q5_0 time % | 29.8% | < 15% |
| Speedup vs scalar | 1x | 4-8x |

---

## Reference: llama.cpp Implementation

From `llama.cpp/ggml/src/ggml-quants.c`:

```c
void ggml_vec_dot_q5_0_q8_0(int n, float * restrict s, size_t bs,
                            const void * restrict vx, size_t bx,
                            const void * restrict vy, size_t by, int nrc) {
    // SSE/AVX2/AVX512 implementations with proper SIMD
}
```

Key insight: llama.cpp's implementation uses `_mm256_maddubs_epi16` (AVX2) which we can't use. We need SSE4.1/SSSE3 equivalent.

---

## Timeline

1. **Unit tests**: Create test file first
2. **Reference impl**: Ensure correctness baseline
3. **SSE impl**: Vectorize with SSSE3/SSE4.1
4. **Benchmark**: Compare performance
5. **Integration**: Update codegen to use new kernels
6. **Verify**: Run full inference and measure improvement
