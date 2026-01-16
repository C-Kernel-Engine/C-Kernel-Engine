# Kernel Design Pattern for C-Kernel-Engine

## Philosophy: Separation of Threading from Math

**Core Principle:** Math kernels are single-threaded, fully vectorized. Orchestrators handle threading.

---

## Three-Layer Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 3: Orchestrators (prefill_*, decode_*)                │
│ - Decide parallelism strategy (token vs feature)            │
│ - Call core kernels on [start, end) chunks                  │
│ - Handle OpenMP, false sharing, cache alignment             │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         v
┌──────────────────────────────────────────────────────────────┐
│ Layer 2: Dispatch (if needed)                               │
│ - Shape-specific routing (small-M, large-M, attention)      │
│ - CPUID detection (AVX-512 vs AVX2 vs scalar)              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         v
┌──────────────────────────────────────────────────────────────┐
│ Layer 1: Core Kernels (gelu_vec16_avx512)                   │
│ - Pure math, single-threaded                                │
│ - Vectorized (16 floats per iteration) + scalar tail        │
│ - NO OpenMP, NO threading, NO allocations                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Pattern 1: Element-wise Operations (GELU, ReLU, Tanh)

### Core Kernel (Layer 1)

```c
// gelu_kernels.c
#include <immintrin.h>

// Pure vectorized math, no threading
void gelu_vec16_avx512(float *data, size_t n)
{
    const __m512 c_sqrt_2_over_pi = _mm512_set1_ps(0.7978845608f);
    const __m512 c_coeff = _mm512_set1_ps(0.044715f);
    const __m512 c_half = _mm512_set1_ps(0.5f);
    const __m512 c_one = _mm512_set1_ps(1.0f);

    size_t i = 0;
    // Process 16 floats at a time
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(data + i);

        // x^3
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 x3 = _mm512_mul_ps(x2, x);

        // sqrt(2/π) * (x + 0.044715 * x^3)
        __m512 inner = _mm512_fmadd_ps(c_coeff, x3, x);
        inner = _mm512_mul_ps(c_sqrt_2_over_pi, inner);

        // tanh(inner) - approximate with fast tanh
        __m512 tanh_inner = fast_tanh_avx512(inner);

        // 0.5 * x * (1 + tanh(inner))
        __m512 result = _mm512_add_ps(c_one, tanh_inner);
        result = _mm512_mul_ps(result, x);
        result = _mm512_mul_ps(result, c_half);

        _mm512_storeu_ps(data + i, result);
    }

    // Scalar tail (0-15 remaining elements)
    for (; i < n; i++) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Fast tanh approximation for AVX-512
static inline __m512 fast_tanh_avx512(__m512 x)
{
    // Pade approximant: tanh(x) ≈ x(27 + x²)/(27 + 9x²)
    const __m512 c_27 = _mm512_set1_ps(27.0f);
    const __m512 c_9 = _mm512_set1_ps(9.0f);

    __m512 x2 = _mm512_mul_ps(x, x);

    __m512 num = _mm512_fmadd_ps(x2, c_27, c_27);
    num = _mm512_mul_ps(x, num);

    __m512 den = _mm512_fmadd_ps(x2, c_9, c_27);

    return _mm512_div_ps(num, den);
}
```

### Orchestrator (Layer 3)

```c
// mlp_prefill.c

// Prefill: Large batch, token-parallel
void mlp_gelu_prefill(float *activations, size_t num_tokens, size_t hidden_dim)
{
    // Each token is independent: parallelize over tokens
    // Each thread gets contiguous chunk of tokens (good cache locality)

    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        float *token_data = activations + t * hidden_dim;

        // Call single-threaded vectorized kernel
        gelu_vec16_avx512(token_data, hidden_dim);
    }

    // No false sharing: each thread writes to different tokens
    // Cache-friendly: each token's hidden_dim values are contiguous
}

// Decode: Single token, feature-parallel
void mlp_gelu_decode(float *activations, size_t hidden_dim)
{
    if (hidden_dim < 512) {
        // Small hidden_dim: single-threaded is faster (avoid overhead)
        gelu_vec16_avx512(activations, hidden_dim);
        return;
    }

    // Large hidden_dim: parallelize across features
    // Chunk size: 64-byte aligned (16 floats for AVX-512)
    const size_t chunk_size = 256; // 1KB per thread

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        size_t chunk_start = tid * chunk_size;
        size_t chunk_end = (tid + 1) * chunk_size;
        if (chunk_end > hidden_dim) chunk_end = hidden_dim;

        if (chunk_start < hidden_dim) {
            gelu_vec16_avx512(activations + chunk_start,
                              chunk_end - chunk_start);
        }
    }
}
```

---

## Pattern 2: Reduction Operations (Softmax, LayerNorm, RMSNorm)

### Core Kernel (Layer 1)

```c
// softmax_kernels.c

// Pure vectorized softmax for a single vector
void softmax_vec16_avx512(float *logits, size_t n)
{
    // Step 1: Find max (for numerical stability)
    __m512 vmax = _mm512_set1_ps(-INFINITY);
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(logits + i);
        vmax = _mm512_max_ps(vmax, x);
    }

    float max_val = _mm512_reduce_max_ps(vmax);

    // Scalar tail for max
    for (; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // Step 2: Compute exp(x - max) and sum
    __m512 vsum = _mm512_setzero_ps();
    __m512 vmax_broadcast = _mm512_set1_ps(max_val);

    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(logits + i);
        __m512 shifted = _mm512_sub_ps(x, vmax_broadcast);
        __m512 expx = exp_approx_avx512(shifted);

        _mm512_storeu_ps(logits + i, expx);
        vsum = _mm512_add_ps(vsum, expx);
    }

    float sum = _mm512_reduce_add_ps(vsum);

    // Scalar tail for exp
    for (; i < n; i++) {
        float val = expf(logits[i] - max_val);
        logits[i] = val;
        sum += val;
    }

    // Step 3: Normalize
    __m512 vsum_inv = _mm512_set1_ps(1.0f / sum);

    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(logits + i);
        __m512 normalized = _mm512_mul_ps(x, vsum_inv);
        _mm512_storeu_ps(logits + i, normalized);
    }

    // Scalar tail for normalize
    for (; i < n; i++) {
        logits[i] /= sum;
    }
}

// Fast exp approximation (5-6 digits accuracy)
static inline __m512 exp_approx_avx512(__m512 x)
{
    // exp(x) ≈ 2^(x/ln(2)) using _mm512_scalef_ps
    const __m512 log2e = _mm512_set1_ps(1.442695040f); // 1/ln(2)

    __m512 scaled = _mm512_mul_ps(x, log2e);
    return _mm512_scalef_ps(_mm512_set1_ps(1.0f), scaled);
}
```

### Orchestrator (Layer 3)

```c
// attention_prefill.c

// Prefill: Each head computes softmax over seq_len
void attention_softmax_prefill(float *attn_scores,
                               size_t num_heads,
                               size_t num_tokens,
                               size_t seq_len)
{
    // Parallelize over (head, token) pairs
    // Each thread gets one attention vector (seq_len values)

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t h = 0; h < num_heads; h++) {
        for (size_t t = 0; t < num_tokens; t++) {
            float *scores = attn_scores
                          + h * num_tokens * seq_len
                          + t * seq_len;

            // Single-threaded vectorized softmax
            softmax_vec16_avx512(scores, seq_len);
        }
    }
}

// Decode: Single token, but multiple heads
void attention_softmax_decode(float *attn_scores,
                              size_t num_heads,
                              size_t seq_len)
{
    // Parallelize over heads (if num_heads >= 4)
    if (num_heads >= 4) {
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < num_heads; h++) {
            float *scores = attn_scores + h * seq_len;
            softmax_vec16_avx512(scores, seq_len);
        }
    } else {
        // Few heads: single-threaded
        for (size_t h = 0; h < num_heads; h++) {
            float *scores = attn_scores + h * seq_len;
            softmax_vec16_avx512(scores, seq_len);
        }
    }
}
```

---

## Pattern 3: GEMM (Special Case)

GEMM is different - it benefits from internal parallelism at the blocking level.

### Approach A: External Threading Only (Simple)

```c
// gemm_orchestrator.c

void gemm_prefill(const float *A, const float *B, float *C,
                  size_t M, size_t N, size_t K)
{
    // Parallelize over M (rows of C)
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;

            // Vectorized dot product
            gemm_dot_vec16_avx512(A + i*K, B + j, N, K, &sum);

            C[i*N + j] = sum;
        }
    }
}
```

### Approach B: BLIS-style Internal Blocking (Advanced)

```c
// gemm_blocked.c

// No OpenMP inside - called by orchestrator
void gemm_kernel_16x2_avx512(const float *A, const float *B, float *C,
                             size_t K, size_t lda, size_t ldb, size_t ldc)
{
    // 16×2 microkernel: computes 16 rows × 2 cols of C
    // Pure AVX-512, no threading

    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();

    for (size_t k = 0; k < K; k++) {
        __m512 a = _mm512_loadu_ps(A + k * lda);
        __m512 b0 = _mm512_set1_ps(B[k * ldb + 0]);
        __m512 b1 = _mm512_set1_ps(B[k * ldb + 1]);

        c0 = _mm512_fmadd_ps(a, b0, c0);
        c1 = _mm512_fmadd_ps(a, b1, c1);
    }

    _mm512_storeu_ps(C + 0, c0);
    _mm512_storeu_ps(C + ldc, c1);
}

// Orchestrator calls microkernel with blocking
void gemm_blocked_threaded(const float *A, const float *B, float *C,
                           size_t M, size_t N, size_t K)
{
    const size_t MC = 384;  // L2 cache blocking
    const size_t NC = 4096; // L3 cache blocking
    const size_t KC = 384;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i = 0; i < M; i += MC) {
        for (size_t j = 0; j < N; j += NC) {
            for (size_t k = 0; k < K; k += KC) {
                size_t ib = MIN(MC, M - i);
                size_t jb = MIN(NC, N - j);
                size_t kb = MIN(KC, K - k);

                // Call microkernel on block (no threading inside)
                gemm_block_16x2(A + i*K + k,
                               B + k*N + j,
                               C + i*N + j,
                               ib, jb, kb, K, N, N);
            }
        }
    }
}
```

---

## Pattern 4: RoPE, RMSNorm, LayerNorm

Same philosophy - vectorized core, orchestrated threading.

### RMSNorm Core Kernel

```c
// rmsnorm_kernels.c

void rmsnorm_vec16_avx512(float *output,
                          const float *input,
                          const float *gamma,
                          size_t dim,
                          float eps)
{
    // Step 1: Compute sum of squares
    __m512 vsum = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        __m512 x = _mm512_loadu_ps(input + i);
        vsum = _mm512_fmadd_ps(x, x, vsum);
    }

    float sum_sq = _mm512_reduce_add_ps(vsum);

    for (; i < dim; i++) {
        sum_sq += input[i] * input[i];
    }

    // Step 2: RMS = sqrt(mean + eps)
    float rms = sqrtf(sum_sq / dim + eps);
    float rms_inv = 1.0f / rms;

    // Step 3: Normalize and scale by gamma
    __m512 vrms_inv = _mm512_set1_ps(rms_inv);

    i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 x = _mm512_loadu_ps(input + i);
        __m512 g = _mm512_loadu_ps(gamma + i);

        __m512 normalized = _mm512_mul_ps(x, vrms_inv);
        __m512 scaled = _mm512_mul_ps(normalized, g);

        _mm512_storeu_ps(output + i, scaled);
    }

    for (; i < dim; i++) {
        output[i] = (input[i] * rms_inv) * gamma[i];
    }
}
```

### RMSNorm Orchestrator

```c
// transformer_prefill.c

void rmsnorm_prefill(float *output,
                     const float *input,
                     const float *gamma,
                     size_t num_tokens,
                     size_t dim,
                     float eps)
{
    // Token-parallel: each token normalized independently
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        rmsnorm_vec16_avx512(output + t * dim,
                             input + t * dim,
                             gamma,
                             dim,
                             eps);
    }
}

void rmsnorm_decode(float *output,
                    const float *input,
                    const float *gamma,
                    size_t dim,
                    float eps)
{
    // Single token: no threading needed
    rmsnorm_vec16_avx512(output, input, gamma, dim, eps);
}
```

---

## Key Design Rules

### ✅ DO

1. **Keep kernels pure**: No OpenMP, no allocations, no I/O
2. **Vectorize aggressively**: AVX-512 first, then fallbacks
3. **Handle tail explicitly**: Don't rely on over-reads
4. **Cache-align chunks**: 64-byte boundaries for [start, end)
5. **Document FLOPS**: Comment expected performance
6. **Validate numerically**: Test against PyTorch

### ❌ DON'T

1. **Don't put OpenMP in kernels**: Threading belongs in orchestrators
2. **Don't use dynamic allocation in hot path**: Pre-allocate buffers
3. **Don't ignore false sharing**: Ensure threads write to separate cache lines
4. **Don't over-thread**: Small workloads are faster single-threaded
5. **Don't forget tail cases**: Always handle n % 16 ≠ 0
6. **Don't skip validation**: Numerical bugs compound

---

## File Organization

```
src/
├── kernels/                    # Layer 1: Pure math kernels
│   ├── gelu_kernels.c         # gelu_vec16_avx512()
│   ├── softmax_kernels.c      # softmax_vec16_avx512()
│   ├── rmsnorm_kernels.c      # rmsnorm_vec16_avx512()
│   ├── rope_kernels.c         # rope_vec16_avx512()
│   ├── gemm_kernels.c         # gemm_kernel_16x2_avx512()
│   └── layernorm_kernels.c    # layernorm_vec16_avx512()
│
├── orchestrators/              # Layer 3: Threading logic
│   ├── prefill/
│   │   ├── mlp_prefill.c      # Token-parallel GELU/matmul
│   │   ├── attention_prefill.c # Head-parallel attention
│   │   └── norm_prefill.c     # Token-parallel norms
│   │
│   └── decode/
│       ├── mlp_decode.c       # Feature-parallel or single-threaded
│       ├── attention_decode.c # Head-parallel softmax
│       └── norm_decode.c      # Single-threaded norms
│
└── dispatch/                   # Layer 2: Routing (optional)
    ├── shape_dispatch.c       # Small-M vs large-M
    └── cpu_dispatch.c         # AVX-512 vs AVX2 vs scalar
```

---

## Performance Expectations

| Kernel | Input Size | Expected GFLOPS | Notes |
|--------|-----------|-----------------|-------|
| **GELU** | 1M floats | 180-220 | Limited by memory BW |
| **Softmax** | seq=2048 | 150-200 | 3 passes (max, exp, div) |
| **RMSNorm** | dim=4096 | 200-250 | 2 passes (sum_sq, scale) |
| **GEMM** | 512×768×768 | 1200-1400 | 80-90% of MKL |
| **RoPE** | 128 heads×64 dim | 160-200 | Trig functions expensive |

---

## Validation Strategy

For each kernel:

1. **Unit test**: Known input → expected output
2. **PyTorch comparison**: diff < 1e-5 (forward), < 1e-3 (backward)
3. **Numerical gradient check**: Finite difference vs analytical
4. **Performance test**: Measure GFLOPS, compare to theoretical peak
5. **Threading test**: Verify no race conditions (Valgrind, TSan)

---

## Example: Complete GELU Implementation

```c
// src/kernels/gelu_kernels.c
#include "gelu_kernels.h"
#include <immintrin.h>
#include <math.h>

void gelu_vec16_avx512(float *data, size_t n)
{
    // [Implementation as shown above]
}

void gelu_backward_vec16_avx512(const float *input,
                                const float *d_output,
                                float *d_input,
                                size_t n)
{
    // Vectorized backward pass
    // [Implementation details...]
}

// src/orchestrators/prefill/mlp_prefill.c
#include "gelu_kernels.h"
#include <omp.h>

void mlp_gelu_prefill(float *activations,
                      size_t num_tokens,
                      size_t hidden_dim)
{
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        gelu_vec16_avx512(activations + t * hidden_dim, hidden_dim);
    }
}

// validation/test_gelu.py
import torch
import numpy as np
import ctypes

# Load C library
lib = ctypes.CDLL('./libckernel_engine.so')

def test_gelu_forward():
    # Test data
    x = torch.randn(512, 768, dtype=torch.float32)

    # PyTorch reference
    gelu_pytorch = torch.nn.functional.gelu(x, approximate='tanh')

    # C implementation
    x_c = x.numpy().copy()
    lib.gelu_vec16_avx512(
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(x_c.size)
    )

    # Compare
    diff = np.abs(gelu_pytorch.numpy() - x_c)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"GELU Forward Test:")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    assert max_diff < 1e-5, f"GELU forward failed: max diff {max_diff}"
    print("  ✓ PASSED")

if __name__ == '__main__':
    test_gelu_forward()
```

---

## Next Steps

1. **Vectorize existing kernels**: Add `*_vec16_avx512()` versions
2. **Add orchestrators**: Create prefill/decode wrappers
3. **Benchmark**: Measure actual GFLOPS vs theoretical
4. **Validate**: Test every kernel against PyTorch
5. **Document**: Add performance numbers to each kernel

---

**Last Updated**: 2025-11-23
**Status**: Design pattern established, ready for implementation
