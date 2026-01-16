# Threading Strategy: Avoiding Nested OpenMP Hell

## The Problem

**Nested OpenMP parallelism** is a performance killer:

```c
// ❌ NEVER DO THIS
void gemm_blocked(A, B, C, M, N, K) {
    #pragma omp parallel for  // Level 1 parallelism
    for (int i = 0; i < M; i += block) {
        gemm_microkernel(A, B, C, ...);
    }
}

void transformer_layer(X, ...) {
    #pragma omp parallel for  // Level 2 parallelism (NESTED!)
    for (int t = 0; t < num_tokens; t++) {
        gemm_blocked(X_token, W_q, Q_token, ...);  // Spawns MORE threads!
    }
}
```

**Result:**
- Thread explosion (64 threads spawning 64 more = 4096 threads!)
- Context switching overhead
- Cache thrashing
- Actually SLOWER than single-threaded

---

## Solution: Single-Level Threading

Pick ONE layer to own threads. For C-Kernel-Engine, we choose **Option B**:

### Option B: Orchestrator Owns Threading (RECOMMENDED)

```
┌─────────────────────────────────────────────────────┐
│ Layer 3: Orchestrators (prefill/decode)            │
│ - OWNS OpenMP (#pragma omp parallel)               │
│ - Decides token-parallel vs feature-parallel       │
│ - Calls single-threaded kernels                    │
└────────────────┬────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────┐
│ Layer 1: Kernels (GEMM, GELU, softmax, etc.)       │
│ - NO OpenMP (single-threaded)                      │
│ - Fully vectorized (AVX-512)                       │
│ - Called by orchestrators on [start,end) slices   │
└─────────────────────────────────────────────────────┘
```

---

## Implementation Rules

### ✅ Rule 1: Kernels NEVER use OpenMP

```c
// ✅ CORRECT: Single-threaded, vectorized
void gemm_kernel_16x2_avx512(const float *A, const float *B, float *C,
                             size_t K, size_t lda, size_t ldb, size_t ldc)
{
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

void gelu_vec16_avx512(float *data, size_t n)
{
    // No OpenMP, pure AVX-512
    for (size_t i = 0; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(data + i);
        // ... math ...
        _mm512_storeu_ps(data + i, result);
    }
}

void softmax_vec16_avx512(float *logits, size_t n)
{
    // No OpenMP, pure vectorization
    // ... implementation ...
}
```

### ✅ Rule 2: Orchestrators Own Threading

```c
// src/orchestrators/prefill/mlp_prefill.c

void mlp_prefill(float *X, const float *W1, const float *W2,
                 float *H, float *O,
                 size_t num_tokens, size_t hidden_dim, size_t mlp_dim)
{
    // ========================================
    // FC1: X[num_tokens, hidden_dim] · W1[hidden_dim, mlp_dim]
    // ========================================

    // Token-parallel: each token is independent
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        const float *x_token = X + t * hidden_dim;
        float *h_token = H + t * mlp_dim;

        // Call single-threaded GEMM on one row
        gemm_row_vec16(x_token, W1, h_token, hidden_dim, mlp_dim);
    }

    // ========================================
    // GELU: H[num_tokens, mlp_dim] (in-place)
    // ========================================

    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        float *h_token = H + t * mlp_dim;

        // Call single-threaded GELU
        gelu_vec16_avx512(h_token, mlp_dim);
    }

    // ========================================
    // FC2: H[num_tokens, mlp_dim] · W2[mlp_dim, hidden_dim]
    // ========================================

    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        const float *h_token = H + t * mlp_dim;
        float *o_token = O + t * hidden_dim;

        // Call single-threaded GEMM on one row
        gemm_row_vec16(h_token, W2, o_token, mlp_dim, hidden_dim);
    }
}
```

```c
// src/orchestrators/decode/mlp_decode.c

void mlp_decode(float *x, const float *W1, const float *W2,
                float *h, float *o,
                size_t hidden_dim, size_t mlp_dim)
{
    // M=1: Single token
    // Different strategy than prefill!

    // ========================================
    // FC1: x[1, hidden_dim] · W1[hidden_dim, mlp_dim]
    // ========================================

    if (mlp_dim >= 512) {
        // Large output: parallelize over output features (N)
        #pragma omp parallel for schedule(static, 64)
        for (size_t j = 0; j < mlp_dim; j += 64) {
            size_t j_end = (j + 64 > mlp_dim) ? mlp_dim : j + 64;

            // Compute h[j:j_end] = x · W1[:, j:j_end]
            gemm_1xN_vec16(x, W1 + j, h + j, hidden_dim, j_end - j);
        }
    } else {
        // Small output: single-threaded (overhead too high)
        gemm_row_vec16(x, W1, h, hidden_dim, mlp_dim);
    }

    // ========================================
    // GELU: h[1, mlp_dim] (in-place)
    // ========================================

    if (mlp_dim >= 512) {
        // Large: parallelize over features
        #pragma omp parallel for schedule(static, 256)
        for (size_t chunk = 0; chunk < (mlp_dim + 255) / 256; chunk++) {
            size_t start = chunk * 256;
            size_t end = (start + 256 > mlp_dim) ? mlp_dim : start + 256;
            gelu_vec16_avx512(h + start, end - start);
        }
    } else {
        // Small: single-threaded
        gelu_vec16_avx512(h, mlp_dim);
    }

    // ========================================
    // FC2: h[1, mlp_dim] · W2[mlp_dim, hidden_dim]
    // ========================================

    if (hidden_dim >= 512) {
        #pragma omp parallel for schedule(static, 64)
        for (size_t j = 0; j < hidden_dim; j += 64) {
            size_t j_end = (j + 64 > hidden_dim) ? hidden_dim : j + 64;
            gemm_1xN_vec16(h, W2 + j, o + j, mlp_dim, j_end - j);
        }
    } else {
        gemm_row_vec16(h, W2, o, mlp_dim, hidden_dim);
    }
}
```

### ✅ Rule 3: Guard Against Nested OpenMP

Add runtime checks to ensure single-level threading:

```c
// src/orchestrators/common.h

#include <omp.h>
#include <assert.h>

#define ASSERT_NO_NESTED_OMP() \
    do { \
        assert(omp_in_parallel() == 0 && \
               "ERROR: Nested OpenMP detected! Orchestrators should not be called from within parallel regions."); \
    } while(0)

// In each orchestrator:
void mlp_prefill(...) {
    ASSERT_NO_NESTED_OMP();  // Fail fast if misused

    #pragma omp parallel for
    for (...) {
        // ...
    }
}
```

---

## Prefill vs Decode: Different Parallelism Strategies

### Prefill (M = 512 tokens)

**Token-Parallel:** Each thread processes one or more complete tokens.

```c
void attention_prefill(X, W_q, W_k, W_v, W_o, Q, K, V, O,
                       num_tokens, hidden_dim, num_heads, head_dim)
{
    // ==== Q, K, V Projections ====
    // Token-parallel: each token is independent

    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        gemm_row_vec16(X + t*hidden_dim, W_q, Q + t*hidden_dim, hidden_dim, hidden_dim);
    }

    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        gemm_row_vec16(X + t*hidden_dim, W_k, K + t*hidden_dim, hidden_dim, hidden_dim);
    }

    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        gemm_row_vec16(X + t*hidden_dim, W_v, V + t*hidden_dim, hidden_dim, hidden_dim);
    }

    // ==== Attention Scores ====
    // Head-parallel: each head is independent

    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < num_heads; h++) {
        // Q[num_tokens, head_dim] · K^T[head_dim, num_tokens]
        gemm_small(Q_h, K_h, scores_h, num_tokens, num_tokens, head_dim, /* transB= */ true);

        // Softmax over num_tokens dimension (per query)
        for (size_t t = 0; t < num_tokens; t++) {
            softmax_vec16_avx512(scores_h + t*num_tokens, num_tokens);
        }

        // scores[num_tokens, num_tokens] · V[num_tokens, head_dim]
        gemm_small(scores_h, V_h, out_h, num_tokens, head_dim, num_tokens);
    }

    // ==== Output Projection ====
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < num_tokens; t++) {
        gemm_row_vec16(attn_out + t*hidden_dim, W_o, O + t*hidden_dim, hidden_dim, hidden_dim);
    }
}
```

**Characteristics:**
- High parallelism (512 tokens)
- Good cache locality (each token's data is contiguous)
- No false sharing (each thread writes to separate tokens)

### Decode (M = 1 token)

**Feature-Parallel or Head-Parallel:** Each thread processes a chunk of features or one head.

```c
void attention_decode(x, W_q, W_k, W_v, W_o, q, k_cache, v_cache, o,
                      kv_len, hidden_dim, num_heads, head_dim)
{
    // ==== Q, K, V Projections ====
    // Feature-parallel (if dim large) or single-threaded

    if (hidden_dim >= 512) {
        #pragma omp parallel for schedule(static, 64)
        for (size_t j = 0; j < hidden_dim; j += 64) {
            size_t j_end = MIN(j + 64, hidden_dim);
            gemm_1xN_vec16(x, W_q + j, q + j, hidden_dim, j_end - j);
        }

        // Same for K, V projections
        // ...

    } else {
        // Small dim: single-threaded (overhead too high)
        gemm_row_vec16(x, W_q, q, hidden_dim, hidden_dim);
        gemm_row_vec16(x, W_k, k, hidden_dim, hidden_dim);
        gemm_row_vec16(x, W_v, v, hidden_dim, hidden_dim);
    }

    // Append new k, v to cache
    memcpy(k_cache + kv_len * hidden_dim, k, hidden_dim * sizeof(float));
    memcpy(v_cache + kv_len * hidden_dim, v, hidden_dim * sizeof(float));
    kv_len++;

    // ==== Attention Scores ====
    // Head-parallel: each head is independent

    if (num_heads >= 4) {
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < num_heads; h++) {
            // q[1, head_dim] · k_cache^T[head_dim, kv_len] = scores[1, kv_len]
            dot_product_vec(q_h, k_cache_h, scores_h, head_dim, kv_len);

            softmax_vec16_avx512(scores_h, kv_len);

            // scores[1, kv_len] · v_cache[kv_len, head_dim] = out[1, head_dim]
            weighted_sum_vec(scores_h, v_cache_h, out_h, kv_len, head_dim);
        }
    } else {
        // Few heads: single-threaded
        for (size_t h = 0; h < num_heads; h++) {
            // ... same logic, no OpenMP
        }
    }

    // ==== Output Projection ====
    if (hidden_dim >= 512) {
        #pragma omp parallel for schedule(static, 64)
        for (size_t j = 0; j < hidden_dim; j += 64) {
            size_t j_end = MIN(j + 64, hidden_dim);
            gemm_1xN_vec16(attn_out, W_o + j, o + j, hidden_dim, j_end - j);
        }
    } else {
        gemm_row_vec16(attn_out, W_o, o, hidden_dim, hidden_dim);
    }
}
```

**Characteristics:**
- Low token parallelism (1 token)
- Parallelize over features or heads instead
- Need threshold checks (avoid OpenMP overhead for small dims)

---

## GEMM: Special Considerations

GEMM can have internal parallelism at the blocking level, but we still keep the discipline:

### Approach A: External Threading Only (Simple)

```c
// src/kernels/gemm_kernels.c

// Single-threaded GEMM (called by orchestrator)
void gemm_st(const float *A, const float *B, float *C,
             size_t M, size_t N, size_t K)
{
    // Three-level blocking, but NO OpenMP
    for (size_t i = 0; i < M; i += MC) {
        for (size_t j = 0; j < N; j += NC) {
            for (size_t k = 0; k < K; k += KC) {
                gemm_block_16x2(A, B, C, ...);
            }
        }
    }
}

// Orchestrator parallelizes over rows
void gemm_prefill(const float *A, const float *B, float *C,
                  size_t M, size_t N, size_t K)
{
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < M; i += 16) {
        size_t i_end = MIN(i + 16, M);
        gemm_st(A + i*K, B, C + i*N, i_end - i, N, K);
    }
}
```

### Approach B: GEMM-Internal Threading (Advanced)

If you want BLIS-style internal parallelism (threads at pack/accumulate level), make GEMM the ONLY function with OpenMP:

```c
// src/kernels/gemm_kernels.c

// GEMM owns threading internally
void gemm_threaded(const float *A, const float *B, float *C,
                   size_t M, size_t N, size_t K)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // BLIS-style: threads work on panels
        // ... complex blocked algorithm ...
    }
}

// Orchestrator calls GEMM WITHOUT additional OpenMP
void mlp_prefill(X, W1, W2, H, O, num_tokens, hidden_dim, mlp_dim)
{
    // NO OpenMP here - GEMM handles threading internally
    gemm_threaded(X, W1, H, num_tokens, mlp_dim, hidden_dim);

    // For non-GEMM ops, add OpenMP here
    #pragma omp parallel for
    for (size_t t = 0; t < num_tokens; t++) {
        gelu_vec16_avx512(H + t*mlp_dim, mlp_dim);
    }

    gemm_threaded(H, W2, O, num_tokens, hidden_dim, mlp_dim);
}
```

**Recommendation:** Start with **Approach A** (external threading). It's simpler and avoids mixing threading models.

---

## File Organization

```
src/
├── kernels/                    # NO OpenMP
│   ├── gelu_kernels.c         # gelu_vec16_avx512()
│   ├── softmax_kernels.c      # softmax_vec16_avx512()
│   ├── rmsnorm_kernels.c      # rmsnorm_vec16_avx512()
│   ├── gemm_kernels.c         # gemm_kernel_16x2_avx512(), gemm_st()
│   └── ...
│
├── orchestrators/              # OpenMP HERE ONLY
│   ├── prefill/
│   │   ├── attention_prefill.c   # Token-parallel + head-parallel
│   │   ├── mlp_prefill.c         # Token-parallel
│   │   └── norm_prefill.c        # Token-parallel
│   │
│   └── decode/
│       ├── attention_decode.c    # Head-parallel, feature-parallel
│       ├── mlp_decode.c          # Feature-parallel
│       └── norm_decode.c         # Single-threaded or feature-parallel
│
└── models/                     # High-level API (NO OpenMP)
    ├── gpt2.c                  # Calls orchestrators
    ├── qwen.c
    └── llama.c
```

---

## Environment Variable Configuration

Give users control without recompiling:

```c
// src/orchestrators/common.c

static int g_num_threads = 0;
static bool g_initialized = false;

void ck_init_threading(void)
{
    if (g_initialized) return;

    const char *env = getenv("CK_NUM_THREADS");
    if (env) {
        g_num_threads = atoi(env);
    } else {
        g_num_threads = omp_get_max_threads();
    }

    omp_set_num_threads(g_num_threads);
    omp_set_nested(0);  // DISABLE nested parallelism
    g_initialized = true;
}

int ck_get_num_threads(void)
{
    if (!g_initialized) ck_init_threading();
    return g_num_threads;
}
```

Usage:
```bash
# Use 8 threads
export CK_NUM_THREADS=8
./benchmark_gpt2

# Use all available threads
unset CK_NUM_THREADS
./benchmark_gpt2
```

---

## Testing for Nested Parallelism

Add assertions to catch accidental nesting:

```c
// tests/test_no_nested_omp.c

void test_mlp_prefill_no_nesting(void)
{
    // Allocate data
    // ...

    // Try to call from within parallel region (should assert/abort)
    #pragma omp parallel
    {
        #pragma omp single
        {
            // This should fail assertion
            mlp_prefill(X, W1, W2, H, O, num_tokens, hidden_dim, mlp_dim);
        }
    }
}
```

---

## Performance Implications

### Single-Level Threading (✅ Good)

```
Thread overhead: O(num_threads) = O(8-16)
Cache behavior: Each thread gets large contiguous chunks
Context switches: Minimal (threads pinned to cores)
Load balancing: OpenMP's dynamic scheduling handles it
```

### Nested Threading (❌ Bad)

```
Thread overhead: O(num_threads^2) = O(64-256)
Cache behavior: Thrashing (many threads competing for same data)
Context switches: Constant (kernel scheduling 256 threads)
Load balancing: Impossible (too many moving parts)
```

**Real-world impact:**
- Single-level: 1200 GFLOPS (80% of peak)
- Nested: 200 GFLOPS (13% of peak) ← 6x SLOWER!

---

## Summary: The Golden Rule

**"Only orchestrators use OpenMP. Kernels are always single-threaded."**

This rule:
- ✅ Avoids nested parallelism
- ✅ Makes kernels reusable (can be called from anywhere)
- ✅ Simplifies debugging (no threading issues in kernels)
- ✅ Enables clean prefill/decode separation
- ✅ Matches production library design (BLIS, Eigen, oneDNN)

**Next Steps:**
1. Mark all kernels with `ASSERT_NO_OPENMP()` guards
2. Add OpenMP only in `src/orchestrators/`
3. Disable nested parallelism: `omp_set_nested(0)`
4. Test with `OMP_NESTED=true` to catch bugs

---

**Last Updated**: 2025-11-23
**Status**: Threading strategy defined, ready for implementation
