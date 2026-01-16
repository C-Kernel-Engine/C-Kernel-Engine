# AI-Specific Shape Optimizations

## Introduction

Traditional BLAS libraries optimize for **general** matrix shapes. But AI workloads (especially transformers) have **specific, recurring patterns** that we can exploit.

This document shows how to optimize for **LLM-specific shapes** from `01-llm-kernel-shapes.md`.

---

## LLM Kernel Shape Families (Review)

From `01-llm-kernel-shapes.md`, we have these key operations:

```
1. QKV Projection:   [T, D] · [D, 3D] → [T, 3D]
2. Attention Output: [T, D] · [D, D]  → [T, D]
3. MLP Up-project:   [T, D] · [D, 4D] → [T, 4D]
4. MLP Down-project: [T, 4D] · [4D, D] → [T, D]
5. Attention Scores: [T, d] · [d, T]  → [T, T]  (per head)
6. Attention Values: [T, T] · [T, d]  → [T, d]  (per head)
```

Where:
- `T` = sequence length (1-4096)
- `D` = embedding dimension (768, 1024, 2048, 4096)
- `d` = head dimension (D / num_heads, typically 64 or 128)

**Key Observation:** T varies dramatically (1 to 4096), but D and d are fixed!

---

## Shape Characteristic 1: Variable M (Token Dimension)

### Small M: Autoregressive Decoding (M=1-16)

**Use case:** Generating tokens one at a time

```python
# GPT-2 autoregressive generation
for step in range(max_new_tokens):
    # Process only the LAST token
    logits = model(tokens[-1:])  # M=1
    next_token = sample(logits)
    tokens.append(next_token)
```

**GEMM shapes:**
```
QKV:   [1, 768] · [768, 2304] → [1, 2304]  (M=1, N=2304, K=768)
MLP1:  [1, 768] · [768, 3072] → [1, 3072]  (M=1, N=3072, K=768)
MLP2:  [1, 3072] · [3072, 768] → [1, 768]  (M=1, N=768, K=3072)
```

**Problem with general GEMM:**
- Cache blocking assumes M ≥ 64
- Microkernel (6×8 or 16×2) designed for larger M
- Overhead of setup dominates for M=1

**Solution: Small-M Kernel**

```c
// Optimized for M ≤ 16
void gemm_small_m(const float *A, const float *B, const float *bias,
                  float *C, int M, int N, int K)
{
    // Don't block on M dimension (it's already small!)
    // Block only on N and K

    const int Nc = 256;  // N blocking
    const int Kc = 512;  // K blocking

    for (int jj = 0; jj < N; jj += Nc) {
        int n_cur = (jj + Nc <= N) ? Nc : (N - jj);

        for (int kk = 0; kk < K; kk += Kc) {
            int k_cur = (kk + Kc <= K) ? Kc : (K - kk);

            // Process ALL rows at once (M is small!)
            for (int i = 0; i < M; i++) {
                __m512 bias_vec = bias ? _mm512_set1_ps(bias[jj]) : _mm512_setzero_ps();

                for (int j = jj; j < jj + n_cur; j += 16) {
                    __m512 sum = (kk == 0) ? bias_vec : _mm512_loadu_ps(&C[i * N + j]);

                    // Inner product: A[i, kk:kk+k_cur] · B[kk:kk+k_cur, j:j+16]
                    for (int k = kk; k < kk + k_cur; k++) {
                        __m512 a = _mm512_set1_ps(A[i * K + k]);
                        __m512 b = _mm512_loadu_ps(&B[k * N + j]);
                        sum = _mm512_fmadd_ps(a, b, sum);
                    }

                    _mm512_storeu_ps(&C[i * N + j], sum);
                }
            }
        }
    }
}
```

**Key differences:**
- No M blocking (iterate over all M rows)
- Broadcast A[i, k] across N dimension
- Focus on vectorizing N dimension

---

### Large M: Prompt Processing (M=256-4096)

**Use case:** Initial prompt encoding

```python
# Encode entire prompt at once
prompt = "Once upon a time, ..."
tokens = tokenize(prompt)  # M = len(tokens) = 512
logits = model(tokens)       # Process all tokens in parallel
```

**GEMM shapes:**
```
QKV:   [512, 768] · [768, 2304] → [512, 2304]  (M=512, N=2304, K=768)
MLP1:  [512, 768] · [768, 3072] → [512, 3072]  (M=512, N=3072, K=768)
```

**Optimization: Full Blocking**

Use the traditional three-level blocking strategy from `01-cache-hierarchy-blocking.md`:

```c
void gemm_large_m(const float *A, const float *B, const float *bias,
                  float *C, int M, int N, int K)
{
    const int Mr = 16, Nr = 2;  // Microkernel dimensions
    const int Mc = 384, Nc = 96, Kc = 384;  // Cache block sizes

    for (int jj = 0; jj < N; jj += Nc) {
        for (int pp = 0; pp < K; pp += Kc) {
            for (int ii = 0; ii < M; ii += Mc) {
                // Call microkernel
                for (int i = ii; i < ii + Mc && i < M; i += Mr) {
                    for (int j = jj; j < jj + Nc && j < N; j += Nr) {
                        gemm_microkernel_16x2(
                            &A[i * K + pp], &B[pp * N + j],
                            &C[i * N + j], Kc, K, N, N);
                    }
                }
            }
        }
    }
}
```

---

## Shape Characteristic 2: Small K (Attention Dimension)

### Attention Score GEMM: K=64 or K=128

```
Attention scores: Q[T, d] · K^T[d, T] → scores[T, T]

Where d = D / num_heads = 768 / 12 = 64  (common)
                        or 1024 / 8 = 128  (less common)
```

**GEMM shape:**
```
M = T (512, variable)
N = T (512, same as M)
K = d (64 or 128, FIXED and SMALL!)
```

**Optimization: K-Unrolled Kernel**

Since K is small and fixed, we can **fully unroll the K loop**:

```c
// Specialized kernel for K=64
void gemm_attention_k64(const float *Q, const float *K_T,
                        float *scores, int T)
{
    // Q: [T × 64]
    // K_T: [64 × T] (already transposed)
    // scores: [T × T]

    #pragma omp parallel for
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            // Dot product: Q[i, :] · K_T[:, j]
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();

            // Unroll K=64 into 4 chunks of 16
            __m512 q0 = _mm512_loadu_ps(&Q[i * 64 + 0]);
            __m512 k0 = _mm512_loadu_ps(&K_T[0 * T + j]);  // Wait, layout is wrong...
        }
    }
}
```

**Actually**, K_T should be transposed. Let me reconsider:

```c
// Attention: Q[T, d] · K[T, d]^T = Q[T, d] · K_T[d, T] → scores[T, T]

// If K is stored row-major as [T, d], we need to transpose access:
// scores[i, j] = sum_k Q[i, k] * K[j, k]

void gemm_attention_k64(const float *Q, const float *K,
                        float *scores, int T)
{
    #pragma omp parallel for
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j += 16) {  // Vectorize over j
            __m512 sum_vec = _mm512_setzero_ps();

            // K=64: unroll completely
            #pragma unroll
            for (int k = 0; k < 64; k++) {
                __m512 q = _mm512_set1_ps(Q[i * 64 + k]);  // Broadcast Q[i, k]
                __m512 kv = _mm512_loadu_ps(&K[j * 64 + k]);  // Load K[j:j+16, k]
                sum_vec = _mm512_fmadd_ps(q, kv, sum_vec);
            }

            _mm512_storeu_ps(&scores[i * T + j], sum_vec);
        }
    }
}
```

**Wait**, the access pattern `K[j * 64 + k]` loads from different rows, which is strided! Let me fix:

```c
// Better: Transpose K first, or use gather
void gemm_attention_k64_optimized(const float *Q, const float *K,
                                  float *scores, int T)
{
    // Assume K is stored as [T, 64], need K^T which is [64, T]
    // Option 1: Transpose K beforehand (preprocessing)
    // Option 2: Use different access pattern

    // Let's transpose K into K_T[64, T]
    float *K_T = aligned_alloc(64, 64 * T * sizeof(float));
    #pragma omp parallel for
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < 64; d++) {
            K_T[d * T + t] = K[t * 64 + d];
        }
    }

    // Now compute: Q[T, 64] · K_T[64, T] → scores[T, T]
    #pragma omp parallel for
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j += 16) {
            __m512 sum_vec = _mm512_setzero_ps();

            for (int k = 0; k < 64; k++) {
                __m512 q = _mm512_set1_ps(Q[i * 64 + k]);
                __m512 kt = _mm512_loadu_ps(&K_T[k * T + j]);
                sum_vec = _mm512_fmadd_ps(q, kt, sum_vec);
            }

            _mm512_storeu_ps(&scores[i * T + j], sum_vec);
        }
    }

    free(K_T);
}
```

**Key insight:** For attention with small K, the transpose overhead is amortized.

---

## Shape Characteristic 3: Rectangular GEMMs

### MLP Up-Projection: [T, D] · [D, 4D]

```
Example: [512, 768] · [768, 3072] → [512, 3072]

N = 4*D >> D (output is 4x wider than input)
```

**Optimization: Wide-N Blocking**

```c
void gemm_wide_n(const float *A, const float *B, const float *bias,
                 float *C, int M, int N, int K)
{
    // N is large, so block aggressively on N
    const int Nc = 512;  // Larger than usual

    for (int jj = 0; jj < N; jj += Nc) {
        // Process Nc columns at a time
        // Keep this B panel in cache while iterating over M

        for (int i = 0; i < M; i++) {
            for (int j = jj; j < jj + Nc && j < N; j += 16) {
                __m512 sum = bias ? _mm512_set1_ps(bias[j]) : _mm512_setzero_ps();

                for (int k = 0; k < K; k++) {
                    __m512 a = _mm512_set1_ps(A[i * K + k]);
                    __m512 b = _mm512_loadu_ps(&B[k * N + j]);
                    sum = _mm512_fmadd_ps(a, b, sum);
                }

                _mm512_storeu_ps(&C[i * N + j], sum);
            }
        }
    }
}
```

---

## Dispatch Strategy

### Shape-Based Dispatcher

```c
void sgemm_optimized(const float *A, const float *B, const float *bias,
                     float *C, int M, int N, int K)
{
    // Dispatch based on shape characteristics

    // Small M: Autoregressive decode
    if (M <= 16) {
        gemm_small_m(A, B, bias, C, M, N, K);
    }
    // Attention scores: Small K, square-ish
    else if (K == 64 || K == 128) {
        if (M >= 256 && N >= 256 && abs(M - N) < 64) {
            gemm_attention_k64(A, B, scores, M);  // Assume N == M
        } else {
            gemm_general(A, B, bias, C, M, N, K);
        }
    }
    // Wide N: MLP up-projection
    else if (N >= 4 * M) {
        gemm_wide_n(A, B, bias, C, M, N, K);
    }
    // Large M: Prompt processing
    else if (M >= 128) {
        gemm_large_m(A, B, bias, C, M, N, K);
    }
    // Default
    else {
        gemm_general(A, B, bias, C, M, N, K);
    }
}
```

---

## Fusion Opportunities

### 1. QKV Projection + Split

Traditional:
```c
// QKV projection
gemm(X, W_qkv, qkv, T, 3*D, D);  // [T, D] · [D, 3D] → [T, 3D]

// Split into Q, K, V
memcpy(Q, &qkv[0*D], T * D * sizeof(float));
memcpy(K, &qkv[1*D], T * D * sizeof(float));
memcpy(V, &qkv[2*D], T * D * sizeof(float));
```

Fused:
```c
// GEMM that directly writes to Q, K, V
gemm_qkv_split(X, W_qkv, Q, K, V, T, D);

// Inside the kernel:
for (int i = 0; i < T; i++) {
    // Compute row i of QKV
    for (int j = 0; j < D; j++) {
        Q[i * D + j] = ...;  // Output to Q directly
    }
    for (int j = 0; j < D; j++) {
        K[i * D + j] = ...;  // Output to K directly
    }
    for (int j = 0; j < D; j++) {
        V[i * D + j] = ...;  // Output to V directly
    }
}
```

**Benefit:** Eliminates 3*T*D memory stores and loads

---

### 2. MLP Up-Projection + GELU

Traditional:
```c
gemm(X, W_1, H, T, 4*D, D);  // [T, D] · [D, 4D] → [T, 4D]
gelu(H, T * 4 * D);           // In-place GELU
```

Fused:
```c
gemm_gelu(X, W_1, H, T, 4*D, D);

// Inside the kernel, after computing each element:
float h = sum + bias[j];
h = gelu_approx(h);  // Inline GELU
C[i * N + j] = h;
```

**GELU Approximation:**
```c
static inline float gelu_approx(float x) {
    // Tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float c1 = 0.797884560802865;  // sqrt(2/π)
    const float c2 = 0.044715;
    float x3 = x * x * x;
    float t = c1 * (x + c2 * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}
```

**Benefit:** Eliminates one pass over T*4D elements (~2MB for T=512, D=768)

---

## Performance Targets

### Baseline: Our Current Implementation

| Shape | Description | Current (GFLOPS) | Target (GFLOPS) | Gap |
|-------|-------------|------------------|-----------------|-----|
| [1, 768] · [768, 2304] | Decode QKV | ? | 50+ | TBD |
| [512, 768] · [768, 2304] | Prompt QKV | 250 | 1000+ | 4x |
| [512, 64] · [64, 512] | Attention | ? | 800+ | TBD |

### After Optimizations

| Optimization | Expected Speedup | Priority |
|-------------|------------------|----------|
| **Small-M kernel** | 3-5x (for M=1) | ⭐⭐⭐ Critical |
| **Attention K=64 kernel** | 1.5-2x | ⭐⭐ High |
| **BRGEMM batching** | 2-3x (multi-head) | ⭐⭐⭐ Critical |
| **Fusion (GELU, bias)** | 1.2-1.5x | ⭐ Medium |
| **Wide-N blocking** | 1.3-1.8x | ⭐ Medium |

---

## Implementation Checklist

### Phase 1: Critical Path (Week 1-2)
- [ ] Implement `gemm_small_m` for M ≤ 16
- [ ] Test on shape [1, 768] · [768, 2304]
- [ ] Profile: Should hit 50+ GFLOPS (vs. current ~10?)
- [ ] Validate: diff < 1e-5 vs. naive

### Phase 2: Batching (Week 3-4)
- [ ] Implement `sgemm_batched_strided` (from `02-brgemm-architecture.md`)
- [ ] Test on multi-head attention (12 heads × [512, 64] · [64, 512])
- [ ] Profile: Should be 2-3x faster than 12 separate GEMMs
- [ ] Validate: diff < 1e-5

### Phase 3: Fusion (Week 5-6)
- [ ] Implement `gemm_gelu` fusion
- [ ] Test on MLP: [512, 768] · [768, 3072] with GELU
- [ ] Profile: Should save ~1-2 ms per layer
- [ ] Validate: diff < 1e-5 (GELU is approximate, but should be close)

### Phase 4: Polish (Week 7-8)
- [ ] Implement shape dispatcher `sgemm_optimized`
- [ ] Add prefetching to all kernels
- [ ] Tune block sizes for target hardware
- [ ] Full profiling suite on all LLM shapes

---

## Validation Strategy

### Test Matrix

```c
struct TestShape {
    int M, N, K;
    const char *description;
};

const struct TestShape llm_shapes[] = {
    // Autoregressive decode
    {1, 768, 768, "Decode: attn output"},
    {1, 2304, 768, "Decode: QKV"},
    {1, 3072, 768, "Decode: MLP up"},
    {1, 768, 3072, "Decode: MLP down"},

    // Prompt processing
    {512, 768, 768, "Prompt: attn output"},
    {512, 2304, 768, "Prompt: QKV"},
    {512, 3072, 768, "Prompt: MLP up"},
    {512, 768, 3072, "Prompt: MLP down"},

    // Attention (per head)
    {512, 512, 64, "Attention: scores (d=64)"},
    {512, 512, 128, "Attention: scores (d=128)"},
    {512, 64, 512, "Attention: values (d=64)"},
};

void test_all_shapes() {
    for (int i = 0; i < sizeof(llm_shapes) / sizeof(llm_shapes[0]); i++) {
        test_shape(llm_shapes[i].M, llm_shapes[i].N, llm_shapes[i].K,
                   llm_shapes[i].description);
    }
}
```

---

**Next:** [08-computational-complexity.md](08-computational-complexity.md) for roofline analysis and hardware limits.
