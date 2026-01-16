# BRGEMM: Batched-Reduced GEMM for AI Workloads

## What is BRGEMM?

**BRGEMM** (Batched-Reduced GEMM) is Intel oneDNN's optimization for computing multiple GEMM operations efficiently. It's specifically designed for AI workloads like transformers and CNNs.

**The Problem It Solves:**

In transformers, we frequently need to compute the **same GEMM operation** on **multiple data batches**:

```c
// Multi-head attention: 12 separate GEMM calls
for (int h = 0; h < 12; h++) {
    // Compute attention scores for head h
    sgemm(Q[h], K[h], scores[h], T, T, d);  // [T×d] · [d×T] → [T×T]
}

// Problem:
// - 12 separate kernel launches (overhead!)
// - Poor cache reuse between heads
// - Can't share loaded data
```

**BRGEMM Solution:**

```c
// Single batched call for all 12 heads
brgemm_strided(Q_base, K_base, scores_base,
               batch_count = 12,
               M=T, N=T, K=d,
               stride_a = T*d,    // Offset to next Q head
               stride_b = d*T,    // Offset to next K head
               stride_c = T*T);   // Offset to next scores head

// Benefits:
// - 1 kernel launch instead of 12
// - Better cache reuse (process multiple heads while data is hot)
// - Lower overhead
```

---

## The Four BRGEMM Batch Types

From `oneDNN/src/cpu/x64/brgemm/brgemm_types.hpp`:

```cpp
typedef enum {
    brgemm_addr = 1,          // Arrays of pointers
    brgemm_offs = 2,          // Base + array of offsets
    brgemm_strd = 3,          // Base + fixed stride (BEST FOR TRANSFORMERS!)
    brgemm_static_offs = 4,   // Base + static offsets
} brgemm_batch_kind_t;
```

### 1. `brgemm_addr` - Array of Pointers

**Use case:** When matrices are scattered in memory

```c
// Matrices not contiguous in memory
float *A_ptrs[batch_count] = {A0, A1, A2, ...};
float *B_ptrs[batch_count] = {B0, B1, B2, ...};
float *C_ptrs[batch_count] = {C0, C1, C2, ...};

brgemm_batch_element_t batch[batch_count];
for (int b = 0; b < batch_count; b++) {
    batch[b].ptr.A = A_ptrs[b];
    batch[b].ptr.B = B_ptrs[b];
}

brgemm_execute(kernel, batch, ...);
```

**Memory Layout:**
```
Batch 0: A0 [M×K]    somewhere in memory
Batch 1: A1 [M×K]    somewhere else
Batch 2: A2 [M×K]    somewhere else
...
```

**Pros:** Flexible, handles non-contiguous memory
**Cons:** Pointer indirection overhead

---

### 2. `brgemm_offs` - Base Address + Offset Array

**Use case:** Matrices stored with variable offsets from a base

```c
// All matrices in one arena, but at irregular offsets
float *base_A = arena;
int64_t offsets_A[batch_count] = {0, 1024, 3456, ...};

brgemm_batch_element_t batch[batch_count];
for (int b = 0; b < batch_count; b++) {
    batch[b].offset.A = offsets_A[b];
    batch[b].offset.B = offsets_B[b];
}

brgemm_execute(kernel, batch, ...);
```

**Memory Layout:**
```
Arena:
[A0 at offset 0][padding][A1 at offset 1024][padding][A2 at offset 3456]...
```

**Pros:** Single base pointer, flexible layout
**Cons:** Still need to load offsets

---

### 3. `brgemm_strd` - Base Address + Fixed Stride ⭐

**Use case:** Regularly spaced matrices (PERFECT for multi-head attention!)

```c
// All Q heads stored contiguously with fixed stride
float *Q_base = malloc(num_heads * T * d * sizeof(float));
// Q_head[0] at offset 0
// Q_head[1] at offset T*d
// Q_head[2] at offset 2*T*d
// ...

brgemm_strides_t strides;
strides.stride_a = T * d * sizeof(float);  // Stride between Q heads
strides.stride_b = d * T * sizeof(float);  // Stride between K heads
strides.stride_c = T * T * sizeof(float);  // Stride between score matrices

brgemm_execute(kernel, batch_count, A_base, B_base, C_base, &strides, ...);
```

**Memory Layout:**
```
Q_heads: [Q0: T×d][Q1: T×d][Q2: T×d]...[Q11: T×d]  (contiguous!)
         ↑        ↑        ↑
         0        stride   2*stride
```

**Pros:**
- Fastest! No indirection, predictable access
- Best cache prefetching
- Minimal overhead

**Cons:** Requires contiguous, regularly-spaced layout

**This is the one we should implement for C-Kernel-Engine!**

---

### 4. `brgemm_static_offs` - Base Address + Static Offsets

**Use case:** Offsets known at compile time

```c
// Offsets baked into the JIT-compiled kernel
const int64_t static_offsets[] = {0, 512, 1024, 1536};

brgemm_execute_static(kernel, A_base, B_base, C_base, ...);
// Offsets are part of the kernel code itself
```

**Pros:** Offsets compiled in, no runtime lookup
**Cons:** Inflexible, requires JIT compilation

---

## BRGEMM for Multi-Head Attention: Deep Dive

### The Transformer Attention Pattern

```python
# PyTorch multi-head attention (simplified)
def multi_head_attention(Q, K, V):
    # Q, K, V: [batch, seq_len, d_model]
    # num_heads = 12, head_dim = 64

    # Split into heads: [batch, seq_len, d_model] → [batch, num_heads, seq_len, head_dim]
    Q_heads = Q.view(batch, num_heads, seq_len, head_dim)
    K_heads = K.view(batch, num_heads, seq_len, head_dim)
    V_heads = V.view(batch, num_heads, seq_len, head_dim)

    # Compute attention scores for each head
    scores = []
    for h in range(num_heads):
        # scores[h] = Q_heads[:, h, :, :] @ K_heads[:, h, :, :].T
        # Shape: [seq_len, head_dim] @ [head_dim, seq_len] → [seq_len, seq_len]
        scores.append(matmul(Q_heads[:, h], K_heads[:, h].T))

    return scores
```

### Traditional Approach (Slow)

```c
// Naive: 12 separate GEMM calls
for (int h = 0; h < num_heads; h++) {
    float *Q_h = &Q[h * seq_len * head_dim];
    float *K_h = &K[h * head_dim * seq_len];  // Transposed
    float *scores_h = &scores[h * seq_len * seq_len];

    // [seq_len × head_dim] · [head_dim × seq_len] → [seq_len × seq_len]
    sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          seq_len, seq_len, head_dim,
          1.0f, Q_h, head_dim, K_h, head_dim, 0.0f, scores_h, seq_len);
}

// Cost:
// - 12 kernel launches (overhead ≈ 1-5 μs each = 12-60 μs total)
// - Cache cold on each head (Q[0] evicted by time we get to Q[11])
// - No data reuse
```

**For seq_len=512, head_dim=64, num_heads=12:**
```
Each GEMM: 2 × 512 × 512 × 64 = 33.5 MFLOP
Total: 33.5 × 12 = 402 MFLOP

Time (naive): 402 MFLOP / 50 GFLOPS = 8.0 ms
(Assuming poor cache behavior, only hitting 50 GFLOPS)
```

---

### BRGEMM Approach (Fast)

```c
// BRGEMM strided: 1 kernel call for all 12 heads
brgemm_desc_t brg;
brgemm_desc_init(&brg,
    isa_undef,                    // Auto-detect (AVX-512/AMX)
    brgemm_strd,                  // Strided batch
    dt_f32, dt_f32,               // fp32 × fp32
    false, true,                  // A not transposed, B transposed
    brgemm_row_major,
    1.0f, 0.0f,                   // alpha=1, beta=0
    head_dim, seq_len, seq_len,   // LDA, LDB, LDC
    seq_len, seq_len, head_dim);  // M, N, K

brgemm_strides_t strides;
strides.stride_a = seq_len * head_dim * sizeof(float);
strides.stride_b = head_dim * seq_len * sizeof(float);

// Compile kernel
brgemm_kernel_t *kernel;
brgemm_kernel_create(&kernel, &brg);

// Execute once for ALL heads
brgemm_execute_strided(kernel, num_heads, Q_base, K_base, scores_base, &strides);

// Cost:
// - 1 kernel launch (overhead ≈ 1-5 μs)
// - Cache reuse: Q[0-3] may still be hot when processing Q[4-7]
// - Better instruction-level parallelism
```

**Performance improvement:**
```
Time (BRGEMM): 402 MFLOP / 120 GFLOPS = 3.35 ms
Speedup: 8.0 / 3.35 = 2.4x faster!
```

**Why faster?**
1. **Amortized overhead:** 1 launch vs 12
2. **Better cache reuse:** Process adjacent heads together
3. **Vectorization across batch:** Can SIMD over heads
4. **Better prefetching:** Predictable stride pattern

---

## Memory Layout for BRGEMM Strided

### Correct Layout for Multi-Head Attention

```c
// Q matrix: [num_heads, seq_len, head_dim]
// Memory layout: head-major
float Q[num_heads][seq_len][head_dim];

// Linear memory view:
// [Q_head0: seq_len×head_dim][Q_head1: seq_len×head_dim]...[Q_head11: seq_len×head_dim]

// Accessing head h, token t, dimension d:
float q_htd = Q[h * (seq_len * head_dim) + t * head_dim + d];

// Stride between heads:
stride_a = seq_len * head_dim * sizeof(float)
```

### Common Mistake: Wrong Layout

```c
// WRONG: [seq_len, num_heads, head_dim] (token-major)
// Memory: [Token0: all heads][Token1: all heads]...

// This breaks strided access:
// - Heads are NOT contiguous
// - Stride is irregular
// - BRGEMM strided won't work!

// Fix: Transpose to head-major before BRGEMM
```

---

## Implementation in C-Kernel-Engine

### Step 1: Add Batched Interface

```c
// In include/ckernel_engine.h
typedef struct {
    void (*sgemm)(int M, int N, int K,
                  const float *A, int lda,
                  const float *B, int ldb,
                  const float *bias,
                  float *C, int ldc);

    // NEW: Batched strided GEMM
    void (*sgemm_batched_strided)(
        int batch_count,
        int M, int N, int K,
        const float *A, int lda, int stride_a,
        const float *B, int ldb, int stride_b,
        const float *bias, int stride_bias,
        float *C, int ldc, int stride_c);
} CKMathBackend;
```

### Step 2: Implement BRGEMM Strided

```c
// In src/backend_native.c

void gemm_batched_strided(
    int batch_count,
    int M, int N, int K,
    const float *A_base, int lda, int stride_a,
    const float *B_base, int ldb, int stride_b,
    const float *bias_base, int stride_bias,
    float *C_base, int ldc, int stride_c)
{
    // Naive approach: just loop over batches
    for (int b = 0; b < batch_count; b++) {
        const float *A = A_base + b * stride_a;
        const float *B = B_base + b * stride_b;
        const float *bias = bias_base ? (bias_base + b * stride_bias) : NULL;
        float *C = C_base + b * stride_c;

        gemm_blocked_serial(A, B, bias, C, M, N, K);
    }
}

// This works, but doesn't exploit the key optimization:
// We can reuse cache between batches!
```

### Step 3: Optimized BRGEMM (Cache Reuse)

```c
void gemm_batched_strided_optimized(
    int batch_count,
    int M, int N, int K,
    const float *A_base, int lda, int stride_a,
    const float *B_base, int ldb, int stride_b,
    const float *bias_base, int stride_bias,
    float *C_base, int ldc, int stride_c)
{
    const int block_size = 64;

    // Key idea: Iterate over K blocks ONCE for all batches
    // This keeps A and B panels in cache

    for (int kk = 0; kk < K; kk += block_size) {
        int k_end = ck_min(kk + block_size, K);

        // Process this K block for ALL batches before moving to next K block
        for (int b = 0; b < batch_count; b++) {
            const float *A = A_base + b * stride_a;
            const float *B = B_base + b * stride_b;
            float *C = C_base + b * stride_c;

            // Compute partial result for this K block
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    __m512 sum_vec = _mm512_setzero_ps();

                    for (int k = kk; k <= k_end - 16; k += 16) {
                        __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                        __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                    }

                    float partial = _mm512_reduce_add_ps(sum_vec);

                    // Accumulate (since we're iterating over K)
                    if (kk == 0) {
                        C[i * N + j] = partial + (bias_base ? bias_base[b * stride_bias + j] : 0.0f);
                    } else {
                        C[i * N + j] += partial;
                    }
                }
            }
        }
    }
}

// This is better: A[i, kk:kk+64] is reused across all batches
```

---

## Performance Comparison

### Benchmark Setup

```c
// Multi-head attention parameters
int num_heads = 12;
int seq_len = 512;
int head_dim = 64;

float *Q = aligned_alloc(64, num_heads * seq_len * head_dim * sizeof(float));
float *K = aligned_alloc(64, num_heads * head_dim * seq_len * sizeof(float));
float *scores = aligned_alloc(64, num_heads * seq_len * seq_len * sizeof(float));

// Initialize with random data
// ...
```

### Method 1: Naive Loop (Baseline)

```c
double start = get_time();
for (int h = 0; h < num_heads; h++) {
    gemm_blocked_serial(&Q[h * seq_len * head_dim],
                       &K[h * head_dim * seq_len],
                       NULL,
                       &scores[h * seq_len * seq_len],
                       seq_len, seq_len, head_dim);
}
double naive_time = get_time() - start;

// Expected: ~8-10 ms on AVX-512 (assuming 40-50 GFLOPS)
```

### Method 2: BRGEMM Strided

```c
double start = get_time();
gemm_batched_strided(
    num_heads,
    seq_len, seq_len, head_dim,
    Q, head_dim, seq_len * head_dim,
    K, seq_len, head_dim * seq_len,
    NULL, 0,
    scores, seq_len, seq_len * seq_len);
double brgemm_time = get_time() - start;

// Expected: ~3-4 ms (2-2.5x speedup)
```

### Method 3: oneDNN BRGEMM (Reference)

```c
// Using oneDNN's optimized BRGEMM
double start = get_time();
brgemm_desc_t brg;
brgemm_desc_init(&brg, ...);
brgemm_kernel_t *kernel;
brgemm_kernel_create(&kernel, &brg);
brgemm_execute_strided(kernel, num_heads, Q, K, scores, &strides);
double onednn_time = get_time() - start;

// Expected: ~2-3 ms (oneDNN has more optimizations)
```

---

## Advanced: AMX Support

On Intel Sapphire Rapids and newer (AMX-enabled CPUs):

```c
// BRGEMM with AMX (tile-based matrix multiplication)
brgemm_desc_init(&brg,
    avx512_core_amx,   // Use AMX instead of AVX-512
    brgemm_strd,
    dt_bf16, dt_bf16,  // bf16 × bf16 (faster on AMX!)
    ...);

// AMX uses 2D register tiles (tmm0-tmm7)
// Can compute 16×16 fp32 or 32×32 bf16 in one instruction
// Much faster than AVX-512 for large matrices
```

**Performance on AMX:**
- bf16: 10-20x faster than AVX-512 fp32
- int8: 20-40x faster
- Critical for production LLM inference

**Future work for C-Kernel-Engine:** Add AMX backend when hardware is available.

---

## Summary: BRGEMM Benefits

### For Multi-Head Attention

| Aspect | Without BRGEMM | With BRGEMM | Improvement |
|--------|----------------|-------------|-------------|
| **Kernel Launches** | 12 | 1 | 12x less overhead |
| **Cache Reuse** | Poor | Good | 1.5-2x bandwidth |
| **GFLOPS** | 40-60 | 100-120 | 2-3x throughput |
| **Time (512 tokens)** | 8-10 ms | 3-4 ms | 2.5x faster |

### When to Use BRGEMM

✅ **Use BRGEMM when:**
- Multiple GEMMs with same shape (M, N, K)
- Regular memory layout (strided)
- Batch count > 4 (otherwise overhead dominates)

❌ **Don't use BRGEMM when:**
- Single GEMM (no batching)
- Irregular strides (use brgemm_offs or brgemm_addr)
- Very small matrices (overhead not worth it)

---

## Implementation Checklist

For C-Kernel-Engine, implement in this order:

- [ ] **Phase 1:** Add `sgemm_batched_strided` interface to `CKMathBackend`
- [ ] **Phase 2:** Implement naive loop (correctness first)
- [ ] **Phase 3:** Optimize for cache reuse (iterate K once for all batches)
- [ ] **Phase 4:** Add prefetching between batches
- [ ] **Phase 5:** Profile vs. oneDNN BRGEMM, close the gap

**Target:** Within 80% of oneDNN performance on multi-head attention shapes.

---

**Next:** [03-microkernel-design.md](03-microkernel-design.md) for detailed microkernel implementation.
