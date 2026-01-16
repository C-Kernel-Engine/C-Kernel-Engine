# Microkernel Design Principles

## What is a Microkernel?

The **microkernel** is the innermost computational loop in GEMM. It's a small, highly-optimized function that computes:

```
C[Mr×Nr] += A[Mr×K] * B[K×Nr]
```

Where:
- `Mr` = microkernel M dimension (rows of A, rows of C)
- `Nr` = microkernel N dimension (cols of B, cols of C)
- `K` = reduction dimension (unrolled in the microkernel loop)

**Key characteristics:**
1. **Register-blocked:** Accumulates result in registers, not memory
2. **Hand-optimized:** Often assembly or carefully-written intrinsics
3. **Fixed size:** Mr×Nr is constant (e.g., 6×8, 16×2)
4. **SIMD-optimized:** Uses AVX-512, AMX, or other vector instructions

---

## BLIS 6×16 Microkernel Analysis

From `/home/antshiv/Workspace/3rd-Party/MathLibrary/blis/kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c`:

### Function Signature

```c
void bli_sgemm_haswell_asm_6x16(
    dim_t      m,         // Should be 6 (or ≤ 6 for edge)
    dim_t      n,         // Should be 16 (or ≤ 16 for edge)
    dim_t      k,         // Reduction dimension
    const void* alpha,    // Scaling factor
    const void* a,        // A panel [6 × k]
    const void* b,        // B panel [k × 16]
    const void* beta,     // C scaling factor
    void*      c,         // C tile [6 × 16]
    inc_t      rs_c,      // Row stride of C
    inc_t      cs_c,      // Column stride of C
    const auxinfo_t* data,
    const cntx_t*    cntx
);
```

### Register Allocation (6×16 microkernel)

```asm
// From the assembly code:

// Zero all accumulator registers (ymm4-ymm15)
vxorps( ymm4, ymm4, ymm4)     // C[0,0:7]
vmovaps( ymm4, ymm5)           // C[0,8:15]
vmovaps( ymm4, ymm6)           // C[1,0:7]
vmovaps( ymm4, ymm7)           // C[1,8:15]
vmovaps( ymm4, ymm8)           // C[2,0:7]
vmovaps( ymm4, ymm9)           // C[2,8:15]
vmovaps( ymm4, ymm10)          // C[3,0:7]
vmovaps( ymm4, ymm11)          // C[3,8:15]
vmovaps( ymm4, ymm12)          // C[4,0:7]
vmovaps( ymm4, ymm13)          // C[4,8:15]
vmovaps( ymm4, ymm14)          // C[5,0:7]
vmovaps( ymm4, ymm15)          // C[5,8:15]

// Total: 12 ymm registers for 6×16 = 96 float accumulators
// (ymm registers are 256-bit, hold 8 floats each)
```

**Register Mapping:**
```
C[6×16] stored in ymm4-ymm15:

         N: 0-7          8-15
M: 0  →  ymm4           ymm5
   1  →  ymm6           ymm7
   2  →  ymm8           ymm9
   3  →  ymm10          ymm11
   4  →  ymm12          ymm13
   5  →  ymm14          ymm15
```

### Inner Loop (K dimension)

```asm
// Pre-load B panels
vmovaps(mem(rbx, -4*32), ymm0)  // Load B[k, 0:7]
vmovaps(mem(rbx, -3*32), ymm1)  // Load B[k, 8:15]

// Main K loop (unrolled by 4)
label(.SLOOPKITER)

    // Iteration 0
    vmovaps(mem(rbx, -2*32), ymm2)      // Prefetch next B
    vbroadcastss(mem(rax), xmm3)        // Broadcast A[0,k]
    vfmadd231ps(ymm0, ymm3, ymm4)       // C[0,0:7] += A[0,k] * B[k,0:7]
    vfmadd231ps(ymm1, ymm3, ymm5)       // C[0,8:15] += A[0,k] * B[k,8:15]

    vbroadcastss(mem(rax, r8, 1), xmm3) // Broadcast A[1,k]
    vfmadd231ps(ymm0, ymm3, ymm6)       // C[1,0:7] += A[1,k] * B[k,0:7]
    vfmadd231ps(ymm1, ymm3, ymm7)       // C[1,8:15] += A[1,k] * B[k,8:15]

    // ... (repeat for rows 2-5)

    add(imm(4*4), rax)                  // A += 4 elements
    add(imm(4*32), rbx)                 // B += 4*16 elements

    dec(rsi)                             // k_iter--
    jne(.SLOOPKITER)                     // Loop if k_iter > 0
```

**Key Insight: Broadcast Pattern**

```
For each k iteration:
  Load B[k, 0:15] once (2 ymm loads)
  Broadcast A[0:5, k] to each row (6 broadcasts)
  FMA into C[0:5, 0:15] (12 FMA operations)

Memory ops: 2 loads + 6 broadcasts = 8 mem ops
Compute ops: 12 FMAs = 96 FLOPs (8 ops/FMA * 12)
Arithmetic Intensity: 96 / (8 * 32 bytes) = 0.375 FLOPs/byte

But for K=64:
  Total FLOPs: 96 * 64 = 6,144
  Total bytes: (8 * 32) * 64 = 16,384
  AI = 6144 / 16384 = 0.375 FLOPs/byte (STILL LOW!)

Wait, this doesn't match our earlier calculation...
```

**Re-analysis: Panel Reuse**

Actually, the B panel is loaded once and reused 6 times (for each A row):

```
For K=64 iterations:
  Load B[0:63, 0:15] = 64 * 16 = 1024 floats = 4096 bytes
  Load A[0:5, 0:63] = 6 * 64 = 384 floats = 1536 bytes
  Compute C[0:5, 0:15] = 2 * 6 * 16 * 64 = 12,288 FLOPs

  AI = 12,288 / (4096 + 1536) = 12,288 / 5,632 = 2.18 FLOPs/byte
```

This is better, but still not ideal. The magic happens with **higher-level blocking** (L1/L2 cache reuse).

---

## OpenBLAS 16×2 Microkernel Analysis

From `/home/antshiv/Workspace/3rd-Party/MathLibrary/OpenBLAS/kernel/x86_64/dgemm_kernel_16x2_haswell.S`:

### Why 16×2?

16×2 is asymmetric on purpose:
- **M=16:** Fits perfectly in one AVX-512 register (16 doubles) or two ymm (16 doubles)
- **N=2:** Only 2 columns → minimal B loads

### Kernel Macro (Assembly)

```asm
.macro KERNEL16x3_SUBN
    prefetcht0    A_PR1(AO)                   // Prefetch next A
    vbroadcastsd  -12 * SIZE(BO), %ymm1       // Broadcast B[k, col0]
    vmovaps       -16 * SIZE(AO), %ymm0       // Load A[0:3, k]
    VFMADD231PD_  %ymm4,%ymm1,%ymm0          // C[0:3, col0] += A * B
    vbroadcastsd  -11 * SIZE(BO), %ymm2       // Broadcast B[k, col1]
    VFMADD231PD_  %ymm5,%ymm2,%ymm0          // C[0:3, col1] += A * B
    vbroadcastsd  -10 * SIZE(BO), %ymm3       // Broadcast B[k, col2]
    VFMADD231PD_  %ymm6,%ymm3,%ymm0          // C[0:3, col2] += A * B

    vmovaps       -12 * SIZE(AO), %ymm0       // Load A[4:7, k]
    VFMADD231PD_  %ymm7,%ymm1,%ymm0          // C[4:7, col0] += A * B
    VFMADD231PD_  %ymm8,%ymm2,%ymm0          // C[4:7, col1] += A * B
    VFMADD231PD_  %ymm9,%ymm3,%ymm0          // C[4:7, col2] += A * B

    // ... (repeat for A[8:11, k] and A[12:15, k])

    addq    $ 3*SIZE , BO                    // B += 3 elements
    addq    $ 16*SIZE, AO                    // A += 16 elements
.endm
```

**Register Allocation (16×2 for doubles, 16×3 shown):**
```
C[16×3] in ymm4-ymm15:
  ymm4  = C[0:3, col0]
  ymm5  = C[0:3, col1]
  ymm6  = C[0:3, col2]
  ymm7  = C[4:7, col0]
  ymm8  = C[4:7, col1]
  ymm9  = C[4:7, col2]
  ymm10 = C[8:11, col0]
  ymm11 = C[8:11, col1]
  ymm12 = C[8:11, col2]
  ymm13 = C[12:15, col0]
  ymm14 = C[12:15, col1]
  ymm15 = C[12:15, col2]
```

### Why This Works

**Small N (=2):**
- Minimizes B memory traffic (only 2 columns)
- Maximizes A reuse (16 rows share same B values)

**Large M (=16):**
- Vectorizes perfectly with AVX (4 doubles/ymm) or AVX-512 (8 doubles/zmm)
- Good parallelism over M dimension

**Trade-off:**
- BLIS 6×16: Wide N → better for column-major, more B reuse per A load
- OpenBLAS 16×2: Tall M → better for row-major, more A reuse per B load

---

## Microkernel Design Patterns

### Pattern 1: Broadcast-FMA (BLIS 6×16)

```c
// Pseudocode for 6×16 microkernel
void gemm_6x16_broadcast(const float *A, const float *B, float *C, int K) {
    __m256 c[6][2];  // 6 rows × 2 ymm (for 16 cols)

    // Zero accumulators
    for (int i = 0; i < 6; i++) {
        c[i][0] = _mm256_setzero_ps();
        c[i][1] = _mm256_setzero_ps();
    }

    for (int k = 0; k < K; k++) {
        // Load B[k, 0:15] once
        __m256 b0 = _mm256_loadu_ps(&B[k * 16 + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[k * 16 + 8]);

        // Broadcast A[i, k] and FMA for each row
        for (int i = 0; i < 6; i++) {
            __m256 a = _mm256_broadcast_ss(&A[i * K + k]);
            c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
        }
    }

    // Write back
    for (int i = 0; i < 6; i++) {
        _mm256_storeu_ps(&C[i * 16 + 0], c[i][0]);
        _mm256_storeu_ps(&C[i * 16 + 8], c[i][1]);
    }
}
```

**Pros:**
- B loaded once per K iteration
- Good for wide N

**Cons:**
- Many broadcasts (1 per row per K)

---

### Pattern 2: Load-Broadcast (OpenBLAS 16×2)

```c
// Pseudocode for 16×2 microkernel
void gemm_16x2_load(const float *A, const float *B, float *C, int K) {
    __m512 c0 = _mm512_setzero_ps();  // C[0:15, 0]
    __m512 c1 = _mm512_setzero_ps();  // C[0:15, 1]

    for (int k = 0; k < K; k++) {
        // Load A[0:15, k] once (16 elements = 1 zmm)
        __m512 a = _mm512_loadu_ps(&A[k * 16]);

        // Broadcast B[k, 0] and B[k, 1]
        __m512 b0 = _mm512_set1_ps(B[k * 2 + 0]);
        __m512 b1 = _mm512_set1_ps(B[k * 2 + 1]);

        // FMA
        c0 = _mm512_fmadd_ps(a, b0, c0);
        c1 = _mm512_fmadd_ps(a, b1, c1);
    }

    // Write back
    _mm512_storeu_ps(&C[0], c0);
    _mm512_storeu_ps(&C[16], c1);
}
```

**Pros:**
- Minimal B loads (only 2 scalars per K)
- A loaded once and used for both columns

**Cons:**
- Small N → many microkernel invocations needed

---

## Choosing Microkernel Dimensions

### Register Constraint

```
Available registers: R (e.g., 32 AVX-512 registers)
Vector width: W (e.g., 16 floats for zmm)

C accumulators need: ceil(Mr * Nr / W) registers
A temp: 1-2 registers
B temp: 1-2 registers (or broadcasts, which reuse)
Pointers/counters: ~4 registers

Total: ceil(Mr * Nr / W) + 4 + 2 ≤ R

For AVX-512 (R=32, W=16):
  ceil(Mr * Nr / 16) + 6 ≤ 32
  Mr * Nr ≤ 26 * 16 = 416

  Examples that fit:
  - 16×2 = 32 → ceil(32/16) + 6 = 8 ✓
  - 8×4 = 32 → ceil(32/16) + 6 = 8 ✓
  - 6×8 = 48 → ceil(48/16) + 6 = 10 ✓
  - 16×16 = 256 → ceil(256/16) + 6 = 22 ✓
  - 20×20 = 400 → ceil(400/16) + 6 = 31 ✓ (tight!)
```

### Performance Trade-offs

| Dimension | Pros | Cons | Best For |
|-----------|------|------|----------|
| **16×2** | Minimal B traffic, simple | Many invocations | Small N, autoregressive |
| **6×8** | Balanced, fits many archs | Non-power-of-2 awkward | General purpose |
| **8×6** | Same as 6×8, transposed | Same | Different loop order |
| **16×16** | Maximum FLOP rate | High register pressure | Large matrices |
| **4×12** | Small M, wide N | - | Attention Q·K^T |

---

## Implementation for C-Kernel-Engine

### Step 1: Choose Dimensions

Based on LLM shapes from `01-llm-kernel-shapes.md`:

**Target shapes:**
1. **Autoregressive decode:** M=1-16, N=large (768-4096), K=large (768-4096)
   → Use **16×2** or **16×4** microkernel (small N, tall M)

2. **Prompt processing:** M=large (256-2048), N=large, K=large
   → Use **6×8** or **8×6** microkernel (balanced)

3. **Attention scores:** M=T (512), N=T (512), K=d (64-128)
   → Use **16×16** microkernel (square, medium size)

### Step 2: Implement 16×2 Microkernel (Priority)

```c
// src/microkernel_16x2.c

static inline void gemm_microkernel_16x2_avx512(
    const float *A,      // [16 × K], row-major
    const float *B,      // [K × 2], row-major
    float *C,            // [16 × 2], row-major
    int K,
    int lda,             // Leading dimension of A (usually K)
    int ldb,             // Leading dimension of B (usually 2)
    int ldc)             // Leading dimension of C (usually 2)
{
    __m512 c0 = _mm512_setzero_ps();  // C[0:15, 0]
    __m512 c1 = _mm512_setzero_ps();  // C[0:15, 1]

    for (int k = 0; k < K; k++) {
        // Load A[0:15, k]
        __m512 a = _mm512_loadu_ps(&A[k * lda]);

        // Broadcast B[k, 0] and B[k, 1]
        __m512 b0 = _mm512_set1_ps(B[k * ldb + 0]);
        __m512 b1 = _mm512_set1_ps(B[k * ldb + 1]);

        // FMA
        c0 = _mm512_fmadd_ps(a, b0, c0);
        c1 = _mm512_fmadd_ps(a, b1, c1);
    }

    // Write back C[0:15, 0:1]
    for (int i = 0; i < 16; i++) {
        C[i * ldc + 0] = ((float*)&c0)[i];
        C[i * ldc + 1] = ((float*)&c1)[i];
    }
}

// BETTER: Use masked stores or transpose for column writes
```

Wait, there's an issue with the writeback. Let me fix:

```c
static inline void gemm_microkernel_16x2_avx512(
    const float *A,      // [16 × K], A[i,k] = A[i*lda + k]
    const float *B,      // [K × 2], B[k,j] = B[k*ldb + j]
    float *C,            // [16 × 2], C[i,j] = C[i*ldc + j]
    int K, int lda, int ldb, int ldc)
{
    // For row-major C with ldc=2, we can store directly
    // But for general ldc, we need to scatter or transpose

    // Simplified version assuming ldc is such that we can vectorize:
    // This assumes C is stored with rows contiguous

    float c_local[16][2] __attribute__((aligned(64)));

    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();

    for (int k = 0; k < K; k++) {
        __m512 a = _mm512_loadu_ps(&A[k * lda]);
        __m512 b0 = _mm512_set1_ps(B[k * ldb + 0]);
        __m512 b1 = _mm512_set1_ps(B[k * ldb + 1]);
        c0 = _mm512_fmadd_ps(a, b0, c0);
        c1 = _mm512_fmadd_ps(a, b1, c1);
    }

    // Store to local buffer
    _mm512_store_ps(&c_local[0][0], c0);
    _mm512_store_ps(&c_local[0][1], c1); // Wait, this doesn't work...

    // Actually, let's store as two separate zmm and then copy:
    float temp0[16] __attribute__((aligned(64)));
    float temp1[16] __attribute__((aligned(64)));
    _mm512_store_ps(temp0, c0);
    _mm512_store_ps(temp1, c1);

    for (int i = 0; i < 16; i++) {
        C[i * ldc + 0] = temp0[i];
        C[i * ldc + 1] = temp1[i];
    }
}
```

**Better Approach: Store as 32 consecutive floats, then transpose**

```c
void gemm_microkernel_16x2_avx512_optimized(
    const float *A, const float *B, float *C,
    int K, int lda, int ldb, int ldc)
{
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();

    for (int k = 0; k < K; k++) {
        __m512 a = _mm512_loadu_ps(&A[k * lda]);
        __m512 b0 = _mm512_set1_ps(B[k * ldb + 0]);
        __m512 b1 = _mm512_set1_ps(B[k * ldb + 1]);
        c0 = _mm512_fmadd_ps(a, b0, c0);
        c1 = _mm512_fmadd_ps(a, b1, c1);
    }

    // If ldc == 2, we can interleave and store:
    if (ldc == 2) {
        // Interleave c0 and c1 using shuffle/permute
        // Then store 32 floats contiguously
        // (Complex, but worth it for performance)
        // TODO: Implement with _mm512_mask_blend_ps or similar
    } else {
        // Fallback: scalar stores
        float temp0[16] __attribute__((aligned(64)));
        float temp1[16] __attribute__((aligned(64)));
        _mm512_store_ps(temp0, c0);
        _mm512_store_ps(temp1, c1);

        for (int i = 0; i < 16; i++) {
            C[i * ldc + 0] = temp0[i];
            C[i * ldc + 1] = temp1[i];
        }
    }
}
```

---

## Integrating Microkernel into GEMM

### High-Level Flow

```c
void gemm_with_microkernel(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    const int Mr = 16;  // Microkernel M dimension
    const int Nr = 2;   // Microkernel N dimension

    for (int i = 0; i < M; i += Mr) {
        int m_cur = (i + Mr <= M) ? Mr : (M - i);

        for (int j = 0; j < N; j += Nr) {
            int n_cur = (j + Nr <= N) ? Nr : (N - j);

            if (m_cur == Mr && n_cur == Nr) {
                // Full microkernel
                gemm_microkernel_16x2_avx512(&A[i*K], &B[j], &C[i*N + j],
                                             K, K, N, N);
            } else {
                // Edge case: use scalar code or masked SIMD
                gemm_edge_case(&A[i*K], &B[j], &C[i*N + j],
                               m_cur, n_cur, K, K, N, N);
            }
        }
    }
}
```

---

## Summary: Microkernel Checklist

- [ ] **Choose dimensions** based on target shapes (16×2 for LLM decode)
- [ ] **Implement core loop** with FMA instructions
- [ ] **Handle edge cases** (M/N not divisible by Mr/Nr)
- [ ] **Optimize writeback** (interleave stores if possible)
- [ ] **Add prefetching** (A_PR1, B_PR1 offsets)
- [ ] **Profile** with `perf stat -e cycles,instructions`
- [ ] **Validate** numerical correctness (diff < 1e-5)

**Next:** [04-ai-shape-optimizations.md](04-ai-shape-optimizations.md) for shape-specific dispatch strategy.
