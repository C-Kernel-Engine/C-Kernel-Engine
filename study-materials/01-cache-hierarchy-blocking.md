# Cache Hierarchy and Blocking Strategy

## The Central Question: Why 6×8 or 16×2 Instead of 64×64?

When you first see block sizes like 6×8, 8×6, or 16×2 in BLIS and OpenBLAS, it seems counterintuitive. After all, cache lines are 64 bytes, so shouldn't we block at 64×64?

**The answer lies in understanding THREE levels of blocking, not just one.**

## The Three-Level Blocking Hierarchy

### Level 1: Register Blocking (Microkernel) - 6×8, 8×6, 16×2
**Target:** CPU Register File (32 AVX-512 registers on x86-64)

### Level 2: L1 Cache Blocking - ~384 × ~96
**Target:** L1 Data Cache (32 KB typical)

### Level 3: L2/L3 Cache Blocking - ~4096 × ~256
**Target:** L2/L3 Cache (256 KB - 2 MB typical)

Let's understand each level.

---

## Level 1: Register Blocking (The Microkernel)

### Why 6×8 in BLIS?

The **microkernel** is the innermost computation loop that accumulates into registers. Its size is determined by **register count**, not cache size.

#### Hardware Constraints (AVX-512 on x86-64)

```
Available registers:
- 32 ZMM registers (zmm0-zmm31), each holds 16 floats (512 bits)
- We need registers for:
  * Accumulators (C matrix tiles)
  * A matrix elements
  * B matrix elements
  * Temporary values
```

#### BLIS 6×8 Microkernel Register Allocation

```c
// Pseudo-code for 6×8 microkernel
void gemm_microkernel_6x8(float *A, float *B, float *C, int K) {
    // Accumulator registers: 6×8 = 48 floats
    // But we pack into vectors of 16 floats each
    // So we need: ceil(6×8 / 16) = 3 vector registers
    __m512 c00_15, c16_31, c32_47;  // 3 registers for C

    // Actually, it's organized as 6 rows × 8 cols:
    __m512 c0_vec[3];  // Row 0-1 (16 elements, covers columns 0-7 twice)
    __m512 c1_vec[3];  // Row 2-3
    __m512 c2_vec[3];  // Row 4-5
    // Total: 9 registers for C accumulators? No!

    // Real layout: 6 rows, 8 columns, but vectorized cleverly
    // Let's see how BLIS actually does it...
}
```

Wait, let me recalculate this properly.

#### The Real Reason for 6×8

A 6×8 tile means:
- **6 rows** from matrix A (M dimension)
- **8 columns** from matrix B (N dimension)
- Produces a **6×8 output** tile in C

In registers:
```
C[6×8] accumulator needs 48 scalar values

With AVX-512 (16 floats per register):
- We can't fit 48 floats perfectly into registers
- But we organize as: 6 vectors × 8 cols = requires thinking differently

Actually, BLIS uses:
- Load 1 vector from A (broadcasts across or loads 6 elements)
- Load 1 vector from B (8 elements fit in one zmm if we process 8 at once)
- Accumulate into C tile

Let me look at the actual pattern...
```

#### OpenBLAS 16×2 Microkernel

This is easier to understand:

```c
// 16×2 means:
// - 16 rows (M dimension)
// - 2 columns (N dimension)

__m512 c_col0[16/16];  // 16 elements for column 0 = 1 zmm register
__m512 c_col1[16/16];  // 16 elements for column 1 = 1 zmm register
// Total: 2 zmm registers for C

__m512 a_vec[16/16];   // 16 elements from A = 1 zmm register
__m512 b_col0;         // Broadcast B[0,k] across all 16 floats
__m512 b_col1;         // Broadcast B[1,k] across all 16 floats

for (int k = 0; k < K; k++) {
    a_vec = _mm512_loadu_ps(&A[k]);  // Load 16 elements from A

    b_col0 = _mm512_set1_ps(B[0*K + k]);  // Broadcast B[0,k]
    c_col0 = _mm512_fmadd_ps(a_vec, b_col0, c_col0);

    b_col1 = _mm512_set1_ps(B[1*K + k]);  // Broadcast B[1,k]
    c_col1 = _mm512_fmadd_ps(a_vec, b_col1, c_col1);
}
// Total register usage: ~4 zmm registers
```

**Why 16×2?**
- 16 rows fit perfectly in one AVX-512 register
- Process 2 columns at a time (minimizes B loads)
- Only needs 4-5 registers total
- Leaves plenty of registers for loop counters, pointers, etc.

#### The Register Constraint Formula

```
Available registers: R (typically 32 for AVX-512)
Registers needed for C: Mr × Nc / 16  (assuming fp32, 16 per zmm)
Registers needed for A: 1-2 (temporary loads)
Registers needed for B: Nc (one per column, broadcast)
Registers for pointers/counters: ~4

Total needed: (Mr × Nc / 16) + Nc + 2 + 4 < R

For 16×2:
  (16 × 2 / 16) + 2 + 2 + 4 = 2 + 2 + 2 + 4 = 10 registers ✓

For 6×8:
  More complex due to non-power-of-2, but works with clever packing
```

### Key Insight 1: Register Blocking ≠ Cache Line Size

**Cache line is 64 BYTES**, not 64 floats!
- 64 bytes = 16 floats (fp32) = 1 AVX-512 register

**The microkernel is bounded by REGISTER COUNT, not cache lines.**

---

## Level 2: L1 Cache Blocking

Now we block for the L1 cache (32 KB typical).

### Memory Layout in GEMM

```
C[M×N] = A[M×K] × B[K×N]

For a blocked computation:
- Load a panel of A: [Mc × Kc]
- Load a panel of B: [Kc × Nc]
- Compute C tile: [Mc × Nc]
```

### L1 Cache Capacity

```
L1 Data Cache: 32 KB = 32,768 bytes = 8,192 floats

We want to fit:
- A panel: Mc × Kc floats
- B panel: Kc × Nc floats
- C tile: Mc × Nc floats (if we want to accumulate in cache)

Total: (Mc × Kc) + (Kc × Nc) + (Mc × Nc) ≤ 8,192 floats
```

### Typical L1 Block Sizes

**BLIS typically uses:**
```
Mc = 384  (divisible by 6, the microkernel M size)
Nc = 4096 (divisible by 8, the microkernel N size)
Kc = 384  (inner dimension)

But wait, this doesn't fit in L1!
384 × 384 + 384 × 4096 + 384 × 4096 = 3,294,720 floats >> 8,192

So BLIS is NOT blocking for L1 with these sizes...
```

Actually, the L1 blocking is more subtle. Let me reconsider:

### The Real L1 Blocking Strategy

The **microkernel** itself fits in L1:
```
6×8 microkernel processes:
- A panel: 6 rows × K elements
- B panel: K elements × 8 cols
- C tile: 6 × 8 = 48 floats (accumulates in registers!)

For K iterations (the inner loop):
- A accesses: 6 floats per iteration
- B accesses: 8 floats per iteration
- Total per iteration: 14 floats = 56 bytes

For K=384 (a typical Kc):
- A: 6 × 384 = 2,304 floats = 9,216 bytes
- B: 8 × 384 = 3,072 floats = 12,288 bytes
- Total: 21,504 bytes ✓ Fits in 32 KB L1
```

**This is why the microkernel dimensions matter!**
- Small Mr (6 rows) and small Nr (8 cols) ensure the panels fit in L1
- The C tile (6×8) stays in registers

---

## Level 3: L2/L3 Cache Blocking

This is where the traditional "64×64 blocking" makes sense, but actually we go much larger.

### L2 Cache Capacity

```
L2 Cache: 256 KB - 1 MB (varies by CPU)
L3 Cache: 2 MB - 32 MB (shared across cores)

For L2 blocking:
- Block size: 256 × 256 floats = 65,536 floats × 4 bytes = 262 KB
- This is the "Mc × Kc" panel of A that we want to keep in L2
```

### OpenBLAS Parameters (from dgemm_kernel_16x2_haswell.S)

```c
DGEMM_DEFAULT_P = 192   // Mc (M-dimension blocking for L2)
DGEMM_DEFAULT_Q = 128   // Nc (N-dimension blocking for L2)
// Kc is determined dynamically

Memory for one L2 block:
A panel: 192 × Kc
B panel: Kc × 128
```

### Our 64×64 Blocking

In C-Kernel-Engine, we use:
```c
const int block_size = 64;  // Single block size for all dimensions

for (int ii = 0; ii < M; ii += 64) {
    for (int jj = 0; jj < N; jj += 64) {
        for (int kk = 0; kk < K; kk += 64) {
            // Process 64×64×64 block
        }
    }
}
```

**Memory footprint:**
```
A block: 64 × 64 = 4,096 floats = 16 KB
B block: 64 × 64 = 4,096 floats = 16 KB
C block: 64 × 64 = 4,096 floats = 16 KB
Total: 48 KB - fits in L1! ✓

But this is TOO SMALL for L2/L3 blocking!
```

**Problem:** We're not utilizing the full cache hierarchy effectively.

---

## The Complete Picture: Three-Level Blocking

### Optimal Strategy (BLIS-style)

```
┌─────────────────────────────────────────────────────────┐
│ Level 3: L3 Cache Blocking (Largest)                    │
│   Block size: Mc=4096, Nc=4096, Kc=256                  │
│   Load entire A panel into L3                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Level 2: L2 Cache Blocking (Medium)                     │
│   Block size: Mc=384, Nc=96, Kc=384                     │
│   Keep B panel in L2 as we iterate over M               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Level 1: Register Blocking (Microkernel)                │
│   Block size: Mr=6, Nr=8, Kr=1                          │
│   Accumulate in registers, write once to C              │
└─────────────────────────────────────────────────────────┘
```

### Comparison: Our Implementation vs. BLIS

| Aspect | C-Kernel-Engine (Ours) | BLIS | OpenBLAS |
|--------|------------------------|------|----------|
| **Microkernel** | None (AVX-512 in loops) | 6×8 | 16×2 |
| **L1 Blocking** | 64×64×64 (implicit) | Mr×Nr in regs | 16×2 in regs |
| **L2 Blocking** | 64×64×64 (same!) | ~384×384×96 | ~192×128×Kc |
| **L3 Blocking** | None | ~4096×4096×256 | Implicit |
| **Levels** | 1 (cache-only) | 3 (reg + L1/L2/L3) | 3 (reg + L2/L3) |

**Our Issue:** We're only doing L1 cache blocking, and it's too small to effectively use L2/L3.

---

## Computational Complexity Analysis

### FLOPs: The Math

For `C[M×N] = A[M×K] × B[K×N]`:
```
Total FLOPs = 2 × M × N × K  (multiply + add per element)

For M=N=K=1024:
  FLOPs = 2 × 1024³ = 2,147,483,648 FLOPs ≈ 2.15 GFLOP
```

### Memory Traffic: The Bottleneck

#### Naive GEMM (No Blocking)

```
for i in 0..M:
  for j in 0..N:
    for k in 0..K:
      C[i,j] += A[i,k] * B[k,j]

Memory accesses per iteration:
- Read A[i,k]: 1 load
- Read B[k,j]: 1 load
- Read C[i,j]: 1 load (for accumulation)
- Write C[i,j]: 1 store
Total: 4 memory ops per 2 FLOPs

For M=N=K=1024:
  Memory ops = 4 × 1024³ = 4.3 billion memory ops
  Bytes = 4.3B × 4 bytes = 17.2 GB
```

But wait, matrices are only:
```
A: 1024 × 1024 = 1 MB
B: 1024 × 1024 = 1 MB
C: 1024 × 1024 = 1 MB
Total: 3 MB

Why 17.2 GB of traffic?
→ CACHE MISSES! Each element accessed multiple times from DRAM.
```

#### Blocked GEMM

With proper blocking, we read A and B **once from DRAM**, then reuse from cache.

**Optimal Traffic:**
```
Read A: M × K floats = 1 MB
Read B: K × N floats = 1 MB
Write C: M × N floats = 1 MB
Total: 3 MB ✓
```

**Arithmetic Intensity:**
```
AI = FLOPs / Bytes
   = (2 × M × N × K) / (M×K + K×N + M×N) floats
   = (2 × 1024³) / (3 × 1024²) / 4 bytes
   = 2.15 GFLOP / 3 MB
   = 2,147,483,648 / 3,145,728
   = 682 FLOPs/byte

This is the IDEAL. Realistic is lower due to cache capacity.
```

### Hardware Peak Performance

**Your CPU (likely Intel Haswell/Skylake AVX-512):**
```
FMA units: 2 per core (can do 2 FMAs per cycle)
AVX-512: 16 fp32 ops per FMA instruction
Clock: ~3.0 GHz

Peak GFLOPS per core:
  = 2 FMA/cycle × 16 ops/FMA × 2 flops/op × 3.0 GHz
  = 2 × 16 × 2 × 3.0 = 192 GFLOPS per core

For 8 cores: 192 × 8 = 1,536 GFLOPS peak
```

**Memory Bandwidth:**
```
DDR4-2400: ~75 GB/s (typical dual-channel)
DDR4-3200: ~100 GB/s

Arithmetic Intensity needed to be compute-bound:
  AI > GFLOPS / Bandwidth
  AI > 1536 GFLOPS / 75 GB/s
  AI > 20.5 FLOPs/byte

For GEMM with good blocking: AI ≈ 200-600 FLOPs/byte
→ Compute-bound (good!) ✓
```

---

## Why Our 64×64 Blocking Works (But Isn't Optimal)

### What We're Doing Right

1. **Cache-Friendly:** 64×64×64 blocks fit in L1 (48 KB)
2. **AVX-512 Vectorization:** Using 16-wide SIMD
3. **Bias Fusion:** Reducing memory traffic

### What We're Missing

1. **No Microkernel:** Not maximizing register reuse
2. **Single Block Size:** Not adapting to M/N/K dimensions
3. **No L2/L3 Blocking:** Underutilizing larger caches
4. **No Prefetching:** Suffering from cache miss latency
5. **No Shape Specialization:** Same kernel for M=1 and M=512

### Performance Gap Estimate

```
Our current performance (rough estimate):
  64×64 blocking: ~40-60 GFLOPS on 8 cores
  % of peak: 40 / 1536 = 2.6% (single core equiv)

BLIS/OpenBLAS performance:
  Optimized blocking: ~120-150 GFLOPS on 8 cores
  % of peak: ~10% per core (good for CPU-only)

Gap: 2-3x slower
Reason: Blocking strategy + microkernel design
```

---

## Action Items: How to Improve

### Priority 1: Add Register Blocking (Microkernel)

Instead of:
```c
for (int k = kk; k <= k_end - 16; k += 16) {
    __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
    __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
    sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
}
```

Use a proper microkernel:
```c
// 16×2 microkernel (OpenBLAS style)
void gemm_microkernel_16x2(const float *A, const float *B, float *C,
                            int K, int lda, int ldb, int ldc) {
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();

    for (int k = 0; k < K; k++) {
        __m512 a = _mm512_loadu_ps(&A[k * lda]);  // 16 elements

        __m512 b0 = _mm512_set1_ps(B[0 * ldb + k]);  // Broadcast
        c0 = _mm512_fmadd_ps(a, b0, c0);

        __m512 b1 = _mm512_set1_ps(B[1 * ldb + k]);  // Broadcast
        c1 = _mm512_fmadd_ps(a, b1, c1);
    }

    _mm512_storeu_ps(&C[0 * ldc], c0);
    _mm512_storeu_ps(&C[1 * ldc], c1);
}
```

### Priority 2: Add Three-Level Blocking

```c
// Pseudocode
void gemm_optimized(A, B, C, M, N, K) {
    // Level 3: L3 cache blocking
    for (int ii = 0; ii < M; ii += Mc) {  // Mc = 384
        for (int jj = 0; jj < N; jj += Nc) {  // Nc = 4096
            for (int kk = 0; kk < K; kk += Kc) {  // Kc = 256

                // Level 2: L2 cache blocking (if needed)

                // Level 1: Microkernel
                for (int i = ii; i < ii + Mc; i += Mr) {  // Mr = 16
                    for (int j = jj; j < jj + Nc; j += Nr) {  // Nr = 2
                        gemm_microkernel_16x2(&A[i*K + kk],
                                               &B[kk*N + j],
                                               &C[i*N + j],
                                               Kc, K, N, N);
                    }
                }
            }
        }
    }
}
```

### Priority 3: Shape-Specific Kernels

```c
// Dispatch based on M size
if (M <= 16) {
    gemm_small_m(A, B, C, M, N, K);  // Optimized for autoregressive decode
} else if (M >= 128) {
    gemm_large_m(A, B, C, M, N, K);  // Full blocking strategy
} else {
    gemm_medium_m(A, B, C, M, N, K);  // Hybrid approach
}
```

---

## Summary: The Blocking Hierarchy

| Level | Target | Size | Purpose |
|-------|--------|------|---------|
| **Microkernel (Mr×Nr)** | Registers | 6×8, 8×6, 16×2 | Maximize register reuse, minimize load/store |
| **L1 Block** | L1 Cache | Microkernel panels | Keep innermost loop hot in L1 |
| **L2 Block (Mc×Nc)** | L2 Cache | 384×96, 192×128 | Reuse B panel across multiple A rows |
| **L3 Block** | L3 Cache | 4096×4096 | Minimize DRAM traffic |

**Cache line size (64 bytes) matters for alignment, but block sizes are determined by:**
1. **Register count** → Microkernel dimensions
2. **Cache capacity** → Block sizes at each level
3. **Memory bandwidth** → Minimize DRAM traffic

---

**Next:** [08-computational-complexity.md](08-computational-complexity.md) for detailed roofline analysis.
