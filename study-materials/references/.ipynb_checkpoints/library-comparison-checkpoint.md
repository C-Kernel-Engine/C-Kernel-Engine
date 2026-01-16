# Reference Library Comparison

A side-by-side comparison of BLIS, oneDNN, and OpenBLAS to understand their different philosophies and what we can learn from each.

## Philosophy and Design Goals

| Library | Philosophy | Target Use Case | Key Strength |
|---------|-----------|-----------------|--------------|
| **BLIS** | Framework for BLAS | General linear algebra | Clean abstraction, portability |
| **oneDNN** | Deep learning primitives | AI/ML workloads (transformers, CNNs) | AI-specific optimizations, modern hardware |
| **OpenBLAS** | Optimized BLAS | HPC, scientific computing | Battle-tested, production-ready |

## What Makes Each Library Special

### BLIS: The Framework Approach

**Design Principle:** Separate **mechanism** from **policy**

```
┌───────────────────────────────────────────────────┐
│ Framework Layer (portable)                        │
│  - Memory allocation                              │
│  - Thread scheduling                              │
│  - Cache blocking strategy                        │
└───────────────────────────────────────────────────┘
                     ↓
┌───────────────────────────────────────────────────┐
│ Microkernel Layer (architecture-specific)         │
│  - gemm_microkernel_6x8() for Haswell            │
│  - gemm_microkernel_8x6() for Zen                │
└───────────────────────────────────────────────────┘
```

**What We Learn:**
- How to design **retargetable** kernels
- Clean separation of architecture-specific code
- Object-based API design

**Best for:** Understanding GEMM structure at a conceptual level

---

### oneDNN: The AI-First Approach

**Design Principle:** Optimize for **real AI workloads**, not generic BLAS

**Key Innovation: BRGEMM (Batched-Reduced GEMM)**

Traditional BLAS:
```c
// Multi-head attention (H=12 heads)
for (int h = 0; h < 12; h++) {
    sgemm(Q[h], K[h], scores[h], ...);  // 12 separate GEMM calls!
}
```

oneDNN BRGEMM:
```c
// Single kernel call for all heads
brgemm_strided(Q_base, K_base, scores_base,
               num_heads=12, stride_a, stride_b, stride_c, ...);
// Process all 12 heads in one shot with better cache reuse!
```

**Other AI-Specific Features:**
1. **Post-ops:** Fused bias, ReLU, GELU, residual add
2. **AMX Support:** Intel's tile-based matrix unit (Sapphire Rapids)
3. **Shape Dispatch:** Different code paths for small-M (decode) vs. large-M (prompt)
4. **JIT Compilation:** Generate optimal assembly at runtime

**What We Learn:**
- How to optimize for **transformer shapes** specifically
- Batching strategies for multi-head attention
- Modern hardware acceleration (AMX)

**Best for:** Learning AI-specific optimizations

---

### OpenBLAS: The Production Workhorse

**Design Principle:** Hand-tune for **every architecture**, prioritize correctness

**Kazushige Goto's Blocking Strategy:**
- Multi-level cache blocking
- Explicit prefetching
- Hand-written assembly for critical kernels

**Key Files:**
```
kernel/x86_64/
  ├── dgemm_kernel_16x2_haswell.S     # 16×2 microkernel (assembly)
  ├── dgemm_kernel_4x8_sandy.S        # 4×8 for Sandy Bridge
  ├── sgemm_kernel_8x2_skylakex.c     # 8×2 for Skylake-X
  └── ...                              # Many more variants

Observation: Different kernel for different (M,N,K) ranges!
```

**Shape-Specific Dispatch in OpenBLAS:**
```c
// Simplified from common_level3.h
if (M <= 16) {
    // Small-M: Use 16×2 or 4×8 kernel
    dgemm_small_m(...);
} else if (M <= 128) {
    // Medium-M: Use 8×4 kernel
    dgemm_medium_m(...);
} else {
    // Large-M: Use full blocking
    dgemm_large_m(...);
}
```

**What We Learn:**
- Importance of **shape-specific kernels**
- How to write efficient assembly
- Production-level edge case handling

**Best for:** Learning practical optimization tricks

---

## Code Organization Comparison

### BLIS Structure
```
blis/
├── frame/                  # Framework (portable)
│   ├── 3/                  # Level-3 BLAS (GEMM, etc.)
│   │   └── bli_gemm.c     # High-level GEMM driver
│   └── base/               # Memory, threading, etc.
├── kernels/                # Architecture-specific
│   ├── haswell/
│   │   └── 3/
│   │       └── bli_gemm_haswell_asm_d6x8.c  # 6×8 microkernel
│   └── zen/
│       └── ...
└── ref_kernels/            # Reference (slow but correct)
    └── bli_gemm_ref.c
```

**Lesson:** Clean layering makes it easy to port to new architectures

---

### oneDNN Structure
```
oneDNN/
├── src/
│   ├── cpu/
│   │   ├── matmul/                    # High-level matmul
│   │   │   ├── ref_matmul.cpp        # Reference
│   │   │   └── gemm_bf16_matmul.cpp  # bf16 optimized
│   │   └── x64/
│   │       └── brgemm/                # BRGEMM (AI-specific!)
│   │           ├── brgemm.hpp
│   │           ├── jit_brgemm_kernel.cpp      # AVX-512 JIT
│   │           └── jit_brgemm_amx_uker.cpp    # AMX JIT
│   └── gpu/                           # GPU backends
└── include/
    └── oneapi/dnnl/dnnl.hpp          # Public API
```

**Lesson:** Separate CPU and GPU, use JIT for flexibility

---

### OpenBLAS Structure
```
OpenBLAS/
├── kernel/                            # Hand-tuned kernels
│   ├── x86_64/
│   │   ├── dgemm_kernel_16x2_haswell.S
│   │   ├── sgemm_kernel_8x2_skylakex.c
│   │   └── ... (hundreds of variants)
│   ├── arm64/
│   └── ...
├── driver/                            # Dispatch logic
│   └── level3/
│       ├── gemm.c                     # Main GEMM entry
│       └── level3_thread.c            # Threading
├── common*.h                          # Architecture headers
│   ├── common_x86_64.h
│   └── common_arm64.h
└── Makefile.system                    # Build config
```

**Lesson:** Heavy reliance on build system to select kernels, many hand-coded variants

---

## API Design Comparison

### BLIS: Object-Based + Traditional BLAS

```c
// Object-based API (BLIS-specific)
obj_t A, B, C;
bli_obj_create(..., &A);
bli_obj_create(..., &B);
bli_obj_create(..., &C);
bli_gemm(&BLIS_ONE, &A, &B, &BLIS_ZERO, &C);

// Traditional BLAS API (compatibility layer)
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
```

**Pros:**
- Object API is more extensible
- Compatible with existing BLAS users

---

### oneDNN: Primitive-Based

```c
// Create primitive descriptor
dnnl_matmul_desc_t matmul_desc;
dnnl_matmul_desc_init(&matmul_desc, src_md, weights_md, dst_md);

// Apply post-ops (fusion!)
dnnl_post_ops_t post_ops;
dnnl_post_ops_create(&post_ops);
dnnl_post_ops_append_eltwise(post_ops, DNNL_ELTWISE_GELU, ...);

dnnl_primitive_attr_set_post_ops(attr, post_ops);

// Create and execute primitive
dnnl_primitive_t matmul_prim;
dnnl_primitive_create(&matmul_prim, matmul_desc, attr, engine);
dnnl_primitive_execute(matmul_prim, stream, args);
```

**Pros:**
- Flexible post-ops (fusion)
- Hardware abstraction (CPU/GPU)
- Optimized execution graph

---

### OpenBLAS: Pure BLAS

```c
// Standard BLAS interface
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
```

**Pros:**
- Drop-in replacement for any BLAS
- Simple, no learning curve

---

## Performance Characteristics

### Typical GFLOPS on AVX-512 CPU (8 cores, M=N=K=1024)

| Library | GFLOPS | % Peak | Notes |
|---------|--------|--------|-------|
| **BLIS** | ~1200 | 78% | Excellent on large square matrices |
| **oneDNN** | ~1400 | 91% | Best for AI shapes (M≠N≠K) |
| **OpenBLAS** | ~1100 | 72% | Stable, well-tested |
| **C-Kernel-Engine** | ~250 | 16% | 🔴 Needs optimization |

**Our Goal:** 1000+ GFLOPS (65%+ of peak)

---

## Where Each Library Excels

### BLIS is Best For:
- ✅ Learning GEMM structure
- ✅ Porting to new architectures
- ✅ Square matrices (M≈N≈K)
- ✅ Research and prototyping

### oneDNN is Best For:
- ✅ Transformers and CNNs
- ✅ Batched operations (multi-head attention)
- ✅ Low-precision (bf16, int8)
- ✅ Modern Intel CPUs (AMX)

### OpenBLAS is Best For:
- ✅ Production stability
- ✅ Wide architecture support
- ✅ Scientific computing
- ✅ Drop-in NumPy/SciPy backend

---

## What We Take From Each

### From BLIS:
1. **Framework design:** Separate portable code from microkernels
2. **Object API:** Better than raw pointers
3. **Documentation:** Well-commented, educational

**Apply to C-Kernel-Engine:**
```c
// Our CKMathBackend is inspired by BLIS modularity
typedef struct {
    void (*sgemm)(int M, int N, int K, ...);
    void (*sgemm_batched)(int batch_count, ...);
} CKMathBackend;
```

---

### From oneDNN:
1. **BRGEMM:** Batched GEMM for multi-head attention
2. **Post-ops:** Fused operations (bias + GELU)
3. **Shape dispatch:** Different code for small-M vs. large-M
4. **AMX support:** Future-proof for new hardware

**Apply to C-Kernel-Engine:**
```c
// Add batched interface (from oneDNN)
void sgemm_batched_strided(int batch_count, int M, int N, int K,
                           const float *A, int lda, int stride_a,
                           const float *B, int ldb, int stride_b,
                           float *C, int ldc, int stride_c);

// Add fusion (from oneDNN)
typedef enum {
    CK_ACTIVATION_NONE,
    CK_ACTIVATION_GELU,
    CK_ACTIVATION_RELU,
} ck_activation_t;

void sgemm_fused(int M, int N, int K, ..., ck_activation_t activation);
```

---

### From OpenBLAS:
1. **Shape-specific kernels:** Different kernel for each M/N/K range
2. **Hand-tuned assembly:** When intrinsics aren't enough
3. **Prefetching:** Explicit software prefetch in loops
4. **Edge cases:** Handling small/odd matrix sizes

**Apply to C-Kernel-Engine:**
```c
// Dispatch based on shape (from OpenBLAS)
if (M <= 16) {
    gemm_small_m_kernel(A, B, C, M, N, K);  // Optimized for decode
} else if (K >= 64 && K <= 128) {
    gemm_attention_kernel(A, B, C, M, N, K);  // Attention-specific
} else {
    gemm_general_kernel(A, B, C, M, N, K);  // Default
}
```

---

## Detailed Study Plan

### Phase 1: Understand BLIS Microkernel (Week 1-2)

**Files to Study:**
1. `blis/kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c`
   - How does 6×8 register blocking work?
   - How do they handle edge cases (M/N not divisible by 6/8)?

2. `blis/frame/3/bli_gemm.c`
   - How does the high-level driver call microkernels?
   - What's the blocking hierarchy?

**Experiment:**
- Extract the microkernel, run standalone
- Measure performance vs. our current kernel
- Document register usage

---

### Phase 2: Study oneDNN BRGEMM (Week 3-4)

**Files to Study:**
1. `oneDNN/src/cpu/x64/brgemm/brgemm_types.hpp`
   - What are the batch types (strided, offset, addr)?
   - How is memory layout specified?

2. `oneDNN/src/cpu/x64/brgemm/jit_brgemm_kernel.cpp`
   - How does JIT code generation work?
   - What optimizations are applied?

3. `oneDNN/examples/` (if available)
   - How do users call BRGEMM?

**Experiment:**
- Build oneDNN
- Run matmul benchmark on LLM shapes
- Compare performance to our implementation
- Profile with perf to see where time is spent

---

### Phase 3: Analyze OpenBLAS Dispatch (Week 5-6)

**Files to Study:**
1. `OpenBLAS/kernel/x86_64/dgemm_kernel_16x2_haswell.S`
   - Line-by-line assembly analysis
   - How do they use prefetch?
   - Register allocation strategy

2. `OpenBLAS/driver/level3/gemm.c`
   - How does shape dispatch work?
   - What are the cutoff points for small-M vs. large-M?

**Experiment:**
- Test OpenBLAS on range of M values (1, 2, 4, 8, ..., 1024)
- Plot GFLOPS vs. M
- Identify where kernel changes happen
- Compare to our single kernel

---

## Profiling Comparison Template

Use this template to compare libraries:

```markdown
## Performance: M=512, N=768, K=768 (Typical LLM Linear Layer)

| Library | GFLOPS | IPC | L1 Miss % | L3 Miss % | Time (ms) |
|---------|--------|-----|-----------|-----------|-----------|
| BLIS | | | | | |
| oneDNN | | | | | |
| OpenBLAS | | | | | |
| C-Kernel-Engine | | | | | |

### Analysis
- Winner: [library name]
- Why: [explain based on profiling data]
- Our gap: [X%]
- Key difference: [what they do that we don't]

---

## Performance: M=1, N=768, K=768 (Autoregressive Decode - Critical!)

| Library | GFLOPS | IPC | L1 Miss % | L3 Miss % | Time (μs) |
|---------|--------|-----|-----------|-----------|-----------|
| BLIS | | | | | |
| oneDNN | | | | | |
| OpenBLAS | | | | | |
| C-Kernel-Engine | | | | | |

### Analysis
- This is the MOST important shape for LLM inference!
- Small M (=1) requires different optimization strategy
- [Document findings]
```

---

## Summary: Learning from the Best

**BLIS teaches us:** Clean abstraction and framework design
**oneDNN teaches us:** AI-specific optimizations and modern hardware
**OpenBLAS teaches us:** Production pragmatism and shape handling

**Our strategy:** Take the best from each:
- BLIS-style modularity (CKMathBackend interface)
- oneDNN-style AI focus (BRGEMM, fusion)
- OpenBLAS-style pragmatism (shape dispatch, hand-tuned kernels)

**Next Steps:**
1. Profile all three libraries on LLM shapes
2. Document performance gaps
3. Prioritize optimizations based on impact
4. Iterate: implement → measure → analyze → repeat

---

**Last Updated:** 2025-11-23
**Status:** Reference guide for studying the three libraries
