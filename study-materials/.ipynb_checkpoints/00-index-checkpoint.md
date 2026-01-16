# Study Materials Index

This directory contains in-depth study materials for understanding high-performance GEMM optimization for AI/LLM workloads, based on analysis of industry-leading open-source libraries.

## Libraries Under Study

We are studying three world-class open-source GEMM libraries located in `/home/antshiv/Workspace/3rd-Party/MathLibrary/`:

### 1. **BLIS** (BSD-3-Clause License)
- **Location:** `3rd-Party/MathLibrary/blis/`
- **Repository:** https://github.com/flame/blis
- **Focus:** Framework for instantiating BLAS-like libraries
- **Key Innovation:** Clean separation of microkernel from framework
- **Why Study:** Best-in-class API design, portable microkernel abstraction
- **Award:** 2023 James H. Wilkinson Prize for Numerical Software
- **What We Learn:** How to design modular, retargetable GEMM kernels
- **Key Files:**
  - `kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c` - 6×8 microkernel
  - `frame/` - Framework code separating policy from mechanism
  - `ref_kernels/` - Reference implementations for validation

### 2. **oneDNN** (Apache 2.0 License)
- **Location:** `3rd-Party/MathLibrary/oneDNN/`
- **Repository:** https://github.com/oneapi-src/oneDNN
- **Focus:** Deep Neural Network primitives (Intel)
- **Key Innovation:** BRGEMM (Batched-Reduced GEMM) for AI workloads
- **Why Study:** State-of-the-art AI-specific optimizations, AMX support
- **What We Learn:** How to optimize for transformer/CNN shapes specifically
- **Key Files:**
  - `src/cpu/x64/brgemm/brgemm.hpp` - BRGEMM API
  - `src/cpu/x64/brgemm/brgemm_types.hpp` - Batch types and layout
  - `src/cpu/x64/brgemm/jit_brgemm_kernel.cpp` - JIT code generation
  - `src/cpu/x64/brgemm/jit_brgemm_amx_uker.cpp` - AMX tile operations
  - `src/cpu/matmul/` - High-level matmul implementations

### 3. **OpenBLAS** (BSD-3-Clause License)
- **Location:** `3rd-Party/MathLibrary/OpenBLAS/`
- **Repository:** https://github.com/OpenMathLib/OpenBLAS
- **Focus:** Optimized BLAS implementation (continuation of GotoBLAS)
- **Key Innovation:** Kazushige Goto's blocking strategy, hand-tuned assembly
- **Why Study:** Production-proven, wide hardware support, shape-specific kernels
- **What We Learn:** How to handle diverse CPU architectures and edge cases
- **Key Files:**
  - `kernel/x86_64/dgemm_kernel_16x2_haswell.S` - 16×2 assembly kernel
  - `kernel/x86_64/` - Architecture-specific kernels
  - `common_*.h` - Platform abstraction headers
  - `driver/level3/` - High-level GEMM driver code

## Our Current Implementation vs. Reference Libraries

### C-Kernel-Engine (Our Implementation)
**Location:** `/home/antshiv/Workspace/C-Kernel-Engine/`

| Aspect | Our Implementation | BLIS | oneDNN | OpenBLAS |
|--------|-------------------|------|--------|----------|
| **Block Size** | 64×64×64 | 6×8 microkernel | Dynamic (JIT) | 16×2, 8×4, 4×12 |
| **SIMD** | AVX-512 intrinsics | AVX-512 + assembly | AVX-512 + AMX | AVX-512 assembly |
| **Batching** | None (single GEMM) | None | BRGEMM (strided/offset) | None |
| **Bias Fusion** | Yes (optional) | No | Yes (post-ops) | No |
| **Threading** | OpenMP (#pragma) | Custom | TBB/OpenMP | OpenMP |
| **Shape Dispatch** | Single kernel | Generic framework | Shape-specific | Shape-specific |
| **Prefetching** | None | Configurable | Aggressive | Manual in asm |
| **API Design** | `CKMathBackend` struct | Object-based | Primitive-based | BLAS-compatible |

### Key Differences to Study

1. **Block Size Choice**
   - **Ours:** 64×64 - matches cache line
   - **Theirs:** 6×8, 8×6, 16×2 - matches register file
   - **Study Goal:** Understand register blocking vs. cache blocking hierarchy

2. **Kernel Specialization**
   - **Ours:** Single kernel for all shapes
   - **Theirs:** Different kernels for different (M,N,K) ranges
   - **Study Goal:** When to use small-M, large-M, attention-specific kernels

3. **Memory Access Patterns**
   - **Ours:** Simple blocked iteration
   - **Theirs:** Panel-panel, software prefetch, non-temporal stores
   - **Study Goal:** Cache-conscious algorithm design

4. **Batching Support**
   - **Ours:** Call GEMM N times for N operations
   - **oneDNN:** Single BRGEMM call with stride
   - **Study Goal:** Multi-head attention optimization

## Core Concepts Documentation

1. **[Cache Hierarchy and Blocking Strategy](01-cache-hierarchy-blocking.md)**
   - Why block sizes are 6×8, 8×6, 16×2 instead of 64×64
   - Cache line alignment vs. register blocking
   - L1/L2/L3 blocking strategies
   - Three-level blocking hierarchy
   - Computational complexity analysis

2. **[BRGEMM: Batched-Reduced GEMM for AI](02-brgemm-architecture.md)**
   - Intel oneDNN's innovation for transformers
   - Strided batching for multi-head attention
   - Memory access patterns
   - Performance comparison

3. **[Microkernel Design Principles](03-microkernel-design.md)**
   - BLIS framework approach
   - Register blocking and SIMD utilization
   - Panel-panel vs. panel-block multiplication
   - Assembly vs. intrinsics trade-offs

4. **[AI-Specific Shape Optimizations](04-ai-shape-optimizations.md)**
   - LLM kernel shape families (from our 01-llm-kernel-shapes.md)
   - Small-M vs. Large-M kernels
   - Attention-specific optimizations
   - MLP layer optimizations

5. **[AMX and Future Hardware](05-amx-future-hardware.md)**
   - Intel Advanced Matrix Extensions (Sapphire Rapids+)
   - Tile-based matrix multiplication
   - BF16/INT8 acceleration
   - Migration path from AVX-512

6. **[Prefetching and Memory Optimization](06-prefetching-memory.md)**
   - Software prefetching strategies
   - Non-temporal loads for streaming data
   - NUMA awareness
   - Memory advice hints

7. **[Fused Operations and Post-Ops](07-fused-operations.md)**
   - Bias fusion inside GEMM
   - Activation function fusion (GELU, ReLU, SiLU)
   - Residual connection fusion
   - Performance impact analysis

8. **[Computational Complexity Analysis](08-computational-complexity.md)**
   - FLOPs vs. memory bandwidth
   - Roofline model for GEMM
   - Arithmetic intensity
   - Hardware limits and bottlenecks

## Profiling and Performance Analysis

### Tools and Methodology

**[Performance Profiling Guide](profiling/00-profiling-methodology.md)**

We will use the following tools to profile and compare our implementation:

1. **Intel VTune Profiler**
   - Microarchitecture analysis
   - Cache miss rates
   - TLB miss rates
   - Pipeline stalls
   - **How to profile:** [profiling/01-vtune-analysis.md](profiling/01-vtune-analysis.md)

2. **perf (Linux perf_events)**
   - Hardware counters
   - CPU cycles, instructions
   - Cache references/misses
   - Branch prediction
   - **How to profile:** [profiling/02-perf-analysis.md](profiling/02-perf-analysis.md)

3. **likwid-perfctr**
   - Performance counter analysis
   - Memory bandwidth measurement
   - Roofline model generation
   - **How to profile:** [profiling/03-likwid-analysis.md](profiling/03-likwid-analysis.md)

4. **Custom Microbenchmarks**
   - Cycle-accurate timing
   - GFLOPS measurement
   - Memory bandwidth validation
   - **How to implement:** [profiling/04-custom-benchmarks.md](profiling/04-custom-benchmarks.md)

### Profiling Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Baseline Measurement                            │
│  - Profile our current implementation                   │
│  - Measure GFLOPS, cache misses, bandwidth              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: Reference Library Profiling                     │
│  - Profile BLIS, oneDNN, OpenBLAS on same shapes        │
│  - Compare metrics side-by-side                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Gap Analysis                                    │
│  - Identify performance gaps (e.g., 2x slower on M=1)   │
│  - Identify bottlenecks (cache misses, bandwidth)       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: Optimization Implementation                     │
│  - Apply learnings from reference libraries             │
│  - Implement shape-specific kernels, prefetch, etc.     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: Re-measure and Iterate                          │
│  - Profile optimized implementation                     │
│  - Compare against Step 1 baseline                      │
│  - Repeat until performance goals met                   │
└─────────────────────────────────────────────────────────┘
```

### Performance Tracking

**[Performance History Log](profiling/performance-history.md)**
- Track GFLOPS over time as optimizations are applied
- Compare against reference libraries
- Document what worked and what didn't

**Metrics to Track:**
- **GFLOPS:** Actual floating-point operations per second
- **% Peak:** Percentage of theoretical hardware peak
- **Cache Miss Rate:** L1/L2/L3 cache misses per 1000 instructions
- **Memory Bandwidth:** GB/s memory throughput
- **Arithmetic Intensity:** FLOPs per byte of memory traffic
- **Cycle Breakdown:** % cycles in compute vs. memory vs. stalls

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- **[Priority 1: Shape-Specific Kernels](implementation/01-shape-specific-kernels.md)**
  - Small-M kernel (M ≤ 16)
  - Large-M kernel (M ≥ 64)
  - Attention kernel (K ∈ [64, 128])

### Phase 2: Batching (Weeks 3-4)
- **[Priority 2: BRGEMM Strided Batching](implementation/02-brgemm-batching.md)**
  - Multi-head attention optimization
  - Strided batch interface

### Phase 3: Fusion (Weeks 5-6)
- **[Priority 3: Fused Operations](implementation/03-fused-operations.md)**
  - Bias + GELU fusion
  - Residual connection fusion

### Phase 4: Advanced (Weeks 7-8)
- **[Priority 4: Prefetching and Memory](implementation/04-prefetch-memory.md)**
  - Software prefetching
  - Non-temporal stores
  - NUMA awareness

## Validation Strategy

**[Validation and Testing](validation/00-validation-strategy.md)**

1. **Numerical Correctness**
   - Compare against naive reference implementation
   - Maximum difference threshold: 1e-5 (from C-Transformer experience)
   - Test all LLM shapes from `01-llm-kernel-shapes.md`

2. **Performance Validation**
   - Benchmark against BLIS, oneDNN, OpenBLAS
   - Must be within 80% of best library for each shape
   - Document any shape where we're slower and why

3. **Continuous Integration**
   - Automated tests on every commit
   - Performance regression detection
   - Cross-platform validation (x86-64, ARM)

## Study Order Recommendation

**For First-Time Readers:**

1. **Start Here:** [01-cache-hierarchy-blocking.md](01-cache-hierarchy-blocking.md)
   - Answers "Why 6×8 and not 64×64?"
   - Fundamental to everything else

2. **Understand the Math:** [08-computational-complexity.md](08-computational-complexity.md)
   - Hardware limits and bottlenecks
   - Roofline model

3. **Learn the Structure:** [03-microkernel-design.md](03-microkernel-design.md)
   - How BLIS separates concerns
   - Microkernel design patterns

4. **Apply to AI:** [04-ai-shape-optimizations.md](04-ai-shape-optimizations.md)
   - LLM-specific optimizations
   - Our use case

5. **Modern Techniques:** [02-brgemm-architecture.md](02-brgemm-architecture.md)
   - How oneDNN optimizes transformers
   - Batching strategies

6. **Profile and Iterate:** [profiling/00-profiling-methodology.md](profiling/00-profiling-methodology.md)
   - Measure, analyze, optimize
   - Repeat

## Contributing to Study Materials

As we learn and implement, we'll continuously update these materials with:
- **Experimental results** from our profiling
- **Code snippets** showing what worked and what didn't
- **Performance comparisons** before/after optimizations
- **Lessons learned** from debugging and tuning

Each document should be a living resource that grows with our understanding.

## References and External Resources

### Academic Papers
- BLIS: "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" (ACM TOMS)
- GotoBLAS: "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn)
- Roofline Model: "Roofline: An Insightful Visual Performance Model" (Williams et al.)

### Online Resources
- BLIS Documentation: https://github.com/flame/blis/wiki
- oneDNN Performance Guide: https://oneapi-src.github.io/oneDNN/
- Intel Optimization Manual: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html

### Video Lectures
- "The Science of High-Performance Computing" (SHPC Group, UT Austin)
- "BLAS-like Library Instantiation Software" (BLIS tutorials)

---

**Last Updated:** 2025-11-23
**Status:** Initial structure, documents to be written
**Maintainer:** Anthony Shivakumar
