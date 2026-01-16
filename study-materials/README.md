# C-Kernel-Engine Study Materials

This directory contains comprehensive study materials for understanding and implementing high-performance GEMM kernels optimized for AI/LLM workloads.

## Quick Start

**New to GEMM optimization?** Start here:
1. Read [00-index.md](00-index.md) for the overview
2. Read [01-cache-hierarchy-blocking.md](01-cache-hierarchy-blocking.md) to understand "Why 6×8 and not 64×64?"
3. Read [profiling/00-profiling-methodology.md](profiling/00-profiling-methodology.md) to learn how to measure performance

## What We're Studying

We're analyzing three world-class open-source GEMM libraries:

### 📚 Reference Libraries (in `/home/antshiv/Workspace/3rd-Party/MathLibrary/`)

1. **BLIS** - Framework-based BLAS with clean microkernel abstraction
2. **oneDNN** - Intel's deep learning primitives with BRGEMM for transformers
3. **OpenBLAS** - Production BLAS with hand-tuned assembly kernels

### 🎯 Our Implementation (`/home/antshiv/Workspace/C-Kernel-Engine/`)

Current status: Basic AVX-512 blocked GEMM (64×64×64 blocks)
Goal: Match or exceed reference libraries for AI-specific shapes

## Core Questions We Answer

### 1. **Why are microkernel sizes 6×8 or 16×2 instead of 64×64?**
→ See [01-cache-hierarchy-blocking.md](01-cache-hierarchy-blocking.md)

**TL;DR:** Cache lines are 64 *bytes* (16 floats), but microkernels are sized for the *register file* (32 AVX-512 registers), not cache lines. We need THREE levels of blocking: register (6×8), L1/L2 cache (384×96), and L3 cache (4096×4096).

### 2. **How do reference libraries optimize for AI workloads?**
→ See [02-brgemm-architecture.md](02-brgemm-architecture.md) and [04-ai-shape-optimizations.md](04-ai-shape-optimizations.md)

**TL;DR:** They use batched GEMM (BRGEMM) for multi-head attention, shape-specific kernels for small-M (autoregressive) vs. large-M (prompt), and fused operations (bias + activation).

### 3. **What's the computational complexity and hardware limits?**
→ See [08-computational-complexity.md](08-computational-complexity.md)

**TL;DR:** GEMM is compute-bound with good blocking (AI = 200-600 FLOPs/byte). Hardware peak is ~192 GFLOPS/core on AVX-512, but realistic is 80-120 GFLOPS/core due to memory and pipeline constraints.

### 4. **How do we profile and compare implementations?**
→ See [profiling/00-profiling-methodology.md](profiling/00-profiling-methodology.md)

**TL;DR:** Use `perf stat` for hardware counters, track GFLOPS/IPC/cache-misses, compare against reference libraries on same shapes, identify bottlenecks, optimize, repeat.

## Directory Structure

```
study-materials/
├── 00-index.md                          # Full index of all materials
├── README.md                            # This file (quick start)
│
├── Core Concepts (to be written)
├── 01-cache-hierarchy-blocking.md       # ✅ Why 6×8 not 64×64
├── 02-brgemm-architecture.md            # BRGEMM for transformers
├── 03-microkernel-design.md             # BLIS microkernel patterns
├── 04-ai-shape-optimizations.md         # LLM-specific optimizations
├── 05-amx-future-hardware.md            # Intel AMX (Sapphire Rapids)
├── 06-prefetching-memory.md             # Memory optimization
├── 07-fused-operations.md               # Operator fusion
├── 08-computational-complexity.md       # FLOPs, bandwidth, roofline
│
├── profiling/
│   ├── 00-profiling-methodology.md      # ✅ How to measure performance
│   ├── 01-vtune-analysis.md             # Intel VTune usage
│   ├── 02-perf-analysis.md              # Linux perf usage
│   ├── 03-likwid-analysis.md            # likwid-perfctr usage
│   ├── 04-custom-benchmarks.md          # Custom microbenchmarks
│   └── performance-history.md           # Track improvements over time
│
├── implementation/
│   ├── 01-shape-specific-kernels.md     # Small-M, large-M, attention
│   ├── 02-brgemm-batching.md            # Strided batching
│   ├── 03-fused-operations.md           # Bias+GELU fusion
│   └── 04-prefetch-memory.md            # Prefetching strategy
│
├── references/
│   ├── blis-analysis.md                 # Deep dive into BLIS
│   ├── onednn-brgemm.md                 # oneDNN BRGEMM analysis
│   └── openblas-kernels.md              # OpenBLAS kernel study
│
└── validation/
    └── 00-validation-strategy.md        # Testing and correctness
```

## Key Comparisons: Our Implementation vs. References

| Aspect | C-Kernel-Engine | BLIS | oneDNN | OpenBLAS |
|--------|-----------------|------|--------|----------|
| **Block Size** | 64×64×64 (single) | 6×8 micro + multi-level | Dynamic | 16×2, 8×4, etc. |
| **SIMD** | AVX-512 intrinsics | AVX-512 asm | AVX-512/AMX | AVX-512 asm |
| **Batching** | None | None | BRGEMM ✓ | None |
| **Shape Dispatch** | Single kernel | Framework | Shape-specific ✓ | Shape-specific ✓ |
| **AI Focus** | Learning | Generic BLAS | High ✓ | Generic BLAS |

**Gap to close:** 2-3x performance difference on AI workloads

## Performance Targets

Based on AVX-512 CPU (8 cores, ~3.0 GHz):

| Shape | Theoretical Peak | Target (80%) | Current | Status |
|-------|------------------|--------------|---------|--------|
| **M=512, N=768, K=768** | ~1536 GFLOPS | ~1200 | ~250 | 🔴 20% |
| **M=1, N=768, K=768** | Critical path | Fast | ? | ⚪ TBD |
| **M=512, N=512, K=64** | Attention | Fast | ? | ⚪ TBD |

## Study Plan

### Week 1-2: Foundation
- [ ] Read 01-cache-hierarchy-blocking.md
- [ ] Read 08-computational-complexity.md
- [ ] Set up profiling (perf, benchmarks)
- [ ] Measure baseline performance
- [ ] Compare against BLIS/oneDNN/OpenBLAS

### Week 3-4: Microkernel
- [ ] Read 03-microkernel-design.md
- [ ] Study BLIS 6×8 kernel
- [ ] Study OpenBLAS 16×2 kernel
- [ ] Implement 16×2 microkernel
- [ ] Profile and validate

### Week 5-6: AI Optimizations
- [ ] Read 04-ai-shape-optimizations.md
- [ ] Read 02-brgemm-architecture.md
- [ ] Implement shape-specific dispatch
- [ ] Implement BRGEMM strided batching
- [ ] Profile on LLM shapes

### Week 7-8: Advanced
- [ ] Read 06-prefetching-memory.md
- [ ] Read 07-fused-operations.md
- [ ] Add prefetching
- [ ] Add fused bias+activation
- [ ] Final profiling and tuning

## Contributing

As we learn and experiment, we should:
1. **Document findings** in the relevant markdown files
2. **Track performance** in `profiling/performance-history.md`
3. **Add examples** from actual code (ours and reference libraries)
4. **Note what worked** and what didn't

This is a living resource that grows with our understanding.

## External Links

### Reference Library Repositories
- BLIS: https://github.com/flame/blis
- oneDNN: https://github.com/oneapi-src/oneDNN
- OpenBLAS: https://github.com/OpenMathLib/OpenBLAS

### Papers and Resources
- BLIS Paper: ["BLIS: A Framework for Rapidly Instantiating BLAS Functionality"](http://dl.acm.org/authorize?N91172) (ACM TOMS)
- GotoBLAS: ["Anatomy of High-Performance Matrix Multiplication"](http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- Roofline Model: ["Roofline: An Insightful Visual Performance Model"](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
- Intel Optimization Manual: [Intel® 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)

### Educational Resources
- BLIS Wiki: https://github.com/flame/blis/wiki
- oneDNN Developer Guide: https://oneapi-src.github.io/oneDNN/
- SHPC Group (UT Austin): http://shpc.ices.utexas.edu/

---

**Status:** Initial structure created 2025-11-23
**Next:** Write remaining concept documents as we study the reference libraries
