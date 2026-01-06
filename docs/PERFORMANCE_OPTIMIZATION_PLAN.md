# Performance Optimization Plan: C-Kernel-Engine (Phase 1.5)

## Overview

As we push for parity with and superiority over `llama.cpp`, we shift from correctness testing to deep performance engineering. This plan outlines a rigorous, data-driven approach using industry-standard low-level profiling tools to optimize the core 10-20 kernels that drive inference.

**Philosophy:** "Measure, Measure, Measure." We do not guess where bottlenecks are; we observe them in hardware counters and assembly dumps.

---

## 1. The Tooling Stack

We will employ a tiered profiling strategy:

| Tier | Tool | Purpose | Key Metrics |
|------|------|---------|-------------|
| **1. Macro** | `perf` + FlameGraph | Hotspot identification | CPU time, Call stacks |
| **2. Micro** | Intel VTune Profiler | Deep micro-arch analysis | Front-end bound, Back-end bound, Retiring, Bad Speculation |
| **3. Assembly** | `objdump` / `godbolt` | Instruction Verification | Vectorization (AVX2/512), Register spills, Loop unrolling |
| **4. Memory** | `perf stat` / VTune | Cache & Bandwidth | L1/L2/LLC Misses, DRAM Bandwidth, Store-to-Load Forwarding |

---

## 2. Profiling Workflow

### Step 1: Establish Baseline (The "Litmus Test")
Before optimizing, we must record the current performance of a kernel against `llama.cpp`.

*   **Action:** Create a comparative benchmark (extending `benchmarks/perf_gemm_micro.c`) that runs both CK and GGML implementations of the same kernel side-by-side.
*   **Output:** `ck_vs_ggml_gflops.csv`

### Step 2: Static Analysis (Assembly Inspection)
We verify that the compiler is doing what we expect before running the code.

*   **Command:** `objdump -d -M intel -S build/libckernel_engine.so > kernel_dump.asm`
*   **Checklist:**
    *   Are we using YMM/ZMM registers (AVX2/AVX512)?
    *   Are FMA instructions present (`vfmadd...`)?
    *   Are there unnecessary register spills (push/pop inside inner loops)?
    *   Is the loop prologue/epilogue overhead significant?

### Step 3: Dynamic Analysis (The "Torture Test")
Run the kernel in a tight loop with `perf` to capture hardware events.

*   **Command:**
    ```bash
    perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./bench_kernel
    ```
*   **Interpretation:**
    *   **High IPC (> 2):** Good. Compute bound.
    *   **High L1 Misses:** Tiling/Blocking size is too large or strides are bad.
    *   **High LLC Misses:** We are memory bandwidth bound. Needs fusion or better prefetching.

### Step 4: Fusion Analysis
Identify "Memory Wall" victims.

*   **Pattern:** `Kernel A (Write DRAM)` -> `Kernel B (Read DRAM)`
*   **Goal:** Fuse `A + B` so intermediate data stays in registers/L1.
*   **Candidates:**
    *   `GEMV + Activation` (e.g., `MatMul + Silu`)
    *   `RMSNorm + Q/K/V Projection` (Potential)
    *   `RoPE + Cache Write`

---

## 3. Targeted Kernels (The "Vital Few")

We will focus deeply on these ~15 kernels:

### Compute Heavy (Gemm/Gemv)
*   `gemm_q4_k` (Bulk decode)
*   `gemv_q4_k` (Token generation - usually bandwidth bound)
*   `gemm_f16` / `gemm_bf16`

### Bandwidth/Cache Heavy (Activations)
*   `rmsnorm` (Reads all weights, minimal math)
*   `softmax` (Reduction heavy)
*   `rope` (Strided access patterns, tricky for cache)
*   `swiglu` (Element-wise, easily fused)

---

## 4. Optimization Tactics

1.  **Cache Blocking (Tiling):**
    *   Adjust `Mc`, `Nc`, `Kc` block sizes to fit exactly in L1 (32KB) and L2 (256KB-1MB).
    *   *Tool:* VTune "Memory Access" analysis.

2.  **Register Blocking:**
    *   Ensure the inner kernel computes a roughly 6x16 or 8x24 micro-tile to hide FMA latencies.
    *   *Tool:* Assembly inspection (look for independent accumulation chains).

3.  **Prefetching:**
    *   Explicit software prefetching (`__builtin_prefetch`) for irregular access (like indirect lookups in Quantization).
    *   *Tool:* `perf stat` (stalls on load).

4.  **Kernel Fusion (The Big Win):**
    *   **Objective:** Eliminate the "Write to DRAM" phase of intermediate tensors.
    *   **Implementation:** Write custom kernels that take `f(x)` and immediately apply `g(x)` before storing.

---

## 5. Automation

We will build a script `scripts/profile_kernel.py` that:
1.  Compiles the specific kernel test case.
2.  Runs `objdump` and extracts the inner loop assembly.
3.  Runs `perf stat` and grabs key counters.
4.  Compares IPC/Throughput against the `llama.cpp` baseline.
5.  Generates a mini-report.

## Next Steps

1.  **Setup VTune/Perf:** Ensure environment supports hardware counters (may need sudo or `sysctl` tweaks).
2.  **Benchmark Suite:** Standardize `benchmarks/` to cover all 10-20 kernels, not just GEMM.
3.  **Fusion Pilot:** Attempt a manual fusion of `RMSNorm + Q_Proj` to measure DRAM traffic reduction.
