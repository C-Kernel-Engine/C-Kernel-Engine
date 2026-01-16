# Profiling Methodology for GEMM Optimization

## Overview

Performance optimization is a **data-driven** process. We measure, analyze bottlenecks, apply optimizations, and measure again. This document describes our methodology for profiling C-Kernel-Engine GEMM kernels and comparing them against BLIS, oneDNN, and OpenBLAS.

---

## Tools We Use

### 1. **perf** (Linux perf_events)
- **Purpose:** Hardware performance counter analysis
- **Installation:** Usually pre-installed on Linux
- **What it measures:**
  - CPU cycles, instructions
  - Cache references, cache misses (L1/L2/L3)
  - TLB misses
  - Branch mispredictions
  - Stalled cycles (frontend, backend)

### 2. **Intel VTune Profiler** (Optional, if available)
- **Purpose:** Microarchitecture-level analysis
- **Installation:** Free download from Intel
- **What it measures:**
  - Port utilization (which execution units are busy)
  - Pipeline bottlenecks
  - Memory bandwidth saturation
  - Vectorization efficiency

### 3. **likwid-perfctr** (Optional)
- **Purpose:** Performance counter wrapper with better output
- **Installation:** `git clone https://github.com/RRZE-HPC/likwid`
- **What it measures:**
  - Same as perf, but with cleaner output
  - Built-in metrics (e.g., "FLOPS_DP", "MEM")
  - Roofline model data

### 4. **Custom Cycle-Accurate Timing**
- **Purpose:** Microbenchmarking specific code sections
- **Implementation:** Use `RDTSC` instruction or `clock_gettime()`

---

## Profiling Workflow

### Step 1: Baseline Measurement

**Goal:** Establish current performance of C-Kernel-Engine

```bash
# Navigate to C-Kernel-Engine directory
cd /home/antshiv/Workspace/C-Kernel-Engine

# Build with optimization and debug symbols
gcc -O3 -march=native -mavx512f -g -fopenmp \
    src/backend_native.c \
    -o test_gemm \
    -I include/

# Create test harness
cat > test/benchmark_gemm.c << 'EOF'
#include "ckernel_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int M = 512, N = 768, K = 768;

    float *A = aligned_alloc(64, M * K * sizeof(float));
    float *B = aligned_alloc(64, N * K * sizeof(float));
    float *C = aligned_alloc(64, M * N * sizeof(float));
    float *bias = aligned_alloc(64, N * sizeof(float));

    // Initialize with random data
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N * K; i++) B[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N; i++) bias[i] = (float)rand() / RAND_MAX;

    // Warmup
    gemm_blocked_serial(A, B, bias, C, M, N, K);

    // Benchmark
    int iterations = 100;
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        gemm_blocked_serial(A, B, bias, C, M, N, K);
    }
    double end = get_time();

    double elapsed = (end - start) / iterations;
    double gflops = (2.0 * M * N * K) / elapsed / 1e9;

    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("Time: %.6f ms\n", elapsed * 1000);
    printf("GFLOPS: %.2f\n", gflops);

    free(A); free(B); free(C); free(bias);
    return 0;
}
EOF

gcc -O3 -march=native -mavx512f -fopenmp \
    test/benchmark_gemm.c src/backend_native.c \
    -o benchmark_gemm -I include/

# Run baseline
./benchmark_gemm
```

**Expected Output:**
```
M=512, N=768, K=768
Time: 2.345 ms
GFLOPS: 253.42
```

Record this in `profiling/performance-history.md`.

---

### Step 2: Detailed Performance Counter Analysis

```bash
# Profile with perf
perf stat -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,\
branches,branch-misses \
./benchmark_gemm

# Example output:
#  Performance counter stats for './benchmark_gemm':
#
#      5,234,567,890  cycles
#      3,456,789,012  instructions              #  0.66  insn per cycle
#        234,567,890  cache-references
#         12,345,678  cache-misses              #  5.27% of all cache refs
#        890,123,456  L1-dcache-loads
#         45,678,901  L1-dcache-load-misses     #  5.13% of all L1-dcache accesses
#         23,456,789  LLC-loads
#          1,234,567  LLC-load-misses           #  5.26% of all LL-cache accesses
```

**Key Metrics to Track:**

1. **IPC (Instructions Per Cycle):** Should be > 2.0 for well-optimized GEMM
   - Ours: 0.66 (low! indicates stalls)
   - Target: 2.0-3.0

2. **Cache Miss Rate:**
   - L1: Should be < 5%
   - L3: Should be < 1%

3. **Memory Bandwidth Utilization:**
   ```
   Bandwidth = (cache-misses × 64 bytes) / time
   ```

---

### Step 3: Compare Against Reference Libraries

#### Profile BLIS

```bash
cd /home/antshiv/Workspace/3rd-Party/MathLibrary/blis

# Build BLIS
./configure --enable-cblas auto
make -j8

# Create test for same shape
cat > test_blis.c << 'EOF'
#include "blis.h"
#include <time.h>
#include <stdio.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int M = 512, N = 768, K = 768;

    float *A = aligned_alloc(64, M * K * sizeof(float));
    float *B = aligned_alloc(64, K * N * sizeof(float));
    float *C = aligned_alloc(64, M * N * sizeof(float));

    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    // BLIS uses column-major, so transpose or use row-major API
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
              M, N, K, &alpha, A, 1, K, B, 1, N, &beta, C, 1, N);

    // Benchmark
    int iterations = 100;
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
                  M, N, K, &alpha, A, 1, K, B, 1, N, &beta, C, 1, N);
    }
    double end = get_time();

    double elapsed = (end - start) / iterations;
    double gflops = (2.0 * M * N * K) / elapsed / 1e9;

    printf("BLIS: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Time: %.6f ms\n", elapsed * 1000);
    printf("GFLOPS: %.2f\n", gflops);

    return 0;
}
EOF

gcc -O3 -march=native -fopenmp test_blis.c -o test_blis \
    -I include/blis -L lib -lblis -lm -lpthread

./test_blis
```

#### Profile OpenBLAS

```bash
cd /home/antshiv/Workspace/3rd-Party/MathLibrary/OpenBLAS

# Build OpenBLAS
make -j8

# Test (similar to BLIS, using cblas_sgemm)
```

#### Profile oneDNN

```bash
cd /home/antshiv/Workspace/3rd-Party/MathLibrary/oneDNN

# Build oneDNN
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Test using oneDNN matmul primitive
```

---

### Step 4: Side-by-Side Comparison

Create a comparison table:

| Library | M=512, N=768, K=768 | M=1, N=768, K=768 | M=512, N=512, K=64 |
|---------|---------------------|-------------------|---------------------|
| **C-Kernel-Engine** | 253 GFLOPS | ? | ? |
| **BLIS** | ? | ? | ? |
| **oneDNN** | ? | ? | ? |
| **OpenBLAS** | ? | ? | ? |

Fill in the "?" by running benchmarks for each shape.

---

### Step 5: Identify Bottlenecks

Use `perf record` and `perf report` for hotspot analysis:

```bash
# Record performance data
perf record -e cycles:pp -g ./benchmark_gemm

# Analyze hotspots
perf report

# Example output:
# 45.23%  benchmark_gemm  [.] gemm_blocked_serial
# 12.34%  benchmark_gemm  [.] _mm512_loadu_ps
#  8.76%  benchmark_gemm  [.] _mm512_fmadd_ps
```

**Questions to Ask:**
1. Is the hotspot in the right place? (Should be in compute kernel)
2. Are we spending time in memory access? (loads/stores)
3. Are there unexpected function calls? (indicates missed inlining)

---

### Step 6: Roofline Model Analysis

The roofline model helps us understand if we're compute-bound or memory-bound.

```
Peak Performance (GFLOPS): P_peak = 192 GFLOPS/core (AVX-512)
Memory Bandwidth (GB/s): B_mem = 75 GB/s (DDR4-2400)

Ridge Point = P_peak / B_mem = 192 / 75 = 2.56 FLOPs/byte

If our arithmetic intensity > 2.56, we should be compute-bound.
If our arithmetic intensity < 2.56, we're memory-bound.

For GEMM with good blocking:
  AI = (2 * M * N * K) / (M*K + K*N + M*N) / 4 bytes
  For M=N=K=512:
    AI = (2 * 512^3) / (3 * 512^2) / 4 = 85 FLOPs/byte

  85 >> 2.56, so we should be compute-bound ✓
```

**If we're NOT hitting peak performance:**
- Check IPC (should be > 2.0)
- Check cache miss rate
- Check vectorization (should use zmm registers)

---

## Performance Tracking Template

Create `profiling/performance-history.md`:

```markdown
# Performance History

## Baseline (2025-11-23)

### Configuration
- CPU: Intel Xeon (AVX-512)
- Cores: 8
- Memory: DDR4-2400, 64 GB
- Compiler: GCC 11.3.0
- Flags: -O3 -march=native -mavx512f -fopenmp

### Results: M=512, N=768, K=768

| Implementation | GFLOPS | % Peak | Time (ms) | Cache Miss Rate | IPC |
|----------------|--------|--------|-----------|-----------------|-----|
| C-Kernel-Engine | 253.4 | 16.5% | 2.345 | 5.27% L1 | 0.66 |
| BLIS | - | - | - | - | - |
| oneDNN | - | - | - | - | - |
| OpenBLAS | - | - | - | - | - |

### Bottlenecks Identified
1. Low IPC (0.66) indicates stalls - likely memory stalls
2. L1 cache miss rate is acceptable (5.27%)
3. Not utilizing full register file (no microkernel)

### Next Steps
- Implement 16×2 microkernel
- Add three-level blocking
- Profile again

---

## Optimization 1: 16×2 Microkernel (2025-11-XX)

### Changes
- Added gemm_microkernel_16x2() function
- Modified blocking to use microkernel

### Results: M=512, N=768, K=768

| Implementation | GFLOPS | Δ from Baseline | IPC |
|----------------|--------|-----------------|-----|
| C-Kernel-Engine (new) | XXX | +XX% | X.XX |

### Analysis
- [Document what worked and what didn't]

---
```

---

## Profiling Checklist

Before calling an optimization "done", verify:

- [ ] **Correctness:** Numerical diff < 1e-5 vs. reference
- [ ] **Performance:** Benchmark on all LLM shapes from `01-llm-kernel-shapes.md`
- [ ] **Comparison:** Compare against BLIS/oneDNN/OpenBLAS for same shapes
- [ ] **Bottleneck Analysis:** Profile with perf to identify remaining issues
- [ ] **Documentation:** Update performance history with results
- [ ] **Regression Test:** Ensure previous shapes didn't get slower

---

## Advanced Profiling Techniques

### Memory Bandwidth Measurement

Use `likwid-perfctr` for accurate memory bandwidth:

```bash
likwid-perfctr -C 0 -g MEM ./benchmark_gemm

# Output shows:
# Memory bandwidth [MBytes/s]: 45234.56
# Memory data volume [GBytes]: 12.34
```

### Vectorization Analysis

Check if compiler vectorized your code:

```bash
gcc -O3 -march=native -mavx512f -fopt-info-vec-all \
    src/backend_native.c -c 2>&1 | grep "vectorized"

# Should see:
# src/backend_native.c:131:21: optimized: loop vectorized using 64 byte vectors
```

### Assembly Inspection

Look at generated assembly:

```bash
objdump -d -M intel benchmark_gemm | grep -A 50 "gemm_blocked_serial"

# Look for:
# - vfmadd instructions (AVX-512 FMA)
# - zmm register usage (zmm0-zmm31)
# - Prefetch instructions (prefetcht0, prefetchnta)
```

---

## Summary

**Profiling is iterative:**
1. Measure baseline
2. Identify bottleneck
3. Apply optimization
4. Measure again
5. Repeat

**Key metrics:**
- **GFLOPS:** Higher is better (target: 80%+ of reference libraries)
- **IPC:** Should be > 2.0 for well-optimized code
- **Cache Miss Rate:** L1 < 5%, L3 < 1%
- **Memory Bandwidth:** Should be close to hardware limit if memory-bound

**Next:** [01-vtune-analysis.md](01-vtune-analysis.md) for detailed VTune usage.
