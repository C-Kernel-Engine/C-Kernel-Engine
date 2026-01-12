# Profiling and Performance Optimization Plan

## System Profile
- **CPU**: Intel i7-3630QM (Ivy Bridge, 3rd Gen, 2012)
- **Cores**: 4 physical, 8 threads (HT)
- **SIMD**: AVX only (**NO AVX2, NO FMA**)
- **Cache**: L1d 32KB/core, L2 256KB/core, L3 6MB shared
- **Memory**: DDR3 (likely 12.8 GB/s peak)

## Critical Finding: SIMD Gap

| Feature | Your CPU | Modern (AVX2) | llama.cpp Optimized |
|---------|----------|---------------|---------------------|
| AVX | Yes | Yes | Uses |
| AVX2 | **NO** | Yes | Uses heavily |
| FMA | **NO** | Yes | Critical for GEMV |
| AVX-512 | **NO** | Some | Optional |

This is **NOT a 30x code quality gap** - much of the difference is hardware limitations.
llama.cpp on AVX-only is also significantly slower than AVX2+FMA.

---

## Part 1: Profiling Setup

### Step 1: Build Standalone Binary for Profiling

Create a standalone `ck-profile` binary that doesn't require Python:

```bash
# Create profiling build directory
mkdir -p build/profile

# Build standalone profiling binary
gcc -O3 -g -fno-omit-frame-pointer -mavx \
    -DPROFILE_BUILD \
    tools/ck_profile.c \
    ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.c \
    src/*.c src/kernels/*.c \
    -I include \
    -o build/profile/ck-profile \
    -lm -fopenmp
```

### Step 2: Create Profiling Wrapper

File: `tools/ck_profile.c`

```c
// See generated file below
```

### Step 3: Run perf Profiling

```bash
# Record profile (100 tokens)
perf record -g -F 999 ./build/profile/ck-profile \
    ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump \
    100

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# Or simpler perf report
perf report --stdio --sort=dso,symbol
```

### Step 4: Install Flamegraph Tools

```bash
# Clone flamegraph tools
git clone https://github.com/brendangregg/FlameGraph.git ~/FlameGraph
export PATH="$HOME/FlameGraph:$PATH"

# Or use perf's built-in
perf report --hierarchy
```

---

## Part 2: Performance Gap Analysis

### Expected Breakdown (estimated)

| Component | % Time | Optimization Opportunity |
|-----------|--------|-------------------------|
| Q4_K GEMV | ~60-70% | SIMD, cache blocking |
| Attention | ~15-20% | Fused softmax, better memory |
| RMSNorm | ~5-8% | SIMD, loop fusion |
| Memory copy/init | ~5-10% | Eliminate redundant ops |
| Other | ~5% | - |

### Why llama.cpp is Faster

1. **GEMV Kernels** (biggest impact):
   - Hand-tuned intrinsics for Q4_K dequant + multiply + accumulate
   - Explicit prefetching (`_mm_prefetch`)
   - Cache-aware blocking (process 32 floats at a time)
   - On AVX-only: uses `_mm256_dp_ps` tricks

2. **Memory Layout**:
   - Quantized weights stored in SIMD-friendly format
   - Scale factors interleaved with data
   - Minimizes cache misses during dequant

3. **Threading**:
   - OpenMP parallel GEMV with good load balancing
   - Avoids false sharing

4. **Attention**:
   - Online softmax (no separate max pass)
   - Better blocking for KV cache access

---

## Part 3: Optimization Roadmap

### Phase 1: Quick Wins (1-2 days, expect 2-3x speedup)

1. **Enable OpenMP in GEMV kernels**
   ```c
   #pragma omp parallel for reduction(+:sum)
   for (int i = 0; i < n; i += 32) { ... }
   ```

2. **Add prefetching to Q4_K dequant**
   ```c
   _mm_prefetch(src + 64, _MM_HINT_T0);
   ```

3. **Fuse RMSNorm epilog into attention input**

### Phase 2: Kernel Rewrite (3-5 days, expect 3-5x speedup)

1. **Rewrite `gemv_q4_k` with explicit AVX**:
   - Process 8 floats per iteration (AVX)
   - Inline dequantization
   - Accumulate in registers, reduce at end

2. **Optimize attention for decode**:
   - Fused QK^T + softmax + V matmul
   - Better cache blocking for KV

3. **Memory pooling**:
   - Reuse scratch buffers across layers
   - Align all buffers to 32 bytes

### Phase 3: Architecture Changes (1-2 weeks, expect 5-10x speedup)

1. **Adopt llama.cpp Q4_K format exactly**
   - Match block layout for direct comparison
   - Reuse their dequant intrinsics

2. **Flash Attention style decode**
   - Online softmax
   - Tile-based processing

3. **Speculative decoding infrastructure**

---

## Part 4: Immediate Action Items

### Create Profiling Binary

```bash
# Step 1: Create the profiling tool
cat > tools/ck_profile.c << 'PROFILE_EOF'
// ... see generated code ...
PROFILE_EOF

# Step 2: Build with symbols
make clean && make CFLAGS="-O3 -g -fno-omit-frame-pointer -mavx"

# Step 3: Profile
perf record -g ./build/profile/ck-profile weights.bump 50
perf report
```

### Benchmark Baseline

```bash
# Time 100 token generation
time python scripts/v6/ck_run_v6.py run \
    hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
    --prompt "Hello" --max-tokens 100

# Compare with llama.cpp
./llama-cli -m qwen2-0_5b-instruct-q4_k_m.gguf -p "Hello" -n 100
```

---

## Part 5: Expected Results

### Current State (estimated)
- CK-Engine: ~5-10 tok/s
- llama.cpp: ~30-50 tok/s on same CPU
- Gap: ~5-10x (not 30x - verify this)

### After Phase 1 (OpenMP + prefetch)
- CK-Engine: ~15-25 tok/s
- Gap: ~2-3x

### After Phase 2 (kernel rewrite)
- CK-Engine: ~25-40 tok/s
- Gap: ~1.2-1.5x

### After Phase 3 (architecture)
- CK-Engine: ~30-50 tok/s
- Gap: ~1x (parity)

---

## Notes on AVX-only Optimization

Since you don't have AVX2/FMA, focus on:

1. **Use `_mm256_dp_ps`** for dot products (AVX has this)
2. **Avoid FMA patterns** - use separate mul+add
3. **256-bit operations** are still 2x wider than SSE
4. **Memory bandwidth** is often the bottleneck on older CPUs

llama.cpp's `ggml_vec_dot_q4_K_q8_K` has AVX-only paths you can reference.
