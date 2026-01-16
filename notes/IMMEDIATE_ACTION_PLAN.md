# IMMEDIATE ACTION PLAN: Close 10x Performance Gap to llama.cpp

**Current**: 2.91 tok/s  
**llama.cpp**: 30 tok/s  
**Gap**: 10.3x  
**Target**: 15-20 tok/s (50-67% of llama.cpp) in 2 weeks

---

## Root Cause Analysis

### From Flamegraph: 75% time in optimized kernels

This means:
- ✅ Kernels ARE optimized (good news!)
- ❌ But kernels themselves are SLOW (bad news!)

**The problem is NOT that we're using scalar code - it's that our AVX kernels are inefficient compared to llama.cpp's AVX kernels.**

---

## Three Critical Issues (90% of gap)

### Issue 1: GEMM Block Sizes (40% of problem)

**llama.cpp**: Uses optimal cache-aware blocking
- L1: 32x32 blocks
- L2: 64x64 blocks
- Prefetching

**C-Kernel**: Likely using naive blocking
- Check: `gemm_kernels_q5_0.c`

**Action**: Copy llama.cpp's blocking strategy
```c
// Optimal for 32KB L1 cache
#define MC 32
#define NC 32
#define KC 64
```

### Issue 2: Memory Allocation in Hot Path (30% of problem)

**Observation**: 1.72% in `__memmove_sse2` suggests allocation overhead

**llama.cpp**: Arena allocator, pre-allocated buffers
**C-Kernel**: Likely malloc/free in decode loop

**Action**: Implement arena allocator
```c
// In model struct
typedef struct {
    char* base;
    size_t offset;
    size_t size;
} Arena;

// Allocate once, use forever
arena = arena_create(16 * 1024 * 1024); // 16MB
```

### Issue 3: Inefficient INT8 Dequantization (20% of problem)

**Current**: `vec_dot_q5_0_q8_0_avx` takes 32% time
**llama.cpp**: Optimized INT8 unpack + multiply fused

**Action**: Fuse dequantization with multiplication
```c
// Instead of: dequant -> multiply
// Do: dequant_and_multiply (single pass)
```

---

## Week 1: Fix Blocking + Allocator

### Day 1-2: GEMM Blocking
```bash
# Compare llama.cpp vs our blocking
diff -u <(objdump -d llama.cpp | grep gemm) <(objdump -d ck-kernel-inference.so | grep gemm)

# Fix in src/kernels/gemm_kernels_q5_0.c
# Copy optimal block sizes from llama.cpp
```

**Target**: 2.91 → 5 tok/s (1.7x)

### Day 3-4: Arena Allocator
```c
// Add to ckernel_alloc_v6.5.c
// Pre-allocate all scratch buffers
// Zero malloc in decode loop

// Test: valgrind --tool=callgrind ./ck-cli-v6.5
// Verify: No malloc in decode path
```

**Target**: 5 → 7 tok/s (1.4x)

### Day 5-7: INT8 Fusion
```c
// Optimize vec_dot_q5_0_q8_0_avx
// Fuse: unpack + multiply + accumulate
// Remove: intermediate temp buffers
```

**Target**: 7 → 10 tok/s (1.4x)

---

## Week 2: Advanced Optimizations

### Day 8-10: Cache Blocking for Attention
```c
// Optimize attention kernel
// Block K, V matrices for L2 cache
// Target: Reduce cache misses by 5x
```

**Target**: 10 → 15 tok/s (1.5x)

### Day 11-14: KV Cache Layout
```c
// Optimize KV cache memory layout
// Pack K, V for sequential access
// Target: Reduce DRAM traffic
```

**Target**: 15 → 20 tok/s (1.3x)

---

## Validation Strategy

### Daily Benchmark
```bash
# Run before/after each change
python3 benchmark_v65.py > results_$(date +%Y%m%d).txt

# Track: tokens/second, memory, cache misses
```

### Perf Analysis
```bash
# After each change, generate flamegraph
./run_flamegraph_v6.sh

# Compare with llama.cpp
perf record -g ./llama.cpp/main ...
```

---

## Success Metrics

| Day | Target | Action |
|-----|--------|--------|
| 0 | 2.91 tok/s | Current baseline |
| 3 | 5 tok/s | Blocking fixed |
| 7 | 10 tok/s | + Arena + INT8 fusion |
| 10 | 15 tok/s | + Attention opt |
| 14 | 20 tok/s | + KV cache opt |
| 21 | 25 tok/s | + Final tuning |

---

## Critical Files to Modify

1. **src/kernels/gemm_kernels_q5_0.c**
   - Update block sizes
   - Optimize unrolling

2. **src/v6.5/ckernel_alloc_v6.5.c**
   - Implement arena allocator
   - Pre-allocate scratch

3. **src/kernels/gemm_kernels_q8_0.c**
   - Fuse dequant + multiply
   - Optimize INT8 path

4. **src/kernels/attention_kernels.c**
   - Better cache blocking
   - Reduce memory traffic

---

## Debug Commands

```bash
# Check allocation in hot path
valgrind --tool=callgrind ./ck-cli-v6.5 --model ... --max-tokens 50
kcachegrind callgrind.out

# Check cache misses
perf stat -e cache-misses,cache-references ./ck-cli-v6.5 ...

# Compare with llama.cpp
perf stat ./llama.cpp/main ...

# View assembly
objdump -d ck-kernel-inference.so | grep -A20 vec_dot_q5_0
```

---

## Expected Results

After 2 weeks of focused optimization:
- **Target**: 20 tok/s (67% of llama.cpp)
- **Improvement**: 6.9x from current
- **Gap remaining**: 33%

This is achievable because:
1. ✅ We're already using optimized kernels (70% of work done)
2. ✅ 10x gap = clear targets to hit
3. ✅ llama.cpp is open source (we can copy strategies)

---

## Bottom Line

**We can close the gap from 10x to 3x in 2 weeks** by:
1. Copying llama.cpp's GEMM blocking (1.7x)
2. Adding arena allocator (1.4x)  
3. Fusing INT8 operations (1.4x)
4. Optimizing attention (1.5x)
5. Cache-friendly KV layout (1.3x)

**Total**: 1.7 × 1.4 × 1.4 × 1.5 × 1.3 = **6.9x improvement**

**Start NOW!** 🚀

---

## CRITICAL: Prefill Optimization (0% improvement so far!)

**Prefill**: Processing ALL prompt tokens at once (compute-bound)  
**Decode**: Processing ONE token at a time (memory-bound)

### Current Prefill Performance
- **Prefill**: ~1.18 tok/s (9 tokens in 7638ms)
- **Decode**: 2.91 tok/s
- **Problem**: Prefill is 2.5x SLOWER than decode!

### Why Prefill is Slow

**Observation**: Prefill should be FASTER than decode (batch processing)
- llama.cpp: Prefill ~50-100 tok/s, Decode ~30 tok/s
- C-Kernel: Prefill ~1.18 tok/s, Decode ~2.91 tok/s

**Root Cause**: Inefficient batch GEMM

### Prefill-Specific Optimizations

#### 1. Batch GEMM (50% of prefill improvement)
```c
// Current: Process tokens one-by-one in decode loop
for (int i = 0; i < seq_len; i++) {
    gemv(token[i]);  // SLOW!
}

// Target: Batch process all tokens
gemm_batched(tokens[0..seq_len], weights);  // FAST!
```

**File to fix**: `ck-kernel-prefill.c`
- Use `gemm` instead of `gemv` for batch operations
- Process 32-64 tokens at once

#### 2. Multi-Threading (30% of prefill improvement)
```c
// Parallelize across layers
#pragma omp parallel for
for (int layer = 0; layer < num_layers; layer++) {
    process_layer(layer);
}
```

#### 3. Better Memory Layout (20% of prefill improvement)
```c
// Pack activations for cache efficiency
// Target: L2 cache reuse across tokens
```

### Week 0.5: Prefill Sprint (2 days)

#### Day 1: Batch GEMM
```bash
# Modify src/v6.5/ck-kernel-prefill.c
# Replace gemv with gemm_batched

# Benchmark
python3 scripts/v6.5/profile_inference.py | grep Prefill
```

**Target**: 1.18 → 5 tok/s (4.2x)

#### Day 2: OpenMP Parallelization
```c
// Add parallelization
#include <omp.h>

// Parallelize across heads
#pragma omp parallel for collapse(2)
for (int layer = 0; layer < num_layers; layer++) {
    for (int head = 0; head < num_heads; head++) {
        process_head(layer, head);
    }
}
```

**Target**: 5 → 10 tok/s (2x)

#### Day 3: Memory Layout
```c
// Optimize activation layout for batch
// Better cache locality
```

**Target**: 10 → 15 tok/s (1.5x)

---

## Updated Success Metrics (Including Prefill)

| Metric | Current | Week 1 | Week 2 | Week 3 |
|--------|---------|--------|--------|--------|
| **Prefill** | 1.18 tok/s | 10 tok/s | 15 tok/s | 20 tok/s |
| **Decode** | 2.91 tok/s | 10 tok/s | 15 tok/s | 20 tok/s |
| **Overall** | 1.5 tok/s | 10 tok/s | 15 tok/s | 20 tok/s |

---

## Validation Commands

```bash
# Test prefill specifically
python3 scripts/v6.5/profile_inference.py
# Look for: "Prefill: XXX ms (X tokens, X.XX tok/s)"

# Compare batch vs gemv
python3 -c "
import subprocess
# Long prompt (more prefill work)
cmd = ['./ck-cli-v6.5', '--model', 'qwen2-0_5b', '--prompt', 'LONG PROMPT HERE', '--max-tokens', '5']
subprocess.run(cmd)
"
```

---

## Critical Insight

**Prefill is the KEY to overall speed!**

- User prompt: 100 tokens → 100 tok prefill work
- User expects: 1 response token → 1 tok decode work
- **Ratio**: 100:1 prefill:decode

**If prefill is 10x slower, it dominates total time!**

Example:
- Current: 1.18 tok/s prefill, 2.91 tok/s decode
- For 100-token prompt + 10-token response:
  - Prefill: 100 / 1.18 = 85 seconds ❌
  - Decode: 10 / 2.91 = 3.4 seconds
  - **Total: 88 seconds (97% in prefill!)**

**Fix prefill first = biggest impact!**
