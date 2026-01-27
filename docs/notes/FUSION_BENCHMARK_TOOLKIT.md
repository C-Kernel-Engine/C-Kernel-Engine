# Fusion Benchmark Toolkit

## Overview

This toolkit helps you determine if the **1.45x "fusion speedup"** is real or a benchmark artifact. It provides multiple scripts and guides for proper fusion analysis.

---

## Quick Start

### Step 1: Check Your Environment

```bash
# Check if perf is available (for memory profiling)
python3 scripts/final_fusion_test.py --check-perf
```

### Step 2: Run Original Benchmark

```bash
# This shows the 1.45x speedup
python3 scripts/bench_mega_fused_attention_prefill.py --q8-outproj --seq-lens 32,64 --iters 3
```

**Expected output**:
```
synthetic-q5wv tokens=32 baseline=26272 us fused=18985 us speedup=1.38x
synthetic-q5wv tokens=64 baseline=54855 us fused=38246 us speedup=1.43x
```

### Step 3: Analyze Results

```bash
# Interactive analysis tool
python3 scripts/final_fusion_test.py --seq-lens 32,64 --iters 3
```

This will:
- Run the original benchmark
- Explain why the speedup is likely an artifact
- Show you next steps

---

## Files in This Toolkit

### 📊 Benchmark Scripts

1. **`scripts/bench_mega_fused_attention_prefill.py`**
   - **Original benchmark** (shows 1.45x speedup)
   - Uses Python loop for baseline (slow)
   - Uses optimized C kernels for fused (fast)
   - **This is the source of the "artifact"**

2. **`scripts/simple_fusion_test_v2.py`**
   - **Simple comparison** of fused vs separate
   - Shows 1.30x speedup (also likely artifact)
   - Useful for quick testing
   - **Usage**: `python3 scripts/simple_fusion_test_v2.py`

3. **`scripts/benchmark_fusion_real.py`**
   - **Proper benchmark** with optimized baselines
   - Tests multiple fusion variants
   - Ready to use once kernel issues are fixed
   - **Usage**: `python3 scripts/benchmark_fusion_real.py`

4. **`scripts/final_fusion_test.py`**
   - **Interactive analysis tool**
   - Runs original benchmark and explains results
   - Guides you through the investigation
   - **Usage**: `python3 scripts/final_fusion_test.py --seq-lens 32,64`

### 📚 Documentation

1. **`FUSION_ANALYSIS_SUMMARY.md`**
   - **Complete analysis** of the fusion findings
   - Explains root cause of the artifact
   - Shows expected outcomes
   - **READ THIS FIRST**

2. **`MEMORY_PROFILING_GUIDE.md`**
   - **How to measure DRAM traffic**
   - perf, VTune, likwid alternatives
   - What metrics to look for
   - **Essential for validation**

3. **`FUSION_BENCHMARK_TOOLKIT.md`** (this file)
   - Overview of all tools
   - Quick start guide
   - File descriptions

---

## Understanding the Results

### If You See 1.3-1.5x Speedup

This is likely an **artifact** from:

| Component | Baseline | Fused |
|-----------|----------|-------|
| Flattening | Python loop (448 iterations) | C kernel |
| GEMM | N×GEMV (896 serial calls) | Tiled kernel |

**The "speedup" is from**:
- ✅ Using optimized C instead of Python
- ✅ Using correct memory layout
- ❌ **NOT from fusion**

### What Real Fusion Benefit Looks Like

After fixing the benchmark:

```
Baseline (optimized separate):  18.2 ms
Fused (true fusion):           17.8 ms
Speedup:                       1.02x
```

**Or worse**:
```
Baseline (optimized separate):  18.2 ms
Fused (with overhead):         18.5 ms
Speedup:                       0.98x (fusion is slower!)
```

---

## Memory Profiling (To Validate)

Since `perf` is restricted, use alternatives:

### Option 1: Intel VTune (Recommended if available)

```bash
# Check if VTune is installed
which vtune || echo "VTune not available"

# Run memory access analysis
vtune -collect memory-access \
    -result-dir vtune_results \
    -- python3 scripts/simple_fusion_test_v2.py

# Generate report
vtune -report memory-access -result-dir vtune_results
```

**Look for**:
- LLC-load-misses (DRAM traffic)
- Lower in fused version = real benefit

### Option 2: likwid (Open Source)

```bash
# Install: git clone https://github.com/RRZE-HPC/likwid.git && make install
likwid-perfctr -C 0 -g MEM -- python3 scripts/simple_fusion_test_v2.py
```

### Option 3: Add Canary Detection (In C Code)

```c
// Add to kernel
float canary_before = get_canary_value();

// Run computation
mega_fused_attention(...);

// Check if canary changed (indicates DRAM write)
if (canary_changed()) {
    printf("WARNING: Intermediates touched DRAM!\n");
}
```

---

## Expected Timeline

### Day 1: Run Initial Benchmarks
```bash
# 1. Quick test
python3 scripts/simple_fusion_test_v2.py

# 2. Original benchmark
python3 scripts/bench_mega_fused_attention_prefill.py --q8-outproj --seq-lens 64 --iters 5

# 3. Analysis
python3 scripts/final_fusion_test.py --seq-lens 64 --iters 5
```

**Outcome**: Confirm 1.3-1.5x speedup (artifact)

### Day 2: Memory Profiling
```bash
# If VTune available:
vtune -collect memory-access -- python3 simple_fusion_test_v2.py
vtune -report summary

# If not, add canary detection to C code
```

**Outcome**: See if fusion reduces LLC misses

### Day 3: Fix Baseline
Modify `bench_mega_fused_attention_prefill.py`:
- Replace Python loop with `ck_gemm_nt_head_major_*`
- Use optimized kernels for baseline
- Re-run benchmark

**Outcome**: Speedup drops to 1.05x (minimal)

---

## What We Expect to Find

### Confirmed Artifact (Most Likely)

After proper analysis:
- Original benchmark: 1.45x (Python vs C)
- Fixed benchmark: 1.05x (optimized vs optimized)
- **Conclusion**: v6.6 was correct, fusion doesn't help at this scale

### Real Fusion Benefit (Unlikely but Possible)

If fusion genuinely helps:
- Memory profiling shows 20-30% LLC miss reduction
- Fixed benchmark still shows 1.2x+ speedup
- **Conclusion**: Fusion helps for this specific workload

### Fusion Hurts Performance (Also Possible)

If fusion has overhead:
- Memory profiling shows similar LLC misses
- Fixed benchmark shows 0.95x (slower)
- **Conclusion**: Don't use fusion for Qwen2-0.5B

---

## Key Takeaways

1. **The 1.45x speedup is likely an artifact** from comparing Python+C vs optimized C

2. **Real fusion benefit is expected to be minimal** (1.05x or less) based on v6.6 analysis

3. **Memory profiling is essential** to validate whether fusion reduces DRAM traffic

4. **Proper baseline is critical** - must use optimized kernels for both paths

5. **Fusion only helps when intermediates exceed cache** (L1/L2), which doesn't happen at Qwen2-0.5B scale

---

## Quick Commands Reference

```bash
# Quick test
python3 scripts/simple_fusion_test_v2.py

# Full analysis
python3 scripts/final_fusion_test.py --seq-lens 64 --iters 5

# Check perf
python3 scripts/final_fusion_test.py --check-perf

# Original benchmark (shows artifact)
python3 scripts/bench_mega_fused_attention_prefill.py --q8-outproj --seq-lens 64 --iters 5

# With perf (if available)
perf stat -e LLC-load-misses -- python3 scripts/simple_fusion_test_v2.py

# With VTune (if available)
vtune -collect memory-access -- python3 scripts/simple_fusion_test_v2.py
```

---

## Support

- **Analysis**: Read `FUSION_ANALYSIS_SUMMARY.md`
- **Profiling**: Read `MEMORY_PROFILING_GUIDE.md`
- **Code**: Review `src/kernels/fused/mega_fused_attention_prefill.c`
- **Tests**: Check `unittest/test_mega_fused_attention_prefill.py`