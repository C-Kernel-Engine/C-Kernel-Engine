# Memory Profiling Guide: Measuring DRAM Traffic

## Quick Answer: Use `perf` First, VTune If Available

### For Quick Diagnostics (Recommended Start)
```bash
# Basic cache miss analysis
perf stat -e cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./benchmark

# Memory access patterns
perf mem record -- ./benchmark
perf mem report
```

### For Deep Analysis
**Intel VTune Profiler** (if you have access):
```bash
vtune -collect memory-access -result-dir vtune_results -- ./benchmark
vtune -report summary
```

---

## Tool Comparison

| Tool | Setup | Detail Level | Cost | Best For |
|------|-------|--------------|------|----------|
| **perf** | Minimal | Good | Free | Quick diagnostics, CI/CD |
| **VTune** | Moderate | Excellent | Paid (free for non-commercial) | Deep optimization |
| **likwid** | Moderate | Very Good | Free | Open-source alternative to VTune |
| **Intel PCM** | Minimal | Limited | Free | CPU-level metrics only |

---

## Recommended Workflow

### Phase 1: Quick Check with perf

```bash
# 1. Install perf (usually pre-installed on Linux)
# 2. Run with performance counters
perf stat -e \
    L1-dcache-loads,\
    L1-dcache-load-misses,\
    LLC-loads,\
    LLC-load-misses,\
    cycles,\
    instructions,\
    -- \
    python3 scripts/benchmark_fusion_real.py --seq-lens 64 --iters 5

# Expected output:
#   Performance counter stats for 'python3 scripts/benchmark_fusion_real.py':

#   L1-dcache-loads:          123,456,789
#   L1-dcache-load-misses:       5,678,912  (4.60%)
#   LLC-loads:                  12,345,678
#   LLC-load-misses:             1,234,567  (10.00%)
#   cycles:                   456,789,012
#   instructions:              234,567,890
```

**Interpretation**:
- **L1 miss rate < 5%**: Excellent cache efficiency
- **L1 miss rate 5-10%**: Good
- **L1 miss rate > 10%**: Room for optimization
- **LLC miss rate**: Shows DRAM traffic. Lower is better.

---

### Phase 2: Detailed Memory Analysis with perf mem

```bash
# Record memory access patterns
perf mem record -- python3 scripts/benchmark_fusion_real.py --seq-lens 64 --iters 1

# Generate detailed report
perf mem report --sort=local_weight,sample

# Shows:
#   - Which functions access memory
#   - Load/store breakdown
#   - Memory type (L1, L2, LLC, DRAM)
```

---

### Phase 3: Deep Dive with VTune (If Available)

#### Setup VTune
```bash
# Install (choose one):
# Ubuntu/Debian:
sudo apt install intel-vtune

# Or download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html

# Verify installation
vtune --help
```

#### Run Memory Access Analysis
```bash
# Create result directory
mkdir -p vtune_results

# Collect memory access patterns
vtune -collect memory-access -result-dir vtune_results -- python3 scripts/benchmark_fusion_real.py --seq-lens 64 --iters 1

# Generate report
vtune -report summary -result-dir vtune_results

# Generate detailed hotspot report
vtune -report hotspots -result-dir vtune_results

# Generate memory access report (THIS IS WHAT WE WANT!)
vtune -report memory-access -result-dir vtune_results
```

#### Key VTune Metrics for DRAM Analysis

1. **Memory Access Summary**:
   ```
   Memory Access:
     High Traffic:           12.3% of cycles
     Bandwidth Utilization:   45.2%
     DRAM Reads:             1.23 GB
     DRAM Writes:            0.87 GB
   ```

2. **Cache Miss Analysis**:
   ```
   L1 Miss Rate:            5.2% (Good)
   L2 Miss Rate:           12.3% (Check tiling)
   LLC Miss Rate:          45.6% (High - indicates DRAM traffic)
   DRAM Bandwidth:         18.5 GB/s (Out of 25 GB/s max = 74%)
   ```

3. **Top Memory Consumers**:
   ```
   Function                     Memory GB  % of Total
   mega_fused_attention_prefill    2.34      45.2%
   rmsnorm_fp16                   1.23      23.7%
   flash_attention               0.89      17.2%
   ```

---

### Phase 4: Alternative - likwid (Open Source)

If you don't have VTune, `likwid` is excellent:

```bash
# Install likwid
git clone https://github.com/RRZE-HPC/likwid.git
cd likwid
make install

# Run with memory counters
likwid-perfctr -C 0 -g MEM ./benchmark

# Or use likwid-pin for thread pinning
likwid-pin -c 0-3 ./benchmark
```

---

## Script for Automated Memory Benchmarking

```bash
#!/bin/bash
# run_memory_bench.sh

echo "=== Phase 1: Quick perf check ==="
perf stat -e cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
    -- python3 scripts/benchmark_fusion_real.py --seq-lens 64 --iters 5 2>&1 | tee perf_quick.txt

echo ""
echo "=== Phase 2: Memory access patterns ==="
perf mem record -- python3 scripts/benchmark_fusion_real.py --seq-lens 64 --iters 1
perf mem report > perf_mem_report.txt

echo ""
echo "=== Phase 3: VTune (if available) ==="
if command -v vtune &> /dev/null; then
    vtune -collect memory-access -result-dir vtune_results \
        -- python3 scripts/benchmark_fusion_real.py --seq-lens 64 --iters 1
    vtune -report memory-access -result-dir vtune_results > vtune_memory_report.txt
else
    echo "VTune not available. Install with: apt install intel-vtune"
fi

echo ""
echo "Results:"
echo "  - perf_quick.txt: Basic performance counters"
echo "  - perf_mem_report.txt: Memory access patterns"
echo "  - vtune_memory_report.txt: Detailed DRAM analysis (if VTune available)"
```

---

## What to Look For

### Signs of Good Fusion
```
Separate (baseline):
  L1 miss rate:  8.5%
  LLC miss rate: 35.2%
  DRAM traffic:  2.3 GB

Fused:
  L1 miss rate:  7.8% (slightly better)
  LLC miss rate: 28.1% (20% reduction!)
  DRAM traffic:  1.8 GB (22% reduction!)
```

### Signs of Poor/No Fusion Benefit
```
Separate (baseline):
  L1 miss rate:  5.2%
  LLC miss rate: 12.3%
  DRAM traffic:  1.2 GB

Fused:
  L1 miss rate:  5.3% (same)
  LLC miss rate: 12.1% (same)
  DRAM traffic:  1.2 GB (no change)
```

---

## Quick Test Command

```bash
# Easiest way to start:
perf stat -e L1-dcache-load-misses,LLC-load-misses,cycles,instructions \
    -- python3 scripts/benchmark_fusion_real.py --seq-lens 64 --iters 3

# Look for:
# - LLC-load-misses reduction in fused version
# - Similar L1 misses (intermediates fit in L1 anyway)
# - Overall cycle reduction matching speedup
```

---

## Recommended Approach

1. **Start with `perf`** (always available)
2. **If you have VTune**, use it for detailed analysis
3. **`likwid`** is great open-source alternative

The goal is to see **LLC miss reduction** as the primary indicator of fusion benefit, since L1 misses won't change much (intermediates are small and fit in L1).

**Key metric**: `LLC-load-misses` - this directly measures DRAM traffic. Lower is better.