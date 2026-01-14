# VTune Profiling Guide

## Quick Start

### 1. Single Run VTune (Easiest)
```bash
# Collect hotspots
vtune -collect hotspots -result-dir vtune_results -- ./build/ck-cli-v6.5 --model qwen2-0_5b-instruct-q4_k_m --max-tokens 100 --prompt "Hello"

# Generate report
vtune -report -result-dir vtune_results -format html -output vtune_report.html

# View report
open vtune_report.html  # macOS
# or
firefox vtune_report.html  # Linux
```

### 2. VTune Server Mode (Remote Access)
```bash
# Start VTune server
vtune -start-server

# Run analysis
vtune -collect hotspots -server-collect -- ./build/ck-cli-v6.5 --model qwen2-0_5b-instruct-q4_k_m --max-tokens 100 --prompt "Hello"

# Stop server
vtune -stop-server
```

**Port Forward for Remote Access:**
```bash
# From your local machine, forward VTune web server
ssh -L 8080:localhost:8080 user@avx512-machine

# Then open browser: http://localhost:8080
```

## Analysis Types

### Hotspots (CPU Time)
```bash
vtune -collect hotspots -result-dir vtune_hotspots -- ./build/ck-cli-v6.5 --model qwen2-0_5b-instruct-q4_k_m --max-tokens 100 --prompt "Hello"
vtune -report -result-dir vtune_hotspots
```
**What it shows:** Which functions consume the most CPU time. Expected findings:
- `gemv_q4_k_q8_k_vnni()` or similar kernels
- Memory access functions
- Token sampling code

### Memory Access Analysis
```bash
vtune -collect memory-access -result-dir vtune_memory -- ./build/ck-cli-v6.5 --model qwen2-0_5b-instruct-q4_k_m --max-tokens 100 --prompt "Hello"
vtune -report -result-dir vtune_memory
```
**What it shows:** Cache misses, DRAM access, memory bandwidth. Expected findings:
- LLC misses (last level cache - goes to DRAM)
- L1/L2 hit rates
- Memory bandwidth utilization

### Microarchitecture Analysis
```bash
vtune -collect uarch-exploration -result-dir vtune_uarch -- ./build/ck-cli-v6.5 --model qwen2-0_5b-instruct-q4_k_m --max-tokens 100 --prompt "Hello"
vtune -report -result-dir vtune_uarch
```
**What it shows:** CPU pipeline utilization, branch prediction, etc.

## Interpreting Results

### Look For:

1. **Top Functions**
   - If `gemv_q4_k_q8_k_*` is top → Kernel is the bottleneck (good!)
   - If `quantize_row_q8_k_*` is top → Quantization overhead
   - If memory functions are top → Memory bottleneck

2. **Memory Analysis**
   - LLC Load Misses → High = poor cache locality, goes to DRAM
   - L1 Hit Rate → High = good (target >95%)
   - Memory Bandwidth → High = bandwidth-limited

3. **CPU Utilization**
   - If single-core shows kernel is bottleneck → Need faster kernels
   - If single-core shows memory is bottleneck → Need better cache locality
   - Multi-core scaling → Poor = need parallelization

## Quick Commands

```bash
# Hotspots (find slow functions)
vtune -collect hotspots -result-dir vtune -- ./build/ck-cli-v6.5 --model qwen2-0_5b-instruct-q4_k_m --max-tokens 100 --prompt "Hello"

# Memory (find cache issues)
vtune -collect memory-access -result-dir vtune_mem -- ./build/ck-cli-v6.5 --model qwen2-0_5b-instruct-q4_k_m --max-tokens 100 --prompt "Hello"

# Both together
vtune -collect both -result-dir vtune_both -- ./build/ck-cli-v6.5 --model qwen2-0_5b-instruct-q4_k_m --max-tokens 100 --prompt "Hello"
```

## Expected Findings

### Single-Core Decode
**Goal:** Identify if single-core is already fast enough
- Hotspots should show GEMM kernel time
- If kernel < 50% of time → memory/quantization overhead

### Multi-Core Scaling
**Goal:** Identify why parallel scaling is poor
- Compare single-core vs multi-core hotspots
- If kernel time similar → need parallel decode loop
- If kernel time different → threading overhead

## Share Results

**To share with us:**
```bash
# Package VTune results
tar -czf vtune_results.tar.gz vtune_results/

# Or just the report
vtune -report -result-dir vtune_results -format html -output vtune_report.html
```

Then share the `vtune_results.tar.gz` or `vtune_report.html` file.
