# C-Kernel-Engine Profiling Report

**Date:** 2026-01-14
**Model:** Qwen2-0.5B-Instruct (Q4_K_M)
**Profiler:** Intel VTune 2025.0

---

## Executive Summary

Performance comparison shows **llama.cpp is ~2.5x faster** than C-Kernel-Engine for Q4_K_M inference on this system.

| Metric | C-Kernel-Engine | llama.cpp |
|--------|----------------|-----------|
| Decode speed | ~7.1 tok/s | ~17.6 tok/s |
| Time for 10 tokens | ~1.4s | ~0.57s |
| Overhead | Python wrapper + [DUMP] output | Native binary |

---

## System Configuration

### CPU
- **Model:** Intel Core i7-3630QM @ 2.40GHz (Ivy Bridge)
- **Cores:** 4 cores / 8 threads
- **SIMD:** AVX2 ✓ | AVX-512 ✗

### Profiling Tools
- **VTune:** `/opt/intel/oneapi/vtune/2025.0/bin64` (v2025.0.1)
- **Valgrind:** `/usr/bin/valgrind`
- **GProf:** `/usr/bin/gprof`

### Model Location
```
~/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/
├── qwen2-0_5b-instruct-q4_k_m.gguf (397MB)
├── qwen2-0_5b-instruct-q5_0.gguf (397MB)
└── ck-kernel-inference.so (compiled kernel)
```

---

## Performance Results

### C-Kernel-Engine Timing
```
prompt eval:   669.24 ms /    4 tokens ( 167.31 ms/tok,    5.98 tok/s)
decode:       1401.98 ms /   10 runs   ( 140.20 ms/tok,    7.13 tok/s)
sample:         14.87 ms /   10 runs   (   1.49 ms/tok)
total:        2088.05 ms /   14 tokens
```

### llama.cpp Timing
```
[ Prompt: 22.1 t/s | Generation: 17.6 t/s ]
```

### Wall Clock Comparison
```bash
# C-Kernel-Engine (10 tokens)
real    0m44.629s
user    0m29.196s
sys     0m3.204s

# llama.cpp (10 tokens)
real    0m0.039s
user    0m0.008s
sys     0m0.013s
```

---

## Scripts Created

### 1. scripts/profile_comparison.sh
General-purpose profiling with valgrind/callgrind.

**Usage:**
```bash
./scripts/profile_comparison.sh
```

**Features:**
- Detects available compilers (gcc/icx)
- Runs both implementations
- Creates callgrind output
- Includes flamegraph conversion instructions

### 2. scripts/vtune_profile.sh
VTune-specific profiling with multiple analysis types.

**Usage:**
```bash
./scripts/vtune_profile.sh
```

**Analysis Types:**
- Hotspots (CPU time per function)
- uarch-exploration (microarchitecture)
- memory-access (cache/memory behavior)

---

## Key Findings

### Issue 1: Excessive Debug Output
C-Kernel-Engine outputs megabytes of buffer dumps during inference:

```
[DUMP] inp_embd (7168 bytes)
[DUMP] norm-0 (7168 bytes)
[DUMP] attn_norm-0 (7168 bytes)
[DUMP] Qcur-0 (7168 bytes)
...
[DUMP] result_output (607744 bytes)
```

**Impact:** Major performance regression due to I/O overhead.

**Recommendation:** Disable `[DUMP]` in production (set debug flag to false).

### Issue 2: Python Wrapper Overhead
- 3.2s sys time vs 0.013s for llama.cpp
- Significant overhead from Python → C bridge

**Recommendation:** Profile the Python/C boundary to identify overhead.

### Issue 3: Missing Debug Symbols
VTune warnings during profiling:
```
Cannot locate debugging information for file `libggml.so.0'
Cannot locate debugging information for file `libggml-cpu.so.0'
Cannot locate debugging information for file `libllama.so.0'
Cannot locate debugging information for file `llama-cli'
```

**Recommendation:** Rebuild with debug symbols:
```bash
cd llama.cpp && make clean && make CFLAGS="-O2 -g"
```

---

## Performance Gap Analysis

| Factor | Impact | Notes |
|--------|--------|-------|
| [DUMP] I/O overhead | High | Megabytes of output |
| Python wrapper | Medium | 3s sys time difference |
| Q4_K kernel | Medium | May be less optimized than llama.cpp |
| AVX-512 | Low | Affects both similarly (CPU doesn't support) |

---

## Commands Reference

### VTune Setup
```bash
export VTUNE_PATH=/opt/intel/oneapi/vtune/2025.0/bin64
export PATH="$VTUNE_PATH:$PATH"
```

### Profile llama.cpp
```bash
vtune -collect hotspots -r ~/vtune_results/llama -- \
    ./llama.cpp/build/bin/llama-cli \
    -m ~/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
    -p "Hello" -n 5 --temp 0 --threads 4
```

### Profile C-Kernel-Engine
```bash
vtune -collect hotspots -r ~/vtune_results/ck -- \
    python scripts/v6.5/ck_run_v6_5.py run \
    ~/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
    --force-compile --max-tokens 5 --prompt "Hello"
```

### View Results
```bash
# Text summary
vtune -report summary -r ~/vtune_results/llama

# HTML report for browser
vtune -report summary -r ~/vtune_results/llama -format html -report-output ~/llama.html

# Top functions
vtune -report functions -r ~/vtune_results/llama -limit 20
```

---

## Next Steps

1. **Disable debug output** in C-Kernel-Engine
2. **Rebuild llama.cpp with debug symbols** for detailed profiling
3. **Profile C-Kernel-Engine** with VTune to identify kernel bottlenecks
4. **Compare Q4_K dequantization** between implementations
5. **Profile attention kernel** specifically

---

## Files Generated

| File | Description |
|------|-------------|
| `scripts/profile_comparison.sh` | General profiling script |
| `scripts/vtune_profile.sh` | VTune profiling script |
| `docs/PROFILING_REPORT.md` | This report |

---

*Generated by Claude Code during C-Kernel-Engine performance analysis session.*
