# Computational Complexity Analysis

## Introduction

Understanding the **computational complexity** and **hardware limits** is critical for GEMM optimization. This document analyzes:
1. **FLOPs:** How much compute is required
2. **Memory bandwidth:** How much data needs to move
3. **Arithmetic intensity:** Ratio of compute to memory
4. **Roofline model:** Determining if we're compute-bound or memory-bound

---

## GEMM Computational Complexity

### FLOPs Calculation

For `C[M×N] = A[M×K] × B[K×N]`:

```
Each output element C[i,j] requires:
  - K multiplications
  - K additions
  - Total: 2K operations (1 multiply-add counts as 2 FLOPs)

Total FLOPs = M × N × 2K

Example: M=N=K=1024
  FLOPs = 1024 × 1024 × 2 × 1024
        = 2,147,483,648 FLOPs
        ≈ 2.15 GFLOP
```

### LLM Workload Examples

| Operation | Shape | FLOPs | Notes |
|-----------|-------|-------|-------|
| **Decode QKV** | [1, 768] · [768, 2304] | 2 × 1 × 2304 × 768 = 3.5 MFLOP | Per token |
| **Prompt QKV** | [512, 768] · [768, 2304] | 2 × 512 × 2304 × 768 = 1.8 GFLOP | Per prompt |
| **MLP Up** | [512, 768] · [768, 3072] | 2 × 512 × 3072 × 768 = 2.4 GFLOP | Per layer |
| **Attention** | [512, 64] · [64, 512] | 2 × 512 × 512 × 64 = 33.5 MFLOP | Per head |

**Full GPT-2 (12 layers, 12 heads):**
```
Per token (decode): 3.5M × 12 layers × (4 ops) ≈ 168 MFLOP
Per prompt (512 tok): 1.8G × 12 × 4 + attention ≈ 90 GFLOP
```

---

## Memory Traffic Analysis

### Naive GEMM (No Blocking)

```c
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];  // (*)
        }
        C[i * N + j] = sum;
    }
}
```

**Memory accesses at line (*):**
- Load `A[i * K + k]`: 1 load per K iteration
- Load `B[k * N + j]`: 1 load per K iteration  (CACHE MISS!)
- Total per (i,j,k): 2 loads

**Total memory operations:**
```
Loads: M × N × K × 2 = 2MNK loads
Stores: M × N = MN stores
Total: 2MNK + MN ≈ 2MNK memory ops (for large K)

For M=N=K=1024:
  Memory ops = 2 × 1024³ ≈ 2.15 billion ops
  Bytes = 2.15B × 4 bytes = 8.6 GB
```

But the matrices only occupy:
```
A: M × K = 1024 × 1024 × 4 = 4 MB
B: K × N = 1024 × 1024 × 4 = 4 MB
C: M × N = 1024 × 1024 × 4 = 4 MB
Total: 12 MB

Why 8.6 GB traffic for 12 MB data?
→ CACHE MISSES! Data reloaded from DRAM many times.
```

### Optimal GEMM (With Blocking)

With proper cache blocking, each matrix element is loaded from DRAM **once**:

```
Loads from DRAM:
  A: M × K floats
  B: K × N floats
Stores to DRAM:
  C: M × N floats

Total bytes = (MK + KN + MN) × 4 bytes

For M=N=K=1024:
  Total bytes = (1024² + 1024² + 1024²) × 4
              = 3 × 1024² × 4
              = 12 MB ✓
```

**Arithmetic Intensity (Optimal):**
```
AI = FLOPs / Bytes
   = (2 × M × N × K) / ((M×K + K×N + M×N) × 4)

For M=N=K:
  AI = (2 × N³) / (3 × N² × 4)
     = N / 6

For N=1024:
  AI = 1024 / 6 ≈ 171 FLOPs/byte
```

**This is excellent!** High AI means compute-bound, which is what we want.

---

## Hardware Performance Limits

### Your CPU (Assumed: Intel AVX-512 system)

**Processor specs (typical):**
```
Architecture: Intel Haswell/Skylake/Cascade Lake with AVX-512
Cores: 8 (assumption)
Clock: 3.0 GHz (base)
Vector width: 512 bits (16 fp32 per register)
FMA units: 2 per core (Skylake+) or 1 (Haswell)
```

### Peak GFLOPS

**Single Core:**
```
FMA throughput: 2 FMAs/cycle (Skylake+)
Operations per FMA: 16 fp32 × 2 (mul + add) = 32 FLOPs
Clock: 3.0 GHz

Peak = 2 FMA/cycle × 32 FLOPs/FMA × 3.0 GHz
     = 192 GFLOPS/core
```

**8 Cores:**
```
Peak = 192 × 8 = 1,536 GFLOPS
```

**Realistic (80% efficiency):**
```
Realistic = 1,536 × 0.8 = 1,229 GFLOPS
```

### Memory Bandwidth

**DDR4 Memory:**
```
DDR4-2400: ~19 GB/s per channel
Dual-channel: 2 × 19 = 38 GB/s
Quad-channel (server): 4 × 19 = 76 GB/s

Typical workstation: 40-80 GB/s
```

### Roofline Model

The **Roofline Model** shows performance limits based on arithmetic intensity.

**Ridge Point:**
```
Ridge = Peak GFLOPS / Memory BW
      = 1536 GFLOPS / 76 GB/s
      = 20.2 FLOPs/byte
```

**Interpretation:**
- If AI > 20.2: Compute-bound (good for GEMM!)
- If AI < 20.2: Memory-bound (bad, need better blocking)

**For our optimal GEMM (AI = 171 FLOPs/byte):**
```
171 >> 20.2 → COMPUTE-BOUND ✓
```

We should be able to hit peak GFLOPS!

---

## Roofline Plot

```
                   Peak GFLOPS (1536)
                   ─────────────────────────────────────
Performance       │                                      │
(GFLOPS)          │                    ╱─────────────────┤ Compute-bound region
              1500│                  ╱                   │
                  │                ╱                     │
              1000│              ╱                       │
                  │            ╱                         │
               500│          ╱                           │
                  │        ╱                             │
                  │      ╱                               │
                0 │────╱─────────────────────────────────│
                  0   20.2 (ridge)  100        200   FLOPs/byte
                      ↑                            ↑
                  Memory-bound               Our GEMM (AI=171)
```

**Our GEMM should be in the compute-bound region**, hitting ~1200-1500 GFLOPS on 8 cores.

---

## Current Performance Gap Analysis

### Measured Performance (Estimated)

```
Our current implementation (64×64 blocking):
  M=N=K=1024: ~250 GFLOPS on 8 cores
  % of peak: 250 / 1536 = 16.3%
  Per-core: 250 / 8 = 31 GFLOPS/core
  % of core peak: 31 / 192 = 16.1%
```

**This is LOW!** We should be hitting 80%+ of peak.

### Why Are We Slow?

Let's analyze with `perf stat`:

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,\
           L1-dcache-loads,L1-dcache-load-misses ./benchmark_gemm

# Example output (hypothetical):
  5,234,567,890  cycles
  3,456,789,012  instructions      # IPC = 0.66 (LOW!)
    234,567,890  cache-references
     12,345,678  cache-misses      # 5.3% miss rate (OK)
```

**IPC (Instructions Per Cycle) = 0.66**

This is the smoking gun! We're stalled waiting for something.

**Ideal IPC for GEMM:**
```
Theoretical max IPC: 4-5 (out-of-order execution)
Well-optimized GEMM: 2.5-3.5 IPC
Our current: 0.66 IPC

→ We're spending 2/3 of cycles STALLED
```

**Likely causes:**
1. **Cache misses:** Even with 5.3% L1 miss rate, could be hitting L2/L3
2. **No prefetching:** Not loading next data in advance
3. **No register blocking:** Not keeping accumulators in registers
4. **Poor instruction mix:** Too many loads, not enough compute

---

## Target Performance

### After Microkernel + Blocking

**Expected:**
```
Microkernel (16×2 or 6×8): 2.5-3.0 IPC
Three-level blocking: Minimal cache misses (<1% L3)

Expected GFLOPS: 1200-1400 (80-90% of peak)
Expected per-core: 150-175 GFLOPS (80-90% of 192)
```

### After AI-Specific Optimizations

**Small-M (decode, M=1):**
```
Current (est.): 10 GFLOPS (latency-bound, many overheads)
Optimized: 80-100 GFLOPS (better IPC, less overhead)
Speedup: 8-10x
```

**Attention (K=64):**
```
Current (est.): 300 GFLOPS (suboptimal for small K)
Optimized: 800-1000 GFLOPS (K-unrolled, better cache)
Speedup: 2.7-3.3x
```

---

## Computational Costs in LLM Inference

### GPT-2 (124M parameters, 12 layers)

**Single token generation (M=1):**
```
QKV:   1 × 768 × 2304 × 2 = 3.5 MFLOP
Attn:  12 heads × (1 × 512 × 64 × 2 + 1 × 64 × 512 × 2) = 1.6 MFLOP
Out:   1 × 768 × 768 × 2 = 1.2 MFLOP
MLP:   1 × 768 × 3072 × 2 + 1 × 3072 × 768 × 2 = 9.4 MFLOP

Per layer: 3.5 + 1.6 + 1.2 + 9.4 = 15.7 MFLOP
All 12 layers: 15.7 × 12 = 188 MFLOP

At 100 GFLOPS: 188 MFLOP / 100 GFLOPS = 1.88 ms/token
→ 532 tokens/sec
```

**Prompt encoding (M=512):**
```
Per layer: ~2.5 GFLOP (scales with M²)
All 12 layers: 30 GFLOP

At 1200 GFLOPS: 30 GFLOP / 1200 GFLOPS = 25 ms
→ For 512 tokens, 25 ms total (0.05 ms/token)
```

---

## Memory Bandwidth Requirements

### Best Case (Perfect Blocking)

```
For M=N=K=1024:
  Data: 12 MB
  Compute time at 1200 GFLOPS: 2.15 GFLOP / 1200 = 1.79 ms

Required bandwidth: 12 MB / 1.79 ms = 6.7 GB/s

Available bandwidth: 76 GB/s

→ Only using 8.8% of bandwidth! ✓ Definitely compute-bound.
```

### Worst Case (No Cache Reuse)

```
Data: 8.6 GB (naive, no blocking)
Compute time: 1.79 ms (same FLOPs)

Required bandwidth: 8.6 GB / 1.79 ms = 4,804 GB/s

Available bandwidth: 76 GB/s

→ Need 63x more bandwidth! → Memory-bound ✗
→ This is why blocking is critical
```

---

## Performance Validation Checklist

When profiling optimized kernels, verify:

### Compute-Bound Checklist
- [ ] **IPC ≥ 2.5:** Good instruction-level parallelism
- [ ] **L3 miss rate < 1%:** Effective blocking
- [ ] **GFLOPS ≥ 80% peak:** Close to hardware limit
- [ ] **CPU utilization ≈ 100%:** Not waiting on I/O

### Memory-Bound Warning Signs
- [ ] **IPC < 1.0:** Lots of stalls
- [ ] **High cache miss rate:** Poor blocking
- [ ] **GFLOPS < 30% peak:** Not using FMA units
- [ ] **Low memory bandwidth:** Data not in cache

---

## Summary: Key Metrics

| Metric | Formula | Target (GEMM) | Our Current | After Opt |
|--------|---------|---------------|-------------|-----------|
| **IPC** | instructions/cycles | 2.5-3.5 | 0.66 | 2.8-3.2 |
| **GFLOPS** | FLOPs/time | 1200-1400 | 250 | 1200+ |
| **% Peak** | GFLOPS/peak | 80-90% | 16% | 80%+ |
| **L3 Miss %** | L3_misses/L3_refs | <1% | ~5% | <1% |
| **AI** | FLOPs/bytes | >20 | 171 (theor) | 171 |
| **BW Usage** | bytes/time | <10% | ~50% | <10% |

**Goal:** Move from **memory-bound** (due to poor cache reuse) to **compute-bound** (hitting CPU FLOP limits).

---

**Next Steps:**
1. Profile current implementation with `perf stat`
2. Measure actual GFLOPS, IPC, cache misses
3. Implement microkernel (target: IPC > 2.5)
4. Re-measure and validate

**Tools:**
- See [profiling/00-profiling-methodology.md](profiling/00-profiling-methodology.md)
- Use `perf stat`, `likwid-perfctr`, or VTune

---

**References:**
- [01-cache-hierarchy-blocking.md](01-cache-hierarchy-blocking.md) - Why blocking matters
- [03-microkernel-design.md](03-microkernel-design.md) - How to improve IPC
- [profiling/00-profiling-methodology.md](profiling/00-profiling-methodology.md) - How to measure
