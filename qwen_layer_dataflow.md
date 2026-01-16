# Qwen2-0.5B Layer Data Flow Analysis

## Model Configuration
```
hidden_size (D)      = 896
num_heads (H)        = 14
num_kv_heads (H_kv)  = 2
head_dim             = 64
intermediate_size    = 4864
rope_theta           = 1,000,000
rms_norm_eps         = 1e-6
```

---

## Single Layer Forward Pass (T=1 token decode)

### Operation 1: RMSNorm (Pre-Attention)
```
INPUTS:
  - input[1, 896]        READ from DRAM    3.5 KB
  - ln1_gamma[896]       READ from DRAM    3.5 KB (weight)

OUTPUTS:
  - ln1_out[1, 896]      WRITE to DRAM     3.5 KB  ⚠️ BOTTLENECK
  - ln1_rstd[1]          WRITE to DRAM     4 bytes

COMPUTE: ~1800 FLOPs (sum squares, rsqrt, scale)
MEMORY:  ~10 KB transferred
```

### Operation 2: QKV Linear Projection
```
INPUTS:
  - ln1_out[1, 896]      READ from DRAM    3.5 KB   ← Just written!
  - Wq[14, 64, 896]      READ from DRAM    3.2 MB  (weight)
  - Wk[2, 64, 896]       READ from DRAM    460 KB  (weight)
  - Wv[2, 64, 896]       READ from DRAM    460 KB  (weight)
  - bq[14, 64]           READ from DRAM    3.5 KB  (bias)
  - bk[2, 64]            READ from DRAM    512 B   (bias)
  - bv[2, 64]            READ from DRAM    512 B   (bias)

OUTPUTS:
  - Q[14, 1, 64]         WRITE to DRAM     3.5 KB  ⚠️
  - K[2, 1, 64]          WRITE to DRAM     512 B   ⚠️
  - V[2, 1, 64]          WRITE to DRAM     512 B   ⚠️

COMPUTE:
  Q: 14 heads × 64 × 896 × 2 = 1.6M FLOPs
  K: 2 heads × 64 × 896 × 2 = 230K FLOPs
  V: 2 heads × 64 × 896 × 2 = 230K FLOPs
  Total: ~2M FLOPs

MEMORY: ~4.1 MB transferred (dominated by weight reads)
```

### Operation 3: RoPE (Rotary Position Embedding)
```
INPUTS:
  - Q[14, 1, 64]         READ from DRAM    3.5 KB   ← Just written!
  - K[2, 1, 64]          READ from DRAM    512 B    ← Just written!
  - cos_cache[pos, 32]   READ from DRAM    128 B
  - sin_cache[pos, 32]   READ from DRAM    128 B

OUTPUTS:
  - Q[14, 1, 64]         WRITE to DRAM (in-place)  3.5 KB  ⚠️
  - K[2, 1, 64]          WRITE to DRAM (in-place)  512 B   ⚠️

COMPUTE: ~2K FLOPs (rotations)
MEMORY:  ~8 KB transferred
```

### Operation 4: Scaled Dot-Product Attention (GQA)
```
INPUTS:
  - Q[14, 1, 64]         READ from DRAM    3.5 KB   ← Just written!
  - K[2, T_cache, 64]    READ from KV cache (can be huge!)
  - V[2, T_cache, 64]    READ from KV cache

OUTPUTS:
  - attn_out[14, 1, 64]  WRITE to DRAM     3.5 KB  ⚠️

COMPUTE:
  QK^T: 14 × 1 × T_cache × 64 × 2 FLOPs
  Softmax: 14 × T_cache FLOPs
  Attn×V: 14 × 1 × 64 × T_cache × 2 FLOPs

MEMORY: Depends on cache length. For T_cache=100: ~25 KB
```

### Operation 5: Output Projection + Residual
```
INPUTS:
  - attn_out[14, 1, 64]  READ from DRAM    3.5 KB   ← Just written!
  - Wo[896, 896]         READ from DRAM    3.2 MB  (weight)
  - bo[896]              READ from DRAM    3.5 KB  (bias)
  - input[1, 896]        READ from DRAM    3.5 KB  (residual)

OUTPUTS:
  - residual1[1, 896]    WRITE to DRAM     3.5 KB  ⚠️

COMPUTE: 896 × 896 × 2 = 1.6M FLOPs
MEMORY: ~3.2 MB transferred
```

### Operation 6: RMSNorm (Pre-MLP)
```
INPUTS:
  - residual1[1, 896]    READ from DRAM    3.5 KB   ← Just written!
  - ln2_gamma[896]       READ from DRAM    3.5 KB  (weight)

OUTPUTS:
  - ln2_out[1, 896]      WRITE to DRAM     3.5 KB  ⚠️

COMPUTE: ~1800 FLOPs
MEMORY: ~10 KB transferred
```

### Operation 7: Gate + Up Projection (SwiGLU MLP)
```
INPUTS:
  - ln2_out[1, 896]      READ from DRAM    3.5 KB   ← Just written!
  - W1[9728, 896]        READ from DRAM    34.6 MB (weight) ← BIGGEST!
  - b1[9728]             READ from DRAM    38 KB   (bias)

OUTPUTS:
  - fc1_out[1, 9728]     WRITE to DRAM     38 KB   ⚠️

COMPUTE: 9728 × 896 × 2 = 17.4M FLOPs
MEMORY: ~34.7 MB transferred (dominated by W1)
```

### Operation 8: SwiGLU Activation
```
INPUTS:
  - fc1_out[1, 9728]     READ from DRAM    38 KB    ← Just written!
    (split into gate[4864] and up[4864])

OUTPUTS:
  - swiglu_out[1, 4864]  WRITE to DRAM     19 KB   ⚠️

COMPUTE:
  sigmoid(gate): 4864 × ~10 = 48K FLOPs
  gate × sigmoid(gate) × up: 4864 × 3 = 15K FLOPs

MEMORY: ~57 KB transferred
```

### Operation 9: Down Projection + Residual
```
INPUTS:
  - swiglu_out[1, 4864]  READ from DRAM    19 KB    ← Just written!
  - W2[896, 4864]        READ from DRAM    17.3 MB (weight)
  - b2[896]              READ from DRAM    3.5 KB  (bias)
  - residual1[1, 896]    READ from DRAM    3.5 KB  (residual)

OUTPUTS:
  - output[1, 896]       WRITE to DRAM     3.5 KB

COMPUTE: 896 × 4864 × 2 = 8.7M FLOPs
MEMORY: ~17.3 MB transferred
```

---

## Summary: One Layer

### Weight Memory (Read-Only)
| Weight | Shape | Size |
|--------|-------|------|
| ln1_gamma | [896] | 3.5 KB |
| Wq | [14, 64, 896] | 3.2 MB |
| Wk | [2, 64, 896] | 460 KB |
| Wv | [2, 64, 896] | 460 KB |
| Wo | [896, 896] | 3.2 MB |
| ln2_gamma | [896] | 3.5 KB |
| W1 (gate+up) | [9728, 896] | **34.6 MB** |
| W2 (down) | [896, 4864] | 17.3 MB |
| **Total** | | **~59 MB** |

### Activation Memory (Read+Write per token)
| Buffer | Shape | Size | Written By | Read By |
|--------|-------|------|------------|---------|
| ln1_out | [1, 896] | 3.5 KB | RMSNorm1 | QKV Proj |
| Q | [14, 1, 64] | 3.5 KB | QKV Proj | RoPE, Attention |
| K | [2, 1, 64] | 512 B | QKV Proj | RoPE, Attention |
| V | [2, 1, 64] | 512 B | QKV Proj | Attention |
| attn_out | [14, 1, 64] | 3.5 KB | Attention | Out Proj |
| residual1 | [1, 896] | 3.5 KB | Out Proj | RMSNorm2, Down Proj |
| ln2_out | [1, 896] | 3.5 KB | RMSNorm2 | Gate+Up Proj |
| fc1_out | [1, 9728] | 38 KB | Gate+Up | SwiGLU |
| swiglu_out | [1, 4864] | 19 KB | SwiGLU | Down Proj |
| output | [1, 896] | 3.5 KB | Down Proj | Next Layer |

### DRAM Round-Trips per Layer
```
Current: 9 write + 9 read = 18 DRAM accesses for activations
         + weight streaming (~59 MB)

With Fusion: 2 writes (after attention block, after MLP block)
```

---

## Bottleneck Analysis

### Your perf data shows:
- Most CPUs at 10% utilization
- One CPU at 100% (likely the thread doing GEMM)
- This means: **Memory-bound, not compute-bound**

### Why:
```
Compute per layer:  ~31M FLOPs
Memory per layer:   ~59 MB weights + ~150 KB activations

At 100 GB/s DRAM bandwidth:
  Time to transfer: 59 MB / 100 GB/s = 0.59 ms

At 100 GFLOPS compute:
  Time to compute: 31M / 100G = 0.31 ms

Ratio: Memory takes 2x longer than compute!
```

---

## Fusion Opportunities

### Fusion 1: RMSNorm + QKV Projection
```
Before: RMSNorm → DRAM → QKV
After:  RMSNorm → L2 cache → QKV

Saves: 2 × 3.5 KB DRAM traffic per layer
       More importantly: eliminates latency stall
```

### Fusion 2: RMSNorm + Gate+Up (MLP)
```
Before: RMSNorm → DRAM → Gate+Up
After:  RMSNorm → L2 cache → Gate+Up

Saves: 2 × 3.5 KB DRAM traffic
```

### Fusion 3: Gate+Up + SwiGLU + Down
```
Before: Gate+Up → DRAM (38KB) → SwiGLU → DRAM (19KB) → Down
After:  Gate+Up → L2 → SwiGLU → L2 → Down

Saves: 38 + 19 + 38 + 19 = 114 KB DRAM traffic
       This is the BIGGEST win for MLP!
```

### Expected Speedup with Full Fusion
```
Current DRAM traffic (activations): ~300 KB/layer
Fused DRAM traffic (activations):   ~14 KB/layer

For decode (T=1): 20x reduction in activation memory traffic
Practical speedup: 2-5x (weights still dominate)
```
