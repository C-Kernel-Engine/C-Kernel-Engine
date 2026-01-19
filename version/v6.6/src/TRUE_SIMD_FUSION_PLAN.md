# True SIMD Fusion Implementation Plan

## Overview

This document details the implementation plan for **true register-level SIMD fusion** across all three fusion blocks. The key insight is: **process OUTPUT cache-line by cache-line**, keeping intermediate values in YMM registers and never writing them to memory.

## The Problem with Current "Fused" Kernels

Current fusion is **fake fusion** - just function bundling:

```
┌─────────────────────────────────────────────────────────────────────┐
│ FAKE FUSION (what we had)                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  normed = rmsnorm(x)         ← writes 896 floats to L1/L2          │
│  q = gemv(Wq, normed)        ← reads normed from L1/L2             │
│  k = gemv(Wk, normed)        ← reads normed from L1/L2 (again!)    │
│  v = gemv(Wv, normed)        ← reads normed from L1/L2 (again!)    │
│                                                                     │
│  Memory traffic: 4x normed[] = 14KB extra cache pressure            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ TRUE SIMD FUSION (what we want)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  For each output cache line [j:j+8]:                               │
│    acc[0:7] = 0  (in YMM registers)                                │
│    For each input cache line [i:i+8]:                              │
│      normed = x[i:i+8] * rms_weight[i:i+8] * scale  ← IN YMM REG!  │
│      acc[k] += W[j+k, i:i+8] · normed              ← FMADD in reg  │
│    Store output[j:j+8]                                             │
│                                                                     │
│  Memory traffic: 0 bytes for normed[] - NEVER leaves registers!    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Files

### Block 1: RMSNorm + QKV Projection

| File | Function | Status |
|------|----------|--------|
| `src/kernels/fused/rmsnorm_qkv.c` | `rmsnorm_qkv_fp32_fused()` | ✅ Done (fake fusion) |
| `src/kernels/fused/rmsnorm_qkv.c` | `rmsnorm_qkv_fp32_fused_v2()` | ✅ Done (TRUE fusion) |
| `src/kernels/fused/rmsnorm_qkv.c` | `rmsnorm_qkv_q4k_fused_v2()` | 🔲 TODO |

### Block 2: Attention + MLP

| File | Function | Status |
|------|----------|--------|
| `src/kernels/fused/attention_mlp_fused.c` | `attention_mlp_fused_fp32()` | ✅ Done (scalar loops - SLOW) |
| `src/kernels/fused/attention_mlp_fused.c` | `attention_mlp_fused_v2()` | 🔲 TODO |
| `src/kernels/fused/attention_mlp_fused.c` | `attention_mlp_fused_q4k_v2()` | 🔲 TODO |

### Block 3: RMSNorm + lm_head

| File | Function | Status |
|------|----------|--------|
| `src/kernels/fused/rmsnorm_lmhead.c` | `rmsnorm_lmhead_fp32_fused()` | 🔲 TODO |
| `src/kernels/fused/rmsnorm_lmhead.c` | `rmsnorm_lmhead_q4k_fused()` | 🔲 TODO |

---

## Block 1: True SIMD Fusion Pattern (FP32)

Already implemented in `rmsnorm_qkv_fp32_fused_v2()`:

```c
void rmsnorm_qkv_fp32_fused_v2(
    const float *x,           /* [embed_dim] input */
    const float *rms_weight,  /* [embed_dim] RMSNorm gamma */
    const float *wq,          /* [q_dim, embed_dim] row-major */
    const float *wk,          /* [kv_dim, embed_dim] */
    const float *wv,          /* [kv_dim, embed_dim] */
    float *q_out, float *k_out, float *v_out,
    int embed_dim, int q_dim, int kv_dim, float eps
) {
    // Step 1: Compute RMS scale (full pass required)
    float scale = compute_rms_scale(x, embed_dim, eps);
    __m256 vscale = _mm256_set1_ps(scale);

    // Step 2: Process 8 outputs at a time (one cache line)
    for (int j = 0; j < q_dim; j += 8) {
        // 8 accumulators - ALL stay in YMM registers
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        // ... acc2-acc7

        // For each input cache line
        for (int i = 0; i + 7 < embed_dim; i += 8) {
            // Compute normed IN REGISTER - never stored!
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);

            // Use immediately for all 8 outputs
            __m256 w0 = _mm256_loadu_ps(wq + (j+0)*embed_dim + i);
            acc0 = _mm256_fmadd_ps(w0, normed, acc0);
            // ... repeat for acc1-acc7
        }

        // Horizontal sum and store output
        q_out[j+0] = hsum256_ps(acc0);
        // ... j+1 to j+7
    }
    // Same pattern for K and V projections
}
```

### Key Performance Characteristics

- **Register pressure**: 8 accumulators + 1 normed + ~2 temp = 11 YMM registers (16 available)
- **Memory access**: Input + weights loaded once, normed NEVER stored
- **Cache efficiency**: Each input cache line loaded once, used 8x
- **Instruction mix**: 1 MUL + 1 MUL + 8 FMADD per input chunk = 10 FMA-equivalent ops

---

## Block 1: True SIMD Fusion Pattern (Q4_K)

The Q4_K version is more complex because dequantization has block structure:

```c
void rmsnorm_qkv_q4k_fused_v2(
    const float *x,           /* [embed_dim] input */
    const float *rms_weight,  /* [embed_dim] RMSNorm gamma */
    const void *wq,           /* Q4_K quantized [q_dim, embed_dim] */
    const void *wk,           /* Q4_K quantized */
    const void *wv,           /* Q4_K quantized */
    float *q_out, float *k_out, float *v_out,
    int embed_dim, int q_dim, int kv_dim, float eps
) {
    float scale = compute_rms_scale(x, embed_dim, eps);
    __m256 vscale = _mm256_set1_ps(scale);

    // Q4_K has 256-element blocks (QK_K = 256)
    // Each block: 144 bytes → 256 floats
    const int QK_K = 256;
    const int blocks_per_row = embed_dim / QK_K;

    // Process 8 output rows at a time
    for (int j = 0; j < q_dim; j += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        // ... acc1-acc7

        // Process input in Q4_K block chunks
        for (int b = 0; b < blocks_per_row; b++) {
            int base_i = b * QK_K;

            // For each 32-element super-block within Q4_K block
            for (int sb = 0; sb < 8; sb++) {
                int i = base_i + sb * 32;

                // Compute normed for 32 elements (4x YMM)
                // Load 32 x[] values
                __m256 vx0 = _mm256_loadu_ps(x + i);
                __m256 vx1 = _mm256_loadu_ps(x + i + 8);
                __m256 vx2 = _mm256_loadu_ps(x + i + 16);
                __m256 vx3 = _mm256_loadu_ps(x + i + 24);

                __m256 vrms0 = _mm256_loadu_ps(rms_weight + i);
                __m256 vrms1 = _mm256_loadu_ps(rms_weight + i + 8);
                __m256 vrms2 = _mm256_loadu_ps(rms_weight + i + 16);
                __m256 vrms3 = _mm256_loadu_ps(rms_weight + i + 24);

                __m256 normed0 = _mm256_mul_ps(_mm256_mul_ps(vx0, vrms0), vscale);
                __m256 normed1 = _mm256_mul_ps(_mm256_mul_ps(vx1, vrms1), vscale);
                __m256 normed2 = _mm256_mul_ps(_mm256_mul_ps(vx2, vrms2), vscale);
                __m256 normed3 = _mm256_mul_ps(_mm256_mul_ps(vx3, vrms3), vscale);

                // For each of 8 output rows, dequant + FMADD
                for (int k = 0; k < 8 && j+k < q_dim; k++) {
                    // Get Q4_K block pointer for row j+k
                    const block_q4_K *blk = get_q4k_block(wq, j+k, b);

                    // Extract scale/min for this super-block
                    uint8_t sc = blk->scales[sb];
                    float d = GGML_FP16_TO_FP32(blk->d) * (sc & 0xF);
                    float m = GGML_FP16_TO_FP32(blk->dmin) * (sc >> 4);
                    __m256 vd = _mm256_set1_ps(d);
                    __m256 vm = _mm256_set1_ps(m);

                    // Dequant 32 weights and accumulate with normed
                    // This is the inner loop - needs careful optimization
                    __m256 w0 = dequant_q4k_8(blk->qs + sb*16 + 0, vd, vm);
                    __m256 w1 = dequant_q4k_8(blk->qs + sb*16 + 4, vd, vm);
                    __m256 w2 = dequant_q4k_8(blk->qs + sb*16 + 8, vd, vm);
                    __m256 w3 = dequant_q4k_8(blk->qs + sb*16 + 12, vd, vm);

                    acc[k] = _mm256_fmadd_ps(w0, normed0, acc[k]);
                    acc[k] = _mm256_fmadd_ps(w1, normed1, acc[k]);
                    acc[k] = _mm256_fmadd_ps(w2, normed2, acc[k]);
                    acc[k] = _mm256_fmadd_ps(w3, normed3, acc[k]);
                }
            }
        }

        // Store outputs
        for (int k = 0; k < 8 && j+k < q_dim; k++) {
            q_out[j+k] = hsum256_ps(acc[k]);
        }
    }
    // Repeat for K, V
}
```

---

## Block 2: Attention + MLP True Fusion

The current attention_mlp_fused is SLOWER than separate because it uses scalar loops. Here's the v2 plan:

### What Block 2 Fuses

```
attention_out = softmax(Q @ K.T / sqrt(d)) @ V   [already optimized]
proj_out = attention_out @ W_o                   [GEMV]
residual = hidden + proj_out                     [vector add]
normed = rmsnorm(residual)                       [IN REGISTER!]
gate = normed @ W_gate                           [GEMV]
up = normed @ W_up                               [GEMV]
act = gate * silu(up)                            [element-wise]
mlp_out = act @ W_down                           [GEMV]
output = residual + mlp_out                      [vector add]
```

### True Fusion Points

The key fusion is keeping `normed` in registers while computing gate and up:

```c
void attention_mlp_fused_v2(
    /* ... attention inputs ... */
    const float *w_gate,      /* [hidden_dim, embed_dim] */
    const float *w_up,        /* [hidden_dim, embed_dim] */
    const float *w_down,      /* [embed_dim, hidden_dim] */
    const float *rms_weight,
    float *output,
    /* ... dimensions ... */
) {
    // 1. Attention (use existing optimized kernel)
    attention_forward(...);

    // 2. Output projection: proj = attn_out @ W_o
    gemv_optimized(w_o, attn_out, proj_out, ...);

    // 3. First residual: residual = hidden + proj_out
    vec_add(hidden, proj_out, residual, embed_dim);

    // 4. TRUE FUSED: RMSNorm + Gate/Up projection
    // Process 8 gate + 8 up outputs at a time
    float scale = compute_rms_scale(residual, embed_dim, eps);
    __m256 vscale = _mm256_set1_ps(scale);

    for (int j = 0; j < hidden_dim; j += 8) {
        __m256 gate_acc[8] = {0};
        __m256 up_acc[8] = {0};

        for (int i = 0; i + 7 < embed_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(residual + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);
            // normed NEVER stored!

            for (int k = 0; k < 8; k++) {
                __m256 wg = _mm256_loadu_ps(w_gate + (j+k)*embed_dim + i);
                __m256 wu = _mm256_loadu_ps(w_up + (j+k)*embed_dim + i);
                gate_acc[k] = _mm256_fmadd_ps(wg, normed, gate_acc[k]);
                up_acc[k] = _mm256_fmadd_ps(wu, normed, up_acc[k]);
            }
        }

        // Store gate and up, compute SiLU-gated activation
        for (int k = 0; k < 8; k++) {
            float g = hsum256_ps(gate_acc[k]);
            float u = hsum256_ps(up_acc[k]);
            intermediate[j+k] = g * silu(u);  // Fused gate * silu(up)
        }
    }

    // 5. Down projection: mlp = intermediate @ W_down
    gemv_optimized(w_down, intermediate, mlp_out, ...);

    // 6. Final residual: output = residual + mlp_out
    vec_add(residual, mlp_out, output, embed_dim);
}
```

---

## Block 3: RMSNorm + lm_head

Simple fusion - similar to Block 1 Q projection only:

```c
void rmsnorm_lmhead_fp32_fused(
    const float *hidden,       /* [embed_dim] last hidden state */
    const float *rms_weight,   /* [embed_dim] final RMSNorm */
    const float *w_lm_head,    /* [vocab_size, embed_dim] */
    float *logits,             /* [vocab_size] output */
    int embed_dim,
    int vocab_size,
    float eps
) {
    float scale = compute_rms_scale(hidden, embed_dim, eps);
    __m256 vscale = _mm256_set1_ps(scale);

    // Process 8 vocab outputs at a time
    for (int j = 0; j < vocab_size; j += 8) {
        __m256 acc[8] = {0};

        for (int i = 0; i + 7 < embed_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(hidden + i);
            __m256 vrms = _mm256_loadu_ps(rms_weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vscale);

            for (int k = 0; k < 8 && j+k < vocab_size; k++) {
                __m256 w = _mm256_loadu_ps(w_lm_head + (j+k)*embed_dim + i);
                acc[k] = _mm256_fmadd_ps(w, normed, acc[k]);
            }
        }

        for (int k = 0; k < 8 && j+k < vocab_size; k++) {
            logits[j+k] = hsum256_ps(acc[k]);
        }
    }
}
```

---

## Unit Test Strategy

### Test File: `unittest/test_fusion_benchmark.py`

```python
"""
Benchmark true SIMD fusion vs fake fusion vs separate kernels.
"""
import ctypes
import numpy as np
import time

lib = ctypes.CDLL("build/libckernel_engine.so")

# Setup function signatures
lib.rmsnorm_qkv_fp32_fused.argtypes = [...]      # v1 fake fusion
lib.rmsnorm_qkv_fp32_fused_v2.argtypes = [...]   # v2 true fusion
lib.rmsnorm_qkv_separate_fp32.argtypes = [...]   # separate baseline

def benchmark(fn, iterations=1000, warmup=100):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations * 1e6  # microseconds

def test_rmsnorm_qkv_fusion():
    # Qwen2-0.5B dimensions
    embed_dim = 896
    q_dim = 896      # num_heads * head_dim = 14 * 64
    kv_dim = 128     # num_kv_heads * head_dim = 2 * 64

    # Allocate aligned arrays
    x = np.random.randn(embed_dim).astype(np.float32)
    rms_w = np.random.randn(embed_dim).astype(np.float32)
    wq = np.random.randn(q_dim, embed_dim).astype(np.float32)
    wk = np.random.randn(kv_dim, embed_dim).astype(np.float32)
    wv = np.random.randn(kv_dim, embed_dim).astype(np.float32)

    q_out = np.zeros(q_dim, dtype=np.float32)
    k_out = np.zeros(kv_dim, dtype=np.float32)
    v_out = np.zeros(kv_dim, dtype=np.float32)
    normed = np.zeros(embed_dim, dtype=np.float32)

    # Benchmark separate
    def run_separate():
        lib.rmsnorm_qkv_separate_fp32(...)
    t_separate = benchmark(run_separate)

    # Benchmark v1 (fake fusion)
    def run_v1():
        lib.rmsnorm_qkv_fp32_fused(...)
    t_v1 = benchmark(run_v1)

    # Benchmark v2 (true SIMD fusion)
    def run_v2():
        lib.rmsnorm_qkv_fp32_fused_v2(...)
    t_v2 = benchmark(run_v2)

    print(f"Separate:    {t_separate:.1f} µs")
    print(f"Fused v1:    {t_v1:.1f} µs (speedup: {t_separate/t_v1:.2f}x)")
    print(f"Fused v2:    {t_v2:.1f} µs (speedup: {t_separate/t_v2:.2f}x)")
    print(f"v2 vs v1:    {t_v1/t_v2:.2f}x faster")

    # Correctness check
    lib.rmsnorm_qkv_fp32_fused_v2(...)
    q_v2 = q_out.copy()
    lib.rmsnorm_qkv_separate_fp32(...)
    q_ref = q_out.copy()

    assert np.allclose(q_v2, q_ref, rtol=1e-5), "Mismatch!"
```

### Expected Results

| Kernel | Time (µs) | Speedup | Notes |
|--------|-----------|---------|-------|
| Separate (baseline) | ~50 | 1.00x | 4x normed reads from cache |
| Fused v1 (fake) | ~44 | 1.14x | Function bundling only |
| **Fused v2 (true)** | ~30 | **1.67x** | normed never leaves registers |

---

## Constraints

1. **NO MALLOC/FREE IN KERNELS**: All memory from stack or caller-provided buffers
2. **Stack arrays max 16KB**: For `float arr[4096]` on stack, safe limit
3. **Caller provides scratch**: For larger intermediates, caller allocates

---

## Implementation Order

1. ✅ `rmsnorm_qkv_fp32_fused_v2()` - TRUE SIMD fusion for Block 1 FP32
2. 🔲 `unittest/test_fusion_benchmark.py` - Benchmark v1 vs v2 vs separate
3. 🔲 `rmsnorm_qkv_q4k_fused_v2()` - TRUE SIMD fusion for Block 1 Q4_K
4. 🔲 `attention_mlp_fused_v2()` - TRUE SIMD fusion for Block 2
5. 🔲 `rmsnorm_lmhead_fp32_fused()` - Block 3 implementation

---

## Success Criteria

| Block | Current | Target | Metric |
|-------|---------|--------|--------|
| Block 1 (RMSNorm+QKV) | ~1.0x | ~1.0x | **NO BENEFIT** - normed (3.5KB) fits in L1 |
| Block 2 (Attn+MLP) | 0.60x (SLOW!) | ≥1.3x | Fix scalar loops, fuse gate/up |
| Block 3 (RMSNorm+lmhead) | N/A | ~1.0x | normed fits in L1 |
| **Full decode** | TBD | **≥1.3x** | Block 2 dominates |

---

## Critical Finding: When Fusion Helps

After benchmarking, we discovered that **fusion only helps when intermediate buffers exceed L1/L2 cache**:

### Block 1 Analysis (RMSNorm + QKV)

For Qwen2-0.5B (embed_dim=896):
- `normed` buffer: 896 × 4 bytes = **3.5 KB** (easily fits in L1)
- Weight matrices: 4 MB+ (dominate memory traffic regardless)

**Result**: Fusion provides **~0% speedup** because:
1. normed fits in L1 - no cache spill benefit
2. Fused versions recompute normed per output row = MORE operations
3. Weight loading dominates memory traffic

**Recommendation**: Skip Block 1 fusion for models with embed_dim < 2048.

### Block 2 Analysis (Attention + MLP)

For Qwen2-0.5B (intermediate_dim=4864):
- `gate_out`: 4864 × 4 = **19.5 KB** (L1 pressure!)
- `up_out`: 4864 × 4 = **19.5 KB** (L1 pressure!)
- Combined: **39 KB** (exceeds typical 32KB L1D)

**Result**: Fusion SHOULD help here by:
1. Fusing gate + up projection (compute together, one pass through normed)
2. Fusing SwiGLU + down projection (no intermediate buffer store)

**BUT**: Current implementation uses scalar loops for GEMV = SLOW!

### The Real Problem

The current "fused" Block 2 implementation (`attention_mlp_fused_fp32`) uses:
- **Scalar loops** for output projection (lines 255-262)
- **Scalar loops** for gate projection (lines 293-300)
- **Scalar loops** for up projection (lines 303-310)
- **Scalar loops** for down projection (lines 338-345)

The only SIMD is for RMSNorm and SwiGLU activation.

The "separate" baseline would use **optimized GEMV kernels** that are:
- Heavily SIMDized (AVX2/AVX-512)
- Cache-optimized with blocking
- Potentially MKL-backed

**This is why the fused version is 0.60x (SLOWER)**!
