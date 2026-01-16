# GEMM in Transformers: The 90% Rule

## Executive Summary

**~85-95% of transformer FLOPs go through GEMM**. Everything else (GELU, LayerNorm, softmax, RoPE) is cheap in comparison.

---

## GEMM Operations in a Transformer Layer

### 1. Attention Block (4 GEMMs)

```c
// Input: X[T, D]  (T tokens, D hidden dimension)

// Q projection: X[T, D] · W_q[D, D] = Q[T, D]
gemm(X, W_q, Q, T, D, D);

// K projection: X[T, D] · W_k[D, D] = K[T, D]
gemm(X, W_k, K, T, D, D);

// V projection: X[T, D] · W_v[D, D] = V[T, D]
gemm(X, W_v, V, T, D, D);

// Output projection: X[T, D] · W_o[D, D] = O[T, D]
gemm(X_attn, W_o, O, T, D, D);
```

**Total FLOPs (Attention Projections):** 4 × (2 × T × D × D) = 8TD²

### 2. Attention Scores (2 more GEMMs)

These can be implemented as specialized loops OR as GEMMs:

```c
// Option A: Specialized loop (what you have now)
for (size_t h = 0; h < num_heads; h++) {
    // Q[T, d_head] · K^T[d_head, T] = scores[T, T]
    attention_scores(Q_h, K_h, scores, T, d_head);

    softmax(scores, T, T);

    // scores[T, T] · V[T, d_head] = out[T, d_head]
    attention_output(scores, V_h, out_h, T, d_head);
}

// Option B: Call GEMM (more general)
for (size_t h = 0; h < num_heads; h++) {
    // Q · K^T
    gemm(Q_h, K_h, scores, T, T, d_head, /* transB= */ true);

    softmax(scores, T, T);

    // scores · V
    gemm(scores, V_h, out_h, T, d_head, T);
}
```

**FLOPs per head:**
- Q·K^T: 2 × T × T × d_head
- scores·V: 2 × T × T × d_head
- **Total per head:** 4T²d_head

**Total (all heads):** num_heads × 4T²d_head = 4T²D (where D = num_heads × d_head)

### 3. MLP Block (2 GEMMs)

```c
// FC1 (expand): X[T, D] · W1[D, 4D] = H[T, 4D]
gemm(X, W1, H, T, 4*D, D);

gelu(H, T * 4*D);

// FC2 (project): H[T, 4D] · W2[4D, D] = O[T, D]
gemm(H, W2, O, T, D, 4*D);
```

**Total FLOPs (MLP):** 2 × T × D × 4D + 2 × T × 4D × D = 16TD²

### 4. Output/LM Head (1 GEMM)

```c
// Final layer: X[T, D] · W_vocab[D, V] = logits[T, V]
gemm(X, W_vocab, logits, T, V, D);
```

**Total FLOPs (LM Head):** 2 × T × V × D

---

## Total FLOPs Breakdown (GPT-2 124M, 12 layers)

**Assumptions:**
- T = 512 tokens
- D = 768 (hidden dimension)
- V = 50257 (vocab size)
- num_heads = 12
- d_head = 64

### Per-Layer FLOPs

| Component | Operation | FLOPs | % of Layer |
|-----------|-----------|-------|------------|
| **Attention Projections** | 4 GEMMs (QKV + O) | 8TD² = 8×512×768² ≈ 2.42 GFLOP | **60%** |
| **Attention Scores** | Q·K^T, scores·V | 4T²D = 4×512²×768 ≈ 0.40 GFLOP | **10%** |
| **MLP** | 2 GEMMs (up + down) | 16TD² = 16×512×768² ≈ 4.83 GFLOP | **30%** |
| **LayerNorm** | 2× (before attn, before MLP) | Negligible | <1% |
| **GELU** | 1× (in MLP) | Negligible | <1% |
| **Softmax** | 1× (in attention) | Negligible | <1% |
| **Residual Adds** | 2× | Negligible | <1% |

**Total per layer:** ~7.65 GFLOP
**GEMM:** ~7.25 GFLOP (95%)
**Non-GEMM:** ~0.40 GFLOP (5%)

### Full Model (12 layers + LM head)

| Component | FLOPs |
|-----------|-------|
| **12 Layers (GEMM)** | 12 × 7.25 = 87 GFLOP |
| **12 Layers (non-GEMM)** | 12 × 0.40 = 4.8 GFLOP |
| **LM Head (GEMM)** | 2 × 512 × 50257 × 768 ≈ 39.6 GFLOP |
| **Total GEMM** | 126.6 GFLOP (93%) |
| **Total Non-GEMM** | 4.8 GFLOP (7%) |
| **Grand Total** | 131.4 GFLOP |

**Conclusion:** GEMM is 93% of compute. Optimize GEMM first!

---

## Prefill vs Decode: Shape Changes

### Prefill (M=512)

All matrices have M=512 (many tokens):

```c
// Attention projections
gemm(X[512,768], W_q[768,768], Q[512,768], 512, 768, 768);  // 1.2 GFLOP
gemm(X[512,768], W_k[768,768], K[512,768], 512, 768, 768);  // 1.2 GFLOP
gemm(X[512,768], W_v[768,768], V[512,768], 512, 768, 768);  // 1.2 GFLOP

// Attention scores (per head, 12 heads)
gemm(Q[512,64], K^T[64,512], scores[512,512], 512, 512, 64);  // 33.5 MFLOP × 12 = 0.40 GFLOP

// MLP
gemm(X[512,768], W1[768,3072], H[512,3072], 512, 3072, 768);  // 2.4 GFLOP
gemm(H[512,3072], W2[3072,768], O[512,768], 512, 768, 3072);  // 2.4 GFLOP
```

**Characteristics:**
- Large M (512)
- GEMM is compute-bound (good cache reuse)
- Parallelize over M (token-parallel)

### Decode (M=1)

All matrices have M=1 (single token):

```c
// Attention projections
gemm(X[1,768], W_q[768,768], Q[1,768], 1, 768, 768);  // 2.4 MFLOP
gemm(X[1,768], W_k[768,768], K[1,768], 1, 768, 768);  // 2.4 MFLOP
gemm(X[1,768], W_v[768,768], V[1,768], 1, 768, 768);  // 2.4 MFLOP

// Attention scores (per head, T_cache=512 KV tokens)
gemm(Q[1,64], K_cache^T[64,512], scores[1,512], 1, 512, 64);  // 0.06 MFLOP × 12 = 0.7 MFLOP

// MLP
gemm(X[1,768], W1[768,3072], H[1,3072], 1, 3072, 768);  // 4.7 MFLOP
gemm(H[1,3072], W2[3072,768], O[1,768], 1, 768, 3072);  // 4.7 MFLOP
```

**Characteristics:**
- Tiny M (1)
- GEMM is memory-bound (poor cache reuse)
- Parallelize over N or K (feature-parallel)
- Need specialized small-M kernels

---

## Should You Implement Attention Scores as GEMM?

### Option A: Specialized Loops (Current)

**Pros:**
- More control over memory layout (head-major vs token-major)
- Easier to fuse with softmax
- Simpler for small T (decode)

**Cons:**
- Duplicate code vs your GEMM implementation
- Potentially slower than optimized GEMM for large T

### Option B: Call GEMM (General)

**Pros:**
- Reuse your optimized GEMM kernel
- Better for large T (prefill, long context)
- Less code to maintain

**Cons:**
- Need to handle transpose
- Harder to fuse with softmax
- May have overhead for small T

### Recommendation: Hybrid

```c
void attention_scores_prefill(/* ... */) {
    if (T >= 512) {
        // Large T: use GEMM (better performance)
        for (size_t h = 0; h < num_heads; h++) {
            gemm(Q_h, K_h, scores, T, T, d_head, /* transB= */ true);
            softmax(scores, T, T);
            gemm(scores, V_h, out_h, T, d_head, T);
        }
    } else {
        // Small T: use specialized loop
        attention_scores_loop(Q, K, V, scores, out, T, num_heads, d_head);
    }
}

void attention_scores_decode(/* ... */) {
    // M=1: always use specialized loop (GEMM overhead too high)
    for (size_t h = 0; h < num_heads; h++) {
        // Q[1,64] · K_cache^T[64,T_cache] = scores[1,T_cache]
        dot_product_vec(Q_h, K_cache_h, scores, d_head, T_cache);

        softmax(scores, 1, T_cache);

        // scores[1,T_cache] · V_cache[T_cache,64] = out[1,64]
        weighted_sum_vec(scores, V_cache_h, out_h, T_cache, d_head);
    }
}
```

---

## Takeaway for C-Kernel-Engine

**Your GEMM kernels will be called in these patterns:**

### Prefill (M=512, N=768-3072, K=768)
```c
// Attention: 4 calls per layer × 12 layers = 48 GEMM calls
// MLP: 2 calls per layer × 12 layers = 24 GEMM calls
// LM head: 1 call
// Total: 73 GEMM calls per forward pass
```

### Decode (M=1, N=768-3072, K=768)
```c
// Attention: 4 calls per layer × 12 layers = 48 GEMM calls
// MLP: 2 calls per layer × 12 layers = 24 GEMM calls
// LM head: 1 call
// Total: 73 GEMM calls per forward pass
```

**Key insight:** Same number of calls, but vastly different shapes (M=512 vs M=1).

You need:
1. **General GEMM** for prefill (M×N×K)
2. **Small-M GEMM** for decode (1×N×K)
3. Possibly **specialized attention kernels** for very small T

---

## FLOPs vs Memory Bandwidth

### Prefill (M=512)

**Arithmetic Intensity (AI):**
```
FLOPs = 2 × 512 × 768 × 768 = 1.2 GFLOP
Bytes = (512×768 + 768×768 + 512×768) × 4 = 4.7 MB
AI = 1.2 GFLOP / 4.7 MB = 255 FLOPs/byte
```

**Ridge point (Xeon):** ~20 FLOPs/byte

**Conclusion:** 255 >> 20 → **Compute-bound** (good!)

### Decode (M=1)

**Arithmetic Intensity:**
```
FLOPs = 2 × 1 × 768 × 768 = 2.4 MFLOP
Bytes = (1×768 + 768×768 + 1×768) × 4 = 2.4 MB
AI = 2.4 MFLOP / 2.4 MB = 1 FLOPs/byte
```

**Conclusion:** 1 << 20 → **Memory-bound** (bad!)

This is why small-M kernels need different optimization strategies:
- Minimize memory traffic
- Parallelize over N or K (not M)
- Use vector loads/broadcasts efficiently

---

## Summary: Where to Focus Optimization

| Component | % of FLOPs | Optimization Priority |
|-----------|------------|----------------------|
| **GEMM (projections)** | 85-90% | **CRITICAL** |
| **GEMM (attention)** | 3-5% | Medium (for long context) |
| **GELU** | 1-2% | Low (already fast) |
| **LayerNorm** | 1-2% | Low |
| **Softmax** | <1% | Low |
| **RoPE** | <1% | Low |
| **Residual adds** | <0.5% | Very low |

**80/20 Rule:** Optimize GEMM projections (QKV, MLP) first. Everything else combined is <10% of compute.

---

**Last Updated**: 2025-11-23
**Status**: Analysis complete, ready for implementation planning
