# Kernel Parallelism and Backends (Decoder + Vision Encoder)

This document captures how we think about:

- Which operations use GEMM vs custom kernels.
- Where parallelism lives (microkernel vs orchestrator).
- How this applies to decoder-only LLMs and vision encoders.

The goal: avoid nested OpenMP, keep kernels reusable, and make prefill/decode behavior explicit.

---

## 1. Decoder-Only Transformer: Kernel Vocabulary

Let:
- `T` = tokens (sequence length)
- `D` = model dimension
- `H` = attention heads
- `d = D / H` = per-head dimension

### 1.1 Core Ops vs Kernels

| Block        | Math / Shape                                      | Uses GEMM?         | Kernel Type                 |
|-------------|----------------------------------------------------|--------------------|-----------------------------|
| Q/K/V proj  | `X[T,D] · W_[q,k,v][D,D] + b[D]`                  | **Yes**            | GEMM + bias                 |
| Attn output | `X[T,D] · W_o[D,D] + b[D]`                        | **Yes**            | GEMM + bias                 |
| MLP FC1     | `X[T,D] · W1[D,4D] + b1[4D]`                      | **Yes**            | GEMM + bias                 |
| MLP FC2     | `H[T,4D] · W2[4D,D] + b2[D]`                      | **Yes**            | GEMM + bias                 |
| QKᵀ scores  | `S = Q[T,d] · K[T,d]^T → [T,T]`                   | Optional (GEMM)    | Specialized dot-product     |
| SV          | `O = P[T,T] · V[T,d] → [T,d]`                     | Optional (GEMM)    | Specialized matmul          |
| Softmax     | Row-wise softmax w/ causal mask on `S[h,i,:]`     | **No**             | Custom softmax kernel       |
| LayerNorm   | Per-token normalization over `D`                  | **No**             | Custom LN/RMSNorm kernel    |
| GELU/SwiGLU | Elementwise nonlinearity over `[T,4D]`            | **No**             | Activation kernel           |
| Residual    | `Y = X + F(X)`                                    | **No**             | Simple vector add kernel    |
| Logits      | `X[T,D] · W_vocab[D,V] + b[V]`                    | **Yes**            | GEMM + bias                 |

---

## 2. Parallelism Layers: Microkernels vs Orchestrators

We split responsibilities:

- **Microkernel**: Single-threaded, vectorized (AVX/AMX), math-only.
- **Orchestrator**: Decides *how* to parallelize (prefill vs decode, token vs feature/head) and uses OpenMP (or other threading).

### 2.1 Microkernels (No OpenMP Inside)

Examples:

- `gemm_core(...)` – small blocked GEMM microkernel (Mr × Nr) using AVX/AMX.
- `layernorm_forward_slice(...)` – LN over `D` features for a contiguous set of tokens.
- `layernorm_backward_slice(...)`
- `gelu_vec16_avx512(...)` – GELU over a contiguous span using AVX-512 (16 floats at a time).
- `softmax_row_kernel(...)` – stable softmax over one row (with mask).

**Properties:**

- No threading pragmas.
- Operate on contiguous `float*` spans (or small tiles).
- Designed to be called from:
  - Xeon/ARM orchestrators.
  - DSP/embedded environments.
  - Potential GPU wrappers where appropriate.

### 2.2 Orchestrators (Prefill vs Decode)

For decoder-only LLMs we care about two main execution regimes:

- **Prefill**: `T` is large (full context), batch may be 1–N.
- **Decode**: `T` is tiny (often 1), but `D` is large.

We design orchestrators that choose parallelism accordingly:

| Mode    | Typical Shape          | Parallelism Choice                  |
|---------|------------------------|-------------------------------------|
| Prefill | `T ≫ 1`, `D` moderate | **Token-parallel** over `T`         |
| Decode  | `T ≈ 1`, `D` large    | **Feature/head-parallel** over `D` or heads |

Examples (conceptual):

```c
// Prefill: token-parallel GELU over [T × D]
void gelu_prefill_token_parallel(float *data, int tokens, int d_model, int num_threads)
{
    size_t stride = (size_t)d_model;

    #pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < tokens; ++t) {
        float *row = data + (size_t)t * stride;
        gelu_vec16_avx512(row, (size_t)d_model);  // microkernel
    }
}

// Decode: feature-parallel GELU over [T × D] with T small
void gelu_decode_feature_parallel(float *data, int tokens, int d_model, int num_threads)
{
    size_t total = (size_t)tokens * (size_t)d_model;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();

        size_t chunk = (total + nth - 1) / nth;
        size_t start = tid * chunk;
        size_t end   = start + chunk;
        if (end > total) end = total;

        if (start < end) {
            gelu_vec16_avx512(data + start, end - start);  // microkernel
        }
    }
}
```

The same pattern applies to LayerNorm, softmax, and even attention matmuls:

- Prefill: block over tokens.
- Decode: block over features/heads.

**Key rule**: only orchestrators use OpenMP. Microkernels are thread-agnostic.

---

## 3. Vision Encoders: How They Fit

Vision encoders fall into two main categories:

1. **CNN-based encoders** (ResNet-style)
2. **Vision Transformers (ViT, Swin, etc.)**

### 3.1 ViT / Vision Transformers

Much of the math is identical to LLM decoders, just on image patches instead of tokens.

Typical pipeline:

1. **Patch embedding**:
   - Option A: Conv with kernel = patch size (e.g., 16×16 conv).
   - Option B: Explicit patch extraction (`im2col`-like) + linear projection.
2. **Add positional encodings**.
3. **Transformer encoder blocks**:
   - LayerNorm.
   - Multi-head self-attention (full attention, no causal mask).
   - MLP with GELU/SwiGLU.

So from a kernel perspective:

| Block             | Math / Shape                            | New Kernel Needed?       |
|------------------|------------------------------------------|--------------------------|
| Patch embedding  | Conv or matmul over patches             | **Conv or GEMM-based**   |
| Positional enc   | Adds or RoPE                            | Already covered (RoPE)   |
| Self-attention   | Same as decoder but no causal mask      | Reuse attention kernels  |
| LayerNorm/RMSNorm| Same                                    | Reuse LN/RMSNorm kernels |
| MLP              | Same (FC1 → GELU/SwiGLU → FC2)          | Reuse MLP kernels        |

The main *additional* kernel family to support CNN-style encoders or patch embedding is **2D convolution**.

### 3.2 Convolution Kernels

You have two realistic paths:

1. **Conv via GEMM**:
   - `im2col` (or equivalent) to flatten patches into `[N_patches, K]`.
   - GEMM `[N_patches, K] · [K, C_out]` with your existing GEMM kernels.
   - Good for first versions, reuses all GEMM infrastructure.

2. **Direct conv kernel**:
   - Microkernel that slides a kernel window over input feature maps.
   - Vectorized over channels and/or spatial positions.
   - Orchestrators:
     - Prefill-like: parallel over image tiles/batches.
     - Decode-like: if doing streaming vision, parallel over channels.

You can extend the same parallelism table:

| Domain   | Op type  | Microkernel          | Orchestrator Parallelism                 |
|----------|----------|----------------------|------------------------------------------|
| Decoder  | GEMM     | `gemm_core`          | token/feature/head (prefill/decode)      |
| Decoder  | GELU     | `gelu_vec16_avx512`  | token/feature (prefill/decode)           |
| Decoder  | LN/RMS   | `ln_slice_core`      | token/feature                             |
| Decoder  | Softmax  | `softmax_row_core`   | rows (tokens or heads)                   |
| Decoder  | Attn mat | `attn_qk_core`, etc. | tokens/heads                             |
| Vision   | Conv     | `conv2d_core`        | batch / tiles / channels                 |
| Vision   | Patch MLP| same as decoder MLP  | same as decoder                          |

---

## 4. Summary

1. **GEMM** powers:
   - Q/K/V projections, attention output, MLP FC1/FC2, logits, and optionally QKᵀ/SV.
2. **Custom kernels (no GEMM)** handle:
   - GELU/SwiGLU, LayerNorm/RMSNorm, softmax, RoPE, residual adds.
3. **Microkernels**:
   - Single-threaded, vectorized, cache-line aware.
4. **Orchestrators**:
   - Own multi-core parallelism (OpenMP or other).
   - Have explicit modes for decoder **prefill** and **decode**.
5. **Vision encoders**:
   - Reuse almost all decoder kernels (LN, MLP, attention).
   - Need one more family for conv/patch embedding, which can initially be implemented via GEMM (`im2col` + GEMM).

This structure keeps C-Kernel-Engine:

- Easy to validate against PyTorch (each microkernel has a clear math contract).
- Easy to port across backends (Xeon, ARM, DSP, etc.).
- Explicit about execution regimes (prefill vs decode, LLM vs vision).

