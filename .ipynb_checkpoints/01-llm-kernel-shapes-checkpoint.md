# LLM Kernel Shape Families

Let:
- `T` = context length (number of tokens)
- `D` = embedding dimension
- `H` = number of attention heads
- `d = D / H` = per-head dimension

The core workloads for a single-batch transformer can be covered by a small set of matrix shapes.

## 1. Linear `Wx + b` Blocks (Most FLOPs)

These cover QKV projections, attention output projections, and the MLP.

Given input `X[T, D]`:

- **QKV projection**  
  `X[T, D] · W_qkv[D, 3D] + b_qkv[3D] → [T, 3D]`

- **Attention output projection**  
  `X[T, D] · W_o[D, D] + b_o[D] → [T, D]`

- **MLP feed-forward 1**  
  `X[T, D] · W_1[D, 4D] + b_1[4D] → [T, 4D]`

- **MLP feed-forward 2**  
  `H[T, 4D] · W_2[4D, D] + b_2[D] → [T, D]`

All are of the general GEMM+bias form:

- `[T, D] · [D, k·D] + bias[k·D]` with `k ∈ {1,3,4}`

These are the primary targets for high-performance GEMM in C Kernel Engine.

## 2. Attention Score and Value GEMMs

Per head (with `d = D / H`):

- **Scores (QK^T)**  
  `Q[T, d] · K[T, d]^T → S[T, T]`  
  GEMM view: `[T, d] · [d, T] → [T, T]`

- **Apply scores to values (SV)**  
  `P[T, T] · V[T, d] → [T, d]`  
  GEMM view: `[T, T] · [T, d] → [T, d]`

These are still regular GEMMs, but now `K` or `N` are `d` or `T` instead of `D` or `k·D`. They deserve their own tuning because they stress different cache and blocking regimes.

## 3. Backward-Pass GEMMs (Training)

Training reuses the same families of shapes, plus transposed forms for weight gradients. Example:

- **Weight gradient for MLP 1**  
  `dW_1 = X^T[T, D]^T · dH[T, 4D] → [D, 4D]`

General pattern:

- `[D, T] · [T, k·D] → [D, k·D]` and similar variations.

If the engine supports training, kernels should be optimized for these transposed cases as well.

## 4. Elementwise / Reduction Kernels

Non-GEMM but important for correctness and latency:

- LayerNorm
- GELU / SwiGLU
- Softmax (with masking)
- Residual adds and simple pointwise ops

These are primarily bandwidth-bound and typically cheap compared to GEMMs, but they still benefit from:

- Cache-friendly layouts (contiguous tokens or features)
- SIMD vectorization
- Consistent memory alignment with the GEMM kernels

## Design Rule for C Kernel Engine

Instead of “support every BLAS shape,” C Kernel Engine **optimizes hard** for:

1. `[T, D] · [D, D/3D/4D] (+bias)` – projections and MLP.
2. `[T, d] · [d, T]` and `[T, T] · [T, d]` – attention scores and values.
3. Their transposed variants needed for backprop.
