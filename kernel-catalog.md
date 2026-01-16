# Kernel Catalog

This document lists the high-performance C kernels available in the engine. These kernels are designed to be "micro-libraries"—you can copy `src/kernels/rope_kernels.c` into your own project without taking the rest of the engine.

## Naming Convention

- **`_naive`**: Reference implementation, easy to read, slow.
- **`_parallel`**: OpenMP-accelerated for multi-core CPUs.
- **`_avx512`**: Explicit intrinsics for AVX-512 (x86_64).
- **`_head_major`**: Optimized memory layout where the "Head" dimension is outermost or stride-optimized.

## 1. Attention & RoPE

| Kernel | Source File | Description |
| :--- | :--- | :--- |
| `attention_forward_causal_head_major` | `attention_kernels.c` | Standard Scaled Dot-Product Attention (SDPA) with causal masking. Expects `[Head, Token, Dim]` layout. |
| `rope_forward` | `rope_kernels.c` | Rotary Positional Embeddings. Rotates query/key vectors in-place. |
| `causal_softmax_head_major` | `softmax_kernels.c` | Softmax applied to attention scores, masking out future tokens (causal mask). |

## 2. Elementwise & Activation

| Kernel | Source File | Description |
| :--- | :--- | :--- |
| `gelu_fast_inplace` | `gelu_kernels.c` | Gaussian Error Linear Unit. Uses the fast approximation (tanh). |
| `swiglu_forward` | `swiglu_kernels.c` | SwiGLU activation (Gated Linear Unit with Swish). Used in Llama/Mistral. Input size is `2 * dim`. |
| `sigmoid_forward` | `sigmoid_kernels.c` | Standard logistic sigmoid function. |

## 3. Normalization

| Kernel | Source File | Description |
| :--- | :--- | :--- |
| `rmsnorm_forward` | `rmsnorm_kernels.c` | Root Mean Square Normalization. Lighter than LayerNorm (no mean subtraction). |
| `layernorm_naive_serial` | `layernorm_kernels.c` | Standard LayerNorm. Subtracts mean, divides by variance. |

## 4. GEMM (Matrix Multiplication)

| Kernel | Source File | Description |
| :--- | :--- | :--- |
| `gemm_naive_parallel` | `gemm_kernels.c` | OpenMP-parallelized SGEMM ($C = A \times B$). |
| `gemm_avx512_parallel` | `gemm_kernels.c` | Hand-tuned AVX-512 implementation for max throughput on modern Intel/AMD chips. |

## Memory Layouts

Most kernels assume **Row-Major** contiguous memory unless specified otherwise.

- **Matrices**: `[Rows, Cols]`
- **Attention Tensors**: `[Heads, Tokens, Head_Dim]` (Head-Major) helps with cache locality during the attention loop.
