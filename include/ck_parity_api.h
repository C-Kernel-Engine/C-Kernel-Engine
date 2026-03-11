/**
 * @file ck_parity_api.h
 * @brief C-Kernel-Engine Parity Testing API
 *
 * Exposes individual CK kernels for parity testing against llama.cpp/ggml.
 * This API mirrors the test-kernel-parity.cpp interface in llama.cpp.
 *
 * Usage:
 *   1. Build as shared library: libck_parity.so
 *   2. Load from Python using ctypes
 *   3. Call functions with matching signatures to test-kernel-parity.cpp
 */

#ifndef CK_PARITY_API_H
#define CK_PARITY_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants (must match llama.cpp/ggml)
 * ============================================================================ */

#define CK_QK_K 256      /* Elements per K-quant super-block */
#define CK_QK4_0 32      /* Elements per Q4_0 block */
#define CK_QK8_0 32      /* Elements per Q8_0 block */

/* Block sizes in bytes */
#define CK_BLOCK_Q4_K_SIZE 144
#define CK_BLOCK_Q6_K_SIZE 210
#define CK_BLOCK_Q8_K_SIZE 292
#define CK_BLOCK_Q4_0_SIZE 18

/* ============================================================================
 * Dequantization Tests
 * ============================================================================ */

/**
 * @brief Dequantize Q4_K data to FP32
 * @param src Input Q4_K blocks
 * @param dst Output FP32 values
 * @param n Number of elements (must be multiple of 256)
 */
void ck_test_dequant_q4_k(const void *src, float *dst, int n);

/**
 * @brief Dequantize Q6_K data to FP32
 */
void ck_test_dequant_q6_k(const void *src, float *dst, int n);

/**
 * @brief Dequantize Q4_0 data to FP32
 */
void ck_test_dequant_q4_0(const void *src, float *dst, int n);

/* ============================================================================
 * Quantization Tests
 * ============================================================================ */

/**
 * @brief Quantize FP32 to Q8_K (for activations)
 * @param src Input FP32 values
 * @param dst Output Q8_K blocks
 * @param n Number of elements (must be multiple of 256)
 */
void ck_test_quantize_q8_k(const float *src, void *dst, int n);

/* ============================================================================
 * GEMV (Matrix-Vector) Tests
 * ============================================================================ */

/**
 * @brief Q4_K GEMV - dot product of quantized weights and FP32 input
 *
 * Internally quantizes input to Q8_K, then computes dot product.
 *
 * @param weight_q4k Q4_K quantized weights [cols]
 * @param input_f32 FP32 input vector [cols]
 * @param output Output scalar [1]
 * @param cols Number of columns (must be multiple of 256)
 */
void ck_test_gemv_q4_k(const void *weight_q4k,
                       const float *input_f32,
                       float *output,
                       int cols);

/**
 * @brief Q6_K GEMV
 */
void ck_test_gemv_q6_k(const void *weight_q6k,
                       const float *input_f32,
                       float *output,
                       int cols);

/**
 * @brief Q5_0 GEMV - matrix-vector multiply with Q5_0 weights
 *
 * @param weight_q5_0 Q5_0 quantized weights [rows * cols]
 * @param input_f32 FP32 input vector [cols]
 * @param output FP32 output vector [rows]
 * @param rows Number of output rows
 * @param cols Number of columns (must be multiple of 32)
 */
void ck_test_gemv_q5_0(const void *weight_q5_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols);

/**
 * @brief Q8_0 GEMV - matrix-vector multiply with Q8_0 weights
 *
 * @param weight_q8_0 Q8_0 quantized weights [rows * cols]
 * @param input_f32 FP32 input vector [cols]
 * @param output FP32 output vector [rows]
 * @param rows Number of output rows
 * @param cols Number of columns (must be multiple of 32)
 */
void ck_test_gemv_q8_0(const void *weight_q8_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols);

/**
 * @brief Q5_0 x Q8_0 quantized GEMV - matches llama.cpp's approach
 *
 * This version quantizes the input to Q8_0 first, then uses integer
 * dot products (like llama.cpp does). Use this for parity testing.
 *
 * @param weight_q5_0 Q5_0 quantized weights [rows * cols]
 * @param input_f32 FP32 input vector [cols] - will be quantized to Q8_0
 * @param output FP32 output vector [rows]
 * @param rows Number of output rows
 * @param cols Number of columns (must be multiple of 32)
 */
void ck_test_gemv_q5_0_q8_0(const void *weight_q5_0,
                             const float *input_f32,
                             float *output,
                             int rows, int cols);

/**
 * @brief Q8_0 x Q8_0 quantized GEMV - matches llama.cpp's approach
 *
 * This version quantizes the input to Q8_0 first, then uses integer
 * dot products (like llama.cpp does). Use this for parity testing.
 *
 * @param weight_q8_0 Q8_0 quantized weights [rows * cols]
 * @param input_f32 FP32 input vector [cols] - will be quantized to Q8_0
 * @param output FP32 output vector [rows]
 * @param rows Number of output rows
 * @param cols Number of columns (must be multiple of 32)
 */
void ck_test_gemv_q8_0_q8_0(const void *weight_q8_0,
                             const float *input_f32,
                             float *output,
                             int rows, int cols);

/* ============================================================================
 * Direct Vec Dot Tests (pre-quantized inputs, no FP32 conversion)
 * ============================================================================ */

/**
 * @brief Direct Q5_0 x Q8_0 dot product (takes pre-quantized Q8_0 input)
 *
 * This is a "direct" test that bypasses FP32-to-Q8_0 conversion.
 * Useful for isolating kernel bugs from quantization bugs.
 *
 * @param weight_q5_0 Q5_0 quantized weights [cols]
 * @param input_q8_0 Q8_0 quantized input [cols] (pre-quantized!)
 * @param output Output scalar [1]
 * @param cols Number of elements (must be multiple of 32)
 */
void ck_test_vec_dot_q5_0_q8_0(const void *weight_q5_0,
                                const void *input_q8_0,
                                float *output,
                                int cols);

/**
 * @brief Direct Q8_0 x Q8_0 dot product (takes pre-quantized Q8_0 input)
 *
 * @param weight_q8_0 Q8_0 quantized weights [cols]
 * @param input_q8_0 Q8_0 quantized input [cols] (pre-quantized!)
 * @param output Output scalar [1]
 * @param cols Number of elements (must be multiple of 32)
 */
void ck_test_vec_dot_q8_0_q8_0(const void *weight_q8_0,
                                const void *input_q8_0,
                                float *output,
                                int cols);

/* ============================================================================
 * GEMM (Matrix-Matrix) Tests
 * ============================================================================ */

/**
 * @brief Q4_K GEMM - batched matrix multiply with quantized weights
 *
 * Computes: output[t,r] = sum_k(weight[r,k] * input[t,k])
 *
 * @param weight_q4k Q4_K quantized weights [rows, cols]
 * @param input_f32 FP32 input [n_tokens, cols]
 * @param output FP32 output [n_tokens, rows]
 * @param rows Number of output rows
 * @param cols Number of columns (must be multiple of 256)
 * @param n_tokens Batch size
 */
void ck_test_gemm_q4_k(const void *weight_q4k,
                       const float *input_f32,
                       float *output,
                       int rows, int cols, int n_tokens);

/**
 * @brief Q6_K GEMM - batched matrix multiply with Q6_K weights
 *
 * Computes: output[t,r] = sum_k(weight[r,k] * input[t,k])
 *
 * @param weight_q6k Q6_K quantized weights [rows, cols]
 * @param input_f32 FP32 input [n_tokens, cols]
 * @param output FP32 output [n_tokens, rows]
 * @param rows Number of output rows
 * @param cols Number of columns (must be multiple of 256)
 * @param n_tokens Batch size
 */
void ck_test_gemm_q6_k(const void *weight_q6k,
                       const float *input_f32,
                       float *output,
                       int rows, int cols, int n_tokens);

/**
 * @brief Q5_0 GEMM - batched matrix multiply with Q5_0 weights (32-element blocks)
 *
 * Computes: output[t,r] = sum_k(weight[r,k] * input[t,k])
 *
 * @param weight_q5_0 Q5_0 quantized weights [rows, cols]
 * @param input_f32 FP32 input [n_tokens, cols]
 * @param output FP32 output [n_tokens, rows]
 * @param rows Number of output rows
 * @param cols Number of columns (must be multiple of 32)
 * @param n_tokens Batch size
 */
void ck_test_gemm_q5_0(const void *weight_q5_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols, int n_tokens);

/**
 * @brief Q8_0 GEMM - batched matrix multiply with Q8_0 weights (32-element blocks)
 *
 * Computes: output[t,r] = sum_k(weight[r,k] * input[t,k])
 *
 * @param weight_q8_0 Q8_0 quantized weights [rows, cols]
 * @param input_f32 FP32 input [n_tokens, cols]
 * @param output FP32 output [n_tokens, rows]
 * @param rows Number of output rows
 * @param cols Number of columns (must be multiple of 32)
 * @param n_tokens Batch size
 */
void ck_test_gemm_q8_0(const void *weight_q8_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols, int n_tokens);

/* ============================================================================
 * Activation Kernels
 * ============================================================================ */

/**
 * @brief RMSNorm
 *
 * Computes: output = (input / rms(input)) * weight
 * where rms(x) = sqrt(mean(x^2) + eps)
 *
 * @param input Input tensor [n_tokens, dim]
 * @param weight Normalization weights [dim]
 * @param output Output tensor [n_tokens, dim]
 * @param n_tokens Number of tokens
 * @param dim Hidden dimension
 * @param eps Epsilon for numerical stability
 */
void ck_test_rmsnorm(const float *input,
                     const float *weight,
                     float *output,
                     int n_tokens, int dim, float eps);

/**
 * @brief RoPE (Rotary Position Embedding)
 *
 * Applies rotary position embeddings to Q and K tensors.
 *
 * NOTE: CK uses rotate-half format (split first/second halves)
 *       while some implementations use interleaved format.
 *       The test harness should account for this.
 *
 * @param q Query tensor [n_tokens, n_heads * head_dim], modified in-place
 * @param k Key tensor [n_tokens, n_heads_kv * head_dim], modified in-place
 * @param n_tokens Number of tokens
 * @param n_heads Number of query heads
 * @param n_heads_kv Number of key/value heads
 * @param head_dim Dimension per head
 * @param pos_offset Starting position for RoPE
 * @param theta RoPE base frequency (typically 10000.0)
 */
void ck_test_rope(float *q, float *k,
                  int n_tokens, int n_heads, int n_heads_kv, int head_dim,
                  int pos_offset, float theta);

/**
 * @brief RoPE with interleaved format (for llama.cpp compatibility)
 *
 * Uses interleaved format: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
 */
void ck_test_rope_interleaved(float *q, float *k,
                              int n_tokens, int n_heads, int n_heads_kv, int head_dim,
                              int pos_offset, float theta);

/**
 * @brief SwiGLU activation
 *
 * Computes: output = SiLU(gate) * up
 * where SiLU(x) = x * sigmoid(x)
 *
 * @param gate_up Input tensor [n_tokens, 2 * intermediate_dim]
 *               Layout: [gate_0..gate_D-1, up_0..up_D-1] per token
 * @param output Output tensor [n_tokens, intermediate_dim]
 * @param n_tokens Number of tokens
 * @param intermediate_dim Intermediate dimension
 */
void ck_test_swiglu(const float *gate_up,
                    float *output,
                    int n_tokens, int intermediate_dim);

/**
 * @brief Softmax (simple, non-causal)
 *
 * Computes: output[i] = exp(input[i]) / sum(exp(input))
 *
 * @param input Input tensor [n]
 * @param output Output tensor [n]
 * @param n Number of elements
 */
void ck_test_softmax(const float *input, float *output, int n);

/**
 * @brief Gated DeltaNet autoregressive update.
 *
 * Layout:
 *   q, k, v   [num_heads, state_dim]
 *   g, beta   [num_heads]
 *   state_*   [num_heads, state_dim, state_dim] row-major per head
 *   out       [num_heads, state_dim]
 *
 * This mirrors the single-token recurrent update used by qwen3next in
 * llama.cpp after projections/convolution but before output projection.
 */
void ck_test_gated_deltanet_autoregressive(const float *q,
                                           const float *k,
                                           const float *v,
                                           const float *g,
                                           const float *beta,
                                           const float *state_in,
                                           float *state_out,
                                           float *out,
                                           int num_heads,
                                           int state_dim,
                                           float norm_eps);

/* ============================================================================
 * Attention Kernels
 * ============================================================================ */

/**
 * @brief Multi-head causal attention for prefill (head-major layout)
 *
 * Layout (head-major, matches llama.cpp test):
 *   Q: [num_heads, tokens, head_dim]
 *   K: [num_kv_heads, seq_len, head_dim]
 *   V: [num_kv_heads, seq_len, head_dim]
 *   out: [num_heads, tokens, head_dim]
 *
 * Supports GQA (grouped-query attention) where num_heads > num_kv_heads.
 * Causal masking: token t can only attend to positions 0..t (inclusive).
 *
 * @param q          Query [num_heads, tokens, head_dim]
 * @param k          Key [num_kv_heads, seq_len, head_dim]
 * @param v          Value [num_kv_heads, seq_len, head_dim]
 * @param out        Output [num_heads, tokens, head_dim]
 * @param num_heads  Number of query heads
 * @param num_kv_heads Number of key/value heads (for GQA)
 * @param tokens     Number of query tokens
 * @param seq_len    Key/value sequence length (for prefill: seq_len == tokens)
 * @param head_dim   Dimension per head
 */
void ck_test_attention_causal(const float *q,
                               const float *k,
                               const float *v,
                               float *out,
                               int num_heads,
                               int num_kv_heads,
                               int tokens,
                               int seq_len,
                               int head_dim);

/* ============================================================================
 * Mega-Fused Kernels
 * ============================================================================ */

/**
 * @brief Test mega-fused OutProj + MLP kernel (Q5_0 weights)
 *
 * This tests the mega_fused_outproj_mlp_prefill kernel which fuses:
 * 1. Quantize attention output (head-major) to Q8_0
 * 2. OutProj: attn_out @ W_o (Q5_0) → h1
 * 3. Residual: h1 += residual
 * 4. RMSNorm: h1 → ln2_out
 * 5. MLP: silu(ln2_out @ W_gate) * (ln2_out @ W_up) @ W2
 * 6. Residual: output += h1
 *
 * @param attn_out     Attention output [num_heads, tokens, head_dim] (FP32, head-major)
 * @param residual     Residual input [tokens, embed_dim] (FP32)
 * @param ln2_gamma    RMSNorm gamma [embed_dim] (FP32)
 * @param wo           OutProj weights [embed_dim, embed_dim] (Q5_0)
 * @param w1           MLP W1 weights [2*intermediate, embed_dim] (Q5_0)
 * @param w2           MLP W2 weights [embed_dim, intermediate] (Q4_K or Q6_K)
 * @param output       Output [tokens, embed_dim] (FP32)
 * @param tokens       Number of tokens
 * @param num_heads    Number of attention heads
 * @param head_dim     Dimension per head
 * @param embed_dim    Embedding dimension (= num_heads * head_dim)
 * @param intermediate MLP intermediate dimension
 * @param eps          RMSNorm epsilon
 * @param w2_is_q6k    If true, W2 is Q6_K; if false, W2 is Q4_K
 */
void ck_test_outproj_mlp_fused_q5_0(
    const float *attn_out,
    const float *residual,
    const float *ln2_gamma,
    const void *wo,
    const void *w1,
    const void *w2,
    float *output,
    int tokens,
    int num_heads,
    int head_dim,
    int embed_dim,
    int intermediate,
    float eps,
    int w2_is_q6k);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Get Q4_K block size in bytes
 */
int ck_get_block_q4_k_size(void);

/**
 * @brief Get Q6_K block size in bytes
 */
int ck_get_block_q6_k_size(void);

/**
 * @brief Get Q8_K block size in bytes
 */
int ck_get_block_q8_k_size(void);

/**
 * @brief Get QK_K (elements per super-block)
 */
int ck_get_qk_k(void);

#ifdef __cplusplus
}
#endif

#endif /* CK_PARITY_API_H */
