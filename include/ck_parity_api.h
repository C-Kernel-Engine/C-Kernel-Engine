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
