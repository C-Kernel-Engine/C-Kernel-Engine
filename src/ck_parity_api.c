/**
 * @file ck_parity_api.c
 * @brief C-Kernel-Engine Parity Testing API Implementation
 *
 * Wraps CK kernels for parity testing against llama.cpp/ggml.
 */

#include "ck_parity_api.h"
#include "ckernel_quant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* External kernel function declarations */

/* Dequantization kernels (from dequant_kernels.c) */
extern void dequant_q4_k_row(const void *src, float *dst, size_t n_elements);
extern void dequant_q6_k_row(const void *src, float *dst, size_t n_elements);
extern void dequant_q4_0_row(const void *src, float *dst, size_t n_elements);

/* Quantization kernels (from gemm_kernels_q4k_q8k.c) */
extern void quantize_row_q8_k(const float *x, void *vy, int k);

/* GEMV/GEMM kernels */
extern void gemv_q4_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemm_nt_q4_k_q8_k(const void *A_q8, const void *B, const float *bias,
                              float *C, int M, int N, int K);

/* Q6_K x Q8_K kernels (from gemm_kernels_q6k_q8k.c) */
extern void gemv_q6_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemm_nt_q6_k_q8_k(const void *A_q8, const void *B, const float *bias,
                              float *C, int M, int N, int K);

/* Q8_0 x Q8_0 batch GEMM (from gemm_batch_int8.c) */
extern void gemm_nt_q8_0_q8_0(const void *A_q8, const void *B_q8, const float *bias,
                              float *C, int M, int N, int K);

/* Q5_0 x Q8_0 batch GEMM (from gemm_kernels_q5_0.c) */
extern void gemm_nt_q5_0_q8_0(const void *A_q8, const void *B_q5, const float *bias,
                              float *C, int M, int N, int K);

/* Q5_0 and Q8_0 GEMV kernels (from gemm_kernels_q5_0.c, gemm_kernels_q8_0.c) */
extern void gemv_q5_0(float *y, const void *W, const float *x, int M, int K);
extern void gemv_q8_0(float *y, const void *W, const float *x, int M, int K);

/* Quantized dot product kernels for parity with llama.cpp */
extern void gemv_q5_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q8_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);

/* Direct vec_dot kernels (single dot product, not GEMV) */
extern void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy);
extern void vec_dot_q8_0_q8_0(int n, float *s, const void *vx, const void *vy);

/* Q8_0 quantization (for input) */
extern void quantize_row_q8_0(const float *x, void *vy, int k);

/* RMSNorm kernel (from rmsnorm_kernels.c) */
extern void rmsnorm_forward(const float *input, const float *gamma,
                            float *output, float *rstd_cache,
                            int tokens, int d_model, int aligned_embed_dim, float eps);

/* RoPE kernels (from rope_kernels.c) */
extern void rope_forward_qk(float *q, float *k,
                            const float *cos_cache, const float *sin_cache,
                            int num_heads, int num_kv_heads, int num_tokens,
                            int head_dim, int aligned_head_dim, int pos_offset);
extern void rope_precompute_cache(float *cos_cache, float *sin_cache,
                                  int max_seq_len, int head_dim, float base,
                                  int rotary_dim, const char *scaling_type,
                                  float scaling_factor);

/* SwiGLU kernel (from swiglu_kernels.c) */
extern void swiglu_forward(const float *input, float *output, int tokens, int dim);

/* Attention kernels (from attention_kernels.c / attention_flash_true.c) */
extern void attention_forward_causal_head_major_gqa_flash_strided(
    const float *q, const float *k, const float *v, float *output,
    int num_heads, int num_kv_heads, int num_tokens,
    int head_dim, int aligned_head_dim, int kv_stride_tokens);

/* ============================================================================
 * Dequantization Tests
 * ============================================================================ */

void ck_test_dequant_q4_k(const void *src, float *dst, int n)
{
    dequant_q4_k_row(src, dst, (size_t)n);
}

void ck_test_dequant_q6_k(const void *src, float *dst, int n)
{
    dequant_q6_k_row(src, dst, (size_t)n);
}

void ck_test_dequant_q4_0(const void *src, float *dst, int n)
{
    dequant_q4_0_row(src, dst, (size_t)n);
}

/* ============================================================================
 * Quantization Tests
 * ============================================================================ */

void ck_test_quantize_q8_k(const float *src, void *dst, int n)
{
    quantize_row_q8_k(src, dst, n);
}

/* ============================================================================
 * GEMV Tests
 * ============================================================================ */

void ck_test_gemv_q4_k(const void *weight_q4k,
                       const float *input_f32,
                       float *output,
                       int cols)
{
    /* Allocate Q8_K buffer for quantized activations */
    int n_blocks = cols / CK_QK_K;
    block_q8_K *q8_data = (block_q8_K *)malloc(n_blocks * sizeof(block_q8_K));
    if (!q8_data) {
        *output = 0.0f;
        return;
    }

    /* Quantize input to Q8_K */
    quantize_row_q8_k(input_f32, q8_data, cols);

    /* Compute dot product using GEMV with M=1 */
    gemv_q4_k_q8_k(output, weight_q4k, q8_data, 1, cols);

    free(q8_data);
}

void ck_test_gemv_q6_k(const void *weight_q6k,
                       const float *input_f32,
                       float *output,
                       int cols)
{
    /* Match runtime decode path:
     * 1) quantize FP32 activation to Q8_K
     * 2) run Q6_K x Q8_K GEMV kernel with M=1
     */
    int n_blocks = cols / CK_QK_K;
    block_q8_K *q8_data = (block_q8_K *)malloc(n_blocks * sizeof(block_q8_K));
    if (!q8_data) {
        *output = 0.0f;
        return;
    }

    quantize_row_q8_k(input_f32, q8_data, cols);
    gemv_q6_k_q8_k(output, weight_q6k, q8_data, 1, cols);
    free(q8_data);
}

void ck_test_gemv_q5_0(const void *weight_q5_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols)
{
    /* Match llama.cpp's test_gemv_q5_0:
     * 1. Quantize input to Q8_0 format
     * 2. Use quantized dot product (vec_dot_q5_0_q8_0)
     *
     * This ensures parity with llama.cpp which always uses the
     * quantized path, NOT the FP32 dequantization path.
     */
    int n_blocks = cols / CK_QK8_0;
    block_q8_0 *q8_data = (block_q8_0 *)malloc(n_blocks * sizeof(block_q8_0));
    if (!q8_data) {
        for (int r = 0; r < rows; r++) output[r] = 0.0f;
        return;
    }

    /* Quantize input to Q8_0 */
    quantize_row_q8_0(input_f32, q8_data, cols);

    /* Call the quantized GEMV kernel (same as ck_test_gemv_q5_0_q8_0) */
    gemv_q5_0_q8_0(output, weight_q5_0, q8_data, rows, cols);

    free(q8_data);
}

void ck_test_gemv_q8_0(const void *weight_q8_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols)
{
    /* Match llama.cpp's test_gemv_q8_0:
     * 1. Quantize input to Q8_0 format
     * 2. Use quantized dot product (vec_dot_q8_0_q8_0)
     *
     * This ensures parity with llama.cpp which always uses the
     * quantized path, NOT the FP32 dequantization path.
     */
    int n_blocks = cols / CK_QK8_0;
    block_q8_0 *q8_data = (block_q8_0 *)malloc(n_blocks * sizeof(block_q8_0));
    if (!q8_data) {
        for (int r = 0; r < rows; r++) output[r] = 0.0f;
        return;
    }

    /* Quantize input to Q8_0 */
    quantize_row_q8_0(input_f32, q8_data, cols);

    /* Call the quantized GEMV kernel (same as ck_test_gemv_q8_0_q8_0) */
    gemv_q8_0_q8_0(output, weight_q8_0, q8_data, rows, cols);

    free(q8_data);
}

void ck_test_gemv_q5_0_q8_0(const void *weight_q5_0,
                             const float *input_f32,
                             float *output,
                             int rows, int cols)
{
    /* This matches llama.cpp's approach:
     * 1. Quantize input to Q8_0 format
     * 2. Use quantized dot product (integer math)
     * 3. Scale at the end
     */
    int n_blocks = cols / CK_QK8_0;
    block_q8_0 *q8_data = (block_q8_0 *)malloc(n_blocks * sizeof(block_q8_0));
    if (!q8_data) {
        for (int r = 0; r < rows; r++) output[r] = 0.0f;
        return;
    }

    /* Quantize input to Q8_0 */
    quantize_row_q8_0(input_f32, q8_data, cols);

    /* Call the quantized GEMV kernel */
    gemv_q5_0_q8_0(output, weight_q5_0, q8_data, rows, cols);

    free(q8_data);
}

void ck_test_gemv_q8_0_q8_0(const void *weight_q8_0,
                             const float *input_f32,
                             float *output,
                             int rows, int cols)
{
    /* This matches llama.cpp's approach:
     * 1. Quantize input to Q8_0 format
     * 2. Use quantized dot product (integer math)
     * 3. Scale at the end
     */
    int n_blocks = cols / CK_QK8_0;
    block_q8_0 *q8_data = (block_q8_0 *)malloc(n_blocks * sizeof(block_q8_0));
    if (!q8_data) {
        for (int r = 0; r < rows; r++) output[r] = 0.0f;
        return;
    }

    /* Quantize input to Q8_0 */
    quantize_row_q8_0(input_f32, q8_data, cols);

    /* Call the quantized GEMV kernel */
    gemv_q8_0_q8_0(output, weight_q8_0, q8_data, rows, cols);

    free(q8_data);
}

/* ============================================================================
 * Direct Vec Dot Tests (pre-quantized inputs, no FP32 conversion)
 * ============================================================================ */

/**
 * @brief Direct Q5_0 x Q8_0 dot product test (takes pre-quantized Q8_0 input)
 *
 * This is a "direct" test that bypasses FP32-to-Q8_0 conversion.
 * Useful for isolating kernel bugs from quantization bugs.
 *
 * @param weight_q5_0  Q5_0 quantized weights [cols]
 * @param input_q8_0   Q8_0 quantized input [cols] (pre-quantized!)
 * @param output       Output scalar [1]
 * @param cols         Number of elements (must be multiple of 32)
 */
void ck_test_vec_dot_q5_0_q8_0(const void *weight_q5_0,
                                const void *input_q8_0,
                                float *output,
                                int cols)
{
    vec_dot_q5_0_q8_0(cols, output, weight_q5_0, input_q8_0);
}

/**
 * @brief Direct Q8_0 x Q8_0 dot product test (takes pre-quantized Q8_0 input)
 *
 * @param weight_q8_0  Q8_0 quantized weights [cols]
 * @param input_q8_0   Q8_0 quantized input [cols] (pre-quantized!)
 * @param output       Output scalar [1]
 * @param cols         Number of elements (must be multiple of 32)
 */
void ck_test_vec_dot_q8_0_q8_0(const void *weight_q8_0,
                                const void *input_q8_0,
                                float *output,
                                int cols)
{
    vec_dot_q8_0_q8_0(cols, output, weight_q8_0, input_q8_0);
}

/* ============================================================================
 * GEMM Tests
 * ============================================================================ */

void ck_test_gemm_q4_k(const void *weight_q4k,
                       const float *input_f32,
                       float *output,
                       int rows, int cols, int n_tokens)
{
    /* Allocate Q8_K buffer for quantized activations */
    int n_blocks_per_row = cols / CK_QK_K;
    block_q8_K *q8_data = (block_q8_K *)malloc(n_tokens * n_blocks_per_row * sizeof(block_q8_K));
    if (!q8_data) {
        memset(output, 0, n_tokens * rows * sizeof(float));
        return;
    }

    /* Quantize all input tokens */
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_k(input_f32 + t * cols,
                          q8_data + t * n_blocks_per_row, cols);
    }

    /* Use gemm_nt_q4_k_q8_k: C[M,N] = A[M,K] * B[N,K]^T
     * Our layout: output[n_tokens, rows] = input[n_tokens, cols] * weight[rows, cols]^T
     * So: M = n_tokens, N = rows, K = cols
     */
    gemm_nt_q4_k_q8_k(q8_data, weight_q4k, NULL, output, n_tokens, rows, cols);

    free(q8_data);
}

/**
 * @brief Test Q6_K x Q8_K GEMM (batch matrix multiply)
 *
 * Used for MLP W2 (down projection) with Q6_K weights.
 */
void ck_test_gemm_q6_k(const void *weight_q6k,
                       const float *input_f32,
                       float *output,
                       int rows, int cols, int n_tokens)
{
    /* Allocate Q8_K buffer for quantized activations */
    int n_blocks_per_row = cols / CK_QK_K;
    block_q8_K *q8_data = (block_q8_K *)malloc(n_tokens * n_blocks_per_row * sizeof(block_q8_K));
    if (!q8_data) {
        memset(output, 0, n_tokens * rows * sizeof(float));
        return;
    }

    /* Quantize all input tokens */
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_k(input_f32 + t * cols,
                          q8_data + t * n_blocks_per_row, cols);
    }

    /* Use gemm_nt_q6_k_q8_k: C[M,N] = A[M,K] * B[N,K]^T
     * Our layout: output[n_tokens, rows] = input[n_tokens, cols] * weight[rows, cols]^T
     * So: M = n_tokens, N = rows, K = cols
     */
    gemm_nt_q6_k_q8_k(q8_data, weight_q6k, NULL, output, n_tokens, rows, cols);

    free(q8_data);
}

/**
 * @brief Test Q8_0 x Q8_0 GEMM (batch matrix multiply)
 *
 * Used for attention V projection with Q8_0 weights.
 */
void ck_test_gemm_q8_0(const void *weight_q8_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols, int n_tokens)
{
    /* Allocate Q8_0 buffer for quantized activations */
    int n_blocks_per_row = cols / CK_QK8_0;
    block_q8_0 *q8_data = (block_q8_0 *)malloc(n_tokens * n_blocks_per_row * sizeof(block_q8_0));
    if (!q8_data) {
        memset(output, 0, n_tokens * rows * sizeof(float));
        return;
    }

    /* Quantize all input tokens */
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_0(input_f32 + t * cols,
                          q8_data + t * n_blocks_per_row, cols);
    }

    /* Use gemm_nt_q8_0_q8_0: C[M,N] = A[M,K] * B[N,K]^T
     * Our layout: output[n_tokens, rows] = input[n_tokens, cols] * weight[rows, cols]^T
     * So: M = n_tokens, N = rows, K = cols
     */
    gemm_nt_q8_0_q8_0(q8_data, weight_q8_0, NULL, output, n_tokens, rows, cols);

    free(q8_data);
}

/**
 * @brief Test Q5_0 x Q8_0 GEMM (batch matrix multiply)
 *
 * Used for MLP W1 (gate/up projection) and attention Q/K with Q5_0 weights.
 */
void ck_test_gemm_q5_0(const void *weight_q5_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols, int n_tokens)
{
    /* Allocate Q8_0 buffer for quantized activations */
    int n_blocks_per_row = cols / CK_QK8_0;
    block_q8_0 *q8_data = (block_q8_0 *)malloc(n_tokens * n_blocks_per_row * sizeof(block_q8_0));
    if (!q8_data) {
        memset(output, 0, n_tokens * rows * sizeof(float));
        return;
    }

    /* Quantize all input tokens */
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_0(input_f32 + t * cols,
                          q8_data + t * n_blocks_per_row, cols);
    }

    /* Use gemm_nt_q5_0_q8_0: C[M,N] = A[M,K] * B[N,K]^T
     * Our layout: output[n_tokens, rows] = input[n_tokens, cols] * weight[rows, cols]^T
     * So: M = n_tokens, N = rows, K = cols
     */
    gemm_nt_q5_0_q8_0(q8_data, weight_q5_0, NULL, output, n_tokens, rows, cols);

    free(q8_data);
}

/* ============================================================================
 * Activation Kernels
 * ============================================================================ */

void ck_test_rmsnorm(const float *input,
                     const float *weight,
                     float *output,
                     int n_tokens, int dim, float eps)
{
    /* CK rmsnorm_forward has aligned_embed_dim parameter
     * For testing, use dim as aligned_embed_dim (no padding) */
    rmsnorm_forward(input, weight, output, NULL, n_tokens, dim, dim, eps);
}

void ck_test_rope(float *q, float *k,
                  int n_tokens, int n_heads, int n_heads_kv, int head_dim,
                  int pos_offset, float theta)
{
    /* Precompute cos/sin cache */
    int half_dim = head_dim / 2;
    int max_seq = pos_offset + n_tokens;

    float *cos_cache = (float *)malloc(max_seq * half_dim * sizeof(float));
    float *sin_cache = (float *)malloc(max_seq * half_dim * sizeof(float));
    if (!cos_cache || !sin_cache) {
        free(cos_cache);
        free(sin_cache);
        return;
    }

    rope_precompute_cache(cos_cache, sin_cache, max_seq, head_dim, theta,
                          head_dim, "none", 1.0f);

    /* CK RoPE expects layout [num_heads, num_tokens, head_dim]
     * Reshape from [n_tokens, n_heads * head_dim] to [n_heads, n_tokens, head_dim]
     */
    float *q_reorder = (float *)malloc(n_heads * n_tokens * head_dim * sizeof(float));
    float *k_reorder = (float *)malloc(n_heads_kv * n_tokens * head_dim * sizeof(float));

    if (q_reorder && k_reorder) {
        /* Reorder Q: [T, H*D] -> [H, T, D] */
        for (int t = 0; t < n_tokens; t++) {
            for (int h = 0; h < n_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    q_reorder[h * n_tokens * head_dim + t * head_dim + d] =
                        q[t * n_heads * head_dim + h * head_dim + d];
                }
            }
        }

        /* Reorder K: [T, H_kv*D] -> [H_kv, T, D] */
        for (int t = 0; t < n_tokens; t++) {
            for (int h = 0; h < n_heads_kv; h++) {
                for (int d = 0; d < head_dim; d++) {
                    k_reorder[h * n_tokens * head_dim + t * head_dim + d] =
                        k[t * n_heads_kv * head_dim + h * head_dim + d];
                }
            }
        }

        /* Apply RoPE */
        rope_forward_qk(q_reorder, k_reorder,
                        cos_cache, sin_cache,
                        n_heads, n_heads_kv, n_tokens,
                        head_dim, head_dim, pos_offset);

        /* Reorder back: [H, T, D] -> [T, H*D] */
        for (int t = 0; t < n_tokens; t++) {
            for (int h = 0; h < n_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    q[t * n_heads * head_dim + h * head_dim + d] =
                        q_reorder[h * n_tokens * head_dim + t * head_dim + d];
                }
            }
        }

        for (int t = 0; t < n_tokens; t++) {
            for (int h = 0; h < n_heads_kv; h++) {
                for (int d = 0; d < head_dim; d++) {
                    k[t * n_heads_kv * head_dim + h * head_dim + d] =
                        k_reorder[h * n_tokens * head_dim + t * head_dim + d];
                }
            }
        }
    }

    free(q_reorder);
    free(k_reorder);
    free(cos_cache);
    free(sin_cache);
}

void ck_test_rope_interleaved(float *q, float *k,
                              int n_tokens, int n_heads, int n_heads_kv, int head_dim,
                              int pos_offset, float theta)
{
    /* Interleaved RoPE format (matches llama.cpp):
     * (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
     * Applied to consecutive pairs of elements
     */

    /* Precompute inverse frequencies */
    float *inv_freq = (float *)malloc((head_dim / 2) * sizeof(float));
    if (!inv_freq) return;

    for (int i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0f / powf(theta, (float)(2 * i) / head_dim);
    }

    /* Apply RoPE to Q */
    for (int t = 0; t < n_tokens; t++) {
        int pos = pos_offset + t;
        for (int h = 0; h < n_heads; h++) {
            float *qh = q + t * n_heads * head_dim + h * head_dim;

            for (int i = 0; i < head_dim / 2; i++) {
                float freq = pos * inv_freq[i];
                float cos_val = cosf(freq);
                float sin_val = sinf(freq);

                /* Interleaved format */
                float x0 = qh[i * 2];
                float x1 = qh[i * 2 + 1];
                qh[i * 2]     = x0 * cos_val - x1 * sin_val;
                qh[i * 2 + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    /* Apply RoPE to K */
    for (int t = 0; t < n_tokens; t++) {
        int pos = pos_offset + t;
        for (int h = 0; h < n_heads_kv; h++) {
            float *kh = k + t * n_heads_kv * head_dim + h * head_dim;

            for (int i = 0; i < head_dim / 2; i++) {
                float freq = pos * inv_freq[i];
                float cos_val = cosf(freq);
                float sin_val = sinf(freq);

                float x0 = kh[i * 2];
                float x1 = kh[i * 2 + 1];
                kh[i * 2]     = x0 * cos_val - x1 * sin_val;
                kh[i * 2 + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    free(inv_freq);
}

void ck_test_swiglu(const float *gate_up,
                    float *output,
                    int n_tokens, int intermediate_dim)
{
    swiglu_forward(gate_up, output, n_tokens, intermediate_dim);
}

void ck_test_softmax(const float *input, float *output, int n)
{
    /* Find max for numerical stability */
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        output[i] *= inv_sum;
    }
}

/* ============================================================================
 * Attention Kernels
 * ============================================================================ */

void ck_test_attention_causal(const float *q,
                               const float *k,
                               const float *v,
                               float *out,
                               int num_heads,
                               int num_kv_heads,
                               int tokens,
                               int seq_len,
                               int head_dim)
{
    /* For prefill, seq_len == tokens, and kv_stride == tokens.
     * The CK kernel expects strided KV layout with kv_stride_tokens parameter.
     * For parity testing with contiguous tensors, kv_stride = seq_len.
     */
    attention_forward_causal_head_major_gqa_flash_strided(
        q, k, v, out,
        num_heads, num_kv_heads, tokens,
        head_dim, head_dim,  /* aligned_head_dim = head_dim for testing */
        seq_len              /* kv_stride_tokens = seq_len for contiguous KV */
    );
}

/* ============================================================================
 * Mega-Fused OutProj + MLP Kernels
 * ============================================================================ */

/* External declaration for mega_fused_outproj_mlp_prefill */
extern void mega_fused_outproj_mlp_prefill(
    float *output,
    const float *attn_out,
    const float *residual,
    const float *ln2_gamma,
    const void *wo, const float *bo, int wo_dt,
    const void *w1, const float *b1, int w1_dt,
    const void *w2, const float *b2, int w2_dt,
    int tokens,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int aligned_head_dim,
    int intermediate_dim,
    int aligned_intermediate_dim,
    float eps,
    void *scratch);

extern size_t mega_fused_outproj_mlp_prefill_scratch_size(
    int tokens,
    int aligned_embed_dim,
    int num_heads,
    int aligned_head_dim,
    int aligned_intermediate_dim);

/**
 * @brief Test mega-fused OutProj + MLP kernel (Q5_0 weights)
 *
 * This is a simplified wrapper for parity testing that:
 * - Uses Q5_0 for W_o and W1 weights
 * - Uses Q4_K for W2 weights
 * - Allocates scratch internally
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
    int w2_is_q6k)
{
    /* CK uses dtype enum: CK_DT_Q5_0 = 11, CK_DT_Q4_K = 7, CK_DT_Q6_K = 8 */
    const int CK_DT_Q5_0_VAL = 11;
    const int CK_DT_Q4_K_VAL = 7;
    const int CK_DT_Q6_K_VAL = 8;

    /* For parity testing, aligned = actual (no padding) */
    int aligned_embed_dim = embed_dim;
    int aligned_head_dim = head_dim;
    int aligned_intermediate = intermediate;

    /* Ensure intermediate is multiple of 256 (QK_K) for K-quants */
    if ((intermediate % 256) != 0) {
        aligned_intermediate = ((intermediate + 255) / 256) * 256;
    }

    /* Allocate scratch */
    size_t scratch_size = mega_fused_outproj_mlp_prefill_scratch_size(
        tokens, aligned_embed_dim, num_heads, aligned_head_dim, aligned_intermediate);

    void *scratch = malloc(scratch_size);
    if (!scratch) {
        return;
    }

    /* Call the mega-fused kernel */
    mega_fused_outproj_mlp_prefill(
        output,
        attn_out,
        residual,
        ln2_gamma,
        wo, NULL, CK_DT_Q5_0_VAL,          /* W_o with Q5_0 */
        w1, NULL, CK_DT_Q5_0_VAL,          /* W1 with Q5_0 */
        w2, NULL, w2_is_q6k ? CK_DT_Q6_K_VAL : CK_DT_Q4_K_VAL,  /* W2 with Q4_K or Q6_K */
        tokens,
        embed_dim,
        aligned_embed_dim,
        num_heads,
        aligned_head_dim,
        intermediate,
        aligned_intermediate,
        eps,
        scratch
    );

    free(scratch);
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

int ck_get_block_q4_k_size(void)
{
    return sizeof(block_q4_K);
}

int ck_get_block_q6_k_size(void)
{
    return sizeof(block_q6_K);
}

int ck_get_block_q8_k_size(void)
{
    return sizeof(block_q8_K);
}

int ck_get_qk_k(void)
{
    return QK_K;
}
