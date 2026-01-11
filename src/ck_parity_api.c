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

/* Q5_0 and Q8_0 GEMV kernels (from gemm_kernels_q5_0.c, gemm_kernels_q8_0.c) */
extern void gemv_q5_0(float *y, const void *W, const float *x, int M, int K);
extern void gemv_q8_0(float *y, const void *W, const float *x, int M, int K);

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
                                  int max_seq_len, int head_dim, float base);

/* SwiGLU kernel (from swiglu_kernels.c) */
extern void swiglu_forward(const float *input, float *output, int tokens, int dim);

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
    /* Q6_K GEMV is not yet implemented in CK - provide reference impl */
    /* For now, dequantize and compute in FP32 */
    float *weight_f32 = (float *)malloc(cols * sizeof(float));
    if (!weight_f32) {
        *output = 0.0f;
        return;
    }

    dequant_q6_k_row(weight_q6k, weight_f32, cols);

    /* Dot product in FP32 */
    double sum = 0.0;
    for (int i = 0; i < cols; i++) {
        sum += (double)weight_f32[i] * (double)input_f32[i];
    }
    *output = (float)sum;

    free(weight_f32);
}

void ck_test_gemv_q5_0(const void *weight_q5_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols)
{
    /* Call the Q5_0 GEMV kernel directly */
    gemv_q5_0(output, weight_q5_0, input_f32, rows, cols);
}

void ck_test_gemv_q8_0(const void *weight_q8_0,
                       const float *input_f32,
                       float *output,
                       int rows, int cols)
{
    /* Call the Q8_0 GEMV kernel directly */
    gemv_q8_0(output, weight_q8_0, input_f32, rows, cols);
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

    rope_precompute_cache(cos_cache, sin_cache, max_seq, head_dim, theta);

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
