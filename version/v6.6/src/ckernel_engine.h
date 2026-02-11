/*
 * ckernel_engine.h - v6.6 kernel declarations
 *
 * Provides all kernel function prototypes needed by generated v6.6 code.
 * This is the complete set used by codegen_v6_6.py output.
 *
 * NOTE: This header must include prototypes for ANY kernel that codegen may emit.
 * If codegen selects a kernel that lacks a declaration here, model_v6_6.c will
 * fail to compile with "call to undeclared function" errors.
 *
 * When adding new kernels to the kernel_maps/ registry, ensure their prototypes
 * are added here as well to maintain build correctness.
 */

#ifndef CKERNEL_ENGINE_H
#define CKERNEL_ENGINE_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * DATA TYPE ENUMS
 * ============================================================================ */

#define CK_DT_FP32    0
#define CK_DT_FP16    1
#define CK_DT_BF16    2
#define CK_DT_Q8_0    3
#define CK_DT_Q5_0    4
#define CK_DT_Q4_K    5
#define CK_DT_Q6_K    6
#define CK_DT_I8      7
#define CK_DT_INT32   8

/* ============================================================================
 * DTYPE HELPERS
 * ============================================================================ */

static inline float ck_dtype_bytes_per_elem(int dtype) {
    switch (dtype) {
        case CK_DT_FP32:  return 4.0f;
        case CK_DT_FP16:
        case CK_DT_BF16:  return 2.0f;
        case CK_DT_I8:    return 1.0f;
        case CK_DT_INT32: return 4.0f;
        case CK_DT_Q8_0:  return 1.0625f;   /* 32 bytes + 2 byte scale per 32 elements */
        case CK_DT_Q5_0:  return 0.6875f;   /* 20 bytes + 2 byte scale per 32 elements */
        case CK_DT_Q4_K:  return 0.5625f;   /* 144 bytes per 256 elements */
        case CK_DT_Q6_K:  return 0.8203125f; /* 210 bytes per 256 elements */
        default:          return 4.0f;
    }
}

/* Legacy alias for compatibility */
#define ck_dtype_bytes(dtype) ((size_t)ck_dtype_bytes_per_elem(dtype))

static inline size_t ck_dtype_row_bytes(int dtype, size_t elements) {
    return (size_t)(elements * ck_dtype_bytes_per_elem(dtype));
}

static inline size_t ck_dtype_block_size(int dtype) {
    switch (dtype) {
        case CK_DT_Q8_0:  return 32;
        case CK_DT_Q5_0:  return 32;
        case CK_DT_Q4_K:
        case CK_DT_Q6_K:  return 256;
        default:          return 1;
    }
}

/* ============================================================================
 * EMBEDDING KERNELS
 * ============================================================================ */

void embedding_forward_q8_0(const int32_t *token_ids,
                            int num_tokens,
                            int vocab_size,
                            const void *embedding_table,
                            const float *pos_embed,
                            float *output,
                            int embed_dim,
                            int aligned_embed_dim,
                            int context_size,
                            int add_pos_embed);

/* ============================================================================
 * NORMALIZATION KERNELS
 * ============================================================================ */

void rmsnorm_forward(const float *input,
                     const float *gamma,
                     float *output,
                     float *rstd,
                     int num_tokens,
                     int embed_dim,
                     int aligned_embed_dim,
                     float eps);

/* Per-head RMSNorm on Q and K (Qwen3-style QK norm).
 * Wraps rmsnorm_forward twice: once for Q [num_heads*num_tokens, head_dim],
 * once for K [num_kv_heads*num_tokens, head_dim]. Operates in-place on
 * scratch buffers between QKV projection and RoPE. */
void qk_norm_forward(float *q, float *k,
                     const float *q_gamma, const float *k_gamma,
                     int num_heads, int num_kv_heads,
                     int num_tokens, int head_dim, float eps);

/* ============================================================================
 * QUANTIZATION KERNELS
 * ============================================================================ */

void quantize_row_q8_0(const float *x, void *y, int k);
void quantize_batch_q8_0(const float *x, void *y, int num_rows, int k);
void quantize_row_q8_k(const float *x, void *y, int k);
void quantize_batch_q8_k(const float *x, void *y, int num_rows, int k);

/* ============================================================================
 * GEMV KERNELS (decode mode - single token)
 * ============================================================================ */

void gemv_q4_k(float *y, const void *W, const float *x, int M, int K);
void gemv_q5_0(float *y, const void *W, const float *x, int M, int K);
void gemv_q5_1(float *y, const void *W, const float *x, int M, int K);
void gemv_q5_1_q8_1(float *y, const void *W, const float *x, int M, int K);
void gemv_q5_1_q8_1_ref(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q5_k(float *y, const void *W, const float *x, int M, int K);
void gemv_q5_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q6_k(float *y, const void *W, const float *x, int M, int K);
void gemv_q8_0(float *y, const void *W, const float *x, int M, int K);
void gemv_q8_0_q8_0_contract(float *y, const void *W, const float *x, int M, int K);

/* INT8 activation variants */
void gemv_q4_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q5_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q6_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q8_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);

// Fused GEMV: quantize(FP32->Q8_0) + GEMV(Q5_0 weights) + bias add
void gemv_fused_q5_0_bias_dispatch(float *y, const void *W, const float *x,
                                    const float *bias, int M, int K);

// Fused GEMV: quantize(FP32->Q8_0) + GEMV(Q8_0 weights) + bias add
void gemv_fused_q8_0_bias_dispatch(float *y, const void *W, const float *x,
                                    const float *bias, int M, int K);

/* OpenMP-parallel GEMV variants (for single-stream decode throughput) */
void gemv_q8_0_q8_0_parallel_omp(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q5_0_q8_0_parallel_omp(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_fused_q5_0_bias_parallel_omp(float *y, const void *W, const float *x,
                                const float *bias, int M, int K);

/* ============================================================================
 * GEMM KERNELS (prefill mode - multiple tokens)
 * ============================================================================ */

void gemm_blocked_serial(const float *A, const float *B, const float *bias,
                        float *C, int M, int N, int K);

void gemm_nt_q4_k(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q5_0(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q5_1(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q5_1_q8_1(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q5_1_q8_1_ref(const void *A_q8, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q5_k(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q5_k_q8_k(const void *A_q8, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q6_k(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q8_0(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q8_0_q8_0_contract(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);

/* INT8 activation GEMM variants (quantized input for prefill) */
void gemm_nt_q4_k_q8_k(const void *A_q8, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q5_0_q8_0(const void *A_q8, const void *B_q5, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q6_k_q8_k(const void *A_q8, const void *B, const float *bias, float *C, int M, int N, int K);
void gemm_nt_q8_0_q8_0(const void *A_q8, const void *B, const float *bias, float *C, int M, int N, int K);

/* ============================================================================
 * ROPE KERNELS
 * ============================================================================ */

void rope_precompute_cache_split(float *cos_cache,
                                float *sin_cache,
                                int max_seq_len,
                                int head_dim,
                                float base);

void rope_precompute_cache(float *cos_cache,
                           float *sin_cache,
                           int max_seq_len,
                           int head_dim,
                           float base,
                           int rotary_dim,
                           const char *scaling_type,
                           float scaling_factor);

void rope_forward_qk(float *q,
                     float *k,
                     const float *cos_cache,
                     const float *sin_cache,
                     int num_heads,
                     int num_kv_heads,
                     int num_tokens,
                     int head_dim,
                     int aligned_head_dim,
                     int pos);

void rope_forward_qk_with_rotary_dim(float *q,
                                     float *k,
                                     const float *cos_cache,
                                     const float *sin_cache,
                                     int num_heads,
                                     int num_kv_heads,
                                     int num_tokens,
                                     int head_dim,
                                     int aligned_head_dim,
                                     int pos,
                                     int rotary_dim);

/* ============================================================================
 * ATTENTION KERNELS
 * ============================================================================ */

void attention_forward_decode_head_major_gqa_flash(const float *q_token,
                                                   const float *k_cache,
                                                   const float *v_cache,
                                                   float *out_token,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int kv_tokens,
                                                   int cache_capacity,
                                                   int head_dim,
                                                   int aligned_head_dim);

void attention_forward_causal_head_major_gqa_flash_strided(const float *q,
                                                           const float *k,
                                                           const float *v,
                                                           float *out,
                                                           int num_heads,
                                                           int num_kv_heads,
                                                           int num_tokens,
                                                           int head_dim,
                                                           int aligned_head_dim,
                                                           int stride);

/* Sliding window attention variants */
void attention_forward_decode_head_major_gqa_flash_sliding(const float *q_token,
                                                          const float *k_cache,
                                                          const float *v_cache,
                                                          float *out_token,
                                                          int num_heads,
                                                          int num_kv_heads,
                                                          int kv_tokens,
                                                          int cache_capacity,
                                                          int head_dim,
                                                          int aligned_head_dim,
                                                          int sliding_window);

void attention_forward_causal_head_major_gqa_flash_strided_sliding(const float *q,
                                                                  const float *k,
                                                                  const float *v,
                                                                  float *out,
                                                                  int num_heads,
                                                                  int num_kv_heads,
                                                                  int num_tokens,
                                                                  int head_dim,
                                                                  int aligned_head_dim,
                                                                  int stride,
                                                                  int sliding_window);

/* ============================================================================
 * KV CACHE KERNELS
 * ============================================================================ */

void kv_cache_store(float *__restrict kv_cache_k,
                    float *__restrict kv_cache_v,
                    const float *__restrict k,
                    const float *__restrict v,
                    int layer,
                    int pos,
                    int num_kv_heads,
                    int head_dim,
                    int max_seq_len);

/* ============================================================================
 * ACTIVATION KERNELS
 * ============================================================================ */

void geglu_forward_fp32(const float *x, float *out, int tokens, int dim);
void geglu_forward_bf16(const uint16_t *x, uint16_t *out, int tokens, int dim, float *scratch);
void swiglu_forward(const float *input, float *output, int num_tokens, int dim);

/* ============================================================================
 * RESIDUAL KERNELS
 * ============================================================================ */

void ck_residual_add_token_major(const float *a, const float *b, float *out,
                                 int tokens, int aligned_embed_dim);

/* ============================================================================
 * UTILITY KERNELS
 * ============================================================================ */

void add_inplace_f32(float *a, const float *b, size_t n);

#endif /* CKERNEL_ENGINE_H */
