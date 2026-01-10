#include "ckernel_orchestration.h"

#include "ckernel_engine.h"
#include "ckernel_dtype.h"
#include "ckernel_quant.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ============================================================================
 * TRUE FLASH ATTENTION (O(1) for decode)
 *
 * New implementation based on Tri Dao's Flash Attention algorithm.
 * Provides O(1) complexity for decode instead of O(n).
 *
 * Reference: attention_flash_decode() in src/kernels/attention_flash_true.c
 * ============================================================================ */

/**
 * @brief Wrapper to call TRUE flash attention from orchestration layer
 *
 * @param q_token Query token [H, D_h]
 * @param k_cache Cached keys [T_k, H, D_h]
 * @param v_cache Cached values [T_k, H, D_h]
 * @param out_token Output [H, D_h]
 * @param num_heads Number of heads
 * @param num_kv_heads Number of KV heads (for GQA)
 * @param kv_tokens Number of tokens in KV cache
 * @param cache_capacity Cache capacity
 * @param head_dim Head dimension
 * @param aligned_head_dim Aligned head dimension
 */
void ck_attention_flash_decode_wrapper(
    const float *q_token,
    const float *k_cache,
    const float *v_cache,
    float *out_token,
    int num_heads,
    int num_kv_heads,
    int kv_tokens,
    int cache_capacity,
    int head_dim,
    int aligned_head_dim)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    static int use_strict = -1;
    if (use_strict < 0) {
        const char *env = getenv("CK_FLASH_ATTN_STRICT");
        use_strict = (env && env[0] && env[0] != '0') ? 1 : 0;
    }

    if (use_strict) {
        attention_forward_decode_head_major_gqa_regular(q_token,
                                                        k_cache,
                                                        v_cache,
                                                        out_token,
                                                        num_heads,
                                                        num_kv_heads,
                                                        kv_tokens,
                                                        cache_capacity,
                                                        head_dim,
                                                        aligned_head_dim);
        return;
    }

    // Scale factor: 1/sqrt(head_dim)
    const float scale = 1.0f / sqrtf((float)head_dim);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

#pragma omp parallel for schedule(static) if(num_heads > 1)
    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
        const float *k_head = k_cache + (size_t)kv_head * head_stride;
        const float *v_head = v_cache + (size_t)kv_head * head_stride;
        float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;

        // Use aligned_head_dim as D_h so per-token stride matches the cache layout.
        attention_flash_decode(out_head,
                               q_head,
                               k_head,
                               v_head,
                               1,
                               kv_tokens,
                               1,
                               aligned_head_dim,
                               scale);
    }
}

void ck_residual_add_token_major(const float *a,
                                 const float *b,
                                 float *out,
                                 int tokens,
                                 int aligned_embed_dim)
{
    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;
    for (size_t i = 0; i < total; ++i) {
        out[i] = a[i] + b[i];
    }
}

void ck_residual_add_backward(const float *d_out,
                              float *d_a,
                              float *d_b,
                              int tokens,
                              int aligned_embed_dim)
{
    if (!d_out || !d_a || !d_b) {
        return;
    }
    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;
    for (size_t i = 0; i < total; ++i) {
        float v = d_out[i];
        d_a[i] = v;
        d_b[i] = v;
    }
}

void ck_qkv_project_head_major(const float *input,
                               const float *wq, const float *bq,
                               const float *wk, const float *bk,
                               const float *wv, const float *bv,
                               float *q, float *k, float *v,
                               int tokens,
                               int kv_stride_tokens,
                               int aligned_embed_dim,
                               int num_heads,
                               int num_kv_heads,
                               int aligned_head_dim)
{
    if (!input || !wq || !wk || !wv || !q || !k || !v) {
        return;
    }
    if (kv_stride_tokens < tokens) {
        return;
    }

    size_t head_weight_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    size_t q_head_stride = (size_t)tokens * (size_t)aligned_head_dim;
    size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *wq_h = wq + (size_t)h * head_weight_stride;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;

        gemm_blocked_serial(input, wq_h, bq_h, q_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *wk_h = wk + (size_t)h * head_weight_stride;
        const float *wv_h = wv + (size_t)h * head_weight_stride;

        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;

        float *k_h = k + (size_t)h * kv_head_stride;
        float *v_h = v + (size_t)h * kv_head_stride;

        gemm_blocked_serial(input, wk_h, bk_h, k_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
        gemm_blocked_serial(input, wv_h, bv_h, v_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
    }
}

static int ck_layer_debug_enabled(void)
{
    static int cached = -2;
    if (cached != -2) {
        return cached;
    }
    const char *env = getenv("CK_LAYER_DEBUG");
    if (env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y')) {
        cached = 1;
    } else {
        cached = 0;
    }
    return cached;
}

static void ck_debug_check_buffer(const char *stage, const float *buf, int size)
{
    if (!ck_layer_debug_enabled() || !buf) {
        return;
    }
    int nan_count = 0, inf_count = 0;
    float min_val = 1e38f, max_val = -1e38f;
    for (int i = 0; i < size; ++i) {
        float v = buf[i];
        if (isnan(v)) {
            nan_count++;
        } else if (isinf(v)) {
            inf_count++;
        } else {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
    }
    if (nan_count > 0 || inf_count > 0) {
        fprintf(stderr, "[LAYER_DEBUG] %-30s size=%5d nan=%d inf=%d\n",
                stage, size, nan_count, inf_count);
    } else {
        fprintf(stderr, "[LAYER_DEBUG] %-30s size=%5d range=[%.3e, %.3e]\n",
                stage, size, min_val, max_val);
    }
}

static void ck_debug_check_q8k(const char *stage, const void *q8_buf, int num_blocks)
{
    if (!ck_layer_debug_enabled() || !q8_buf) {
        return;
    }
    const block_q8_K *blocks = (const block_q8_K *)q8_buf;
    int nan_scale = 0, inf_scale = 0;
    float min_d = 1e38f, max_d = -1e38f;
    for (int i = 0; i < num_blocks; ++i) {
        float d = blocks[i].d;
        if (isnan(d)) {
            nan_scale++;
        } else if (isinf(d)) {
            inf_scale++;
        } else {
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
        }
    }
    if (nan_scale > 0 || inf_scale > 0) {
        fprintf(stderr, "[LAYER_DEBUG] %-30s blocks=%d nan_scale=%d inf_scale=%d\n",
                stage, num_blocks, nan_scale, inf_scale);
    } else {
        fprintf(stderr, "[LAYER_DEBUG] %-30s blocks=%d scale_range=[%.3e, %.3e]\n",
                stage, num_blocks, min_d, max_d);
    }
}

static void ck_debug_check_q4k_weights(const char *stage, const void *q4_buf, int num_blocks)
{
    if (!ck_layer_debug_enabled() || !q4_buf) {
        return;
    }
    const block_q4_K *blocks = (const block_q4_K *)q4_buf;
    int nan_d = 0, nan_dmin = 0;
    float min_d = 1e38f, max_d = -1e38f;
    for (int i = 0; i < num_blocks; ++i) {
        float d = CK_FP16_TO_FP32(blocks[i].d);
        float dm = CK_FP16_TO_FP32(blocks[i].dmin);
        if (isnan(d)) nan_d++;
        if (isnan(dm)) nan_dmin++;
        if (!isnan(d) && !isinf(d)) {
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
        }
    }
    if (nan_d > 0 || nan_dmin > 0) {
        fprintf(stderr, "[LAYER_DEBUG] %-30s blocks=%d nan_d=%d nan_dmin=%d\n",
                stage, num_blocks, nan_d, nan_dmin);
    } else {
        fprintf(stderr, "[LAYER_DEBUG] %-30s blocks=%d d_range=[%.3e, %.3e]\n",
                stage, num_blocks, min_d, max_d);
    }
}

static int ck_q8k_activations_enabled(void)
{
    static int cached = -2;
    if (cached != -2) {
        return cached;
    }

    const char *env = getenv("CK_Q8K_ACTIVATIONS");
    if (!env || !env[0]) {
        cached = ck_strict_parity_enabled() ? 0 : 1;
        return cached;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' ||
        env[0] == 'f' || env[0] == 'F') {
        cached = 0;
    } else {
        cached = 1;
    }
    return cached;
}

void ck_gemm_nt_quant(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K,
                      CKDataType dtype)
{
    switch (dtype) {
    case CK_DT_FP32:
        gemm_blocked_serial(A, (const float *)B, bias, C, M, N, K);
        break;
    case CK_DT_Q4_K:
        gemm_nt_q4_k(A, B, bias, C, M, N, K);
        break;
    case CK_DT_Q6_K:
        gemm_nt_q6_k(A, B, bias, C, M, N, K);
        break;
    case CK_DT_Q4_0:
        gemm_nt_q4_0(A, B, bias, C, M, N, K);
        break;
    case CK_DT_Q4_1:
        gemm_nt_q4_1(A, B, bias, C, M, N, K);
        break;
    case CK_DT_Q5_0:
        gemm_nt_q5_0(A, B, bias, C, M, N, K);
        break;
    case CK_DT_Q5_1:
        gemm_nt_q5_1(A, B, bias, C, M, N, K);
        break;
    case CK_DT_Q8_0:
        gemm_nt_q8_0(A, B, bias, C, M, N, K);
        break;
    default:
        break;
    }
}

/* ============================================================================
 * Q4_K (Q4_K_M) forward-only paths
 * ============================================================================
 *
 * These helpers keep the same activation layouts as the fp32 code paths, but
 * accept weight matrices stored as GGML-compatible Q4_K blocks. This is meant
 * for weight-only quantized inference: activations remain fp32 by default, but
 * decode can switch to Q8_K activations via CK_Q8K_ACTIVATIONS=1.
 *
 * Important constraints:
 *   - Q4_K kernels require K (the input dimension) to be a multiple of 256.
 *   - For attention output projection we assume the concatenated head vector
 *     has length aligned_embed_dim (i.e., num_heads * aligned_head_dim matches).
 */

static void ck_qkv_project_head_major_q4_k(const float *input,
                                          const void *wq, const float *bq,
                                          const void *wk, const float *bk,
                                          const void *wv, const float *bv,
                                          float *q, float *k, float *v,
                                          int tokens,
                                          int kv_stride_tokens,
                                          int aligned_embed_dim,
                                          int num_heads,
                                          int num_kv_heads,
                                          int aligned_head_dim)
{
    if (!input || !wq || !wk || !wv || !q || !k || !v) {
        return;
    }
    if (kv_stride_tokens < tokens) {
        return;
    }

    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t head_w_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const size_t q_head_stride = (size_t)tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;

    const uint8_t *wq_bytes = (const uint8_t *)wq;
    const uint8_t *wk_bytes = (const uint8_t *)wk;
    const uint8_t *wv_bytes = (const uint8_t *)wv;

    for (int h = 0; h < num_heads; ++h) {
        const void *wq_h = wq_bytes + (size_t)h * head_w_bytes;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;

        gemm_nt_q4_k(input, wq_h, bq_h, q_h,
                     tokens, aligned_head_dim, aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const void *wk_h = wk_bytes + (size_t)h * head_w_bytes;
        const void *wv_h = wv_bytes + (size_t)h * head_w_bytes;

        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;

        float *k_h = k + (size_t)h * kv_head_stride;
        float *v_h = v + (size_t)h * kv_head_stride;

        gemm_nt_q4_k(input, wk_h, bk_h, k_h,
                     tokens, aligned_head_dim, aligned_embed_dim);
        gemm_nt_q4_k(input, wv_h, bv_h, v_h,
                     tokens, aligned_head_dim, aligned_embed_dim);
    }
}

static void ck_qkv_project_head_major_quant(const float *input,
                                            const void *wq, const float *bq, CKDataType wq_dtype,
                                            const void *wk, const float *bk, CKDataType wk_dtype,
                                            const void *wv, const float *bv, CKDataType wv_dtype,
                                            float *q, float *k, float *v,
                                            int tokens,
                                            int kv_stride_tokens,
                                            int aligned_embed_dim,
                                            int num_heads,
                                            int num_kv_heads,
                                            int aligned_head_dim)
{
    if (!input || !wq || !wk || !wv || !q || !k || !v) {
        return;
    }
    if (kv_stride_tokens < tokens) {
        return;
    }

    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t q_head_stride = (size_t)tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;

    const size_t wq_head_bytes = ck_dtype_row_bytes(wq_dtype, head_w_elems);
    const size_t wk_head_bytes = ck_dtype_row_bytes(wk_dtype, head_w_elems);
    const size_t wv_head_bytes = ck_dtype_row_bytes(wv_dtype, head_w_elems);

    const uint8_t *wq_bytes = (const uint8_t *)wq;
    const uint8_t *wk_bytes = (const uint8_t *)wk;
    const uint8_t *wv_bytes = (const uint8_t *)wv;

    for (int h = 0; h < num_heads; ++h) {
        const void *wq_h = (wq_dtype == CK_DT_FP32)
            ? (const void *)((const float *)wq + (size_t)h * head_w_elems)
            : (const void *)(wq_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;

        ck_gemm_nt_quant(input, wq_h, bq_h, q_h,
                         tokens, aligned_head_dim, aligned_embed_dim, wq_dtype);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const void *wk_h = (wk_dtype == CK_DT_FP32)
            ? (const void *)((const float *)wk + (size_t)h * head_w_elems)
            : (const void *)(wk_bytes + (size_t)h * wk_head_bytes);
        const void *wv_h = (wv_dtype == CK_DT_FP32)
            ? (const void *)((const float *)wv + (size_t)h * head_w_elems)
            : (const void *)(wv_bytes + (size_t)h * wv_head_bytes);

        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;

        float *k_h = k + (size_t)h * kv_head_stride;
        float *v_h = v + (size_t)h * kv_head_stride;

        ck_gemm_nt_quant(input, wk_h, bk_h, k_h,
                         tokens, aligned_head_dim, aligned_embed_dim, wk_dtype);
        ck_gemm_nt_quant(input, wv_h, bv_h, v_h,
                         tokens, aligned_head_dim, aligned_embed_dim, wv_dtype);
    }
}

static void ck_attention_project_head_major_q4_k(const float *attn_out,
                                                 const void *wo,
                                                 const float *bo,
                                                 float *out,
                                                 float *scratch,
                                                 int tokens,
                                                 int aligned_embed_dim,
                                                 int num_heads,
                                                 int aligned_head_dim)
{
    if (!attn_out || !wo || !out || !scratch) {
        return;
    }

    /* Flatten head-major [H, T, ad] into token-major [T, H*ad] */
    const int K = num_heads * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }

    const size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;

    for (int t = 0; t < tokens; ++t) {
        float *dst = scratch + (size_t)t * (size_t)aligned_embed_dim;
        for (int h = 0; h < num_heads; ++h) {
            const float *src = attn_out + (size_t)h * head_in_stride + (size_t)t * (size_t)aligned_head_dim;
            memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                   src,
                   (size_t)aligned_head_dim * sizeof(float));
        }
    }

    gemm_nt_q4_k(scratch, wo, bo, out,
                 tokens, aligned_embed_dim, aligned_embed_dim);
}

static void ck_attention_project_head_major_quant(const float *attn_out,
                                                  const void *wo,
                                                  const float *bo,
                                                  float *out,
                                                  float *scratch,
                                                  int tokens,
                                                  int aligned_embed_dim,
                                                  int num_heads,
                                                  int aligned_head_dim,
                                                  CKDataType wo_dtype)
{
    if (!attn_out || !wo || !out || !scratch) {
        return;
    }

    if (wo_dtype == CK_DT_FP32) {
        ck_attention_project_head_major(attn_out,
                                        (const float *)wo,
                                        bo,
                                        out,
                                        scratch,
                                        tokens,
                                        aligned_embed_dim,
                                        num_heads,
                                        aligned_head_dim);
        return;
    }

    const int K = num_heads * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }

    const size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;

    for (int t = 0; t < tokens; ++t) {
        float *dst = scratch + (size_t)t * (size_t)aligned_embed_dim;
        for (int h = 0; h < num_heads; ++h) {
            const float *src = attn_out + (size_t)h * head_in_stride + (size_t)t * (size_t)aligned_head_dim;
            memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                   src,
                   (size_t)aligned_head_dim * sizeof(float));
        }
    }

    ck_gemm_nt_quant(scratch, wo, bo, out,
                     tokens, aligned_embed_dim, aligned_embed_dim, wo_dtype);
}

static void ck_mlp_swiglu_forward_q4_k(const float *input,
                                      const void *w1,
                                      const float *b1,
                                      const void *w2,
                                      const float *b2,
                                      float *fc1_out,
                                      float *swiglu_out,
                                      float *output,
                                      int tokens,
                                      int aligned_embed_dim,
                                      int aligned_intermediate_dim)
{
    int up_dim = 2 * aligned_intermediate_dim;
    gemm_nt_q4_k(input, w1, b1, fc1_out,
                 tokens, up_dim, aligned_embed_dim);

    swiglu_forward(fc1_out, swiglu_out, tokens, aligned_intermediate_dim);

    gemm_nt_q4_k(swiglu_out, w2, b2, output,
                 tokens, aligned_embed_dim, aligned_intermediate_dim);
}

static void ck_mlp_swiglu_forward_quant(const float *input,
                                        const void *w1,
                                        const float *b1,
                                        CKDataType w1_dtype,
                                        const void *w2,
                                        const float *b2,
                                        CKDataType w2_dtype,
                                        float *fc1_out,
                                        float *swiglu_out,
                                        float *output,
                                        int tokens,
                                        int aligned_embed_dim,
                                        int aligned_intermediate_dim)
{
    int up_dim = 2 * aligned_intermediate_dim;
    ck_gemm_nt_quant(input, w1, b1, fc1_out,
                     tokens, up_dim, aligned_embed_dim, w1_dtype);

    swiglu_forward(fc1_out, swiglu_out, tokens, aligned_intermediate_dim);

    ck_gemm_nt_quant(swiglu_out, w2, b2, output,
                     tokens, aligned_embed_dim, aligned_intermediate_dim, w2_dtype);
}

static void ck_mlp_swiglu_forward_q4_k_q8_k(const float *input,
                                            const void *w1,
                                            const float *b1,
                                            const void *w2,
                                            const float *b2,
                                            float *fc1_out,
                                            float *swiglu_out,
                                            float *output,
                                            int aligned_embed_dim,
                                            int aligned_intermediate_dim)
{
    if (!input || !w1 || !w2 || !fc1_out || !swiglu_out || !output) {
        return;
    }
    if ((aligned_embed_dim % QK_K) != 0 || (aligned_intermediate_dim % QK_K) != 0) {
        return;
    }

    const int up_dim = 2 * aligned_intermediate_dim;
    const int q8_blocks_embed = aligned_embed_dim / QK_K;
    const int q8_blocks_inter = aligned_intermediate_dim / QK_K;
    const int q8_blocks_max = (q8_blocks_embed > q8_blocks_inter) ? q8_blocks_embed : q8_blocks_inter;
    block_q8_K q8_buf[q8_blocks_max];

    quantize_row_q8_k(input, q8_buf, aligned_embed_dim);
    gemm_nt_q4_k_q8_k(q8_buf, w1, b1, fc1_out,
                      /*M=*/1, /*N=*/up_dim, /*K=*/aligned_embed_dim);

    swiglu_forward(fc1_out, swiglu_out, /*tokens=*/1, aligned_intermediate_dim);

    quantize_row_q8_k(swiglu_out, q8_buf, aligned_intermediate_dim);
    gemm_nt_q4_k_q8_k(q8_buf, w2, b2, output,
                      /*M=*/1, /*N=*/aligned_embed_dim, /*K=*/aligned_intermediate_dim);
}

static void ck_qkv_project_head_major_ref(const float *input,
                                          const float *wq, const float *bq,
                                          const float *wk, const float *bk,
                                          const float *wv, const float *bv,
                                          float *q, float *k, float *v,
                                          int tokens,
                                          int kv_stride_tokens,
                                          int aligned_embed_dim,
                                          int num_heads,
                                          int num_kv_heads,
                                          int aligned_head_dim)
{
    if (!input || !wq || !wk || !wv || !q || !k || !v) {
        return;
    }
    if (kv_stride_tokens < tokens) {
        return;
    }

    size_t head_weight_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    size_t q_head_stride = (size_t)tokens * (size_t)aligned_head_dim;
    size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *wq_h = wq + (size_t)h * head_weight_stride;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * q_head_stride;

        gemm_naive_parallel(input, wq_h, bq_h, q_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *wk_h = wk + (size_t)h * head_weight_stride;
        const float *wv_h = wv + (size_t)h * head_weight_stride;

        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;

        float *k_h = k + (size_t)h * kv_head_stride;
        float *v_h = v + (size_t)h * kv_head_stride;

        gemm_naive_parallel(input, wk_h, bk_h, k_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
        gemm_naive_parallel(input, wv_h, bv_h, v_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
    }
}

static void ck_add_inplace(float *dst,
                           const float *src,
                           int tokens,
                           int aligned_embed_dim)
{
    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;
    for (size_t i = 0; i < total; ++i) {
        dst[i] += src[i];
    }
}

void ck_attention_project_head_major(const float *attn_out,
                                     const float *wo,
                                     const float *bo,
                                     float *out,
                                     float *scratch,
                                     int tokens,
                                     int aligned_embed_dim,
                                     int num_heads,
                                     int aligned_head_dim)
{
    if (!attn_out || !wo || !out) {
        return;
    }
    if (num_heads > 1 && !scratch) {
        return;
    }

    size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;
    size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *head_in = attn_out + (size_t)h * head_in_stride;
        const float *wo_h = wo + (size_t)h * head_weight_stride;

        if (h == 0) {
            gemm_blocked_serial(head_in, wo_h, bo, out,
                                tokens, aligned_embed_dim, aligned_head_dim);
        } else {
            gemm_blocked_serial(head_in, wo_h, NULL, scratch,
                                tokens, aligned_embed_dim, aligned_head_dim);
            ck_add_inplace(out, scratch, tokens, aligned_embed_dim);
        }
    }
}

static void ck_attention_project_head_major_ref(const float *attn_out,
                                                const float *wo,
                                                const float *bo,
                                                float *out,
                                                float *scratch,
                                                int tokens,
                                                int aligned_embed_dim,
                                                int num_heads,
                                                int aligned_head_dim)
{
    if (!attn_out || !wo || !out) {
        return;
    }
    if (num_heads > 1 && !scratch) {
        return;
    }

    size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;
    size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *head_in = attn_out + (size_t)h * head_in_stride;
        const float *wo_h = wo + (size_t)h * head_weight_stride;

        if (h == 0) {
            gemm_naive_parallel(head_in, wo_h, bo, out,
                                tokens, aligned_embed_dim, aligned_head_dim);
        } else {
            gemm_naive_parallel(head_in, wo_h, NULL, scratch,
                                tokens, aligned_embed_dim, aligned_head_dim);
            ck_add_inplace(out, scratch, tokens, aligned_embed_dim);
        }
    }
}

void ck_attention_project_head_major_backward(const float *d_out,
                                              const float *attn_out,
                                              const float *wo,
                                              float *d_attn_out,
                                              float *d_wo,
                                              float *d_bo,
                                              int tokens,
                                              int aligned_embed_dim,
                                              int num_heads,
                                              int aligned_head_dim)
{
    if (!d_out || !attn_out || !wo || !d_attn_out || !d_wo || !d_bo) {
        return;
    }

    // Bias gradient: sum over tokens once (bias is applied once in forward).
    for (int d = 0; d < aligned_embed_dim; ++d) {
        d_bo[d] = 0.0f;
    }
    for (int t = 0; t < tokens; ++t) {
        const float *row = d_out + (size_t)t * (size_t)aligned_embed_dim;
        for (int d = 0; d < aligned_embed_dim; ++d) {
            d_bo[d] += row[d];
        }
    }

    size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;
    size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

    float *tmp_b = (float *)calloc((size_t)aligned_embed_dim, sizeof(float));
    if (!tmp_b) {
        return;
    }

    for (int h = 0; h < num_heads; ++h) {
        const float *head_in = attn_out + (size_t)h * head_in_stride;
        const float *wo_h = wo + (size_t)h * head_weight_stride;
        float *d_head_in = d_attn_out + (size_t)h * head_in_stride;
        float *d_wo_h = d_wo + (size_t)h * head_weight_stride;

        memset(tmp_b, 0, (size_t)aligned_embed_dim * sizeof(float));
        fc2_backward_kernel(d_out,
                            head_in,
                            wo_h,
                            d_head_in,
                            d_wo_h,
                            tmp_b,
                            tokens,
                            aligned_head_dim,
                            aligned_embed_dim,
                            1);
    }

    free(tmp_b);
}

void ck_qkv_project_head_major_backward(const float *d_q,
                                        const float *d_k,
                                        const float *d_v,
                                        const float *input,
                                        const float *wq,
                                        const float *bq,
                                        const float *wk,
                                        const float *bk,
                                        const float *wv,
                                        const float *bv,
                                        float *d_input,
                                        float *d_wq,
                                        float *d_bq,
                                        float *d_wk,
                                        float *d_bk,
                                        float *d_wv,
                                        float *d_bv,
                                        float *scratch,
                                        int tokens,
                                        int aligned_embed_dim,
                                        int num_heads,
                                        int num_kv_heads,
                                        int aligned_head_dim,
                                        int num_threads)
{
    if (!d_q || !d_k || !d_v || !input || !wq || !wk || !wv ||
        !d_input || !d_wq || !d_bq || !d_wk || !d_bk || !d_wv || !d_bv || !scratch) {
        return;
    }

    size_t total_in = (size_t)tokens * (size_t)aligned_embed_dim;
    for (size_t i = 0; i < total_in; ++i) {
        d_input[i] = 0.0f;
    }

    size_t head_weight_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    size_t head_out_stride = (size_t)tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *d_q_h = d_q + (size_t)h * head_out_stride;
        const float *wq_h = wq + (size_t)h * head_weight_stride;
        float *d_wq_h = d_wq + (size_t)h * head_weight_stride;
        float *d_bq_h = d_bq + (size_t)h * (size_t)aligned_head_dim;

        fc2_backward_kernel(d_q_h,
                            input,
                            wq_h,
                            scratch,
                            d_wq_h,
                            d_bq_h,
                            tokens,
                            aligned_embed_dim,
                            aligned_head_dim,
                            num_threads);
        ck_add_inplace(d_input, scratch, tokens, aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *d_k_h = d_k + (size_t)h * head_out_stride;
        const float *d_v_h = d_v + (size_t)h * head_out_stride;

        const float *wk_h = wk + (size_t)h * head_weight_stride;
        const float *wv_h = wv + (size_t)h * head_weight_stride;

        float *d_wk_h = d_wk + (size_t)h * head_weight_stride;
        float *d_wv_h = d_wv + (size_t)h * head_weight_stride;

        float *d_bk_h = d_bk + (size_t)h * (size_t)aligned_head_dim;
        float *d_bv_h = d_bv + (size_t)h * (size_t)aligned_head_dim;

        fc2_backward_kernel(d_k_h,
                            input,
                            wk_h,
                            scratch,
                            d_wk_h,
                            d_bk_h,
                            tokens,
                            aligned_embed_dim,
                            aligned_head_dim,
                            num_threads);
        ck_add_inplace(d_input, scratch, tokens, aligned_embed_dim);

        fc2_backward_kernel(d_v_h,
                            input,
                            wv_h,
                            scratch,
                            d_wv_h,
                            d_bv_h,
                            tokens,
                            aligned_embed_dim,
                            aligned_head_dim,
                            num_threads);
        ck_add_inplace(d_input, scratch, tokens, aligned_embed_dim);
    }
}

void ck_mlp_swiglu_forward(const float *input,
                           const float *w1,
                           const float *b1,
                           const float *w2,
                           const float *b2,
                           float *fc1_out,
                           float *swiglu_out,
                           float *output,
                           int tokens,
                           int aligned_embed_dim,
                           int aligned_intermediate_dim)
{
    int up_dim = 2 * aligned_intermediate_dim;
    gemm_blocked_serial(input, w1, b1, fc1_out,
                        tokens, up_dim, aligned_embed_dim);

    swiglu_forward(fc1_out, swiglu_out, tokens, aligned_intermediate_dim);

    gemm_blocked_serial(swiglu_out, w2, b2, output,
                        tokens, aligned_embed_dim, aligned_intermediate_dim);
}

static void ck_mlp_swiglu_forward_ref(const float *input,
                                      const float *w1,
                                      const float *b1,
                                      const float *w2,
                                      const float *b2,
                                      float *fc1_out,
                                      float *swiglu_out,
                                      float *output,
                                      int tokens,
                                      int aligned_embed_dim,
                                      int aligned_intermediate_dim)
{
    int up_dim = 2 * aligned_intermediate_dim;
    gemm_naive_parallel(input, w1, b1, fc1_out,
                        tokens, up_dim, aligned_embed_dim);

    swiglu_forward(fc1_out, swiglu_out, tokens, aligned_intermediate_dim);

    gemm_naive_parallel(swiglu_out, w2, b2, output,
                        tokens, aligned_embed_dim, aligned_intermediate_dim);
}

void ck_layer_forward_rmsnorm_swiglu(const CKLayerForwardParams *p)
{
    if (!p) {
        return;
    }

    rmsnorm_forward(p->input,
                    p->ln1_gamma,
                    p->ln1_out,
                    p->ln1_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_qkv_project_head_major(p->ln1_out,
                              p->wq, p->bq,
                              p->wk, p->bk,
                              p->wv, p->bv,
                              p->q, p->k, p->v,
                              p->tokens,
                              p->tokens,
                              p->aligned_embed_dim,
                              p->num_heads,
                              p->num_kv_heads,
                              p->aligned_head_dim);

    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(p->q,
                        p->k,
                        p->rope_cos,
                        p->rope_sin,
                        p->num_heads,
                        p->num_kv_heads,
                        p->tokens,
                        p->head_dim,
                        p->aligned_head_dim,
                        p->rope_pos_offset);
    }

    if (p->scores) {
        attention_forward_causal_head_major_gqa(p->q,
                                               p->k,
                                               p->v,
                                               p->scores,
                                               p->attn_out,
                                               p->num_heads,
                                               p->num_kv_heads,
                                               p->tokens,
                                               p->head_dim,
                                               p->aligned_head_dim,
                                               p->aligned_context_window);
    } else {
        attention_forward_causal_head_major_gqa_flash(p->q,
                                                     p->k,
                                                     p->v,
                                                     p->attn_out,
                                                     p->num_heads,
                                                     p->num_kv_heads,
                                                     p->tokens,
                                                     p->head_dim,
                                                     p->aligned_head_dim);
    }

    ck_attention_project_head_major(p->attn_out,
                                    p->wo,
                                    p->bo,
                                    p->proj_tmp,
                                    p->proj_scratch,
                                    p->tokens,
                                    p->aligned_embed_dim,
                                    p->num_heads,
                                    p->aligned_head_dim);

    ck_residual_add_token_major(p->input,
                                p->proj_tmp,
                                p->residual1,
                                p->tokens,
                                p->aligned_embed_dim);

    rmsnorm_forward(p->residual1,
                    p->ln2_gamma,
                    p->ln2_out,
                    p->ln2_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_mlp_swiglu_forward(p->ln2_out,
                          p->w1,
                          p->b1,
                          p->w2,
                          p->b2,
                          p->fc1_out,
                          p->swiglu_out,
                          p->mlp_out,
                          p->tokens,
                          p->aligned_embed_dim,
                          p->aligned_intermediate_dim);

    ck_residual_add_token_major(p->residual1,
                                p->mlp_out,
                                p->output,
                                p->tokens,
                                p->aligned_embed_dim);
}

void ck_layer_forward_rmsnorm_swiglu_ref(const CKLayerForwardParams *p)
{
    if (!p) {
        return;
    }

    rmsnorm_forward(p->input,
                    p->ln1_gamma,
                    p->ln1_out,
                    p->ln1_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_qkv_project_head_major_ref(p->ln1_out,
                                  p->wq, p->bq,
                                  p->wk, p->bk,
                                  p->wv, p->bv,
                                  p->q, p->k, p->v,
                                  p->tokens,
                                  p->tokens,
                                  p->aligned_embed_dim,
                                  p->num_heads,
                                  p->num_kv_heads,
                                  p->aligned_head_dim);

    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(p->q,
                        p->k,
                        p->rope_cos,
                        p->rope_sin,
                        p->num_heads,
                        p->num_kv_heads,
                        p->tokens,
                        p->head_dim,
                        p->aligned_head_dim,
                        p->rope_pos_offset);
    }

    if (p->scores) {
        attention_forward_causal_head_major_gqa(p->q,
                                               p->k,
                                               p->v,
                                               p->scores,
                                               p->attn_out,
                                               p->num_heads,
                                               p->num_kv_heads,
                                               p->tokens,
                                               p->head_dim,
                                               p->aligned_head_dim,
                                               p->aligned_context_window);
    } else {
        attention_forward_causal_head_major_gqa_flash(p->q,
                                                     p->k,
                                                     p->v,
                                                     p->attn_out,
                                                     p->num_heads,
                                                     p->num_kv_heads,
                                                     p->tokens,
                                                     p->head_dim,
                                                     p->aligned_head_dim);
    }

    ck_attention_project_head_major_ref(p->attn_out,
                                        p->wo,
                                        p->bo,
                                        p->proj_tmp,
                                        p->proj_scratch,
                                        p->tokens,
                                        p->aligned_embed_dim,
                                        p->num_heads,
                                        p->aligned_head_dim);

    ck_residual_add_token_major(p->input,
                                p->proj_tmp,
                                p->residual1,
                                p->tokens,
                                p->aligned_embed_dim);

    rmsnorm_forward(p->residual1,
                    p->ln2_gamma,
                    p->ln2_out,
                    p->ln2_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_mlp_swiglu_forward_ref(p->ln2_out,
                              p->w1,
                              p->b1,
                              p->w2,
                              p->b2,
                              p->fc1_out,
                              p->swiglu_out,
                              p->mlp_out,
                              p->tokens,
                              p->aligned_embed_dim,
                              p->aligned_intermediate_dim);

    ck_residual_add_token_major(p->residual1,
                                p->mlp_out,
                                p->output,
                                p->tokens,
                                p->aligned_embed_dim);
}

void ck_mlp_swiglu_forward_fused_token(const float *input_row,
                                       const float *w1,
                                       const float *b1,
                                       const float *w2,
                                       const float *b2,
                                       float *swiglu_row,
                                       float *output_row,
                                       int aligned_embed_dim,
                                       int aligned_intermediate_dim)
{
    if (!input_row || !w1 || !w2 || !swiglu_row || !output_row) {
        return;
    }

    const float *w_gate = w1;
    const float *w_up = w1 + (size_t)aligned_intermediate_dim * (size_t)aligned_embed_dim;
    const float *b_gate = b1;
    const float *b_up = b1 ? (b1 + aligned_intermediate_dim) : NULL;

    gemm_swiglu_fused(input_row,
                      w_gate,
                      w_up,
                      b_gate,
                      b_up,
                      swiglu_row,
                      /*M=*/1,
                      /*N=*/aligned_intermediate_dim,
                      /*K=*/aligned_embed_dim);

    gemm_blocked_serial(swiglu_row, w2, b2, output_row,
                        /*M=*/1,
                        /*N=*/aligned_embed_dim,
                        /*K=*/aligned_intermediate_dim);
}

void ck_mlp_swiglu_forward_fully_fused_token(const float *input_row,
                                              const float *w1,
                                              const float *b1,
                                              const float *w2,
                                              const float *b2,
                                              float *output_row,
                                              int aligned_embed_dim,
                                              int aligned_intermediate_dim)
{
    if (!input_row || !w1 || !w2 || !output_row) {
        return;
    }

    // Split w1 into gate and up projections
    // w1 layout: [2 * aligned_intermediate_dim, aligned_embed_dim]
    //   First half: W_gate [aligned_intermediate_dim, aligned_embed_dim]
    //   Second half: W_up [aligned_intermediate_dim, aligned_embed_dim]
    const float *w_gate = w1;
    const float *w_up = w1 + (size_t)aligned_intermediate_dim * (size_t)aligned_embed_dim;

    // Split b1 into gate and up biases (if present)
    const float *b_gate = b1;
    const float *b_up = b1 ? (b1 + aligned_intermediate_dim) : NULL;

    // w2 is W_down: [aligned_embed_dim, aligned_intermediate_dim]
    const float *w_down = w2;
    const float *b_down = b2;

    // Call the fully fused kernel - eliminates DRAM round-trip for swiglu
    // Uses aligned dimensions since weights are stored with alignment padding
    fused_mlp_swiglu_decode_v2(input_row,
                                w_gate,
                                w_up,
                                w_down,
                                b_gate,
                                b_up,
                                b_down,
                                output_row,
                                aligned_embed_dim,
                                aligned_intermediate_dim);
}

void ck_layer_forward_rmsnorm_swiglu_decode(const CKLayerForwardParams *p,
                                           int token_index,
                                           int cache_capacity)
{
    if (!p) {
        return;
    }
    if (!p->input || !p->ln1_gamma || !p->ln2_gamma || !p->ln1_out || !p->ln2_out ||
        !p->wq || !p->wk || !p->wv || !p->wo || !p->w1 || !p->w2 ||
        !p->k || !p->v ||
        !p->proj_tmp || !p->residual1 || !p->fc1_out || !p->swiglu_out || !p->mlp_out || !p->output) {
        return;
    }
    if (token_index < 0 || cache_capacity <= 0 || token_index >= cache_capacity) {
        return;
    }
    if (p->num_heads <= 0 || p->num_kv_heads <= 0 || p->aligned_head_dim <= 0) {
        return;
    }

    const int D = p->embed_dim;
    const int aligned_D = p->aligned_embed_dim;
    const int H = p->num_heads;
    const int H_kv = p->num_kv_heads;
    const int hd = p->head_dim;
    const int ad = p->aligned_head_dim;
    const int aligned_intermediate = p->aligned_intermediate_dim;

    /* Decode buffers are single-token; token_index only applies to KV cache. */
    const size_t token_slot = 0;
    const float *input_row = p->input + token_slot * (size_t)aligned_D;
    float *ln1_row = p->ln1_out + token_slot * (size_t)aligned_D;
    float *ln2_row = p->ln2_out + token_slot * (size_t)aligned_D;
    float *proj_row = p->proj_tmp + token_slot * (size_t)aligned_D;
    float *residual_row = p->residual1 + token_slot * (size_t)aligned_D;
    float *mlp_row = p->mlp_out + token_slot * (size_t)aligned_D;
    float *out_row = p->output + token_slot * (size_t)aligned_D;

    float ln1_rstd_tmp = 0.0f;
    float ln2_rstd_tmp = 0.0f;
    float *ln1_rstd = p->ln1_rstd ? (p->ln1_rstd + token_slot) : &ln1_rstd_tmp;
    float *ln2_rstd = p->ln2_rstd ? (p->ln2_rstd + token_slot) : &ln2_rstd_tmp;

    // Scratch for a single token in head-major layout: [head, aligned_head_dim].
    size_t q_elems = (size_t)H * (size_t)ad;
    size_t kv_elems = (size_t)H_kv * (size_t)ad;
    float q_token[q_elems];
    float k_token[kv_elems];
    float v_token[kv_elems];
    float attn_token[q_elems];

    // LN1 / RMSNorm.
    rmsnorm_forward(input_row,
                    p->ln1_gamma,
                    ln1_row,
                    ln1_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    // Project Q/K/V for the new token.
    ck_qkv_project_head_major_token(ln1_row,
                                    p->wq, p->bq,
                                    p->wk, p->bk,
                                    p->wv, p->bv,
                                    q_token, k_token, v_token,
                                    aligned_D,
                                    H,
                                    H_kv,
                                    ad);

    // RoPE for the new token at absolute position `p->rope_pos_offset`.
    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(q_token,
                        k_token,
                        p->rope_cos,
                        p->rope_sin,
                        H,
                        H_kv,
                        /*num_tokens=*/1,
                        hd,
                        ad,
                        p->rope_pos_offset);
    }

    // Update KV cache (stores k/v for this token and clears padded lanes).
    kv_cache_write_head_major(k_token,
                              v_token,
                              p->k,
                              p->v,
                              H_kv,
                              token_index,
                              cache_capacity,
                              hd,
                              ad);

    // Decode attention for this token using the KV cache.
    ck_attention_flash_decode_wrapper(q_token,
                                   p->k,
                                   p->v,
                                   attn_token,
                                   H,
                                   H_kv,
                                   /*kv_tokens=*/token_index + 1,
                                   cache_capacity,
                                   hd,
                                   ad);

    // Output projection (Wo) into token-major buffer (decode-specialized).
    ck_attention_project_head_major_decode_token(attn_token,
                                                 p->wo,
                                                 p->bo,
                                                 proj_row,
                                                 D,
                                                 aligned_D,
                                                 H,
                                                 ad);

    // Residual + LN2 / RMSNorm.
    ck_residual_add_token_major(input_row,
                                proj_row,
                                residual_row,
                                /*tokens=*/1,
                                aligned_D);

    rmsnorm_forward(residual_row,
                    p->ln2_gamma,
                    ln2_row,
                    ln2_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    // MLP block for this token.
    int up_dim = 2 * aligned_intermediate;
    float *fc1_row = p->fc1_out + token_slot * (size_t)up_dim;
    float *swiglu_row = p->swiglu_out + token_slot * (size_t)aligned_intermediate;

    ck_mlp_swiglu_forward(ln2_row,
                          p->w1,
                          p->b1,
                          p->w2,
                          p->b2,
                          fc1_row,
                          swiglu_row,
                          mlp_row,
                          /*tokens=*/1,
                          aligned_D,
                          aligned_intermediate);

    // Final residual.
    ck_residual_add_token_major(residual_row,
                                mlp_row,
                                out_row,
                                /*tokens=*/1,
                                aligned_D);
}

void ck_layer_forward_rmsnorm_swiglu_decode_fused(const CKLayerForwardParams *p,
                                                 int token_index,
                                                 int cache_capacity)
{
    if (!p) {
        return;
    }
    if (!p->input || !p->ln1_gamma || !p->ln2_gamma || !p->ln1_out || !p->ln2_out ||
        !p->wq || !p->wk || !p->wv || !p->wo || !p->w1 || !p->w2 ||
        !p->k || !p->v || !p->swiglu_out ||
        !p->proj_tmp || !p->residual1 || !p->mlp_out || !p->output) {
        return;
    }
    if (token_index < 0 || cache_capacity <= 0 || token_index >= cache_capacity) {
        return;
    }
    if (p->num_heads <= 0 || p->num_kv_heads <= 0 || p->aligned_head_dim <= 0) {
        return;
    }

    const int D = p->embed_dim;
    const int aligned_D = p->aligned_embed_dim;
    const int H = p->num_heads;
    const int H_kv = p->num_kv_heads;
    const int hd = p->head_dim;
    const int ad = p->aligned_head_dim;
    const int aligned_intermediate = p->aligned_intermediate_dim;

    /* Decode buffers are single-token; token_index only applies to KV cache. */
    const size_t token_slot = 0;
    const float *input_row = p->input + token_slot * (size_t)aligned_D;
    float *ln1_row = p->ln1_out + token_slot * (size_t)aligned_D;
    float *ln2_row = p->ln2_out + token_slot * (size_t)aligned_D;
    float *proj_row = p->proj_tmp + token_slot * (size_t)aligned_D;
    float *residual_row = p->residual1 + token_slot * (size_t)aligned_D;
    float *swiglu_row = p->swiglu_out + token_slot * (size_t)aligned_intermediate;
    float *mlp_row = p->mlp_out + token_slot * (size_t)aligned_D;
    float *out_row = p->output + token_slot * (size_t)aligned_D;

    float ln1_rstd_tmp = 0.0f;
    float ln2_rstd_tmp = 0.0f;
    float *ln1_rstd = p->ln1_rstd ? (p->ln1_rstd + token_slot) : &ln1_rstd_tmp;
    float *ln2_rstd = p->ln2_rstd ? (p->ln2_rstd + token_slot) : &ln2_rstd_tmp;

    // Scratch for a single token in head-major layout: [head, aligned_head_dim].
    size_t q_elems = (size_t)H * (size_t)ad;
    size_t kv_elems = (size_t)H_kv * (size_t)ad;
    float q_token[q_elems];
    float k_token[kv_elems];
    float v_token[kv_elems];
    float attn_token[q_elems];

    // LN1 / RMSNorm.
    rmsnorm_forward(input_row,
                    p->ln1_gamma,
                    ln1_row,
                    ln1_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    // Project Q/K/V for the new token.
    ck_qkv_project_head_major_token(ln1_row,
                                    p->wq, p->bq,
                                    p->wk, p->bk,
                                    p->wv, p->bv,
                                    q_token, k_token, v_token,
                                    aligned_D,
                                    H,
                                    H_kv,
                                    ad);

    // RoPE for the new token at absolute position `p->rope_pos_offset`.
    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(q_token,
                        k_token,
                        p->rope_cos,
                        p->rope_sin,
                        H,
                        H_kv,
                        /*num_tokens=*/1,
                        hd,
                        ad,
                        p->rope_pos_offset);
    }

    // Update KV cache (stores k/v for this token and clears padded lanes).
    kv_cache_write_head_major(k_token,
                              v_token,
                              p->k,
                              p->v,
                              H_kv,
                              token_index,
                              cache_capacity,
                              hd,
                              ad);

    // Decode attention for this token using the KV cache.
    ck_attention_flash_decode_wrapper(q_token,
                                   p->k,
                                   p->v,
                                   attn_token,
                                   H,
                                   H_kv,
                                   /*kv_tokens=*/token_index + 1,
                                   cache_capacity,
                                   hd,
                                   ad);

    // Output projection (Wo) into token-major buffer (decode-specialized).
    ck_attention_project_head_major_decode_token(attn_token,
                                                 p->wo,
                                                 p->bo,
                                                 proj_row,
                                                 D,
                                                 aligned_D,
                                                 H,
                                                 ad);

    // Residual + LN2 / RMSNorm.
    ck_residual_add_token_major(input_row,
                                proj_row,
                                residual_row,
                                /*tokens=*/1,
                                aligned_D);

    rmsnorm_forward(residual_row,
                    p->ln2_gamma,
                    ln2_row,
                    ln2_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    // MLP block for this token (fully fused - all 3 projections in one pass).
    // Eliminates DRAM round-trip for swiglu intermediate values.
    ck_mlp_swiglu_forward_fully_fused_token(ln2_row,
                                             p->w1,
                                             p->b1,
                                             p->w2,
                                             p->b2,
                                             mlp_row,
                                             aligned_D,
                                             aligned_intermediate);

    // Final residual.
    ck_residual_add_token_major(residual_row,
                                mlp_row,
                                out_row,
                                /*tokens=*/1,
                                aligned_D);
}

static void ck_qkv_project_head_major_token_q4_k(const float *input_row,
                                                 const void *wq, const float *bq,
                                                 const void *wk, const float *bk,
                                                 const void *wv, const float *bv,
                                                 float *q_token,
                                                 float *k_token,
                                                 float *v_token,
                                                 int aligned_embed_dim,
                                                 int num_heads,
                                                 int num_kv_heads,
                                                 int aligned_head_dim)
{
    if (!input_row || !wq || !wk || !wv || !q_token || !k_token || !v_token) {
        return;
    }

    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t head_w_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);

    const uint8_t *wq_bytes = (const uint8_t *)wq;
    const uint8_t *wk_bytes = (const uint8_t *)wk;
    const uint8_t *wv_bytes = (const uint8_t *)wv;

    for (int h = 0; h < num_heads; ++h) {
        const void *wq_h = wq_bytes + (size_t)h * head_w_bytes;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q_token + (size_t)h * (size_t)aligned_head_dim;
        gemm_nt_q4_k(input_row, wq_h, bq_h, q_h,
                     /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const void *wk_h = wk_bytes + (size_t)h * head_w_bytes;
        const void *wv_h = wv_bytes + (size_t)h * head_w_bytes;
        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k_token + (size_t)h * (size_t)aligned_head_dim;
        float *v_h = v_token + (size_t)h * (size_t)aligned_head_dim;
        gemm_nt_q4_k(input_row, wk_h, bk_h, k_h,
                     /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
        gemm_nt_q4_k(input_row, wv_h, bv_h, v_h,
                     /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
    }
}

static void ck_qkv_project_head_major_token_q4_k_q8_k(const block_q8_K *input_q8,
                                                      const void *wq, const float *bq,
                                                      const void *wk, const float *bk,
                                                      const void *wv, const float *bv,
                                                      float *q_token,
                                                      float *k_token,
                                                      float *v_token,
                                                      int aligned_embed_dim,
                                                      int num_heads,
                                                      int num_kv_heads,
                                                      int aligned_head_dim)
{
    if (!input_q8 || !wq || !wk || !wv || !q_token || !k_token || !v_token) {
        return;
    }

    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t head_w_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);

    const uint8_t *wq_bytes = (const uint8_t *)wq;
    const uint8_t *wk_bytes = (const uint8_t *)wk;
    const uint8_t *wv_bytes = (const uint8_t *)wv;

    for (int h = 0; h < num_heads; ++h) {
        const void *wq_h = wq_bytes + (size_t)h * head_w_bytes;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q_token + (size_t)h * (size_t)aligned_head_dim;
        gemm_nt_q4_k_q8_k(input_q8, wq_h, bq_h, q_h,
                          /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const void *wk_h = wk_bytes + (size_t)h * head_w_bytes;
        const void *wv_h = wv_bytes + (size_t)h * head_w_bytes;
        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k_token + (size_t)h * (size_t)aligned_head_dim;
        float *v_h = v_token + (size_t)h * (size_t)aligned_head_dim;
        gemm_nt_q4_k_q8_k(input_q8, wk_h, bk_h, k_h,
                          /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
        gemm_nt_q4_k_q8_k(input_q8, wv_h, bv_h, v_h,
                          /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
    }
}

static void ck_qkv_project_head_major_q4_k_q8_k(const float *input,
                                                const void *wq, const float *bq,
                                                const void *wk, const float *bk,
                                                const void *wv, const float *bv,
                                                float *q, float *k, float *v,
                                                int tokens,
                                                int kv_stride_tokens,
                                                int aligned_embed_dim,
                                                int num_heads,
                                                int num_kv_heads,
                                                int aligned_head_dim)
{
    if (!input || !wq || !wk || !wv || !q || !k || !v) {
        return;
    }
    if (tokens <= 0 || aligned_embed_dim <= 0) {
        return;
    }
    if (kv_stride_tokens < tokens) {
        return;
    }
    if ((aligned_embed_dim % QK_K) != 0) {
        return;
    }

    const int q8_blocks = aligned_embed_dim / QK_K;
    block_q8_K q8_buf[q8_blocks];
    const size_t q_head_stride = (size_t)tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;

    float q_token[num_heads * aligned_head_dim];
    float k_token[num_kv_heads * aligned_head_dim];
    float v_token[num_kv_heads * aligned_head_dim];

    for (int t = 0; t < tokens; ++t) {
        const float *input_row = input + (size_t)t * (size_t)aligned_embed_dim;
        quantize_row_q8_k(input_row, q8_buf, aligned_embed_dim);

        ck_qkv_project_head_major_token_q4_k_q8_k(q8_buf,
                                                  wq, bq,
                                                  wk, bk,
                                                  wv, bv,
                                                  q_token,
                                                  k_token,
                                                  v_token,
                                                  aligned_embed_dim,
                                                  num_heads,
                                                  num_kv_heads,
                                                  aligned_head_dim);

        for (int h = 0; h < num_heads; ++h) {
            float *q_dst = q + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
            memcpy(q_dst,
                   q_token + (size_t)h * (size_t)aligned_head_dim,
                   (size_t)aligned_head_dim * sizeof(float));
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            float *k_dst = k + (size_t)h * kv_head_stride + (size_t)t * (size_t)aligned_head_dim;
            float *v_dst = v + (size_t)h * kv_head_stride + (size_t)t * (size_t)aligned_head_dim;
            memcpy(k_dst,
                   k_token + (size_t)h * (size_t)aligned_head_dim,
                   (size_t)aligned_head_dim * sizeof(float));
            memcpy(v_dst,
                   v_token + (size_t)h * (size_t)aligned_head_dim,
                   (size_t)aligned_head_dim * sizeof(float));
        }
    }
}

static void ck_attention_project_head_major_q4_k_q8_k(const float *attn_out,
                                                      const void *wo,
                                                      const float *bo,
                                                      float *out,
                                                      int tokens,
                                                      int aligned_embed_dim,
                                                      int num_heads,
                                                      int aligned_head_dim)
{
    if (!attn_out || !wo || !out) {
        return;
    }
    if (tokens <= 0 || aligned_embed_dim <= 0) {
        return;
    }
    if ((aligned_embed_dim % QK_K) != 0) {
        return;
    }

    const int K = num_heads * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }

    const int q8_blocks = aligned_embed_dim / QK_K;
    block_q8_K q8_buf[q8_blocks];
    float attn_token[aligned_embed_dim];
    const size_t head_stride = (size_t)tokens * (size_t)aligned_head_dim;

    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const float *src = attn_out + (size_t)h * head_stride + (size_t)t * (size_t)aligned_head_dim;
            memcpy(attn_token + (size_t)h * (size_t)aligned_head_dim,
                   src,
                   (size_t)aligned_head_dim * sizeof(float));
        }

        quantize_row_q8_k(attn_token, q8_buf, aligned_embed_dim);
        gemm_nt_q4_k_q8_k(q8_buf, wo, bo,
                          out + (size_t)t * (size_t)aligned_embed_dim,
                          /*M=*/1, /*N=*/aligned_embed_dim, /*K=*/aligned_embed_dim);
    }
}

static void ck_mlp_swiglu_forward_q4_k_q8_k_prefill(const float *input,
                                                    const void *w1,
                                                    const float *b1,
                                                    const void *w2,
                                                    const float *b2,
                                                    float *fc1_out,
                                                    float *swiglu_out,
                                                    float *output,
                                                    int tokens,
                                                    int aligned_embed_dim,
                                                    int aligned_intermediate_dim)
{
    if (!input || !w1 || !w2 || !fc1_out || !swiglu_out || !output) {
        return;
    }
    if (tokens <= 0) {
        return;
    }
    if ((aligned_embed_dim % QK_K) != 0 || (aligned_intermediate_dim % QK_K) != 0) {
        return;
    }

    const int up_dim = 2 * aligned_intermediate_dim;
    const int q8_blocks_embed = aligned_embed_dim / QK_K;
    const int q8_blocks_inter = aligned_intermediate_dim / QK_K;
    const int q8_blocks_max = (q8_blocks_embed > q8_blocks_inter) ? q8_blocks_embed : q8_blocks_inter;
    block_q8_K q8_buf[q8_blocks_max];

    for (int t = 0; t < tokens; ++t) {
        const float *input_row = input + (size_t)t * (size_t)aligned_embed_dim;
        float *fc1_row = fc1_out + (size_t)t * (size_t)up_dim;

        quantize_row_q8_k(input_row, q8_buf, aligned_embed_dim);
        gemm_nt_q4_k_q8_k(q8_buf, w1, b1, fc1_row,
                          /*M=*/1, /*N=*/up_dim, /*K=*/aligned_embed_dim);
    }

    swiglu_forward(fc1_out, swiglu_out, tokens, aligned_intermediate_dim);

    for (int t = 0; t < tokens; ++t) {
        const float *swiglu_row = swiglu_out + (size_t)t * (size_t)aligned_intermediate_dim;
        float *out_row = output + (size_t)t * (size_t)aligned_embed_dim;

        quantize_row_q8_k(swiglu_row, q8_buf, aligned_intermediate_dim);
        gemm_nt_q4_k_q8_k(q8_buf, w2, b2, out_row,
                          /*M=*/1, /*N=*/aligned_embed_dim, /*K=*/aligned_intermediate_dim);
    }
}

static void ck_qkv_project_head_major_token_quant(const float *input_row,
                                                  const void *wq, const float *bq, CKDataType wq_dtype,
                                                  const void *wk, const float *bk, CKDataType wk_dtype,
                                                  const void *wv, const float *bv, CKDataType wv_dtype,
                                                  float *q_token,
                                                  float *k_token,
                                                  float *v_token,
                                                  int aligned_embed_dim,
                                                  int num_heads,
                                                  int num_kv_heads,
                                                  int aligned_head_dim)
{
    if (!input_row || !wq || !wk || !wv || !q_token || !k_token || !v_token) {
        return;
    }

    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wq_head_bytes = ck_dtype_row_bytes(wq_dtype, head_w_elems);
    const size_t wk_head_bytes = ck_dtype_row_bytes(wk_dtype, head_w_elems);
    const size_t wv_head_bytes = ck_dtype_row_bytes(wv_dtype, head_w_elems);

    const uint8_t *wq_bytes = (const uint8_t *)wq;
    const uint8_t *wk_bytes = (const uint8_t *)wk;
    const uint8_t *wv_bytes = (const uint8_t *)wv;

    for (int h = 0; h < num_heads; ++h) {
        const void *wq_h = (wq_dtype == CK_DT_FP32)
            ? (const void *)((const float *)wq + (size_t)h * head_w_elems)
            : (const void *)(wq_bytes + (size_t)h * wq_head_bytes);
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q_token + (size_t)h * (size_t)aligned_head_dim;
        ck_gemm_nt_quant(input_row, wq_h, bq_h, q_h,
                         /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim, wq_dtype);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const void *wk_h = (wk_dtype == CK_DT_FP32)
            ? (const void *)((const float *)wk + (size_t)h * head_w_elems)
            : (const void *)(wk_bytes + (size_t)h * wk_head_bytes);
        const void *wv_h = (wv_dtype == CK_DT_FP32)
            ? (const void *)((const float *)wv + (size_t)h * head_w_elems)
            : (const void *)(wv_bytes + (size_t)h * wv_head_bytes);
        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k_token + (size_t)h * (size_t)aligned_head_dim;
        float *v_h = v_token + (size_t)h * (size_t)aligned_head_dim;
        ck_gemm_nt_quant(input_row, wk_h, bk_h, k_h,
                         /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim, wk_dtype);
        ck_gemm_nt_quant(input_row, wv_h, bv_h, v_h,
                         /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim, wv_dtype);
    }
}

void ck_layer_forward_rmsnorm_swiglu_q4_k(const CKLayerForwardParamsQ4K *p)
{
    if (!p) {
        return;
    }

    const int aligned_D = p->aligned_embed_dim;
    const int aligned_intermediate = p->aligned_intermediate_dim;

    rmsnorm_forward(p->input,
                    p->ln1_gamma,
                    p->ln1_out,
                    p->ln1_rstd,
                    p->tokens,
                    p->embed_dim,
                    aligned_D,
                    p->eps);

    if (ck_q8k_activations_enabled()) {
        if ((aligned_D % QK_K) == 0 && (aligned_intermediate % QK_K) == 0) {
            ck_qkv_project_head_major_q4_k_q8_k(p->ln1_out,
                                                p->wq, p->bq,
                                                p->wk, p->bk,
                                                p->wv, p->bv,
                                                p->q, p->k, p->v,
                                                p->tokens,
                                                p->tokens,
                                                aligned_D,
                                                p->num_heads,
                                                p->num_kv_heads,
                                                p->aligned_head_dim);

            if (p->rope_cos && p->rope_sin) {
                rope_forward_qk(p->q,
                                p->k,
                                p->rope_cos,
                                p->rope_sin,
                                p->num_heads,
                                p->num_kv_heads,
                                p->tokens,
                                p->head_dim,
                                p->aligned_head_dim,
                                p->rope_pos_offset);
            }

            if (p->scores) {
                attention_forward_causal_head_major_gqa(p->q,
                                                       p->k,
                                                       p->v,
                                                       p->scores,
                                                       p->attn_out,
                                                       p->num_heads,
                                                       p->num_kv_heads,
                                                       p->tokens,
                                                       p->head_dim,
                                                       p->aligned_head_dim,
                                                       p->aligned_context_window);
            } else {
                attention_forward_causal_head_major_gqa_flash(p->q,
                                                             p->k,
                                                             p->v,
                                                             p->attn_out,
                                                             p->num_heads,
                                                             p->num_kv_heads,
                                                             p->tokens,
                                                             p->head_dim,
                                                             p->aligned_head_dim);
            }

            ck_attention_project_head_major_q4_k_q8_k(p->attn_out,
                                                      p->wo,
                                                      p->bo,
                                                      p->proj_tmp,
                                                      p->tokens,
                                                      aligned_D,
                                                      p->num_heads,
                                                      p->aligned_head_dim);

            ck_residual_add_token_major(p->input,
                                        p->proj_tmp,
                                        p->residual1,
                                        p->tokens,
                                        aligned_D);

            rmsnorm_forward(p->residual1,
                            p->ln2_gamma,
                            p->ln2_out,
                            p->ln2_rstd,
                            p->tokens,
                            p->embed_dim,
                            aligned_D,
                            p->eps);

            ck_mlp_swiglu_forward_q4_k_q8_k_prefill(p->ln2_out,
                                                    p->w1,
                                                    p->b1,
                                                    p->w2,
                                                    p->b2,
                                                    p->fc1_out,
                                                    p->swiglu_out,
                                                    p->mlp_out,
                                                    p->tokens,
                                                    aligned_D,
                                                    aligned_intermediate);

            ck_residual_add_token_major(p->residual1,
                                        p->mlp_out,
                                        p->output,
                                        p->tokens,
                                        aligned_D);
            return;
        }
    }

    ck_qkv_project_head_major_q4_k(p->ln1_out,
                                  p->wq, p->bq,
                                  p->wk, p->bk,
                                  p->wv, p->bv,
                                  p->q, p->k, p->v,
                                  p->tokens,
                                  p->tokens,
                                  aligned_D,
                                  p->num_heads,
                                  p->num_kv_heads,
                                  p->aligned_head_dim);

    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(p->q,
                        p->k,
                        p->rope_cos,
                        p->rope_sin,
                        p->num_heads,
                        p->num_kv_heads,
                        p->tokens,
                        p->head_dim,
                        p->aligned_head_dim,
                        p->rope_pos_offset);
    }

    if (p->scores) {
        attention_forward_causal_head_major_gqa(p->q,
                                               p->k,
                                               p->v,
                                               p->scores,
                                               p->attn_out,
                                               p->num_heads,
                                               p->num_kv_heads,
                                               p->tokens,
                                               p->head_dim,
                                               p->aligned_head_dim,
                                               p->aligned_context_window);
    } else {
        attention_forward_causal_head_major_gqa_flash(p->q,
                                                     p->k,
                                                     p->v,
                                                     p->attn_out,
                                                     p->num_heads,
                                                     p->num_kv_heads,
                                                     p->tokens,
                                                     p->head_dim,
                                                     p->aligned_head_dim);
    }

    ck_attention_project_head_major_q4_k(p->attn_out,
                                         p->wo,
                                         p->bo,
                                         p->proj_tmp,
                                         p->proj_scratch,
                                         p->tokens,
                                         p->aligned_embed_dim,
                                         p->num_heads,
                                         p->aligned_head_dim);

    ck_residual_add_token_major(p->input,
                                p->proj_tmp,
                                p->residual1,
                                p->tokens,
                                p->aligned_embed_dim);

    rmsnorm_forward(p->residual1,
                    p->ln2_gamma,
                    p->ln2_out,
                    p->ln2_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_mlp_swiglu_forward_q4_k(p->ln2_out,
                               p->w1,
                               p->b1,
                               p->w2,
                               p->b2,
                               p->fc1_out,
                               p->swiglu_out,
                               p->mlp_out,
                               p->tokens,
                               p->aligned_embed_dim,
                               p->aligned_intermediate_dim);

    ck_residual_add_token_major(p->residual1,
                                p->mlp_out,
                                p->output,
                                p->tokens,
                                p->aligned_embed_dim);
}

void ck_layer_forward_rmsnorm_swiglu_decode_q4_k(const CKLayerForwardParamsQ4K *p,
                                                int token_index,
                                                int cache_capacity)
{
    if (!p) {
        return;
    }
    if (!p->input || !p->ln1_gamma || !p->ln2_gamma || !p->ln1_out || !p->ln2_out ||
        !p->wq || !p->wk || !p->wv || !p->wo || !p->w1 || !p->w2 ||
        !p->k || !p->v ||
        !p->proj_tmp || !p->residual1 || !p->fc1_out || !p->swiglu_out || !p->mlp_out || !p->output) {
        return;
    }
    if (token_index < 0 || cache_capacity <= 0 || token_index >= cache_capacity) {
        return;
    }

    const int D = p->embed_dim;
    const int aligned_D = p->aligned_embed_dim;
    const int H = p->num_heads;
    const int H_kv = p->num_kv_heads;
    const int hd = p->head_dim;
    const int ad = p->aligned_head_dim;
    const int aligned_intermediate = p->aligned_intermediate_dim;
    const int K_concat = H * ad;

    /* Decode buffers are single-token; token_index only applies to KV cache. */
    const size_t token_slot = 0;
    const float *input_row = p->input + token_slot * (size_t)aligned_D;
    float *ln1_row = p->ln1_out + token_slot * (size_t)aligned_D;
    float *ln2_row = p->ln2_out + token_slot * (size_t)aligned_D;
    float *proj_row = p->proj_tmp + token_slot * (size_t)aligned_D;
    float *residual_row = p->residual1 + token_slot * (size_t)aligned_D;
    float *mlp_row = p->mlp_out + token_slot * (size_t)aligned_D;
    float *out_row = p->output + token_slot * (size_t)aligned_D;

    float ln1_rstd_tmp = 0.0f;
    float ln2_rstd_tmp = 0.0f;
    float *ln1_rstd = p->ln1_rstd ? (p->ln1_rstd + token_slot) : &ln1_rstd_tmp;
    float *ln2_rstd = p->ln2_rstd ? (p->ln2_rstd + token_slot) : &ln2_rstd_tmp;

    /* Scratch for a single token in head-major layout: [head, aligned_head_dim]. */
    size_t q_elems = (size_t)H * (size_t)ad;
    size_t kv_elems = (size_t)H_kv * (size_t)ad;
    float q_token[q_elems];
    float k_token[kv_elems];
    float v_token[kv_elems];
    float attn_token[q_elems];

    /* LN1 / RMSNorm. */
    ck_debug_check_buffer("input_row", input_row, aligned_D);
    rmsnorm_forward(input_row,
                    p->ln1_gamma,
                    ln1_row,
                    ln1_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);
    ck_debug_check_buffer("ln1_out (after rmsnorm)", ln1_row, aligned_D);

    if (ck_q8k_activations_enabled()) {
        if ((aligned_D % QK_K) == 0 && (aligned_intermediate % QK_K) == 0) {
        const int q8_blocks_embed = aligned_D / QK_K;
        const int q8_blocks_inter = aligned_intermediate / QK_K;
        const int q8_blocks_max = (q8_blocks_embed > q8_blocks_inter) ? q8_blocks_embed : q8_blocks_inter;
        block_q8_K q8_buf[q8_blocks_max];

        /* Project Q/K/V with Q8_K activations. */
        quantize_row_q8_k(ln1_row, q8_buf, aligned_D);
        ck_debug_check_q8k("q8_buf (after quantize)", q8_buf, q8_blocks_embed);
        ck_debug_check_q4k_weights("wq weights", p->wq, (aligned_D / QK_K) * (H * ad));
        ck_qkv_project_head_major_token_q4_k_q8_k(q8_buf,
                                                  p->wq, p->bq,
                                                  p->wk, p->bk,
                                                  p->wv, p->bv,
                                                  q_token, k_token, v_token,
                                                  aligned_D,
                                                  H,
                                                  H_kv,
                                                  ad);
        ck_debug_check_buffer("q_token (after QKV proj)", q_token, (int)q_elems);
        ck_debug_check_buffer("k_token (after QKV proj)", k_token, (int)kv_elems);
        ck_debug_check_buffer("v_token (after QKV proj)", v_token, (int)kv_elems);

        /* RoPE for the new token at absolute position `p->rope_pos_offset`. */
        if (p->rope_cos && p->rope_sin) {
            rope_forward_qk(q_token,
                            k_token,
                            p->rope_cos,
                            p->rope_sin,
                            H,
                            H_kv,
                            /*num_tokens=*/1,
                            hd,
                            ad,
                            p->rope_pos_offset);
        }

        /* Update KV cache. */
        kv_cache_write_head_major(k_token,
                                  v_token,
                                  p->k,
                                  p->v,
                                  H_kv,
                                  token_index,
                                  cache_capacity,
                                  hd,
                                  ad);

        /* Decode attention for this token using the KV cache. */
        attention_forward_decode_head_major_gqa_regular(q_token,
                                                      p->k,
                                                      p->v,
                                                      attn_token,
                                                      H,
                                                      H_kv,
                                                      /*kv_tokens=*/token_index + 1,
                                                      cache_capacity,
                                                      hd,
                                                      ad);
        ck_debug_check_buffer("attn_token (after attention)", attn_token, (int)q_elems);

        /* Quantized output projection (Wo) with Q8_K activations. */
        quantize_row_q8_k(attn_token, q8_buf, aligned_D);
        gemm_nt_q4_k_q8_k(q8_buf,
                          p->wo,
                          p->bo,
                          proj_row,
                          /*M=*/1,
                          aligned_D,
                          /*K=*/K_concat);
        ck_debug_check_buffer("proj_row (after Wo proj)", proj_row, aligned_D);

        for (int j = D; j < aligned_D; ++j) {
            proj_row[j] = 0.0f;
        }

        /* Residual + LN2 / RMSNorm. */
        ck_residual_add_token_major(input_row,
                                    proj_row,
                                    residual_row,
                                    /*tokens=*/1,
                                    aligned_D);

        rmsnorm_forward(residual_row,
                        p->ln2_gamma,
                        ln2_row,
                        ln2_rstd,
                        /*tokens=*/1,
                        D,
                        aligned_D,
                        p->eps);

        /* MLP block for this token (Q8_K activations). */
        int up_dim = 2 * aligned_intermediate;
        float *fc1_row = p->fc1_out + token_slot * (size_t)up_dim;
        float *swiglu_row = p->swiglu_out + token_slot * (size_t)aligned_intermediate;

        ck_mlp_swiglu_forward_q4_k_q8_k(ln2_row,
                                        p->w1,
                                        p->b1,
                                        p->w2,
                                        p->b2,
                                        fc1_row,
                                        swiglu_row,
                                        mlp_row,
                                        aligned_D,
                                        aligned_intermediate);
        ck_debug_check_buffer("mlp_row (after MLP)", mlp_row, aligned_D);

        /* Final residual. */
        ck_residual_add_token_major(residual_row,
                                    mlp_row,
                                    out_row,
                                    /*tokens=*/1,
                                    aligned_D);
        ck_debug_check_buffer("out_row (final output)", out_row, aligned_D);
        return;
        }
    }

    /* Project Q/K/V for the new token (Q4_K weights). */
    ck_qkv_project_head_major_token_q4_k(ln1_row,
                                         p->wq, p->bq,
                                         p->wk, p->bk,
                                         p->wv, p->bv,
                                         q_token, k_token, v_token,
                                         aligned_D,
                                         H,
                                         H_kv,
                                         ad);

    /* RoPE for the new token at absolute position `p->rope_pos_offset`. */
    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(q_token,
                        k_token,
                        p->rope_cos,
                        p->rope_sin,
                        H,
                        H_kv,
                        /*num_tokens=*/1,
                        hd,
                        ad,
                        p->rope_pos_offset);
    }

    /* Update KV cache. */
    kv_cache_write_head_major(k_token,
                              v_token,
                              p->k,
                              p->v,
                              H_kv,
                              token_index,
                              cache_capacity,
                              hd,
                              ad);

    /* Decode attention for this token using the KV cache. */
    ck_attention_flash_decode_wrapper(q_token,
                                   p->k,
                                   p->v,
                                   attn_token,
                                   H,
                                   H_kv,
                                   /*kv_tokens=*/token_index + 1,
                                   cache_capacity,
                                   hd,
                                   ad);

    /* Quantized output projection: Wo is stored as a flattened Q4_K matrix. */
    gemm_nt_q4_k(attn_token,
                 p->wo,
                 p->bo,
                 proj_row,
                 /*M=*/1,
                 aligned_D,
                 /*K=*/K_concat);

    for (int j = D; j < aligned_D; ++j) {
        proj_row[j] = 0.0f;
    }

    /* Residual + LN2 / RMSNorm. */
    ck_residual_add_token_major(input_row,
                                proj_row,
                                residual_row,
                                /*tokens=*/1,
                                aligned_D);

    rmsnorm_forward(residual_row,
                    p->ln2_gamma,
                    ln2_row,
                    ln2_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    /* MLP block for this token. */
    int up_dim = 2 * aligned_intermediate;
    float *fc1_row = p->fc1_out + token_slot * (size_t)up_dim;
    float *swiglu_row = p->swiglu_out + token_slot * (size_t)aligned_intermediate;

    ck_mlp_swiglu_forward_q4_k(ln2_row,
                               p->w1,
                               p->b1,
                               p->w2,
                               p->b2,
                               fc1_row,
                               swiglu_row,
                               mlp_row,
                               /*tokens=*/1,
                               aligned_D,
                               aligned_intermediate);

    /* Final residual. */
    ck_residual_add_token_major(residual_row,
                                mlp_row,
                                out_row,
                                /*tokens=*/1,
                                aligned_D);
}

void ck_layer_forward_rmsnorm_swiglu_quant(const CKLayerForwardParamsQ4K *p)
{
    if (!p) {
        return;
    }

    rmsnorm_forward(p->input,
                    p->ln1_gamma,
                    p->ln1_out,
                    p->ln1_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_qkv_project_head_major_quant(p->ln1_out,
                                   p->wq, p->bq, p->wq_dtype,
                                   p->wk, p->bk, p->wk_dtype,
                                   p->wv, p->bv, p->wv_dtype,
                                   p->q, p->k, p->v,
                                   p->tokens,
                                   p->tokens,
                                   p->aligned_embed_dim,
                                   p->num_heads,
                                   p->num_kv_heads,
                                   p->aligned_head_dim);

    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(p->q,
                        p->k,
                        p->rope_cos,
                        p->rope_sin,
                        p->num_heads,
                        p->num_kv_heads,
                        p->tokens,
                        p->head_dim,
                        p->aligned_head_dim,
                        p->rope_pos_offset);
    }

    if (p->scores) {
        attention_forward_causal_head_major_gqa(p->q,
                                               p->k,
                                               p->v,
                                               p->scores,
                                               p->attn_out,
                                               p->num_heads,
                                               p->num_kv_heads,
                                               p->tokens,
                                               p->head_dim,
                                               p->aligned_head_dim,
                                               p->aligned_context_window);
    } else {
        attention_forward_causal_head_major_gqa_flash(p->q,
                                                     p->k,
                                                     p->v,
                                                     p->attn_out,
                                                     p->num_heads,
                                                     p->num_kv_heads,
                                                     p->tokens,
                                                     p->head_dim,
                                                     p->aligned_head_dim);
    }

    ck_attention_project_head_major_quant(p->attn_out,
                                          p->wo,
                                          p->bo,
                                          p->proj_tmp,
                                          p->proj_scratch,
                                          p->tokens,
                                          p->aligned_embed_dim,
                                          p->num_heads,
                                          p->aligned_head_dim,
                                          p->wo_dtype);

    ck_residual_add_token_major(p->input,
                                p->proj_tmp,
                                p->residual1,
                                p->tokens,
                                p->aligned_embed_dim);

    rmsnorm_forward(p->residual1,
                    p->ln2_gamma,
                    p->ln2_out,
                    p->ln2_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_mlp_swiglu_forward_quant(p->ln2_out,
                                p->w1,
                                p->b1,
                                p->w1_dtype,
                                p->w2,
                                p->b2,
                                p->w2_dtype,
                                p->fc1_out,
                                p->swiglu_out,
                                p->mlp_out,
                                p->tokens,
                                p->aligned_embed_dim,
                                p->aligned_intermediate_dim);

    ck_residual_add_token_major(p->residual1,
                                p->mlp_out,
                                p->output,
                                p->tokens,
                                p->aligned_embed_dim);
}

void ck_layer_forward_rmsnorm_swiglu_decode_quant(const CKLayerForwardParamsQ4K *p,
                                                 int token_index,
                                                 int cache_capacity)
{
    if (!p) {
        return;
    }
    if (!p->input || !p->ln1_gamma || !p->ln2_gamma || !p->ln1_out || !p->ln2_out ||
        !p->wq || !p->wk || !p->wv || !p->wo || !p->w1 || !p->w2 ||
        !p->k || !p->v ||
        !p->proj_tmp || !p->proj_scratch || !p->residual1 || !p->fc1_out || !p->swiglu_out || !p->mlp_out || !p->output) {
        return;
    }
    if (token_index < 0 || cache_capacity <= 0 || token_index >= cache_capacity) {
        return;
    }

    const int D = p->embed_dim;
    const int aligned_D = p->aligned_embed_dim;
    const int H = p->num_heads;
    const int H_kv = p->num_kv_heads;
    const int hd = p->head_dim;
    const int ad = p->aligned_head_dim;
    const int aligned_intermediate = p->aligned_intermediate_dim;
    const int K_concat = H * ad;

    /* Decode buffers are single-token; token_index only applies to KV cache. */
    const size_t token_slot = 0;
    const float *input_row = p->input + token_slot * (size_t)aligned_D;
    float *ln1_row = p->ln1_out + token_slot * (size_t)aligned_D;
    float *ln2_row = p->ln2_out + token_slot * (size_t)aligned_D;
    float *proj_row = p->proj_tmp + token_slot * (size_t)aligned_D;
    float *residual_row = p->residual1 + token_slot * (size_t)aligned_D;
    float *mlp_row = p->mlp_out + token_slot * (size_t)aligned_D;
    float *out_row = p->output + token_slot * (size_t)aligned_D;

    float ln1_rstd_tmp = 0.0f;
    float ln2_rstd_tmp = 0.0f;
    float *ln1_rstd = p->ln1_rstd ? (p->ln1_rstd + token_slot) : &ln1_rstd_tmp;
    float *ln2_rstd = p->ln2_rstd ? (p->ln2_rstd + token_slot) : &ln2_rstd_tmp;

    size_t q_elems = (size_t)H * (size_t)ad;
    size_t kv_elems = (size_t)H_kv * (size_t)ad;
    float q_token[q_elems];
    float k_token[kv_elems];
    float v_token[kv_elems];
    float attn_token[q_elems];

    rmsnorm_forward(input_row,
                    p->ln1_gamma,
                    ln1_row,
                    ln1_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    ck_qkv_project_head_major_token_quant(ln1_row,
                                          p->wq, p->bq, p->wq_dtype,
                                          p->wk, p->bk, p->wk_dtype,
                                          p->wv, p->bv, p->wv_dtype,
                                          q_token, k_token, v_token,
                                          aligned_D,
                                          H,
                                          H_kv,
                                          ad);

    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(q_token,
                        k_token,
                        p->rope_cos,
                        p->rope_sin,
                        H,
                        H_kv,
                        /*num_tokens=*/1,
                        hd,
                        ad,
                        p->rope_pos_offset);
    }

    kv_cache_write_head_major(k_token,
                              v_token,
                              p->k,
                              p->v,
                              H_kv,
                              token_index,
                              cache_capacity,
                              hd,
                              ad);

    ck_attention_flash_decode_wrapper(q_token,
                                   p->k,
                                   p->v,
                                   attn_token,
                                   H,
                                   H_kv,
                                   /*kv_tokens=*/token_index + 1,
                                   cache_capacity,
                                   hd,
                                   ad);

    if (p->wo_dtype == CK_DT_FP32) {
        ck_attention_project_head_major_decode_token(attn_token,
                                                     (const float *)p->wo,
                                                     p->bo,
                                                     proj_row,
                                                     D,
                                                     aligned_D,
                                                     H,
                                                     ad);
    } else {
        /* Quantized attention output projection - handle all quant types */
        ck_gemm_nt_quant(attn_token,
                         p->wo,
                         p->bo,
                         proj_row,
                         /*M=*/1,
                         aligned_D,
                         /*K=*/K_concat,
                         p->wo_dtype);
        for (int j = D; j < aligned_D; ++j) {
            proj_row[j] = 0.0f;
        }
    }

    ck_residual_add_token_major(input_row,
                                proj_row,
                                residual_row,
                                /*tokens=*/1,
                                aligned_D);

    rmsnorm_forward(residual_row,
                    p->ln2_gamma,
                    ln2_row,
                    ln2_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    int up_dim = 2 * aligned_intermediate;
    float *fc1_row = p->fc1_out + token_slot * (size_t)up_dim;
    float *swiglu_row = p->swiglu_out + token_slot * (size_t)aligned_intermediate;

    ck_mlp_swiglu_forward_quant(ln2_row,
                                p->w1,
                                p->b1,
                                p->w1_dtype,
                                p->w2,
                                p->b2,
                                p->w2_dtype,
                                fc1_row,
                                swiglu_row,
                                mlp_row,
                                /*tokens=*/1,
                                aligned_D,
                                aligned_intermediate);

    ck_residual_add_token_major(residual_row,
                                mlp_row,
                                out_row,
                                /*tokens=*/1,
                                aligned_D);
}

void ck_layer_backward_rmsnorm_swiglu(const CKLayerBackwardParams *p)
{
    if (!p) {
        return;
    }

    int T = p->tokens;
    int aligned_embed = p->aligned_embed_dim;
    int aligned_head = p->aligned_head_dim;
    int aligned_intermediate = p->aligned_intermediate_dim;
    int up_dim = 2 * aligned_intermediate;
    int num_threads = 1;

    // 1) Residual add (output = residual1 + mlp_out)
    ck_residual_add_backward(p->d_output, p->d_residual1, p->d_mlp_out, T, aligned_embed);

    // 2) MLP down proj backward
    fc2_backward_kernel(p->d_mlp_out,
                        p->swiglu_out,
                        p->w2,
                        p->d_swiglu_out,
                        p->d_w2,
                        p->d_b2,
                        T,
                        aligned_intermediate,
                        aligned_embed,
                        num_threads);

    // 3) SwiGLU backward
    swiglu_backward(p->fc1_out, p->d_swiglu_out, p->d_fc1_out, T, aligned_intermediate);

    // 4) MLP up proj backward
    fc1_backward_kernel(p->d_fc1_out,
                        p->ln2_out,
                        p->w1,
                        p->d_ln2_out,
                        p->d_w1,
                        p->d_b1,
                        T,
                        aligned_embed,
                        up_dim,
                        num_threads);

    // 5) RMSNorm (ln2) backward; reuse d_output as scratch for d_residual1_from_ln2
    rmsnorm_backward(p->d_ln2_out,
                     p->residual1,
                     p->ln2_gamma,
                     p->ln2_rstd,
                     p->d_output,
                     p->d_ln2_gamma,
                     T,
                     p->embed_dim,
                     aligned_embed);
    ck_add_inplace(p->d_residual1, p->d_output, T, aligned_embed);

    // 6) Residual add (residual1 = input + proj_tmp)
    ck_residual_add_backward(p->d_residual1, p->d_input, p->d_proj_tmp, T, aligned_embed);

    // 7) Attention projection backward
    ck_attention_project_head_major_backward(p->d_proj_tmp,
                                             p->attn_out,
                                             p->wo,
                                             p->d_attn_out,
                                             p->d_wo,
                                             p->d_bo,
                                             T,
                                             aligned_embed,
                                             p->num_heads,
                                             aligned_head);

    // 8) Attention backward
    attention_backward_causal_head_major_gqa(p->d_attn_out,
                                             p->q,
                                             p->k,
                                             p->v,
                                             p->scores,
                                             p->d_q,
                                             p->d_k,
                                             p->d_v,
                                             p->d_scores,
                                             p->num_heads,
                                             p->num_kv_heads,
                                             T,
                                             p->head_dim,
                                             aligned_head,
                                             p->aligned_context_window);

    // 9) RoPE backward (if enabled)
    if (p->rope_cos && p->rope_sin) {
        rope_backward_qk(p->d_q,
                         p->d_k,
                         p->d_q,
                         p->d_k,
                         p->rope_cos,
                         p->rope_sin,
                         p->num_heads,
                         p->num_kv_heads,
                         T,
                         p->head_dim,
                         aligned_head,
                         p->rope_pos_offset);
    }

    // 10) QKV projection backward (scratch uses d_proj_tmp)
    ck_qkv_project_head_major_backward(p->d_q,
                                       p->d_k,
                                       p->d_v,
                                       p->ln1_out,
                                       p->wq,
                                       p->bq,
                                       p->wk,
                                       p->bk,
                                       p->wv,
                                       p->bv,
                                       p->d_ln1_out,
                                       p->d_wq,
                                       p->d_bq,
                                       p->d_wk,
                                       p->d_bk,
                                       p->d_wv,
                                       p->d_bv,
                                       p->d_proj_tmp,
                                       T,
                                       aligned_embed,
                                       p->num_heads,
                                       p->num_kv_heads,
                                       aligned_head,
                                       num_threads);

    // 11) RMSNorm (ln1) backward; reuse d_ln1_out as scratch for d_input_from_ln1
    rmsnorm_backward(p->d_ln1_out,
                     p->input,
                     p->ln1_gamma,
                     p->ln1_rstd,
                     p->d_ln1_out,
                     p->d_ln1_gamma,
                     T,
                     p->embed_dim,
                     aligned_embed);
    ck_add_inplace(p->d_input, p->d_ln1_out, T, aligned_embed);
}
