#include "ckernel_orchestration.h"

#include "ckernel_engine.h"

#include <stddef.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX__) && !defined(__AVX512F__)
static inline float ck_hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

static inline float ck_dot_f32(const float *a, const float *b, int len)
{
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i <= len - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#elif defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= len - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float sum = ck_hsum256_ps(acc);
    for (; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

void ck_qkv_project_head_major_token(const float *input_row,
                                    const float *wq, const float *bq,
                                    const float *wk, const float *bk,
                                    const float *wv, const float *bv,
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

    const int q_out = num_heads * aligned_head_dim;
    gemm_blocked_serial(input_row, wq, bq, q_token,
                        /*tokens=*/1, q_out, aligned_embed_dim);

    size_t head_weight_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
#pragma omp parallel for schedule(static) if(num_kv_heads > 1)
    for (int h = 0; h < num_kv_heads; ++h) {
        const float *wk_h = wk + (size_t)h * head_weight_stride;
        const float *wv_h = wv + (size_t)h * head_weight_stride;
        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k_token + (size_t)h * (size_t)aligned_head_dim;
        float *v_h = v_token + (size_t)h * (size_t)aligned_head_dim;

        gemm_blocked_serial(input_row, wk_h, bk_h, k_h,
                            /*tokens=*/1, aligned_head_dim, aligned_embed_dim);
        gemm_blocked_serial(input_row, wv_h, bv_h, v_h,
                            /*tokens=*/1, aligned_head_dim, aligned_embed_dim);
    }
}

void ck_attention_project_head_major_decode_token(const float *attn_token,
                                                 const float *wo,
                                                 const float *bo,
                                                 float *out_token,
                                                 int embed_dim,
                                                 int aligned_embed_dim,
                                                 int num_heads,
                                                 int aligned_head_dim)
{
    const size_t head_in_stride = (size_t)aligned_head_dim;
    const size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

#pragma omp parallel for schedule(static)
    for (int j = 0; j < embed_dim; ++j) {
        float sum = bo ? bo[j] : 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            const float *head_in = attn_token + (size_t)h * head_in_stride;
            const float *wo_row = wo + (size_t)h * head_weight_stride + (size_t)j * (size_t)aligned_head_dim;
            sum += ck_dot_f32(head_in, wo_row, aligned_head_dim);
        }
        out_token[j] = sum;
    }

    for (int j = embed_dim; j < aligned_embed_dim; ++j) {
        out_token[j] = 0.0f;
    }
}

// Per-output dot products keep the attn token hot, but can be slower than a matvec
// microkernel on some CPUs.
static void ck_attention_project_head_major_decode_token_residual(const float *attn_token,
                                                                  const float *wo,
                                                                  const float *bo,
                                                                  const float *residual_in,
                                                                  float *proj_out,
                                                                  float *residual_out,
                                                                  int embed_dim,
                                                                  int aligned_embed_dim,
                                                                  int num_heads,
                                                                  int aligned_head_dim)
{
    const size_t head_in_stride = (size_t)aligned_head_dim;
    const size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

#pragma omp parallel for schedule(static)
    for (int j = 0; j < embed_dim; ++j) {
        float sum = bo ? bo[j] : 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            const float *head_in = attn_token + (size_t)h * head_in_stride;
            const float *wo_row = wo + (size_t)h * head_weight_stride + (size_t)j * (size_t)aligned_head_dim;
            sum += ck_dot_f32(head_in, wo_row, aligned_head_dim);
        }
        if (proj_out) {
            proj_out[j] = sum;
        }
        residual_out[j] = sum + residual_in[j];
    }

    for (int j = embed_dim; j < aligned_embed_dim; ++j) {
        if (proj_out) {
            proj_out[j] = 0.0f;
        }
        residual_out[j] = 0.0f;
    }
}

static void ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_impl(const CKLayerForwardParams *p,
                                                                   int token_index,
                                                                   int cache_capacity,
                                                                   int fuse_mlp)
{
    if (!p) {
        return;
    }
    if (!p->input || !p->ln1_gamma || !p->ln2_gamma || !p->wq || !p->wk || !p->wv || !p->wo ||
        !p->w1 || !p->w2 || !p->k || !p->v || !p->residual1 || !p->ln2_out || !p->swiglu_out ||
        !p->mlp_out || !p->output) {
        return;
    }
    if (!fuse_mlp && !p->fc1_out) {
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
    float *proj_row = NULL;
    float *residual_row = p->residual1 + token_slot * (size_t)aligned_D;
    float *ln2_row = p->ln2_out + token_slot * (size_t)aligned_D;
    float *swiglu_row = p->swiglu_out + token_slot * (size_t)aligned_intermediate;
    float *mlp_row = p->mlp_out + token_slot * (size_t)aligned_D;
    float *out_row = p->output + token_slot * (size_t)aligned_D;

    float ln1_rstd_tmp = 0.0f;
    float ln2_rstd_tmp = 0.0f;
    float *ln2_rstd = p->ln2_rstd ? (p->ln2_rstd + token_slot) : &ln2_rstd_tmp;

    float ln1_row[aligned_D];

    rmsnorm_forward(input_row,
                    p->ln1_gamma,
                    ln1_row,
                    &ln1_rstd_tmp,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    size_t q_elems = (size_t)H * (size_t)ad;
    size_t kv_elems = (size_t)H_kv * (size_t)ad;
    float q_token[q_elems];
    float k_token[kv_elems];
    float v_token[kv_elems];
    float attn_token[q_elems];

    ck_qkv_project_head_major_token(ln1_row,
                                    p->wq, p->bq,
                                    p->wk, p->bk,
                                    p->wv, p->bv,
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

    ck_attention_project_head_major_decode_token_residual(attn_token,
                                                          p->wo,
                                                          p->bo,
                                                          input_row,
                                                          proj_row,
                                                          residual_row,
                                                          D,
                                                          aligned_D,
                                                          H,
                                                          ad);

    rmsnorm_forward(residual_row,
                    p->ln2_gamma,
                    ln2_row,
                    ln2_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    if (fuse_mlp) {
        // Fully fused MLP avoids writing SwiGLU activations to DRAM (aligned dims
        // match padded weight layout).
        ck_mlp_swiglu_forward_fully_fused_token(ln2_row,
                                                 p->w1,
                                                 p->b1,
                                                 p->w2,
                                                 p->b2,
                                                 mlp_row,
                                                 aligned_D,
                                                 aligned_intermediate);
    } else {
        int up_dim = 2 * aligned_intermediate;
        float *fc1_row = p->fc1_out + token_slot * (size_t)up_dim;

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
    }

    ck_residual_add_token_major(residual_row,
                                mlp_row,
                                out_row,
                                /*tokens=*/1,
                                aligned_D);
}

void ck_layer_forward_rmsnorm_swiglu_decode_fused_attn(const CKLayerForwardParams *p,
                                                      int token_index,
                                                      int cache_capacity)
{
    ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_impl(p,
                                                           token_index,
                                                           cache_capacity,
                                                           /*fuse_mlp=*/0);
}

void ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_mlp(const CKLayerForwardParams *p,
                                                          int token_index,
                                                          int cache_capacity)
{
    ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_impl(p,
                                                           token_index,
                                                           cache_capacity,
                                                           /*fuse_mlp=*/1);
}
