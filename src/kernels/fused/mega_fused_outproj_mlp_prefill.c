/**
 * @file mega_fused_outproj_mlp_prefill.c
 * @brief Mega-fused post-attention block for prefill
 *
 * OutProj → Residual → RMSNorm2 → MLP → Residual
 *
 * Plan summary:
 *   1) Quantize head-major attn_out to Q8_0
 *   2) Out-proj with Q5_0/Q8_0 weights → h1 (post-attn) in scratch
 *   3) Add residual (input) into h1
 *   4) RMSNorm2(h1) → ln2_out (scratch)
 *   5) Fused MLP (quant W1/W2) → output
 *   6) Add h1 residual into output
 *
 * Goal: avoid DRAM writes between attention out-proj and MLP output.
 * All intermediates live in scratch buffers from the bump allocator.
 */

#include "ckernel_engine.h"
#include "ckernel_quant.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__AVX__)
#include <immintrin.h>
#endif

static size_t align_up_size(size_t value, size_t align)
{
    return (value + align - 1) & ~(align - 1);
}

/* Note: add_inplace_f32 is declared in ckernel_engine.h and defined elsewhere */

static void quantize_attn_out_head_major_q8_0(const float *attn_out,
                                              uint8_t *dst,
                                              int tokens,
                                              int num_heads,
                                              int aligned_head_dim)
{
    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0,
                                                   (size_t)aligned_head_dim);
    const size_t head_stride = (size_t)tokens * (size_t)aligned_head_dim;
    for (int h = 0; h < num_heads; ++h) {
        const float *head = attn_out + (size_t)h * head_stride;
        for (int t = 0; t < tokens; ++t) {
            const float *row = head + (size_t)t * (size_t)aligned_head_dim;
            uint8_t *out = dst + ((size_t)h * (size_t)tokens + (size_t)t) *
                                  q8_row_bytes;
            quantize_row_q8_0(row, out, aligned_head_dim);
        }
    }
}

static void out_proj_head_major_q5_0_q8_0(const uint8_t *attn_q8,
                                          const void *wo,
                                          const float *bias,
                                          float *output,
                                          int tokens,
                                          int aligned_embed_dim,
                                          int num_heads,
                                          int aligned_head_dim)
{
#define OUTPROJ_TILE_N 8
    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0,
                                                   (size_t)aligned_head_dim);
    const int blocks_per_head = aligned_head_dim / QK5_0;
    const int blocks_per_row = aligned_embed_dim / QK5_0;
    const block_q5_0 *weights = (const block_q5_0 *)wo;

    for (int t = 0; t < tokens; ++t) {
        float *out_row = output + (size_t)t * (size_t)aligned_embed_dim;
        for (int n = 0; n < aligned_embed_dim; n += OUTPROJ_TILE_N) {
            const int tile = (n + OUTPROJ_TILE_N <= aligned_embed_dim)
                                 ? OUTPROJ_TILE_N
                                 : (aligned_embed_dim - n);
            float sum[OUTPROJ_TILE_N];
            for (int i = 0; i < tile; ++i) {
                sum[i] = bias ? bias[n + i] : 0.0f;
            }

            for (int h = 0; h < num_heads; ++h) {
                const uint8_t *a_row = attn_q8 +
                                       ((size_t)h * (size_t)tokens + (size_t)t) *
                                       q8_row_bytes;
                const block_q5_0 *w_row_base = weights +
                                               (size_t)n * (size_t)blocks_per_row +
                                               (size_t)h * (size_t)blocks_per_head;
                for (int i = 0; i < tile; ++i) {
                    const block_q5_0 *w_head = w_row_base +
                                               (size_t)i * (size_t)blocks_per_row;
                    float partial = 0.0f;
                    vec_dot_q5_0_q8_0(aligned_head_dim, &partial, w_head, a_row);
                    sum[i] += partial;
                }
            }

            for (int i = 0; i < tile; ++i) {
                out_row[n + i] = sum[i];
            }
        }
    }
#undef OUTPROJ_TILE_N
}

static void out_proj_head_major_q8_0_q8_0(const uint8_t *attn_q8,
                                          const void *wo,
                                          const float *bias,
                                          float *output,
                                          int tokens,
                                          int aligned_embed_dim,
                                          int num_heads,
                                          int aligned_head_dim)
{
#define OUTPROJ_TILE_N 8
    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0,
                                                   (size_t)aligned_head_dim);
    const int blocks_per_head = aligned_head_dim / QK8_0;
    const int blocks_per_row = aligned_embed_dim / QK8_0;
    const block_q8_0 *weights = (const block_q8_0 *)wo;

    for (int t = 0; t < tokens; ++t) {
        float *out_row = output + (size_t)t * (size_t)aligned_embed_dim;
        for (int n = 0; n < aligned_embed_dim; n += OUTPROJ_TILE_N) {
            const int tile = (n + OUTPROJ_TILE_N <= aligned_embed_dim)
                                 ? OUTPROJ_TILE_N
                                 : (aligned_embed_dim - n);
            float sum[OUTPROJ_TILE_N];
            for (int i = 0; i < tile; ++i) {
                sum[i] = bias ? bias[n + i] : 0.0f;
            }

            for (int h = 0; h < num_heads; ++h) {
                const uint8_t *a_row = attn_q8 +
                                       ((size_t)h * (size_t)tokens + (size_t)t) *
                                       q8_row_bytes;
                const block_q8_0 *w_row_base = weights +
                                               (size_t)n * (size_t)blocks_per_row +
                                               (size_t)h * (size_t)blocks_per_head;
                for (int i = 0; i < tile; ++i) {
                    const block_q8_0 *w_head = w_row_base +
                                               (size_t)i * (size_t)blocks_per_row;
                    float partial = 0.0f;
                    vec_dot_q8_0_q8_0(aligned_head_dim, &partial, w_head, a_row);
                    sum[i] += partial;
                }
            }

            for (int i = 0; i < tile; ++i) {
                out_row[n + i] = sum[i];
            }
        }
    }
#undef OUTPROJ_TILE_N
}

size_t mega_fused_outproj_mlp_prefill_scratch_size(int tokens,
                                                   int aligned_embed_dim,
                                                   int num_heads,
                                                   int aligned_head_dim,
                                                   int aligned_intermediate_dim)
{
    if (tokens <= 0 || aligned_embed_dim <= 0 || num_heads <= 0 ||
        aligned_head_dim <= 0 || aligned_intermediate_dim <= 0) {
        return 0;
    }

    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0,
                                                   (size_t)aligned_head_dim);
    const size_t attn_q8_bytes = (size_t)num_heads * (size_t)tokens * q8_row_bytes;
    const size_t h1_bytes = (size_t)tokens * (size_t)aligned_embed_dim * sizeof(float);
    const size_t ln2_bytes = h1_bytes;
    const size_t mlp_scratch = fused_mlp_swiglu_prefill_w1w2_quant_scratch_size(
        aligned_embed_dim, aligned_intermediate_dim);

    return align_up_size(attn_q8_bytes, 64) +
           align_up_size(h1_bytes, 64) +
           align_up_size(ln2_bytes, 64) +
           align_up_size(mlp_scratch, 64);
}

void mega_fused_outproj_mlp_prefill(
    float *output,
    const float *attn_out,
    const float *residual,
    const float *ln2_gamma,
    const void *wo, const float *bo, CKDataType wo_dt,
    const void *w1, const float *b1, CKDataType w1_dt,
    const void *w2, const float *b2, CKDataType w2_dt,
    int tokens,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int aligned_head_dim,
    int intermediate_dim,
    int aligned_intermediate_dim,
    float eps,
    void *scratch)
{
    if (!output || !attn_out || !residual || !ln2_gamma ||
        !wo || !w1 || !w2 || !scratch) {
        return;
    }
    if (tokens <= 0 || embed_dim <= 0 || aligned_embed_dim <= 0 ||
        num_heads <= 0 || aligned_head_dim <= 0 ||
        intermediate_dim <= 0 || aligned_intermediate_dim <= 0) {
        return;
    }
    if (aligned_embed_dim < embed_dim || aligned_head_dim <= 0 ||
        aligned_intermediate_dim < intermediate_dim) {
        return;
    }
    if (aligned_embed_dim != num_heads * aligned_head_dim) {
        return;
    }
    if ((aligned_embed_dim % 32) != 0 || (aligned_head_dim % 32) != 0) {
        return;
    }
    if ((aligned_intermediate_dim % QK_K) != 0) {
        return;
    }
    if (wo_dt != CK_DT_Q5_0 && wo_dt != CK_DT_Q8_0) {
        return;
    }
    if (w1_dt != CK_DT_Q5_0 && w1_dt != CK_DT_Q8_0) {
        return;
    }
    if (w2_dt != CK_DT_Q4_K && w2_dt != CK_DT_Q6_K) {
        return;
    }

    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0,
                                                   (size_t)aligned_head_dim);
    const size_t attn_q8_bytes = (size_t)num_heads * (size_t)tokens * q8_row_bytes;
    const size_t h1_bytes = (size_t)tokens * (size_t)aligned_embed_dim * sizeof(float);
    const size_t ln2_bytes = h1_bytes;

    uint8_t *scratch_bytes = (uint8_t *)scratch;
    uint8_t *attn_q8 = scratch_bytes;
    scratch_bytes += align_up_size(attn_q8_bytes, 64);
    float *h1 = (float *)scratch_bytes;
    scratch_bytes += align_up_size(h1_bytes, 64);
    float *ln2_out = (float *)scratch_bytes;
    scratch_bytes += align_up_size(ln2_bytes, 64);
    void *mlp_scratch = (void *)scratch_bytes;

    quantize_attn_out_head_major_q8_0(attn_out,
                                      attn_q8,
                                      tokens,
                                      num_heads,
                                      aligned_head_dim);

    if (wo_dt == CK_DT_Q8_0) {
        out_proj_head_major_q8_0_q8_0(attn_q8,
                                      wo,
                                      bo,
                                      h1,
                                      tokens,
                                      aligned_embed_dim,
                                      num_heads,
                                      aligned_head_dim);
    } else {
        out_proj_head_major_q5_0_q8_0(attn_q8,
                                      wo,
                                      bo,
                                      h1,
                                      tokens,
                                      aligned_embed_dim,
                                      num_heads,
                                      aligned_head_dim);
    }

    for (int t = 0; t < tokens; ++t) {
        const float *res_row = residual + (size_t)t * (size_t)aligned_embed_dim;
        float *h1_row = h1 + (size_t)t * (size_t)aligned_embed_dim;
        add_inplace_f32(h1_row, res_row, aligned_embed_dim);
    }

    rmsnorm_forward(h1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    tokens,
                    embed_dim,
                    aligned_embed_dim,
                    eps);

    fused_mlp_swiglu_prefill_w1w2_quant(ln2_out,
                                        w1,
                                        b1,
                                        w1_dt,
                                        w2,
                                        b2,
                                        w2_dt,
                                        output,
                                        tokens,
                                        embed_dim,
                                        aligned_embed_dim,
                                        intermediate_dim,
                                        aligned_intermediate_dim,
                                        mlp_scratch);

    for (int t = 0; t < tokens; ++t) {
        const float *h1_row = h1 + (size_t)t * (size_t)aligned_embed_dim;
        float *out_row = output + (size_t)t * (size_t)aligned_embed_dim;
        add_inplace_f32(out_row, h1_row, aligned_embed_dim);
    }
}
