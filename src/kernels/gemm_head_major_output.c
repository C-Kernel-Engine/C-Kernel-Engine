/**
 * @file gemm_head_major_output.c
 * @brief Output projection from head-major attention (NO LAYOUT CONVERSION)
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. NO memcpy for layout - use strided access, not copies
 * 4. API must define: inputs, outputs, workspace, and memory layouts
 * 5. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * PROBLEM THIS SOLVES:
 * ====================
 * The standard mega_fused_attention_prefill has a bottleneck:
 *   attn_out [num_heads, tokens, head_dim] (head-major)
 *       → flatten_head_major() - 448 memcpy calls for 32 tokens × 14 heads!
 *       → token-major buffer
 *       → GEMM output projection
 *
 * This kernel eliminates the flatten by reading head-major data directly with
 * strided access. The output projection computes:
 *
 *   output[t, n] = bias[n] + sum_h wo[n, h*head_dim:(h+1)*head_dim] @ attn_out[h, t, :]
 *
 * where wo is Q5_0 quantized [embed_dim, embed_dim] and attn_out is head-major.
 *
 * Expected speedup: 1.5-2x by eliminating 448 small memcpy calls.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "ckernel_quant.h"
#include "ckernel_dtype.h"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Forward declaration from dequant_kernels.c */
void dequant_q5_0_block(const block_q5_0 *block, float *output);
void dequant_q5_0_row(const void *src, float *dst, size_t n_elements);

/* ============================================================================
 * Scalar reference: Output projection from head-major attention
 * ============================================================================ */

/**
 * @brief Output projection reading head-major attention output (Q5_0 weights)
 *
 * @param output Output [tokens, embed_dim] (token-major, written contiguously)
 * @param attn_out Attention output [num_heads, tokens, head_dim] (head-major, strided)
 * @param wo Output weights in Q5_0 format [embed_dim, embed_dim]
 * @param bias Optional bias [embed_dim]
 * @param tokens Number of tokens
 * @param embed_dim Output embedding dimension
 * @param num_heads Number of attention heads
 * @param head_dim Head dimension (must be multiple of 32 for Q5_0)
 */
void gemv_nt_q5_0_head_major_output(float *output,
                                     const float *attn_out,
                                     const void *wo,
                                     const float *bias,
                                     int tokens,
                                     int embed_dim,
                                     int num_heads,
                                     int head_dim)
{
    if (!output || !attn_out || !wo) return;
    if (tokens <= 0 || embed_dim <= 0 || num_heads <= 0 || head_dim <= 0) return;

    const int blocks_per_head = head_dim / QK5_0;
    const int blocks_per_row = embed_dim / QK5_0;
    const block_q5_0 *weights = (const block_q5_0 *)wo;

    /* Strides for head-major layout */
    const size_t token_stride = head_dim;           /* attn_out[h][t] offset */
    const size_t head_stride = (size_t)tokens * token_stride;  /* attn_out[h] offset */

    /* Initialize output with bias (if provided) */
    if (bias) {
        for (int t = 0; t < tokens; t++) {
            float *out_row = output + (size_t)t * embed_dim;
            for (int n = 0; n < embed_dim; n++) {
                out_row[n] = bias[n];
            }
        }
    } else {
        memset(output, 0, (size_t)tokens * embed_dim * sizeof(float));
    }

    /* Accumulate contributions from each head */
    for (int h = 0; h < num_heads; h++) {
        const float *head_data = attn_out + (size_t)h * head_stride;

        /* For each output row (n) corresponding to this head's slice */
        const int head_offset = h * blocks_per_head;

        for (int n_block = 0; n_block < blocks_per_head; n_block++) {
            for (int n = 0; n < embed_dim; n++) {
                const block_q5_0 *w_row = weights + (size_t)n * blocks_per_row + head_offset + n_block;
                const float d = CK_FP16_TO_FP32(w_row->d);

                /* Get high bits */
                uint32_t qh;
                memcpy(&qh, w_row->qh, sizeof(qh));

                /* Accumulate for all tokens at once (better cache reuse) */
                for (int t = 0; t < tokens; t++) {
                    const float *token_vec = head_data + (size_t)t * token_stride + (size_t)n_block * QK5_0;
                    float sum = 0.0f;

                    /* Q5_0 dot product for this block */
                    for (int j = 0; j < QK5_0 / 2; j++) {
                        const uint8_t packed = w_row->qs[j];
                        const int lo = (packed & 0x0F);
                        const int hi = (packed >> 4);
                        const int xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                        const int xh_1 = ((qh >> (j + 12))) & 0x10;
                        const int q0 = (lo | xh_0) - 16;
                        const int q1 = (hi | xh_1) - 16;

                        sum += d * (float)q0 * token_vec[j];
                        sum += d * (float)q1 * token_vec[j + 16];
                    }

                    output[(size_t)t * embed_dim + n] += sum;
                }
            }
        }
    }
}

/* ============================================================================
 * Vectorized version with AVX (dot product over decoded 32-wide blocks)
 * ============================================================================ */

#if defined(__AVX__)

static inline float hsum256_ps_head_major(__m256 v)
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

static inline float dot_fp32_q5_0_block_avx(const block_q5_0 *block,
                                            const float *x)
{
    float w[QK5_0];
    dequant_q5_0_block(block, w);

    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < QK5_0; i += 8) {
        const __m256 wv = _mm256_loadu_ps(w + i);
        const __m256 xv = _mm256_loadu_ps(x + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
    }
    return hsum256_ps_head_major(acc);
}

static inline void accum_8rows_q5_0_block_avx(float *out,
                                              const block_q5_0 *w0,
                                              const block_q5_0 *w1,
                                              const block_q5_0 *w2,
                                              const block_q5_0 *w3,
                                              const block_q5_0 *w4,
                                              const block_q5_0 *w5,
                                              const block_q5_0 *w6,
                                              const block_q5_0 *w7,
                                              const float *x)
{
    float w_dec[8][QK5_0];
    dequant_q5_0_block(w0, w_dec[0]);
    dequant_q5_0_block(w1, w_dec[1]);
    dequant_q5_0_block(w2, w_dec[2]);
    dequant_q5_0_block(w3, w_dec[3]);
    dequant_q5_0_block(w4, w_dec[4]);
    dequant_q5_0_block(w5, w_dec[5]);
    dequant_q5_0_block(w6, w_dec[6]);
    dequant_q5_0_block(w7, w_dec[7]);

    __m256 acc = _mm256_loadu_ps(out);
    for (int i = 0; i < QK5_0; i++) {
        const __m256 wv = _mm256_setr_ps(
            w_dec[0][i], w_dec[1][i], w_dec[2][i], w_dec[3][i],
            w_dec[4][i], w_dec[5][i], w_dec[6][i], w_dec[7][i]);
        const __m256 xv = _mm256_set1_ps(x[i]);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
    }
    _mm256_storeu_ps(out, acc);
}

void gemv_nt_q5_0_head_major_output_avx(float *output,
                                        const float *attn_out,
                                        const void *wo,
                                        const float *bias,
                                        int tokens,
                                        int embed_dim,
                                        int num_heads,
                                        int head_dim)
{
    if (!output || !attn_out || !wo) return;
    if (tokens <= 0 || embed_dim <= 0 || num_heads <= 0 || head_dim <= 0) return;

    const int blocks_per_head = head_dim / QK5_0;
    const int blocks_per_row = embed_dim / QK5_0;
    const block_q5_0 *weights = (const block_q5_0 *)wo;

    const size_t token_stride = head_dim;
    const size_t head_stride = (size_t)tokens * token_stride;

    if (bias) {
        for (int t = 0; t < tokens; t++) {
            float *out_row = output + (size_t)t * embed_dim;
            for (int n = 0; n < embed_dim; n++) {
                out_row[n] = bias[n];
            }
        }
    } else {
        memset(output, 0, (size_t)tokens * embed_dim * sizeof(float));
    }

    for (int h = 0; h < num_heads; h++) {
        const float *head_data = attn_out + (size_t)h * head_stride;
        const int head_offset = h * blocks_per_head;

        int n = 0;
        for (; n + 7 < embed_dim; n += 8) {
            for (int n_block = 0; n_block < blocks_per_head; n_block++) {
                const block_q5_0 *w0 = weights + (size_t)(n + 0) * blocks_per_row + head_offset + n_block;
                const block_q5_0 *w1 = weights + (size_t)(n + 1) * blocks_per_row + head_offset + n_block;
                const block_q5_0 *w2 = weights + (size_t)(n + 2) * blocks_per_row + head_offset + n_block;
                const block_q5_0 *w3 = weights + (size_t)(n + 3) * blocks_per_row + head_offset + n_block;
                const block_q5_0 *w4 = weights + (size_t)(n + 4) * blocks_per_row + head_offset + n_block;
                const block_q5_0 *w5 = weights + (size_t)(n + 5) * blocks_per_row + head_offset + n_block;
                const block_q5_0 *w6 = weights + (size_t)(n + 6) * blocks_per_row + head_offset + n_block;
                const block_q5_0 *w7 = weights + (size_t)(n + 7) * blocks_per_row + head_offset + n_block;

                for (int t = 0; t < tokens; t++) {
                    const float *token_vec =
                        head_data + (size_t)t * token_stride + (size_t)n_block * QK5_0;
                    float *out_row = output + (size_t)t * embed_dim + n;
                    accum_8rows_q5_0_block_avx(out_row, w0, w1, w2, w3, w4, w5, w6, w7, token_vec);
                }
            }
        }

        for (; n < embed_dim; n++) {
            const block_q5_0 *w_row = weights + (size_t)n * blocks_per_row + head_offset;
            for (int n_block = 0; n_block < blocks_per_head; n_block++) {
                const block_q5_0 *w_block = w_row + n_block;
                for (int t = 0; t < tokens; t++) {
                    const float *token_vec =
                        head_data + (size_t)t * token_stride + (size_t)n_block * QK5_0;
                    output[(size_t)t * embed_dim + n] +=
                        dot_fp32_q5_0_block_avx(w_block, token_vec);
                }
            }
        }
    }
}

#endif /* __AVX__ */

/* ============================================================================
 * Generic dispatch
 * ============================================================================ */

/**
 * @brief Output projection from head-major attention (auto-dispatch)
 *
 * This replaces flatten_head_major() + ck_gemm_nt_quant() with a single
 * strided-access kernel that reads head-major attention output directly.
 */
void ck_gemm_nt_head_major_q5_0(const float *attn_out,  /* [num_heads, tokens, head_dim] */
                                 const void *wo,
                                 const float *bias,
                                 float *output,         /* [tokens, embed_dim] */
                                 int tokens,
                                 int embed_dim,
                                 int num_heads,
                                 int head_dim)
{
#if defined(__AVX__)
    gemv_nt_q5_0_head_major_output_avx(output, attn_out, wo, bias,
                                       tokens, embed_dim, num_heads, head_dim);
#else
    gemv_nt_q5_0_head_major_output(output, attn_out, wo, bias,
                                   tokens, embed_dim, num_heads, head_dim);
#endif
}

/* ============================================================================
 * Q8_0 variant (for V weights which are often Q8_0)
 * ============================================================================ */

/**
 * @brief Output projection from head-major attention (Q8_0 weights)
 */
void ck_gemm_nt_head_major_q8_0(const float *attn_out,
                                 const void *wo,
                                 const float *bias,
                                 float *output,
                                 int tokens,
                                 int embed_dim,
                                 int num_heads,
                                 int head_dim)
{
    if (!output || !attn_out || !wo) return;
    if (tokens <= 0 || embed_dim <= 0 || num_heads <= 0 || head_dim <= 0) return;

    const int blocks_per_head = head_dim / QK8_0;
    const int blocks_per_row = embed_dim / QK8_0;
    const block_q8_0 *weights = (const block_q8_0 *)wo;

    const size_t token_stride = head_dim;
    const size_t head_stride = (size_t)tokens * token_stride;

    /* Initialize output */
    if (bias) {
        for (int t = 0; t < tokens; t++) {
            float *out_row = output + (size_t)t * embed_dim;
            for (int n = 0; n < embed_dim; n++) {
                out_row[n] = bias[n];
            }
        }
    } else {
        memset(output, 0, (size_t)tokens * embed_dim * sizeof(float));
    }

    /* Accumulate from each head */
    for (int h = 0; h < num_heads; h++) {
        const float *head_data = attn_out + (size_t)h * head_stride;
        const int head_offset = h * blocks_per_head;

        for (int n_block = 0; n_block < blocks_per_head; n_block++) {
            for (int n = 0; n < embed_dim; n++) {
                const block_q8_0 *w_row = weights + (size_t)n * blocks_per_row + head_offset + n_block;
                const float d = CK_FP16_TO_FP32(w_row->d);

                for (int t = 0; t < tokens; t++) {
                    const float *token_vec = head_data + (size_t)t * token_stride + (size_t)n_block * QK8_0;
                    float sum = 0.0f;

                    for (int j = 0; j < QK8_0; j++) {
                        sum += d * (float)w_row->qs[j] * token_vec[j];
                    }

                    output[(size_t)t * embed_dim + n] += sum;
                }
            }
        }
    }
}
