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
 * Vectorized version with AVX (8 floats at a time)
 * ============================================================================ */

#if defined(__AVX__) && defined(__F16C__)
#include <immintrin.h>

/**
 * @brief Optimized version with AVX SIMD
 *
 * Key optimizations:
 * 1. Process 8 output rows at a time using AVX
 * 2. Accumulate across heads for better cache utilization
 * 3. Use FMAC for multiply-accumulate
 */
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

    /* Process heads sequentially, accumulating into output */
    for (int h = 0; h < num_heads; h++) {
        const float *head_data = attn_out + (size_t)h * head_stride;
        const int head_offset = h * blocks_per_head;

        /* Process output rows in chunks of 8 for AVX */
        int n = 0;
        for (; n + 7 < embed_dim; n += 8) {
            /* Process 8 output rows at once */
            for (int n_block = 0; n_block < blocks_per_head; n_block++) {
                const size_t w_offset = (size_t)(n + head_offset + n_block) * blocks_per_row + n_block;

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                __m256 acc4 = _mm256_setzero_ps();
                __m256 acc5 = _mm256_setzero_ps();
                __m256 acc6 = _mm256_setzero_ps();
                __m256 acc7 = _mm256_setzero_ps();

                /* For each token */
                for (int t = 0; t < tokens; t++) {
                    const float *token_vec = head_data + (size_t)t * token_stride + (size_t)n_block * QK5_0;

                    /* Load 8 weight blocks */
                    const block_q5_0 *w0 = weights + w_offset;
                    const block_q5_0 *w1 = w0 + blocks_per_row;
                    const block_q5_0 *w2 = w1 + blocks_per_row;
                    const block_q5_0 *w3 = w2 + blocks_per_row;
                    const block_q5_0 *w4 = w3 + blocks_per_row;
                    const block_q5_0 *w5 = w4 + blocks_per_row;
                    const block_q5_0 *w6 = w5 + blocks_per_row;
                    const block_q5_0 *w7 = w6 + blocks_per_row;

                    const float d0 = CK_FP16_TO_FP32(w0->d);
                    const float d1 = CK_FP16_TO_FP32(w1->d);
                    const float d2 = CK_FP16_TO_FP32(w2->d);
                    const float d3 = CK_FP16_TO_FP32(w3->d);
                    const float d4 = CK_FP16_TO_FP32(w4->d);
                    const float d5 = CK_FP16_TO_FP32(w5->d);
                    const float d6 = CK_FP16_TO_FP32(w6->d);
                    const float d7 = CK_FP16_TO_FP32(w7->d);

                    /* Dot products for each output row */
                    for (int j = 0; j < 16; j++) {
                        const uint8_t p0 = w0->qs[j];
                        const uint8_t p1 = w1->qs[j];
                        const uint8_t p2 = w2->qs[j];
                        const uint8_t p3 = w3->qs[j];
                        const uint8_t p4 = w4->qs[j];
                        const uint8_t p5 = w5->qs[j];
                        const uint8_t p6 = w6->qs[j];
                        const uint8_t p7 = w7->qs[j];

                        const float tv0 = token_vec[j];
                        const float tv1 = token_vec[j + 16];

                        /* Extract low nibbles */
                        const int lo0 = (p0 & 0x0F) - 8;
                        const int lo1 = (p1 & 0x0F) - 8;
                        const int lo2 = (p2 & 0x0F) - 8;
                        const int lo3 = (p3 & 0x0F) - 8;
                        const int lo4 = (p4 & 0x0F) - 8;
                        const int lo5 = (p5 & 0x0F) - 8;
                        const int lo6 = (p6 & 0x0F) - 8;
                        const int lo7 = (p7 & 0x0F) - 8;

                        __m256 xv = _mm256_set1_ps(tv0);
                        __m256 qw = _mm256_setr_ps(lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7);
                        __m256 vw = _mm256_setr_ps(d0, d1, d2, d3, d4, d5, d6, d7);
                        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_mul_ps(qw, vw), xv));

                        /* Extract high nibbles */
                        const int hi0 = (p0 >> 4) - 8;
                        const int hi1 = (p1 >> 4) - 8;
                        const int hi2 = (p2 >> 4) - 8;
                        const int hi3 = (p3 >> 4) - 8;
                        const int hi4 = (p4 >> 4) - 8;
                        const int hi5 = (p5 >> 4) - 8;
                        const int hi6 = (p6 >> 4) - 8;
                        const int hi7 = (p7 >> 4) - 8;

                        xv = _mm256_set1_ps(tv1);
                        qw = _mm256_setr_ps(hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7);
                        vw = _mm256_setr_ps(d0, d1, d2, d3, d4, d5, d6, d7);
                        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_mul_ps(qw, vw), xv));
                    }

                    /* Combine low and high accumulators */
                    __m256 total = _mm256_add_ps(acc0, acc1);

                    /* Store to output */
                    float *out_row = output + (size_t)t * embed_dim + n;
                    __m256 out_val = _mm256_loadu_ps(out_row);
                    out_val = _mm256_add_ps(out_val, total);
                    _mm256_storeu_ps(out_row, out_val);
                }
            }
        }

        /* Handle remaining output rows with scalar */
        for (; n < embed_dim; n++) {
            for (int n_block = 0; n_block < blocks_per_head; n_block++) {
                const block_q5_0 *w_row = weights + (size_t)(n + head_offset + n_block) * blocks_per_row + n_block;
                const float d = CK_FP16_TO_FP32(w_row->d);

                uint32_t qh;
                memcpy(&qh, w_row->qh, sizeof(qh));

                for (int t = 0; t < tokens; t++) {
                    const float *token_vec = head_data + (size_t)t * token_stride + (size_t)n_block * QK5_0;
                    float sum = 0.0f;

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
#if defined(__AVX__) && defined(__F16C__)
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
