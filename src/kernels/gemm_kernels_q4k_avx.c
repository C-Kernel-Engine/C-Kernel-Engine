/**
 * @file gemm_kernels_q4k_avx.c
 * @brief AVX Q4_K x Q8_K matvec kernel for Sandy/Ivy Bridge
 *
 * Uses _mm_maddubs_epi16 (SSSE3) for efficient u8*s8 multiply-add while
 * maintaining our scale format from unpack_q4_k_scales.
 *
 * Key improvement over SSE: _mm_maddubs_epi16 processes 16 pairs per
 * instruction vs SSE's _mm_cvtepu8_epi16 + _mm_madd_epi16 (8 pairs).
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

#if defined(__AVX__)
#include <immintrin.h>

static inline int32_t hsum_epi32_sse(__m128i v) {
    __m128i shuf = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sums = _mm_add_epi32(v, shuf);
    shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
    sums = _mm_add_epi32(sums, shuf);
    return _mm_cvtsi128_si32(sums);
}

void gemv_q4_k_q8_k_avx(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *bx = (const block_q8_K *)x_q8;
    const int nb = K / QK_K;

    const __m128i mask_low = _mm_set1_epi8(0x0F);
    const __m128i ones = _mm_set1_epi16(1);

    for (int row = 0; row < M; ++row) {
        const block_q4_K *x = blocks + row * nb;
        float sumf = 0.0f;

        for (int i = 0; i < nb; ++i) {
            /* Prefetch next block */
            if (i + 1 < nb) {
                _mm_prefetch((const char *)&x[i + 1], _MM_HINT_T0);
                _mm_prefetch((const char *)&bx[i + 1], _MM_HINT_T0);
            }

            /* Unpack scales using our format */
            uint8_t sc[8], m_val[8];
            unpack_q4_k_scales(x[i].scales, sc, m_val);

            const float d = CK_FP16_TO_FP32(x[i].d) * bx[i].d;
            const float dmin = CK_FP16_TO_FP32(x[i].dmin) * bx[i].d;

            const uint8_t *q4 = x[i].qs;
            const int8_t *q8 = bx[i].qs;

            int is = 0;

            /* Process 4 groups of 64 elements */
            for (int j = 0; j < QK_K; j += 64) {
                __m128i acc_lo = _mm_setzero_si128();
                __m128i acc_hi = _mm_setzero_si128();

                /* Process 32 bytes of Q4 (64 elements via nibbles) */
                for (int l = 0; l < 32; l += 16) {
                    /* Load 16 bytes of Q4 */
                    __m128i q4_vec = _mm_loadu_si128((const __m128i *)(q4 + l));

                    /* Extract low and high nibbles */
                    __m128i q4_lo = _mm_and_si128(q4_vec, mask_low);
                    __m128i q4_hi = _mm_and_si128(_mm_srli_epi16(q4_vec, 4), mask_low);

                    /* Load Q8 values - low nibbles correspond to j+l, high to j+32+l */
                    __m128i q8_lo_vec = _mm_loadu_si128((const __m128i *)(q8 + j + l));
                    __m128i q8_hi_vec = _mm_loadu_si128((const __m128i *)(q8 + j + 32 + l));

                    /* _mm_maddubs_epi16: unsigned*signed multiply-add
                     * Multiplies 16 pairs of bytes, returns 8 int16 sums
                     * Result[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1] */
                    __m128i prod_lo = _mm_maddubs_epi16(q4_lo, q8_lo_vec);
                    __m128i prod_hi = _mm_maddubs_epi16(q4_hi, q8_hi_vec);

                    /* Convert to int32 by multiplying by 1 with madd_epi16 */
                    acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(prod_lo, ones));
                    acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(prod_hi, ones));
                }

                int32_t sum_q4q8_lo = hsum_epi32_sse(acc_lo);
                int32_t sum_q4q8_hi = hsum_epi32_sse(acc_hi);

                /* bsums: each bsum is 16 elements */
                int32_t bsum_lo = (int32_t)bx[i].bsums[j / 16] +
                                  (int32_t)bx[i].bsums[j / 16 + 1];
                int32_t bsum_hi = (int32_t)bx[i].bsums[(j + 32) / 16] +
                                  (int32_t)bx[i].bsums[(j + 32) / 16 + 1];

                sumf += d * (float)sc[is] * (float)sum_q4q8_lo;
                sumf -= dmin * (float)m_val[is] * (float)bsum_lo;
                sumf += d * (float)sc[is + 1] * (float)sum_q4q8_hi;
                sumf -= dmin * (float)m_val[is + 1] * (float)bsum_hi;

                q4 += 32;
                is += 2;
            }
        }
        y[row] = sumf;
    }
}

/* ============================================================================
 * PARALLEL SIMD VERSION
 *
 * Combines AVX SIMD with parallel row splitting for maximum throughput.
 * OpenMP lives in orchestration layer - this kernel receives ith/nth.
 *
 * Prefetch strategy:
 * - Prefetch 2-4 rows ahead (hide memory latency ~50-70ns)
 * - Each row = (K/256) * 144 bytes = ~576 bytes for K=1024
 * - Computation per row ~ 100ns, so prefetch 1-2 rows ahead
 * ============================================================================ */

void gemv_q4_k_q8_k_parallel_simd(float *y,
                                   const void *W,
                                   const void *x_q8,
                                   int M, int K,
                                   int ith, int nth)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) return;
    if (ith < 0 || nth <= 0 || ith >= nth) return;

    /* Compute row range for this thread */
    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;

    if (r0 >= M) return;

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *bx = (const block_q8_K *)x_q8;
    const int nb = K / QK_K;
    const size_t bytes_per_row = (size_t)nb * sizeof(block_q4_K);

    const __m128i mask_low = _mm_set1_epi8(0x0F);
    const __m128i ones = _mm_set1_epi16(1);

    /* Prefetch first few rows for this thread */
    const int PREFETCH_ROWS = 4;
    for (int p = 0; p < PREFETCH_ROWS && r0 + p < r1; ++p) {
        const char *row_ptr = (const char *)(blocks + (r0 + p) * nb);
        _mm_prefetch(row_ptr, _MM_HINT_T0);
        _mm_prefetch(row_ptr + 64, _MM_HINT_T0);  /* Second cache line */
    }

    for (int row = r0; row < r1; ++row) {
        /* Prefetch rows ahead to hide memory latency */
        if (row + PREFETCH_ROWS < r1) {
            const char *prefetch_ptr = (const char *)(blocks + (row + PREFETCH_ROWS) * nb);
            _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
            _mm_prefetch(prefetch_ptr + 64, _MM_HINT_T0);
            _mm_prefetch(prefetch_ptr + 128, _MM_HINT_T0);
        }

        const block_q4_K *x = blocks + row * nb;
        float sumf = 0.0f;

        for (int i = 0; i < nb; ++i) {
            /* Prefetch next block within row */
            if (i + 1 < nb) {
                _mm_prefetch((const char *)&x[i + 1], _MM_HINT_T0);
            }

            /* Unpack scales using our format */
            uint8_t sc[8], m_val[8];
            unpack_q4_k_scales(x[i].scales, sc, m_val);

            const float d = CK_FP16_TO_FP32(x[i].d) * bx[i].d;
            const float dmin = CK_FP16_TO_FP32(x[i].dmin) * bx[i].d;

            const uint8_t *q4 = x[i].qs;
            const int8_t *q8 = bx[i].qs;

            int is = 0;

            /* Process 4 groups of 64 elements */
            for (int j = 0; j < QK_K; j += 64) {
                __m128i acc_lo = _mm_setzero_si128();
                __m128i acc_hi = _mm_setzero_si128();

                /* Process 32 bytes of Q4 (64 elements via nibbles) */
                for (int l = 0; l < 32; l += 16) {
                    /* Load 16 bytes of Q4 */
                    __m128i q4_vec = _mm_loadu_si128((const __m128i *)(q4 + l));

                    /* Extract low and high nibbles */
                    __m128i q4_lo = _mm_and_si128(q4_vec, mask_low);
                    __m128i q4_hi = _mm_and_si128(_mm_srli_epi16(q4_vec, 4), mask_low);

                    /* Load Q8 values */
                    __m128i q8_lo_vec = _mm_loadu_si128((const __m128i *)(q8 + j + l));
                    __m128i q8_hi_vec = _mm_loadu_si128((const __m128i *)(q8 + j + 32 + l));

                    /* _mm_maddubs_epi16: unsigned*signed multiply-add */
                    __m128i prod_lo = _mm_maddubs_epi16(q4_lo, q8_lo_vec);
                    __m128i prod_hi = _mm_maddubs_epi16(q4_hi, q8_hi_vec);

                    /* Convert to int32 */
                    acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(prod_lo, ones));
                    acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(prod_hi, ones));
                }

                int32_t sum_q4q8_lo = hsum_epi32_sse(acc_lo);
                int32_t sum_q4q8_hi = hsum_epi32_sse(acc_hi);

                /* bsums: each bsum is 16 elements */
                int32_t bsum_lo = (int32_t)bx[i].bsums[j / 16] +
                                  (int32_t)bx[i].bsums[j / 16 + 1];
                int32_t bsum_hi = (int32_t)bx[i].bsums[(j + 32) / 16] +
                                  (int32_t)bx[i].bsums[(j + 32) / 16 + 1];

                sumf += d * (float)sc[is] * (float)sum_q4q8_lo;
                sumf -= dmin * (float)m_val[is] * (float)bsum_lo;
                sumf += d * (float)sc[is + 1] * (float)sum_q4q8_hi;
                sumf -= dmin * (float)m_val[is + 1] * (float)bsum_hi;

                q4 += 32;
                is += 2;
            }
        }
        y[row] = sumf;
    }
}

#else
/* Fallback for non-AVX builds */
void gemv_q4_k_q8_k_ref(float *y, const void *W, const void *x_q8, int M, int K);

void gemv_q4_k_q8_k_avx(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
}

/* Parallel fallback when no AVX */
void gemv_q4_k_q8_k_parallel(float *y, const void *W, const void *x_q8,
                              int M, int K, int ith, int nth);

void gemv_q4_k_q8_k_parallel_simd(float *y,
                                   const void *W,
                                   const void *x_q8,
                                   int M, int K,
                                   int ith, int nth)
{
    /* Fall back to reference parallel version */
    gemv_q4_k_q8_k_parallel(y, W, x_q8, M, K, ith, nth);
}
#endif
