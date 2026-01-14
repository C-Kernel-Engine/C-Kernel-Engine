/**
 * @file gemm_kernels_q6k_q8k.c
 * @brief Q6_K (weights) x Q8_K (activations) kernels for inference
 *
 * Implements decode-style matvec/matmul where weights are Q6_K and the
 * activations are quantized on-the-fly to Q8_K. This is inference-only;
 * no backward pass is provided here.
 *
 * Q6_K Format (256 weights per block):
 *   - d: FP16 super-block scale
 *   - ql: 128 bytes (low 4 bits of each weight)
 *   - qh: 64 bytes (high 2 bits of each weight)
 *   - scales: 16 int8 sub-block scales
 *
 * Q8_K Format (256 weights per block):
 *   - d: FP32 scale
 *   - qs: 256 int8 values
 *   - bsums: 16 int16 block sums
 */

#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "ckernel_quant.h"

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__) || defined(__SSSE3__)
#include <immintrin.h>
#endif

/* Forward declarations for SIMD implementations */
void gemv_q6_k_q8_k_avx2(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q6_k_q8_k_sse(float *y, const void *W, const void *x_q8, int M, int K);

/* ============================================================================
 * Reference Implementation
 * ============================================================================ */

/**
 * @brief Scalar dot product for Q6_K x Q8_K
 *
 * Q6_K layout: 256 weights per block
 *   - ql[0..127]: low 4 bits for all 256 weights (packed 2 per byte)
 *   - qh[0..63]: high 2 bits for all 256 weights (packed 4 per byte)
 *   - scales[0..15]: int8 scale for each 16-weight sub-block
 *   - d: FP16 super-block scale
 *
 * The dequantization formula for each weight is:
 *   weight = d * scale[sub] * (q6_value - 32)
 * where q6_value is the 6-bit unsigned value (0..63).
 */
static float dot_q6_k_q8_k_ref(const block_q6_K *w,
                                const block_q8_K *x,
                                int K)
{
    const int nb = K / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(w[i].d) * x[i].d;

        const uint8_t *ql = w[i].ql;
        const uint8_t *qh = w[i].qh;
        const int8_t *sc = w[i].scales;
        const int8_t *q8 = x[i].qs;

        /* Process 256 weights in 2 iterations of 128 */
        for (int n = 0; n < QK_K; n += 128) {
            /* Each iteration processes 128 weights:
             * - ql[0..63] contains low 4 bits
             * - qh[0..31] contains high 2 bits
             * - Interleaved pattern: weights 0-31, 32-63, 64-95, 96-127
             */
            for (int l = 0; l < 32; ++l) {
                /* Sub-block index: each scale covers 16 weights */
                const int is = l / 16;

                /* Extract 6-bit values from packed format */
                /* q1: weights l+0 (low nibble of ql[l], bits 0-1 of qh[l]) */
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                /* q2: weights l+32 (low nibble of ql[l+32], bits 2-3 of qh[l]) */
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                /* q3: weights l+64 (high nibble of ql[l], bits 4-5 of qh[l]) */
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                /* q4: weights l+96 (high nibble of ql[l+32], bits 6-7 of qh[l]) */
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                /* Accumulate: d * scale * q6 * q8 */
                sumf += d * (float)sc[is + 0] * (float)q1 * (float)q8[l + 0];
                sumf += d * (float)sc[is + 2] * (float)q2 * (float)q8[l + 32];
                sumf += d * (float)sc[is + 4] * (float)q3 * (float)q8[l + 64];
                sumf += d * (float)sc[is + 6] * (float)q4 * (float)q8[l + 96];
            }
            q8 += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    return sumf;
}

void gemv_q6_k_q8_k_ref(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    const block_q6_K *blocks = (const block_q6_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q6_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q6_k_q8_k_ref(w_row, x, K);
    }
}

/* ============================================================================
 * SSE4.1 Implementation (for Ivy Bridge and older AVX-without-AVX2 CPUs)
 *
 * Uses 128-bit SSE operations with maddubs for integer multiply-add.
 * Handles the -32 offset using bsums from Q8_K.
 * ============================================================================ */

#if defined(__SSSE3__)

/* Scale shuffle indices for Q6_K - maps scale index to 16-byte shuffle pattern */
static const int8_t q6k_scale_shuffle[8][16] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 },  /* is=0: scales[0,1] */
    { 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 },  /* is=1: scales[2,3] */
    { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5 },  /* is=2: scales[4,5] */
    { 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 },  /* is=3: scales[6,7] */
    { 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 },  /* is=4: scales[8,9] */
    {10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11 },  /* is=5: scales[10,11] */
    {12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13 },  /* is=6: scales[12,13] */
    {14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15 },  /* is=7: scales[14,15] */
};

static float dot_q6_k_q8_k_sse(const block_q6_K *w,
                                const block_q8_K *x,
                                int K)
{
    const int nb = K / QK_K;
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m15 = _mm_set1_epi8(15);

    __m128 acc = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(w[i].d) * x[i].d;

        const uint8_t *ql = w[i].ql;
        const uint8_t *qh = w[i].qh;
        const int8_t *q8 = x[i].qs;

        /* Load scales and precompute the -32 offset contribution using bsums */
        const __m128i scales = _mm_loadu_si128((const __m128i *)w[i].scales);
        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i *)x[i].bsums);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i *)x[i].bsums + 1);

        /* Compute: sum(scale * bsum) * 32 for the -32 offset */
        const __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
        const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        const __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
        const __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int is = 0;

        /* Process 256 weights in 2 iterations of 128 */
        for (int j = 0; j < QK_K / 128; ++j) {
            /* Load high bits */
            const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i *)qh);
            qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i *)qh);
            qh += 16;

            /* Extract and shift high bits into position */
            const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
            const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
            const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
            const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
            const __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
            const __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
            const __m128i q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
            const __m128i q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

            /* Load low bits */
            const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i *)ql);
            ql += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i *)ql);
            ql += 16;
            const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i *)ql);
            ql += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i *)ql);
            ql += 16;

            /* Combine low and high bits to get 6-bit values (unsigned 0..63) */
            const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), q4h_0);
            const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), q4h_1);
            const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), q4h_2);
            const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), q4h_3);
            const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), q4h_4);
            const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), q4h_5);
            const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), q4h_6);
            const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), q4h_7);

            /* Load Q8_K values */
            const __m128i q8_0 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i *)q8);
            q8 += 16;

            /* Multiply: maddubs treats first arg as unsigned, second as signed */
            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            /* Get scales for this iteration */
            const __m128i scale_0 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)q6k_scale_shuffle[is + 0]));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)q6k_scale_shuffle[is + 1]));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)q6k_scale_shuffle[is + 2]));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)q6k_scale_shuffle[is + 3]));
            is += 4;

            /* Scale the products and widen to 32-bit */
            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8)), p16_7);

            /* Accumulate */
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));
        }

        /* Subtract the -32 offset contribution */
        sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
        sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);

        /* Combine and convert to float */
        __m128i sumi = _mm_add_epi32(sumi_0, sumi_1);
        __m128 sumf_vec = _mm_mul_ps(_mm_set1_ps(d), _mm_cvtepi32_ps(sumi));

        /* Horizontal sum */
        sumf_vec = _mm_hadd_ps(sumf_vec, sumf_vec);
        sumf_vec = _mm_hadd_ps(sumf_vec, sumf_vec);
        acc = _mm_add_ss(acc, sumf_vec);
    }

    return _mm_cvtss_f32(acc);
}

void gemv_q6_k_q8_k_sse(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    const block_q6_K *blocks = (const block_q6_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q6_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q6_k_q8_k_sse(w_row, x, K);
    }
}
#endif /* __SSSE3__ */

/* ============================================================================
 * AVX2 Implementation (for modern CPUs with AVX2)
 * ============================================================================ */

#if defined(__AVX2__)

/* Scale shuffle for AVX2 - 32-byte version */
static const int8_t q6k_scale_shuffle_avx2[4][32] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
    { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
    { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 },
    { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 },
};

static inline __m128i get_scale_shuffle_avx2(int i) {
    static const uint8_t patterns[8][16] = {
        { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 },
        { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5 },
        { 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 },
        { 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 },
        {10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11 },
        {12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13 },
        {14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15 },
    };
    return _mm_loadu_si128((const __m128i *)patterns[i]);
}

static float dot_q6_k_q8_k_avx2(const block_q6_K *w,
                                 const block_q8_K *x,
                                 int K)
{
    const int nb = K / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(w[i].d) * x[i].d;

        const uint8_t *q4 = w[i].ql;
        const uint8_t *qh = w[i].qh;
        const int8_t *q8 = x[i].qs;

        const __m128i scales = _mm_loadu_si128((const __m128i *)w[i].scales);

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle_avx2(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle_avx2(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle_avx2(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle_avx2(is + 3));
            is += 4;

            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
            const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
            const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;

            /* Compute -32 * q8 contribution */
            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            /* Multiply q4 * q8 */
            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            /* Subtract offset: (q4 - 32) * q8 = q4*q8 - 32*q8 */
            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            /* Apply scales */
            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

void gemv_q6_k_q8_k_avx2(float *y,
                          const void *W,
                          const void *x_q8,
                          int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    const block_q6_K *blocks = (const block_q6_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q6_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q6_k_q8_k_avx2(w_row, x, K);
    }
}
#endif /* __AVX2__ */

/* ============================================================================
 * Dispatch Functions
 * ============================================================================ */

/**
 * @brief Q6_K x Q8_K dot product (single row)
 */
void vec_dot_q6_k_q8_k(int n, float *s, const void *vx, const void *vy)
{
    if (!s || !vx || !vy || n <= 0) {
        return;
    }

    const block_q6_K *x = (const block_q6_K *)vx;
    const block_q8_K *y = (const block_q8_K *)vy;

#if defined(__AVX2__)
    *s = dot_q6_k_q8_k_avx2(x, y, n);
#elif defined(__SSSE3__)
    *s = dot_q6_k_q8_k_sse(x, y, n);
#else
    *s = dot_q6_k_q8_k_ref(x, y, n);
#endif
}

/**
 * @brief GEMV: y = W @ x where W is Q6_K and x is Q8_K
 */
void gemv_q6_k_q8_k(float *y,
                     const void *W,
                     const void *x_q8,
                     int M, int K)
{
#if defined(__AVX2__)
    gemv_q6_k_q8_k_avx2(y, W, x_q8, M, K);
#elif defined(__SSSE3__)
    gemv_q6_k_q8_k_sse(y, W, x_q8, M, K);
#else
    gemv_q6_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}

/**
 * @brief GEMM: Y = W @ X^T where W is Q6_K and X is Q8_K
 *
 * @param Y Output matrix [N x M] in row-major
 * @param W Weight matrix in Q6_K format [M x K]
 * @param X_q8 Input matrix in Q8_K format [N x K]
 * @param M Number of output rows (output dim)
 * @param N Number of input vectors (batch size)
 * @param K Input dimension
 */
void gemm_q6_k_q8_k(float *Y,
                     const void *W,
                     const void *X_q8,
                     int M, int N, int K)
{
    if (!Y || !W || !X_q8 || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    const block_q8_K *X = (const block_q8_K *)X_q8;
    const int blocks_per_vec = K / QK_K;

    for (int n = 0; n < N; ++n) {
        const block_q8_K *x_row = X + (size_t)n * (size_t)blocks_per_vec;
        gemv_q6_k_q8_k(&Y[n * M], W, x_row, M, K);
    }
}

/**
 * @brief NT GEMM: C = A @ B^T where A is Q8_K and B is Q6_K
 *
 * This is the typical inference pattern:
 *   - A: Activations in Q8_K format [M x K]
 *   - B: Weights in Q6_K format [N x K]
 *   - C: Output [M x N]
 *
 * @param A_q8 Input activations in Q8_K format
 * @param B Weight matrix in Q6_K format
 * @param bias Optional bias vector [N]
 * @param C Output matrix
 * @param M Batch size (number of tokens)
 * @param N Output dimension
 * @param K Input dimension
 */
void gemm_nt_q6_k_q8_k(const void *A_q8,
                        const void *B,
                        const float *bias,
                        float *C,
                        int M, int N, int K)
{
    if (!A_q8 || !B || !C) {
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    gemm_q6_k_q8_k(C, B, A_q8, /*M_out=*/N, /*N_batch=*/M, K);

    if (!bias) {
        return;
    }

    for (int i = 0; i < M; ++i) {
        float *row = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            row[j] += bias[j];
        }
    }
}
