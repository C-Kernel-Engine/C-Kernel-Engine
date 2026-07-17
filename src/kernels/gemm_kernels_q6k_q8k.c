/**
 * @file gemm_kernels_q6k_q8k.c
 * @brief Q6_K (weights) x Q8_K (activations) kernels for inference
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
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
#include <stdlib.h>

#include "ckernel_engine.h"
#include "ckernel_quant.h"

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__) || defined(__SSSE3__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* Forward declarations for SIMD implementations */
void gemv_q6_k_q8_k_avx512(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q6_k_q8_k_avx512_vbmi(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q6_k_q8_k_avx2(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q6_k_q8_k_avx(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q6_k_q8_k_sse(float *y, const void *W, const void *x_q8, int M, int K);

static int ck_q6k_q8k_force_ref(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("CK_DEBUG_Q6K_Q8K_REF");
        cached = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return cached;
}

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
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(w[i].d) * x[i].d;

        const uint8_t *ql = w[i].ql;
        const uint8_t *qh = w[i].qh;
        const int8_t *sc = w[i].scales;
        const int8_t *q8 = x[i].qs;
        int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};

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

                aux32[l & 7] += (int)sc[is + 0] * (int)q1 * (int)q8[l + 0];
                aux32[l & 7] += (int)sc[is + 2] * (int)q2 * (int)q8[l + 32];
                aux32[l & 7] += (int)sc[is + 4] * (int)q3 * (int)q8[l + 64];
                aux32[l & 7] += (int)sc[is + 6] * (int)q4 * (int)q8[l + 96];
            }
            q8 += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }

        for (int l = 0; l < 8; ++l) {
            sums[l] += d * (float)aux32[l];
        }
    }

    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    return sumf;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
static float dot_q6_k_q8_k_neon(const block_q6_K *w,
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

        int8_t wvals[QK_K];
        int8_t svals[QK_K];

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;

                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                const int base = n;
                wvals[base + l + 0] = q1;
                wvals[base + l + 32] = q2;
                wvals[base + l + 64] = q3;
                wvals[base + l + 96] = q4;

                svals[base + l + 0] = sc[is + 0];
                svals[base + l + 32] = sc[is + 2];
                svals[base + l + 64] = sc[is + 4];
                svals[base + l + 96] = sc[is + 6];
            }

            ql += 64;
            qh += 32;
            sc += 8;
        }

        int32x4_t acc = vdupq_n_s32(0);
        for (int j = 0; j < QK_K; j += 16) {
            const int8x16_t wv = vld1q_s8(&wvals[j]);
            const int8x16_t sv = vld1q_s8(&svals[j]);
            const int8x16_t xv = vld1q_s8(&q8[j]);

            const int16x8_t ws0 = vmull_s8(vget_low_s8(wv), vget_low_s8(sv));
            const int16x8_t ws1 = vmull_s8(vget_high_s8(wv), vget_high_s8(sv));
            const int16x8_t x0 = vmovl_s8(vget_low_s8(xv));
            const int16x8_t x1 = vmovl_s8(vget_high_s8(xv));

            int32x4_t p0 = vmull_s16(vget_low_s16(ws0), vget_low_s16(x0));
            p0 = vmlal_s16(p0, vget_high_s16(ws0), vget_high_s16(x0));

            int32x4_t p1 = vmull_s16(vget_low_s16(ws1), vget_low_s16(x1));
            p1 = vmlal_s16(p1, vget_high_s16(ws1), vget_high_s16(x1));

            acc = vaddq_s32(acc, p0);
            acc = vaddq_s32(acc, p1);
        }

        int32_t lanes[4];
        vst1q_s32(lanes, acc);
        sumf += d * (float)(lanes[0] + lanes[1] + lanes[2] + lanes[3]);
    }

    return sumf;
}
#endif

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
 * AVX Implementation (for Sandy/Ivy Bridge - AVX without AVX2)
 *
 * Same as SSE but with prefetching for next block.
 * Uses 128-bit integer ops (AVX doesn't add 256-bit int ops).
 * ============================================================================ */

#if defined(__AVX__) && !defined(__AVX2__)

static float dot_q6_k_q8_k_avx(const block_q6_K *w,
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

        /* Prefetch next block */
        if (i + 1 < nb) {
            _mm_prefetch((const char *)&w[i + 1], _MM_HINT_T0);
            _mm_prefetch((const char *)&x[i + 1], _MM_HINT_T0);
        }

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

void gemv_q6_k_q8_k_avx(float *y,
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
        y[row] = dot_q6_k_q8_k_avx(w_row, x, K);
    }
}

#endif /* __AVX__ && !__AVX2__ */

/* ============================================================================
 * AVX2 Implementation (for modern CPUs with AVX2)
 * ============================================================================ */

#if defined(__AVX2__)

/* Match ggml's x86 hsum_float_8 reduction tree exactly. Replacing the final
 * two additions with _mm_hadd_ps changes one FP32 ULP for realistic MLP-down
 * rows and can cross the absolute parity gate on AVX2/AVX-512 runners. */
static inline float ck_q6k_hsum_float_8(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

/* Q6_K optimization note:
 * ----------------------
 * llama.cpp's newer Q6_K AVX2 work inspired an experiment that separates the
 * zero-point correction from the hot unpack/dot loop by using Q8_K block sums
 * (bsums). That can reduce repeated subtract-32 work and improve some CPU
 * layouts, but it also changes reduction order and instruction balance.
 *
 * Current llama.cpp x86 kernels use the AVX2 reduction tree even in AVX-512
 * builds. CK uses the same tree for its AVX2 and non-VBMI AVX-512 paths. Keep
 * the final horizontal reduction in sync with ggml: an equivalent _mm_hadd_ps
 * tree differs by one FP32 ULP on realistic MLP-down rows.
 */

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
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(w[i].d) * x[i].d;

        const uint8_t *q4 = w[i].ql;
        const uint8_t *qh = w[i].qh;
        const int8_t *q8 = x[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)x[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i *)w[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(
            _mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(
                _mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(
                _mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(
                scales, get_scale_shuffle_avx2(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(
                scales, get_scale_shuffle_avx2(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(
                scales, get_scale_shuffle_avx2(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(
                scales, get_scale_shuffle_avx2(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return ck_q6k_hsum_float_8(acc);
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
 * AVX-512 Implementation
 *
 * Uses 512-bit ZMM registers to process 64 bytes at a time.
 * Processes entire 256-element Q6_K block in fewer iterations.
 * ============================================================================ */

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VBMI__)

/**
 * @brief AVX-512 dot product for Q6_K x Q8_K with VBMI
 *
 * Uses AVX-512 VBMI for efficient byte permutation.
 */
static float dot_q6_k_q8_k_avx512_vbmi(const block_q6_K *w,
                                        const block_q8_K *x,
                                        int K)
{
    const int nb = K / QK_K;
    const __m512i m4 = _mm512_set1_epi8(0xF);
    const __m512i m2 = _mm512_set1_epi8(3);
    const __m512i m32s = _mm512_set1_epi8(32);

    __m512 acc = _mm512_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(w[i].d) * x[i].d;

        const uint8_t *ql = w[i].ql;
        const uint8_t *qh = w[i].qh;
        const int8_t *q8 = x[i].qs;
        const int8_t *sc = w[i].scales;

        __m512i sumi = _mm512_setzero_si512();

        /* Process 256 weights in one iteration using AVX-512 */
        /* Load 64 bytes of low bits (covers 128 weights, need 2 loads for full block) */
        const __m512i q4bits1 = _mm512_loadu_si512((const __m512i *)ql);       /* ql[0..63] */
        const __m512i q4bits2 = _mm512_loadu_si512((const __m512i *)(ql + 64)); /* ql[64..127] */

        /* Load 64 bytes of high bits */
        const __m512i q4bitsH = _mm512_loadu_si512((const __m512i *)qh);

        /* Extract high 2-bit contributions for each group of 32 weights */
        /* Group 0: bits 0-1 of qh -> weights 0-31 */
        const __m512i q4h_0 = _mm512_slli_epi16(_mm512_and_si512(q4bitsH, m2), 4);
        /* Group 1: bits 2-3 of qh -> weights 32-63 */
        const __m512i q4h_1 = _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(q4bitsH, 2), m2), 4);
        /* Group 2: bits 4-5 of qh -> weights 64-95 */
        const __m512i q4h_2 = _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(q4bitsH, 4), m2), 4);
        /* Group 3: bits 6-7 of qh -> weights 96-127 */
        const __m512i q4h_3 = _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(q4bitsH, 6), m2), 4);

        /* Combine low nibbles with high bits to get 6-bit values (0-63) */
        /* First 64 weights: low nibbles of ql[0..63] */
        const __m512i q6_0 = _mm512_or_si512(_mm512_and_si512(q4bits1, m4), q4h_0);
        const __m512i q6_1 = _mm512_or_si512(_mm512_and_si512(q4bits2, m4), q4h_1);
        /* Second 64 weights: high nibbles of ql[0..63] */
        const __m512i q6_2 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(q4bits1, 4), m4), q4h_2);
        const __m512i q6_3 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(q4bits2, 4), m4), q4h_3);

        /* Load Q8_K values (256 int8 values = 4 x 64) */
        const __m512i q8_0 = _mm512_loadu_si512((const __m512i *)q8);
        const __m512i q8_1 = _mm512_loadu_si512((const __m512i *)(q8 + 64));
        const __m512i q8_2 = _mm512_loadu_si512((const __m512i *)(q8 + 128));
        const __m512i q8_3 = _mm512_loadu_si512((const __m512i *)(q8 + 192));

        /* Compute 32 * q8 for the offset subtraction */
        __m512i q8s_0 = _mm512_maddubs_epi16(m32s, q8_0);
        __m512i q8s_1 = _mm512_maddubs_epi16(m32s, q8_1);
        __m512i q8s_2 = _mm512_maddubs_epi16(m32s, q8_2);
        __m512i q8s_3 = _mm512_maddubs_epi16(m32s, q8_3);

        /* Multiply unsigned q6 * signed q8 */
        __m512i p16_0 = _mm512_maddubs_epi16(q6_0, q8_0);
        __m512i p16_1 = _mm512_maddubs_epi16(q6_1, q8_1);
        __m512i p16_2 = _mm512_maddubs_epi16(q6_2, q8_2);
        __m512i p16_3 = _mm512_maddubs_epi16(q6_3, q8_3);

        /* Subtract offset: (q6 - 32) * q8 = q6*q8 - 32*q8 */
        p16_0 = _mm512_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm512_sub_epi16(p16_1, q8s_1);
        p16_2 = _mm512_sub_epi16(p16_2, q8s_2);
        p16_3 = _mm512_sub_epi16(p16_3, q8s_3);

        /* Load and broadcast scales using VBMI permute
         * Each scale applies to 16 weights, so we need to broadcast appropriately
         * scales[0..15] for the 16 sub-blocks */
        const __m128i scales_128 = _mm_loadu_si128((const __m128i *)sc);

        /* Create scale broadcast patterns for 64 weights (4 scales per 64 weights) */
        /* Pattern: each scale repeated 16 times for 16 weights */
        const __m512i scale_idx_0 = _mm512_set_epi8(
            3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
            2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        const __m512i scale_idx_1 = _mm512_set_epi8(
            7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
            6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
            5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
            4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4);
        const __m512i scale_idx_2 = _mm512_set_epi8(
            11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
            10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
            9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
            8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8);
        const __m512i scale_idx_3 = _mm512_set_epi8(
            15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
            14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
            13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
            12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12);

        /* Broadcast scales to 512-bit using VBMI permutexvar */
        const __m512i scales_512 = _mm512_broadcast_i32x4(scales_128);
        const __m512i sc_0 = _mm512_permutexvar_epi8(scale_idx_0, scales_512);
        const __m512i sc_1 = _mm512_permutexvar_epi8(scale_idx_1, scales_512);
        const __m512i sc_2 = _mm512_permutexvar_epi8(scale_idx_2, scales_512);
        const __m512i sc_3 = _mm512_permutexvar_epi8(scale_idx_3, scales_512);

        /* Sign-extend scales to 16-bit and multiply with products */
        /* For efficiency, we process in two halves (low and high 256 bits) */
        __m512i p32_0 = _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(sc_0)), p16_0);
        __m512i p32_1 = _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(sc_1)), p16_1);
        __m512i p32_2 = _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(sc_2)), p16_2);
        __m512i p32_3 = _mm512_madd_epi16(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(sc_3)), p16_3);

        /* Accumulate */
        sumi = _mm512_add_epi32(sumi, p32_0);
        sumi = _mm512_add_epi32(sumi, p32_1);
        sumi = _mm512_add_epi32(sumi, p32_2);
        sumi = _mm512_add_epi32(sumi, p32_3);

        /* Scale by d and accumulate */
        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }

    return _mm512_reduce_add_ps(acc);
}

void gemv_q6_k_q8_k_avx512_vbmi(float *y,
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
        y[row] = dot_q6_k_q8_k_avx512_vbmi(w_row, x, K);
    }
}

#endif /* __AVX512F__ && __AVX512BW__ && __AVX512VBMI__ */

#if defined(__AVX512F__) && defined(__AVX512BW__)

/**
 * @brief AVX-512 dot product for Q6_K x Q8_K
 *
 * Works on all AVX-512 CPUs (Skylake-X and newer).
 * Uses same algorithm as AVX2, but benefits from AVX-512's wider FMA
 * and efficient horizontal reduction.
 */
static float dot_q6_k_q8_k_avx512(const block_q6_K *w,
                                   const block_q8_K *x,
                                   int K)
{
    const int nb = K / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    /* Use 256-bit float accumulator, same as AVX2 */
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(w[i].d) * x[i].d;

        const uint8_t *q4 = w[i].ql;
        const uint8_t *qh = w[i].qh;
        const int8_t *q8 = x[i].qs;

        const __m128i scales = _mm_loadu_si128((const __m128i *)w[i].scales);

        /* Use 256-bit integer accumulator, same as AVX2 */
        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        /* Process 256 weights in 2 iterations of 128 (same structure as AVX2) */
        for (int j = 0; j < QK_K / 128; ++j) {
            /* Get scale shuffle patterns - identical to AVX2 */
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

            const __m128i scale_0 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)patterns[is + 0]));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)patterns[is + 1]));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)patterns[is + 2]));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)patterns[is + 3]));
            is += 4;

            /* Load low bits */
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
            qh += 32;

            /* Extract high 2-bit contributions */
            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
            const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
            const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

            /* Combine low + high bits to get 6-bit values */
            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            /* Load Q8_K values */
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;

            /* Compute 32 * q8 for offset */
            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            /* Multiply q4 * q8 (unsigned * signed) */
            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            /* Subtract offset: (q4 - 32) * q8 = q4*q8 - 32*q8 */
            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            /* Apply scales - produces 8 int32 each */
            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            /* Accumulate all 4 into sumi (same as AVX2) */
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        /* Scale by d and accumulate */
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return ck_q6k_hsum_float_8(acc);
}

void gemv_q6_k_q8_k_avx512(float *y,
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
        y[row] = dot_q6_k_q8_k_avx512(w_row, x, K);
    }
}

#endif /* __AVX512F__ && __AVX512BW__ */

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

    /* This is the architecture-neutral scalar oracle. Production x86 dispatch
     * uses the separately parity-tested SIMD reduction tree. */
    *s = dot_q6_k_q8_k_ref(x, y, n);
}

/**
 * @brief GEMV: y = W @ x where W is Q6_K and x is Q8_K
 */
void gemv_q6_k_q8_k(float *y,
                     const void *W,
                     const void *x_q8,
                     int M, int K)
{
    if (ck_strict_parity_enabled() || ck_q6k_q8k_force_ref()) {
        gemv_q6_k_q8_k_ref(y, W, x_q8, M, K);
        return;
    }

#if defined(__AVX2__)
    /* llama.cpp's x86 Q6_K production graph keeps the AVX2 reduction order
     * even when AVX-512 is available. Wider ISA availability is not a license
     * to change this numerical contract. */
    gemv_q6_k_q8_k_avx2(y, W, x_q8, M, K);
    return;
#elif defined(__AVX__)
    gemv_q6_k_q8_k_avx(y, W, x_q8, M, K);
    return;
#elif defined(__SSE4_1__)
    gemv_q6_k_q8_k_sse(y, W, x_q8, M, K);
    return;
#endif
    gemv_q6_k_q8_k_ref(y, W, x_q8, M, K);
}

const char *ck_q6_k_q8_k_provider_name(void)
{
    if (ck_strict_parity_enabled() || ck_q6k_q8k_force_ref()) {
        return "q6_k_q8_k_ref";
    }
#if defined(__AVX2__)
    return "q6_k_q8_k_avx2";
#elif defined(__AVX__)
    return "q6_k_q8_k_avx";
#elif defined(__SSE4_1__)
    return "q6_k_q8_k_sse";
#else
    return "q6_k_q8_k_ref";
#endif
}

/* ============================================================================
 * PARALLEL VERSIONS (for parallel orchestration)
 *
 * These receive ith (thread index) and nth (total threads) from orchestration.
 * OpenMP lives in orchestration layer, NOT here.
 *
 * Naming: *_parallel = receives ith/nth, processes only its portion
 * ============================================================================ */

/**
 * @brief Parallel reference GEMV for Q6_K × Q8_K
 *
 * Caller provides ith (thread index) and nth (total threads).
 * Each thread processes rows [r0, r1).
 */
void gemv_q6_k_q8_k_parallel(float *y,
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

    const block_q6_K *blocks = (const block_q6_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = r0; row < r1; ++row) {
        const block_q6_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q6_k_q8_k_ref(w_row, x, K);
    }
}

/**
 * @brief Parallel SIMD GEMV for Q6_K × Q8_K
 *
 * Uses best available SIMD (AVX/SSE) with row prefetching.
 * Caller provides ith/nth from OpenMP region.
 */
void gemv_q6_k_q8_k_parallel_simd(float *y,
                                   const void *W,
                                   const void *x_q8,
                                   int M, int K,
                                   int ith, int nth)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) return;
    if (ith < 0 || nth <= 0 || ith >= nth) return;

    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;

    if (r0 >= M) return;

    const block_q6_K *blocks = (const block_q6_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;
    const int strict = ck_strict_parity_enabled() || ck_q6k_q8k_force_ref();

    for (int row = r0; row < r1; ++row) {
        const block_q6_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
#if defined(__AVX2__)
        y[row] = strict ? dot_q6_k_q8_k_ref(w_row, x, K)
                        : dot_q6_k_q8_k_avx2(w_row, x, K);
#elif defined(__AVX__)
        y[row] = strict ? dot_q6_k_q8_k_ref(w_row, x, K)
                        : dot_q6_k_q8_k_avx(w_row, x, K);
#elif defined(__SSE4_1__)
        y[row] = strict ? dot_q6_k_q8_k_ref(w_row, x, K)
                        : dot_q6_k_q8_k_sse(w_row, x, K);
#else
        y[row] = dot_q6_k_q8_k_ref(w_row, x, K);
#endif
    }
}

static inline float ck_dot_q6_k_q8_k_fast_or_ref(const block_q6_K *w,
                                                  const block_q8_K *x,
                                                  int K);

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

    /* Prefill GEMM is the hot Qwen2/Qwen3.5 MLP-down path. Keep decode
     * gemv_q6_k_q8_k() conservative, but allow GEMM/prefill to use the
     * parity-gated SIMD dot helper by default. CK strict parity and
     * CK_DEBUG_Q6K_Q8K_REF=1 still force the scalar reference reduction. */
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q6_K *W = (const block_q6_K *)B;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;

    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; ++n) {
            const block_q6_K *w_row = W + (size_t)n * (size_t)blocks_per_row;
            const float b = bias ? bias[n] : 0.0f;
            c_row[n] = ck_dot_q6_k_q8_k_fast_or_ref(w_row, a_row, K) + b;
        }
    }
}

static inline float ck_dot_q6_k_q8_k_fast_or_ref(const block_q6_K *w,
                                                  const block_q8_K *x,
                                                  int K)
{
    if (ck_strict_parity_enabled() || ck_q6k_q8k_force_ref()) {
        return dot_q6_k_q8_k_ref(w, x, K);
    }
#if defined(__AVX2__)
    return dot_q6_k_q8_k_avx2(w, x, K);
#elif defined(__AVX__)
    return dot_q6_k_q8_k_avx(w, x, K);
#elif defined(__SSE4_1__)
    return dot_q6_k_q8_k_sse(w, x, K);
#else
    return dot_q6_k_q8_k_ref(w, x, K);
#endif
}

/**
 * @brief Compute one C[m0:m1, n0:n1] tile for Q8_K activations x Q6_K weights.
 *
 * Pure tile math only: no threadpool, no global scheduling, no allocation.
 * The orchestrator decides how to split tile jobs across cores.
 */
void gemm_nt_q6_k_q8_k_tile(const void *A_q8,
                             const void *B,
                             const float *bias,
                             float *C,
                             int M, int N, int K,
                             int m0, int m1,
                             int n0, int n1)
{
    if (!A_q8 || !B || !C) {
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0 || K % QK_K != 0) {
        return;
    }
    if (m0 < 0) m0 = 0;
    if (n0 < 0) n0 = 0;
    if (m1 > M) m1 = M;
    if (n1 > N) n1 = N;
    if (m0 >= m1 || n0 >= n1) {
        return;
    }

    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q6_K *W = (const block_q6_K *)B;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;

    for (int n = n0; n < n1; ++n) {
        const block_q6_K *w_row = W + (size_t)n * (size_t)blocks_per_row;
        const float b = bias ? bias[n] : 0.0f;
        for (int m = m0; m < m1; ++m) {
            const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
            C[(size_t)m * (size_t)N + (size_t)n] = ck_dot_q6_k_q8_k_fast_or_ref(w_row, a_row, K) + b;
        }
    }
}

/**
 * @brief Experimental single-thread tiled NT GEMM wrapper.
 *
 * Kept as a separate symbol from gemm_nt_q6_k_q8_k for benchmarks and parity.
 * Production prefill should prefer the v8 2D tile scheduler when enabled.
 */
void gemm_nt_q6_k_q8_k_tiled(const void *A_q8,
                              const void *B,
                              const float *bias,
                              float *C,
                              int M, int N, int K)
{
    enum { TILE_M = 8, TILE_N = 16 };
    for (int n0 = 0; n0 < N; n0 += TILE_N) {
        const int n1 = (n0 + TILE_N < N) ? (n0 + TILE_N) : N;
        for (int m0 = 0; m0 < M; m0 += TILE_M) {
            const int m1 = (m0 + TILE_M < M) ? (m0 + TILE_M) : M;
            gemm_nt_q6_k_q8_k_tile(A_q8, B, bias, C, M, N, K, m0, m1, n0, n1);
        }
    }
}
