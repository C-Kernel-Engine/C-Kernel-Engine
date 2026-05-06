/**
 * @file quantize_row_q8_k_avx512.c
 * @brief AVX-512 entrypoint for exact Q8_K row quantization
 */

#include <assert.h>
#include <math.h>
#include <string.h>

#include "ckernel_quant.h"

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#endif

void quantize_row_q8_k_ref(const float *x, void *vy, int k);

void quantize_row_q8_k_avx512(const float *x, void *vy, int k) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (!x || !vy || k <= 0) {
        return;
    }
    assert(k % QK_K == 0);

    const int nb = k / QK_K;
    block_q8_K *y = (block_q8_K *)vy;

    for (int i = 0; i < nb; ++i) {
        /* Preserve the scalar signed-max tie contract used by llama.cpp/ref. */
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            const float xv = x[j];
            const float ax = fabsf(xv);
            if (ax > amax) {
                amax = ax;
                max = xv;
            }
        }

        if (amax == 0.0f) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, sizeof(y[i].qs));
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }

        const float iscale = -127.0f / max;
        const __m512 v_iscale = _mm512_set1_ps(iscale);

        for (int j = 0; j < QK_K; j += 16) {
            const __m512 xf = _mm512_loadu_ps(x + j);
            const __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(xf, v_iscale));
            const __m128i q8 = _mm512_cvtsepi32_epi8(q32);
            _mm_storeu_si128((__m128i *)(y[i].qs + j), q8);

            const int sum = _mm512_reduce_add_epi32(q32);
            y[i].bsums[j / 16] = (int16_t)sum;
        }

        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
#else
    quantize_row_q8_k_ref(x, vy, k);
#endif
}
