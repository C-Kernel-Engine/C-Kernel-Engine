#include <assert.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

#include "ckernel_quant.h"

static inline int ck_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void quantize_row_q8_k_sse(const float *x, void *vy, int k) {
    if (!x || !vy || k <= 0) {
        return;
    }
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    block_q8_K *y = (block_q8_K *)vy;

    for (int i = 0; i < nb; ++i) {
        float max = 0.0f;
        
        // SSE max absolute value
        __m128 v_max = _mm_setzero_ps();
        for (int j = 0; j < QK_K; j += 4) {
            __m128 v = _mm_loadu_ps(x + j);
            __m128 v_abs = _mm_andnot_ps(_mm_set1_ps(-0.0f), v);
            v_max = _mm_max_ps(v_max, v_abs);
        }
        
        // Horizontal max
        v_max = _mm_max_ps(v_max, _mm_shuffle_ps(v_max, v_max, _MM_SHUFFLE(1, 0, 3, 2)));
        v_max = _mm_max_ps(v_max, _mm_shuffle_ps(v_max, v_max, _MM_SHUFFLE(0, 1, 0, 1)));
        _mm_store_ss(&max, v_max);

        if (max == 0.0f) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, sizeof(y[i].qs));
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }

        const float iscale = -127.0f / max;
        __m128 v_iscale = _mm_set1_ps(iscale);
        
        // Quantize and compute bsums in SSE
        for (int j = 0; j < QK_K; j += 16) {
            __m128 x0 = _mm_loadu_ps(x + j + 0);
            __m128 x1 = _mm_loadu_ps(x + j + 4);
            __m128 x2 = _mm_loadu_ps(x + j + 8);
            __m128 x3 = _mm_loadu_ps(x + j + 12);

            __m128i q0 = _mm_cvtps_epi32(_mm_mul_ps(x0, v_iscale));
            __m128i q1 = _mm_cvtps_epi32(_mm_mul_ps(x1, v_iscale));
            __m128i q2 = _mm_cvtps_epi32(_mm_mul_ps(x2, v_iscale));
            __m128i q3 = _mm_cvtps_epi32(_mm_mul_ps(x3, v_iscale));

            // Pack i32 -> i16 -> i8
            __m128i q01 = _mm_packs_epi32(q0, q1);
            __m128i q23 = _mm_packs_epi32(q2, q3);
            __m128i q0123 = _mm_packs_epi16(q01, q23);

            _mm_storeu_si128((__m128i *)(y[i].qs + j), q0123);

            // Compute bsum for these 16 elements
            // Each bsum[j/16] covers 16 elements
            __m128i p01 = _mm_add_epi16(q01, q23);
            p01 = _mm_add_epi16(p01, _mm_shuffle_epi32(p01, _MM_SHUFFLE(1, 0, 3, 2)));
            p01 = _mm_add_epi16(p01, _mm_shufflelo_epi16(p01, _MM_SHUFFLE(1, 0, 3, 2)));
            int16_t bsum = (int16_t)_mm_extract_epi16(p01, 0) + (int16_t)_mm_extract_epi16(p01, 1);
            y[i].bsums[j / 16] = bsum;
        }

        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}
