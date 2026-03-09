/**
 * @file quantize_row_q8_k_sse.c
 * @brief SSE-optimized Q8_K row quantization kernel
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 */

#include <assert.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

#include "ckernel_quant.h"

void quantize_row_q8_k_sse(const float *x, void *vy, int k) {
    if (!x || !vy || k <= 0) {
        return;
    }
    assert(k % QK_K == 0);

    const int nb = k / QK_K;
    block_q8_K *y = (block_q8_K *)vy;

    for (int i = 0; i < nb; ++i) {
        /* Keep the exact signed-max selection contract from llama.cpp/ref. */
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
        const __m128 v_iscale = _mm_set1_ps(iscale);

        for (int j = 0; j < QK_K; j += 16) {
            const __m128 x0 = _mm_loadu_ps(x + j + 0);
            const __m128 x1 = _mm_loadu_ps(x + j + 4);
            const __m128 x2 = _mm_loadu_ps(x + j + 8);
            const __m128 x3 = _mm_loadu_ps(x + j + 12);

            const __m128i q0 = _mm_cvtps_epi32(_mm_mul_ps(x0, v_iscale));
            const __m128i q1 = _mm_cvtps_epi32(_mm_mul_ps(x1, v_iscale));
            const __m128i q2 = _mm_cvtps_epi32(_mm_mul_ps(x2, v_iscale));
            const __m128i q3 = _mm_cvtps_epi32(_mm_mul_ps(x3, v_iscale));

            const __m128i q01 = _mm_packs_epi32(q0, q1);
            const __m128i q23 = _mm_packs_epi32(q2, q3);
            const __m128i q0123 = _mm_packs_epi16(q01, q23);

            _mm_storeu_si128((__m128i *)(y[i].qs + j), q0123);

            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j + ii];
            }
            y[i].bsums[j / 16] = (int16_t)sum;
        }

        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}
