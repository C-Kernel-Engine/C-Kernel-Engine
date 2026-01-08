#pragma GCC target("avx,sse4.1")
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

// AVX1 horizontal sum helper
static inline float hsum256_ps_fused(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// AVX1 horizontal max helper
static inline float hmax256_ps_fused(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtss_f32(max128);
}

/**
 * Fused RMSNorm + Q8_K Quantization
 * 
 * Benefits:
 * 1. Single pass over input data (reduces DRAM pressure)
 * 2. Normalization results stay in registers for quantization
 * 3. Keeps hot data in L1/L2 cache
 */
void rmsnorm_q8_k_fused(const float *input,
                        const float *gamma,
                        void *vy,
                        int tokens,
                        int d_model,
                        int aligned_embed_dim,
                        float eps)
{
    const int T = tokens;
    const int D = d_model;
    block_q8_K *y = (block_q8_K *)vy;

    for (int t = 0; t < T; ++t) {
        const float *x = input + (size_t)t * aligned_embed_dim;
        
        // 1. Compute sum of squares using AVX
        __m256 sum_sq_vec = _mm256_setzero_ps();
        for (int d = 0; d < D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, _mm256_mul_ps(xv, xv));
        }
        float sum_sq = hsum256_ps_fused(sum_sq_vec);
        float rstd = 1.0f / sqrtf(sum_sq / (float)D + eps);
        __m256 vrstd = _mm256_set1_ps(rstd);

        // 2. We need the max absolute value of the NORMALIZED data for quantization
        // y_i = gamma_i * (x_i * rstd)
        // We do this in blocks of QK_K (256) to match Q8_K layout
        for (int b = 0; b < D / QK_K; ++b) {
            const float *xb = x + b * QK_K;
            const float *gb = gamma + b * QK_K;
            block_q8_K *out_block = &y[t * (D / QK_K) + b];

            // Local normalization and max search
            __m256 v_max_abs = _mm256_setzero_ps();
            float norm_buf[QK_K];

            for (int d = 0; d < QK_K; d += 8) {
                __m256 xv = _mm256_loadu_ps(&xb[d]);
                __m256 gv = _mm256_loadu_ps(&gb[d]);
                __m256 normalized = _mm256_mul_ps(_mm256_mul_ps(xv, vrstd), gv);
                
                _mm256_storeu_ps(&norm_buf[d], normalized);
                
                __m256 v_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), normalized);
                v_max_abs = _mm256_max_ps(v_max_abs, v_abs);
            }

            float max_val = hmax256_ps_fused(v_max_abs);
            if (max_val == 0.0f) {
                out_block->d = 0.0f;
                memset(out_block->qs, 0, QK_K);
                memset(out_block->bsums, 0, sizeof(out_block->bsums));
                continue;
            }

            // 3. Quantize to Q8_K
            float iscale = -127.0f / max_val;
            __m256 v_iscale = _mm256_set1_ps(iscale);
            out_block->d = 1.0f / iscale;

            for (int j = 0; j < QK_K; j += 16) {
                // AVX1 doesn't have 256-bit integer conversion, so we use 128-bit SSE for packing
                __m128 n0 = _mm_loadu_ps(&norm_buf[j + 0]);
                __m128 n1 = _mm_loadu_ps(&norm_buf[j + 4]);
                __m128 n2 = _mm_loadu_ps(&norm_buf[j + 8]);
                __m128 n3 = _mm_loadu_ps(&norm_buf[j + 12]);

                __m128i q0 = _mm_cvtps_epi32(_mm_mul_ps(n0, _mm256_castps256_ps128(v_iscale)));
                __m128i q1 = _mm_cvtps_epi32(_mm_mul_ps(n1, _mm256_castps256_ps128(v_iscale)));
                __m128i q2 = _mm_cvtps_epi32(_mm_mul_ps(n2, _mm256_castps256_ps128(v_iscale)));
                __m128i q3 = _mm_cvtps_epi32(_mm_mul_ps(n3, _mm256_castps256_ps128(v_iscale)));

                __m128i q01 = _mm_packs_epi32(q0, q1);
                __m128i q23 = _mm_packs_epi32(q2, q3);
                __m128i q0123 = _mm_packs_epi16(q01, q23);

                _mm_storeu_si128((__m128i *)(out_block->qs + j), q0123);

                // Compute bsum for 16 elements
                __m128i p01 = _mm_add_epi16(q01, q23);
                p01 = _mm_add_epi16(p01, _mm_shuffle_epi32(p01, _MM_SHUFFLE(1, 0, 3, 2)));
                p01 = _mm_add_epi16(p01, _mm_shufflelo_epi16(p01, _MM_SHUFFLE(1, 0, 3, 2)));
                int16_t bsum = (int16_t)_mm_extract_epi16(p01, 0) + (int16_t)_mm_extract_epi16(p01, 1);
                out_block->bsums[j / 16] = bsum;
            }
        }
    }
}
