#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

// SSE4.1 implementation of Q4_K * Q8_K dot product
// Compatible with standard AVX (Sandy Bridge/Ivy Bridge)

static inline int32_t hsum_epi32_sse(__m128i v) {
    __m128i shuf = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sums = _mm_add_epi32(v, shuf);
    shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
    sums = _mm_add_epi32(sums, shuf);
    return _mm_cvtsi128_si32(sums);
}

void gemv_q4_k_q8_k_sse(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    const __m128i mask_low = _mm_set1_epi8(0x0F);

    for (int row = 0; row < M; ++row) {
        float sumf = 0.0f;
        const block_q4_K *w_row = blocks + row * blocks_per_row;

        for (int i = 0; i < blocks_per_row; ++i) {
            const block_q4_K *b4 = &w_row[i];
            const block_q8_K *b8 = &x[i];

            // Unpack scales (same as ref)
            uint8_t sc[8], m_val[8];
            unpack_q4_k_scales(b4->scales, sc, m_val);

            float d = CK_FP16_TO_FP32(b4->d) * b8->d;
            float dmin = CK_FP16_TO_FP32(b4->dmin) * b8->d;

            int is = 0;
            int q_offset = 0;

            // Process 4 chunks of 64 elements (256 total)
            for (int j = 0; j < QK_K; j += 64) {
                // We process 32 bytes of qs (covering 64 elements via low/high nibbles)
                // We access qs[0..31] relative to q_offset

                // Accumulators for this 64-element chunk
                __m128i acc_lo = _mm_setzero_si128();
                __m128i acc_hi = _mm_setzero_si128();

                // Inner loop: 2 iters of 16 bytes (32 elements)
                for (int l = 0; l < 32; l += 16) {
                    // Load 16 bytes of Q4
                    __m128i q4_vec = _mm_loadu_si128((const __m128i *)(b4->qs + q_offset + l));

                    // Low nibbles -> correspond to q8_lo (elements j+l .. j+l+15)
                    __m128i q4_lo = _mm_and_si128(q4_vec, mask_low);
                    
                    // High nibbles -> correspond to q8_hi (elements j+32+l .. j+32+l+15)
                    __m128i q4_hi = _mm_and_si128(_mm_srli_epi16(q4_vec, 4), mask_low);

                    // Load Q8
                    __m128i q8_lo_vec = _mm_loadu_si128((const __m128i *)(b8->qs + j + l));
                    __m128i q8_hi_vec = _mm_loadu_si128((const __m128i *)(b8->qs + j + 32 + l));

                    // Expand and Multiply-Add: Q4(u8) * Q8(s8) -> i32
                    // Since Q4 is u8 and Q8 is s8, we use intermediate i16
                    
                    // LO PART
                    __m128i q4_lo_16_L = _mm_cvtepu8_epi16(q4_lo); // lower 8 -> 16
                    __m128i q8_lo_16_L = _mm_cvtepi8_epi16(q8_lo_vec);
                    __m128i prod_lo_L = _mm_madd_epi16(q4_lo_16_L, q8_lo_16_L); // i32
                    acc_lo = _mm_add_epi32(acc_lo, prod_lo_L);

                    __m128i q4_lo_16_H = _mm_cvtepu8_epi16(_mm_srli_si128(q4_lo, 8)); // upper 8 -> 16
                    __m128i q8_lo_16_H = _mm_cvtepi8_epi16(_mm_srli_si128(q8_lo_vec, 8));
                    __m128i prod_lo_H = _mm_madd_epi16(q4_lo_16_H, q8_lo_16_H); // i32
                    acc_lo = _mm_add_epi32(acc_lo, prod_lo_H);

                    // HI PART
                    __m128i q4_hi_16_L = _mm_cvtepu8_epi16(q4_hi);
                    __m128i q8_hi_16_L = _mm_cvtepi8_epi16(q8_hi_vec);
                    __m128i prod_hi_L = _mm_madd_epi16(q4_hi_16_L, q8_hi_16_L);
                    acc_hi = _mm_add_epi32(acc_hi, prod_hi_L);

                    __m128i q4_hi_16_H = _mm_cvtepu8_epi16(_mm_srli_si128(q4_hi, 8));
                    __m128i q8_hi_16_H = _mm_cvtepi8_epi16(_mm_srli_si128(q8_hi_vec, 8));
                    __m128i prod_hi_H = _mm_madd_epi16(q4_hi_16_H, q8_hi_16_H);
                    acc_hi = _mm_add_epi32(acc_hi, prod_hi_H);
                }

                int32_t sum_q4q8_lo = hsum_epi32_sse(acc_lo);
                int32_t sum_q4q8_hi = hsum_epi32_sse(acc_hi);

                /* bsums: each bsum is 16 elements */
                int32_t bsum_lo = (int32_t)b8->bsums[j / 16] +
                                  (int32_t)b8->bsums[j / 16 + 1];
                int32_t bsum_hi = (int32_t)b8->bsums[(j + 32) / 16] +
                                  (int32_t)b8->bsums[(j + 32) / 16 + 1];

                sumf += d * (float)sc[is] * (float)sum_q4q8_lo;
                sumf -= dmin * (float)m_val[is] * (float)bsum_lo;
                sumf += d * (float)sc[is + 1] * (float)sum_q4q8_hi;
                sumf -= dmin * (float)m_val[is + 1] * (float)bsum_hi;

                q_offset += 32;
                is += 2;
            }
        }
        y[row] = sumf;
    }
}
