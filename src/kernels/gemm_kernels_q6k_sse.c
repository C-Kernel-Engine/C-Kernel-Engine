#pragma GCC target("sse4.1,ssse3")
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

// Forward declarations
void quantize_row_q8_k(const float *x, void *vy, int k);

/**
 * SSE Optimized dot product for Q6_K x Q8_K
 * Q6_K layout: 
 *   ql: 128 bytes (low 4 bits)
 *   qh: 64 bytes (high 2 bits)
 *   scales: 16 bytes (int8 scales)
 *   d: fp16 super-scale
 */
static inline float dot_q6_k_q8_k_256_sse(const block_q6_K *bw, const block_q8_K *ba) {
    const uint8_t *ql = bw->ql;
    const uint8_t *qh = bw->qh;
    const int8_t  *sc = bw->scales;
    const int8_t  *qa = ba->qs;

    double sum = 0.0;
    float d = CK_FP16_TO_FP32(bw->d);

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            // Unpack 4 weights at a time to match scalar reference logic
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            sum += (double)(d * (float)sc[is + 0] * (float)q1) * (double)qa[l + 0];
            sum += (double)(d * (float)sc[is + 2] * (float)q2) * (double)qa[l + 32];
            sum += (double)(d * (float)sc[is + 4] * (float)q3) * (double)qa[l + 64];
            sum += (double)(d * (float)sc[is + 6] * (float)q4) * (double)qa[l + 96];
        }
        qa += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }

    return (float)(sum * ba->d);
}

// Fallback to ref if K not aligned
void gemm_nt_q6_k_sse(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    if (K % QK_K != 0) {
        gemm_nt_q6_k_ref(A, B, bias, C, M, N, K);
        return;
    }

    size_t q8_size = (K / QK_K) * sizeof(block_q8_K);
    block_q8_K *A_q8 = (block_q8_K *)alloca(q8_size);

    const block_q6_K *weights = (const block_q6_K *)B;
    const int blocks_per_row = K / QK_K;

    for (int m = 0; m < M; m++) {
        quantize_row_q8_k(&A[m * K], A_q8, K);

        for (int n = 0; n < N; n++) {
            float sumf = 0.0f;
            const block_q6_K *w_row = weights + n * blocks_per_row;

            for (int b = 0; b < blocks_per_row; b++) {
                sumf += dot_q6_k_q8_k_256_sse(&w_row[b], &A_q8[b]);
            }

            C[m * N + n] = sumf + (bias ? bias[n] : 0.0f);
        }
    }
}
