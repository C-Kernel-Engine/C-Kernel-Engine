/**
 * @file gemm_kernels_q6k.c
 * @brief GEMM/GEMV kernels with Q6_K quantized weights
 *
 * Implements matrix multiplication where:
 *   - Activations (input): FP32
 *   - Weights: Q6_K (6-bit k-quant, int8 scales)
 *   - Output: FP32
 *
 * Reference implementation only (scalar) for correctness.
 */

#include <stdint.h>
#include <stddef.h>
#include "ckernel_quant.h"

/* ============================================================================
 * GEMV: y = W @ x  (W is Q6_K, x and y are FP32)
 * ============================================================================ */

static float dot_q6_k_ref(const block_q6_K *w,
                          const float *x,
                          int K)
{
    const int blocks_per_row = K / QK_K;
    double sum = 0.0;

    for (int b = 0; b < blocks_per_row; ++b) {
        const block_q6_K *block = &w[b];
        const float d = GGML_FP16_TO_FP32(block->d);

        const uint8_t *ql = block->ql;
        const uint8_t *qh = block->qh;
        const int8_t *sc = block->scales;
        const float *xp = x + (size_t)b * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                sum += (double)(d * (float)sc[is + 0] * (float)q1) * (double)xp[l + 0];
                sum += (double)(d * (float)sc[is + 2] * (float)q2) * (double)xp[l + 32];
                sum += (double)(d * (float)sc[is + 4] * (float)q3) * (double)xp[l + 64];
                sum += (double)(d * (float)sc[is + 6] * (float)q4) * (double)xp[l + 96];
            }
            xp += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    return (float)sum;
}

void gemv_q6_k(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
    if (!y || !W || !x) {
        return;
    }
    if (M <= 0 || K <= 0) {
        return;
    }

    const block_q6_K *blocks = (const block_q6_K *)W;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q6_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q6_k_ref(w_row, x, K);
    }
}

void gemm_q6_k(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K)
{
    if (!Y || !W || !X) {
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    for (int n = 0; n < N; ++n) {
        gemv_q6_k(&Y[n * M], W, &X[n * K], M, K);
    }
}

void gemm_nt_q6_k(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
    if (!A || !B || !C) {
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    /* gemm_q6_k produces Y as [batch x M_out] where:
     *   batch = M (tokens)
     *   M_out = N (output channels) */
    gemm_q6_k(C, B, A, /*M_out=*/N, /*N_batch=*/M, K);

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
