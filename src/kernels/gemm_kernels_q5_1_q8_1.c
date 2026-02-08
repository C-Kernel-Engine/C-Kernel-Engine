/**
 * @file gemm_kernels_q5_1_q8_1.c
 * @brief Q5_1 x Q8_1 contract kernels used for ggml parity (Gemma-sensitive path)
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "ckernel_quant.h"

#define CK_Q51_STACK_Q8_BLOCKS 256

/* Q8_1 is used only by this contract kernel path. Keep a local definition so
 * this file does not depend on global header churn.
 * Layout matches ggml: fp16 d, fp16 s, 32 int8 quants. */
#ifndef QK8_1
#define QK8_1 32
#endif

typedef struct {
    ck_half d;
    ck_half s;
    int8_t qs[QK8_1];
} block_q8_1;

/* Quantize one FP32 row to Q8_1 blocks (ggml-compatible scalar path). */
static void quantize_row_q8_1_scalar(const float *x, block_q8_1 *y, int k) {
    const int nb = k / QK8_1;
    for (int b = 0; b < nb; ++b) {
        const float *xb = x + (size_t)b * QK8_1;
        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            float av = xb[j] >= 0.0f ? xb[j] : -xb[j];
            if (av > amax) amax = av;
        }

        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        y[b].d = CK_FP32_TO_FP16(d);

        int sum = 0;
        for (int j = 0; j < QK8_1; ++j) {
            int q = (int)roundf(xb[j] * id);
            y[b].qs[j] = (int8_t)q;
            sum += q;
        }
        y[b].s = CK_FP32_TO_FP16((float)sum * d);
    }
}

/* One 32-element block dot: Q5_1(weights) x Q8_1(activations). */
static float dot_q5_1_q8_1_block(const block_q5_1 *w, const block_q8_1 *x) {
    uint32_t qh;
    memcpy(&qh, w->qh, sizeof(qh));

    int sumi0 = 0;
    int sumi1 = 0;
    for (int j = 0; j < QK5_1 / 2; ++j) {
        const uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
        const uint8_t xh1 = (uint8_t)(((qh >> (j + 12))     ) & 0x10);
        const int32_t q0 = (int32_t)((w->qs[j] & 0x0F) | xh0);
        const int32_t q1 = (int32_t)((w->qs[j] >> 4) | xh1);
        sumi0 += q0 * (int32_t)x->qs[j];
        sumi1 += q1 * (int32_t)x->qs[j + QK5_1 / 2];
    }

    const float wd = CK_FP16_TO_FP32(w->d);
    const float wm = CK_FP16_TO_FP32(w->m);
    const float xd = CK_FP16_TO_FP32(x->d);
    const float xs = CK_FP16_TO_FP32(x->s);
    return (wd * xd) * (float)(sumi0 + sumi1) + wm * xs;
}

void gemv_q5_1_q8_1_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0 || (K % QK5_1) != 0) {
        return;
    }

    const block_q5_1 *blocks = (const block_q5_1 *)W;
    const block_q8_1 *x = (const block_q8_1 *)x_q8;
    const int blocks_per_row = K / QK5_1;

    for (int row = 0; row < M; ++row) {
        const block_q5_1 *w_row = &blocks[row * blocks_per_row];
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            sum += dot_q5_1_q8_1_block(&w_row[b], &x[b]);
        }
        y[row] = sum;
    }
}

void gemm_nt_q5_1_q8_1_ref(const void *A_q8,
                           const void *B,
                           const float *bias,
                           float *C,
                           int M, int N, int K)
{
    if (!A_q8 || !B || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK5_1) != 0) {
        return;
    }

    const block_q8_1 *A = (const block_q8_1 *)A_q8;
    const block_q5_1 *W = (const block_q5_1 *)B;
    const int blocks_per_row = K / QK5_1;

    for (int m = 0; m < M; ++m) {
        const block_q8_1 *a_row = &A[m * blocks_per_row];
        for (int n = 0; n < N; ++n) {
            const block_q5_1 *w_row = &W[n * blocks_per_row];
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q5_1_q8_1_block(&w_row[b], &a_row[b]);
            }
            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

void gemv_q5_1_q8_1(float *y,
                    const void *W,
                    const float *x,
                    int M,
                    int K)
{
    if (!y || !W || !x || M <= 0 || K <= 0 || (K % QK5_1) != 0) {
        return;
    }

    const int blocks_per_row = K / QK5_1;
    if (blocks_per_row > CK_Q51_STACK_Q8_BLOCKS) {
        return;
    }

    block_q8_1 x_q8[CK_Q51_STACK_Q8_BLOCKS];
    quantize_row_q8_1_scalar(x, x_q8, K);
    gemv_q5_1_q8_1_ref(y, W, x_q8, M, K);
}

void gemm_nt_q5_1_q8_1(const float *A,
                       const void *B,
                       const float *bias,
                       float *C,
                       int M,
                       int N,
                       int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK5_1) != 0) {
        return;
    }

    const int blocks_per_row = K / QK5_1;
    if (blocks_per_row > CK_Q51_STACK_Q8_BLOCKS) {
        return;
    }

    for (int m = 0; m < M; ++m) {
        block_q8_1 a_q8[CK_Q51_STACK_Q8_BLOCKS];
        quantize_row_q8_1_scalar(&A[m * K], a_q8, K);
        gemm_nt_q5_1_q8_1_ref(a_q8, B, bias, &C[m * N], 1, N, K);
    }
}
