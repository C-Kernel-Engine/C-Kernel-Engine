/**
 * @file gemm_kernels_q8_0_q8_0_contract.c
 * @brief FP32 API adapters that enforce Q8_0 x Q8_0 activation contract
 */

#include "ckernel_engine.h"
#include "ckernel_quant.h"

#define CK_Q80_STACK_Q8_BLOCKS 1024

void gemv_q8_0_q8_0_contract(float *y,
                             const void *W,
                             const float *x,
                             int M,
                             int K)
{
    if (!y || !W || !x || M <= 0 || K <= 0) {
        return;
    }

    if ((K % QK8_0) != 0) {
        gemv_q8_0(y, W, x, M, K);
        return;
    }

    const int blocks_per_row = K / QK8_0;
    if (blocks_per_row > CK_Q80_STACK_Q8_BLOCKS) {
        gemv_q8_0(y, W, x, M, K);
        return;
    }

    block_q8_0 x_q8[CK_Q80_STACK_Q8_BLOCKS];
    quantize_row_q8_0(x, x_q8, K);
    gemv_q8_0_q8_0(y, W, x_q8, M, K);
}

void gemm_nt_q8_0_q8_0_contract(const float *A,
                                const void *B,
                                const float *bias,
                                float *C,
                                int M,
                                int N,
                                int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    for (int m = 0; m < M; ++m) {
        gemv_q8_0_q8_0_contract(&C[m * N], B, &A[m * K], N, K);
        if (bias) {
            for (int n = 0; n < N; ++n) {
                C[m * N + n] += bias[n];
            }
        }
    }
}
