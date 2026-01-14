/**
 * @file gemm_kernels_q4k_q8k.c
 * @brief Q4_K (weights) x Q8_K (activations) kernels for inference
 *
 * Implements decode-style matvec/matmul where weights are Q4_K and the
 * activations are quantized on-the-fly to Q8_K. This is inference-only;
 * no backward pass is provided here.
 */

#include <assert.h>
#include <math.h>
#include <string.h>

#include "ckernel_quant.h"

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K);

void gemv_q4_k_q8_k_vnni(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K);

void gemv_q4_k_q8_k_avx(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

void gemv_q4_k_q8_k_sse(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

static inline int ck_nearest_int(float fval) {
    /* Bit-level round-to-nearest from llama.cpp (fast + deterministic). */
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void quantize_row_q8_k_ref(const float *x, void *vy, int k) {
    if (!x || !vy || k <= 0) {
        return;
    }
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    block_q8_K *y = (block_q8_K *)vy;

    for (int i = 0; i < nb; ++i) {
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, sizeof(y[i].qs));
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }

        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = ck_nearest_int(iscale * x[j]);
            if (v > 127) {
                v = 127;
            }
            if (v < -128) {
                v = -128;
            }
            y[i].qs[j] = (int8_t)v;
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            const int8_t *qs = &y[i].qs[j * 16];
            for (int ii = 0; ii < 16; ++ii) {
                sum += qs[ii];
            }
            y[i].bsums[j] = (int16_t)sum;
        }

        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}

void quantize_row_q8_k_sse(const float *x, void *vy, int k);

void quantize_row_q8_k(const float *x, void *vy, int k) {
#if defined(__SSE4_1__)
    quantize_row_q8_k_sse(x, vy, k);
#else
    quantize_row_q8_k_ref(x, vy, k);
#endif
}

static float dot_q4_k_q8_k_ref(const block_q4_K *w,
                               const block_q8_K *x,
                               int k)
{
    const int nb = k / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        uint8_t sc[8], m_val[8];
        unpack_q4_k_scales(w[i].scales, sc, m_val);

        const float d = CK_FP16_TO_FP32(w[i].d) * x[i].d;
        const float dmin = CK_FP16_TO_FP32(w[i].dmin) * x[i].d;

        /* Q4_K layout: process 64 elements at a time
         * - Low nibbles of qs[0..31] → elements 0..31 → uses sc[0], m[0]
         * - High nibbles of qs[0..31] → elements 32..63 → uses sc[1], m[1]
         * - Low nibbles of qs[32..63] → elements 64..95 → uses sc[2], m[2]
         * - etc.
         */
        int is = 0;
        int q_offset = 0;

        for (int j = 0; j < QK_K; j += 64) {
            const uint8_t *qs = &w[i].qs[q_offset];
            const int8_t *q8_lo = &x[i].qs[j];       /* Elements j to j+31 */
            const int8_t *q8_hi = &x[i].qs[j + 32];  /* Elements j+32 to j+63 */

            /* Sum for low nibbles (elements j to j+31) */
            int32_t sum_q4q8_lo = 0;
            for (int l = 0; l < 32; ++l) {
                int q4_val = qs[l] & 0x0F;
                sum_q4q8_lo += q4_val * q8_lo[l];
            }

            /* Sum for high nibbles (elements j+32 to j+63) */
            int32_t sum_q4q8_hi = 0;
            for (int l = 0; l < 32; ++l) {
                int q4_val = qs[l] >> 4;
                sum_q4q8_hi += q4_val * q8_hi[l];
            }

            /* bsums: each bsum is 16 elements */
            int32_t bsum_lo = (int32_t)x[i].bsums[j / 16] +
                              (int32_t)x[i].bsums[j / 16 + 1];
            int32_t bsum_hi = (int32_t)x[i].bsums[(j + 32) / 16] +
                              (int32_t)x[i].bsums[(j + 32) / 16 + 1];

            /* Accumulate: d * sc * sum(q4*q8) - dmin * m * sum(q8) */
            sumf += d * (float)sc[is] * (float)sum_q4q8_lo;
            sumf -= dmin * (float)m_val[is] * (float)bsum_lo;
            sumf += d * (float)sc[is + 1] * (float)sum_q4q8_hi;
            sumf -= dmin * (float)m_val[is + 1] * (float)bsum_hi;

            q_offset += 32;
            is += 2;
        }
    }

    return sumf;
}

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q4_k_q8_k_ref(w_row, x, K);
    }
}

void gemv_q4_k_q8_k_amx(float *y, const void *W, const void *x_q8, int M, int K);

void gemv_q4_k_q8_k(float *y,
                    const void *W,
                    const void *x_q8,
                    int M, int K)
{
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    /* AMX: Highest performance for large M (prefill) */
    gemv_q4_k_q8_k_amx(y, W, x_q8, M, K);
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
    /* VNNI: Best for decode (single token) */
    gemv_q4_k_q8_k_vnni(y, W, x_q8, M, K);
#elif defined(__AVX2__)
    gemv_q4_k_q8_k_avx2(y, W, x_q8, M, K);
#elif defined(__AVX__)
    /* AVX version uses maddubs_epi16 (more efficient than SSE) */
    gemv_q4_k_q8_k_avx(y, W, x_q8, M, K);
#elif defined(__SSE4_1__)
    gemv_q4_k_q8_k_sse(y, W, x_q8, M, K);
#else
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}

void gemm_q4_k_q8_k_ref(float *Y,
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
        gemv_q4_k_q8_k_ref(&Y[n * M], W, x_row, M, K);
    }
}

void gemm_q4_k_q8_k(float *Y,
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
        gemv_q4_k_q8_k(&Y[n * M], W, x_row, M, K);
    }
}

void gemm_nt_q4_k_q8_k(const void *A_q8,
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

    gemm_q4_k_q8_k(C, B, A_q8, /*M_out=*/N, /*N_batch=*/M, K);

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
