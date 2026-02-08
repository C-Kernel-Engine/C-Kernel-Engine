/**
 * @file gemm_kernels_q5_k.c
 * @brief GEMM/GEMV kernels with Q5_K quantized weights
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * Implements matrix multiplication where:
 *   - Activations (input): FP32 (quantized internally to Q8_K for dot path)
 *   - Weights: Q5_K (5-bit super-block quant)
 *   - Output: FP32
 *
 * Q5_K Format (256 weights per super-block):
 *   - d: FP16 super-block scale
 *   - dmin: FP16 super-block minimum
 *   - scales[12]: 8 sub-block scales + 8 sub-block mins (6 bits each, packed)
 *   - qh[32]: high bits for 256 weights (1 bit each)
 *   - qs[128]: low 4 bits for 256 weights (4 bits each)
 *
 * Total: 2 + 2 + 12 + 32 + 128 = 176 bytes per 256 weights = 5.5 bits/weight
 *
 * Dequantization formula (matches llama.cpp):
 *   w = d * scale * q - dmin * mins
 *   where q = qs_val | (qh_bit << 4) = 5-bit value [0, 31]
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "ckernel_quant.h"

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__)
#include <immintrin.h>
#endif

/* Q5_K constants */
#define QK_K 256
#define CK_Q5K_STACK_Q8_BLOCKS 128

/* Q5_K block definition is required by this kernel file.
 * Keep a local ggml-compatible layout to decouple from shared headers. */
typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} block_q5_K;

/* Unpack 8 per-subblock scales and mins from packed Q5_K scale bytes.
 * This mirrors the packing contract used by llama.cpp. */
static inline void unpack_q5_k_scales(const uint8_t *scales,
                                      uint8_t *sc,
                                      uint8_t *m) {
    sc[0] = scales[0] & 0x3F;
    sc[1] = scales[1] & 0x3F;
    sc[2] = scales[2] & 0x3F;
    sc[3] = scales[3] & 0x3F;

    m[0] = scales[4] & 0x3F;
    m[1] = scales[5] & 0x3F;
    m[2] = scales[6] & 0x3F;
    m[3] = scales[7] & 0x3F;

    sc[4] = (scales[8]  & 0x0F) | ((scales[0] >> 6) << 4);
    sc[5] = (scales[9]  & 0x0F) | ((scales[1] >> 6) << 4);
    sc[6] = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4);
    sc[7] = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4);

    m[4] = (scales[8]  >> 4) | ((scales[4] >> 6) << 4);
    m[5] = (scales[9]  >> 4) | ((scales[5] >> 6) << 4);
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4);
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4);
}

/* quantize_row_q8_k() is implemented in gemm_kernels_q4k_q8k.c */
void quantize_row_q8_k(const float *x, void *vy, int k);

static inline int ck_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static void quantize_row_q8_k_scalar(const float *x, block_q8_K *y) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        float ax = x[j] < 0.0f ? -x[j] : x[j];
        if (ax > amax) {
            amax = ax;
            max = x[j];
        }
    }
    if (amax == 0.0f) {
        y->d = 0.0f;
        memset(y->qs, 0, sizeof(y->qs));
        memset(y->bsums, 0, sizeof(y->bsums));
        return;
    }

    const float iscale = -127.0f / max;
    for (int j = 0; j < QK_K; ++j) {
        int q = ck_nearest_int(iscale * x[j]);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        y->qs[j] = (int8_t)q;
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        int sum = 0;
        const int8_t *q = &y->qs[j * 16];
        for (int l = 0; l < 16; ++l) {
            sum += q[l];
        }
        y->bsums[j] = (int16_t)sum;
    }
    y->d = 1.0f / iscale;
}

/* Llama-compatible dot path: Q5_K weights x Q8_K activations for one 256-value block. */
static float dot_q5_k_q8_k_block(const block_q5_K *w, const block_q8_K *x) {
    static const uint32_t kmask1 = 0x3f3f3f3fU;
    static const uint32_t kmask2 = 0x0f0f0f0fU;
    static const uint32_t kmask3 = 0x03030303U;

    uint32_t utmp[4] = {0, 0, 0, 0};
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[2];

    int8_t aux8[QK_K];
    int16_t aux16[8];
    int32_t aux32[8];
    memset(aux32, 0, sizeof(aux32));

    const uint8_t *q4 = w->qs;
    const uint8_t *hm = w->qh;
    int8_t *a = aux8;
    uint8_t m = 1;

    for (int j = 0; j < QK_K / 64; ++j) {
        for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
        for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
        a += 32;
        m <<= 1;

        for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
        for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
        a += 32;
        m <<= 1;

        q4 += 32;
    }

    memcpy(utmp, w->scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    {
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
    }
    utmp[0] &= kmask1;

    int sumi = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
        sumi += (int)x->bsums[j] * (int)mins[j / 2];
    }

    a = aux8;
    const int8_t *q8 = x->qs;
    int is = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
        const int32_t scale = (int32_t)scales[is++];

        for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;

        for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;

        for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;

        for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;
    }

    int32_t doti = 0;
    for (int l = 0; l < 8; ++l) {
        doti += aux32[l];
    }

    const float d = CK_FP16_TO_FP32(w->d) * x->d;
    const float dmin = CK_FP16_TO_FP32(w->dmin) * x->d;
    return d * (float)doti - dmin * (float)sumi;
}

/* FP32 fallback for oversized K (very rare for current models). */
static void gemv_q5_k_ref_fp32(float *y, const void *W, const float *x, int M, int K)
{
    const block_q5_K *blocks = (const block_q5_K *)W;
    const int blocks_per_row = K / QK_K;

    for (int m = 0; m < M; m++) {
        const float *x_row = x;
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_K *block = &blocks[m * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float dmin = CK_FP16_TO_FP32(block->dmin);
            const uint8_t *qh = block->qh;
            const uint8_t *qs = block->qs;
            uint8_t sc_arr[8], m_arr[8];
            unpack_q5_k_scales(block->scales, sc_arr, m_arr);

            for (int sb = 0; sb < 8; sb++) {
                const float d_sub = d * (float)sc_arr[sb];
                const float m_sub = dmin * (float)m_arr[sb];
                const int qs_offset = sb * 16;
                const int qh_offset = sb * 4;

                for (int i = 0; i < 32; i++) {
                    const uint8_t qs_val = (uint8_t)((qs[qs_offset + i / 2] >> (4 * (i % 2))) & 0xF);
                    const uint8_t qh_bit = (uint8_t)((qh[qh_offset + i / 8] >> (i % 8)) & 1);
                    const uint8_t q = (uint8_t)(qs_val | (qh_bit << 4));
                    sum += (d_sub * (float)q - m_sub) * x_row[b * QK_K + sb * 32 + i];
                }
            }
        }

        y[m] = sum;
    }
}

static void gemm_nt_q5_k_ref_fp32(const float *A,
                                  const void *B,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    const block_q5_K *blocks = (const block_q5_K *)B;
    const int blocks_per_col = K / QK_K;

    for (int m = 0; m < M; m++) {
        const float *a_row = &A[m * K];

        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            const block_q5_K *w_row = &blocks[n * blocks_per_col];
            for (int b = 0; b < blocks_per_col; b++) {
                const block_q5_K *block = &w_row[b];
                const float d = CK_FP16_TO_FP32(block->d);
                const float dmin = CK_FP16_TO_FP32(block->dmin);
                uint8_t sc_arr[8], m_arr[8];
                unpack_q5_k_scales(block->scales, sc_arr, m_arr);

                for (int sb = 0; sb < 8; sb++) {
                    const float d_sub = d * (float)sc_arr[sb];
                    const float m_sub = dmin * (float)m_arr[sb];
                    const int qs_offset = sb * 16;
                    const int qh_offset = sb * 4;

                    for (int i = 0; i < 32; i++) {
                        const uint8_t qs_val = (uint8_t)((block->qs[qs_offset + i / 2] >> (4 * (i % 2))) & 0xF);
                        const uint8_t qh_bit = (uint8_t)((block->qh[qh_offset + i / 8] >> (i % 8)) & 1);
                        const uint8_t q = (uint8_t)(qs_val | (qh_bit << 4));
                        sum += (d_sub * (float)q - m_sub) * a_row[b * QK_K + sb * 32 + i];
                    }
                }
            }

            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

/* ============================================================================
 * Q5_K x Q8_K Kernels (explicit contract)
 *
 * WHY THESE EXIST:
 *   llama.cpp's Q5_K matmul contract is "Q5_K weights x Q8_K activations".
 *   The activation quantization is part of the numerical contract, not just
 *   an optimization. If we accidentally do FP32 activation dot here, we can
 *   get large parity drift at attn_proj/mlp_down while tests still pass if
 *   they compare against FP32-dequant references.
 *
 *   These entry points make the contract explicit in code:
 *     - gemv_q5_k_q8_k(): decode-style matrix-vector (single token)
 *     - gemm_nt_q5_k_q8_k(): prefill-style matrix-matrix
 * ============================================================================ */

void gemv_q5_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }
    if (K % QK_K != 0) {
        return;
    }

    const block_q5_K *blocks = (const block_q5_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int m = 0; m < M; ++m) {
        const block_q5_K *w_row = &blocks[m * blocks_per_row];
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            sum += dot_q5_k_q8_k_block(&w_row[b], &x[b]);
        }
        y[m] = sum;
    }
}

void gemm_nt_q5_k_q8_k_ref(const void *A_q8,
                           const void *B,
                           const float *bias,
                           float *C,
                           int M, int N, int K)
{
    if (!A_q8 || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }
    if (K % QK_K != 0) {
        return;
    }

    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q5_K *W = (const block_q5_K *)B;
    const int blocks_per_row = K / QK_K;

    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = &A[m * blocks_per_row];
        for (int n = 0; n < N; ++n) {
            const block_q5_K *w_row = &W[n * blocks_per_row];
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q5_k_q8_k_block(&w_row[b], &a_row[b]);
            }
            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

/* ============================================================================
 * FP32 adapter path (keeps existing call sites stable)
 *
 * Existing generated code and orchestration call gemv_q5_k/gemm_nt_q5_k with
 * FP32 activations. These adapter functions quantize activations to Q8_K and
 * then call the explicit Q5_K x Q8_K kernels above.
 * ============================================================================ */

void gemv_q5_k_ref(float *y, const void *W, const float *x, int M, int K)
{
    if (!y || !W || !x || M <= 0 || K <= 0) {
        return;
    }
    if (K % QK_K != 0) {
        gemv_q5_k_ref_fp32(y, W, x, M, K);
        return;
    }

    const block_q5_K *blocks = (const block_q5_K *)W;
    const int blocks_per_row = K / QK_K;
    if (blocks_per_row > CK_Q5K_STACK_Q8_BLOCKS) {
        gemv_q5_k_ref_fp32(y, W, x, M, K);
        return;
    }

    block_q8_K x_q8[CK_Q5K_STACK_Q8_BLOCKS];
    for (int b = 0; b < blocks_per_row; ++b) {
        quantize_row_q8_k_scalar(x + b * QK_K, &x_q8[b]);
    }
    gemv_q5_k_q8_k_ref(y, blocks, x_q8, M, K);
}

/* ============================================================================
 * GEMM NT Reference: C = A @ B^T + bias
 *   - A: FP32 activation matrix [M, K] (quantized internally to Q8_K per row)
 *   - B: Q5_K weight matrix [N, K] (stored transposed, accessed as [N, K])
 *   - bias: Optional FP32 bias [N]
 *   - C: FP32 output matrix [M, N]
 * ============================================================================ */

void gemm_nt_q5_k_ref(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }
    if (K % QK_K != 0) {
        gemm_nt_q5_k_ref_fp32(A, B, bias, C, M, N, K);
        return;
    }

    const block_q5_K *blocks = (const block_q5_K *)B;
    const int blocks_per_col = K / QK_K;
    if (blocks_per_col > CK_Q5K_STACK_Q8_BLOCKS) {
        gemm_nt_q5_k_ref_fp32(A, B, bias, C, M, N, K);
        return;
    }

    for (int m = 0; m < M; ++m) {
        const float *a_row = &A[m * K];
        block_q8_K a_q8[CK_Q5K_STACK_Q8_BLOCKS];
        for (int b = 0; b < blocks_per_col; ++b) {
            quantize_row_q8_k_scalar(a_row + b * QK_K, &a_q8[b]);
        }
        gemm_nt_q5_k_q8_k_ref(a_q8, blocks, bias, &C[m * N], 1, N, K);
    }
}

/* ============================================================================
 * Dispatch wrappers - select best available implementation
 * ============================================================================ */

void gemv_q5_k_q8_k(float *y,
                    const void *W,
                    const void *x_q8,
                    int M, int K)
{
#if defined(__AVX512F__)
    /* TODO: AVX-512 implementation */
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#elif defined(__AVX2__)
    /* TODO: AVX-2 implementation */
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#elif defined(__AVX__)
    /* TODO: AVX implementation */
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#elif defined(__SSE4_1__)
    /* TODO: SSE4.1 implementation */
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#else
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}

void gemm_nt_q5_k_q8_k(const void *A_q8,
                       const void *B,
                       const float *bias,
                       float *C,
                       int M, int N, int K)
{
#if defined(__AVX512F__)
    /* TODO: AVX-512 implementation */
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#elif defined(__AVX2__)
    /* TODO: AVX-2 implementation */
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#elif defined(__AVX__)
    /* TODO: AVX implementation */
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#elif defined(__SSE4_1__)
    /* TODO: SSE4.1 implementation */
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#else
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#endif
}

void gemv_q5_k(float *y, const void *W, const float *x, int M, int K)
{
#if defined(__AVX512F__)
    /* TODO: AVX-512 implementation */
    gemv_q5_k_ref(y, W, x, M, K);
#elif defined(__AVX2__)
    /* TODO: AVX-2 implementation */
    gemv_q5_k_ref(y, W, x, M, K);
#elif defined(__AVX__)
    /* TODO: AVX implementation */
    gemv_q5_k_ref(y, W, x, M, K);
#elif defined(__SSE4_1__)
    /* TODO: SSE4.1 implementation */
    gemv_q5_k_ref(y, W, x, M, K);
#else
    gemv_q5_k_ref(y, W, x, M, K);
#endif
}

void gemm_nt_q5_k(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
#if defined(__AVX512F__)
    /* TODO: AVX-512 implementation */
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#elif defined(__AVX2__)
    /* TODO: AVX-2 implementation */
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#elif defined(__AVX__)
    /* TODO: AVX implementation */
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#elif defined(__SSE4_1__)
    /* TODO: SSE4.1 implementation */
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#else
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#endif
}
