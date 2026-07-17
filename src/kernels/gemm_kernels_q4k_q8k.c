/**
 * @file gemm_kernels_q4k_q8k.c
 * @brief Q4_K (weights) x Q8_K (activations) kernels for inference
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
 * Implements decode-style matvec/matmul where weights are Q4_K and the
 * activations are quantized on-the-fly to Q8_K. This is inference-only;
 * no backward pass is provided here.
 */

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ck_threadpool.h"
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
            /* Keep the active llama.cpp oracle expression intact. In
             * particular, do not insert a store/load rounding barrier between
             * the multiply and nearest_int: that changes tie cases on AVX2. */
            float scaled = iscale * x[j];
            int v = ck_nearest_int(scaled);
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
void quantize_row_q8_k_avx(const float *x, void *vy, int k);
void quantize_row_q8_k_avx2(const float *x, void *vy, int k);
void quantize_row_q8_k_avx512(const float *x, void *vy, int k);

void quantize_row_q8_k(const float *x, void *vy, int k) {
    const char *ref_env = getenv("CK_DEBUG_Q8K_REF");
    if (ref_env && atoi(ref_env) != 0) {
        quantize_row_q8_k_ref(x, vy, k);
        return;
    }
#if defined(__AVX512F__) && defined(__AVX512BW__)
    quantize_row_q8_k_avx512(x, vy, k);
#elif defined(__AVX2__)
    quantize_row_q8_k_avx2(x, vy, k);
#elif defined(__AVX__)
    quantize_row_q8_k_avx(x, vy, k);
#elif defined(__SSE4_1__)
    quantize_row_q8_k_sse(x, vy, k);
#else
    quantize_row_q8_k_ref(x, vy, k);
#endif
}

void quantize_batch_q8_k_4row_nearest_even(const float *x, void *vy,
                                            int num_rows, int k) {
    if (!x || !vy || num_rows <= 0 || k <= 0) {
        return;
    }
    assert(k % QK_K == 0);

    block_q8_K *y = (block_q8_K *)vy;
    const int blocks_per_row = k / QK_K;
    int row0 = 0;

#if defined(__AVX2__)
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    for (; row0 + 3 < num_rows; row0 += 4) {
        for (int block = 0; block < blocks_per_row; ++block) {
            for (int lane = 0; lane < 4; ++lane) {
                const float *src =
                        x + (size_t)(row0 + lane) * (size_t)k +
                        (size_t)block * QK_K;
                block_q8_K *dst =
                        y + (size_t)(row0 + lane) * (size_t)blocks_per_row +
                        (size_t)block;
                __m256 source[QK_K / 8];
                __m256 values0 = _mm256_loadu_ps(src);
                __m256 values1 = _mm256_loadu_ps(src + 8);
                __m256 values2 = _mm256_loadu_ps(src + 16);
                __m256 values3 = _mm256_loadu_ps(src + 24);
                __m256 max_abs = _mm256_max_ps(
                        _mm256_andnot_ps(sign_bit, values0),
                        _mm256_andnot_ps(sign_bit, values1));
                max_abs = _mm256_max_ps(
                        max_abs, _mm256_andnot_ps(sign_bit, values2));
                max_abs = _mm256_max_ps(
                        max_abs, _mm256_andnot_ps(sign_bit, values3));
                __m256 signed_max_mask = _mm256_or_ps(
                        _mm256_or_ps(
                                _mm256_cmp_ps(max_abs, values0, _CMP_EQ_OQ),
                                _mm256_cmp_ps(max_abs, values1, _CMP_EQ_OQ)),
                        _mm256_or_ps(
                                _mm256_cmp_ps(max_abs, values2, _CMP_EQ_OQ),
                                _mm256_cmp_ps(max_abs, values3, _CMP_EQ_OQ)));
                source[0] = values0;
                source[1] = values1;
                source[2] = values2;
                source[3] = values3;

                for (int subblock = 1; subblock < QK_K / 32; ++subblock) {
                    const int vector = subblock * 4;
                    values0 = _mm256_loadu_ps(src + subblock * 32);
                    values1 = _mm256_loadu_ps(src + subblock * 32 + 8);
                    values2 = _mm256_loadu_ps(src + subblock * 32 + 16);
                    values3 = _mm256_loadu_ps(src + subblock * 32 + 24);
                    const __m256 previous_max = max_abs;
                    max_abs = _mm256_max_ps(
                            max_abs, _mm256_andnot_ps(sign_bit, values0));
                    max_abs = _mm256_max_ps(
                            max_abs, _mm256_andnot_ps(sign_bit, values1));
                    max_abs = _mm256_max_ps(
                            max_abs, _mm256_andnot_ps(sign_bit, values2));
                    max_abs = _mm256_max_ps(
                            max_abs, _mm256_andnot_ps(sign_bit, values3));
                    signed_max_mask = _mm256_and_ps(
                            signed_max_mask,
                            _mm256_cmp_ps(previous_max, max_abs, _CMP_EQ_OQ));
                    const __m256 current_mask = _mm256_or_ps(
                            _mm256_or_ps(
                                    _mm256_cmp_ps(max_abs, values0, _CMP_EQ_OQ),
                                    _mm256_cmp_ps(max_abs, values1, _CMP_EQ_OQ)),
                            _mm256_or_ps(
                                    _mm256_cmp_ps(max_abs, values2, _CMP_EQ_OQ),
                                    _mm256_cmp_ps(max_abs, values3, _CMP_EQ_OQ)));
                    signed_max_mask =
                            _mm256_or_ps(signed_max_mask, current_mask);
                    source[vector] = values0;
                    source[vector + 1] = values1;
                    source[vector + 2] = values2;
                    source[vector + 3] = values3;
                }

                __m128 max4 = _mm_max_ps(
                        _mm256_extractf128_ps(max_abs, 1),
                        _mm256_castps256_ps128(max_abs));
                max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
                max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
                const float max_scalar = _mm_cvtss_f32(max4);
                const __m256 winning_lane = _mm256_cmp_ps(
                        _mm256_set1_ps(max_scalar), max_abs, _CMP_EQ_OQ);
                const int signed_winner = _mm256_movemask_ps(
                        _mm256_and_ps(signed_max_mask, winning_lane));
                volatile float inverse_max =
                        max_scalar == 0.0f ? 0.0f : 1.0f / max_scalar;
                volatile float iscale =
                        inverse_max * (signed_winner ? -127.0f : 127.0f);
                const __m256 iscale_vec = _mm256_set1_ps(iscale);

                for (int vector = 0; vector < QK_K / 8; ++vector) {
                    const __m256 rounded = _mm256_round_ps(
                            _mm256_mul_ps(source[vector], iscale_vec),
                            _MM_ROUND_NEAREST);
                    const __m256i integers = _mm256_cvtps_epi32(rounded);
                    int32_t values[8];
                    _mm256_storeu_si256((__m256i *)values, integers);
                    for (int i = 0; i < 8; ++i) {
                        dst->qs[vector * 8 + i] = (int8_t)values[i];
                    }
                }

                for (int group = 0; group < QK_K / 16; ++group) {
                    int sum = 0;
                    for (int i = 0; i < 16; ++i) {
                        sum += dst->qs[group * 16 + i];
                    }
                    dst->bsums[group] = (int16_t)sum;
                }
                dst->d = max_scalar == 0.0f ? 0.0f : 1.0f / iscale;
            }
        }
    }
#endif

    for (; row0 < num_rows; ++row0) {
        quantize_row_q8_k(
                x + (size_t)row0 * (size_t)k,
                y + (size_t)row0 * (size_t)blocks_per_row,
                k);
    }
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

        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            sumi += (int)x[i].bsums[j] * (int)m_val[j / 2];
        }

        int32_t scaled_sum = 0;
        for (int group = 0; group < 4; ++group) {
            const uint8_t *qs = &w[i].qs[group * 32];
            const int8_t *q8_lo = &x[i].qs[group * 64];
            const int8_t *q8_hi = q8_lo + 32;
            int32_t lo = 0;
            int32_t hi = 0;
            for (int l = 0; l < 32; ++l) {
                lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo[l];
                hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi[l];
            }
            scaled_sum += (int32_t)sc[2 * group] * lo;
            scaled_sum += (int32_t)sc[2 * group + 1] * hi;
        }
        sumf += d * (float)scaled_sum - dmin * (float)sumi;
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

/* ============================================================================
 * PARALLEL VERSIONS (for parallel orchestration)
 *
 * These receive ith (thread index) and nth (total threads) from orchestration.
 * OpenMP lives in orchestration layer, NOT here.
 *
 * Naming: *_parallel = receives ith/nth, processes only its portion
 *         *_ref/_avx  = single-threaded, processes all rows
 * ============================================================================ */

void gemv_q4_k_q8_k_parallel(float *y,
                             const void *W,
                             const void *x_q8,
                             int M, int K,
                             int ith, int nth)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }
    if (ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    /* Compute row range for this thread */
    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;

    if (r0 >= M) {
        return;  /* This thread has no work */
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    /* Only process rows [r0, r1) */
    for (int row = r0; row < r1; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q4_k_q8_k_ref(w_row, x, K);
    }
}

void gemv_q4_k_q8_k(float *y,
                    const void *W,
                    const void *x_q8,
                    int M, int K)
{
    const char *ref_env = getenv("CK_DEBUG_Q4K_Q8_REF");
    if (ref_env && atoi(ref_env) != 0) {
        gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
        return;
    }
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && !defined(CK_NO_AVX512_VNNI)
    /* VNNI: Best for decode (single token) - INT8 dot product acceleration */
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

typedef struct {
    float *Y;
    const void *W;
    const block_q8_K *X;
    int M_out;
    int N_batch;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
} gemm_q4_k_q8_k_work_t;

static void gemm_q4_k_q8_k_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_k_q8_k_work_t *a = (gemm_q4_k_q8_k_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    const int dr = (a->M_out + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < a->M_out) ? (r0 + dr) : a->M_out;
    if (r0 >= a->M_out) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)a->W;
    const block_q4_K *w_start = blocks + (size_t)r0 * (size_t)a->blocks_per_row;
    const int rows = r1 - r0;

    for (int n = 0; n < a->N_batch; ++n) {
        const block_q8_K *x_row = a->X + (size_t)n * (size_t)a->blocks_per_vec;
        gemv_q4_k_q8_k(a->Y + (size_t)n * (size_t)a->M_out + (size_t)r0,
                       w_start,
                       x_row,
                       rows,
                       a->K);
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
    const int blocks_per_row = K / QK_K;
    const size_t work_items = (size_t)M * (size_t)N;

    if (work_items >= 4096u && M >= 512 && N > 1) {
        ck_threadpool_t *pool = ck_threadpool_global();
        const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
        int active_threads = pool_threads;
        if (active_threads > M) {
            active_threads = M;
        }
        if (active_threads > 1) {
            gemm_q4_k_q8_k_work_t work = {
                .Y = Y,
                .W = W,
                .X = X,
                .M_out = M,
                .N_batch = N,
                .K = K,
                .blocks_per_vec = blocks_per_vec,
                .blocks_per_row = blocks_per_row,
            };
            ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_k_q8_k_thread_fn, &work);
            return;
        }
    }

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
