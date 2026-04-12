/**
 * @file ck_parallel_decode.c
 * @brief Thread-pool-parallel GEMV dispatch for v7 decode
 *
 * Wraps each GEMV kernel call in a threadpool dispatch.
 * Each wrapper:
 *   1. Packs arguments into a stack-local struct
 *   2. Dispatches to threadpool (all threads execute _parallel_simd variant)
 *   3. Returns when all threads complete
 *
 * For single-thread builds (n_threads=1), dispatch() fast-paths to
 * a direct call with no overhead.
 */

#include "ck_parallel_decode_v8.h"
#include "ck_threadpool.h"
#include "ckernel_quant.h"

#include <stdio.h>
#include <string.h>

/* External parallel GEMV kernels (defined in src/kernels/) */
extern void gemv_q5_0_parallel_simd(float *y, const void *W, const float *x,
                                     int M, int K, int ith, int nth);
extern void gemv_q5_0_q8_0_parallel_simd(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);
extern void gemv_q6_k_q8_k_parallel_simd(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);
extern void gemv_q8_0_q8_0_parallel_simd(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);

/* Q4_K parallel SIMD — check if it exists, otherwise fall back */
extern void gemv_q4_k_q8_k_parallel_simd(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);
extern void gemv_q4_k_q8_k_parallel(float *y, const void *W, const void *x_q8,
                                     int M, int K, int ith, int nth);

/* Serial fallbacks (for correctness reference) */
extern void gemv_q5_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q6_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q4_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q8_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q5_1_q8_1(float *y, const void *W, const float *x, int M, int K);
extern void gemv_q5_k(float *y, const void *W, const float *x, int M, int K);
extern void gemv_q8_0_q8_0_contract(float *y, const void *W, const float *x, int M, int K);

/* Serial fused fallback */
extern void gemv_fused_q5_0_bias_dispatch(float *y, const void *W, const float *x,
                                           const float *bias, int M, int K);

/* Quantization (from ckernel_quant.h / quantize.c) */
extern void quantize_row_q8_0(const float *x, void *vy, int k);

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

void ck_parallel_decode_init(void)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (pool) {
        fprintf(stderr, "[CK parallel decode] Initialized with %d threads\n",
                ck_threadpool_n_threads(pool));
    }
}

void ck_parallel_decode_shutdown(void)
{
    ck_threadpool_global_destroy();
}

/* ============================================================================
 * Argument Packing Structs
 *
 * Each GEMV dispatch needs to pass (y, W, x, M, K) to all threads.
 * We pack into a stack-local struct and pass its address.
 * ============================================================================ */

typedef struct {
    float      *y;
    const void *W;
    const void *x_q8;
    int         M;
    int         K;
} gemv_args_t;

typedef struct {
    float       *y;
    const void  *W;
    const float *x;
    int          M;
    int          K;
    size_t       W_row_bytes;
} gemv_fp32_args_t;

/* Q5_K weights are row-major packed super-blocks.
 * Keep a local layout copy to compute byte offsets without pulling in
 * kernel-private headers. Must stay ggml-compatible. */
typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} ck_block_q5_K_layout_t;

/* ============================================================================
 * Work Functions (called on each thread)
 * ============================================================================ */

static void work_gemv_q5_0_q8_0(int ith, int nth, void *args)
{
    const gemv_args_t *a = (const gemv_args_t *)args;
    gemv_q5_0_q8_0_parallel_simd(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
}

static void work_gemv_q6_k_q8_k(int ith, int nth, void *args)
{
    const gemv_args_t *a = (const gemv_args_t *)args;
    gemv_q6_k_q8_k_parallel_simd(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
}

static void work_gemv_q4_k_q8_k(int ith, int nth, void *args)
{
    const gemv_args_t *a = (const gemv_args_t *)args;
#if defined(CK_TARGET_ARM)
    gemv_q4_k_q8_k_parallel(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
#else
    gemv_q4_k_q8_k_parallel_simd(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
#endif
}

static void work_gemv_q8_0_q8_0(int ith, int nth, void *args)
{
    const gemv_args_t *a = (const gemv_args_t *)args;
    gemv_q8_0_q8_0_parallel_simd(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
}

static void work_gemv_q5_1_q8_1(int ith, int nth, void *args)
{
    const gemv_fp32_args_t *a = (const gemv_fp32_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemv_q5_1_q8_1(
        a->y + r0,
        (const char *)a->W + (size_t)r0 * a->W_row_bytes,
        a->x,
        r1 - r0,
        a->K
    );
}

static void work_gemv_q5_k(int ith, int nth, void *args)
{
    const gemv_fp32_args_t *a = (const gemv_fp32_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemv_q5_k(
        a->y + r0,
        (const char *)a->W + (size_t)r0 * a->W_row_bytes,
        a->x,
        r1 - r0,
        a->K
    );
}

/* ============================================================================
 * Parallel Dispatch Wrappers
 *
 * These have the SAME signature as the serial GEMV functions.
 * They pack args and dispatch to the threadpool.
 * ============================================================================ */

void gemv_q5_0_q8_0_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1) {
        /* Fall back to serial — avoids overhead for single-thread */
        gemv_q5_0_q8_0(y, W, x_q8, M, K);
        return;
    }

    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8, .M = M, .K = K };
    ck_threadpool_dispatch(pool, work_gemv_q5_0_q8_0, &args);
}

void gemv_q6_k_q8_k_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1) {
        gemv_q6_k_q8_k(y, W, x_q8, M, K);
        return;
    }

    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8, .M = M, .K = K };
    ck_threadpool_dispatch(pool, work_gemv_q6_k_q8_k, &args);
}

void gemv_q4_k_q8_k_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1) {
        gemv_q4_k_q8_k(y, W, x_q8, M, K);
        return;
    }

    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8, .M = M, .K = K };
    ck_threadpool_dispatch(pool, work_gemv_q4_k_q8_k, &args);
}

void gemv_q8_0_q8_0_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1) {
        gemv_q8_0_q8_0(y, W, x_q8, M, K);
        return;
    }

    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8, .M = M, .K = K };
    ck_threadpool_dispatch(pool, work_gemv_q8_0_q8_0, &args);
}

void gemv_q5_1_q8_1_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || (K % QK5_1) != 0) {
        gemv_q5_1_q8_1(y, W, x, M, K);
        return;
    }

    const size_t W_row_bytes = (size_t)(K / QK5_1) * sizeof(block_q5_1);
    gemv_fp32_args_t args = {
        .y = y, .W = W, .x = x, .M = M, .K = K, .W_row_bytes = W_row_bytes
    };
    ck_threadpool_dispatch(pool, work_gemv_q5_1_q8_1, &args);
}

void gemv_q5_k_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || (K % QK_K) != 0) {
        gemv_q5_k(y, W, x, M, K);
        return;
    }

    const size_t W_row_bytes = (size_t)(K / QK_K) * sizeof(ck_block_q5_K_layout_t);
    gemv_fp32_args_t args = {
        .y = y, .W = W, .x = x, .M = M, .K = K, .W_row_bytes = W_row_bytes
    };
    ck_threadpool_dispatch(pool, work_gemv_q5_k, &args);
}

/* Keep stack footprint aligned with the contract kernel implementation. */
#define CK_Q80_STACK_Q8_BLOCKS 1024

void gemv_q8_0_q8_0_contract_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1) {
        gemv_q8_0_q8_0_contract(y, W, x, M, K);
        return;
    }

    if ((K % QK8_0) != 0) {
        gemv_q8_0_q8_0_contract(y, W, x, M, K);
        return;
    }

    const int n_blocks = K / QK8_0;
    if (n_blocks <= 0 || n_blocks > CK_Q80_STACK_Q8_BLOCKS) {
        gemv_q8_0_q8_0_contract(y, W, x, M, K);
        return;
    }

    block_q8_0 x_q8_buf[n_blocks];
    quantize_row_q8_0(x, x_q8_buf, K);
    gemv_q8_0_q8_0_parallel_dispatch(y, W, x_q8_buf, M, K);
}

/* ============================================================================
 * Fused GEMV Parallel Dispatch
 *
 * gemv_fused_q5_0_bias = quantize(x) + gemv(W, x_q8) + bias
 *
 * Split into:
 *   1. Quantize FP32 x → Q8_0 (serial, fast: O(K))
 *   2. Parallel GEMV across threads (the expensive part: O(M*K))
 *   3. Add bias (serial, fast: O(M))
 * ============================================================================ */

/* Max K we support on stack: 16384 elements = ~17KB Q8_0 buffer.
 * Qwen2-0.5B max K is 4864, so this is plenty. */
#define FUSED_Q8_STACK_MAX 16384

void gemv_fused_q5_0_bias_parallel_dispatch(
    float *y, const void *W, const float *x, const float *bias, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1) {
        /* Fall back to original fused kernel (serial) */
        gemv_fused_q5_0_bias_dispatch(y, W, x, bias, M, K);
        return;
    }

    /* Step 1: Quantize FP32 x to Q8_0 (serial, fast) */
    int n_blocks = K / QK8_0;
    block_q8_0 x_q8_buf[n_blocks]; /* VLA on stack — safe for K <= ~16K */

    quantize_row_q8_0(x, x_q8_buf, K);

    /* Step 2: Parallel GEMV — reuse existing parallel dispatch */
    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8_buf, .M = M, .K = K };
    ck_threadpool_dispatch(pool, work_gemv_q5_0_q8_0, &args);

    /* Step 3: Add bias (serial, fast) */
    if (bias) {
        for (int i = 0; i < M; i++) {
            y[i] += bias[i];
        }
    }
}
