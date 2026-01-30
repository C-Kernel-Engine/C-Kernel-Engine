/**
 * @file ck_parallel_decode.h
 * @brief Thread-pool-parallel GEMV dispatch for v6.6 decode
 *
 * Provides parallel wrappers for GEMV kernels used in the generated
 * inference code (ck-kernel-inference.c). These functions use
 * ck_threadpool to split GEMV row work across threads.
 *
 * The generated code calls gemv_q5_0_q8_0, gemv_q6_k_q8_k, etc.
 * as serial functions. By defining CK_PARALLEL_DECODE=1, we replace
 * these with parallel dispatch variants that use the global thread pool.
 *
 * Integration:
 *   1. #include "ck_parallel_decode.h" in ck-kernel-inference.c
 *   2. Call ck_parallel_decode_init() in ck_model_init()
 *   3. Call ck_parallel_decode_shutdown() in ck_model_free()
 *   4. GEMV calls are automatically parallelized via macro override
 */

#ifndef CK_PARALLEL_DECODE_H
#define CK_PARALLEL_DECODE_H

#include "ck_threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Lifecycle (call from ck_model_init / ck_model_free)
 * ============================================================================ */

/** Initialize thread pool for parallel decode. Call once at model init. */
void ck_parallel_decode_init(void);

/** Shutdown thread pool. Call at model teardown. */
void ck_parallel_decode_shutdown(void);

/* ============================================================================
 * Parallel GEMV Wrappers
 *
 * Same signatures as the serial GEMV functions, but internally dispatch
 * to thread pool using _parallel_simd variants.
 * ============================================================================ */

/**
 * Parallel gemv_q5_0_q8_0: y[M] = W[M,K] @ x_q8[K]
 * W is Q5_0 quantized, x_q8 is Q8_0 quantized.
 */
void gemv_q5_0_q8_0_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K);

/**
 * Parallel gemv_q6_k_q8_k: y[M] = W[M,K] @ x_q8[K]
 * W is Q6_K quantized, x_q8 is Q8_K quantized.
 */
void gemv_q6_k_q8_k_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K);

/**
 * Parallel gemv_q4_k_q8_k: y[M] = W[M,K] @ x_q8[K]
 * W is Q4_K quantized, x_q8 is Q8_K quantized.
 */
void gemv_q4_k_q8_k_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K);

/**
 * Parallel gemv_q8_0_q8_0: y[M] = W[M,K] @ x_q8[K]
 * Both W and x are Q8_0 quantized.
 */
void gemv_q8_0_q8_0_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K);

/**
 * Parallel gemv_fused_q5_0_bias: y[M] = W[M,K] @ quantize(x[K]) + bias[M]
 * W is Q5_0 quantized, x is FP32 (quantized to Q8_0 internally), bias is FP32.
 *
 * Splits into: (1) quantize x to Q8_0, (2) parallel GEMV, (3) add bias.
 */
void gemv_fused_q5_0_bias_parallel_dispatch(
    float *y, const void *W, const float *x, const float *bias, int M, int K);

/* ============================================================================
 * Macro overrides for generated code
 *
 * When CK_PARALLEL_DECODE is defined, redirect GEMV calls to parallel
 * dispatch versions. This allows the generated code to be parallelized
 * without modifying the code generator.
 * ============================================================================ */

#ifdef CK_PARALLEL_DECODE

#define gemv_q5_0_q8_0(y, W, x, M, K)   gemv_q5_0_q8_0_parallel_dispatch(y, W, x, M, K)
#define gemv_q6_k_q8_k(y, W, x, M, K)   gemv_q6_k_q8_k_parallel_dispatch(y, W, x, M, K)
#define gemv_q4_k_q8_k(y, W, x, M, K)   gemv_q4_k_q8_k_parallel_dispatch(y, W, x, M, K)
#define gemv_q8_0_q8_0(y, W, x, M, K)   gemv_q8_0_q8_0_parallel_dispatch(y, W, x, M, K)
#define gemv_fused_q5_0_bias_dispatch(y, W, x, bias, M, K) \
    gemv_fused_q5_0_bias_parallel_dispatch(y, W, x, bias, M, K)

#endif /* CK_PARALLEL_DECODE */

#ifdef __cplusplus
}
#endif

#endif /* CK_PARALLEL_DECODE_H */
