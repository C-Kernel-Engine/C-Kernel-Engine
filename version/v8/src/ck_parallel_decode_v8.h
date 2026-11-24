/**
 * @file ck_parallel_decode_v8.h
 * @brief Thread-pool-parallel GEMV dispatch for v8 decode
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
 *   1. #include "ck_parallel_decode_v8.h" in ck-kernel-inference.c
 *   2. Call ck_parallel_decode_init() in ck_model_init()
 *   3. Call ck_parallel_decode_shutdown() in ck_model_free()
 *   4. GEMV calls are automatically parallelized via macro override
 */

#ifndef CK_PARALLEL_DECODE_V8_H
#define CK_PARALLEL_DECODE_V8_H

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

/**
 * Parallel grouped DeltaNet decode. Heads own disjoint recurrent-state and
 * output slices, so this preserves the serial provider's per-head arithmetic.
 */
void gated_deltanet_llama_avx2_parallel_forward(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int num_heads, int group_count, int state_dim, float norm_eps);

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

void gemv_q4_k_q8_k_repacked_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K);

/**
 * Parallel gemv_q8_0_q8_0: y[M] = W[M,K] @ x_q8[K]
 * Both W and x are Q8_0 quantized.
 */
void gemv_q8_0_q8_0_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K);

/**
 * Parallel gemv_q5_1_q8_1: y[M] = W[M,K] @ x[K]
 * W is Q5_1 quantized, x is FP32 (quantized internally by kernel).
 */
void gemv_q5_1_q8_1_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K);

/**
 * Parallel gemv_q5_k: y[M] = W[M,K] @ x[K]
 * W is Q5_K quantized, x is FP32 (quantized internally by kernel).
 */
void gemv_q5_k_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K);

/**
 * Parallel gemv_q8_0_q8_0_contract: y[M] = W[M,K] @ x[K]
 * W is Q8_0 quantized, x is FP32; wrapper enforces Q8_0 activation contract.
 */
void gemv_q8_0_q8_0_contract_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K);

/**
 * Parallel gemv_fused_q5_0_bias: y[M] = W[M,K] @ quantize(x[K]) + bias[M]
 * W is Q5_0 quantized, x is FP32 (quantized to Q8_0 internally), bias is FP32.
 *
 * Splits into: (1) quantize x to Q8_0, (2) parallel GEMV, (3) add bias.
 */
void gemv_fused_q5_0_bias_parallel_dispatch(
    float *y, const void *W, const float *x, const float *bias, int M, int K);

/**
 * Parallel RWKV-7 channelmix decode (T=1):
 *   hidden[j] = relu(Wk[j,:] @ xk)^2, out[i] = Wv[i,:] @ hidden
 * Row-splits both GEMVs across the global thread pool using the serial
 * provider's own row primitives, so output is bitwise identical.
 */
int ck_rwkv7_channelmix_decode_parallel(
    const float *x, const float *shift_in, const float *mu,
    const float *Wk, const float *Wv,
    float *shift_out, float *out,
    int dim, int hidden_dim,
    float *scratch, size_t scratch_elems);

/**
 * Reports whether a channelmix binding may use the row-parallel dispatcher.
 * True exactly for the bindings the serial provider accepts (including the
 * in-place carries shift_out == shift_in, shift_out == x, out == x).
 */
int ck_rwkv7_channelmix_parallel_binding_ok(
    const float *x, const float *shift_in, const float *mu,
    const float *Wk, const float *Wv,
    float *shift_out, float *out,
    float *scratch, int dim, int hidden_dim);

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
#define gemv_q8_0_q8_0(y, W, x, M, K)   gemv_q8_0_q8_0_parallel_dispatch(y, W, x, M, K)
#define gemv_q5_1_q8_1(y, W, x, M, K)   gemv_q5_1_q8_1_parallel_dispatch(y, W, x, M, K)
#define gemv_q5_k(y, W, x, M, K)        gemv_q5_k_parallel_dispatch(y, W, x, M, K)
#define gemv_q8_0_q8_0_contract(y, W, x, M, K) \
    gemv_q8_0_q8_0_contract_parallel_dispatch(y, W, x, M, K)
#define gemv_fused_q5_0_bias_dispatch(y, W, x, bias, M, K) \
    gemv_fused_q5_0_bias_parallel_dispatch(y, W, x, bias, M, K)
#define ck_rwkv7_channelmix_decode(...) \
    ck_rwkv7_channelmix_decode_parallel(__VA_ARGS__)

#endif /* CK_PARALLEL_DECODE */

#ifdef __cplusplus
}
#endif

#endif /* CK_PARALLEL_DECODE_V8_H */
