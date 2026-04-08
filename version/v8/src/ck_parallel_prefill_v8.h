/**
 * @file ck_parallel_prefill_v8.h
 * @brief Thread-pool-parallel GEMM dispatch for v8 prefill
 *
 * Provides parallel wrappers for GEMM kernels used in generated prefill code.
 * These functions use ck_threadpool to split GEMM row work (M dimension)
 * across threads. Each thread computes rows [r0, r1) of the output matrix C.
 *
 * Integration:
 *   1. #include "ck_parallel_prefill_v8.h" in ck-kernel-inference.c
 *   2. Call ck_parallel_prefill_init() in ck_model_init()
 *   3. Call ck_parallel_prefill_shutdown() in ck_model_free()
 *   4. GEMM calls are automatically parallelized via macro override
 *
 * All GEMM kernels share the same 7-arg signature:
 *   gemm_nt_<type>(A, B, bias, C, M, N, K)
 *
 * Kernel types:
 *   - gemm_nt_q5_0_q8_0: Q8_0 activations x Q5_0 weights (Q/K proj, out proj, MLP gate+up)
 *   - gemm_nt_q8_0_q8_0: Q8_0 activations x Q8_0 weights (V proj)
 *   - gemm_nt_q6_k_q8_k: Q8_K activations x Q6_K weights (MLP down proj)
 *   - gemm_nt_q5_1_q8_1: FP32 activations x Q5_1 weights
 *   - gemm_nt_q5_k:      FP32 activations x Q5_K weights
 */

#ifndef CK_PARALLEL_PREFILL_V8_H
#define CK_PARALLEL_PREFILL_V8_H

#include "ck_threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Lifecycle */
void ck_parallel_prefill_init(void);
void ck_parallel_prefill_shutdown(void);

/* Parallel GEMM Wrappers - same signatures as serial GEMM functions */
void gemm_nt_q5_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q8_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q6_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q5_1_q8_1_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q5_k_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

/* Macro overrides - when CK_PARALLEL_PREFILL is defined,
 * preprocessor redirects serial gemm_nt_*() calls to thread pool dispatch */
#ifdef CK_PARALLEL_PREFILL

#define gemm_nt_q5_0_q8_0(A, B, bias, C, M, N, K) \
    gemm_nt_q5_0_q8_0_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q8_0_q8_0(A, B, bias, C, M, N, K) \
    gemm_nt_q8_0_q8_0_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q6_k_q8_k(A, B, bias, C, M, N, K) \
    gemm_nt_q6_k_q8_k_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q5_1_q8_1(A, B, bias, C, M, N, K) \
    gemm_nt_q5_1_q8_1_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q5_k(A, B, bias, C, M, N, K) \
    gemm_nt_q5_k_parallel_dispatch(A, B, bias, C, M, N, K)

#endif /* CK_PARALLEL_PREFILL */

#ifdef __cplusplus
}
#endif

#endif /* CK_PARALLEL_PREFILL_V8_H */
