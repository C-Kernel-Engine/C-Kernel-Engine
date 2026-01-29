/*
 * gemv_omp.c - OpenMP-parallel GEMV kernels for decode mode
 *
 * WARNING: These kernels use #pragma omp parallel for, which creates and
 * joins threads on EVERY call. During inference, each decode token invokes
 * kernels 500+ times. OpenMP fork/join overhead (~50-200us per call) makes
 * these SLOWER than serial for real inference workloads:
 *
 *   Measured on i7-3630QM (4C/8T), Qwen 0.5B:
 *     Serial kernels:   170 ms/tok  (5.9 tok/s)
 *     OMP parallel:     327 ms/tok  (3.1 tok/s)  ← 1.9x SLOWER
 *
 * The math is correct (10/10 parity tests pass) but the threading model is
 * wrong for this workload. OpenMP is designed for long-running parallel
 * regions, not thousands of short kernel calls per token.
 *
 * TODO: Replace with a persistent pthread thread pool:
 *   - Create N worker pthreads once at startup
 *   - Workers spin/wait on a barrier or futex
 *   - Kernel dispatch: write work descriptor, signal barrier (~2-5us)
 *   - Workers execute rows, signal completion
 *   - This is what llama.cpp does (ggml_threadpool) to get 30 tok/s
 *     on the same hardware where these OMP kernels get 3.1 tok/s
 *
 * These are parallel variants of the serial GEMV kernels in
 * gemm_kernels_q8_0.c and gemm_kernels_q5_0.c. The serial kernels
 * remain untouched for multi-stream / multi-model serving where
 * per-kernel parallelism is not wanted.
 *
 * All three kernels are row-parallel: each output y[row] is an
 * independent dot product, so we partition rows across threads.
 */

#include <omp.h>
#include "ckernel_quant.h"

/* Existing vec_dot dispatch functions (in gemm_kernels_q8_0.c / gemm_kernels_q5_0.c) */
extern void vec_dot_q8_0_q8_0(int n, float *s, const void *vx, const void *vy);
extern void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy);
extern void quantize_row_q8_0(const float *x, void *y, int k);

/* ---------------------------------------------------------------------------
 * gemv_q8_0_q8_0_parallel_omp  (logits — 37% of decode time)
 *
 * Same 5-param signature as gemv_q8_0_q8_0 — drop-in swap.
 * schedule(static) ensures contiguous row blocks per thread → no false sharing.
 * x_blocks is read-only shared (~1 KB for K=896) → stays in L1.
 * Each thread reads disjoint weight rows → no contention.
 * --------------------------------------------------------------------------- */
void gemv_q8_0_q8_0_parallel_omp(float *y,
                          const void *W,
                          const void *x_q8,
                          int M, int K)
{
    const block_q8_0 *w_blocks = (const block_q8_0 *)W;
    const block_q8_0 *x_blocks = (const block_q8_0 *)x_q8;
    const int blocks_per_row = K / QK8_0;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; row++) {
        vec_dot_q8_0_q8_0(K, &y[row],
                          &w_blocks[row * blocks_per_row],
                          x_blocks);
    }
}

/* ---------------------------------------------------------------------------
 * gemv_q5_0_q8_0_parallel_omp  (mlp_down — 10% of decode time)
 *
 * Same 5-param signature as gemv_q5_0_q8_0 — drop-in swap.
 * --------------------------------------------------------------------------- */
void gemv_q5_0_q8_0_parallel_omp(float *y,
                          const void *W,
                          const void *x_q8,
                          int M, int K)
{
    const block_q5_0 *w_blocks = (const block_q5_0 *)W;
    const block_q8_0 *x_blocks = (const block_q8_0 *)x_q8;
    const int blocks_per_row = K / QK5_0;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; row++) {
        vec_dot_q5_0_q8_0(K, &y[row],
                          &w_blocks[row * blocks_per_row],
                          x_blocks);
    }
}

/* ---------------------------------------------------------------------------
 * gemv_fused_q5_0_bias_parallel_omp  (mlp_gate_up — 44% of decode time)
 *
 * Same 6-param signature as gemv_fused_q5_0_bias_dispatch — drop-in swap.
 * Quantizes x (FP32→Q8_0) once serially, then runs the parallel GEMV.
 * Quantization is O(K) = 896 elements ≈ negligible vs. O(M*K) GEMV.
 * --------------------------------------------------------------------------- */
void gemv_fused_q5_0_bias_parallel_omp(float *y,
                                const void *W,
                                const float *x,
                                const float *bias,
                                int M, int K)
{
    const block_q5_0 *w_blocks = (const block_q5_0 *)W;
    const int blocks_per_row = K / QK5_0;

    /* Quantize input ONCE (serial, fast — K=896 → 28 blocks = 952 bytes) */
    block_q8_0 x_q8[K / QK8_0];
    quantize_row_q8_0(x, (void *)x_q8, K);

    /* Parallel GEMV over output rows */
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; row++) {
        vec_dot_q5_0_q8_0(K, &y[row],
                          &w_blocks[row * blocks_per_row],
                          x_q8);
        if (bias) y[row] += bias[row];
    }
}
