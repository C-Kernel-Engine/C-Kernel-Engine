#pragma once

/*
 * OpenMP-parallel GEMV variants (same signatures as serial counterparts).
 *
 * WARNING: These use #pragma omp parallel for which has high fork/join
 * overhead (~50-200us per call). During inference this makes them SLOWER
 * than the serial versions. Measured 1.9x slower on real workloads.
 *
 * Do NOT use for inference. These exist for:
 *   - Correctness reference (numerically identical to serial)
 *   - Future conversion to a pthread thread pool (persistent threads,
 *     ~2-5us dispatch overhead instead of 50-200us fork/join)
 *
 * The serial kernels in gemm_kernels_q8_0.c / gemm_kernels_q5_0.c are
 * faster for all current use cases.
 */

void gemv_q8_0_q8_0_parallel_omp(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q5_0_q8_0_parallel_omp(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_fused_q5_0_bias_parallel_omp(float *y, const void *W, const float *x,
                                const float *bias, int M, int K);
