/**
 * @file test_gemm_vs_gemv.c
 * @brief Single-core benchmark: GEMM vs GEMV, INT8 vs FP32 activation paths
 *
 * WHAT IT DOES:
 *     - Benchmarks GEMM vs GEMV with varying batch sizes
 *     - Compares INT8 activation path vs FP32 activation path
 *     - Measures compute vs memory bound behavior
 *     - Tests kernel performance at different M values
 *
 * WHEN TO RUN:
 *     - When optimizing GEMM/GEMV kernel selection
 *     - Manual profiling of batch size crossover points
 *     - Performance characterization research
 *
 * TRIGGERED BY:
 *     - NO AUTOMATED TRIGGERS
 *     - Not in any Makefile target
 *     - Manual compilation required (see below)
 *
 * COMPILE:
 *   gcc -O3 -march=native -o test_gemm_vs_gemv test_gemm_vs_gemv.c \
 *       ../../src/kernels/gemm_kernels_q5_0.c ../../src/kernels/gemm_batch_int8.c \
 *       ../../src/kernels/dequant_kernels.c -I../../include -lm
 *
 * RUN:
 *   OMP_NUM_THREADS=1 ./test_gemm_vs_gemv
 *
 * DEPENDENCIES:
 *     - CK-Engine kernel source files
 *     - ckernel_quant.h
 *
 * STATUS: USEFUL BUT ORPHANED
 *     - Provides valuable GEMM/GEMV performance insights
 *     - Not integrated into automated testing
 *     - Consider adding a Makefile target if used regularly
 *
 * RECOMMENDATION: KEEP but consider adding `make bench-gemm-vs-gemv` target
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "ckernel_quant.h"

/* Kernel declarations */
extern void gemv_q5_0(float *y, const void *W, const float *x, int M, int K);
extern void gemm_nt_q5_0(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
extern void gemm_nt_q5_0_q8_0(const void *A_q8, const void *B_q5, const float *bias, float *C, int M, int N, int K);
extern void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy);
extern void quantize_row_q8_0(const float *x, void *vy, int k);

/* Timer utility */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* Q5_0 block allocation */
static void *alloc_q5_0(int rows, int K) {
    int blocks_per_row = K / QK5_0;
    size_t size = (size_t)rows * blocks_per_row * sizeof(block_q5_0);
    void *ptr = aligned_alloc(64, size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

/* Q8_0 block allocation */
static void *alloc_q8_0(int rows, int K) {
    int blocks_per_row = K / QK8_0;
    size_t size = (size_t)rows * blocks_per_row * sizeof(block_q8_0);
    void *ptr = aligned_alloc(64, size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

/* Initialize Q5_0 weights with random-ish values */
static void init_q5_0_weights(void *W, int N, int K) {
    block_q5_0 *blocks = (block_q5_0 *)W;
    int blocks_per_row = K / QK5_0;

    for (int n = 0; n < N; n++) {
        for (int b = 0; b < blocks_per_row; b++) {
            block_q5_0 *blk = &blocks[n * blocks_per_row + b];
            blk->d = CK_FP32_TO_FP16(0.1f);
            uint32_t qh = 0;
            for (int j = 0; j < 16; j++) {
                int q0 = (j + n) % 16;
                int q1 = (j + n + 8) % 16;
                blk->qs[j] = (q1 << 4) | q0;
                if ((j + n) % 3 == 0) qh |= (1u << j);
                if ((j + n + 1) % 3 == 0) qh |= (1u << (j + 16));
            }
            memcpy(blk->qh, &qh, 4);
        }
    }
}

/* Initialize FP32 activations */
static void init_fp32_activations(float *A, int M, int K) {
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)(i % 256 - 128) / 128.0f;
    }
}

/* Initialize Q8_0 activations (quantize from FP32) */
static void init_q8_0_activations(void *A_q8, const float *A_fp32, int M, int K) {
    int blocks_per_row = K / QK8_0;
    block_q8_0 *blocks = (block_q8_0 *)A_q8;

    for (int m = 0; m < M; m++) {
        const float *row = &A_fp32[m * K];
        block_q8_0 *row_blocks = &blocks[m * blocks_per_row];

        for (int b = 0; b < blocks_per_row; b++) {
            const float *src = &row[b * QK8_0];
            block_q8_0 *blk = &row_blocks[b];

            /* Find max absolute value */
            float amax = 0.0f;
            for (int j = 0; j < QK8_0; j++) {
                float v = fabsf(src[j]);
                if (v > amax) amax = v;
            }

            /* Compute scale */
            float d = amax / 127.0f;
            blk->d = CK_FP32_TO_FP16(d);

            /* Quantize */
            float id = (d != 0.0f) ? 127.0f / amax : 0.0f;
            for (int j = 0; j < QK8_0; j++) {
                float v = src[j] * id;
                blk->qs[j] = (int8_t)roundf(fmaxf(-128.0f, fminf(127.0f, v)));
            }
        }
    }
}

/* Benchmark single configuration */
typedef struct {
    double mean_ms;
    double min_ms;
    double max_ms;
    double std_ms;
    double gflops;
} bench_result_t;

static bench_result_t benchmark_gemm_fp32(const float *A, const void *B, float *C,
                                          int M, int N, int K,
                                          int warmup, int runs) {
    bench_result_t res = {0};
    double *times = malloc(runs * sizeof(double));

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        gemm_nt_q5_0(A, B, NULL, C, M, N, K);
    }

    /* Benchmark */
    for (int r = 0; r < runs; r++) {
        double start = get_time_ms();
        gemm_nt_q5_0(A, B, NULL, C, M, N, K);
        times[r] = get_time_ms() - start;
    }

    /* Compute stats */
    double sum = 0, sum_sq = 0;
    res.min_ms = times[0];
    res.max_ms = times[0];
    for (int r = 0; r < runs; r++) {
        sum += times[r];
        sum_sq += times[r] * times[r];
        if (times[r] < res.min_ms) res.min_ms = times[r];
        if (times[r] > res.max_ms) res.max_ms = times[r];
    }
    res.mean_ms = sum / runs;
    res.std_ms = sqrt((sum_sq / runs) - (res.mean_ms * res.mean_ms));

    /* GFLOPS: 2*M*N*K ops (multiply + add per element) */
    double flops = 2.0 * M * N * K;
    res.gflops = (flops / 1e9) / (res.mean_ms / 1000.0);

    free(times);
    return res;
}

static bench_result_t benchmark_gemm_int8(const void *A_q8, const void *B, float *C,
                                          int M, int N, int K,
                                          int warmup, int runs) {
    bench_result_t res = {0};
    double *times = malloc(runs * sizeof(double));

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        gemm_nt_q5_0_q8_0(A_q8, B, NULL, C, M, N, K);
    }

    /* Benchmark */
    for (int r = 0; r < runs; r++) {
        double start = get_time_ms();
        gemm_nt_q5_0_q8_0(A_q8, B, NULL, C, M, N, K);
        times[r] = get_time_ms() - start;
    }

    /* Compute stats */
    double sum = 0, sum_sq = 0;
    res.min_ms = times[0];
    res.max_ms = times[0];
    for (int r = 0; r < runs; r++) {
        sum += times[r];
        sum_sq += times[r] * times[r];
        if (times[r] < res.min_ms) res.min_ms = times[r];
        if (times[r] > res.max_ms) res.max_ms = times[r];
    }
    res.mean_ms = sum / runs;
    res.std_ms = sqrt((sum_sq / runs) - (res.mean_ms * res.mean_ms));

    /* GFLOPS */
    double flops = 2.0 * M * N * K;
    res.gflops = (flops / 1e9) / (res.mean_ms / 1000.0);

    free(times);
    return res;
}

/* Benchmark single-row GEMV (decode mode) */
static bench_result_t benchmark_gemv_fp32(const float *x, const void *W, float *y,
                                          int N, int K,
                                          int warmup, int runs) {
    bench_result_t res = {0};
    double *times = malloc(runs * sizeof(double));

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        gemv_q5_0(y, W, x, N, K);
    }

    /* Benchmark */
    for (int r = 0; r < runs; r++) {
        double start = get_time_ms();
        gemv_q5_0(y, W, x, N, K);
        times[r] = get_time_ms() - start;
    }

    /* Compute stats */
    double sum = 0, sum_sq = 0;
    res.min_ms = times[0];
    res.max_ms = times[0];
    for (int r = 0; r < runs; r++) {
        sum += times[r];
        sum_sq += times[r] * times[r];
        if (times[r] < res.min_ms) res.min_ms = times[r];
        if (times[r] > res.max_ms) res.max_ms = times[r];
    }
    res.mean_ms = sum / runs;
    res.std_ms = sqrt((sum_sq / runs) - (res.mean_ms * res.mean_ms));

    /* GFLOPS */
    double flops = 2.0 * N * K;
    res.gflops = (flops / 1e9) / (res.mean_ms / 1000.0);

    free(times);
    return res;
}

int main(int argc, char **argv) {
    printf("============================================================\n");
    printf("GEMM vs GEMV Benchmark (Single Core, Q5_0 Weights)\n");
    printf("============================================================\n\n");

    /* Model dimensions (like Qwen2-0.5B) */
    int N = 896;      /* hidden_size / output dim */
    int K = 896;      /* input dim */
    int warmup = 5;
    int runs = 20;

    /* Batch sizes to test */
    int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    printf("Configuration:\n");
    printf("  N (output dim): %d\n", N);
    printf("  K (input dim):  %d\n", K);
    printf("  Warmup runs:    %d\n", warmup);
    printf("  Benchmark runs: %d\n", runs);
    printf("\n");

    /* Allocate weights (shared) */
    void *W = alloc_q5_0(N, K);
    if (!W) {
        fprintf(stderr, "Failed to allocate weights\n");
        return 1;
    }
    init_q5_0_weights(W, N, K);

    /* Print header */
    printf("%-8s | %-12s | %-12s | %-10s | %-10s | %-8s\n",
           "M", "FP32 (ms)", "INT8 (ms)", "FP32 GFLOPS", "INT8 GFLOPS", "Speedup");
    printf("---------|--------------|--------------|------------|------------|--------\n");

    for (int bi = 0; bi < num_batches; bi++) {
        int M = batch_sizes[bi];

        /* Allocate activations */
        float *A_fp32 = aligned_alloc(64, (size_t)M * K * sizeof(float));
        void *A_q8 = alloc_q8_0(M, K);
        float *C = aligned_alloc(64, (size_t)M * N * sizeof(float));

        if (!A_fp32 || !A_q8 || !C) {
            fprintf(stderr, "Failed to allocate for M=%d\n", M);
            continue;
        }

        /* Initialize */
        init_fp32_activations(A_fp32, M, K);
        init_q8_0_activations(A_q8, A_fp32, M, K);
        memset(C, 0, (size_t)M * N * sizeof(float));

        /* Benchmark FP32 path */
        bench_result_t fp32_res = benchmark_gemm_fp32(A_fp32, W, C, M, N, K, warmup, runs);

        /* Benchmark INT8 path */
        bench_result_t int8_res = benchmark_gemm_int8(A_q8, W, C, M, N, K, warmup, runs);

        /* Speedup */
        double speedup = fp32_res.mean_ms / int8_res.mean_ms;

        printf("%-8d | %10.3f   | %10.3f   | %10.2f | %10.2f | %6.2fx\n",
               M, fp32_res.mean_ms, int8_res.mean_ms,
               fp32_res.gflops, int8_res.gflops, speedup);

        free(A_fp32);
        free(A_q8);
        free(C);
    }

    printf("\n");

    /* Additional test: GEMV vs GEMM for M=1 */
    printf("============================================================\n");
    printf("GEMV vs GEMM for M=1 (Decode Mode)\n");
    printf("============================================================\n\n");

    float *x = aligned_alloc(64, K * sizeof(float));
    float *y = aligned_alloc(64, N * sizeof(float));
    float *A1 = aligned_alloc(64, K * sizeof(float));
    void *A1_q8 = alloc_q8_0(1, K);
    float *C1 = aligned_alloc(64, N * sizeof(float));

    init_fp32_activations(x, 1, K);
    memcpy(A1, x, K * sizeof(float));
    init_q8_0_activations(A1_q8, A1, 1, K);

    /* GEMV FP32 */
    bench_result_t gemv_fp32 = benchmark_gemv_fp32(x, W, y, N, K, warmup, runs);

    /* GEMM FP32 (M=1, should call GEMV internally) */
    bench_result_t gemm_fp32_m1 = benchmark_gemm_fp32(A1, W, C1, 1, N, K, warmup, runs);

    /* GEMM INT8 (M=1) */
    bench_result_t gemm_int8_m1 = benchmark_gemm_int8(A1_q8, W, C1, 1, N, K, warmup, runs);

    printf("%-20s | %-12s | %-12s\n", "Kernel", "Time (ms)", "GFLOPS");
    printf("---------------------|--------------|------------\n");
    printf("%-20s | %10.3f   | %10.2f\n", "GEMV FP32", gemv_fp32.mean_ms, gemv_fp32.gflops);
    printf("%-20s | %10.3f   | %10.2f\n", "GEMM FP32 (M=1)", gemm_fp32_m1.mean_ms, gemm_fp32_m1.gflops);
    printf("%-20s | %10.3f   | %10.2f\n", "GEMM INT8 (M=1)", gemm_int8_m1.mean_ms, gemm_int8_m1.gflops);

    printf("\n");
    printf("GEMV vs GEMM M=1 (FP32): %.2fx %s\n",
           gemm_fp32_m1.mean_ms / gemv_fp32.mean_ms,
           gemm_fp32_m1.mean_ms > gemv_fp32.mean_ms ? "(GEMV faster)" : "(GEMM faster)");
    printf("FP32 vs INT8 (M=1):      %.2fx %s\n",
           gemm_fp32_m1.mean_ms / gemm_int8_m1.mean_ms,
           gemm_fp32_m1.mean_ms > gemm_int8_m1.mean_ms ? "(INT8 faster)" : "(FP32 faster)");

    free(x);
    free(y);
    free(A1);
    free(A1_q8);
    free(C1);
    free(W);

    printf("\n============================================================\n");
    printf("Summary:\n");
    printf("- FP32 path: dequantize Q5_0 to FP32, FP32 dot product\n");
    printf("- INT8 path: Q8_0 activations, integer dot product (vec_dot_q5_0_q8_0)\n");
    printf("- For M=1 (decode): memory-bound, GEMV optimized\n");
    printf("- For M>1 (prefill): compute-bound, INT8 batch GEMM benefits\n");
    printf("============================================================\n");

    return 0;
}
