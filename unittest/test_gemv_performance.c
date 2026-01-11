/**
 * test_gemv_performance.c - Comprehensive GEMV kernel benchmark
 *
 * Tests all quantized GEMV kernels with realistic model dimensions.
 * Measures throughput (GFLOPS), latency (ms), and compares against
 * theoretical peak and llama.cpp reference numbers.
 *
 * Build:
 *   gcc -O3 -mavx -march=native -fopenmp -I include \
 *       unittest/test_gemv_performance.c \
 *       src/kernels/gemm_kernels_q5_0.c \
 *       src/kernels/gemm_kernels_q5_0_sse_v2.c \
 *       src/kernels/gemm_kernels_q8_0.c \
 *       src/kernels/gemm_kernels_q4k.c \
 *       src/kernels/gemm_kernels_q6k.c \
 *       src/kernels/dequant_kernels.c \
 *       src/kernels/quantize_row_q8_k_sse.c \
 *       -o build/test_gemv_performance -lm
 *
 * Run:
 *   ./build/test_gemv_performance
 *   ./build/test_gemv_performance --quick     # Fast test
 *   ./build/test_gemv_performance --large     # Large dimensions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "ckernel_quant.h"

/* ============================================================================
 * Timer helpers
 * ============================================================================ */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================================
 * FP16 helpers
 * ============================================================================ */

static inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } uf;
    uf.f = f;
    uint32_t u = uf.u;
    uint32_t sign = (u >> 16) & 0x8000;
    int exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FF;

    if (exp <= 0) {
        return sign;
    } else if (exp >= 31) {
        return sign | 0x7C00;
    }
    return sign | (exp << 10) | mant;
}

/* ============================================================================
 * Random weight/activation generation
 * ============================================================================ */

static void fill_random_float(float *arr, int n, float scale) {
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

/* Quantize to Q5_0 format */
static void quantize_to_q5_0(const float *src, void *dst, int M, int K) {
    block_q5_0 *blocks = (block_q5_0 *)dst;
    const int blocks_per_row = K / QK5_0;

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            const float *wp = &src[row * K + b * QK5_0];
            block_q5_0 *blk = &blocks[row * blocks_per_row + b];

            /* Find max for scale */
            float amax = 0.0f;
            for (int i = 0; i < QK5_0; i++) {
                float a = fabsf(wp[i]);
                if (a > amax) amax = a;
            }

            float d = amax / 15.0f;
            float id = (d != 0) ? 1.0f / d : 0.0f;
            blk->d = fp32_to_fp16(d);

            /* Quantize */
            uint32_t qh = 0;
            for (int j = 0; j < QK5_0 / 2; j++) {
                int q0 = (int)(wp[j] * id + 16.5f);
                int q1 = (int)(wp[j + 16] * id + 16.5f);
                if (q0 < 0) q0 = 0;
                if (q0 > 31) q0 = 31;
                if (q1 < 0) q1 = 0;
                if (q1 > 31) q1 = 31;

                blk->qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                if (q0 & 0x10) qh |= (1 << j);
                if (q1 & 0x10) qh |= (1 << (j + 12));
            }
            memcpy(blk->qh, &qh, sizeof(qh));
        }
    }
}

/* Quantize to Q8_0 format */
static void quantize_to_q8_0(const float *src, void *dst, int M, int K) {
    block_q8_0 *blocks = (block_q8_0 *)dst;
    const int blocks_per_row = K / QK8_0;

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            const float *wp = &src[row * K + b * QK8_0];
            block_q8_0 *blk = &blocks[row * blocks_per_row + b];

            float amax = 0.0f;
            for (int i = 0; i < QK8_0; i++) {
                float a = fabsf(wp[i]);
                if (a > amax) amax = a;
            }

            float d = amax / 127.0f;
            float id = (d != 0) ? 1.0f / d : 0.0f;
            blk->d = fp32_to_fp16(d);

            for (int i = 0; i < QK8_0; i++) {
                int q = (int)(wp[i] * id + 0.5f);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                blk->qs[i] = q;
            }
        }
    }
}

/* ============================================================================
 * Kernel declarations
 * ============================================================================ */

void gemv_q5_0(float *y, const void *W, const float *x, int M, int K);
void gemv_q8_0(float *y, const void *W, const float *x, int M, int K);
void gemv_q4_k(float *y, const void *W, const float *x, int M, int K);
void gemv_q6_k(float *y, const void *W, const float *x, int M, int K);

/* ============================================================================
 * Benchmark runner
 * ============================================================================ */

typedef void (*gemv_fn)(float *, const void *, const float *, int, int);

typedef struct {
    const char *name;
    gemv_fn func;
    int block_size;
    int bytes_per_block;
    void (*quantize)(const float *, void *, int, int);
} kernel_info_t;

static double benchmark_kernel(kernel_info_t *kernel, int M, int K,
                               int warmup, int iterations) {
    /* Allocate memory */
    float *W_fp32 = (float *)malloc(M * K * sizeof(float));
    float *x = (float *)malloc(K * sizeof(float));
    float *y = (float *)malloc(M * sizeof(float));

    int blocks_per_row = K / kernel->block_size;
    size_t W_size = (size_t)M * blocks_per_row * kernel->bytes_per_block;
    void *W = malloc(W_size);

    /* Initialize with random data */
    srand(42);
    fill_random_float(W_fp32, M * K, 0.1f);
    fill_random_float(x, K, 1.0f);

    /* Quantize weights */
    if (kernel->quantize) {
        kernel->quantize(W_fp32, W, M, K);
    } else {
        memset(W, 0, W_size);
    }

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        kernel->func(y, W, x, M, K);
    }

    /* Timed runs */
    double t0 = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        kernel->func(y, W, x, M, K);
    }
    double t1 = get_time_ms();

    double time_ms = (t1 - t0) / iterations;

    /* Cleanup */
    free(W_fp32);
    free(x);
    free(y);
    free(W);

    return time_ms;
}

/* ============================================================================
 * Test configurations
 * ============================================================================ */

typedef struct {
    const char *name;
    int M;
    int K;
} test_config_t;

static test_config_t small_configs[] = {
    {"qkv_proj", 896, 896},
    {"mlp_up", 4864, 896},
    {"mlp_down", 896, 4864},
    {NULL, 0, 0}
};

static test_config_t large_configs[] = {
    {"qkv_proj_7b", 4096, 4096},
    {"mlp_up_7b", 11008, 4096},
    {"mlp_down_7b", 4096, 11008},
    {"embed_7b", 32000, 4096},
    {NULL, 0, 0}
};

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv) {
    int quick = 0;
    int large = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
        if (strcmp(argv[i], "--large") == 0) large = 1;
    }

    int warmup = quick ? 2 : 5;
    int iterations = quick ? 10 : 50;

    printf("================================================================================\n");
    printf("  GEMV KERNEL PERFORMANCE BENCHMARK\n");
    printf("================================================================================\n\n");

    /* CPU Info */
    printf("Configuration:\n");
    printf("  Warmup iterations:  %d\n", warmup);
    printf("  Timed iterations:   %d\n", iterations);
    printf("\n");

    /* Define kernels to test */
    kernel_info_t kernels[] = {
        {"Q5_0", gemv_q5_0, QK5_0, sizeof(block_q5_0), quantize_to_q5_0},
        {"Q8_0", gemv_q8_0, QK8_0, sizeof(block_q8_0), quantize_to_q8_0},
        /* Q4_K and Q6_K have different block sizes, skip for now */
        {NULL, NULL, 0, 0, NULL}
    };

    test_config_t *configs = large ? large_configs : small_configs;

    printf("%-12s %-15s %10s %10s %10s %10s\n",
           "Kernel", "Config", "M", "K", "Time(ms)", "GFLOPS");
    printf("--------------------------------------------------------------------------------\n");

    double total_gflops = 0;
    int num_tests = 0;

    for (int k = 0; kernels[k].name; k++) {
        for (int c = 0; configs[c].name; c++) {
            int M = configs[c].M;
            int K = configs[c].K;

            /* Skip if not aligned */
            if (K % kernels[k].block_size != 0) continue;

            double time_ms = benchmark_kernel(&kernels[k], M, K, warmup, iterations);

            /* GFLOPS: 2*M*K FLOPs (multiply-add) */
            double flops = 2.0 * M * K;
            double gflops = (flops / 1e9) / (time_ms / 1000.0);

            printf("%-12s %-15s %10d %10d %10.3f %10.2f\n",
                   kernels[k].name, configs[c].name, M, K, time_ms, gflops);

            total_gflops += gflops;
            num_tests++;
        }
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("Average GFLOPS: %.2f\n\n", total_gflops / num_tests);

    /* Estimate tokens/s for Qwen 0.5B */
    /* Decode uses mostly Q5_0 GEMV: ~324 ops/token, average shape ~2000x2000 */
    double avg_time_per_gemv_ms = 0;
    for (int c = 0; configs[c].name; c++) {
        int M = configs[c].M;
        int K = configs[c].K;
        if (K % QK5_0 == 0) {
            avg_time_per_gemv_ms += benchmark_kernel(&kernels[0], M, K, 2, 10);
        }
    }
    avg_time_per_gemv_ms /= 3;  /* 3 config types */

    double gemv_per_token = 324;  /* Approximate for Qwen 0.5B */
    double ms_per_token = avg_time_per_gemv_ms * gemv_per_token;
    double tok_per_s = 1000.0 / ms_per_token;

    printf("================================================================================\n");
    printf("  DECODE THROUGHPUT ESTIMATE\n");
    printf("================================================================================\n");
    printf("  Average GEMV time:    %.3f ms\n", avg_time_per_gemv_ms);
    printf("  GEMVs per token:      %.0f (estimated)\n", gemv_per_token);
    printf("  Estimated decode:     %.2f tok/s\n", tok_per_s);
    printf("  llama.cpp reference:  ~35 tok/s\n");
    printf("  Gap:                  %.1fx\n", 35.0 / tok_per_s);
    printf("================================================================================\n");

    return 0;
}
