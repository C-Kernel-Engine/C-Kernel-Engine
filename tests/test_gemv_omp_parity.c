/**
 * test_gemv_omp_parity.c - Numerical parity and speed comparison:
 *                           Serial GEMV kernels vs OpenMP-parallel variants
 *
 * Tests all 3 OMP kernels against their serial counterparts:
 *   1. gemv_q8_0_q8_0       vs  gemv_q8_0_q8_0_parallel_omp       (logits)
 *   2. gemv_q5_0_q8_0       vs  gemv_q5_0_q8_0_parallel_omp       (mlp_down)
 *   3. gemv_fused_q5_0_bias vs  gemv_fused_q5_0_bias_parallel_omp  (mlp_gate_up)
 *
 * Validates:
 *   - Bit-exact or near-exact numerical parity (tolerance: 1e-3 abs)
 *   - Speed comparison (serial vs OMP, various thread counts)
 *
 * Build:
 *   make test-gemv-omp
 *
 * Run:
 *   ./build/test_gemv_omp_parity
 *   ./build/test_gemv_omp_parity --quick
 *   ./build/test_gemv_omp_parity --verbose
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "ckernel_quant.h"

/* ============================================================================
 * Timer
 * ============================================================================ */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================================
 * FP16 helper
 * ============================================================================ */

static inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } uf;
    uf.f = f;
    uint32_t u = uf.u;
    uint32_t sign = (u >> 16) & 0x8000;
    int exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | mant;
}

/* ============================================================================
 * Random data generation
 * ============================================================================ */

static void fill_random_float(float *arr, int n, float scale) {
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

static void quantize_to_q5_0(const float *src, void *dst, int M, int K) {
    block_q5_0 *blocks = (block_q5_0 *)dst;
    const int blocks_per_row = K / QK5_0;

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            const float *wp = &src[row * K + b * QK5_0];
            block_q5_0 *blk = &blocks[row * blocks_per_row + b];

            float amax = 0.0f;
            for (int i = 0; i < QK5_0; i++) {
                float a = fabsf(wp[i]);
                if (a > amax) amax = a;
            }

            float d = amax / 15.0f;
            float id = (d != 0) ? 1.0f / d : 0.0f;
            blk->d = fp32_to_fp16(d);

            uint32_t qh = 0;
            for (int j = 0; j < QK5_0 / 2; j++) {
                int q0 = (int)(wp[j] * id + 16.5f);
                int q1 = (int)(wp[j + 16] * id + 16.5f);
                if (q0 < 0) q0 = 0; if (q0 > 31) q0 = 31;
                if (q1 < 0) q1 = 0; if (q1 > 31) q1 = 31;
                blk->qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                if (q0 & 0x10) qh |= (1 << j);
                if (q1 & 0x10) qh |= (1 << (j + 12));
            }
            memcpy(blk->qh, &qh, sizeof(qh));
        }
    }
}

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
 * Kernel declarations (serial)
 * ============================================================================ */

extern void gemv_q8_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q5_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_fused_q5_0_bias_dispatch(float *y, const void *W, const float *x,
                                           const float *bias, int M, int K);

/* Kernel declarations (OMP) */
extern void gemv_q8_0_q8_0_parallel_omp(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q5_0_q8_0_parallel_omp(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_fused_q5_0_bias_parallel_omp(float *y, const void *W, const float *x,
                                      const float *bias, int M, int K);

/* ============================================================================
 * Parity check
 * ============================================================================ */

typedef struct {
    float max_abs_diff;
    float max_rel_diff;
    int   max_diff_idx;
    int   num_diffs;    /* count of elements with abs diff > tolerance */
} parity_result_t;

static parity_result_t check_parity(const float *ref, const float *test, int n,
                                     float abs_tol) {
    parity_result_t r = {0};
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > r.max_abs_diff) {
            r.max_abs_diff = diff;
            r.max_diff_idx = i;
        }
        float ref_abs = fabsf(ref[i]);
        if (ref_abs > 1e-6f) {
            float rel = diff / ref_abs;
            if (rel > r.max_rel_diff) r.max_rel_diff = rel;
        }
        if (diff > abs_tol) r.num_diffs++;
    }
    return r;
}

/* ============================================================================
 * Test dimensions (Qwen2 0.5B model shapes)
 * ============================================================================ */

typedef struct {
    const char *name;
    int M;
    int K;
} test_config_t;

static test_config_t configs_default[] = {
    {"qkv_proj",     896,    896},
    {"mlp_gate_up",  9728,   896},
    {"mlp_down",     896,    4864},
    {"logits",       151936, 896},
    {NULL, 0, 0}
};

static test_config_t configs_quick[] = {
    {"small",   256,  256},
    {"medium",  896,  896},
    {"mlp",     4864, 896},
    {NULL, 0, 0}
};

/* ============================================================================
 * Benchmark helper
 * ============================================================================ */

typedef void (*gemv_q_fn)(float *, const void *, const void *, int, int);
typedef void (*gemv_fused_fn)(float *, const void *, const float *, const float *, int, int);

static double bench_gemv_q(gemv_q_fn fn, float *y, const void *W,
                            const void *x, int M, int K,
                            int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn(y, W, x, M, K);

    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++) fn(y, W, x, M, K);
    double t1 = get_time_ms();

    return (t1 - t0) / iters;
}

static double bench_gemv_fused(gemv_fused_fn fn, float *y, const void *W,
                                const float *x, const float *bias, int M, int K,
                                int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn(y, W, x, bias, M, K);

    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++) fn(y, W, x, bias, M, K);
    double t1 = get_time_ms();

    return (t1 - t0) / iters;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv) {
    int quick   = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0)   quick = 1;
        if (strcmp(argv[i], "--verbose") == 0) verbose = 1;
    }

    int warmup = quick ? 2  : 5;
    int iters  = quick ? 10 : 50;
    float abs_tol = 1e-3f;

    test_config_t *configs = quick ? configs_quick : configs_default;

    int max_threads = omp_get_max_threads();

    printf("================================================================================\n");
    printf("  GEMV Serial vs OpenMP Parity & Speed Test\n");
    printf("================================================================================\n");
    printf("  OpenMP threads:    %d\n", max_threads);
    printf("  Warmup:            %d\n", warmup);
    printf("  Iterations:        %d\n", iters);
    printf("  Abs tolerance:     %.0e\n", abs_tol);
    printf("  Mode:              %s\n", quick ? "quick" : "full");
    printf("================================================================================\n\n");

    int total_pass = 0, total_fail = 0;

    /* =========================================================================
     * Test 1: gemv_q8_0_q8_0 vs gemv_q8_0_q8_0_parallel_omp
     * ========================================================================= */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  TEST 1: gemv_q8_0_q8_0 (serial) vs gemv_q8_0_q8_0_parallel_omp (parallel)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    printf("%-15s %8s %8s  %10s %10s %8s  %s\n",
           "Config", "M", "K", "Serial(ms)", "OMP(ms)", "Speedup", "Parity");
    printf("--------------------------------------------------------------------------------\n");

    for (int c = 0; configs[c].name; c++) {
        int M = configs[c].M, K = configs[c].K;
        if (K % QK8_0 != 0) continue;

        /* Allocate */
        float *W_fp32 = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32 = (float *)malloc(K * sizeof(float));
        float *y_serial = (float *)calloc(M, sizeof(float));
        float *y_omp    = (float *)calloc(M, sizeof(float));

        int bpr_w = K / QK8_0;
        void *W_q8 = malloc((size_t)M * bpr_w * sizeof(block_q8_0));
        void *x_q8 = malloc((size_t)(K / QK8_0) * sizeof(block_q8_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);
        quantize_to_q8_0(W_fp32, W_q8, M, K);
        quantize_to_q8_0(x_fp32, x_q8, 1, K);

        /* Run serial */
        gemv_q8_0_q8_0(y_serial, W_q8, x_q8, M, K);

        /* Run OMP */
        gemv_q8_0_q8_0_parallel_omp(y_omp, W_q8, x_q8, M, K);

        /* Parity */
        parity_result_t pr = check_parity(y_serial, y_omp, M, abs_tol);
        int pass = (pr.max_abs_diff <= abs_tol);
        if (pass) total_pass++; else total_fail++;

        /* Benchmark */
        double t_serial = bench_gemv_q(gemv_q8_0_q8_0, y_serial, W_q8, x_q8, M, K, warmup, iters);
        double t_omp    = bench_gemv_q(gemv_q8_0_q8_0_parallel_omp, y_omp, W_q8, x_q8, M, K, warmup, iters);
        double speedup  = t_serial / t_omp;

        printf("%-15s %8d %8d  %10.3f %10.3f %7.2fx  %s",
               configs[c].name, M, K, t_serial, t_omp, speedup,
               pass ? "PASS" : "FAIL");
        if (!pass || verbose) {
            printf("  (max_abs=%.2e, max_rel=%.2e, diffs=%d)",
                   pr.max_abs_diff, pr.max_rel_diff, pr.num_diffs);
        }
        printf("\n");

        free(W_fp32); free(x_fp32); free(y_serial); free(y_omp);
        free(W_q8); free(x_q8);
    }

    /* =========================================================================
     * Test 2: gemv_q5_0_q8_0 vs gemv_q5_0_q8_0_parallel_omp
     * ========================================================================= */
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  TEST 2: gemv_q5_0_q8_0 (serial) vs gemv_q5_0_q8_0_parallel_omp (parallel)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    printf("%-15s %8s %8s  %10s %10s %8s  %s\n",
           "Config", "M", "K", "Serial(ms)", "OMP(ms)", "Speedup", "Parity");
    printf("--------------------------------------------------------------------------------\n");

    for (int c = 0; configs[c].name; c++) {
        int M = configs[c].M, K = configs[c].K;
        if (K % QK5_0 != 0 || K % QK8_0 != 0) continue;

        /* Allocate */
        float *W_fp32 = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32 = (float *)malloc(K * sizeof(float));
        float *y_serial = (float *)calloc(M, sizeof(float));
        float *y_omp    = (float *)calloc(M, sizeof(float));

        int bpr_w = K / QK5_0;
        void *W_q5 = malloc((size_t)M * bpr_w * sizeof(block_q5_0));
        void *x_q8 = malloc((size_t)(K / QK8_0) * sizeof(block_q8_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);
        quantize_to_q5_0(W_fp32, W_q5, M, K);
        quantize_to_q8_0(x_fp32, x_q8, 1, K);

        /* Run serial */
        gemv_q5_0_q8_0(y_serial, W_q5, x_q8, M, K);

        /* Run OMP */
        gemv_q5_0_q8_0_parallel_omp(y_omp, W_q5, x_q8, M, K);

        /* Parity */
        parity_result_t pr = check_parity(y_serial, y_omp, M, abs_tol);
        int pass = (pr.max_abs_diff <= abs_tol);
        if (pass) total_pass++; else total_fail++;

        /* Benchmark */
        double t_serial = bench_gemv_q(gemv_q5_0_q8_0, y_serial, W_q5, x_q8, M, K, warmup, iters);
        double t_omp    = bench_gemv_q(gemv_q5_0_q8_0_parallel_omp, y_omp, W_q5, x_q8, M, K, warmup, iters);
        double speedup  = t_serial / t_omp;

        printf("%-15s %8d %8d  %10.3f %10.3f %7.2fx  %s",
               configs[c].name, M, K, t_serial, t_omp, speedup,
               pass ? "PASS" : "FAIL");
        if (!pass || verbose) {
            printf("  (max_abs=%.2e, max_rel=%.2e, diffs=%d)",
                   pr.max_abs_diff, pr.max_rel_diff, pr.num_diffs);
        }
        printf("\n");

        free(W_fp32); free(x_fp32); free(y_serial); free(y_omp);
        free(W_q5); free(x_q8);
    }

    /* =========================================================================
     * Test 3: gemv_fused_q5_0_bias_dispatch vs gemv_fused_q5_0_bias_parallel_omp
     * ========================================================================= */
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  TEST 3: gemv_fused_q5_0_bias_dispatch (serial) vs gemv_fused_q5_0_bias_parallel_omp (parallel)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    printf("%-15s %8s %8s  %10s %10s %8s  %s\n",
           "Config", "M", "K", "Serial(ms)", "OMP(ms)", "Speedup", "Parity");
    printf("--------------------------------------------------------------------------------\n");

    for (int c = 0; configs[c].name; c++) {
        int M = configs[c].M, K = configs[c].K;
        if (K % QK5_0 != 0) continue;

        /* Allocate */
        float *W_fp32  = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32  = (float *)malloc(K * sizeof(float));
        float *bias    = (float *)malloc(M * sizeof(float));
        float *y_serial = (float *)calloc(M, sizeof(float));
        float *y_omp    = (float *)calloc(M, sizeof(float));

        int bpr_w = K / QK5_0;
        void *W_q5 = malloc((size_t)M * bpr_w * sizeof(block_q5_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);
        fill_random_float(bias, M, 0.01f);
        quantize_to_q5_0(W_fp32, W_q5, M, K);

        /* Run serial */
        gemv_fused_q5_0_bias_dispatch(y_serial, W_q5, x_fp32, bias, M, K);

        /* Run OMP */
        gemv_fused_q5_0_bias_parallel_omp(y_omp, W_q5, x_fp32, bias, M, K);

        /* Parity */
        parity_result_t pr = check_parity(y_serial, y_omp, M, abs_tol);
        int pass = (pr.max_abs_diff <= abs_tol);
        if (pass) total_pass++; else total_fail++;

        /* Benchmark */
        double t_serial = bench_gemv_fused(gemv_fused_q5_0_bias_dispatch,
                                            y_serial, W_q5, x_fp32, bias, M, K, warmup, iters);
        double t_omp    = bench_gemv_fused(gemv_fused_q5_0_bias_parallel_omp,
                                            y_omp, W_q5, x_fp32, bias, M, K, warmup, iters);
        double speedup  = t_serial / t_omp;

        printf("%-15s %8d %8d  %10.3f %10.3f %7.2fx  %s",
               configs[c].name, M, K, t_serial, t_omp, speedup,
               pass ? "PASS" : "FAIL");
        if (!pass || verbose) {
            printf("  (max_abs=%.2e, max_rel=%.2e, diffs=%d)",
                   pr.max_abs_diff, pr.max_rel_diff, pr.num_diffs);
        }
        printf("\n");

        free(W_fp32); free(x_fp32); free(bias); free(y_serial); free(y_omp);
        free(W_q5);
    }

    /* =========================================================================
     * Test 4: Fused with NULL bias (edge case)
     * ========================================================================= */
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  TEST 4: gemv_fused_q5_0_bias_parallel_omp with NULL bias\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    {
        int M = 896, K = 896;
        float *W_fp32 = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32 = (float *)malloc(K * sizeof(float));
        float *y_serial = (float *)calloc(M, sizeof(float));
        float *y_omp    = (float *)calloc(M, sizeof(float));
        void *W_q5 = malloc((size_t)M * (K / QK5_0) * sizeof(block_q5_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);
        quantize_to_q5_0(W_fp32, W_q5, M, K);

        gemv_fused_q5_0_bias_dispatch(y_serial, W_q5, x_fp32, NULL, M, K);
        gemv_fused_q5_0_bias_parallel_omp(y_omp, W_q5, x_fp32, NULL, M, K);

        parity_result_t pr = check_parity(y_serial, y_omp, M, abs_tol);
        int pass = (pr.max_abs_diff <= abs_tol);
        if (pass) total_pass++; else total_fail++;

        printf("  NULL bias test (M=%d, K=%d): %s", M, K, pass ? "PASS" : "FAIL");
        if (!pass || verbose) {
            printf("  (max_abs=%.2e, max_rel=%.2e)", pr.max_abs_diff, pr.max_rel_diff);
        }
        printf("\n");

        free(W_fp32); free(x_fp32); free(y_serial); free(y_omp); free(W_q5);
    }

    /* =========================================================================
     * Test 5: Thread scaling (OMP_NUM_THREADS sweep)
     * ========================================================================= */
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  TEST 5: Thread scaling for gemv_q8_0_q8_0_parallel_omp (logits M=151936, K=896)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    if (!quick) {
        int M = 151936, K = 896;

        float *W_fp32 = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32 = (float *)malloc(K * sizeof(float));
        float *y = (float *)calloc(M, sizeof(float));
        void *W_q8 = malloc((size_t)M * (K / QK8_0) * sizeof(block_q8_0));
        void *x_q8 = malloc((size_t)(K / QK8_0) * sizeof(block_q8_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);
        quantize_to_q8_0(W_fp32, W_q8, M, K);
        quantize_to_q8_0(x_fp32, x_q8, 1, K);

        /* Serial baseline */
        double t_serial = bench_gemv_q(gemv_q8_0_q8_0, y, W_q8, x_q8, M, K, warmup, iters);
        printf("  Serial:          %10.3f ms\n", t_serial);

        int thread_counts[] = {1, 2, 4, 8, 16};
        int n_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

        for (int t = 0; t < n_counts; t++) {
            int nth = thread_counts[t];
            if (nth > max_threads) break;

            omp_set_num_threads(nth);
            double t_omp = bench_gemv_q(gemv_q8_0_q8_0_parallel_omp, y, W_q8, x_q8, M, K, warmup, iters);
            printf("  OMP nth=%-2d:      %10.3f ms  (%.2fx speedup)\n",
                   nth, t_omp, t_serial / t_omp);
        }

        /* Restore default threads */
        omp_set_num_threads(max_threads);

        free(W_fp32); free(x_fp32); free(y); free(W_q8); free(x_q8);
    } else {
        printf("  (skipped in --quick mode)\n");
    }

    /* =========================================================================
     * Summary
     * ========================================================================= */
    printf("\n================================================================================\n");
    printf("  SUMMARY\n");
    printf("================================================================================\n");
    printf("  Passed: %d\n", total_pass);
    printf("  Failed: %d\n", total_fail);
    printf("================================================================================\n");

    if (total_fail > 0) {
        printf("\n  *** PARITY FAILURE: OMP kernels diverge from serial! ***\n\n");
        return 1;
    }

    printf("\n  All OMP kernels match serial output within tolerance.\n\n");
    return 0;
}
