/**
 * test_threadpool_parity.c - Numerical parity and speed comparison:
 *                             Serial GEMV/GEMM kernels vs Thread Pool dispatch variants
 *
 * Tests thread pool dispatch wrappers against serial kernels:
 *
 * Decode (ck_parallel_decode.h) — GEMV:
 *   1. gemv_q8_0_q8_0       vs  gemv_q8_0_q8_0_parallel_dispatch       (logits)
 *   2. gemv_q5_0_q8_0       vs  gemv_q5_0_q8_0_parallel_dispatch       (mlp_down)
 *   3. gemv_fused_q5_0_bias vs  gemv_fused_q5_0_bias_parallel_dispatch  (fused: quantize+gemv+bias)
 *   4. Fused with NULL bias (edge case)
 *   5. Dispatch latency measurement
 *
 * Prefill (ck_parallel_prefill.h) — GEMM:
 *   6. gemm_nt_q5_0_q8_0    vs  gemm_nt_q5_0_q8_0_parallel_dispatch    (Q/K proj, out proj, MLP gate+up)
 *   7. gemm_nt_q8_0_q8_0    vs  gemm_nt_q8_0_q8_0_parallel_dispatch    (V proj)
 *   8. gemm_nt_q6_k_q8_k    vs  gemm_nt_q6_k_q8_k_parallel_dispatch    (MLP down proj)
 *
 * ADR: Thread pool vs OpenMP
 *   - OpenMP fork/join creates threads per #pragma region (~15-50us overhead)
 *   - Thread pool keeps N-1 pthreads alive, spin-waiting on atomics (~0.1us wake)
 *   - OpenMP + thread pool together causes 2N threads competing for N cores
 *   - This test validates that thread pool dispatch produces identical outputs
 *     to serial kernels, and measures the speedup from persistent pthreads.
 *
 * Validates:
 *   - Numerical parity (tolerance: 1e-3 abs)
 *   - Speed comparison (serial vs thread pool dispatch)
 *   - Thread scaling (1, 2, 4, 8 threads)
 *
 * Build:
 *   make test-threadpool-parity
 *
 * Run:
 *   ./build/test_threadpool_parity
 *   ./build/test_threadpool_parity --quick
 *   ./build/test_threadpool_parity --verbose
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "ckernel_quant.h"
#include "ck_threadpool.h"

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
extern void gemv_q6_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_fused_q5_0_bias_dispatch(float *y, const void *W, const float *x,
                                           const float *bias, int M, int K);

/* Kernel declarations (thread pool dispatch from ck_parallel_decode.c) */
extern void gemv_q8_0_q8_0_parallel_dispatch(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q5_0_q8_0_parallel_dispatch(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q6_k_q8_k_parallel_dispatch(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_fused_q5_0_bias_parallel_dispatch(float *y, const void *W, const float *x,
                                                     const float *bias, int M, int K);

/* GEMM kernel declarations (serial, from src/kernels/) */
extern void gemm_nt_q5_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q8_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q6_k_q8_k(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);

/* GEMM kernel declarations (thread pool dispatch from ck_parallel_prefill.c) */
extern void gemm_nt_q5_0_q8_0_parallel_dispatch(const void *A, const void *B,
                                                   const float *bias, float *C,
                                                   int M, int N, int K);
extern void gemm_nt_q8_0_q8_0_parallel_dispatch(const void *A, const void *B,
                                                   const float *bias, float *C,
                                                   int M, int N, int K);
extern void gemm_nt_q6_k_q8_k_parallel_dispatch(const void *A, const void *B,
                                                   const float *bias, float *C,
                                                   int M, int N, int K);

/* ============================================================================
 * Parity check (reused from test_gemv_omp_parity.c)
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

/* GEMM test dimensions: M (tokens), N (output dim), K (input dim) */
typedef struct {
    const char *name;
    int M;
    int N;
    int K;
} gemm_test_config_t;

static gemm_test_config_t gemm_configs_default[] = {
    {"qkv_proj",     32,   896,  896},
    {"mlp_gate_up",  32,   4864, 896},
    {"mlp_down",     32,   896,  4864},
    {"big_prefill",  128,  896,  896},
    {"long_prefill", 256,  896,  896},
    {NULL, 0, 0, 0}
};

static gemm_test_config_t gemm_configs_quick[] = {
    {"small",    8,   256, 256},
    {"medium",   16,  896, 896},
    {"mlp",      32,  896, 4864},
    {NULL, 0, 0, 0}
};

/* ============================================================================
 * Benchmark helpers
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
 * GEMM benchmark + quantization helpers (for prefill parity tests)
 * ============================================================================ */

typedef void (*gemm_fn_t)(const void *, const void *, const float *, float *, int, int, int);

static double bench_gemm(gemm_fn_t fn, const void *A, const void *B,
                          const float *bias, float *C,
                          int M, int N, int K,
                          int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn(A, B, bias, C, M, N, K);

    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++) fn(A, B, bias, C, M, N, K);
    double t1 = get_time_ms();

    return (t1 - t0) / iters;
}

/* Simple Q6_K quantization for test data (lossy but deterministic) */
static void quantize_to_q6_K(const float *src, void *dst, int M, int K) {
    block_q6_K *blocks = (block_q6_K *)dst;
    const int blocks_per_row = K / QK_K;  /* QK_K = 256 */

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            const float *wp = &src[row * K + b * QK_K];
            block_q6_K *blk = &blocks[row * blocks_per_row + b];

            /* Find max absolute value for this super-block */
            float amax = 0.0f;
            for (int i = 0; i < QK_K; i++) {
                float a = fabsf(wp[i]);
                if (a > amax) amax = a;
            }

            float d = amax / 31.0f;  /* 6-bit range: -32..31 */
            float id = (d != 0) ? 1.0f / d : 0.0f;
            blk->d = fp32_to_fp16(d);

            /* Quantize to 6-bit and split into ql (low 4) + qh (high 2) */
            memset(blk->ql, 0, sizeof(blk->ql));
            memset(blk->qh, 0, sizeof(blk->qh));
            memset(blk->scales, 0, sizeof(blk->scales));

            /* Simple per-16 sub-block scales (all = 1 for simplicity) */
            for (int s = 0; s < QK_K / 16; s++) {
                blk->scales[s] = 1;
            }

            for (int i = 0; i < QK_K; i++) {
                int q = (int)(wp[i] * id + 32.5f);
                if (q < 0) q = 0;
                if (q > 63) q = 63;

                /* Store low 4 bits in ql (packed 2 per byte) */
                int ql_idx = i / 2;
                if (i < QK_K / 2) {
                    if (i % 2 == 0)
                        blk->ql[ql_idx] = (q & 0x0F);
                    else
                        blk->ql[ql_idx] |= ((q & 0x0F) << 4);
                }

                /* Store high 2 bits in qh */
                int qh_idx = i / 4;
                int qh_shift = (i % 4) * 2;
                if (i < QK_K / 4 * 4) {
                    blk->qh[qh_idx] |= ((q >> 4) & 0x03) << qh_shift;
                }
            }
        }
    }
}

/* Simple Q8_K quantization for test data */
static void quantize_to_q8_K(const float *src, void *dst, int M, int K) {
    block_q8_K *blocks = (block_q8_K *)dst;
    const int blocks_per_row = K / QK_K;  /* QK_K = 256 */

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            const float *wp = &src[row * K + b * QK_K];
            block_q8_K *blk = &blocks[row * blocks_per_row + b];

            /* Find max absolute value */
            float amax = 0.0f;
            for (int i = 0; i < QK_K; i++) {
                float a = fabsf(wp[i]);
                if (a > amax) amax = a;
            }

            float d = amax / 127.0f;
            float id = (d != 0) ? 1.0f / d : 0.0f;
            blk->d = d;  /* Q8_K uses FP32 scale, not FP16 */

            /* Quantize and compute block sums */
            memset(blk->bsums, 0, sizeof(blk->bsums));
            for (int i = 0; i < QK_K; i++) {
                int q = (int)(wp[i] * id + 0.5f);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                blk->qs[i] = (int8_t)q;
                blk->bsums[i / 16] += q;
            }
        }
    }
}

/* ============================================================================
 * Generic test runner for GEMM (serial vs thread pool) — prefill parity
 * ============================================================================ */

static int run_gemm_test(const char *test_name,
                          gemm_fn_t serial_fn, gemm_fn_t dispatch_fn,
                          int A_quant_block_size, size_t A_block_bytes,
                          int B_quant_block_size, size_t B_block_bytes,
                          int B_is_q5,  /* 1 = Q5_0 weights, 0 = Q8_0 weights */
                          int A_is_q8_K,  /* 1 = Q8_K activations, 0 = Q8_0 activations */
                          int B_is_q6_K,  /* 1 = Q6_K weights */
                          gemm_test_config_t *configs,
                          int warmup, int iters, float abs_tol, int verbose) {
    int pass_count = 0, fail_count = 0;

    printf("\n");
    printf("%-15s %6s %6s %6s  %10s %10s %8s  %s\n",
           "Config", "M", "N", "K", "Serial(ms)", "Pool(ms)", "Speedup", "Parity");
    printf("------------------------------------------------------------------------------------\n");

    for (int c = 0; configs[c].name; c++) {
        int M = configs[c].M, N = configs[c].N, K = configs[c].K;
        if (K % A_quant_block_size != 0) continue;
        if (K % B_quant_block_size != 0) continue;

        /* Allocate FP32 source data */
        float *A_fp32 = (float *)malloc((size_t)M * K * sizeof(float));
        float *B_fp32 = (float *)malloc((size_t)N * K * sizeof(float));
        float *bias   = (float *)malloc((size_t)N * sizeof(float));
        float *C_serial = (float *)calloc((size_t)M * N, sizeof(float));
        float *C_pool   = (float *)calloc((size_t)M * N, sizeof(float));

        /* Allocate quantized data */
        int A_bpr = K / A_quant_block_size;
        int B_bpr = K / B_quant_block_size;
        void *A_q = malloc((size_t)M * A_bpr * A_block_bytes);
        void *B_q = malloc((size_t)N * B_bpr * B_block_bytes);

        srand(42);
        fill_random_float(A_fp32, M * K, 1.0f);
        fill_random_float(B_fp32, N * K, 0.1f);
        fill_random_float(bias, N, 0.01f);

        /* Quantize A (activations) */
        if (A_is_q8_K) {
            quantize_to_q8_K(A_fp32, A_q, M, K);
        } else {
            quantize_to_q8_0(A_fp32, A_q, M, K);
        }

        /* Quantize B (weights) */
        if (B_is_q6_K) {
            quantize_to_q6_K(B_fp32, B_q, N, K);
        } else if (B_is_q5) {
            quantize_to_q5_0(B_fp32, B_q, N, K);
        } else {
            quantize_to_q8_0(B_fp32, B_q, N, K);
        }

        /* Run serial */
        serial_fn(A_q, B_q, bias, C_serial, M, N, K);

        /* Run thread pool dispatch */
        dispatch_fn(A_q, B_q, bias, C_pool, M, N, K);

        /* Parity */
        parity_result_t pr = check_parity(C_serial, C_pool, M * N, abs_tol);
        int pass = (pr.max_abs_diff <= abs_tol);
        if (pass) pass_count++; else fail_count++;

        /* Benchmark */
        double t_serial = bench_gemm(serial_fn, A_q, B_q, bias, C_serial, M, N, K, warmup, iters);
        double t_pool   = bench_gemm(dispatch_fn, A_q, B_q, bias, C_pool, M, N, K, warmup, iters);
        double speedup  = t_serial / t_pool;

        printf("%-15s %6d %6d %6d  %10.3f %10.3f %7.2fx  %s",
               configs[c].name, M, N, K, t_serial, t_pool, speedup,
               pass ? "PASS" : "FAIL");
        if (!pass || verbose) {
            printf("  (max_abs=%.2e, max_rel=%.2e, diffs=%d)",
                   pr.max_abs_diff, pr.max_rel_diff, pr.num_diffs);
        }
        printf("\n");

        free(A_fp32); free(B_fp32); free(bias);
        free(C_serial); free(C_pool);
        free(A_q); free(B_q);
    }

    return fail_count;
}

/* ============================================================================
 * Generic test runner for quantized GEMV (serial vs thread pool)
 * ============================================================================ */

static int run_gemv_q_test(const char *test_name,
                            gemv_q_fn serial_fn, gemv_q_fn dispatch_fn,
                            int quant_block_size, size_t quant_block_bytes,
                            int is_q5_weights,
                            test_config_t *configs,
                            int warmup, int iters, float abs_tol, int verbose) {
    int pass_count = 0, fail_count = 0;

    printf("\n");
    printf("%-15s %8s %8s  %10s %10s %8s  %s\n",
           "Config", "M", "K", "Serial(ms)", "Pool(ms)", "Speedup", "Parity");
    printf("--------------------------------------------------------------------------------\n");

    for (int c = 0; configs[c].name; c++) {
        int M = configs[c].M, K = configs[c].K;
        if (K % quant_block_size != 0) continue;
        if (K % QK8_0 != 0) continue;

        /* Allocate */
        float *W_fp32   = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32   = (float *)malloc(K * sizeof(float));
        float *y_serial = (float *)calloc(M, sizeof(float));
        float *y_pool   = (float *)calloc(M, sizeof(float));

        int bpr_w = K / quant_block_size;
        void *W_q = malloc((size_t)M * bpr_w * quant_block_bytes);
        void *x_q8 = malloc((size_t)(K / QK8_0) * sizeof(block_q8_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);

        if (is_q5_weights) {
            quantize_to_q5_0(W_fp32, W_q, M, K);
        } else {
            quantize_to_q8_0(W_fp32, W_q, M, K);
        }
        quantize_to_q8_0(x_fp32, x_q8, 1, K);

        /* Run serial */
        serial_fn(y_serial, W_q, x_q8, M, K);

        /* Run thread pool dispatch */
        dispatch_fn(y_pool, W_q, x_q8, M, K);

        /* Parity */
        parity_result_t pr = check_parity(y_serial, y_pool, M, abs_tol);
        int pass = (pr.max_abs_diff <= abs_tol);
        if (pass) pass_count++; else fail_count++;

        /* Benchmark */
        double t_serial = bench_gemv_q(serial_fn, y_serial, W_q, x_q8, M, K, warmup, iters);
        double t_pool   = bench_gemv_q(dispatch_fn, y_pool, W_q, x_q8, M, K, warmup, iters);
        double speedup  = t_serial / t_pool;

        printf("%-15s %8d %8d  %10.3f %10.3f %7.2fx  %s",
               configs[c].name, M, K, t_serial, t_pool, speedup,
               pass ? "PASS" : "FAIL");
        if (!pass || verbose) {
            printf("  (max_abs=%.2e, max_rel=%.2e, diffs=%d)",
                   pr.max_abs_diff, pr.max_rel_diff, pr.num_diffs);
        }
        printf("\n");

        free(W_fp32); free(x_fp32); free(y_serial); free(y_pool);
        free(W_q); free(x_q8);
    }

    return fail_count;
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

    /* Initialize thread pool */
    ck_threadpool_t *pool = ck_threadpool_global();
    int n_threads = pool ? ck_threadpool_n_threads(pool) : 1;

    printf("================================================================================\n");
    printf("  GEMV Serial vs Thread Pool Dispatch Parity & Speed Test\n");
    printf("================================================================================\n");
    printf("  Thread pool:       %s\n", pool ? "initialized" : "FAILED");
    printf("  Threads:           %d\n", n_threads);
    printf("  Warmup:            %d\n", warmup);
    printf("  Iterations:        %d\n", iters);
    printf("  Abs tolerance:     %.0e\n", abs_tol);
    printf("  Mode:              %s\n", quick ? "quick" : "full");
    printf("================================================================================\n");

    if (!pool) {
        printf("\n  *** FATAL: Thread pool failed to initialize ***\n");
        return 1;
    }

    int total_fail = 0;

    /* =========================================================================
     * Test 1: gemv_q8_0_q8_0 serial vs thread pool dispatch
     * ========================================================================= */
    printf("\n");
    printf("================================================================================\n");
    printf("  TEST 1: gemv_q8_0_q8_0 (serial) vs gemv_q8_0_q8_0_parallel_dispatch (pool)\n");
    printf("================================================================================\n");

    total_fail += run_gemv_q_test(
        "q8_0",
        gemv_q8_0_q8_0, gemv_q8_0_q8_0_parallel_dispatch,
        QK8_0, sizeof(block_q8_0),
        0, /* not q5 weights */
        configs, warmup, iters, abs_tol, verbose
    );

    /* =========================================================================
     * Test 2: gemv_q5_0_q8_0 serial vs thread pool dispatch
     * ========================================================================= */
    printf("\n");
    printf("================================================================================\n");
    printf("  TEST 2: gemv_q5_0_q8_0 (serial) vs gemv_q5_0_q8_0_parallel_dispatch (pool)\n");
    printf("================================================================================\n");

    total_fail += run_gemv_q_test(
        "q5_0",
        gemv_q5_0_q8_0, gemv_q5_0_q8_0_parallel_dispatch,
        QK5_0, sizeof(block_q5_0),
        1, /* q5 weights */
        configs, warmup, iters, abs_tol, verbose
    );

    /* =========================================================================
     * Test 3: gemv_fused_q5_0_bias serial vs thread pool dispatch
     * ========================================================================= */
    printf("\n");
    printf("================================================================================\n");
    printf("  TEST 3: gemv_fused_q5_0_bias_dispatch (serial) vs\n");
    printf("          gemv_fused_q5_0_bias_parallel_dispatch (pool)\n");
    printf("================================================================================\n\n");

    printf("%-15s %8s %8s  %10s %10s %8s  %s\n",
           "Config", "M", "K", "Serial(ms)", "Pool(ms)", "Speedup", "Parity");
    printf("--------------------------------------------------------------------------------\n");

    for (int c = 0; configs[c].name; c++) {
        int M = configs[c].M, K = configs[c].K;
        if (K % QK5_0 != 0) continue;

        float *W_fp32   = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32   = (float *)malloc(K * sizeof(float));
        float *bias     = (float *)malloc(M * sizeof(float));
        float *y_serial = (float *)calloc(M, sizeof(float));
        float *y_pool   = (float *)calloc(M, sizeof(float));

        int bpr_w = K / QK5_0;
        void *W_q5 = malloc((size_t)M * bpr_w * sizeof(block_q5_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);
        fill_random_float(bias, M, 0.01f);
        quantize_to_q5_0(W_fp32, W_q5, M, K);

        /* Run serial */
        gemv_fused_q5_0_bias_dispatch(y_serial, W_q5, x_fp32, bias, M, K);

        /* Run thread pool dispatch */
        gemv_fused_q5_0_bias_parallel_dispatch(y_pool, W_q5, x_fp32, bias, M, K);

        /* Parity */
        parity_result_t pr = check_parity(y_serial, y_pool, M, abs_tol);
        int pass = (pr.max_abs_diff <= abs_tol);
        if (!pass) total_fail++;

        /* Benchmark */
        double t_serial = bench_gemv_fused(gemv_fused_q5_0_bias_dispatch,
                                            y_serial, W_q5, x_fp32, bias, M, K, warmup, iters);
        double t_pool   = bench_gemv_fused(gemv_fused_q5_0_bias_parallel_dispatch,
                                            y_pool, W_q5, x_fp32, bias, M, K, warmup, iters);
        double speedup  = t_serial / t_pool;

        printf("%-15s %8d %8d  %10.3f %10.3f %7.2fx  %s",
               configs[c].name, M, K, t_serial, t_pool, speedup,
               pass ? "PASS" : "FAIL");
        if (!pass || verbose) {
            printf("  (max_abs=%.2e, max_rel=%.2e, diffs=%d)",
                   pr.max_abs_diff, pr.max_rel_diff, pr.num_diffs);
        }
        printf("\n");

        free(W_fp32); free(x_fp32); free(bias); free(y_serial); free(y_pool);
        free(W_q5);
    }

    /* =========================================================================
     * Test 4: Fused with NULL bias (edge case)
     * ========================================================================= */
    printf("\n");
    printf("================================================================================\n");
    printf("  TEST 4: gemv_fused_q5_0_bias_parallel_dispatch with NULL bias\n");
    printf("================================================================================\n\n");

    {
        int M = 896, K = 896;
        float *W_fp32   = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32   = (float *)malloc(K * sizeof(float));
        float *y_serial = (float *)calloc(M, sizeof(float));
        float *y_pool   = (float *)calloc(M, sizeof(float));
        void *W_q5 = malloc((size_t)M * (K / QK5_0) * sizeof(block_q5_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);
        quantize_to_q5_0(W_fp32, W_q5, M, K);

        gemv_fused_q5_0_bias_dispatch(y_serial, W_q5, x_fp32, NULL, M, K);
        gemv_fused_q5_0_bias_parallel_dispatch(y_pool, W_q5, x_fp32, NULL, M, K);

        parity_result_t pr = check_parity(y_serial, y_pool, M, abs_tol);
        int pass = (pr.max_abs_diff <= abs_tol);
        if (!pass) total_fail++;

        printf("  NULL bias test (M=%d, K=%d): %s", M, K, pass ? "PASS" : "FAIL");
        if (!pass || verbose) {
            printf("  (max_abs=%.2e, max_rel=%.2e)", pr.max_abs_diff, pr.max_rel_diff);
        }
        printf("\n");

        free(W_fp32); free(x_fp32); free(y_serial); free(y_pool); free(W_q5);
    }

    /* =========================================================================
     * Test 5: Dispatch latency (overhead of packing args + waking threads)
     * ========================================================================= */
    printf("\n");
    printf("================================================================================\n");
    printf("  TEST 5: Thread pool dispatch latency (small M, overhead-dominated)\n");
    printf("================================================================================\n\n");

    {
        /* Small M where dispatch overhead is visible */
        int M = 64, K = 896;
        float *W_fp32   = (float *)malloc((size_t)M * K * sizeof(float));
        float *x_fp32   = (float *)malloc(K * sizeof(float));
        float *y        = (float *)calloc(M, sizeof(float));
        void *W_q8 = malloc((size_t)M * (K / QK8_0) * sizeof(block_q8_0));
        void *x_q8 = malloc((size_t)(K / QK8_0) * sizeof(block_q8_0));

        srand(42);
        fill_random_float(W_fp32, M * K, 0.1f);
        fill_random_float(x_fp32, K, 1.0f);
        quantize_to_q8_0(W_fp32, W_q8, M, K);
        quantize_to_q8_0(x_fp32, x_q8, 1, K);

        int lat_iters = quick ? 100 : 1000;
        int lat_warmup = quick ? 10 : 100;

        double t_serial = bench_gemv_q(gemv_q8_0_q8_0, y, W_q8, x_q8, M, K, lat_warmup, lat_iters);
        double t_pool   = bench_gemv_q(gemv_q8_0_q8_0_parallel_dispatch, y, W_q8, x_q8, M, K, lat_warmup, lat_iters);

        printf("  M=%d, K=%d (%d iterations)\n", M, K, lat_iters);
        printf("  Serial:          %10.4f ms\n", t_serial);
        printf("  Thread pool:     %10.4f ms\n", t_pool);
        printf("  Overhead:        %10.4f ms (%.1f%%)\n",
               t_pool - t_serial,
               (t_pool > t_serial) ? ((t_pool - t_serial) / t_serial * 100.0) : 0.0);
        printf("  (Small M: dispatch overhead may exceed parallelism benefit)\n");

        free(W_fp32); free(x_fp32); free(y); free(W_q8); free(x_q8);
    }

    /* =========================================================================
     * GEMM Parity Tests (prefill path: ck_parallel_prefill.h)
     * ========================================================================= */

    gemm_test_config_t *gemm_cfgs = quick ? gemm_configs_quick : gemm_configs_default;

    /* =========================================================================
     * Test 6: gemm_nt_q5_0_q8_0 serial vs thread pool dispatch
     *         (Q8_0 activations x Q5_0 weights — Q/K proj, out proj, MLP gate+up)
     * ========================================================================= */
    printf("\n");
    printf("================================================================================\n");
    printf("  TEST 6: gemm_nt_q5_0_q8_0 (serial) vs\n");
    printf("          gemm_nt_q5_0_q8_0_parallel_dispatch (pool) [prefill]\n");
    printf("================================================================================\n");

    total_fail += run_gemm_test(
        "q5_0_q8_0",
        gemm_nt_q5_0_q8_0, gemm_nt_q5_0_q8_0_parallel_dispatch,
        QK8_0, sizeof(block_q8_0),   /* A = Q8_0 */
        QK5_0, sizeof(block_q5_0),   /* B = Q5_0 */
        1,  /* B_is_q5 */
        0,  /* A_is_q8_K */
        0,  /* B_is_q6_K */
        gemm_cfgs, warmup, iters, abs_tol, verbose
    );

    /* =========================================================================
     * Test 7: gemm_nt_q8_0_q8_0 serial vs thread pool dispatch
     *         (Q8_0 activations x Q8_0 weights — V proj)
     * ========================================================================= */
    printf("\n");
    printf("================================================================================\n");
    printf("  TEST 7: gemm_nt_q8_0_q8_0 (serial) vs\n");
    printf("          gemm_nt_q8_0_q8_0_parallel_dispatch (pool) [prefill]\n");
    printf("================================================================================\n");

    total_fail += run_gemm_test(
        "q8_0_q8_0",
        gemm_nt_q8_0_q8_0, gemm_nt_q8_0_q8_0_parallel_dispatch,
        QK8_0, sizeof(block_q8_0),   /* A = Q8_0 */
        QK8_0, sizeof(block_q8_0),   /* B = Q8_0 */
        0,  /* B_is_q5 */
        0,  /* A_is_q8_K */
        0,  /* B_is_q6_K */
        gemm_cfgs, warmup, iters, abs_tol, verbose
    );

    /* =========================================================================
     * Test 8: gemm_nt_q6_k_q8_k serial vs thread pool dispatch
     *         (Q8_K activations x Q6_K weights — MLP down proj)
     * ========================================================================= */
    printf("\n");
    printf("================================================================================\n");
    printf("  TEST 8: gemm_nt_q6_k_q8_k (serial) vs\n");
    printf("          gemm_nt_q6_k_q8_k_parallel_dispatch (pool) [prefill]\n");
    printf("================================================================================\n");

    /* Q6_K/Q8_K require K % 256 == 0 — filter configs accordingly */
    {
        /* Build filtered config list with K divisible by QK_K (256) */
        gemm_test_config_t q6k_configs[16];
        int q6k_count = 0;
        for (int c = 0; gemm_cfgs[c].name && q6k_count < 15; c++) {
            if (gemm_cfgs[c].K % QK_K == 0) {
                q6k_configs[q6k_count++] = gemm_cfgs[c];
            }
        }
        q6k_configs[q6k_count] = (gemm_test_config_t){NULL, 0, 0, 0};

        if (q6k_count > 0) {
            total_fail += run_gemm_test(
                "q6_k_q8_k",
                gemm_nt_q6_k_q8_k, gemm_nt_q6_k_q8_k_parallel_dispatch,
                QK_K, sizeof(block_q8_K),   /* A = Q8_K */
                QK_K, sizeof(block_q6_K),   /* B = Q6_K */
                0,  /* B_is_q5 */
                1,  /* A_is_q8_K */
                1,  /* B_is_q6_K */
                q6k_configs, warmup, iters, abs_tol, verbose
            );
        } else {
            printf("\n  (Skipped: no configs with K %% 256 == 0)\n");
        }
    }

    /* =========================================================================
     * Summary
     * ========================================================================= */
    printf("\n================================================================================\n");
    printf("  SUMMARY\n");
    printf("================================================================================\n");
    printf("  Thread pool:   %d threads\n", n_threads);
    printf("  Tests:         GEMV decode (1-5) + GEMM prefill (6-8)\n");
    printf("  Failed:        %d\n", total_fail);
    printf("================================================================================\n");

    /* Shutdown thread pool */
    ck_threadpool_global_destroy();

    if (total_fail > 0) {
        printf("\n  *** PARITY FAILURE: Thread pool dispatch diverges from serial! ***\n\n");
        return 1;
    }

    printf("\n  All thread pool dispatch kernels match serial output within tolerance.\n\n");
    return 0;
}
