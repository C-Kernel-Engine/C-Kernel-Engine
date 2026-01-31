/**
 * test_gemm_avx_bench.c - Benchmark: gemm_nt_q8_0_q8_0 AVX vs scalar reference
 *
 * Measures speedup of the SSE4.1-based _avx kernel over the scalar _ref
 * for the Q8_0 x Q8_0 GEMM used in prefill (V projection, embeddings).
 *
 * On AVX-only CPUs (Sandy/Ivy Bridge), the dispatch chain previously fell
 * through to _ref. This benchmark quantifies the improvement.
 *
 * Also verifies numerical parity between _avx and _ref (tolerance: 1e-3 abs).
 *
 * Tests:
 *   1. Parity: _avx output matches _ref within tolerance
 *   2. Benchmark: _ref vs _avx timing across model-realistic shapes
 *   3. Dispatch: confirms gemm_nt_q8_0_q8_0() now calls _avx (not _ref)
 *
 * Build:
 *   make test-gemm-avx-bench
 *
 * Run:
 *   ./build/test_gemm_avx_bench
 *   ./build/test_gemm_avx_bench --quick
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

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
 * External kernel symbols (all exported from libckernel_engine.so)
 * ============================================================================ */

extern void gemm_nt_q8_0_q8_0_ref(
    const void *A, const void *B, float *C, int M, int N, int K);

extern void gemm_nt_q8_0_q8_0_avx(
    const void *A, const void *B, float *C, int M, int N, int K);

/* Public dispatcher (with bias) — should now call _avx on AVX-only CPUs */
extern void gemm_nt_q8_0_q8_0(
    const void *A, const void *B, const float *bias, float *C, int M, int N, int K);

/* Backend name for logging */
extern const char* gemm_batch_int8_impl_name(void);

/* ============================================================================
 * Test data helpers
 * ============================================================================ */

static void fill_random_q8_0(block_q8_0 *blocks, size_t count, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < count; i++) {
        float scale = ((float)(rand() % 200) - 100) / 1000.0f;
        blocks[i].d = CK_FP32_TO_FP16(scale);
        for (int j = 0; j < QK8_0; j++) {
            blocks[i].qs[j] = (int8_t)(rand() % 256 - 128);
        }
    }
}

static int compare_outputs(const float *ref, const float *test,
                           int count, float abs_tol,
                           float *max_diff_out) {
    float max_diff = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < count; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > abs_tol) mismatches++;
    }
    if (max_diff_out) *max_diff_out = max_diff;
    return mismatches;
}

/* ============================================================================
 * Test configurations (Qwen2-0.5B realistic shapes)
 * ============================================================================ */

typedef struct {
    const char *name;
    int M, N, K;
} bench_config_t;

static bench_config_t configs_default[] = {
    {"qkv_proj",      32,   896,  896},
    {"mlp_gate_up",   32,  4864,  896},
    {"mlp_down",      32,   896, 4864},
    {"big_prefill",  128,   896,  896},
    {"long_prefill", 256,   896,  896},
    {NULL, 0, 0, 0}
};

static bench_config_t configs_quick[] = {
    {"small",    8,  256,  256},
    {"medium",  16,  896,  896},
    {"mlp",     32,  896, 4864},
    {NULL, 0, 0, 0}
};

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv) {
    int quick = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
    }

    bench_config_t *cfgs = quick ? configs_quick : configs_default;
    int warmup = quick ? 1 : 2;
    int iters  = quick ? 3 : 5;
    float abs_tol = 1e-3f;
    int total_fail = 0;

    printf("================================================================================\n");
    printf("  GEMM Q8_0 x Q8_0 Benchmark: AVX (SSE4.1) vs Scalar Reference\n");
    printf("================================================================================\n");
    printf("  Backend:  %s\n", gemm_batch_int8_impl_name());
    printf("  Mode:     %s (%d warmup, %d iters)\n", quick ? "quick" : "full", warmup, iters);
    printf("  Parity:   abs_tol = %.0e\n", abs_tol);
    printf("================================================================================\n\n");

    printf("%-16s %5s %5s %5s  %10s %10s  %7s  %s\n",
           "Config", "M", "N", "K", "ref(ms)", "avx(ms)", "Speedup", "Parity");
    printf("--------------------------------------------------------------------------------\n");

    for (int c = 0; cfgs[c].name; c++) {
        int M = cfgs[c].M;
        int N = cfgs[c].N;
        int K = cfgs[c].K;
        int nb = K / QK8_0;
        int output_count = M * N;

        /* Allocate test data */
        size_t a_count = (size_t)M * nb;
        size_t b_count = (size_t)N * nb;
        block_q8_0 *A = malloc(a_count * sizeof(block_q8_0));
        block_q8_0 *B = malloc(b_count * sizeof(block_q8_0));
        float *C_ref = malloc((size_t)output_count * sizeof(float));
        float *C_avx = malloc((size_t)output_count * sizeof(float));

        if (!A || !B || !C_ref || !C_avx) {
            fprintf(stderr, "ERROR: allocation failed for %s\n", cfgs[c].name);
            total_fail++;
            free(A); free(B); free(C_ref); free(C_avx);
            continue;
        }

        fill_random_q8_0(A, a_count, 42);
        fill_random_q8_0(B, b_count, 123);

        /* Warmup both paths */
        for (int w = 0; w < warmup; w++) {
            gemm_nt_q8_0_q8_0_ref(A, B, C_ref, M, N, K);
            gemm_nt_q8_0_q8_0_avx(A, B, C_avx, M, N, K);
        }

        /* Benchmark: scalar ref */
        double t0 = get_time_ms();
        for (int i = 0; i < iters; i++)
            gemm_nt_q8_0_q8_0_ref(A, B, C_ref, M, N, K);
        double ref_ms = (get_time_ms() - t0) / iters;

        /* Benchmark: AVX */
        t0 = get_time_ms();
        for (int i = 0; i < iters; i++)
            gemm_nt_q8_0_q8_0_avx(A, B, C_avx, M, N, K);
        double avx_ms = (get_time_ms() - t0) / iters;

        /* Parity check (use last iteration's output) */
        float max_diff = 0.0f;
        int mismatches = compare_outputs(C_ref, C_avx, output_count, abs_tol, &max_diff);
        const char *parity = (mismatches == 0) ? "PASS" : "FAIL";
        if (mismatches > 0) total_fail++;

        double speedup = ref_ms / avx_ms;
        printf("%-16s %5d %5d %5d  %10.3f %10.3f  %6.2fx  %s",
               cfgs[c].name, M, N, K, ref_ms, avx_ms, speedup, parity);
        if (mismatches > 0)
            printf("  (%d diffs, max=%.6f)", mismatches, max_diff);
        printf("\n");

        free(A);
        free(B);
        free(C_ref);
        free(C_avx);
    }

    /* Dispatch verification: confirm the public dispatcher routes to _avx */
    printf("\n");
    printf("================================================================================\n");
    printf("  Dispatch Verification\n");
    printf("================================================================================\n");
    printf("  gemm_batch_int8_impl_name() = \"%s\"\n", gemm_batch_int8_impl_name());

    /* Quick functional check: dispatcher output matches _avx output */
    {
        int M = 4, N = 64, K = 128;
        int nb = K / QK8_0;
        size_t a_count = (size_t)M * nb;
        size_t b_count = (size_t)N * nb;
        block_q8_0 *A = malloc(a_count * sizeof(block_q8_0));
        block_q8_0 *B = malloc(b_count * sizeof(block_q8_0));
        float *C_dispatch = malloc((size_t)M * N * sizeof(float));
        float *C_avx_out  = malloc((size_t)M * N * sizeof(float));

        fill_random_q8_0(A, a_count, 99);
        fill_random_q8_0(B, b_count, 77);

        gemm_nt_q8_0_q8_0(A, B, NULL, C_dispatch, M, N, K);
        gemm_nt_q8_0_q8_0_avx(A, B, C_avx_out, M, N, K);

        float max_diff = 0.0f;
        int mismatches = compare_outputs(C_dispatch, C_avx_out, M * N, 0.0f, &max_diff);
        if (mismatches == 0) {
            printf("  Dispatcher output matches _avx: PASS (bit-exact)\n");
        } else {
            printf("  Dispatcher output matches _avx: FAIL (%d diffs, max=%.6f)\n",
                   mismatches, max_diff);
            /* Not a hard failure — on AVX2+ machines the dispatcher calls a different variant */
        }

        free(A); free(B); free(C_dispatch); free(C_avx_out);
    }

    printf("================================================================================\n");
    if (total_fail == 0) {
        printf("  RESULT: ALL PASSED\n");
    } else {
        printf("  RESULT: %d FAILED\n", total_fail);
    }
    printf("================================================================================\n");

    return total_fail > 0 ? 1 : 0;
}
