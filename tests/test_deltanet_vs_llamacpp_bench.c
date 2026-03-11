/**
 * test_deltanet_vs_llamacpp_bench.c - Benchmark CK DeltaNet vs llama.cpp helper
 *
 * Compares the CK public Gated DeltaNet dispatcher against the local
 * llama.cpp kernel parity helper on identical inputs. Reports:
 *   - numerical parity (output + recurrent state)
 *   - mean wall-clock time for CK and llama.cpp
 *   - relative speedup (llama.cpp time / CK time)
 *
 * Build:
 *   make test-deltanet-vs-llamacpp-bench
 *
 * Run:
 *   ./build/test_deltanet_vs_llamacpp_bench
 *   ./build/test_deltanet_vs_llamacpp_bench --quick
 */

#include "ckernel_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void test_gated_deltanet_autoregressive(const float *q,
                                               const float *k,
                                               const float *v,
                                               const float *g,
                                               const float *beta,
                                               const float *state_in,
                                               float *state_out,
                                               float *out,
                                               int num_heads,
                                               int state_dim,
                                               float norm_eps);

extern const char *gated_deltanet_impl_name(void);

typedef struct {
    const char *name;
    int heads;
    int dim;
} bench_config_t;

static bench_config_t configs_default[] = {
    {"tiny_decode",      4,  16},
    {"qwen3next_small",  8,  32},
    {"qwen3next_mid",   16,  64},
    {"qwen3next_large", 16, 128},
    {NULL, 0, 0}
};

static bench_config_t configs_quick[] = {
    {"tiny_decode",      4,  16},
    {"qwen3next_small",  8,  32},
    {"qwen3next_mid",   16,  64},
    {NULL, 0, 0}
};

static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint32_t rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float rng_f32(uint32_t *state, float scale)
{
    uint32_t x = rng_next(state);
    float u = (float)(x & 0x00ffffffu) / 16777216.0f;
    return (u * 2.0f - 1.0f) * scale;
}

static void fill_random_f32(float *dst, size_t n, float scale, uint32_t seed)
{
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = rng_f32(&s, scale);
    }
}

static int compare_arrays(const float *ref,
                          const float *test,
                          int count,
                          float abs_tol,
                          float *max_diff_out)
{
    int mismatches = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < count; ++i) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > abs_tol) {
            mismatches++;
        }
    }

    if (max_diff_out) {
        *max_diff_out = max_diff;
    }
    return mismatches;
}

static int compare_outputs_and_state(const float *out_ref,
                                     const float *out_test,
                                     int out_count,
                                     const float *state_ref,
                                     const float *state_test,
                                     int state_count,
                                     float abs_tol,
                                     float *max_out_diff,
                                     float *max_state_diff)
{
    int out_mm = compare_arrays(out_ref, out_test, out_count, abs_tol, max_out_diff);
    int state_mm = compare_arrays(state_ref, state_test, state_count, abs_tol, max_state_diff);
    return out_mm + state_mm;
}

static double benchmark_ck(const float *q,
                           const float *k,
                           const float *v,
                           const float *g,
                           const float *beta,
                           const float *state_in,
                           float *state_out,
                           float *out,
                           int heads,
                           int dim,
                           float eps,
                           int warmup,
                           int iters)
{
    for (int i = 0; i < warmup; ++i) {
        gated_deltanet_autoregressive_forward(
            q, k, v, g, beta, state_in, state_out, out, heads, dim, eps);
    }

    double start = get_time_ms();
    for (int i = 0; i < iters; ++i) {
        gated_deltanet_autoregressive_forward(
            q, k, v, g, beta, state_in, state_out, out, heads, dim, eps);
    }
    return (get_time_ms() - start) / (double)iters;
}

static double benchmark_llama(const float *q,
                              const float *k,
                              const float *v,
                              const float *g,
                              const float *beta,
                              const float *state_in,
                              float *state_out,
                              float *out,
                              int heads,
                              int dim,
                              float eps,
                              int warmup,
                              int iters)
{
    for (int i = 0; i < warmup; ++i) {
        test_gated_deltanet_autoregressive(
            q, k, v, g, beta, state_in, state_out, out, heads, dim, eps);
    }

    double start = get_time_ms();
    for (int i = 0; i < iters; ++i) {
        test_gated_deltanet_autoregressive(
            q, k, v, g, beta, state_in, state_out, out, heads, dim, eps);
    }
    return (get_time_ms() - start) / (double)iters;
}

int main(int argc, char **argv)
{
    int quick = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        }
    }

    bench_config_t *cfgs = quick ? configs_quick : configs_default;
    int warmup = quick ? 3 : 10;
    int iters = quick ? 40 : 200;
    float abs_tol = 1e-4f;
    int total_fail = 0;

    printf("================================================================================\n");
    printf("  Gated DeltaNet Benchmark: CK vs llama.cpp\n");
    printf("================================================================================\n");
    printf("  CK dispatcher: %s\n", gated_deltanet_impl_name());
    printf("  Mode:          %s (%d warmup, %d iters)\n", quick ? "quick" : "full", warmup, iters);
    printf("  Parity:        abs_tol = %.0e\n", abs_tol);
    printf("================================================================================\n\n");

    printf("%-16s %10s %10s %10s %8s %8s %8s\n",
           "Config", "CK ms", "llama ms", "CK vs ggml", "Out", "State", "Parity");
    printf("--------------------------------------------------------------------------------\n");

    for (int c = 0; cfgs[c].name; ++c) {
        const int H = cfgs[c].heads;
        const int D = cfgs[c].dim;
        const int vec_count = H * D;
        const int state_count = H * D * D;
        const float eps = 1e-6f;

        float *q = malloc((size_t)vec_count * sizeof(float));
        float *k = malloc((size_t)vec_count * sizeof(float));
        float *v = malloc((size_t)vec_count * sizeof(float));
        float *g = malloc((size_t)H * sizeof(float));
        float *beta = malloc((size_t)H * sizeof(float));
        float *state_in = malloc((size_t)state_count * sizeof(float));
        float *out_ck = malloc((size_t)vec_count * sizeof(float));
        float *state_ck = malloc((size_t)state_count * sizeof(float));
        float *out_llama = malloc((size_t)vec_count * sizeof(float));
        float *state_llama = malloc((size_t)state_count * sizeof(float));

        if (!q || !k || !v || !g || !beta || !state_in || !out_ck || !state_ck || !out_llama || !state_llama) {
            fprintf(stderr, "ERROR: allocation failed for %s\n", cfgs[c].name);
            free(q);
            free(k);
            free(v);
            free(g);
            free(beta);
            free(state_in);
            free(out_ck);
            free(state_ck);
            free(out_llama);
            free(state_llama);
            return 1;
        }

        fill_random_f32(q, (size_t)vec_count, 0.25f, 101u + (uint32_t)c);
        fill_random_f32(k, (size_t)vec_count, 0.25f, 202u + (uint32_t)c);
        fill_random_f32(v, (size_t)vec_count, 0.25f, 303u + (uint32_t)c);
        fill_random_f32(g, (size_t)H, 0.10f, 404u + (uint32_t)c);
        fill_random_f32(beta, (size_t)H, 0.50f, 505u + (uint32_t)c);
        fill_random_f32(state_in, (size_t)state_count, 0.20f, 606u + (uint32_t)c);

        test_gated_deltanet_autoregressive(
            q, k, v, g, beta, state_in, state_llama, out_llama, H, D, eps);
        gated_deltanet_autoregressive_forward(
            q, k, v, g, beta, state_in, state_ck, out_ck, H, D, eps);

        float max_out_diff = 0.0f;
        float max_state_diff = 0.0f;
        int mismatches = compare_outputs_and_state(out_llama, out_ck, vec_count,
                                                   state_llama, state_ck, state_count,
                                                   abs_tol, &max_out_diff, &max_state_diff);

        double ck_ms = benchmark_ck(q, k, v, g, beta, state_in, state_ck, out_ck, H, D, eps, warmup, iters);
        double llama_ms = benchmark_llama(q, k, v, g, beta, state_in, state_llama, out_llama, H, D, eps, warmup, iters);
        double speedup = ck_ms > 0.0 ? llama_ms / ck_ms : 0.0;

        printf("%-16s %10.3f %10.3f %10.2fx %8.2e %8.2e %8s\n",
               cfgs[c].name,
               ck_ms,
               llama_ms,
               speedup,
               max_out_diff,
               max_state_diff,
               mismatches == 0 ? "PASS" : "FAIL");

        if (mismatches != 0) {
            total_fail++;
        }

        free(q);
        free(k);
        free(v);
        free(g);
        free(beta);
        free(state_in);
        free(out_ck);
        free(state_ck);
        free(out_llama);
        free(state_llama);
    }

    printf("--------------------------------------------------------------------------------\n");
    if (total_fail == 0) {
        printf("\nRESULT: all CK results match llama.cpp within tolerance.\n");
        return 0;
    }

    printf("\nRESULT: %d config(s) failed parity.\n", total_fail);
    return 1;
}
