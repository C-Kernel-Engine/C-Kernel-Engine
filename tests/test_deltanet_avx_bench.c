/**
 * test_deltanet_avx_bench.c - Benchmark: Gated DeltaNet scalar vs AVX vs AVX2
 *
 * Measures the single-token autoregressive recurrent update used by Qwen3.5 /
 * qwen3next. This benchmark reports parity and timing for:
 *   - scalar reference
 *   - AVX path
 *   - AVX2 path
 *   - AVX-512 path (when compiled)
 *   - public dispatcher
 *
 * Build:
 *   make test-deltanet-avx-bench
 *
 * Run:
 *   ./build/test_deltanet_avx_bench
 *   ./build/test_deltanet_avx_bench --quick
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Public dispatcher */
extern void gated_deltanet_autoregressive_forward(const float *q,
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

/* ISA-specialized benchmark symbols exported from deltanet_kernels.c */
extern void gated_deltanet_autoregressive_forward_ref(const float *q,
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
#if defined(__AVX__)
extern void gated_deltanet_autoregressive_forward_avx(const float *q,
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
#endif
#if defined(__AVX2__)
extern void gated_deltanet_autoregressive_forward_avx2(const float *q,
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
#endif
#if defined(__AVX512F__)
extern void gated_deltanet_autoregressive_forward_avx512(const float *q,
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
#endif
extern const char *gated_deltanet_impl_name(void);

typedef void (*deltanet_fn_t)(const float *, const float *, const float *,
                              const float *, const float *, const float *,
                              float *, float *, int, int, float);

typedef struct {
    const char *name;
    int heads;
    int dim;
} bench_config_t;

typedef struct {
    const char *name;
    deltanet_fn_t fn;
    int available;
} impl_t;

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

int main(int argc, char **argv)
{
    int quick = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        }
    }

    bench_config_t *cfgs = quick ? configs_quick : configs_default;
    int warmup = quick ? 1 : 3;
    int iters = quick ? 5 : 15;
    float abs_tol = 1e-4f;
    int total_fail = 0;

    impl_t impls[] = {
        {"ref", gated_deltanet_autoregressive_forward_ref, 1},
#if defined(__AVX__)
        {"avx", gated_deltanet_autoregressive_forward_avx, 1},
#else
        {"avx", NULL, 0},
#endif
#if defined(__AVX2__)
        {"avx2", gated_deltanet_autoregressive_forward_avx2, 1},
#else
        {"avx2", NULL, 0},
#endif
#if defined(__AVX512F__)
        {"avx512", gated_deltanet_autoregressive_forward_avx512, 1},
#else
        {"avx512", NULL, 0},
#endif
        {"dispatch", gated_deltanet_autoregressive_forward, 1},
        {NULL, NULL, 0}
    };

    printf("================================================================================\n");
    printf("  Gated DeltaNet Benchmark: Scalar vs AVX vs AVX2\n");
    printf("================================================================================\n");
    printf("  Dispatcher: %s\n", gated_deltanet_impl_name());
    printf("  Mode:       %s (%d warmup, %d iters)\n", quick ? "quick" : "full", warmup, iters);
    printf("  Parity:     abs_tol = %.0e\n", abs_tol);
    printf("================================================================================\n\n");

    printf("%-16s %-10s %8s %10s %8s %8s %8s\n",
           "Config", "Impl", "ms", "Speedup", "Out", "State", "Parity");
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
        float *out_ref = malloc((size_t)vec_count * sizeof(float));
        float *state_ref = malloc((size_t)state_count * sizeof(float));

        if (!q || !k || !v || !g || !beta || !state_in || !out_ref || !state_ref) {
            fprintf(stderr, "ERROR: allocation failed for %s\n", cfgs[c].name);
            total_fail++;
            free(q);
            free(k);
            free(v);
            free(g);
            free(beta);
            free(state_in);
            free(out_ref);
            free(state_ref);
            continue;
        }

        fill_random_f32(q, (size_t)vec_count, 0.75f, 11u + (uint32_t)c);
        fill_random_f32(k, (size_t)vec_count, 0.75f, 23u + (uint32_t)c);
        fill_random_f32(v, (size_t)vec_count, 0.75f, 37u + (uint32_t)c);
        fill_random_f32(g, (size_t)H, 0.20f, 41u + (uint32_t)c);
        fill_random_f32(beta, (size_t)H, 1.00f, 53u + (uint32_t)c);
        fill_random_f32(state_in, (size_t)state_count, 0.10f, 67u + (uint32_t)c);

        gated_deltanet_autoregressive_forward_ref(
            q, k, v, g, beta, state_in, state_ref, out_ref, H, D, eps);

        double ref_ms = 0.0;
        for (int i = 0; impls[i].name; ++i) {
            float *out_test = malloc((size_t)vec_count * sizeof(float));
            float *state_test = malloc((size_t)state_count * sizeof(float));
            if (!out_test || !state_test) {
                fprintf(stderr, "ERROR: allocation failed for %s/%s\n", cfgs[c].name, impls[i].name);
                total_fail++;
                free(out_test);
                free(state_test);
                continue;
            }

            if (!impls[i].available || !impls[i].fn) {
                printf("%-16s %-10s %8s %10s %8s %8s %8s\n",
                       cfgs[c].name, impls[i].name, "-", "-", "-", "-", "N/A");
                free(out_test);
                free(state_test);
                continue;
            }

            for (int w = 0; w < warmup; ++w) {
                impls[i].fn(q, k, v, g, beta, state_in, state_test, out_test, H, D, eps);
            }

            double t0 = get_time_ms();
            for (int it = 0; it < iters; ++it) {
                impls[i].fn(q, k, v, g, beta, state_in, state_test, out_test, H, D, eps);
            }
            double elapsed_ms = (get_time_ms() - t0) / (double)iters;

            if (strcmp(impls[i].name, "ref") == 0) {
                ref_ms = elapsed_ms;
            }

            float max_out_diff = 0.0f;
            float max_state_diff = 0.0f;
            int mismatches = compare_outputs_and_state(
                out_ref, out_test, vec_count, state_ref, state_test, state_count,
                abs_tol, &max_out_diff, &max_state_diff);

            const char *speedup = "-";
            char speedup_buf[32];
            if (ref_ms > 0.0 && strcmp(impls[i].name, "ref") != 0) {
                snprintf(speedup_buf, sizeof(speedup_buf), "%.2fx", ref_ms / elapsed_ms);
                speedup = speedup_buf;
            }

            printf("%-16s %-10s %8.3f %10s %8.2e %8.2e %8s\n",
                   cfgs[c].name, impls[i].name, elapsed_ms, speedup,
                   max_out_diff, max_state_diff, mismatches == 0 ? "PASS" : "FAIL");

            if (mismatches != 0) {
                total_fail++;
            }

            free(out_test);
            free(state_test);
        }

        printf("--------------------------------------------------------------------------------\n");

        free(q);
        free(k);
        free(v);
        free(g);
        free(beta);
        free(state_in);
        free(out_ref);
        free(state_ref);
    }

    printf("\n");
    printf("================================================================================\n");
    printf("  RESULT: %s\n", total_fail == 0 ? "ALL PASSED" : "FAILURES DETECTED");
    printf("================================================================================\n");

    return total_fail == 0 ? 0 : 1;
}
