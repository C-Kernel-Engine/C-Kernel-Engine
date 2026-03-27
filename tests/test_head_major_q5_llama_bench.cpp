/**
 * test_head_major_q5_llama_bench.cpp
 *
 * Apples-to-apples benchmark against the local ggml-org/llama.cpp CPU path
 * for Q5_0 output projection from head-major attention output.
 *
 * Compares:
 *   1. CK direct head-major FP32 path: ck_gemm_nt_head_major_q5_0
 *   2. CK flattened Q5_0 x Q8_0 path: flatten + quantize + gemm_nt_q5_0_q8_0
 *   3. llama.cpp exact reference path: flatten + quantize + ggml_vec_dot_q5_0_q8_0
 *
 * The strict parity gate is between (2) and (3), since they implement the
 * same quantized activation math. The direct head-major path is measured and
 * compared numerically, but it is expected to differ because it keeps FP32
 * activations rather than quantizing them to Q8_0.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <time.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"

extern "C" {
void quantize_row_q8_0_ref(const float * x, block_q8_0 * y, int64_t k);
void ggml_vec_dot_q5_0_q8_0(int n, float * s, size_t bs,
                            const void * vx, size_t bx,
                            const void * vy, size_t by, int nrc);

void ck_gemm_nt_head_major_q5_0(const float *attn_out,
                                const void *wo,
                                const float *bias,
                                float *output,
                                int tokens,
                                int embed_dim,
                                int num_heads,
                                int head_dim);

void gemm_nt_q5_0_q8_0(const void *A_q8,
                       const void *B_q5,
                       const float *bias,
                       float *C,
                       int M,
                       int N,
                       int K);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    const char *name;
    int tokens;
    int num_heads;
    int head_dim;
} bench_config_t;

static bench_config_t configs_quick[] = {
    {"small",        4,  4, 32},
    {"qwen_prefill", 16, 14, 64},
    {nullptr, 0, 0, 0}
};

static bench_config_t configs_full[] = {
    {"small",        4,  4, 32},
    {"qwen_prefill", 16, 14, 64},
    {"qwen_long",    32, 14, 64},
    {"wide",         32, 16, 64},
    {nullptr, 0, 0, 0}
};

static void fill_random_f32(float *data, size_t count, float scale, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

static void fill_random_q5_0(block_q5_0 *data, size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> scale_dist(0.005f, 0.05f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (size_t i = 0; i < count; ++i) {
        data[i].d = ggml_fp32_to_fp16(scale_dist(rng));
        for (int j = 0; j < 4; ++j) {
            data[i].qh[j] = (uint8_t)byte_dist(rng);
        }
        for (int j = 0; j < QK5_0 / 2; ++j) {
            data[i].qs[j] = (uint8_t)byte_dist(rng);
        }
    }
}

static void flatten_head_major(const float *attn_out,
                               float *flat,
                               int tokens,
                               int num_heads,
                               int head_dim) {
    const int embed_dim = num_heads * head_dim;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const float *src = attn_out + (size_t)h * (size_t)tokens * (size_t)head_dim +
                               (size_t)t * (size_t)head_dim;
            float *dst = flat + (size_t)t * (size_t)embed_dim + (size_t)h * (size_t)head_dim;
            memcpy(dst, src, (size_t)head_dim * sizeof(float));
        }
    }
}

static void quantize_flat_q8_0(const float *flat,
                               block_q8_0 *flat_q8,
                               int tokens,
                               int embed_dim) {
    const int blocks_per_row = embed_dim / QK8_0;
    for (int t = 0; t < tokens; ++t) {
        quantize_row_q8_0_ref(flat + (size_t)t * (size_t)embed_dim,
                              flat_q8 + (size_t)t * (size_t)blocks_per_row,
                              embed_dim);
    }
}

static void llama_flat_q5_0_q8_0(const block_q8_0 *flat_q8,
                                 const block_q5_0 *weights,
                                 const float *bias,
                                 float *output,
                                 int tokens,
                                 int embed_dim) {
    const int blocks_per_row = embed_dim / QK5_0;
    const size_t row_bytes = (size_t)blocks_per_row * sizeof(block_q5_0);

    for (int t = 0; t < tokens; ++t) {
        const block_q8_0 *a_row = flat_q8 + (size_t)t * (size_t)blocks_per_row;
        float *out_row = output + (size_t)t * (size_t)embed_dim;
        for (int n = 0; n < embed_dim; ++n) {
            float sum = 0.0f;
            const void *w_row = (const uint8_t *)weights + (size_t)n * row_bytes;
            ggml_vec_dot_q5_0_q8_0(embed_dim, &sum, sizeof(float),
                                   w_row, sizeof(block_q5_0),
                                   a_row, sizeof(block_q8_0), 1);
            out_row[n] = bias ? sum + bias[n] : sum;
        }
    }
}

static float max_abs_diff(const float *a, const float *b, size_t count) {
    float md = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > md) {
            md = d;
        }
    }
    return md;
}

static int run_case(const bench_config_t *cfg, int warmup, int iters, float tol) {
    const int tokens = cfg->tokens;
    const int num_heads = cfg->num_heads;
    const int head_dim = cfg->head_dim;
    const int embed_dim = num_heads * head_dim;
    const int blocks_per_row = embed_dim / QK5_0;
    const size_t flat_count = (size_t)tokens * (size_t)embed_dim;
    const size_t q5_count = (size_t)embed_dim * (size_t)blocks_per_row;
    const size_t q8_count = (size_t)tokens * (size_t)blocks_per_row;

    std::vector<float> attn_out((size_t)num_heads * (size_t)tokens * (size_t)head_dim);
    std::vector<float> bias(embed_dim);
    std::vector<block_q5_0> weights(q5_count);

    std::vector<float> flat(flat_count);
    std::vector<block_q8_0> flat_q8(q8_count);
    std::vector<float> out_ck_direct(flat_count);
    std::vector<float> out_ck_q8(flat_count);
    std::vector<float> out_llama(flat_count);

    fill_random_f32(attn_out.data(), attn_out.size(), 0.2f, 42u + (uint32_t)tokens);
    fill_random_f32(bias.data(), bias.size(), 0.1f, 123u + (uint32_t)embed_dim);
    fill_random_q5_0(weights.data(), weights.size(), 777u + (uint32_t)embed_dim);

    auto run_ck_direct = [&]() {
        ck_gemm_nt_head_major_q5_0(attn_out.data(),
                                   weights.data(),
                                   bias.data(),
                                   out_ck_direct.data(),
                                   tokens,
                                   embed_dim,
                                   num_heads,
                                   head_dim);
    };

    auto run_ck_q8 = [&]() {
        flatten_head_major(attn_out.data(), flat.data(), tokens, num_heads, head_dim);
        quantize_flat_q8_0(flat.data(), flat_q8.data(), tokens, embed_dim);
        gemm_nt_q5_0_q8_0(flat_q8.data(),
                          weights.data(),
                          bias.data(),
                          out_ck_q8.data(),
                          tokens,
                          embed_dim,
                          embed_dim);
    };

    auto run_llama = [&]() {
        flatten_head_major(attn_out.data(), flat.data(), tokens, num_heads, head_dim);
        quantize_flat_q8_0(flat.data(), flat_q8.data(), tokens, embed_dim);
        llama_flat_q5_0_q8_0(flat_q8.data(),
                             weights.data(),
                             bias.data(),
                             out_llama.data(),
                             tokens,
                             embed_dim);
    };

    run_ck_direct();
    run_ck_q8();
    run_llama();

    const float diff_ck_q8_vs_llama = max_abs_diff(out_ck_q8.data(), out_llama.data(), flat_count);
    const float diff_ck_direct_vs_llama = max_abs_diff(out_ck_direct.data(), out_llama.data(), flat_count);
    const char *parity = diff_ck_q8_vs_llama <= tol ? "PASS" : "FAIL";

    for (int i = 0; i < warmup; ++i) {
        run_ck_direct();
        run_ck_q8();
        run_llama();
    }

    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        run_ck_direct();
    }
    double ck_direct_ms = (now_ms() - t0) / (double)iters;

    t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        run_ck_q8();
    }
    double ck_q8_ms = (now_ms() - t0) / (double)iters;

    t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        run_llama();
    }
    double llama_ms = (now_ms() - t0) / (double)iters;

    printf("%-14s %6d %6d %6d  %10.3f %10.3f %10.3f  %7.2fx %7.2fx  %9.2e  %9.2e  %s\n",
           cfg->name,
           tokens,
           num_heads,
           head_dim,
           llama_ms,
           ck_q8_ms,
           ck_direct_ms,
           llama_ms / ck_q8_ms,
           llama_ms / ck_direct_ms,
           diff_ck_q8_vs_llama,
           diff_ck_direct_vs_llama,
           parity);

    return diff_ck_q8_vs_llama <= tol ? 0 : 1;
}

int main(int argc, char **argv) {
    int quick = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        }
    }

    ggml_cpu_init();

    const bench_config_t *cfgs = quick ? configs_quick : configs_full;
    const int warmup = quick ? 1 : 2;
    const int iters = quick ? 3 : 5;
    const float tol = 1e-5f;
    int failed = 0;

    printf("==============================================================================================\n");
    printf("  Head-Major Q5 Benchmark: CK vs local llama.cpp exact flatten+Q8_0+vec_dot path\n");
    printf("==============================================================================================\n");
    printf("  Mode:     %s (%d warmup, %d iters)\n", quick ? "quick" : "full", warmup, iters);
    printf("  Note:     ck_q8 and llama share the same quantized-activation math.\n");
    printf("            ck_direct is the FP32 head-major kernel, so its output delta is informational.\n");
    printf("  Parity:   ck_q8 vs llama abs_tol = %.0e\n", tol);
    printf("==============================================================================================\n\n");

    printf("%-14s %6s %6s %6s  %10s %10s %10s  %7s %7s  %9s  %9s  %s\n",
           "Config", "Tok", "Heads", "Dim",
           "llama(ms)", "ck_q8(ms)", "ck_dir(ms)",
           "q8 spd", "dir spd",
           "q8 diff", "dir diff", "Parity");
    printf("--------------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; cfgs[i].name != nullptr; ++i) {
        failed += run_case(&cfgs[i], warmup, iters, tol);
    }

    printf("\n==============================================================================================\n");
    if (failed == 0) {
        printf("  RESULT: ALL PASSED\n");
    } else {
        printf("  RESULT: %d FAILED\n", failed);
    }
    printf("==============================================================================================\n");

    return failed == 0 ? 0 : 1;
}
