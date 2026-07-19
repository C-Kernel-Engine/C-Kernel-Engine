/*
 * bench_qwen3vl_encoder_attention.c
 *
 * Focused benchmark for the Qwen3-VL vision encoder full-attention hot path.
 * Calls the exact public CK kernel selected by the Qwen3-VL vision circuit:
 * attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided.
 */

#include "ckernel_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint32_t rng_state = 0xC0FFEEu;
static uint32_t rng_u32(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rng_f32(float scale) {
    const uint32_t v = rng_u32() >> 8;
    const float u = (float)v / (float)0x00ffffffu;
    return (u * 2.0f - 1.0f) * scale;
}
static void fill_f32(float *p, size_t n, float scale) { for (size_t i = 0; i < n; ++i) p[i] = rng_f32(scale); }
static float checksum_f32(const float *p, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i += 17) s += (double)p[i] * 0.0009765625;
    return (float)s;
}
static int parse_int_arg(int argc, char **argv, const char *name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            const int v = atoi(argv[i + 1]);
            return v > 0 ? v : fallback;
        }
    }
    return fallback;
}
static int has_arg(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], name) == 0) return 1;
    return 0;
}
static const char *parse_string_arg(int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return fallback;
}
static void usage(const char *prog) {
    printf("Usage: %s [--provider auto|tile64|tile336] [--tokens T] [--heads H] [--kv-heads HKV] [--head-dim D] [--aligned-head-dim AD] [--threads N] [--iters N] [--warmup N]\n", prog);
}

typedef void (*attention_provider_fn)(
    const float *, const float *, const float *, float *,
    int, int, int, int, int, int);

static const char *compiler_family(void) {
#if defined(__INTEL_LLVM_COMPILER)
    return "icx";
#elif defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#else
    return "unknown";
#endif
}

int main(int argc, char **argv) {
    if (has_arg(argc, argv, "--help") || has_arg(argc, argv, "-h")) { usage(argv[0]); return 0; }

    const int T = parse_int_arg(argc, argv, "--tokens", 4032);
    const int H = parse_int_arg(argc, argv, "--heads", 16);
    const int HKV = parse_int_arg(argc, argv, "--kv-heads", H);
    const int D = parse_int_arg(argc, argv, "--head-dim", 72);
    const int AD = parse_int_arg(argc, argv, "--aligned-head-dim", D);
    const int threads = parse_int_arg(argc, argv, "--threads", 0);
    const int iters = parse_int_arg(argc, argv, "--iters", 3);
    const int warmup = parse_int_arg(argc, argv, "--warmup", 1);
    const char *provider_name = parse_string_arg(argc, argv, "--provider", "auto");
    attention_provider_fn provider = NULL;
    const char *provider_function = NULL;
    if (strcmp(provider_name, "auto") == 0) {
        provider = attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided;
        provider_function = "attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided";
    } else if (strcmp(provider_name, "tile64") == 0) {
        provider = attention_forward_full_head_major_gqa_tiled64_f16kv_fp32_strided;
        provider_function = "attention_forward_full_head_major_gqa_tiled64_f16kv_fp32_strided";
    } else if (strcmp(provider_name, "tile336") == 0) {
        provider = attention_forward_full_head_major_gqa_tiled336_f16kv_fp32_strided;
        provider_function = "attention_forward_full_head_major_gqa_tiled336_f16kv_fp32_strided";
    } else {
        fprintf(stderr, "unknown provider: %s\n", provider_name);
        usage(argv[0]);
        return 2;
    }

    if (T <= 0 || H <= 0 || HKV <= 0 || HKV > H || D <= 0 || AD < D) { usage(argv[0]); return 2; }
    if (threads > 0) ck_set_num_threads(threads);

    const size_t q_elems = (size_t)H * (size_t)T * (size_t)AD;
    const size_t kv_elems = (size_t)HKV * (size_t)T * (size_t)AD;
    const size_t q_bytes = q_elems * sizeof(float);
    const size_t kv_bytes = kv_elems * sizeof(float);

    float *q = NULL, *k = NULL, *v = NULL, *out = NULL;
    if (posix_memalign((void **)&q, 64, q_bytes) != 0 ||
        posix_memalign((void **)&k, 64, kv_bytes) != 0 ||
        posix_memalign((void **)&v, 64, kv_bytes) != 0 ||
        posix_memalign((void **)&out, 64, q_bytes) != 0) {
        fprintf(stderr, "allocation failed\n");
        free(q); free(k); free(v); free(out);
        return 3;
    }

    rng_state = 0xC0FFEEu;
    fill_f32(q, q_elems, 0.25f);
    fill_f32(k, kv_elems, 0.25f);
    fill_f32(v, kv_elems, 0.25f);
    memset(out, 0, q_bytes);

    printf("Qwen3-VL encoder production attention benchmark\n");
    printf("provider=%s\n", provider_function);
    printf("benchmark_compiler_family=%s benchmark_compiler_version=%s\n",
           compiler_family(), __VERSION__);
    printf("T=%d H=%d HKV=%d D=%d AD=%d threads=%d warmup=%d iters=%d\n", T, H, HKV, D, AD, ck_get_num_threads(), warmup, iters);
    printf("env CK_SPEED_PROFILE=%s CK_ATTENTION_QBLOCK4=%s CK_ATTENTION_QBLOCK8=%s CK_ATTENTION_THREAD_CAP=%s\n",
           getenv("CK_SPEED_PROFILE") ? getenv("CK_SPEED_PROFILE") : "",
           getenv("CK_ATTENTION_QBLOCK4") ? getenv("CK_ATTENTION_QBLOCK4") : "",
           getenv("CK_ATTENTION_QBLOCK8") ? getenv("CK_ATTENTION_QBLOCK8") : "",
           getenv("CK_ATTENTION_THREAD_CAP") ? getenv("CK_ATTENTION_THREAD_CAP") : "");

    for (int i = 0; i < warmup; ++i) {
        provider(q, k, v, out, H, HKV, T, D, AD, T);
    }

    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        provider(q, k, v, out, H, HKV, T, D, AD, T);
    }
    const double avg_ms = (now_ms() - t0) / (double)iters;
    const double q_per_s = ((double)T * (double)H) / (avg_ms / 1000.0);
    const double dot_flop = 2.0 * (double)H * (double)T * (double)T * (double)D;
    const double approx_gflops = dot_flop / (avg_ms / 1000.0) / 1.0e9;

    printf("avg_ms=%.3f query_heads/s=%.1f approx_dot_gflops=%.2f checksum=%.8f\n",
           avg_ms, q_per_s, approx_gflops, checksum_f32(out, q_elems));

    free(q); free(k); free(v); free(out);
    return 0;
}
