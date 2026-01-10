/**
 * Flash Attention decode microbenchmark for perf profiling.
 *
 * Build:
 *   gcc -O3 -g -fno-omit-frame-pointer -march=native -fopenmp \
 *       benchmarks/perf_flash_attn_micro.c \
 *       -Iinclude -Lbuild -lckernel_engine -lm -o build/perf_flash_attn_micro
 *
 * Run:
 *   LD_LIBRARY_PATH=build ./build/perf_flash_attn_micro 1024 4 4 64 200 20
 *
 * perf stat:
 *   LD_LIBRARY_PATH=build perf stat -e cycles,instructions,cache-references,cache-misses, \
 *       L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
 *       ./build/perf_flash_attn_micro 8192 4 4 64 200 20
 *
 * perf record:
 *   LD_LIBRARY_PATH=build perf record -g ./build/perf_flash_attn_micro 8192 4 4 64 200 20
 *
 * Flamegraph:
 *   perf script | ~/Programs/FlameGraph/stackcollapse-perf.pl | \
 *       ~/Programs/FlameGraph/flamegraph.pl > flash_attn_flame.svg
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ckernel_engine.h"
#include "cpu_features.h"

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static size_t align_size(size_t size, size_t align) {
    return (size + align - 1) / align * align;
}

static int align_int(int value, int align) {
    return (value + align - 1) / align * align;
}

static void random_init(float *arr, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        arr[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}

static void print_usage(const char *prog) {
    printf("Usage: %s T_k H H_kv D_h [iters] [warmup]\n", prog);
    printf("  T_k    = context length (KV tokens)\n");
    printf("  H      = num heads\n");
    printf("  H_kv   = num KV heads (GQA/MQA)\n");
    printf("  D_h    = head dim (un-aligned)\n");
    printf("  iters  = iterations (default 200)\n");
    printf("  warmup = warmup iterations (default 20)\n");
}

int main(int argc, char **argv) {
    int T_k = 1024;
    int num_heads = 4;
    int num_kv_heads = 4;
    int head_dim = 64;
    int iters = 200;
    int warmup = 20;

    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        print_usage(argv[0]);
        return 0;
    }

    if (argc >= 5) {
        T_k = atoi(argv[1]);
        num_heads = atoi(argv[2]);
        num_kv_heads = atoi(argv[3]);
        head_dim = atoi(argv[4]);
    }
    if (argc >= 6) {
        iters = atoi(argv[5]);
    }
    if (argc >= 7) {
        warmup = atoi(argv[6]);
    }

    if (T_k <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        fprintf(stderr, "Invalid arguments.\n");
        print_usage(argv[0]);
        return 1;
    }
    if (num_kv_heads > num_heads) {
        fprintf(stderr, "H_kv must be <= H.\n");
        return 1;
    }

    int aligned_head_dim = align_int(head_dim, 16);
    size_t q_elems = (size_t)num_heads * (size_t)aligned_head_dim;
    size_t kv_elems = (size_t)num_kv_heads * (size_t)T_k * (size_t)aligned_head_dim;
    size_t q_bytes = align_size(q_elems * sizeof(float), 64);
    size_t kv_bytes = align_size(kv_elems * sizeof(float), 64);

    float *q = (float *)aligned_alloc(64, q_bytes);
    float *k = (float *)aligned_alloc(64, kv_bytes);
    float *v = (float *)aligned_alloc(64, kv_bytes);
    float *out = (float *)aligned_alloc(64, q_bytes);

    if (!q || !k || !v || !out) {
        fprintf(stderr, "Failed to allocate buffers.\n");
        free(q);
        free(k);
        free(v);
        free(out);
        return 1;
    }

    srand(42);
    random_init(q, q_elems);
    random_init(k, kv_elems);
    random_init(v, kv_elems);
    memset(out, 0, q_elems * sizeof(float));

    printf("=== Flash Attention Decode Microbenchmark ===\n");
    printf("Context (T_k): %d\n", T_k);
    printf("Heads: %d  KV Heads: %d\n", num_heads, num_kv_heads);
    printf("Head dim: %d  Aligned: %d\n", head_dim, aligned_head_dim);
    printf("Iterations: %d  Warmup: %d\n", iters, warmup);
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Tile_k: %d  Fast exp kind: %d\n",
           ck_flash_attn_choose_tile_k(head_dim),
           ck_flash_attn_fast_exp_kind());
    printf("\n");

    print_cpu_info();

    for (int i = 0; i < warmup; ++i) {
        ck_attention_flash_decode_wrapper(q, k, v, out,
                                          num_heads,
                                          num_kv_heads,
                                          T_k,
                                          T_k,
                                          head_dim,
                                          aligned_head_dim);
    }

    double start_ms = get_time_ms();
    for (int i = 0; i < iters; ++i) {
        ck_attention_flash_decode_wrapper(q, k, v, out,
                                          num_heads,
                                          num_kv_heads,
                                          T_k,
                                          T_k,
                                          head_dim,
                                          aligned_head_dim);
    }
    double elapsed_ms = get_time_ms() - start_ms;

    double avg_ms = elapsed_ms / (double)iters;
    double avg_us = avg_ms * 1000.0;
    double tok_s = 1000.0 / avg_ms;

    printf("\n=== Results ===\n");
    printf("Total time:   %.2f ms\n", elapsed_ms);
    printf("Avg per iter: %.2f us\n", avg_us);
    printf("Throughput:   %.2f tok/s (decode)\n", tok_s);
    printf("Sanity: out[0]=%f\n", out[0]);

    free(q);
    free(k);
    free(v);
    free(out);
    return 0;
}
