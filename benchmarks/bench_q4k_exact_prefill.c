/* Focused benchmark for the production-exact Q4_K x Q8_K prefill provider. */

#include "ck_threadpool.h"
#include "ckernel_quant.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void quantize_row_q8_k(const float *x, void *vy, int k);
extern size_t q4_k_packed_meta_x8_block_size(void);
extern void pack_q4_k_to_packed_meta_x8(const void *src, void *dst, int n, int k);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_mreuse(
        const void *a_q8, const void *b_packed_x8, const float *bias, float *c,
        int m, int n, int k, int tile_m, int active_threads);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_4m(
        const void *a_q8, const void *b_packed_x8, const float *bias, float *c,
        int m, int n, int k, int active_threads);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_8m(
        const void *a_q8, const void *b_packed_x8, const float *bias, float *c,
        int m, int n, int k, int active_threads);
extern size_t q4_k_packed_vnni_x8_block_size(void);
extern void pack_q4_k_to_packed_vnni_x8(
        const void *src, void *dst, int n, int k);
extern void gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m(
        const void *a_q8, const void *b_packed_vnni_x8, const float *bias,
        float *c, int m, int n, int k, int active_threads);

static uint32_t rng_state = 0x12345678u;

static uint32_t rng_u32(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static float rng_f32(void)
{
    return ((float)(rng_u32() >> 8) / (float)0x00ffffffu) * 2.0f - 1.0f;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void fill_q4_k(block_q4_K *weights, int n, int k)
{
    const int blocks = n * (k / QK_K);
    for (int b = 0; b < blocks; ++b) {
        block_q4_K *block = &weights[b];
        block->d = GGML_FP32_TO_FP16(0.03125f);
        block->dmin = GGML_FP32_TO_FP16(0.00390625f);
        for (size_t i = 0; i < sizeof(block->scales); ++i) {
            block->scales[i] = (uint8_t)(rng_u32() & 0x3fu);
        }
        for (size_t i = 0; i < sizeof(block->qs); ++i) {
            block->qs[i] = (uint8_t)(rng_u32() & 0xffu);
        }
    }
}

static int parse_positive(const char *value, int fallback)
{
    const int parsed = value ? atoi(value) : 0;
    return parsed > 0 ? parsed : fallback;
}

static void *alloc_aligned(size_t bytes)
{
    void *ptr = NULL;
    return posix_memalign(&ptr, 64, bytes) == 0 ? ptr : NULL;
}

static void run_provider(const char *provider,
                         const void *a_q8,
                         const void *weights_packed,
                         const void *weights_packed_vnni,
                         const float *bias,
                         float *output,
                         int m, int n, int k,
                         int tile_m, int threads)
{
    if (strcmp(provider, "4m") == 0) {
        gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_4m(
                a_q8, weights_packed, bias, output, m, n, k, threads);
        return;
    }
    if (strcmp(provider, "8m") == 0) {
        gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_8m(
                a_q8, weights_packed, bias, output, m, n, k, threads);
        return;
    }
    if (strcmp(provider, "4m-vnni-x8") == 0) {
        gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m(
                a_q8, weights_packed_vnni, bias, output,
                m, n, k, threads);
        return;
    }
    gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_mreuse(
            a_q8, weights_packed, bias, output,
            m, n, k, tile_m, threads);
}

int main(int argc, char **argv)
{
    int m = 1028;
    int n = 4096;
    int k = 4096;
    int threads = 20;
    int tile_m = 8;
    int warmup = 1;
    int iterations = 3;
    const char *provider = "baseline";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) m = parse_positive(argv[++i], m);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = parse_positive(argv[++i], n);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = parse_positive(argv[++i], k);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = parse_positive(argv[++i], threads);
        else if (strcmp(argv[i], "--tile-m") == 0 && i + 1 < argc) tile_m = parse_positive(argv[++i], tile_m);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = parse_positive(argv[++i], warmup);
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = parse_positive(argv[++i], iterations);
        else if (strcmp(argv[i], "--provider") == 0 && i + 1 < argc) provider = argv[++i];
    }

    if ((k % QK_K) != 0 || m <= 0 || n <= 0 ||
        (strcmp(provider, "baseline") != 0 && strcmp(provider, "4m") != 0 &&
         strcmp(provider, "8m") != 0 &&
         strcmp(provider, "4m-vnni-x8") != 0)) {
        fprintf(stderr, "invalid shape M=%d N=%d K=%d\n", m, n, k);
        return 2;
    }

    const size_t a_count = (size_t)m * (size_t)k;
    const size_t a_q8_bytes = (size_t)m * (size_t)(k / QK_K) * sizeof(block_q8_K);
    const size_t w_count = (size_t)n * (size_t)(k / QK_K);
    const size_t w_packed_bytes = (size_t)((n + 7) / 8) *
                                  (size_t)(k / QK_K) *
                                  q4_k_packed_meta_x8_block_size();
    const size_t w_packed_vnni_bytes = (size_t)((n + 7) / 8) *
                                       (size_t)(k / QK_K) *
                                       q4_k_packed_vnni_x8_block_size();
    const size_t c_count = (size_t)m * (size_t)n;

    float *a = alloc_aligned(a_count * sizeof(float));
    void *a_q8 = alloc_aligned(a_q8_bytes);
    block_q4_K *weights = alloc_aligned(w_count * sizeof(block_q4_K));
    void *weights_packed = alloc_aligned(w_packed_bytes);
    void *weights_packed_vnni = alloc_aligned(w_packed_vnni_bytes);
    float *bias = alloc_aligned((size_t)n * sizeof(float));
    float *output = alloc_aligned(c_count * sizeof(float));
    float *reference = alloc_aligned(c_count * sizeof(float));
    if (!a || !a_q8 || !weights || !weights_packed ||
        !weights_packed_vnni ||
        !bias || !output || !reference) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    for (size_t i = 0; i < a_count; ++i) a[i] = rng_f32();
    for (int i = 0; i < n; ++i) bias[i] = rng_f32() * 0.01f;
    fill_q4_k(weights, n, k);
    for (int row = 0; row < m; ++row) {
        quantize_row_q8_k(a + (size_t)row * (size_t)k,
                          (uint8_t *)a_q8 + (size_t)row *
                          (size_t)(k / QK_K) * sizeof(block_q8_K), k);
    }
    pack_q4_k_to_packed_meta_x8(weights, weights_packed, n, k);
    pack_q4_k_to_packed_vnni_x8(weights, weights_packed_vnni, n, k);

    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool) {
        fprintf(stderr, "threadpool initialization failed\n");
        return 2;
    }

    if (strcmp(provider, "baseline") != 0) {
        run_provider("baseline", a_q8, weights_packed,
                     weights_packed_vnni, bias, reference,
                     m, n, k, tile_m, threads);
        run_provider(provider, a_q8, weights_packed,
                     weights_packed_vnni, bias, output,
                     m, n, k, tile_m, threads);
        if (memcmp(reference, output, c_count * sizeof(float)) != 0) {
            size_t first = 0;
            while (first < c_count && reference[first] == output[first]) ++first;
            fprintf(stderr,
                    "provider mismatch at index=%zu baseline=%.9g candidate=%.9g\n",
                    first, reference[first], output[first]);
            return 3;
        }
    }

    for (int i = 0; i < warmup; ++i) {
        run_provider(provider, a_q8, weights_packed,
                     weights_packed_vnni, bias, output,
                     m, n, k, tile_m, threads);
    }

    const double start = now_ms();
    for (int i = 0; i < iterations; ++i) {
        run_provider(provider, a_q8, weights_packed,
                     weights_packed_vnni, bias, output,
                     m, n, k, tile_m, threads);
    }
    const double elapsed_ms = (now_ms() - start) / (double)iterations;

    double checksum = 0.0;
    for (size_t i = 0; i < c_count; ++i) checksum += (double)output[i];
    const double operations = 2.0 * (double)m * (double)n * (double)k;
    printf("provider=%s M=%d N=%d K=%d threads=%d tile_m=%d "
           "time_ms=%.3f gflops=%.3f checksum=%.17g\n",
           provider, m, n, k, threads, tile_m, elapsed_ms,
           operations / (elapsed_ms * 1.0e6), checksum);

    ck_threadpool_global_destroy();
    free(a);
    free(a_q8);
    free(weights);
    free(weights_packed);
    free(weights_packed_vnni);
    free(bias);
    free(output);
    free(reference);
    return 0;
}
