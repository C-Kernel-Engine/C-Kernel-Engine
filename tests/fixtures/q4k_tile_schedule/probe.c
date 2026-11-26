#define _POSIX_C_SOURCE 200809L

#include "gemm_kernels_q4k_q8k_vnni.c"

#include <assert.h>
#include <stdatomic.h>

static void record_tiles(int begin, int end, int step, void *opaque)
{
    _Atomic int *counts = opaque;
    for (int tile = begin; tile < end; tile += step) {
        atomic_fetch_add_explicit(&counts[tile], 1, memory_order_relaxed);
    }
}

static void check_coverage(const char *policy, int active, int jobs, int default_grain)
{
    _Atomic int counts[51];
    for (int i = 0; i < 51; ++i) atomic_init(&counts[i], 0);
    assert(setenv("CK_EXPERIMENT_Q4K_TILE_SCHEDULE", policy, 1) == 0);
    ck_q4k_dispatch_tiles(ck_threadpool_global(), active, jobs, 17,
                          default_grain, "coverage", record_tiles, counts);
    for (int i = 0; i < 51; ++i) assert(atomic_load(&counts[i]) == (i < jobs));
}

static void check_arithmetic(const char *const *policies, int policy_count, int active)
{
    /* Q8 activation storage and packed weights are identical in every replay.
     * Both the row tile and the output group end in a partial tile. */
    enum { rows = 17, cols = 19, width = 2 * QK_K, blocks = width / QK_K };
    block_q4_K weights[cols * blocks];
    block_q8_K input[rows * blocks];
    block_q4_K_packed_meta_x8 meta[(cols + 7) / 8 * blocks];
    block_q4_K_packed_vnni_x8 vnni[(cols + 7) / 8 * blocks];
    float expected[rows * cols], actual[rows * cols], bias[cols];

    for (int i = 0; i < cols; ++i) bias[i] = 0.06f * (i - 3);
    for (int i = 0; i < cols * blocks; ++i) {
        weights[i].d = CK_FP32_TO_FP16(0.006f * (i + 1));
        weights[i].dmin = CK_FP32_TO_FP16(0.003f * (i + 1));
        for (int j = 0; j < K_SCALE_SIZE; ++j) weights[i].scales[j] = 5 + (i + 3 * j) % 50;
        for (int j = 0; j < QK_K / 2; ++j) weights[i].qs[j] = (17 * i + 23 * j) % 256;
    }
    for (int i = 0; i < rows * blocks; ++i) {
        input[i].d = 0.001f * (i + 1);
        for (int j = 0; j < QK_K; ++j) input[i].qs[j] = (13 * i + 3 * j) % 127 - 63;
        for (int j = 0; j < QK_K / 16; ++j) {
            input[i].bsums[j] = 0;
            for (int k = 0; k < 16; ++k) input[i].bsums[j] += input[i].qs[16 * j + k];
        }
    }
    pack_q4_k_to_packed_meta_x8(weights, meta, cols, width);
    pack_q4_k_to_packed_vnni_x8(weights, vnni, cols, width);

    void (*kernels[])(const void *, const void *, const float *, float *, int, int, int, int) = {
        gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_8m,
        gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m,
    };
    const void *packed[] = {meta, vnni};
    for (int i = 0; i < 2; ++i) {
        assert(unsetenv("CK_EXPERIMENT_Q4K_TILE_SCHEDULE") == 0);
        kernels[i](input, packed[i], bias, expected, rows, cols, width, 1);
        for (int k = 0; k < rows * cols; ++k) assert(isfinite(expected[k]));
        for (int j = 0; j < policy_count; ++j) {
            assert(setenv("CK_EXPERIMENT_Q4K_TILE_SCHEDULE", policies[j], 1) == 0);
            for (int k = 0; k < rows * cols; ++k) actual[k] = NAN;
            kernels[i](input, packed[i], bias, actual, rows, cols, width, active);
            assert(memcmp(actual, expected, sizeof(expected)) == 0);
        }
    }
}

int main(void)
{
    const char *policies[] = {"default", "contiguous", "grain4", "whole-group", "unknown"};
    const int count = (int)(sizeof(policies) / sizeof(policies[0]));
    const int active = ck_threadpool_n_threads(ck_threadpool_global());
    for (int i = 0; i < count; ++i) {
        check_coverage(policies[i], 1, 1, 0);
        check_coverage(policies[i], active, 3, 4);
        check_coverage(policies[i], active, 37, 0);
    }
    check_arithmetic(policies, count, active);
    return 0;
}
