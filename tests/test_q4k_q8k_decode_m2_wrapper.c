/* Real cache-backed/threaded M=2 entry against the selected M=1 decode. */
#include "ckernel_quant.h"
#include "ck_threadpool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int requested_workers = 1;
static int pack_calls = 0;

int ck_get_num_threads(void) { return requested_workers; }
int ck_get_physical_cores(void) { return requested_workers; }

void __real_pack_q4_k_to_packed_meta_x8(const void *, void *, int, int);
void __wrap_pack_q4_k_to_packed_meta_x8(const void *src, void *dst, int n, int k)
{
    ++pack_calls;
    __real_pack_q4_k_to_packed_meta_x8(src, dst, n, k);
}

void gemm_nt_q4_k_q8_k_decode_m2(
    const void *, const void *, const float *, float *, int, int, int);
void gemv_q4_k_q8_k_repacked_parallel_dispatch(
    float *, const void *, const void *, int, int);
void ck_q4k_packed_weight_cache_clear(void);

static uint32_t next_u32(uint32_t *seed)
{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return *seed;
}

static int run_case(int n, int k, int with_bias)
{
    const int blocks = k / QK_K;
    const size_t weight_count = (size_t)n * (size_t)blocks;
    const size_t row_bytes = (size_t)blocks * sizeof(block_q8_K);
    const size_t output_bytes = (size_t)2 * (size_t)n * sizeof(float);
    block_q4_K *weights = calloc(2 * weight_count, sizeof(*weights));
    block_q8_K *input = calloc(2 * (size_t)blocks, sizeof(*input));
    float *bias = calloc((size_t)n, sizeof(*bias));
    float *expected = calloc((size_t)2 * (size_t)n, sizeof(*expected));
    float *guarded = malloc(output_bytes + 2 * sizeof(float));
    float *actual = guarded ? guarded + 1 : NULL;
    if (!weights || !input || !bias || !expected || !guarded) {
        free(weights); free(input); free(bias); free(expected); free(guarded);
        return 0;
    }
    uint32_t seed = 0x587aee35u + (uint32_t)n * 13u + (uint32_t)k;
    for (size_t i = 0; i < 2 * weight_count; ++i) {
        weights[i].d = (ck_half)(0x3400u + (next_u32(&seed) & 0x7ffu));
        weights[i].dmin = (ck_half)(0x3000u + (next_u32(&seed) & 0x7ffu));
        for (int j = 0; j < K_SCALE_SIZE; ++j)
            weights[i].scales[j] = (uint8_t)next_u32(&seed);
        for (int j = 0; j < QK_K / 2; ++j)
            weights[i].qs[j] = (uint8_t)next_u32(&seed);
    }
    for (int row = 0; row < 2; ++row) {
        for (int b = 0; b < blocks; ++b) {
            block_q8_K *x = &input[row * blocks + b];
            x->d = (row ? -0.023f : 0.041f) * (float)(b + 1);
            for (int j = 0; j < QK_K; ++j)
                x->qs[j] = (int8_t)((int)(next_u32(&seed) % 255u) - 127);
            for (int j = 0; j < QK_K / 16; ++j) {
                int sum = 0;
                for (int t = 0; t < 16; ++t) sum += x->qs[j * 16 + t];
                x->bsums[j] = (int16_t)sum;
            }
        }
    }
    for (int col = 0; col < n; ++col)
        bias[col] = (float)(col - n / 2) * 0.03125f;

    int pass = 1;
    ck_q4k_packed_weight_cache_clear();
    pack_calls = 0;
    for (int identity = 0; identity < 2; ++identity) {
        block_q4_K *weight = weights + (size_t)identity * weight_count;
        guarded[0] = 71.25f;
        guarded[2 * n + 1] = -19.5f;
        memset(actual, 0x7f, output_bytes);
        /* First call must pack this identity; second must use its cached view. */
        gemm_nt_q4_k_q8_k_decode_m2(input, weight, with_bias ? bias : NULL,
                                     actual, 2, n, k);
        pass &= pack_calls == identity + 1;
        for (int row = 0; row < 2; ++row) {
            gemv_q4_k_q8_k_repacked_parallel_dispatch(
                expected + row * n, weight,
                (const char *)input + (size_t)row * row_bytes, n, k);
            if (with_bias) {
                for (int col = 0; col < n; ++col)
                    expected[row * n + col] += bias[col];
            }
        }
        pass &= pack_calls == identity + 1;
        pass &= memcmp(actual, expected, output_bytes) == 0;
        pass &= guarded[0] == 71.25f && guarded[2 * n + 1] == -19.5f;
        memset(actual, 0x7f, output_bytes);
        gemm_nt_q4_k_q8_k_decode_m2(input, weight, with_bias ? bias : NULL,
                                     actual, 2, n, k);
        pass &= pack_calls == identity + 1;
        pass &= memcmp(actual, expected, output_bytes) == 0;
    }
    /* The two identities differ; a stale cache entry must not reproduce row 0. */
    float *other = calloc((size_t)2 * (size_t)n, sizeof(*other));
    if (!other) pass = 0;
    else {
        gemm_nt_q4_k_q8_k_decode_m2(input, weights, with_bias ? bias : NULL,
                                     other, 2, n, k);
        pass &= memcmp(other, expected, output_bytes) != 0;
        block_q8_K *second = &input[blocks];
        second->qs[0] = (int8_t)(second->qs[0] ^ 0x5a);
        int sum = 0;
        for (int t = 0; t < 16; ++t) sum += second->qs[t];
        second->bsums[0] = (int16_t)sum;
        gemm_nt_q4_k_q8_k_decode_m2(
            input, weights + weight_count, with_bias ? bias : NULL,
            other, 2, n, k);
        pass &= memcmp(other, expected, (size_t)n * sizeof(float)) == 0;
        pass &= memcmp(other + n, expected + n,
                       (size_t)n * sizeof(float)) != 0;
    }
    free(other);
    ck_q4k_packed_weight_cache_clear();
    if (!pass)
        fprintf(stderr, "wrapper mismatch: workers=%d N=%d K=%d bias=%d packs=%d\n",
                requested_workers, n, k, with_bias, pack_calls);
    free(weights); free(input); free(bias); free(expected); free(guarded);
    return pass;
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    requested_workers = atoi(argv[1]);
    if (requested_workers < 1 || requested_workers > 32) return 2;
    printf("resolved_workers=%d\n",
           ck_threadpool_n_threads(ck_threadpool_global()));
    const int ns[] = {1, 7, 8, 9, 17, 32};
    const int ks[] = {256, 512};
    for (size_t i = 0; i < sizeof(ns) / sizeof(ns[0]); ++i)
        for (size_t j = 0; j < sizeof(ks) / sizeof(ks[0]); ++j)
            for (int bias = 0; bias < 2; ++bias)
                if (!run_case(ns[i], ks[j], bias)) return 1;
    ck_threadpool_global_destroy();
    puts("Q4_K/Q8_K real M=2 wrapper: PASS");
    return 0;
}
