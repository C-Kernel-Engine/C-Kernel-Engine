/* Asset-free parity for the packed Q4_K/Q8_K two-row decode primitive. */
#include "ckernel_quant.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t q4_k_packed_meta_x8_block_size(void);
void pack_q4_k_to_packed_meta_x8(const void *, void *, int, int);
void gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order(
    const void *, const void *, const float *, float *, int, int, int);
void gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(
    const void *, const void *, const float *, float *, int, int, int);

static uint32_t next_u32(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static int run_case(int n, int k, int with_bias) {
    const int blocks = k / QK_K;
    const size_t weight_count = (size_t)n * (size_t)blocks;
    const size_t packed_count = (size_t)((n + 7) / 8) * (size_t)blocks;
    block_q4_K *weights = calloc(weight_count, sizeof(*weights));
    block_q8_K *x = calloc((size_t)2 * (size_t)blocks, sizeof(*x));
    void *packed = calloc(packed_count, q4_k_packed_meta_x8_block_size());
    float *expected = calloc((size_t)2 * (size_t)n, sizeof(*expected));
    float *actual = calloc((size_t)2 * (size_t)n, sizeof(*actual));
    float *changed = calloc((size_t)2 * (size_t)n, sizeof(*changed));
    float *bias = with_bias ? calloc((size_t)n, sizeof(*bias)) : NULL;
    if (!weights || !x || !packed || !expected || !actual || !changed ||
        (with_bias && !bias)) {
        free(weights); free(x); free(packed); free(expected);
        free(actual); free(changed); free(bias);
        return 0;
    }
    uint32_t seed = 0x12345678u + (uint32_t)n * 31u + (uint32_t)k;
    for (size_t i = 0; i < weight_count; ++i) {
        weights[i].d = (ck_half)(0x3400u + (next_u32(&seed) & 0x7ffu));
        weights[i].dmin = (ck_half)(0x3000u + (next_u32(&seed) & 0x7ffu));
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            weights[i].scales[j] = (uint8_t)next_u32(&seed);
        }
        for (int j = 0; j < QK_K / 2; ++j) {
            weights[i].qs[j] = (uint8_t)next_u32(&seed);
        }
    }
    for (int row = 0; row < 2; ++row) {
        for (int b = 0; b < blocks; ++b) {
            block_q8_K *block = &x[(size_t)row * (size_t)blocks + (size_t)b];
            block->d = (row ? -0.021f : 0.037f) * (float)(b + 1);
            for (int j = 0; j < QK_K; ++j) {
                block->qs[j] = (int8_t)((int)(next_u32(&seed) % 255u) - 127);
            }
            for (int j = 0; j < QK_K / 16; ++j) {
                int sum = 0;
                for (int t = 0; t < 16; ++t) sum += block->qs[j * 16 + t];
                block->bsums[j] = (int16_t)sum;
            }
        }
    }
    if (bias) {
        for (int j = 0; j < n; ++j) bias[j] = (float)(j - n / 2) * 0.03125f;
    }
    pack_q4_k_to_packed_meta_x8(weights, packed, n, k);
    for (int row = 0; row < 2; ++row) {
        gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order(
            x + (size_t)row * (size_t)blocks, packed, bias,
            expected + (size_t)row * (size_t)n, 1, n, k);
    }
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(
        x, packed, bias, actual, 2, n, k);
    int pass = memcmp(expected, actual, (size_t)2 * (size_t)n * sizeof(float)) == 0;
    if (!pass) fprintf(stderr, "M=2 decode differs: N=%d K=%d bias=%d\n", n, k, with_bias);
    /* Alter one request's quantized row: the other output must stay exact,
     * while the changed row must be observably different. */
    x[blocks].qs[0] = (int8_t)(x[blocks].qs[0] ^ 0x5a);
    x[blocks].bsums[0] = 0;
    for (int t = 0; t < 16; ++t) x[blocks].bsums[0] += x[blocks].qs[t];
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(
        x, packed, bias, changed, 2, n, k);
    pass &= memcmp(changed, expected, (size_t)n * sizeof(float)) == 0;
    pass &= memcmp(changed + n, expected + n, (size_t)n * sizeof(float)) != 0;
    for (int i = 0; i < 2 * n; ++i) actual[i] = 123.0f;
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(
        x, packed, bias, actual, 1, n, k);
    for (int i = 0; i < 2 * n; ++i) pass &= actual[i] == 123.0f;
    free(weights); free(x); free(packed); free(expected);
    free(actual); free(changed); free(bias);
    return pass;
}

int main(void) {
    const int ns[] = {1, 7, 8, 9, 17, 32};
    const int ks[] = {256, 512, 1024};
    for (size_t i = 0; i < sizeof(ns) / sizeof(ns[0]); ++i) {
        for (size_t j = 0; j < sizeof(ks) / sizeof(ks[0]); ++j) {
            for (int bias = 0; bias < 2; ++bias) {
                if (!run_case(ns[i], ks[j], bias)) return 1;
            }
        }
    }
    puts("Q4_K/Q8_K decode M=2 parity: PASS");
    return 0;
}
