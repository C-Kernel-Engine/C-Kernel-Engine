/* Asset-free parity for the packed Q4_K/Q8_K two-row decode primitive. */
#include "ckernel_quant.h"

#include <math.h>
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

/* Independent raw Q4_K field interpretation: do not call the production
 * packer or its scale-unpacking helper. Each block performs its own integer
 * dot, then the declared decode FP32 update. */
static void raw_reference(const block_q4_K *weights, const block_q8_K *x,
                          const float *bias, float *out, int n, int k) {
    const int blocks = k / QK_K;
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < n; ++col) {
            float value = 0.0f, minimum = 0.0f;
            for (int b = 0; b < blocks; ++b) {
                const block_q4_K *w = &weights[(size_t)col * (size_t)blocks + (size_t)b];
                const block_q8_K *a = &x[(size_t)row * (size_t)blocks + (size_t)b];
                int32_t integer_value = 0, integer_minimum = 0;
                for (int pair = 0; pair < 4; ++pair) {
                    const int sub0 = pair * 2, sub1 = sub0 + 1;
                    int scale0, scale1, min0, min1;
                    if (sub0 < 4) {
                        scale0 = w->scales[sub0] & 63;
                        min0 = w->scales[sub0 + 4] & 63;
                    } else {
                        int t = sub0 - 4;
                        scale0 = (w->scales[8 + t] & 15) | (w->scales[t] & 192) >> 2;
                        min0 = (w->scales[8 + t] >> 4) | (w->scales[4 + t] & 192) >> 2;
                    }
                    if (sub1 < 4) {
                        scale1 = w->scales[sub1] & 63;
                        min1 = w->scales[sub1 + 4] & 63;
                    } else {
                        int t = sub1 - 4;
                        scale1 = (w->scales[8 + t] & 15) | (w->scales[t] & 192) >> 2;
                        min1 = (w->scales[8 + t] >> 4) | (w->scales[4 + t] & 192) >> 2;
                    }
                    int dot0 = 0, dot1 = 0;
                    for (int i = 0; i < 32; ++i) {
                        const uint8_t q = w->qs[pair * 32 + i];
                        dot0 += (q & 15) * (int)a->qs[pair * 64 + i];
                        dot1 += (q >> 4) * (int)a->qs[pair * 64 + 32 + i];
                    }
                    const int qsum0 = (int)a->bsums[pair * 4] + (int)a->bsums[pair * 4 + 1];
                    const int qsum1 = (int)a->bsums[pair * 4 + 2] + (int)a->bsums[pair * 4 + 3];
                    integer_value += scale0 * dot0 + scale1 * dot1;
                    integer_minimum += min0 * qsum0 + min1 * qsum1;
                }
                const float d = ck_fp16_to_fp32_soft(w->d) * a->d;
                const float dm = ck_fp16_to_fp32_soft(w->dmin) * a->d;
                value = fmaf((float)integer_value, d, value);
                minimum = fmaf((float)integer_minimum, dm, minimum);
            }
            out[(size_t)row * (size_t)n + (size_t)col] =
                (value - minimum) + (bias ? bias[col] : 0.0f);
        }
    }
}

static int run_case(int n, int k, int with_bias) {
    const int blocks = k / QK_K;
    const size_t weight_count = (size_t)n * (size_t)blocks;
    const size_t packed_count = (size_t)((n + 7) / 8) * (size_t)blocks;
    const size_t packed_bytes = packed_count * q4_k_packed_meta_x8_block_size();
    const size_t output_bytes = (size_t)2 * (size_t)n * sizeof(float);
    const size_t activation_bytes = (size_t)2 * (size_t)blocks * sizeof(block_q8_K);
    block_q4_K *weights = calloc(weight_count, sizeof(*weights));
    block_q8_K *x = calloc((size_t)2 * (size_t)blocks, sizeof(*x));
    void *packed = calloc(1, packed_bytes);
    float *expected = calloc((size_t)2 * (size_t)n, sizeof(*expected));
    float *oracle = calloc((size_t)2 * (size_t)n, sizeof(*oracle));
    float *guarded = calloc((size_t)2 * (size_t)n + 2, sizeof(*guarded));
    float *actual = guarded ? guarded + 1 : NULL;
    float *changed = calloc((size_t)2 * (size_t)n, sizeof(*changed));
    block_q4_K *weight_copy = malloc(weight_count * sizeof(*weight_copy));
    block_q8_K *activation_copy = malloc(activation_bytes);
    void *packed_copy = malloc(packed_bytes);
    float *bias = with_bias ? calloc((size_t)n, sizeof(*bias)) : NULL;
    if (!weights || !x || !packed || !expected || !oracle || !guarded || !changed ||
        !weight_copy || !activation_copy || !packed_copy ||
        (with_bias && !bias)) {
        free(weights); free(x); free(packed); free(expected);
        free(oracle); free(guarded); free(changed); free(bias);
        free(weight_copy); free(activation_copy); free(packed_copy);
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
    /* Boundaries near cancellation and conversion: exact, finite FP16 scales
     * remain representable, while alternating signed Q8 values oppose sums. */
    if (k >= 512) {
        x[0].d = 0.125f;
        x[blocks].d = -0.125f;
    }
    memcpy(weight_copy, weights, weight_count * sizeof(*weight_copy));
    memcpy(activation_copy, x, activation_bytes);
    guarded[0] = 81.25f;
    guarded[(size_t)2 * (size_t)n + 1] = -43.5f;
    pack_q4_k_to_packed_meta_x8(weights, packed, n, k);
    memcpy(packed_copy, packed, packed_bytes);
    for (int row = 0; row < 2; ++row) {
        gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order(
            x + (size_t)row * (size_t)blocks, packed, bias,
            expected + (size_t)row * (size_t)n, 1, n, k);
    }
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(
        x, packed, bias, actual, 2, n, k);
    raw_reference(weights, x, bias, oracle, n, k);
    int pass = memcmp(expected, actual, output_bytes) == 0;
    if (!pass) fprintf(stderr, "M=2 decode differs: N=%d K=%d bias=%d\n", n, k, with_bias);
    if (memcmp(oracle, actual, output_bytes) != 0) {
        fprintf(stderr, "raw Q4_K reference differs: N=%d K=%d bias=%d\n", n, k, with_bias);
        pass = 0;
    }
    /* Alter one request's quantized row: the other output must stay exact,
     * while the changed row must be observably different. */
    x[blocks].qs[0] = (int8_t)(x[blocks].qs[0] ^ 0x5a);
    x[blocks].bsums[0] = 0;
    for (int t = 0; t < 16; ++t) x[blocks].bsums[0] += x[blocks].qs[t];
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(
        x, packed, bias, changed, 2, n, k);
    pass &= memcmp(changed, expected, (size_t)n * sizeof(float)) == 0;
    pass &= memcmp(changed + n, expected + n, (size_t)n * sizeof(float)) != 0;
    memcpy(x, activation_copy, activation_bytes);
    for (int i = 0; i < 2 * n; ++i) actual[i] = 123.0f;
    for (int invalid_m = 0; invalid_m <= 3; ++invalid_m) {
        if (invalid_m == 2) continue;
        gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(
            x, packed, bias, actual, invalid_m, n, k);
    }
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(x, packed, bias, actual, 2, 0, k);
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(x, packed, bias, actual, 2, n, 0);
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(x, packed, bias, actual, 2, n, k + 1);
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(NULL, packed, bias, actual, 2, n, k);
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(x, NULL, bias, actual, 2, n, k);
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order_m2(x, packed, bias, NULL, 2, n, k);
    for (int i = 0; i < 2 * n; ++i) pass &= actual[i] == 123.0f;
    pass &= guarded[0] == 81.25f && guarded[(size_t)2 * (size_t)n + 1] == -43.5f;
    pass &= memcmp(weights, weight_copy, weight_count * sizeof(*weight_copy)) == 0;
    pass &= memcmp(x, activation_copy, activation_bytes) == 0;
    pass &= memcmp(packed, packed_copy, packed_bytes) == 0;
    if (!pass) fprintf(stderr, "decode M=2 contract failed: N=%d K=%d bias=%d\n",
                       n, k, with_bias);
    free(weights); free(x); free(packed); free(expected);
    free(oracle); free(guarded); free(changed); free(bias);
    free(weight_copy); free(activation_copy); free(packed_copy);
    return pass;
}

int main(void) {
    const int ns[] = {1, 7, 8, 9, 17, 32};
    const int ks[] = {256, 512, 1024, 5120};
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
