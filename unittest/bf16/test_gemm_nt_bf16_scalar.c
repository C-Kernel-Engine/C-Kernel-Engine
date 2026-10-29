#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bf16_utils.h"

void gemm_nt_bf16_row_range(const float *A, const void *B, const float *bias,
                           float *C, int M, int N, int K,
                           int row_begin, int row_end);

static float a[8 * 257];
static uint16_t b[513 * 257];
static float bias[513];
static float actual[8 * 513 + 2];
static float expected[8 * 513];

static int run_case(int m, int n, int k, int with_bias)
{
    for (int i = 0; i < m * k; ++i) {
        float value = (float)((i * 37) % 101 - 50) / 19.0f;
        a[i] = i == 0 ? 1.00390625f : value;
    }
    for (int i = 0; i < n * k; ++i) {
        b[i] = float_to_bf16((float)((i * 17) % 89 - 44) / 23.0f);
    }
    for (int j = 0; j < n; ++j) {
        bias[j] = (float)(j % 13 - 6) / 7.0f;
    }
    actual[0] = -317.25f;
    actual[m * n + 1] = -317.25f;
    gemm_nt_bf16_row_range(a, b, with_bias ? bias : NULL,
                           actual + 1, m, n, k, 0, m);
    if (actual[0] != -317.25f || actual[m * n + 1] != -317.25f) {
        fprintf(stderr, "BF16 GEMM wrote outside the output\n");
        return 1;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = with_bias ? bias[j] : 0.0f;
            for (int z = 0; z < k; ++z) {
                const float rounded = bf16_to_float(float_to_bf16(a[i * k + z]));
                sum += rounded * bf16_to_float(b[j * k + z]);
            }
            expected[i * n + j] = sum;
        }
    }
    if (memcmp(actual + 1, expected, (size_t)m * (size_t)n * sizeof(float)) != 0) {
        fprintf(stderr, "BF16 GEMM changed reduction order M=%d N=%d K=%d bias=%d\n",
                m, n, k, with_bias);
        return 1;
    }
    b[0] ^= 0x0040;
    gemm_nt_bf16_row_range(a, b, with_bias ? bias : NULL,
                           actual + 1, m, n, k, 0, m);
    if (memcmp(actual + 1, expected, (size_t)m * (size_t)n * sizeof(float)) == 0) {
        fprintf(stderr, "BF16 GEMM ignored a changed weight\n");
        return 1;
    }
    return 0;
}

int main(void)
{
    const int shapes[][3] = {{1, 7, 13}, {3, 9, 257}, {8, 513, 129}};
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
        for (int with_bias = 0; with_bias <= 1; ++with_bias) {
            if (run_case(shapes[i][0], shapes[i][1], shapes[i][2], with_bias)) {
                return 1;
            }
        }
    }
    puts("BF16 GEMM scalar reduction contract: PASS");
    return 0;
}
