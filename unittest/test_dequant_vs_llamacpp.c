/**
 * test_dequant_vs_llamacpp.c - Compare C-Kernel-Engine dequantization against llama.cpp
 *
 * This test links against llama.cpp's ggml library to use their dequantization
 * functions as ground truth, then compares against our implementations.
 *
 * Build:
 *   gcc -O2 -o test_dequant_vs_llamacpp test_dequant_vs_llamacpp.c \
 *       -I../llama.cpp/ggml/include -I../include \
 *       -L../llama.cpp/build/ggml/src -lggml \
 *       -lm -Wl,-rpath,../llama.cpp/build/ggml/src
 *
 * Run:
 *   ./test_dequant_vs_llamacpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * llama.cpp/ggml types and functions (from ggml-quants.h)
 * ============================================================================ */

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK_K 256
#define K_SCALE_SIZE 12

typedef uint16_t ggml_half;

/* Q4_0 block */
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

/* Q8_0 block */
typedef struct {
    ggml_half d;
    int8_t qs[QK8_0];
} block_q8_0;

/* Q4_K block */
typedef struct {
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
} block_q4_K;

/* Q6_K block */
typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    ggml_half d;
} block_q6_K;

/* ============================================================================
 * FP16 conversion (IEEE 754)
 * ============================================================================ */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t result;

    if (exp == 0) {
        if (mant == 0) {
            result = sign;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            result = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        result = sign | 0x7F800000 | (mant << 13);
    } else {
        result = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    union { uint32_t u; float f; } u;
    u.u = result;
    return u.f;
}

/* ============================================================================
 * llama.cpp reference dequantization (copied exactly from ggml-quants.c)
 * ============================================================================ */

/* get_scale_min_k4 - extract scale and min for Q4_K sub-block */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* Dequantize Q4_0 block (32 values from 18 bytes) */
static void dequant_q4_0_llamacpp(const block_q4_0 *x, float *y) {
    const float d = fp16_to_fp32(x->d);
    for (int i = 0; i < QK4_0 / 2; i++) {
        const int x0 = (x->qs[i] & 0x0F) - 8;
        const int x1 = (x->qs[i] >> 4) - 8;
        y[i] = x0 * d;
        y[i + QK4_0 / 2] = x1 * d;
    }
}

/* Dequantize Q8_0 block (32 values from 34 bytes) */
static void dequant_q8_0_llamacpp(const block_q8_0 *x, float *y) {
    const float d = fp16_to_fp32(x->d);
    for (int i = 0; i < QK8_0; i++) {
        y[i] = x->qs[i] * d;
    }
}

/* Dequantize Q4_K block (256 values from 144 bytes) */
static void dequant_q4_k_llamacpp(const block_q4_K *x, float *y) {
    const float d = fp16_to_fp32(x->d);
    const float dmin = fp16_to_fp32(x->dmin);

    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is, x->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;

        for (int l = 0; l < 32; l++) {
            y[j + l] = d1 * (x->qs[j / 2 + l] & 0xF) - m1;
        }
        for (int l = 0; l < 32; l++) {
            y[j + 32 + l] = d2 * (x->qs[j / 2 + l] >> 4) - m2;
        }
        is += 2;
    }
}

/* Dequantize Q6_K block (256 values from 210 bytes) */
static void dequant_q6_k_llamacpp(const block_q6_K *x, float *y) {
    const float d = fp16_to_fp32(x->d);

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            const uint8_t ql0 = x->ql[n / 2 + l];
            const uint8_t qh0 = x->qh[n / 4 + l];

            const int8_t sc0 = x->scales[n / 16 + 0];
            const int8_t sc1 = x->scales[n / 16 + 1];
            const int8_t sc2 = x->scales[n / 16 + 2];
            const int8_t sc3 = x->scales[n / 16 + 3];

            y[n + l + 0]  = d * sc0 * ((int8_t)((ql0 & 0xF) | ((qh0 & 0x03) << 4)) - 32);
            y[n + l + 32] = d * sc1 * ((int8_t)((ql0 >> 4) | ((qh0 & 0x0C) << 2)) - 32);

            const uint8_t ql1 = x->ql[n / 2 + l + 32];
            const uint8_t qh1 = x->qh[n / 4 + l + 32];

            y[n + l + 64] = d * sc2 * ((int8_t)((ql1 & 0xF) | ((qh1 & 0x03) << 4)) - 32);
            y[n + l + 96] = d * sc3 * ((int8_t)((ql1 >> 4) | ((qh1 & 0x0C) << 2)) - 32);
        }
    }
}

/* ============================================================================
 * C-Kernel-Engine dequantization (our implementation)
 * ============================================================================ */

/* Our Q4_K scale unpacking */
static inline void ck_unpack_q4_k_scales(const uint8_t *scales, uint8_t *sc, uint8_t *m) {
    sc[0] = scales[0] & 0x3F;
    sc[1] = scales[1] & 0x3F;
    sc[2] = scales[2] & 0x3F;
    sc[3] = scales[3] & 0x3F;

    m[0] = scales[4] & 0x3F;
    m[1] = scales[5] & 0x3F;
    m[2] = scales[6] & 0x3F;
    m[3] = scales[7] & 0x3F;

    sc[4] = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4);
    sc[5] = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4);
    sc[6] = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4);
    sc[7] = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4);

    m[4] = (scales[8] >> 4) | ((scales[4] >> 6) << 4);
    m[5] = (scales[9] >> 4) | ((scales[5] >> 6) << 4);
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4);
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4);
}

/* Our Q4_K dequantization */
static void dequant_q4_k_ckernel(const block_q4_K *x, float *y) {
    const float d = fp16_to_fp32(x->d);
    const float dmin = fp16_to_fp32(x->dmin);

    uint8_t sc[8], m[8];
    ck_unpack_q4_k_scales(x->scales, sc, m);

    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        const float d1 = d * sc[is];
        const float m1 = dmin * m[is];
        const float d2 = d * sc[is + 1];
        const float m2 = dmin * m[is + 1];

        for (int l = 0; l < 32; l++) {
            y[j + l] = d1 * (x->qs[j / 2 + l] & 0xF) - m1;
        }
        for (int l = 0; l < 32; l++) {
            y[j + 32 + l] = d2 * (x->qs[j / 2 + l] >> 4) - m2;
        }
        is += 2;
    }
}

/* Our Q8_0 dequantization */
static void dequant_q8_0_ckernel(const block_q8_0 *x, float *y) {
    const float d = fp16_to_fp32(x->d);
    for (int i = 0; i < QK8_0; i++) {
        y[i] = x->qs[i] * d;
    }
}

/* ============================================================================
 * Test Framework
 * ============================================================================ */

#define RED     "\033[91m"
#define GREEN   "\033[92m"
#define YELLOW  "\033[93m"
#define RESET   "\033[0m"

typedef struct {
    const char *name;
    int passed;
    int failed;
    float max_diff;
    int first_diff_idx;
} TestResult;

static int compare_floats(const float *a, const float *b, int n, float tol,
                          float *max_diff, int *first_diff) {
    *max_diff = 0;
    *first_diff = -1;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > *max_diff) {
            *max_diff = diff;
        }
        if (diff > tol && *first_diff < 0) {
            *first_diff = i;
        }
    }
    return *first_diff < 0;
}

static void print_array(const char *name, const float *arr, int n, int max_show) {
    printf("  %s: [", name);
    for (int i = 0; i < (n < max_show ? n : max_show); i++) {
        printf("%.6f%s", arr[i], i < max_show - 1 ? ", " : "");
    }
    if (n > max_show) printf(", ...");
    printf("]\n");
}

/* ============================================================================
 * Test Cases
 * ============================================================================ */

static TestResult test_q4_k_scale_unpacking(void) {
    TestResult result = {"Q4_K Scale Unpacking", 0, 0, 0, -1};

    /* Test multiple scale patterns */
    uint8_t test_cases[][12] = {
        {236, 243, 254, 184, 172, 230, 235, 185, 123, 7, 255, 201},  /* Random */
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},                        /* All zeros */
        {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}, /* All 0xFF */
        {0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0, 0, 0, 0}, /* High bits */
        {0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0xFF, 0xFF, 0xFF, 0xFF},
    };

    for (int tc = 0; tc < 5; tc++) {
        uint8_t sc_llama[8], m_llama[8];
        uint8_t sc_ck[8], m_ck[8];

        /* llama.cpp method */
        for (int j = 0; j < 8; j++) {
            get_scale_min_k4(j, test_cases[tc], &sc_llama[j], &m_llama[j]);
        }

        /* Our method */
        ck_unpack_q4_k_scales(test_cases[tc], sc_ck, m_ck);

        /* Compare */
        int match = 1;
        for (int j = 0; j < 8; j++) {
            if (sc_llama[j] != sc_ck[j] || m_llama[j] != m_ck[j]) {
                match = 0;
                printf("  Case %d, j=%d: llama=(%d,%d) ck=(%d,%d) " RED "MISMATCH" RESET "\n",
                       tc, j, sc_llama[j], m_llama[j], sc_ck[j], m_ck[j]);
                result.failed++;
                break;
            }
        }
        if (match) result.passed++;
    }

    return result;
}

static TestResult test_q4_k_dequant(void) {
    TestResult result = {"Q4_K Dequantization", 0, 0, 0, -1};

    /* Create test block with known values */
    block_q4_K block;
    memset(&block, 0, sizeof(block));

    /* Set scale d=1.0, dmin=0.5 in FP16 */
    block.d = 0x3C00;     /* 1.0 */
    block.dmin = 0x3800;  /* 0.5 */

    /* Set scales: all sub-block scales = 2, mins = 1 */
    for (int i = 0; i < 4; i++) {
        block.scales[i] = 2;      /* scales[0-3] */
        block.scales[i + 4] = 1;  /* mins[0-3] */
    }
    for (int i = 8; i < 12; i++) {
        block.scales[i] = 0x22;   /* scales[4-7]=2, mins[4-7]=2 */
    }

    /* Set some quant values */
    for (int i = 0; i < QK_K / 2; i++) {
        block.qs[i] = 0x84;  /* low=4, high=8 */
    }

    float y_llama[QK_K], y_ck[QK_K];
    dequant_q4_k_llamacpp(&block, y_llama);
    dequant_q4_k_ckernel(&block, y_ck);

    float max_diff;
    int first_diff;
    if (compare_floats(y_llama, y_ck, QK_K, 1e-6f, &max_diff, &first_diff)) {
        result.passed = 1;
        result.max_diff = max_diff;
    } else {
        result.failed = 1;
        result.max_diff = max_diff;
        result.first_diff_idx = first_diff;
        printf("  First diff at [%d]: llama=%.6f, ck=%.6f\n",
               first_diff, y_llama[first_diff], y_ck[first_diff]);
    }

    return result;
}

static TestResult test_q8_0_dequant(void) {
    TestResult result = {"Q8_0 Dequantization", 0, 0, 0, -1};

    block_q8_0 block;

    /* Set scale d=0.01 in FP16 (approximately 0x211E) */
    block.d = 0x211E;

    /* Set quant values: -128 to 127 pattern */
    for (int i = 0; i < QK8_0; i++) {
        block.qs[i] = (int8_t)(i - 16);
    }

    float y_llama[QK8_0], y_ck[QK8_0];
    dequant_q8_0_llamacpp(&block, y_llama);
    dequant_q8_0_ckernel(&block, y_ck);

    float max_diff;
    int first_diff;
    if (compare_floats(y_llama, y_ck, QK8_0, 1e-6f, &max_diff, &first_diff)) {
        result.passed = 1;
        result.max_diff = max_diff;
    } else {
        result.failed = 1;
        result.max_diff = max_diff;
        result.first_diff_idx = first_diff;
    }

    return result;
}

static TestResult test_q4_k_random_blocks(int num_blocks) {
    TestResult result = {"Q4_K Random Blocks", 0, 0, 0, -1};

    for (int b = 0; b < num_blocks; b++) {
        block_q4_K block;

        /* Random data */
        srand(42 + b);
        block.d = rand() & 0xFFFF;
        block.dmin = rand() & 0xFFFF;
        for (int i = 0; i < K_SCALE_SIZE; i++) {
            block.scales[i] = rand() & 0xFF;
        }
        for (int i = 0; i < QK_K / 2; i++) {
            block.qs[i] = rand() & 0xFF;
        }

        float y_llama[QK_K], y_ck[QK_K];
        dequant_q4_k_llamacpp(&block, y_llama);
        dequant_q4_k_ckernel(&block, y_ck);

        float max_diff;
        int first_diff;
        if (compare_floats(y_llama, y_ck, QK_K, 1e-5f, &max_diff, &first_diff)) {
            result.passed++;
            if (max_diff > result.max_diff) result.max_diff = max_diff;
        } else {
            result.failed++;
            if (result.first_diff_idx < 0) {
                result.first_diff_idx = first_diff;
                printf("  Block %d first diff at [%d]: llama=%.6f, ck=%.6f\n",
                       b, first_diff, y_llama[first_diff], y_ck[first_diff]);
            }
        }
    }

    return result;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv) {
    printf("================================================================================\n");
    printf("      C-Kernel-Engine Dequantization vs llama.cpp Reference Test\n");
    printf("================================================================================\n\n");

    TestResult results[10];
    int num_tests = 0;
    int total_passed = 0, total_failed = 0;

    /* Run tests */
    results[num_tests++] = test_q4_k_scale_unpacking();
    results[num_tests++] = test_q4_k_dequant();
    results[num_tests++] = test_q8_0_dequant();
    results[num_tests++] = test_q4_k_random_blocks(100);

    /* Print results */
    printf("\n%-30s %8s %8s %12s\n", "Test", "Passed", "Failed", "Max Diff");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_tests; i++) {
        const char *status;
        const char *color;
        if (results[i].failed == 0) {
            status = "PASS";
            color = GREEN;
            total_passed++;
        } else {
            status = "FAIL";
            color = RED;
            total_failed++;
        }

        printf("%-30s %s%8d%s %8d %12.2e  [%s%s%s]\n",
               results[i].name,
               color, results[i].passed, RESET,
               results[i].failed,
               results[i].max_diff,
               color, status, RESET);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("Total: %d passed, %d failed\n", total_passed, total_failed);
    printf("================================================================================\n");

    return total_failed > 0 ? 1 : 0;
}
