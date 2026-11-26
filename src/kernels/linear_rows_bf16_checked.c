#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "bf16_utils.h"

static int region_valid(const uint16_t *ptr, size_t elements)
{
    return ptr && (uintptr_t)ptr % _Alignof(uint16_t) == 0 &&
           elements <= SIZE_MAX / sizeof(uint16_t) &&
           (uintptr_t)ptr <= UINTPTR_MAX - elements * sizeof(uint16_t);
}

static int overlap(const uint16_t *a, size_t a_count,
                   const uint16_t *b, size_t b_count)
{
    return (uintptr_t)a < (uintptr_t)b + b_count * sizeof(uint16_t) &&
           (uintptr_t)b < (uintptr_t)a + a_count * sizeof(uint16_t);
}

/* Portable reference projection. Products and the ascending reduction
 * stay FP64. A FP32 intermediate is rounded to BF16 RNE on publication. */
int ck_linear_rows_bf16_checked(
    const uint16_t *input, size_t input_elements,
    const uint16_t *weight, size_t weight_elements,
    uint16_t *output, size_t output_elements,
    uint16_t *staged, size_t staged_elements,
    size_t rows, size_t input_width, size_t output_width)
{
    if (!rows || !input_width || !output_width ||
        rows > SIZE_MAX / input_width || output_width > SIZE_MAX / input_width ||
        rows > SIZE_MAX / output_width) return -2;
    const size_t in_count = rows * input_width;
    const size_t w_count = output_width * input_width;
    const size_t out_count = rows * output_width;
    if (input_elements < in_count || weight_elements < w_count ||
        output_elements < out_count || staged_elements < out_count) return -2;
    if (!region_valid(input, input_elements) || !region_valid(weight, weight_elements) ||
        !region_valid(output, output_elements) || !region_valid(staged, staged_elements) ||
        overlap(output, output_elements, input, input_elements) ||
        overlap(output, output_elements, weight, weight_elements) ||
        overlap(output, output_elements, staged, staged_elements) ||
        overlap(staged, staged_elements, input, input_elements) ||
        overlap(staged, staged_elements, weight, weight_elements)) return -1;
    for (size_t i = 0; i < in_count; ++i)
        if (!isfinite(bf16_to_float(input[i]))) return -1;
    for (size_t i = 0; i < w_count; ++i)
        if (!isfinite(bf16_to_float(weight[i]))) return -1;

    for (size_t r = 0; r < rows; ++r) {
        for (size_t o = 0; o < output_width; ++o) {
            const uint16_t *src = input + r * input_width;
            const uint16_t *w = weight + o * input_width;
            double sum = 0.0;
            for (size_t k = 0; k < input_width; ++k) {
                volatile double product = (double)bf16_to_float(src[k]) * bf16_to_float(w[k]);
                sum += product;
            }
            const uint16_t b = float_to_bf16((float)sum);
            if (!isfinite(bf16_to_float(b))) return -1;
            staged[r * output_width + o] = b;
        }
    }
    for (size_t i = 0; i < out_count; ++i) output[i] = staged[i];
    return 0;
}

typedef void (*ck_dgemm_fn)(const char *, const char *, const int *, const int *, const int *,
                           const double *, const double *, const int *, const double *,
                           const int *, const double *, double *, const int *);

static pthread_once_t dgemm_once = PTHREAD_ONCE_INIT;
static ck_dgemm_fn dgemm;

static void bind_dgemm(void)
{
    const char *path = getenv("CK_BLAS_LIBRARY");
    void *library = path && *path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : RTLD_DEFAULT;
    if (path && *path && !library)
        return;
    void *symbol = dlsym(library, "dgemm_");
    _Static_assert(sizeof(symbol) == sizeof(dgemm), "dgemm_ function pointer size");
    memcpy(&dgemm, &symbol, sizeof(dgemm));
}

int ck_linear_rows_bf16_blas_checked(
    const uint16_t *input, size_t input_elements, const uint16_t *weight, size_t weight_elements,
    uint16_t *output, size_t output_elements, uint16_t *staged, size_t staged_elements,
    double *work, size_t work_elements, size_t rows, size_t input_width, size_t output_width)
{
    if (!rows || !input_width || !output_width || rows > INT_MAX ||
        input_width > INT_MAX || output_width > INT_MAX ||
        rows > SIZE_MAX / input_width || output_width > SIZE_MAX / input_width ||
        rows > SIZE_MAX / output_width)
        return -2;
    const size_t a = rows * input_width, b = output_width * input_width, c = rows * output_width;
    if (a > SIZE_MAX - b || a + b > SIZE_MAX - c)
        return -2;
    const size_t work_count = a + b + c;
    if (a > input_elements || b > weight_elements || c > output_elements ||
        c > staged_elements || work_count > work_elements || !work ||
        (uintptr_t)work % _Alignof(double) || work_count > SIZE_MAX / sizeof(double))
        return -2;
    const size_t work_bits = work_count * (sizeof(double) / sizeof(uint16_t));
    const uint16_t *work_region = (const uint16_t*)work;
    if (!region_valid(input, a) || !region_valid(weight, b) ||
        !region_valid(output, c) || !region_valid(staged, c) ||
        !region_valid(work_region, work_bits) ||
        overlap(output, c, input, a) || overlap(output, c, weight, b) ||
        overlap(staged, c, input, a) || overlap(staged, c, weight, b) ||
        overlap(output, c, staged, c) ||
        overlap(work_region, work_bits, input, a) || overlap(work_region, work_bits, weight, b) ||
        overlap(work_region, work_bits, staged, c) || overlap(work_region, work_bits, output, c))
        return -1;
    pthread_once(&dgemm_once, bind_dgemm);
    if (!dgemm)
        return -4;
    double *matrix_a = work, *matrix_b = work + a, *matrix_c = work + a + b;
    for (size_t i = 0; i < a; ++i) {
        matrix_a[i] = bf16_to_float(input[i]);
        if (!isfinite(matrix_a[i]))
            return -3;
    }
    for (size_t i = 0; i < b; ++i) {
        matrix_b[i] = bf16_to_float(weight[i]);
        if (!isfinite(matrix_b[i]))
            return -3;
    }
    const char trans = 'T', no_trans = 'N';
    const int m = (int)rows, n = (int)output_width, k = (int)input_width;
    const double one = 1.0, zero = 0.0;
    dgemm(&trans, &no_trans, &n, &m, &k, &one, matrix_b, &k, matrix_a, &k, &zero, matrix_c, &n);
    for (size_t i = 0; i < c; ++i) {
        staged[i] = float_to_bf16((float)matrix_c[i]);
        if (!isfinite(bf16_to_float(staged[i])))
            return -3;
    }
    memcpy(output, staged, c * sizeof(*output));
    return 0;
}

typedef void (*ck_bf16_gemm_fn)(int, int, int, int, int, int, float,
                               const uint16_t *, int, const uint16_t *, int, float, float *, int);

static pthread_once_t bf16_gemm_once = PTHREAD_ONCE_INIT;
static ck_bf16_gemm_fn bf16_gemm;

static void bind_bf16_gemm(void)
{
    const char *path = getenv("CK_BLAS_LIBRARY");
    void *library = path && *path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : RTLD_DEFAULT;
    if (path && *path && !library)
        return;
    void *symbol = dlsym(library, "cblas_gemm_bf16bf16f32");
    _Static_assert(sizeof(symbol) == sizeof(bf16_gemm), "CBLAS function pointer size");
    memcpy(&bf16_gemm, &symbol, sizeof(bf16_gemm));
}

int ck_linear_rows_bf16_cblas_checked(
    const uint16_t *input, size_t input_elements, const uint16_t *weight, size_t weight_elements,
    uint16_t *output, size_t output_elements, float *work, size_t work_elements,
    size_t rows, size_t input_width, size_t output_width)
{
    if (!rows || !input_width || !output_width || rows > INT_MAX ||
        input_width > INT_MAX || output_width > INT_MAX ||
        rows > SIZE_MAX / input_width || output_width > SIZE_MAX / input_width ||
        rows > SIZE_MAX / output_width)
        return -2;
    const size_t a = rows * input_width, b = output_width * input_width, c = rows * output_width;
    if (a > input_elements || b > weight_elements || c > output_elements || c > work_elements ||
        !work || (uintptr_t)work % _Alignof(float) || c > SIZE_MAX / sizeof(float))
        return -2;
    const uint16_t *work_region = (const uint16_t *)work;
    const size_t work_bits = c * (sizeof(float) / sizeof(uint16_t));
    if (!region_valid(input, a) || !region_valid(weight, b) || !region_valid(output, c) ||
        !region_valid(work_region, work_bits) || overlap(output, c, input, a) ||
        overlap(output, c, weight, b) || overlap(work_region, work_bits, input, a) ||
        overlap(work_region, work_bits, weight, b) || overlap(work_region, work_bits, output, c))
        return -1;
    for (size_t i = 0; i < a; ++i)
        if (!isfinite(bf16_to_float(input[i]))) return -3;
    for (size_t i = 0; i < b; ++i)
        if (!isfinite(bf16_to_float(weight[i]))) return -3;
    pthread_once(&bf16_gemm_once, bind_bf16_gemm);
    if (!bf16_gemm)
        return -4;

    enum { ROW_MAJOR = 101, NO_TRANSPOSE = 111, TRANSPOSE = 112 };
    const int m = (int)rows, n = (int)output_width, k = (int)input_width;
    bf16_gemm(ROW_MAJOR, NO_TRANSPOSE, TRANSPOSE, m, n, k,
              1.0f, input, k, weight, k, 0.0f, work, n);
    for (size_t i = 0; i < c; ++i)
        if (!isfinite(bf16_to_float(float_to_bf16(work[i])))) return -3;
    for (size_t i = 0; i < c; ++i)
        output[i] = float_to_bf16(work[i]);
    return 0;
}
