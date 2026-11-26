#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "bf16_utils.h"

#include "ckernel_engine.h"

static int valid_span(const void *base, size_t length, size_t element_size)
{
    return base && (uintptr_t)base % element_size == 0 &&
           length <= (UINTPTR_MAX - (uintptr_t)base) / element_size;
}

static int overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    return (uintptr_t)a < (uintptr_t)b + b_bytes &&
           (uintptr_t)b < (uintptr_t)a + a_bytes;
}

typedef void (*rmsnorm_forward_fn)(const float *, const float *, float *, float *,
                                  int, int, int, float);

static int rmsnorm_rows_bf16_checked(
    const uint16_t *input, size_t input_elements, const uint16_t *gamma,
    size_t gamma_elements, uint16_t *output, size_t output_elements,
    float *work, size_t work_elements, size_t rows, size_t width, float epsilon,
    rmsnorm_forward_fn normalize)
{
    if (!rows || !width || rows > INT_MAX || width > INT_MAX ||
        rows > (SIZE_MAX - width) / width || !isfinite(epsilon) || epsilon <= 0.0f) {
        return -2;
    }
    const size_t count = rows * width;
    const size_t required_work = count + width;
    if (input_elements < count || gamma_elements < width ||
        output_elements < count || work_elements < required_work) {
        return -2;
    }
    if (!valid_span(input, count, sizeof(*input)) ||
        !valid_span(gamma, width, sizeof(*gamma)) ||
        !valid_span(output, count, sizeof(*output)) ||
        !valid_span(work, required_work, sizeof(*work))) {
        return -1;
    }
    const size_t work_bytes = required_work * sizeof(*work);
    const size_t row_bytes = count * sizeof(*output);
    const size_t gamma_bytes = width * sizeof(*gamma);
    if (overlap(work, work_bytes, input, row_bytes) ||
        overlap(work, work_bytes, gamma, gamma_bytes) ||
        overlap(work, work_bytes, output, row_bytes) ||
        overlap(output, row_bytes, input, row_bytes) ||
        overlap(output, row_bytes, gamma, gamma_bytes)) {
        return -1;
    }
    float *expanded_gamma = work + count;
    for (size_t i = 0; i < width; ++i) {
        expanded_gamma[i] = bf16_to_float(gamma[i]);
        if (!isfinite(expanded_gamma[i])) {
            return -3;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        work[i] = bf16_to_float(input[i]);
        if (!isfinite(work[i])) {
            return -3;
        }
    }
    normalize(
        work, expanded_gamma, work, NULL, (int)rows, (int)width, (int)width, epsilon);
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(work[i])) {
            return -3;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        output[i] = float_to_bf16(work[i]);
    }
    return 0;
}

int ck_rmsnorm_rows_bf16_checked(
    const uint16_t *input, size_t input_elements, const uint16_t *gamma,
    size_t gamma_elements, uint16_t *output, size_t output_elements,
    float *work, size_t work_elements, size_t rows, size_t width, float epsilon)
{
    return rmsnorm_rows_bf16_checked(input, input_elements, gamma, gamma_elements,
        output, output_elements, work, work_elements, rows, width, epsilon,
        rmsnorm_forward_pytorch_bf16_storage);
}

int ck_rmsnorm_rows_weighted_single_round_bf16_checked(
    const uint16_t *input, size_t input_elements, const uint16_t *gamma,
    size_t gamma_elements, uint16_t *output, size_t output_elements,
    float *work, size_t work_elements, size_t rows, size_t width, float epsilon)
{
    return rmsnorm_rows_bf16_checked(input, input_elements, gamma, gamma_elements,
        output, output_elements, work, work_elements, rows, width, epsilon,
        rmsnorm_forward_muse_weighted_pytorch_bf16_storage);
}

int ck_rmsnorm_rows_centered_single_round_bf16_checked(
    const uint16_t *input, size_t input_elements, const uint16_t *gamma,
    size_t gamma_elements, uint16_t *output, size_t output_elements,
    float *work, size_t work_elements, size_t rows, size_t width, float epsilon)
{
    return rmsnorm_rows_bf16_checked(input, input_elements, gamma, gamma_elements,
        output, output_elements, work, work_elements, rows, width, epsilon,
        rmsnorm_forward_muse_centered_pytorch_bf16_storage);
}
