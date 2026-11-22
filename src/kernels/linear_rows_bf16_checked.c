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
