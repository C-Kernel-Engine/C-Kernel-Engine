#include "ckernel_linear_checked.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

static int span_fits(size_t rows, size_t stride, size_t width, size_t elements)
{
    return rows && width && stride >= width &&
           rows - 1 <= (SIZE_MAX - width) / stride &&
           (rows - 1) * stride + width <= elements;
}

static double dot_bias(const float *input, const float *weight,
                       size_t width, float bias)
{
    double sum = (double)bias;
    for (size_t c = 0; c < width; ++c) {
        volatile double product = (double)input[c] * (double)weight[c];
        sum += product;
    }
    return sum;
}

int linear_rows_checked_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements, size_t weight_stride,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t input_channels, size_t output_channels)
{
    if (!input || !weight || !output ||
        !span_fits(rows, input_stride, input_channels, input_elements) ||
        !span_fits(output_channels, weight_stride, input_channels, weight_elements) ||
        !span_fits(rows, output_stride, output_channels, output_elements) ||
        (bias ? bias_elements < output_channels : bias_elements != 0))
        return -1;

    /* Validate every accessed value before the first write. A conservative
     * absolute bound permits one compute pass for ordinary model values.
     * Values near the FP32 limit retain the exact two-pass staging path. */
    double max_input = 0.0;
    double max_weight_l1 = 0.0;
    double max_bias = 0.0;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < input_channels; ++c)
        {
            const float value = input[r * input_stride + c];
            if (!isfinite(value)) return -1;
            const double absolute = fabs((double)value);
            if (absolute > max_input) max_input = absolute;
        }
    for (size_t o = 0; o < output_channels; ++o) {
        if (bias && !isfinite(bias[o])) return -1;
        if (bias && fabs((double)bias[o]) > max_bias)
            max_bias = fabs((double)bias[o]);
        double weight_l1 = 0.0;
        for (size_t c = 0; c < input_channels; ++c)
        {
            const float value = weight[o * weight_stride + c];
            if (!isfinite(value)) return -1;
            weight_l1 += fabs((double)value);
        }
        if (weight_l1 > max_weight_l1) max_weight_l1 = weight_l1;
    }
    /* At most UINT32_MAX positive terms keep FP64 summation error far below
     * the 4x guard, including error in the L1 bound. Every result therefore
     * fits FP32 and publishing cannot fail after the first output write. */
    if (input_channels <= (SIZE_MAX / 4) && input_channels <= UINT32_MAX &&
        max_input * max_weight_l1 + max_bias <= (double)FLT_MAX / 4.0) {
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o < output_channels; ++o)
                output[r * output_stride + o] = (float)dot_bias(
                    input + r * input_stride, weight + o * weight_stride,
                    input_channels, bias ? bias[o] : 0.0f);
        return 0;
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < output_channels; ++o) {
            const double result = dot_bias(input + r * input_stride,
                                           weight + o * weight_stride,
                                           input_channels, bias ? bias[o] : 0.0f);
            if (!isfinite(result) || fabs(result) > FLT_MAX) return -1;
        }
    for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < output_channels; ++o)
            output[r * output_stride + o] = (float)dot_bias(
                input + r * input_stride, weight + o * weight_stride,
                input_channels, bias ? bias[o] : 0.0f);
    return 0;
}
