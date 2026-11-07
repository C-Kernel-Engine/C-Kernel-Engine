#include "ckernel_grouped_weighted_rows.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

static int span_fits(size_t rows, size_t stride, size_t width, size_t elements) {
    return rows && width && stride >= width &&
           rows - 1 <= (SIZE_MAX - width) / stride &&
           (rows - 1) * stride + width <= elements;
}

static int valid_region(const void *ptr, size_t elements, size_t bytes,
                        size_t alignment) {
    return ptr && (uintptr_t)ptr % alignment == 0 &&
           elements <= SIZE_MAX / bytes &&
           (uintptr_t)ptr <= UINTPTR_MAX - elements * bytes;
}

static int overlaps(const void *left, size_t left_bytes,
                    const void *right, size_t right_bytes) {
    uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
    return a < b + right_bytes && b < a + left_bytes;
}

int ck_grouped_softmax_weighted_rows_f32(
    const float *options, size_t option_elements, size_t option_stride,
    const float *fields, size_t field_elements, size_t field_stride,
    const int32_t *group_counts, size_t count_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t groups, size_t option_rows, size_t channels) {
    if (!groups || !option_rows || !channels ||
        groups > SIZE_MAX / channels ||
        option_rows > SIZE_MAX - groups * channels ||
        !span_fits(option_rows, option_stride, channels, option_elements) ||
        !span_fits(groups, field_stride, channels, field_elements) ||
        !span_fits(groups, output_stride, channels, output_elements) ||
        count_elements < groups ||
        scratch_elements < option_rows + groups * channels)
        return -2;
    if (!valid_region(options, option_elements, sizeof(float), _Alignof(float)) ||
        !valid_region(fields, field_elements, sizeof(float), _Alignof(float)) ||
        !valid_region(group_counts, count_elements, sizeof(int32_t), _Alignof(int32_t)) ||
        !valid_region(output, output_elements, sizeof(float), _Alignof(float)) ||
        !valid_region(scratch, scratch_elements, sizeof(float), _Alignof(float)))
        return -1;
    const size_t option_bytes = option_elements * sizeof(float);
    const size_t field_bytes = field_elements * sizeof(float);
    const size_t count_bytes = count_elements * sizeof(int32_t);
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t scratch_bytes = scratch_elements * sizeof(float);
    if (overlaps(output, output_bytes, options, option_bytes) ||
        overlaps(output, output_bytes, fields, field_bytes) ||
        overlaps(output, output_bytes, group_counts, count_bytes) ||
        overlaps(output, output_bytes, scratch, scratch_bytes) ||
        overlaps(scratch, scratch_bytes, options, option_bytes) ||
        overlaps(scratch, scratch_bytes, fields, field_bytes) ||
        overlaps(scratch, scratch_bytes, group_counts, count_bytes))
        return -1;

    size_t counted = 0;
    for (size_t q = 0; q < groups; ++q) {
        int32_t count = group_counts[q];
        if (count <= 0 || (size_t)count > option_rows - counted)
            return -2;
        counted += (size_t)count;
        for (size_t c = 0; c < channels; ++c)
            if (!isfinite(fields[q * field_stride + c])) return -1;
    }
    if (counted != option_rows) return -2;
    for (size_t r = 0; r < option_rows; ++r)
        for (size_t c = 0; c < channels; ++c)
            if (!isfinite(options[r * option_stride + c])) return -1;

    /* Scratch holds all probabilities and staged results. No output is
     * published until every group and output channel has passed validation. */
    size_t first = 0;
    const double scale = 1.0 / sqrt((double)channels);
    for (size_t q = 0; q < groups; ++q) {
        size_t count = (size_t)group_counts[q];
        double maximum = -INFINITY;
        for (size_t i = 0; i < count; ++i) {
            double dot = 0.0;
            for (size_t c = 0; c < channels; ++c) {
                volatile double product =
                    (double)options[(first + i) * option_stride + c] *
                    (double)fields[q * field_stride + c];
                dot += product;
            }
            double logit = dot * scale;
            if (!isfinite(logit) || fabs(logit) > FLT_MAX) return -1;
            scratch[first + i] = (float)logit;
            if ((double)scratch[first + i] > maximum)
                maximum = (double)scratch[first + i];
        }
        double denominator = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double value = exp((double)scratch[first + i] - maximum);
            if (!isfinite(value)) return -1;
            scratch[first + i] = (float)value;
            denominator += (double)scratch[first + i];
        }
        if (!isfinite(denominator) || denominator <= 0.0) return -1;
        for (size_t i = 0; i < count; ++i)
            scratch[first + i] = (float)((double)scratch[first + i] /
                                          denominator);
        for (size_t c = 0; c < channels; ++c) {
            double sum = 0.0;
            for (size_t i = 0; i < count; ++i) {
                volatile double product = (double)scratch[first + i] *
                    (double)options[(first + i) * option_stride + c];
                sum += product;
            }
            if (!isfinite(sum) || fabs(sum) > FLT_MAX) return -1;
            scratch[option_rows + q * channels + c] = (float)sum;
        }
        first += count;
    }
    for (size_t q = 0; q < groups; ++q)
        for (size_t c = 0; c < channels; ++c)
            output[q * output_stride + c] =
                scratch[option_rows + q * channels + c];
    return 0;
}
