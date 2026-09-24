#include "ckernel_runtime_extent.h"

#include <stdint.h>

int ck_runtime_sum_i32_checked(
    const int32_t *values,
    size_t value_elements,
    size_t count,
    int32_t min_value,
    int32_t max_value,
    size_t initial_extent,
    size_t capacity,
    int32_t *valid_extent) {
    if (!valid_extent || (count && !values) || min_value < 0 ||
        max_value < min_value) return CK_RUNTIME_EXTENT_INVALID;
    if (count > value_elements || initial_extent > capacity)
        return CK_RUNTIME_EXTENT_LIMIT;
    if (initial_extent > INT32_MAX || capacity > INT32_MAX)
        return CK_RUNTIME_EXTENT_OVERFLOW;
    if (count > SIZE_MAX / sizeof(int32_t))
        return CK_RUNTIME_EXTENT_OVERFLOW;

    int32_t total = (int32_t)initial_extent;
    for (size_t i = 0; i < count; ++i) {
        const int32_t item = values[i];
        if (item < min_value || item > max_value)
            return CK_RUNTIME_EXTENT_INVALID;
        if (item > INT32_MAX - total)
            return CK_RUNTIME_EXTENT_OVERFLOW;
        total += item;
        if ((size_t)total > capacity)
            return CK_RUNTIME_EXTENT_LIMIT;
    }
    *valid_extent = total;
    return CK_RUNTIME_EXTENT_OK;
}

int ck_runtime_copy_valid_f32(
    const float *input,
    size_t input_elements,
    size_t channels,
    size_t valid_frames,
    size_t input_stride,
    float *output,
    size_t output_elements,
    size_t output_stride) {
    if (!channels || (valid_frames && (!input || !output)) ||
        input_stride < valid_frames || output_stride < valid_frames)
        return CK_RUNTIME_EXTENT_INVALID;
    if (!valid_frames) return CK_RUNTIME_EXTENT_OK;
    if (channels - 1 > (SIZE_MAX - valid_frames) / input_stride ||
        channels - 1 > (SIZE_MAX - valid_frames) / output_stride)
        return CK_RUNTIME_EXTENT_OVERFLOW;
    const size_t input_required = (channels - 1) * input_stride + valid_frames;
    const size_t output_required = (channels - 1) * output_stride + valid_frames;
    if (input_required > SIZE_MAX / sizeof(float) ||
        output_required > SIZE_MAX / sizeof(float))
        return CK_RUNTIME_EXTENT_OVERFLOW;
    if (input_elements < input_required || output_elements < output_required)
        return CK_RUNTIME_EXTENT_LIMIT;
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t frame = 0; frame < valid_frames; ++frame) {
            output[channel * output_stride + frame] =
                input[channel * input_stride + frame];
        }
    }
    return CK_RUNTIME_EXTENT_OK;
}
