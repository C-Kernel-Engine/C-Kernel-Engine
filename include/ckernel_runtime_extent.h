#ifndef CKERNEL_RUNTIME_EXTENT_H
#define CKERNEL_RUNTIME_EXTENT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CKRuntimeExtentStatus {
    CK_RUNTIME_EXTENT_OK = 0,
    CK_RUNTIME_EXTENT_INVALID = -1,
    CK_RUNTIME_EXTENT_LIMIT = -2,
    CK_RUNTIME_EXTENT_OVERFLOW = -3
};

/* Sum a bounded prefix of int32 extents with an optional initial extent.
 * count may be zero. The output is written only after every input and the
 * accumulated result have been validated. No allocation or input mutation. */
int ck_runtime_sum_i32_checked(
    const int32_t *values,
    size_t value_elements,
    size_t count,
    int32_t min_value,
    int32_t max_value,
    size_t initial_extent,
    size_t capacity,
    int32_t *valid_extent);

/* Copy only the valid columns of channel-major FP32 rows. Physical row
 * strides and buffer capacities are independent of valid_frames. The two
 * buffers must not overlap. Invalid metadata leaves output untouched. */
int ck_runtime_copy_valid_f32(
    const float *input,
    size_t input_elements,
    size_t channels,
    size_t valid_frames,
    size_t input_stride,
    float *output,
    size_t output_elements,
    size_t output_stride);

#ifdef __cplusplus
}
#endif

#endif
