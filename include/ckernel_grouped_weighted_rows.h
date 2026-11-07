#ifndef CKE_GROUPED_WEIGHTED_ROWS_H
#define CKE_GROUPED_WEIGHTED_ROWS_H

#include <stddef.h>
#include <stdint.h>

/* For each group, softmax(dot(options[row], field[group]) / sqrt(channels))
 * across its consecutive option rows, then sum the weighted option rows.
 * All buffers are caller-owned. Read buffers may alias; output and scratch
 * must not overlap each other or any read buffer. Scratch needs at least
 * option_rows + groups * channels floats. Rejection leaves output unchanged. */
int ck_grouped_softmax_weighted_rows_f32(
    const float *options, size_t option_elements, size_t option_stride,
    const float *fields, size_t field_elements, size_t field_stride,
    const int32_t *group_counts, size_t count_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t groups, size_t option_rows, size_t channels);

#endif
