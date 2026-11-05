#ifndef CKERNEL_TOKEN_SPAN_MEAN_H
#define CKERNEL_TOKEN_SPAN_MEAN_H

#include <stddef.h>
#include <stdint.h>

/* Mean each half-open [start, end) token span into one output row. */
int ck_token_span_mean_checked_f32(
    const float *input, size_t input_elements, size_t input_stride,
    size_t token_capacity, size_t valid_tokens, size_t channels,
    const int32_t *spans, size_t span_elements, size_t span_count,
    float *output, size_t output_elements, size_t output_stride);

#endif
