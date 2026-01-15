#include <stdint.h>
#include <string.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

// Suppress false positive warnings about uninitialized variables
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

/*
 * BF16 GELU with caller-provided scratch buffer.
 * scratch: [n] floats - caller allocates and reuses
 */
void gelu_fast_inplace_bf16(uint16_t *data, size_t n, float *scratch)
{
    if (!scratch) return;

    bf16_tensor_to_float(data, scratch, n);
    // Use exact version to avoid fast tanh approximation error accumulating
    // with BF16 precision loss. Conversion overhead dominates anyway.
    gelu_exact_inplace(scratch, n);
    float_tensor_to_bf16(scratch, data, n);
}

/*
 * BF16 GELU backward with caller-provided scratch buffers.
 * scratch_input, scratch_d_output, scratch_d_input: each [n] floats
 */
void gelu_backward_exact_bf16(const uint16_t *input,
                              const uint16_t *d_output,
                              uint16_t *d_input,
                              size_t n,
                              float *scratch_input,
                              float *scratch_d_output,
                              float *scratch_d_input)
{
    if (!scratch_input || !scratch_d_output || !scratch_d_input) return;

    bf16_tensor_to_float(input, scratch_input, n);
    bf16_tensor_to_float(d_output, scratch_d_output, n);

    // Use scalar exact version to avoid fast tanh approximation error
    // accumulating with BF16 precision loss.
    gelu_backward_scalar(scratch_input, scratch_d_output, scratch_d_input, n);

    float_tensor_to_bf16(scratch_d_input, d_input, n);
}

/*
 * BF16 GELU backward (fast) with caller-provided scratch buffers.
 */
void gelu_backward_fast_bf16(const uint16_t *input,
                             const uint16_t *d_output,
                             uint16_t *d_input,
                             size_t n,
                             float *scratch_input,
                             float *scratch_d_output,
                             float *scratch_d_input)
{
    if (!scratch_input || !scratch_d_output || !scratch_d_input) return;

    bf16_tensor_to_float(input, scratch_input, n);
    bf16_tensor_to_float(d_output, scratch_d_output, n);

    gelu_backward_fast(scratch_input, scratch_d_output, scratch_d_input, n);

    float_tensor_to_bf16(scratch_d_input, d_input, n);
}
