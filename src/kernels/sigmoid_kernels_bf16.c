#include <stddef.h>
#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

/*
 * BF16 sigmoid forward with caller-provided scratch buffers.
 * scratch_input, scratch_output: each [n] floats
 */
void sigmoid_forward_bf16(const uint16_t *input,
                          uint16_t *output,
                          size_t n,
                          float *scratch_input,
                          float *scratch_output)
{
    if (!input || !output || n == 0) return;
    if (!scratch_input || !scratch_output) return;

    bf16_tensor_to_float(input, scratch_input, n);
    sigmoid_forward(scratch_input, scratch_output, n);
    float_tensor_to_bf16(scratch_output, output, n);
}

/*
 * BF16 sigmoid backward with caller-provided scratch buffers.
 * scratch_input, scratch_d_output, scratch_d_input: each [n] floats
 */
void sigmoid_backward_bf16(const uint16_t *input,
                           const uint16_t *d_output,
                           uint16_t *d_input,
                           size_t n,
                           float *scratch_input,
                           float *scratch_d_output,
                           float *scratch_d_input)
{
    if (!input || !d_output || !d_input || n == 0) return;
    if (!scratch_input || !scratch_d_output || !scratch_d_input) return;

    bf16_tensor_to_float(input, scratch_input, n);
    bf16_tensor_to_float(d_output, scratch_d_output, n);
    sigmoid_backward(scratch_input, scratch_d_output, scratch_d_input, n);
    float_tensor_to_bf16(scratch_d_input, d_input, n);
}
