/* Bounded bidirectional FP32 LSTM scan over token-major rows.
 * Reuses the PyTorch-IFGO single-step kernel; all state and scratch are caller owned.
 */
#include "ckernel_audio.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

static int checked_product(size_t left, size_t right, size_t *result)
{
    if (right != 0u && left > SIZE_MAX / right) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int checked_span(size_t rows, size_t stride, size_t width,
                        size_t *result)
{
    size_t prefix;
    if (rows == 0u || !checked_product(rows - 1u, stride, &prefix) ||
        prefix > SIZE_MAX - width) {
        return 0;
    }
    *result = prefix + width;
    return 1;
}

int audio_lstm_bidirectional_scan_f32(
    const float *input, size_t input_elements,
    const float *weight_ih, size_t weight_ih_elements,
    const float *weight_hh, size_t weight_hh_elements,
    const float *bias_ih, size_t bias_ih_elements,
    const float *bias_hh, size_t bias_hh_elements,
    float *output, size_t output_elements,
    float *hidden_state, size_t hidden_state_elements,
    float *cell_state, size_t cell_state_elements,
    float *gates_scratch, size_t gates_scratch_bytes,
    int tokens, int input_size, int hidden_size,
    size_t input_stride, size_t output_stride)
{
    if (input == NULL || weight_ih == NULL || weight_hh == NULL ||
        bias_ih == NULL || bias_hh == NULL || output == NULL ||
        hidden_state == NULL || cell_state == NULL || gates_scratch == NULL) {
        return -1;
    }
    if (tokens <= 0 || input_size <= 0 || hidden_size <= 0 ||
        hidden_size > INT_MAX / 2) {
        return -2;
    }

    const size_t token_count = (size_t)tokens;
    const size_t input_width = (size_t)input_size;
    const size_t hidden_width = (size_t)hidden_size;
    const size_t output_width = hidden_width * 2u;
    size_t input_required, output_required, gates_count;
    size_t ih_per_direction, hh_per_direction, bias_required;
    size_t ih_required, hh_required, gates_bytes, state_bytes;
    if (input_stride < input_width || output_stride < output_width ||
        !checked_span(token_count, input_stride, input_width, &input_required) ||
        !checked_span(token_count, output_stride, output_width, &output_required) ||
        !checked_product(hidden_width, 4u, &gates_count) ||
        !checked_product(gates_count, input_width, &ih_per_direction) ||
        !checked_product(gates_count, hidden_width, &hh_per_direction) ||
        !checked_product(ih_per_direction, 2u, &ih_required) ||
        !checked_product(hh_per_direction, 2u, &hh_required) ||
        !checked_product(gates_count, 2u, &bias_required) ||
        !checked_product(gates_count, sizeof(float), &gates_bytes) ||
        !checked_product(output_width, sizeof(float), &state_bytes)) {
        return -2;
    }
    if (input_elements < input_required || output_elements < output_required ||
        weight_ih_elements < ih_required || weight_hh_elements < hh_required ||
        bias_ih_elements < bias_required || bias_hh_elements < bias_required ||
        hidden_state_elements < output_width ||
        cell_state_elements < output_width ||
        gates_scratch_bytes < gates_bytes) {
        return -3;
    }

    /* Every invocation starts from zero state, as a batch-one PyTorch LSTM
     * called without initial (h, c) does. Padding outside valid rows is not touched.
     */
    memset(hidden_state, 0, state_bytes);
    memset(cell_state, 0, state_bytes);
    for (size_t index = 0u; index < token_count; ++index) {
        const int status = audio_lstm_step_f32(
            input + index * input_stride,
            weight_ih, weight_hh, bias_ih, bias_hh,
            hidden_state, cell_state,
            output + index * output_stride,
            gates_scratch, gates_scratch_bytes, input_size, hidden_size);
        if (status != 0) {
            return status;
        }
    }
    for (size_t reverse_index = 0u; reverse_index < token_count;
         ++reverse_index) {
        const size_t index = token_count - 1u - reverse_index;
        const int status = audio_lstm_step_f32(
            input + index * input_stride,
            weight_ih + ih_per_direction,
            weight_hh + hh_per_direction,
            bias_ih + gates_count,
            bias_hh + gates_count,
            hidden_state + hidden_width,
            cell_state + hidden_width,
            output + index * output_stride + hidden_width,
            gates_scratch, gates_scratch_bytes, input_size, hidden_size);
        if (status != 0) {
            return status;
        }
    }
    return 0;
}
