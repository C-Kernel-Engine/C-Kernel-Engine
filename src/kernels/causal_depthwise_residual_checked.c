#include "checked_float_buffer.h"
#include <float.h>
#include <string.h>

/* Ordered FP64 arithmetic on FP32 input storage. Weight taps and history run
 * oldest first. The history contains pre-convolution rows, never outputs. */
static double evaluate(const float *input, const float *history, const float *weight,
                       size_t token, size_t channel, size_t channels, size_t taps,
                       double divisor)
{
    const float *filter = weight + channel * taps;
    double sum = 0.0;
    for (size_t tap = 0; tap < taps; ++tap) {
        const size_t row = token + tap;
        const float value = row < taps - 1
            ? history[row * channels + channel]
            : input[(row - (taps - 1)) * channels + channel];
        /* Keep the same multiply and add sequence in every build. */
        const volatile double product = (double)value * (double)filter[tap];
        sum += product;
    }
    return (sum + (double)input[token * channels + channel]) / divisor;
}

int ck_causal_depthwise_residual_checked_f32(
    const float *input, size_t input_elements, const float *weight, size_t weight_elements,
    float *history, size_t history_elements, float *output, size_t output_elements,
    size_t tokens, size_t channels, size_t taps, size_t history_rows)
{
    if (taps < 2 || taps > 256 || history_rows != taps - 1 || tokens > SIZE_MAX - taps) return -2;
    size_t rows, weights, state;
    if (!ck_checked_span(tokens, channels, channels, &rows) ||
        !ck_checked_span(channels, taps, taps, &weights) ||
        !ck_checked_span(taps - 1, channels, channels, &state) ||
        rows > input_elements || weights > weight_elements ||
        state > history_elements || rows > output_elements) return -2;

    if (!ck_checked_region(input, rows) || !ck_checked_region(weight, weights) ||
        !ck_checked_region(history, state) || !ck_checked_region(output, rows) ||
        ck_checked_overlap(history, state, input, rows) ||
        ck_checked_overlap(history, state, weight, weights) ||
        ck_checked_overlap(history, state, output, rows) ||
        ck_checked_overlap(output, rows, input, rows) ||
        ck_checked_overlap(output, rows, weight, weights)) return -1;
    if (!ck_checked_finite(input, rows) || !ck_checked_finite(weight, weights) ||
        !ck_checked_finite(history, state)) return -3;

    const double divisor = sqrt((double)taps + 1.0);
    for (size_t t = 0; t < tokens; ++t)
        for (size_t c = 0; c < channels; ++c)
            if (fabs(evaluate(input, history, weight, t, c, channels, taps, divisor)) > FLT_MAX)
                return -3;
    for (size_t t = 0; t < tokens; ++t)
        for (size_t c = 0; c < channels; ++c)
            output[t * channels + c] = (float)evaluate(input, history, weight, t, c, channels, taps, divisor);

    if (tokens >= taps - 1) {
        memcpy(history, input + rows - state, state * sizeof(float));
    } else {
        memmove(history, history + rows, (state - rows) * sizeof(float));
        memcpy(history + state - rows, input, rows * sizeof(float));
    }
    return 0;
}
