#include "ckernel_audio_snake_checked.h"
#include "checked_float_buffer.h"

#include <math.h>
#include <float.h>

/* For finite arguments, |sinf| <= 1. Leave ample rounding headroom so
 * the separately rounded square, scale and addition cannot overflow. Extreme
 * valid inputs still use the original exact preflight below. No values are
 * written until every element has passed one of these checks. */
static int bounded_periodic_output(float x, float inverse) {
    return fabs((double)x) + fabs((double)inverse) <= (double)FLT_MAX / 4.0;
}

static int snake_value(float x, float alpha, float *value) {
    const float argument = alpha * x;
    if (!isfinite(argument)) return 0;
    const float sine = sinf(argument);
    const float inverse = 1.0f / alpha;
    const float square = sine * sine;
    /* Preserve the FP32 multiplication before addition even when the model
     * library is built with FMA enabled. */
    volatile float scaled = inverse * square;
    const float result = x + scaled;
    if (!isfinite(sine) || !isfinite(inverse) || !isfinite(result)) return 0;
    *value = result;
    return 1;
}

int audio_snake_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *alpha, size_t alpha_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t frames) {
    size_t input_span, output_span;
    if (!channels || !frames || input_stride < frames || output_stride < frames)
        return -1;
    if (!ck_checked_span(channels, frames, input_stride, &input_span) ||
        !ck_checked_span(channels, frames, output_stride, &output_span))
        return -2;
    if (input_elements < input_span || output_elements < output_span ||
        alpha_elements < channels) return -2;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(alpha, channels) ||
        !ck_checked_region(output, output_span) ||
        ck_checked_overlap(input, input_span, output, output_span) ||
        ck_checked_overlap(alpha, channels, input, input_span) ||
        ck_checked_overlap(alpha, channels, output, output_span))
        return -1;

    /* A preflight pass preserves all output, including valid cells, when a
     * late element overflows. No heap or variable-length stack storage. */
    for (size_t channel = 0; channel < channels; ++channel) {
        const float a = alpha[channel];
        if (!isfinite(a) || a == 0.0f) return -3;
        const float inverse = 1.0f / a;
        if (!isfinite(inverse)) return -3;
        for (size_t frame = 0; frame < frames; ++frame) {
            const float x = input[channel * input_stride + frame];
            float result;
            if (!isfinite(x) || !isfinite(a * x)) return -3;
            if (!bounded_periodic_output(x, inverse) &&
                !snake_value(x, a, &result)) return -3;
        }
    }
    for (size_t channel = 0; channel < channels; ++channel)
        for (size_t frame = 0; frame < frames; ++frame) {
            float result = 0.0f;
            /* Preflight established that this evaluation is finite. */
            (void)snake_value(input[channel * input_stride + frame],
                              alpha[channel], &result);
            output[channel * output_stride + frame] = result;
        }
    return 0;
}

static int snake_beta_value(float x, float frequency, float inverse_magnitude,
                            float *value) {
    const float argument = x * frequency;
    if (!isfinite(argument)) return 0;
    const float sine = sinf(argument);
    const float square = sine * sine;
    /* Match the reference's separately rounded multiplication and addition. */
    volatile float periodic = inverse_magnitude * square;
    const float result = x + periodic;
    if (!isfinite(sine) || !isfinite(result)) return 0;
    *value = result;
    return 1;
}

int audio_snake_beta_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *log_alpha, size_t alpha_elements,
    const float *log_beta, size_t beta_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t frames) {
    size_t input_span, output_span;
    if (!channels || !frames || input_stride < frames || output_stride < frames)
        return -1;
    if (!ck_checked_span(channels, frames, input_stride, &input_span) ||
        !ck_checked_span(channels, frames, output_stride, &output_span))
        return -2;
    if (input_elements < input_span || output_elements < output_span ||
        alpha_elements < channels || beta_elements < channels) return -2;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(log_alpha, channels) ||
        !ck_checked_region(log_beta, channels) ||
        !ck_checked_region(output, output_span) ||
        ck_checked_overlap(input, input_span, output, output_span) ||
        ck_checked_overlap(log_alpha, channels, input, input_span) ||
        ck_checked_overlap(log_beta, channels, input, input_span) ||
        ck_checked_overlap(log_alpha, channels, log_beta, channels) ||
        ck_checked_overlap(log_alpha, channels, output, output_span) ||
        ck_checked_overlap(log_beta, channels, output, output_span))
        return -1;

    for (size_t channel = 0; channel < channels; ++channel) {
        if (!isfinite(log_alpha[channel]) || !isfinite(log_beta[channel]))
            return -3;
        const float frequency = expf(log_alpha[channel]);
        const float magnitude = expf(log_beta[channel]);
        const float inverse_magnitude = 1.0f / (magnitude + 1e-9f);
        if (!isfinite(frequency) || !isfinite(magnitude) ||
            !isfinite(inverse_magnitude)) return -3;
        for (size_t frame = 0; frame < frames; ++frame) {
            const float x = input[channel * input_stride + frame];
            float result;
            if (!isfinite(x) || !isfinite(x * frequency)) return -3;
            if (!bounded_periodic_output(x, inverse_magnitude) &&
                !snake_beta_value(x, frequency, inverse_magnitude, &result))
                return -3;
        }
    }
    for (size_t channel = 0; channel < channels; ++channel) {
        const float frequency = expf(log_alpha[channel]);
        const float inverse_magnitude = 1.0f / (expf(log_beta[channel]) + 1e-9f);
        for (size_t frame = 0; frame < frames; ++frame) {
            float result = 0.0f;
            (void)snake_beta_value(input[channel * input_stride + frame],
                                   frequency, inverse_magnitude, &result);
            output[channel * output_stride + frame] = result;
        }
    }
    return 0;
}
