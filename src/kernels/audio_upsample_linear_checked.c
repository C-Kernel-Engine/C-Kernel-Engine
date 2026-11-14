#include "checked_float_buffer.h"

/* Integer-factor, align_corners=false interpolation, frame-major layout.
 * Explicit two-term FP32 reduction; no hidden allocation or persistent state.
 * This is an offline primitive, not a streaming lookahead state machine. */
int audio_upsample_linear_frames_f32_checked(
    const float *input, size_t input_elements, float *output,
    size_t output_elements, size_t frames, size_t channels, size_t factor)
{
    size_t ni, no;
    if (!factor || factor > 65536 || frames > 16777216 ||
        !ck_checked_span(frames, channels, channels, &ni) ||
        ni > SIZE_MAX / factor || input_elements < ni)
        return -2;
    no = ni * factor;
    if (output_elements < no) return -2;
    if (!ck_checked_region(input, ni) || !ck_checked_region(output, no) ||
        ck_checked_overlap(input, ni, output, no)) return -1;
    for (size_t i = 0; i < ni; ++i)
        if (!isfinite(input[i])) return -3;
    /* Preflight arithmetic too: a late invalid output cannot partially publish. */
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t row = 0; row < frames * factor; ++row) {
            /* Double coordinate avoids integer overflow and loss of half-pixels.
             * Cast only the interpolation weight to the declared FP32 domain. */
            double position = ((double)row + 0.5) / (double)factor - 0.5;
            if (position < 0) position = 0;
            size_t left = (size_t)position;
            size_t right = left + 1 < frames ? left + 1 : left;
            float weight = (float)(position - (double)left);
            for (size_t c = 0; c < channels; ++c) {
                float value = (1.0f - weight) * input[left * channels + c]
                    + weight * input[right * channels + c];
                if (!isfinite(value)) return -3;
                if (pass) output[row * channels + c] = value;
            }
        }
    }
    return 0;
}
