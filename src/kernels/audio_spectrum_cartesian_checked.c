#include "checked_float_buffer.h"

/* Frame-major [log magnitude | phase] -> Cartesian bins. No BF16 stores.
 * Clamp before exp to avoid overflow for a finite log magnitude. */
int audio_spectrum_cartesian_f32_checked(
    const float *input, size_t input_elements,
    float *real, size_t real_elements, float *imag, size_t imag_elements,
    size_t frames, size_t bins, float magnitude_limit)
{
    if (!frames || !bins || frames > SIZE_MAX / bins ||
        frames * bins > SIZE_MAX / 2) return -2;
    size_t n = frames * bins;
    if (input_elements < 2*n || real_elements < n || imag_elements < n) return -2;
    if (!(magnitude_limit > 0) || !isfinite(magnitude_limit) ||
        !ck_checked_region(input, 2*n) || !ck_checked_region(real, n) ||
        !ck_checked_region(imag, n) || ck_checked_overlap(input,2*n,real,n) ||
        ck_checked_overlap(input,2*n,imag,n) || ck_checked_overlap(real,n,imag,n)) return -1;
    if (!ck_checked_finite(input, 2*n)) return -3;
    const float threshold = logf(magnitude_limit);
    for (size_t row = 0; row < frames; ++row) {
        for (size_t bin = 0; bin < bins; ++bin) {
            float log_magnitude = input[row*2*bins+bin];
            float phase = input[row*2*bins+bins+bin];
            float magnitude = log_magnitude >= threshold ? magnitude_limit :
                fminf(expf(log_magnitude), magnitude_limit);
            real[row*bins+bin] = magnitude*cosf(phase);
            imag[row*bins+bin] = magnitude*sinf(phase);
        }
    }
    return 0;
}
