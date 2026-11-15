#include "checked_float_buffer.h"
#include "bf16_utils.h"
#include <string.h>

/* Reference inverse real DFT with explicit window and crop. Keeping the window
 * and Cartesian spectrum caller-supplied preserves upstream storage rounding.
 * FP64 recurrence/summation is deliberate: this is a correctness provider,
 * not a claim of PyTorch FFT reduction equivalence or optimized FFT speed. */
int audio_istft_complex_window_plan(
    size_t frames, size_t n_fft, size_t hop, size_t left_crop,
    size_t right_crop, size_t *spectrum, size_t *samples, size_t *scratch)
{
    if (!spectrum || !samples || !scratch || !frames || n_fft < 2 ||
        n_fft > 8192 || (n_fft & 1) || !hop || hop > n_fft ||
        frames - 1 > (SIZE_MAX - n_fft) / hop) return -2;
    size_t padded = (frames - 1) * hop + n_fft;
    size_t bins = n_fft / 2 + 1;
    if (left_crop >= padded || right_crop >= padded - left_crop ||
        frames > SIZE_MAX / bins || padded > SIZE_MAX / 2 ||
        n_fft > SIZE_MAX - 2 * padded) return -2;
    size_t ns = frames * bins, nw = 2 * padded + n_fft;
    if (ns > SIZE_MAX / sizeof(float) || nw > SIZE_MAX / sizeof(double)) return -2;
    *spectrum = ns; *samples = padded - left_crop - right_crop; *scratch = nw;
    return 0;
}

static int inverse_transform(
    const float *real, const float *imag, size_t spectral_elements,
    const float *window, size_t window_elements,
    size_t frames, size_t n_fft, size_t hop, size_t left_crop, size_t right_crop,
    float *output, size_t output_elements, double *scratch, size_t scratch_elements,
    int bf16_envelope)
{
    size_t ns, no, nw;
    if (audio_istft_complex_window_plan(frames, n_fft, hop, left_crop, right_crop,
        &ns, &no, &nw) || spectral_elements < ns || window_elements < n_fft ||
        output_elements < no || scratch_elements < nw) return -2;
    if (!ck_checked_region(real, ns) || !ck_checked_region(imag, ns) ||
        !ck_checked_region(window, n_fft) || !ck_checked_region(output, no) ||
        !scratch || (uintptr_t)scratch % _Alignof(double) ||
        (uintptr_t)scratch > UINTPTR_MAX - nw * sizeof(double)) return -1;
    const float *buffers[] = {real, imag, window, output};
    const size_t lengths[] = {ns, ns, n_fft, no};
    uintptr_t begin = (uintptr_t)scratch, end = begin + nw * sizeof(double);
    for (size_t i = 0; i < 4; ++i) {
        uintptr_t b = (uintptr_t)buffers[i], e = b + lengths[i] * sizeof(float);
        if (begin < e && b < end) return -1;
        if (i < 3 && ck_checked_overlap(buffers[i], lengths[i], output, no)) return -1;
    }
    if (!ck_checked_finite(real, ns) || !ck_checked_finite(imag, ns) ||
        !ck_checked_finite(window, n_fft)) return -3;
    if (bf16_envelope)
        for (size_t n = 0; n < n_fft; ++n)
            if (bf16_to_float(float_to_bf16(window[n])) != window[n]) return -3;
    const size_t padded = no + left_crop + right_crop;
    double *sum = scratch, *envelope = scratch + padded, *frame = envelope + padded;
    memset(scratch, 0, 2 * padded * sizeof(double));
    const double tau = 6.283185307179586476925286766559;
    const size_t bins = n_fft / 2 + 1;
    for (size_t t = 0; t < frames; ++t) {
        size_t base = t * bins;
        for (size_t n = 0; n < n_fft; ++n)
            frame[n] = (double)real[base] + ((n & 1) ? -(double)real[base + bins - 1] : (double)real[base + bins - 1]);
        for (size_t k = 1; k + 1 < bins; ++k) {
            double angle = tau * (double)k / (double)n_fft;
            double step_c = cos(angle), step_s = sin(angle), c = 1.0, s = 0.0;
            for (size_t n = 0; n < n_fft; ++n) {
                frame[n] += 2.0 * ((double)real[base + k] * c - (double)imag[base + k] * s);
                double next_c = c * step_c - s * step_s;
                s = s * step_c + c * step_s; c = next_c;
            }
        }
        for (size_t n = 0; n < n_fft; ++n) {
            double w = window[n]; size_t at = t * hop + n;
            sum[at] += frame[n] / (double)n_fft * w;
            if (!bf16_envelope) envelope[at] += w * w;
        }
    }
    /* PyTorch CPU fold walks window offsets ascending. For one destination
     * sample that is descending contributing frame order; BF16 stores make
     * this distinction observable. Preserve square and each add rounding. */
    if (bf16_envelope) {
        for (size_t t = frames; t-- > 0;) {
            for (size_t n = 0; n < n_fft; ++n) {
                size_t at = t * hop + n;
                float squared = bf16_to_float(float_to_bf16(window[n] * window[n]));
                envelope[at] = bf16_to_float(float_to_bf16((float)envelope[at] + squared));
            }
        }
    }
    for (size_t n = 0; n < no; ++n) {
        size_t at = n + left_crop;
        if (!(envelope[at] > 1e-11) || !isfinite(envelope[at]) ||
            !isfinite((float)(sum[at] / envelope[at]))) return -3;
    }
    for (size_t n = 0; n < no; ++n) {
        size_t at = n + left_crop;
        output[n] = (float)(sum[at] / envelope[at]);
    }
    return 0;
}

int audio_istft_complex_window_f32_checked(
    const float *real, const float *imag, size_t spectral_elements,
    const float *window, size_t window_elements,
    size_t frames, size_t n_fft, size_t hop, size_t left_crop, size_t right_crop,
    float *output, size_t output_elements, double *scratch, size_t scratch_elements)
{
    return inverse_transform(real, imag, spectral_elements, window, window_elements,
        frames, n_fft, hop, left_crop, right_crop, output, output_elements,
        scratch, scratch_elements, 0);
}

int audio_istft_complex_window_bf16_envelope_f32_checked(
    const float *real, const float *imag, size_t spectral_elements,
    const float *window, size_t window_elements,
    size_t frames, size_t n_fft, size_t hop, size_t left_crop, size_t right_crop,
    float *output, size_t output_elements, double *scratch, size_t scratch_elements)
{
    return inverse_transform(real, imag, spectral_elements, window, window_elements,
        frames, n_fft, hop, left_crop, right_crop, output, output_elements,
        scratch, scratch_elements, 1);
}
