#include "ckernel_tts.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int add_size(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int mul_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

int audio_istft_mag_phase_plan_f32(
    size_t frames,
    size_t n_fft,
    size_t hop,
    size_t *spectral_elements,
    size_t *output_samples,
    size_t *scratch_elements) {
    if (!spectral_elements || !output_samples || !scratch_elements ||
        frames < 2 || n_fft < 2 || (n_fft & 1) || !hop || hop > n_fft / 2) {
        return CK_AUDIO_EXTENT_INVALID;
    }
    size_t bins = n_fft / 2 + 1;
    size_t spectrum, output, padded, scratch;
    if (!mul_size(frames, bins, &spectrum) ||
        !mul_size(frames - 1, hop, &output) ||
        !add_size(output, n_fft, &padded) ||
        !mul_size(padded, 2, &scratch) ||
        !add_size(scratch, n_fft, &scratch)) {
        return CK_AUDIO_EXTENT_OVERFLOW;
    }
    if (spectrum > SIZE_MAX / sizeof(float) ||
        output > SIZE_MAX / sizeof(float) ||
        scratch > SIZE_MAX / sizeof(float)) {
        return CK_AUDIO_EXTENT_OVERFLOW;
    }
    *spectral_elements = spectrum;
    *output_samples = output;
    *scratch_elements = scratch;
    return CK_AUDIO_EXTENT_OK;
}

int audio_istft_mag_phase_f32(
    const float *magnitude,
    const float *phase,
    size_t spectral_elements,
    size_t frames,
    size_t n_fft,
    size_t hop,
    float *output,
    size_t output_capacity,
    float *scratch,
    size_t scratch_elements) {
    if (!magnitude || !phase || !output || !scratch)
        return CK_AUDIO_EXTENT_INVALID;
    size_t required_spectrum, output_samples, required_scratch;
    int status = audio_istft_mag_phase_plan_f32(
        frames, n_fft, hop, &required_spectrum, &output_samples,
        &required_scratch);
    if (status != CK_AUDIO_EXTENT_OK) return status;
    if (spectral_elements < required_spectrum ||
        output_capacity < output_samples ||
        scratch_elements < required_scratch) return CK_AUDIO_EXTENT_LIMIT;
    for (size_t i = 0; i < required_spectrum; ++i) {
        if (!isfinite(magnitude[i]) || !isfinite(phase[i]) || magnitude[i] < 0.0f)
            return CK_AUDIO_EXTENT_INVALID;
    }

    const size_t padded_samples = output_samples + n_fft;
    float *overlap = scratch;
    float *window_sum = overlap + padded_samples;
    float *window = window_sum + padded_samples;
    memset(overlap, 0, padded_samples * sizeof(float));
    memset(window_sum, 0, padded_samples * sizeof(float));
    const float tau = 6.2831853071795864769f;
    for (size_t n = 0; n < n_fft; ++n)
        window[n] = 0.5f - 0.5f * cosf(tau * (float)n / (float)n_fft);

    const size_t bins = n_fft / 2 + 1;
    const float inverse_n = 1.0f / (float)n_fft;
    for (size_t frame = 0; frame < frames; ++frame) {
        const size_t base = frame * bins;
        const size_t offset = frame * hop;
        for (size_t n = 0; n < n_fft; ++n) {
            float acc = magnitude[base] * cosf(phase[base]);
            const size_t nyquist = bins - 1;
            float end = magnitude[base + nyquist] * cosf(phase[base + nyquist]);
            acc += (n & 1) ? -end : end;
            for (size_t k = 1; k < nyquist; ++k) {
                const float real = magnitude[base + k] * cosf(phase[base + k]);
                const float imag = magnitude[base + k] * sinf(phase[base + k]);
                const float angle = tau * (float)k * (float)n / (float)n_fft;
                acc += 2.0f * (real * cosf(angle) - imag * sinf(angle));
            }
            const float w = window[n];
            overlap[offset + n] += acc * inverse_n * w;
            window_sum[offset + n] += w * w;
        }
    }

    const size_t left_trim = n_fft / 2;
    for (size_t i = 0; i < output_samples; ++i) {
        const size_t pos = left_trim + i;
        const float norm = window_sum[pos];
        if (!(norm > 1e-11f)) return CK_AUDIO_EXTENT_INVALID;
        if (!isfinite(overlap[pos] / norm)) return CK_AUDIO_EXTENT_INVALID;
    }
    for (size_t i = 0; i < output_samples; ++i) {
        const size_t pos = left_trim + i;
        output[i] = overlap[pos] / window_sum[pos];
    }
    return CK_AUDIO_EXTENT_OK;
}
