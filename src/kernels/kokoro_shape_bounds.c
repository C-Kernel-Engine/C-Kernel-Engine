#include "ckernel_tts.h"

#include <stdint.h>
#include <string.h>

static int checked_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int checked_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

int ck_kokoro_plan_shape_v8(
    const int32_t *durations,
    size_t phoneme_count,
    const CKKokoroShapeLimitsV8 *limits,
    CKKokoroShapeV8 *shape) {
    if (!durations || !limits || !shape || !phoneme_count ||
        !limits->max_phonemes || !limits->max_duration_per_phoneme ||
        !limits->max_expanded_frames || !limits->max_generator_samples ||
        !limits->max_output_samples || !limits->max_alignment_elements ||
        !limits->alignment_channels || !limits->decoder_upsample_factor ||
        !limits->generator_upsample_rate_0 || !limits->generator_upsample_rate_1 ||
        !limits->istft_hop_size || !limits->istft_fft_size) {
        return CK_AUDIO_EXTENT_INVALID;
    }
    memset(shape, 0, sizeof(*shape));
    if (phoneme_count > limits->max_phonemes) return CK_AUDIO_EXTENT_LIMIT;

    size_t frames = 0;
    for (size_t i = 0; i < phoneme_count; ++i) {
        if (durations[i] <= 0) return CK_AUDIO_EXTENT_INVALID;
        if ((size_t)durations[i] > limits->max_duration_per_phoneme)
            return CK_AUDIO_EXTENT_LIMIT;
        if (!checked_add(frames, (size_t)durations[i], &frames))
            return CK_AUDIO_EXTENT_OVERFLOW;
        if (frames > limits->max_expanded_frames)
            return CK_AUDIO_EXTENT_LIMIT;
    }

    size_t align, decoder, generator, istft_frames, padded, output;
    if (!checked_mul(frames, limits->alignment_channels, &align) ||
        !checked_mul(frames, limits->decoder_upsample_factor, &decoder) ||
        !checked_mul(decoder, limits->generator_upsample_rate_0, &generator) ||
        !checked_mul(generator, limits->generator_upsample_rate_1, &generator) ||
        !checked_add(generator / limits->istft_hop_size, 1, &istft_frames) ||
        !checked_add(generator, limits->istft_fft_size, &padded) ||
        !checked_add(generator, limits->istft_fft_size, &output)) {
        return CK_AUDIO_EXTENT_OVERFLOW;
    }
    if (align > SIZE_MAX / sizeof(float) ||
        decoder > SIZE_MAX / sizeof(float) ||
        generator > SIZE_MAX / sizeof(float) ||
        padded > SIZE_MAX / sizeof(float) ||
        output > SIZE_MAX / sizeof(float)) {
        return CK_AUDIO_EXTENT_OVERFLOW;
    }
    if (align > limits->max_alignment_elements ||
        generator > limits->max_generator_samples ||
        output > limits->max_output_samples) {
        return CK_AUDIO_EXTENT_LIMIT;
    }

    shape->phonemes = phoneme_count;
    shape->expanded_frames = frames;
    shape->alignment_elements = align;
    shape->decoder_frames = decoder;
    shape->generator_samples = generator;
    shape->istft_frames = istft_frames;
    shape->istft_padded_samples = padded;
    shape->output_samples_upper_bound = output;
    return CK_AUDIO_EXTENT_OK;
}
