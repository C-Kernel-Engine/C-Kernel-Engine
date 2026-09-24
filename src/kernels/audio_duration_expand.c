#include "ckernel_tts.h"

#include <stdint.h>

int audio_duration_expand_channel_major_f32(
    const float *features,
    size_t input_elements,
    size_t channels,
    size_t phoneme_count,
    size_t input_stride,
    const int32_t *durations,
    size_t expanded_frames,
    float *output,
    size_t output_elements,
    size_t output_stride) {
    if (!features || !durations || !output || !channels || !phoneme_count ||
        !expanded_frames || input_stride < phoneme_count ||
        output_stride < expanded_frames) return CK_AUDIO_EXTENT_INVALID;
    if (channels - 1 > (SIZE_MAX - phoneme_count) / input_stride ||
        channels - 1 > (SIZE_MAX - expanded_frames) / output_stride)
        return CK_AUDIO_EXTENT_OVERFLOW;
    const size_t required_input = (channels - 1) * input_stride + phoneme_count;
    const size_t required_output = (channels - 1) * output_stride + expanded_frames;
    if (required_input > SIZE_MAX / sizeof(float) ||
        required_output > SIZE_MAX / sizeof(float)) return CK_AUDIO_EXTENT_OVERFLOW;
    if (input_elements < required_input || output_elements < required_output)
        return CK_AUDIO_EXTENT_LIMIT;

    size_t sum = 0;
    for (size_t token = 0; token < phoneme_count; ++token) {
        if (durations[token] <= 0) return CK_AUDIO_EXTENT_INVALID;
        if ((size_t)durations[token] > expanded_frames - sum)
            return CK_AUDIO_EXTENT_LIMIT;
        sum += (size_t)durations[token];
    }
    if (sum != expanded_frames) return CK_AUDIO_EXTENT_INVALID;

    for (size_t channel = 0; channel < channels; ++channel) {
        size_t frame = 0;
        for (size_t token = 0; token < phoneme_count; ++token) {
            const float value = features[channel * input_stride + token];
            for (int32_t j = 0; j < durations[token]; ++j) {
                output[channel * output_stride + frame++] = value;
            }
        }
    }
    return CK_AUDIO_EXTENT_OK;
}
