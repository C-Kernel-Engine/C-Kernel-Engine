#include "ckernel_tts.h"

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

static CKKokoroShapeLimitsV8 fixture_limits(void) {
    CKKokoroShapeLimitsV8 x = {
        .max_phonemes = 512,
        .max_duration_per_phoneme = 50,
        .max_expanded_frames = 2000,
        .max_generator_samples = 240000,
        .max_output_samples = 240020,
        .max_alignment_elements = 640 * 2000,
        .alignment_channels = 640,
        .decoder_upsample_factor = 2,
        .generator_upsample_rate_0 = 10,
        .generator_upsample_rate_1 = 6,
        .istft_hop_size = 5,
        .istft_fft_size = 20,
    };
    return x;
}

int main(void) {
    const int32_t durations[] = {2, 3};
    CKKokoroShapeLimitsV8 lim = fixture_limits();
    CKKokoroShapeV8 out;
    assert(ck_kokoro_plan_shape_v8(durations, 2, &lim, &out) == CK_AUDIO_EXTENT_OK);
    assert(out.phonemes == 2);
    assert(out.expanded_frames == 5);
    assert(out.alignment_elements == 3200);
    assert(out.decoder_frames == 10);
    assert(out.generator_samples == 600);
    assert(out.istft_frames == 121);
    assert(out.istft_padded_samples == 620);
    assert(out.output_samples_upper_bound == 620);

    lim.max_output_samples = 619;
    assert(ck_kokoro_plan_shape_v8(durations, 2, &lim, &out) == CK_AUDIO_EXTENT_LIMIT);
    lim = fixture_limits();
    lim.max_alignment_elements = 3199;
    assert(ck_kokoro_plan_shape_v8(durations, 2, &lim, &out) == CK_AUDIO_EXTENT_LIMIT);
    lim = fixture_limits();
    lim.decoder_upsample_factor = SIZE_MAX;
    assert(ck_kokoro_plan_shape_v8(durations, 2, &lim, &out) == CK_AUDIO_EXTENT_OVERFLOW);
    lim = fixture_limits();
    lim.istft_fft_size = SIZE_MAX;
    assert(ck_kokoro_plan_shape_v8(durations, 2, &lim, &out) == CK_AUDIO_EXTENT_OVERFLOW);

    const int32_t too_long[] = {51};
    assert(ck_kokoro_plan_shape_v8(too_long, 1, &lim, &out) == CK_AUDIO_EXTENT_LIMIT);
    const int32_t zero[] = {0};
    assert(ck_kokoro_plan_shape_v8(zero, 1, &lim, &out) == CK_AUDIO_EXTENT_INVALID);
    assert(ck_kokoro_plan_shape_v8(durations, 0, &lim, &out) == CK_AUDIO_EXTENT_INVALID);

    const float features[] = {1.0f, 2.0f, 99.0f, -3.0f, 4.0f, 99.0f};
    float expanded[16];
    for (size_t i = 0; i < 16; ++i) expanded[i] = 99.0f;
    const float expected[] = {1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 99.0f, 99.0f, 99.0f,
                              -3.0f, -3.0f, 4.0f, 4.0f, 4.0f, 99.0f, 99.0f, 99.0f};
    assert(audio_duration_expand_channel_major_f32(
        features, 6, 2, 2, 3, durations, 5, expanded, 16, 8) == CK_AUDIO_EXTENT_OK);
    for (size_t i = 0; i < 16; ++i) assert(expanded[i] == expected[i]);
    assert(audio_duration_expand_channel_major_f32(
        features, 6, 2, 2, 3, durations, 5, expanded, 12, 8) == CK_AUDIO_EXTENT_LIMIT);
    assert(audio_duration_expand_channel_major_f32(
        features, 6, 2, 2, 3, durations, 4, expanded, 16, 8) == CK_AUDIO_EXTENT_LIMIT);
    const int32_t short_durations[] = {1, 1};
    for (size_t i = 0; i < 16; ++i) expanded[i] = 99.0f;
    assert(audio_duration_expand_channel_major_f32(
        features, 6, 2, 2, 3, short_durations, 2, expanded, 16, 8) == CK_AUDIO_EXTENT_OK);
    assert(expanded[0] == 1.0f && expanded[1] == 2.0f);
    assert(expanded[8] == -3.0f && expanded[9] == 4.0f);
    for (size_t i = 2; i < 8; ++i) assert(expanded[i] == 99.0f);
    for (size_t i = 10; i < 16; ++i) assert(expanded[i] == 99.0f);
    const int32_t negative[] = {-1, 1};
    assert(audio_duration_expand_channel_major_f32(
        features, 6, 2, 2, 3, negative, 2, expanded, 16, 8) == CK_AUDIO_EXTENT_INVALID);
    return 0;
}
