#ifndef CKE_V8_KOKORO_SHAPE_BOUNDS_H
#define CKE_V8_KOKORO_SHAPE_BOUNDS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* This plans output extents only. Generated graph scratch is a separate,
 * mandatory per-operation declaration; these values must not be used as an
 * implicit allocation policy. All counts are elements/frames, not bytes. */
typedef struct CKKokoroShapeLimitsV8 {
    size_t max_phonemes;
    size_t max_duration_per_phoneme;
    size_t max_expanded_frames;
    size_t max_generator_samples;
    size_t max_output_samples;
    size_t max_alignment_elements;
    size_t alignment_channels;
    size_t decoder_upsample_factor;
    size_t generator_upsample_rate_0;
    size_t generator_upsample_rate_1;
    size_t istft_hop_size;
    size_t istft_fft_size;
} CKKokoroShapeLimitsV8;

typedef struct CKKokoroShapeV8 {
    size_t phonemes;
    size_t expanded_frames;
    size_t alignment_elements;
    size_t decoder_frames;
    size_t generator_samples;
    size_t istft_frames;
    size_t istft_padded_samples;
    size_t output_samples_upper_bound;
} CKKokoroShapeV8;

enum CKAudioExtentStatus {
    CK_AUDIO_EXTENT_OK = 0,
    CK_AUDIO_EXTENT_INVALID = -1,
    CK_AUDIO_EXTENT_LIMIT = -2,
    CK_AUDIO_EXTENT_OVERFLOW = -3
};

/* Called after model duration postprocessing (round, clamp min=1), before
 * alignment expansion or waveform writes. A conservative output bound includes
 * the inverse-STFT FFT halo; actual output length must be checked again. */
int ck_kokoro_plan_shape_v8(
    const int32_t *durations,
    size_t phoneme_count,
    const CKKokoroShapeLimitsV8 *limits,
    CKKokoroShapeV8 *shape);

/* Expand [channels, phonemes] features to [channels, valid_frames]. The
 * physical row strides may exceed valid extents. Caller owns both buffers;
 * padding is never read or written. Input and output must not overlap. */
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
    size_t output_stride);

/* Centered inverse STFT from frame-major [frame, fft/2+1] magnitude/phase.
 * Periodic Hann, inverse DFT scaled by 1/n_fft, overlap-add normalization, then
 * removal of n_fft/2 samples at each end. Supported geometry is frames >= 2,
 * even n_fft >= 2, and 1 <= hop <= n_fft/2. The output is
 * (frames-1)*hop samples. Scratch is two padded sample arrays plus one Hann
 * window, all caller-owned and disjoint from input/output. The planner rejects
 * overflow before any write. */
int audio_istft_mag_phase_plan_f32(
    size_t frames,
    size_t n_fft,
    size_t hop,
    size_t *spectral_elements,
    size_t *output_samples,
    size_t *scratch_elements);

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
    size_t scratch_elements);

#ifdef __cplusplus
}
#endif

#endif
