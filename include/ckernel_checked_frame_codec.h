#ifndef CKERNEL_CHECKED_FRAME_CODEC_H
#define CKERNEL_CHECKED_FRAME_CODEC_H

#include "ckernel_checked_frame_loop.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CK_FRAME_CODEC_LENGTH_MISMATCH = -4,
    CK_FRAME_CODEC_ENTRY_FAILED = -5,
    CK_FRAME_CODEC_INVALID_OUTPUT = -6
};

/* Execute a declared codec segment after a checked generation loop reaches
 * EOS. Code frames are copied into a frame-major circuit input; the circuit
 * owns every codebook layout edge and all codec arithmetic. Both generated
 * entries use the same compiler-planned arena. Published PCM is unchanged on
 * rejection and valid_samples remains zero. */
typedef struct {
    ck_checked_frame_loop *generation;
    ck_checked_frame_entry codec;
    size_t codec_frame_major_offset;
    size_t expected_codebooks;
    size_t expected_code_frames;
    size_t waveform_offset;
    size_t waveform_valid_samples_offset;
    size_t expected_samples;
    float *output_samples;
    size_t output_capacity;
    size_t *valid_samples;
} ck_checked_frame_codec;

int ck_run_checked_frame_codec(const ck_checked_frame_codec *request);

/* Run a codec compiled for expected_code_frames of storage while publishing
 * only the prefix produced before EOS. The codec must be prefix causal for
 * the selected provider and geometry; callers must certify that property.
 * Unused frame-major code slots are filled with code zero. expected_samples
 * remains the full compiled waveform capacity. */
typedef struct {
    ck_checked_frame_codec base;
    size_t samples_per_code_frame;
} ck_checked_frame_codec_prefix;

int ck_run_checked_frame_codec_prefix(
    const ck_checked_frame_codec_prefix *request);

#ifdef __cplusplus
}
#endif

#endif
