#include "ckernel_checked_frame_codec.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int in_arena(size_t offset, size_t bytes, size_t arena_bytes) {
    return offset <= arena_bytes && bytes <= arena_bytes - offset;
}

static int disjoint(uintptr_t a, size_t a_bytes, uintptr_t b, size_t b_bytes) {
    return a + a_bytes <= b || b + b_bytes <= a;
}

static int run_checked_frame_codec(const ck_checked_frame_codec *request,
                                   size_t samples_per_code_frame,
                                   int prefix_mode,
                                   ck_checked_frame_cancel should_cancel,
                                   void *cancel_context,
                                   int control_required) {
    if (!request || !request->generation || !request->valid_samples ||
        (uintptr_t)request->valid_samples % _Alignof(size_t) != 0 ||
        (uintptr_t)request->valid_samples > UINTPTR_MAX - sizeof(size_t))
        return CK_FRAME_LOOP_INVALID;
    const ck_checked_frame_loop *loop = request->generation;
    const uintptr_t validity = (uintptr_t)request->valid_samples;
    if (!loop->valid_frames ||
        (uintptr_t)loop->valid_frames > UINTPTR_MAX - sizeof(size_t) ||
        !disjoint(validity, sizeof(size_t),
                  (uintptr_t)loop->valid_frames, sizeof(size_t)))
        return CK_FRAME_LOOP_INVALID;
    /* The generation loop writes its validity before codec success. */
    if (request->output_samples && request->output_capacity <= SIZE_MAX / sizeof(float) &&
        (uintptr_t)request->output_samples <= UINTPTR_MAX -
            request->output_capacity * sizeof(float) &&
        !disjoint((uintptr_t)loop->valid_frames, sizeof(size_t),
                  (uintptr_t)request->output_samples,
                  request->output_capacity * sizeof(float)))
        return CK_FRAME_LOOP_INVALID;
    if (request->output_samples && request->output_capacity <= SIZE_MAX / sizeof(float) &&
        (uintptr_t)request->output_samples <= UINTPTR_MAX -
            request->output_capacity * sizeof(float) &&
        !disjoint(validity, sizeof(size_t), (uintptr_t)request->output_samples,
                  request->output_capacity * sizeof(float)))
        return CK_FRAME_LOOP_INVALID;
    if (loop->arena && (uintptr_t)loop->arena <= UINTPTR_MAX - loop->arena_bytes &&
        !disjoint(validity, sizeof(size_t), (uintptr_t)loop->arena,
                  loop->arena_bytes))
        return CK_FRAME_LOOP_INVALID;
    if (loop->output_codes && loop->output_elements <= SIZE_MAX / sizeof(int32_t) &&
        (uintptr_t)loop->output_codes <= UINTPTR_MAX -
            loop->output_elements * sizeof(int32_t) &&
        !disjoint(validity, sizeof(size_t), (uintptr_t)loop->output_codes,
                  loop->output_elements * sizeof(int32_t)))
        return CK_FRAME_LOOP_INVALID;
    if (loop->staging_codes && loop->staging_elements <= SIZE_MAX / sizeof(int32_t) &&
        (uintptr_t)loop->staging_codes <= UINTPTR_MAX -
            loop->staging_elements * sizeof(int32_t) &&
        !disjoint(validity, sizeof(size_t), (uintptr_t)loop->staging_codes,
                  loop->staging_elements * sizeof(int32_t)))
        return CK_FRAME_LOOP_INVALID;
    *request->valid_samples = 0;
    if ((control_required && !should_cancel) ||
        (prefix_mode && !samples_per_code_frame) ||
        !loop->arena || !loop->output_codes || !request->codec ||
        !request->output_samples || !request->expected_code_frames ||
        !request->expected_codebooks ||
        !request->expected_samples || !loop->codebooks ||
        request->expected_codebooks != loop->codebooks ||
        request->expected_code_frames > loop->max_frames ||
        loop->max_frames > SIZE_MAX / loop->codebooks ||
        request->output_capacity > SIZE_MAX / sizeof(float) ||
        request->expected_code_frames > SIZE_MAX / loop->codebooks ||
        request->expected_samples > SIZE_MAX / sizeof(float) ||
        request->output_capacity < request->expected_samples ||
        (samples_per_code_frame &&
         (request->expected_code_frames > SIZE_MAX / samples_per_code_frame ||
          request->expected_samples !=
              request->expected_code_frames * samples_per_code_frame)) ||
        (uintptr_t)request->output_samples % _Alignof(float) != 0)
        return CK_FRAME_LOOP_INVALID;
    const size_t code_elements = request->expected_code_frames * loop->codebooks;
    if (code_elements > SIZE_MAX / sizeof(int32_t) ||
        !in_arena(request->codec_frame_major_offset,
                  code_elements * sizeof(int32_t), loop->arena_bytes) ||
        !in_arena(request->waveform_offset,
                  request->expected_samples * sizeof(float), loop->arena_bytes) ||
        !in_arena(request->waveform_valid_samples_offset,
                  sizeof(int32_t), loop->arena_bytes) ||
        request->codec_frame_major_offset % _Alignof(int32_t) != 0 ||
        request->waveform_offset % _Alignof(float) != 0 ||
        request->waveform_valid_samples_offset % _Alignof(int32_t) != 0 ||
        (uintptr_t)request->output_samples > UINTPTR_MAX -
            request->output_capacity * sizeof(float) ||
        (uintptr_t)loop->arena > UINTPTR_MAX - loop->arena_bytes)
        return CK_FRAME_LOOP_INVALID;
    const uintptr_t arena_begin = (uintptr_t)loop->arena;
    const uintptr_t output_begin = (uintptr_t)request->output_samples;
    const size_t output_bytes = request->output_capacity * sizeof(float);
    if (loop->max_frames * loop->codebooks > SIZE_MAX / sizeof(int32_t) ||
        !loop->staging_codes)
        return CK_FRAME_LOOP_INVALID;
    const size_t code_bytes = loop->max_frames * loop->codebooks * sizeof(int32_t);
    if ((uintptr_t)loop->output_codes > UINTPTR_MAX - code_bytes ||
        (uintptr_t)loop->staging_codes > UINTPTR_MAX - code_bytes)
        return CK_FRAME_LOOP_INVALID;
    if (!disjoint(output_begin, output_bytes, arena_begin, loop->arena_bytes) ||
        !disjoint(output_begin, output_bytes,
                  (uintptr_t)loop->output_codes, code_bytes) ||
        !disjoint(output_begin, output_bytes,
                  (uintptr_t)loop->staging_codes, code_bytes) ||
        !disjoint((uintptr_t)request->valid_samples, sizeof(size_t),
                  output_begin, output_bytes) ||
        !disjoint((uintptr_t)request->valid_samples, sizeof(size_t),
                  arena_begin, loop->arena_bytes) ||
        !disjoint((uintptr_t)request->valid_samples, sizeof(size_t),
                  (uintptr_t)loop->output_codes, code_bytes) ||
        !disjoint((uintptr_t)request->valid_samples, sizeof(size_t),
                  (uintptr_t)loop->staging_codes, code_bytes))
        return CK_FRAME_LOOP_INVALID;
    int generation_status;
    if (should_cancel) {
        ck_checked_frame_loop_control controlled = {
            .base = *loop,
            .should_cancel = should_cancel,
            .cancel_context = cancel_context,
        };
        generation_status = ck_run_checked_frame_loop_control(&controlled);
    } else {
        generation_status = ck_run_checked_frame_loop(loop);
    }
    if (generation_status != CK_FRAME_LOOP_EOS)
        return generation_status;
    if (should_cancel && should_cancel(cancel_context)) {
        *loop->valid_frames = 0;
        return CK_FRAME_LOOP_CANCELLED;
    }
    if (!loop->valid_frames || !*loop->valid_frames ||
        *loop->valid_frames > request->expected_code_frames ||
        (!samples_per_code_frame &&
         *loop->valid_frames != request->expected_code_frames))
        return CK_FRAME_CODEC_LENGTH_MISMATCH;
    if (samples_per_code_frame)
        memset(loop->arena + request->codec_frame_major_offset, 0,
               code_elements * sizeof(int32_t));
    memcpy(loop->arena + request->codec_frame_major_offset,
           loop->output_codes,
           *loop->valid_frames * loop->codebooks * sizeof(int32_t));
    if (request->codec(loop->arena, loop->arena_bytes) != 0)
        return CK_FRAME_CODEC_ENTRY_FAILED;
    if (should_cancel && should_cancel(cancel_context)) {
        *loop->valid_frames = 0;
        return CK_FRAME_LOOP_CANCELLED;
    }
    int32_t produced = 0;
    memcpy(&produced, loop->arena + request->waveform_valid_samples_offset,
           sizeof(produced));
    if (produced < 0 || (size_t)produced != request->expected_samples)
        return CK_FRAME_CODEC_INVALID_OUTPUT;
    const float *samples = (const float *)(const void *)(
        loop->arena + request->waveform_offset);
    for (size_t index = 0; index < request->expected_samples; ++index)
        if (!isfinite(samples[index])) return CK_FRAME_CODEC_INVALID_OUTPUT;
    if (should_cancel && should_cancel(cancel_context)) {
        *loop->valid_frames = 0;
        return CK_FRAME_LOOP_CANCELLED;
    }
    const size_t valid_samples = samples_per_code_frame
        ? *loop->valid_frames * samples_per_code_frame
        : request->expected_samples;
    memcpy(request->output_samples, samples, valid_samples * sizeof(float));
    *request->valid_samples = valid_samples;
    return 0;
}

int ck_run_checked_frame_codec(const ck_checked_frame_codec *request) {
    return run_checked_frame_codec(request, 0, 0, NULL, NULL, 0);
}

int ck_run_checked_frame_codec_prefix(
    const ck_checked_frame_codec_prefix *request) {
    if (!request)
        return CK_FRAME_LOOP_INVALID;
    return run_checked_frame_codec(&request->base,
                                   request->samples_per_code_frame, 1,
                                   NULL, NULL, 0);
}

int ck_run_checked_frame_codec_prefix_control(
    const ck_checked_frame_codec_prefix_control *request) {
    if (!request)
        return CK_FRAME_LOOP_INVALID;
    return run_checked_frame_codec(&request->base.base,
                                   request->base.samples_per_code_frame, 1,
                                   request->should_cancel,
                                   request->cancel_context, 1);
}
