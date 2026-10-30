#include "ckernel_checked_frame_loop.h"

#include <limits.h>
#include <string.h>

static int span(const void *pointer, size_t bytes) {
    return pointer && (uintptr_t)pointer <= UINTPTR_MAX - bytes;
}

static int disjoint(const void *left, size_t left_bytes,
                    const void *right, size_t right_bytes) {
    const uintptr_t a = (uintptr_t)left;
    const uintptr_t b = (uintptr_t)right;
    return a + left_bytes <= b || b + right_bytes <= a;
}

static int scalar_offset(size_t offset, size_t arena_bytes) {
    return offset % _Alignof(int32_t) == 0 &&
           offset <= arena_bytes &&
           sizeof(int32_t) <= arena_bytes - offset;
}

static int32_t read_scalar(const uint8_t *arena, size_t offset) {
    int32_t result;
    memcpy(&result, arena + offset, sizeof(result));
    return result;
}

static void write_scalar(uint8_t *arena, size_t offset, int32_t value) {
    memcpy(arena + offset, &value, sizeof(value));
}

static int run_checked_frame_loop(const ck_checked_frame_loop *request,
                                  ck_checked_frame_cancel should_cancel,
                                  void *cancel_context,
                                  int control_required) {
    if (!request || !request->valid_frames ||
        (uintptr_t)request->valid_frames % _Alignof(size_t) != 0 ||
        !span(request->valid_frames, sizeof(size_t)))
        return CK_FRAME_LOOP_INVALID;
    /* Do not clear validity through an alias of the caller's output. */
    if (request->output_codes && request->output_elements <=
            SIZE_MAX / sizeof(int32_t) &&
        span(request->output_codes,
             request->output_elements * sizeof(int32_t)) &&
        !disjoint(request->valid_frames, sizeof(size_t),
                  request->output_codes,
                  request->output_elements * sizeof(int32_t)))
        return CK_FRAME_LOOP_INVALID;
    *request->valid_frames = 0;
    if ((control_required && !should_cancel) ||
        !request->prefill || !request->frame || !request->arena ||
        !request->prefill_code_offsets || !request->frame_code_offsets ||
        !request->staging_codes || !request->output_codes ||
        (uintptr_t)request->prefill_code_offsets % _Alignof(size_t) != 0 ||
        (uintptr_t)request->frame_code_offsets % _Alignof(size_t) != 0 ||
        (uintptr_t)request->staging_codes % _Alignof(int32_t) != 0 ||
        (uintptr_t)request->output_codes % _Alignof(int32_t) != 0 ||
        !request->codebooks || request->codebooks > INT32_MAX ||
        request->max_frames < 2 || request->max_frames - 1 > INT32_MAX ||
        request->max_frames > SIZE_MAX / request->codebooks)
        return CK_FRAME_LOOP_INVALID;
    const size_t required = request->max_frames * request->codebooks;
    if (required > SIZE_MAX / sizeof(int32_t) ||
        request->staging_elements < required ||
        request->output_elements < required ||
        !span(request->arena, request->arena_bytes) ||
        !span(request->staging_codes, required * sizeof(int32_t)) ||
        !span(request->output_codes, required * sizeof(int32_t)) ||
        !span(request->valid_frames, sizeof(size_t)) ||
        !disjoint(request->arena, request->arena_bytes,
                  request->staging_codes, required * sizeof(int32_t)) ||
        !disjoint(request->arena, request->arena_bytes,
                  request->output_codes, required * sizeof(int32_t)) ||
        !disjoint(request->staging_codes, required * sizeof(int32_t),
                  request->output_codes, required * sizeof(int32_t)) ||
        !disjoint(request->valid_frames, sizeof(size_t),
                  request->arena, request->arena_bytes) ||
        !disjoint(request->valid_frames, sizeof(size_t),
                  request->staging_codes, required * sizeof(int32_t)) ||
        !disjoint(request->valid_frames, sizeof(size_t),
                  request->output_codes, required * sizeof(int32_t)) ||
        !scalar_offset(request->step_index_offset, request->arena_bytes) ||
        !scalar_offset(request->prefill_valid_code_count_offset,
                       request->arena_bytes) ||
        !scalar_offset(request->frame_valid_code_count_offset,
                       request->arena_bytes))
        return CK_FRAME_LOOP_INVALID;
    for (size_t codebook = 0; codebook < request->codebooks; ++codebook) {
        if (!scalar_offset(request->prefill_code_offsets[codebook],
                           request->arena_bytes) ||
            !scalar_offset(request->frame_code_offsets[codebook],
                           request->arena_bytes))
            return CK_FRAME_LOOP_INVALID;
    }
    if (should_cancel && should_cancel(cancel_context))
        return CK_FRAME_LOOP_CANCELLED;
    if (request->prefill(request->arena, request->arena_bytes) != 0)
        return CK_FRAME_LOOP_ENTRY_FAILED;
    if (should_cancel && should_cancel(cancel_context))
        return CK_FRAME_LOOP_CANCELLED;
    if (read_scalar(request->arena,
                    request->prefill_valid_code_count_offset) !=
        (int32_t)(request->codebooks - 1))
        return CK_FRAME_LOOP_INVALID_OUTPUT;
    for (size_t codebook = 0; codebook < request->codebooks; ++codebook)
        request->staging_codes[codebook] = read_scalar(
            request->arena, request->prefill_code_offsets[codebook]);

    for (size_t step = 0; step < request->max_frames - 1; ++step) {
        if (should_cancel && should_cancel(cancel_context))
            return CK_FRAME_LOOP_CANCELLED;
        write_scalar(request->arena, request->step_index_offset,
                     (int32_t)step);
        const int status = request->frame(request->arena,
                                          request->arena_bytes);
        if (should_cancel && should_cancel(cancel_context))
            return CK_FRAME_LOOP_CANCELLED;
        if (status == 1) {
            const size_t frames = step + 1;
            memcpy(request->output_codes, request->staging_codes,
                   frames * request->codebooks * sizeof(int32_t));
            *request->valid_frames = frames;
            return CK_FRAME_LOOP_EOS;
        }
        if (status == 2) return CK_FRAME_LOOP_BUDGET_EXHAUSTED;
        if (status != 0) return CK_FRAME_LOOP_ENTRY_FAILED;
        if (read_scalar(request->arena,
                        request->frame_valid_code_count_offset) !=
            (int32_t)(request->codebooks - 1))
            return CK_FRAME_LOOP_INVALID_OUTPUT;
        for (size_t codebook = 0; codebook < request->codebooks; ++codebook)
            request->staging_codes[(step + 1) * request->codebooks + codebook] =
                read_scalar(request->arena,
                            request->frame_code_offsets[codebook]);
    }
    return CK_FRAME_LOOP_BUDGET_EXHAUSTED;
}

int ck_run_checked_frame_loop(const ck_checked_frame_loop *request) {
    return run_checked_frame_loop(request, NULL, NULL, 0);
}

int ck_run_checked_frame_loop_control(
    const ck_checked_frame_loop_control *request) {
    if (!request)
        return CK_FRAME_LOOP_INVALID;
    return run_checked_frame_loop(&request->base, request->should_cancel,
                                  request->cancel_context, 1);
}
