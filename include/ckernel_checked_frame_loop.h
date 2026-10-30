#ifndef CKERNEL_CHECKED_FRAME_LOOP_H
#define CKERNEL_CHECKED_FRAME_LOOP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ck_checked_frame_entry)(uint8_t *arena, size_t arena_bytes);
typedef int (*ck_checked_frame_cancel)(void *context);

enum {
    CK_FRAME_LOOP_EOS = 0,
    CK_FRAME_LOOP_BUDGET_EXHAUSTED = 2,
    CK_FRAME_LOOP_INVALID = -1,
    CK_FRAME_LOOP_ENTRY_FAILED = -2,
    CK_FRAME_LOOP_INVALID_OUTPUT = -3,
    CK_FRAME_LOOP_CANCELLED = -7
};

/* The offsets are byte offsets into one compiler-planned arena. Each entry
 * executes the neural graph declared by its circuit; this shell only repeats
 * the frame entry and collects its checked integer outputs. The caller owns
 * arena, staging and published output. A failed or budget-limited run leaves
 * output_codes unchanged and publishes zero valid frames. */
typedef struct {
    ck_checked_frame_entry prefill;
    ck_checked_frame_entry frame;
    uint8_t *arena;
    size_t arena_bytes;
    size_t step_index_offset;
    size_t prefill_valid_code_count_offset;
    size_t frame_valid_code_count_offset;
    const size_t *prefill_code_offsets;
    const size_t *frame_code_offsets;
    size_t codebooks;
    size_t max_frames;
    int32_t *staging_codes;
    size_t staging_elements;
    int32_t *output_codes;
    size_t output_elements;
    size_t *valid_frames;
} ck_checked_frame_loop;

int ck_run_checked_frame_loop(const ck_checked_frame_loop *request);

/* The callback is sampled before prefill and between complete frame entries.
 * It must be safe to call from the execution thread and return nonzero when
 * cancellation is requested. Cancellation publishes no codes or valid frames.
 * An entry already in progress runs to its next checked boundary. */
typedef struct {
    ck_checked_frame_loop base;
    ck_checked_frame_cancel should_cancel;
    void *cancel_context;
} ck_checked_frame_loop_control;

int ck_run_checked_frame_loop_control(
    const ck_checked_frame_loop_control *request);

#ifdef __cplusplus
}
#endif

#endif
