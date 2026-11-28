#ifndef CKERNEL_CHECKED_DIFFUSION_SESSION_H
#define CKERNEL_CHECKED_DIFFUSION_SESSION_H

#include "ckernel_checked_diffusion_schedule.h"

typedef int (*ck_diffusion_arena_entry)(uint8_t *arena, size_t bytes);
typedef int (*ck_diffusion_cancel)(void *context);

/* Each BF16 cache plane is [capacity, kv_width]. The two sources are
 * generated temporary rows [prompt_rows or block_rows, kv_width]. */
typedef struct {
    size_t cache, prefill_rows, append_rows;
} ck_diffusion_cache_plane;

typedef struct {
    ck_diffusion_arena_entry entry;
    size_t token_ids, positions, ranges; /* int32 [rows], [rows], [rows,4] */
} ck_diffusion_clean_entry;

typedef struct {
    uint8_t *arena;
    size_t arena_bytes;
    ck_diffusion_clean_entry prefill, append;
    ck_diffusion_arena_entry step;
    ck_diffusion_step_bindings step_bindings;
    size_t positions, conditional_ranges, unconditional_ranges, guidance;

    const ck_diffusion_cache_plane *planes;
    size_t plane_count, kv_width, capacity;
    size_t prompt_rows, block_rows, latent_elements, vocabulary;
    size_t valid_rows; /* initialize to zero. Updated only at commit. */
} ck_diffusion_session;

typedef struct {
    const double *latent, *schedule, *noise;
    size_t latent_elements, schedule_count, noise_elements;
    float guidance;
    int32_t *staged_ids, *output_ids;
    size_t staged_capacity, output_capacity;
    ck_diffusion_cancel should_cancel;
    void *cancel_context;
} ck_diffusion_block_request;

/* Single-owner arena. Bindings must come from the compiled layout.
 * Failed/cancelled evaluations may change scratch. Cache planes, valid_rows,
 * and published block IDs change together only after successful evaluation. */
int ck_prefill_diffusion_session(ck_diffusion_session *session, const int32_t *ids, size_t rows,
                                 ck_diffusion_cancel cancel, void *context);
int ck_append_diffusion_session(ck_diffusion_session *session, const int32_t *ids, size_t rows,
                                ck_diffusion_cancel cancel, void *context);
int ck_generate_diffusion_block(ck_diffusion_session *session, const ck_diffusion_block_request *request);

#endif
