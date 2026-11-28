#ifndef CKERNEL_CHECKED_DIFFUSION_SCHEDULE_H
#define CKERNEL_CHECKED_DIFFUSION_SCHEDULE_H

#include <stddef.h>
#include <stdint.h>

/* Offsets refer to verified typed bindings in one generated step arena. */
typedef struct {
    size_t latent, next_latent;             /* FP64 [latent_elements] */
    size_t noise, gamma;                    /* FP64 [latent_elements] or SIZE_MAX, FP64 [2] */
    size_t self_condition, conditional_mean; /* FP32 [latent_elements] */
    size_t token_ids;                       /* int32 [token_count] */
} ck_diffusion_step_bindings;

typedef struct {
    int (*step)(uint8_t *arena, size_t bytes);
    uint8_t *arena;
    size_t arena_bytes;
    ck_diffusion_step_bindings bindings;
    size_t latent_elements, token_count, vocabulary;
    const double *initial_latent;
    const double *schedule;
    size_t schedule_count;
    const double *noise;
    size_t noise_elements;
    int32_t *output_ids;
    size_t output_capacity;
    int (*should_cancel)(void *context);
    void *cancel_context;
} ck_diffusion_schedule_request;

/* One denoising block. The caller prepares model weights, positions, visibility,
 * read-only clean prefix and guidance in the arena. Noise is explicit, with
 * one row per transition. Deterministic transitions set the noise offset to
 * SIZE_MAX and supply no noise span. A final evaluation commits IDs.
 * Outputs remain untouched on rejection, step failure or cancellation. */
int ck_run_checked_diffusion_schedule(const ck_diffusion_schedule_request *request);

#endif
