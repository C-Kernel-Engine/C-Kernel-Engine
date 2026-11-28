#include "ckernel_checked_diffusion_schedule.h"

#include <math.h>
#include <string.h>

typedef struct {
    const void *pointer;
    size_t bytes;
    size_t alignment;
} schedule_region;

static int disjoint(schedule_region first, schedule_region second)
{
    const uintptr_t a = (uintptr_t)first.pointer, b = (uintptr_t)second.pointer;
    return a + first.bytes <= b || b + second.bytes <= a;
}

static int region_valid(schedule_region region)
{
    const uintptr_t address = (uintptr_t)region.pointer;
    return region.pointer && region.bytes && address % region.alignment == 0 &&
           region.bytes <= UINTPTR_MAX - address;
}

static int valid_regions(const schedule_region *regions, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (!region_valid(regions[i])) return 0;
        for (size_t j = 0; j < i; ++j)
            if (!disjoint(regions[i], regions[j])) return 0;
    }
    return 1;
}

static int validate_request(const ck_diffusion_schedule_request *r)
{
    if (!r || (uintptr_t)r % _Alignof(ck_diffusion_schedule_request) || !r->step || !r->latent_elements || !r->token_count ||
        !r->vocabulary || r->vocabulary > INT32_MAX || r->schedule_count < 2 ||
        r->latent_elements > SIZE_MAX / sizeof(double) / (r->schedule_count - 1) ||
        r->schedule_count > SIZE_MAX / sizeof(double) ||
        r->noise_elements != (r->schedule_count - 1) * r->latent_elements ||
        r->output_capacity < r->token_count || r->output_capacity > SIZE_MAX / sizeof(int32_t))
        return 0;
    const size_t n = r->latent_elements;
    const schedule_region external[] = {
        {r, sizeof(*r), _Alignof(ck_diffusion_schedule_request)},
        {r->arena, r->arena_bytes, _Alignof(double)},
        {r->initial_latent, n * sizeof(double), _Alignof(double)},
        {r->schedule, r->schedule_count * sizeof(double), _Alignof(double)},
        {r->noise, r->noise_elements * sizeof(double), _Alignof(double)},
        {r->output_ids, r->output_capacity * sizeof(int32_t), _Alignof(int32_t)},
    };
    if (!valid_regions(external, sizeof external / sizeof external[0])) return 0;

    const ck_diffusion_step_bindings *b = &r->bindings;
    const size_t offsets[] = {
        b->latent, b->next_latent, b->noise, b->gamma, b->self_condition, b->conditional_mean, b->token_ids
    };
    const size_t doubles = n * sizeof(double), floats = n * sizeof(float);
    const size_t sizes[] = {doubles, doubles, doubles, 2 * sizeof(double), floats, floats,
                            r->token_count * sizeof(int32_t)};
    const size_t alignments[] = {_Alignof(double), _Alignof(double), _Alignof(double), _Alignof(double),
                                 _Alignof(float), _Alignof(float), _Alignof(int32_t)};
    schedule_region slices[sizeof offsets / sizeof offsets[0]];
    for (size_t i = 0; i < sizeof slices / sizeof slices[0]; ++i) {
        if (offsets[i] > r->arena_bytes || sizes[i] > r->arena_bytes - offsets[i]) return 0;
        slices[i] = (schedule_region){r->arena + offsets[i], sizes[i], alignments[i]};
    }
    if (!valid_regions(slices, sizeof slices / sizeof slices[0])) return 0;

    for (size_t i = 0; i < n; ++i)
        if (!isfinite(r->initial_latent[i])) return 0;
    for (size_t i = 0; i < r->noise_elements; ++i)
        if (!isfinite(r->noise[i])) return 0;
    for (size_t i = 0; i < r->schedule_count; ++i)
        if (!isfinite(r->schedule[i]) || (i && r->schedule[i] >= r->schedule[i - 1])) return 0;
    return 1;
}

static int cancelled(const ck_diffusion_schedule_request *r)
{
    return r->should_cancel && r->should_cancel(r->cancel_context);
}

int ck_run_checked_diffusion_schedule(const ck_diffusion_schedule_request *r)
{
    if (!validate_request(r)) return -1;
    if (cancelled(r)) return -7;

    const ck_diffusion_step_bindings *b = &r->bindings;
    uint8_t *arena = r->arena;
    const size_t n = r->latent_elements;
    const size_t final = r->schedule_count - 1;
    double *gamma = (double *)(arena + b->gamma);
    memcpy(arena + b->latent, r->initial_latent, n * sizeof(double));
    memset(arena + b->self_condition, 0, n * sizeof(float));
    for (size_t i = 0; i < r->schedule_count; ++i) {
        if (cancelled(r)) return -7;
        gamma[0] = r->schedule[i];
        if (i == final) {
            gamma[1] = gamma[0];
            memset(arena + b->noise, 0, n * sizeof(double));
        } else {
            gamma[1] = r->schedule[i + 1];
            memcpy(arena + b->noise, r->noise + i * n, n * sizeof(double));
        }
        const int status = r->step(arena, r->arena_bytes);
        if (status) return status;
        if (i != final) {
            memcpy(arena + b->latent, arena + b->next_latent, n * sizeof(double));
            memcpy(arena + b->self_condition, arena + b->conditional_mean, n * sizeof(float));
        }
    }
    const int32_t *ids = (const int32_t *)(arena + b->token_ids);
    for (size_t i = 0; i < r->token_count; ++i)
        if (ids[i] < 0 || (size_t)ids[i] >= r->vocabulary) return -3;
    if (cancelled(r)) return -7;
    memcpy(r->output_ids, ids, r->token_count * sizeof(*ids));
    return 0;
}
