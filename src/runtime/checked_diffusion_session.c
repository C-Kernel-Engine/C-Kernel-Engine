#include "ckernel_checked_diffusion_session.h"

#include <math.h>
#include <string.h>

typedef struct {
    uintptr_t begin, end;
} span;

static int region(span *out, const void *ptr, size_t elements, size_t width)
{
    const uintptr_t begin = (uintptr_t)ptr;
    if (!ptr || begin % width || !elements || elements > (UINTPTR_MAX - begin) / width)
        return 0;
    *out = (span){begin, begin + elements * width};
    return 1;
}

static int disjoint(span a, span b)
{
    return a.end <= b.begin || b.end <= a.begin;
}

static int external(const ck_diffusion_session *s, span value)
{
    span arena, state, bindings;
    return region(&arena, s->arena, s->arena_bytes, 1) &&
           region(&state, s, sizeof(*s), 1) &&
           region(&bindings, s->planes, s->plane_count * sizeof(*s->planes), 1) &&
           disjoint(value, arena) && disjoint(value, state) && disjoint(value, bindings);
}

static int slot(const ck_diffusion_session *s, span *out, size_t offset, size_t count, size_t width)
{
    return offset <= s->arena_bytes && count <= (s->arena_bytes - offset) / width &&
           region(out, s->arena + offset, count, width);
}

static int plane_slots(const ck_diffusion_session *s, size_t i, span slots[3])
{
    const ck_diffusion_cache_plane *p = &s->planes[i];
    return slot(s, &slots[0], p->cache, s->capacity * s->kv_width, sizeof(uint16_t)) &&
           slot(s, &slots[1], p->prefill_rows, s->prompt_rows * s->kv_width, sizeof(uint16_t)) &&
           slot(s, &slots[2], p->append_rows, s->block_rows * s->kv_width, sizeof(uint16_t));
}

static int valid_session(const ck_diffusion_session *s)
{
    if (!s || (uintptr_t)s % _Alignof(ck_diffusion_session) || !s->arena || !s->planes ||
        (uintptr_t)s->planes % _Alignof(ck_diffusion_cache_plane) ||
        !s->prefill.entry || !s->append.entry || !s->step || !s->plane_count ||
        s->plane_count > SIZE_MAX / sizeof(*s->planes) ||
        !s->kv_width || !s->latent_elements || !s->prompt_rows || !s->block_rows ||
        !s->vocabulary || s->vocabulary > INT32_MAX || s->capacity > INT32_MAX ||
        s->block_rows > (size_t)INT32_MAX - s->capacity || s->prompt_rows > s->capacity ||
        s->block_rows > s->capacity - s->prompt_rows || s->capacity > SIZE_MAX / s->kv_width ||
        s->prompt_rows > SIZE_MAX / 4 || s->block_rows > SIZE_MAX / 4 ||
        s->valid_rows > s->capacity ||
        (s->valid_rows && (s->valid_rows < s->prompt_rows ||
                          (s->valid_rows - s->prompt_rows) % s->block_rows)))
        return 0;

    span arena, bindings, state;
    if (!region(&arena, s->arena, s->arena_bytes, 1) ||
        !region(&bindings, s->planes, s->plane_count * sizeof(*s->planes), 1) ||
        !region(&state, s, sizeof(*s), 1) ||
        !disjoint(arena, state) || !disjoint(bindings, state) || !disjoint(bindings, arena))
        return 0;

    const size_t p = s->prompt_rows, b = s->block_rows, n = s->latent_elements;
    const ck_diffusion_step_bindings *d = &s->step_bindings;
    const size_t offsets[] = {s->prefill.token_ids, s->prefill.positions, s->prefill.ranges,
        s->append.token_ids, s->append.positions, s->append.ranges,
        s->positions, s->conditional_ranges, s->unconditional_ranges, s->guidance,
        d->latent, d->next_latent, d->noise, d->gamma, d->self_condition, d->conditional_mean, d->token_ids};
    const size_t counts[] = {p,p,4*p,b,b,4*b,b,4*b,4*b,1,n,n,n,2,n,n,b};
    const size_t widths[] = {4,4,4,4,4,4,4,4,4,4,8,8,8,8,4,4,4};
    span fixed[sizeof offsets / sizeof offsets[0]];
    for (size_t i = 0; i < sizeof fixed / sizeof fixed[0]; ++i) {
        if (!slot(s, &fixed[i], offsets[i], counts[i], widths[i])) return 0;
        for (size_t j = 0; j < i; ++j)
            if (!disjoint(fixed[i], fixed[j])) return 0;
    }

    for (size_t i = 0; i < s->plane_count; ++i) {
        span here[3];
        if (!plane_slots(s, i, here)) return 0;
        for (size_t k = 0; k < 3; ++k) {
            for (size_t j = 0; j < sizeof fixed / sizeof fixed[0]; ++j)
                if (!disjoint(here[k], fixed[j])) return 0;
            for (size_t j = 0; j < k; ++j)
                if (!disjoint(here[k], here[j])) return 0;
        }
        for (size_t j = 0; j < i; ++j) {
            span previous[3];
            if (!plane_slots(s, j, previous)) return 0;
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    if (!disjoint(here[k], previous[l])) return 0;
        }
    }
    return 1;
}

static int cancelled(ck_diffusion_cancel cancel, void *context)
{
    return cancel && cancel(context);
}

static void bind_visibility(uint8_t *arena, size_t prefix, size_t capacity, size_t rows, int causal,
                            size_t positions_offset, size_t ranges_offset)
{
    int32_t *positions = (int32_t *)(arena + positions_offset);
    int32_t *ranges = (int32_t *)(arena + ranges_offset);
    for (size_t i = 0; i < rows; ++i) {
        positions[i] = (int32_t)(prefix + i);
        ranges[4*i] = 0;
        ranges[4*i+1] = (int32_t)prefix;
        ranges[4*i+2] = (int32_t)capacity;
        ranges[4*i+3] = (int32_t)(capacity + (causal ? i + 1 : rows));
    }
}

static int encode_clean(ck_diffusion_session *s, const int32_t *ids, size_t rows, int reset,
                        ck_diffusion_cancel cancel, void *context)
{
    const ck_diffusion_clean_entry *entry = reset ? &s->prefill : &s->append;
    const size_t prefix = reset ? 0 : s->valid_rows;
    span tokens;
    if (rows != (reset ? s->prompt_rows : s->block_rows) ||
        rows > s->capacity - prefix || (!reset && !prefix) ||
        !region(&tokens, ids, rows, sizeof(*ids)) || !external(s, tokens))
        return -1;
    for (size_t i = 0; i < rows; ++i)
        if (ids[i] < 0 || (size_t)ids[i] >= s->vocabulary) return -1;
    if (cancelled(cancel, context)) return -7;
    memcpy(s->arena + entry->token_ids, ids, rows * sizeof(*ids));
    bind_visibility(s->arena, prefix, reset ? 0 : s->capacity, rows, 1, entry->positions, entry->ranges);
    const int status = entry->entry(s->arena, s->arena_bytes);
    if (status) return status;
    if (cancelled(cancel, context)) return -7;

    const size_t stride = s->kv_width * sizeof(uint16_t);
    for (size_t i = 0; i < s->plane_count; ++i) {
        const ck_diffusion_cache_plane *plane = &s->planes[i];
        uint8_t *destination = s->arena + plane->cache;
        const size_t source = reset ? plane->prefill_rows : plane->append_rows;
        memcpy(destination + prefix * stride, s->arena + source, rows * stride);
        if (reset)
            memset(destination + rows * stride, 0, (s->capacity - rows) * stride);
    }
    s->valid_rows = prefix + rows;
    return 0;
}

int ck_prefill_diffusion_session(ck_diffusion_session *session, const int32_t *ids, size_t rows,
                                 ck_diffusion_cancel cancel, void *context)
{
    if (!valid_session(session)) return -1;
    return encode_clean(session, ids, rows, 1, cancel, context);
}

int ck_append_diffusion_session(ck_diffusion_session *session, const int32_t *ids, size_t rows,
                                ck_diffusion_cancel cancel, void *context)
{
    if (!valid_session(session)) return -1;
    return encode_clean(session, ids, rows, 0, cancel, context);
}

int ck_generate_diffusion_block(ck_diffusion_session *s, const ck_diffusion_block_request *r)
{
    if (!r || (uintptr_t)r % _Alignof(ck_diffusion_block_request) || !valid_session(s) ||
        !s->valid_rows || s->block_rows > s->capacity - s->valid_rows ||
        !isfinite(r->guidance) || r->latent_elements != s->latent_elements ||
        r->staged_capacity < s->block_rows || r->output_capacity < s->block_rows)
        return -1;
    span operands[6];
    if (!region(&operands[0], r, sizeof(*r), 1) ||
        !region(&operands[1], r->output_ids, r->output_capacity, sizeof(int32_t)) ||
        !region(&operands[2], r->staged_ids, r->staged_capacity, sizeof(int32_t)) ||
        !region(&operands[3], r->latent, r->latent_elements, sizeof(double)) ||
        !region(&operands[4], r->schedule, r->schedule_count, sizeof(double)) ||
        !region(&operands[5], r->noise, r->noise_elements, sizeof(double)))
        return -1;
    for (size_t i = 0; i < sizeof operands / sizeof operands[0]; ++i) {
        if (!external(s, operands[i])) return -1;
        for (size_t j = 0; j < i; ++j)
            if (!disjoint(operands[i], operands[j])) return -1;
    }
    if (cancelled(r->should_cancel, r->cancel_context)) return -7;

    bind_visibility(s->arena, s->valid_rows, s->capacity, s->block_rows, 0, s->positions, s->conditional_ranges);
    int32_t *ranges = (int32_t *)(s->arena + s->unconditional_ranges);
    for (size_t i = 0; i < s->block_rows; ++i) {
        const int32_t end = (int32_t)s->block_rows;
        const int32_t row[4] = {0, end, end, end};
        memcpy(ranges + 4*i, row, sizeof(row));
    }
    memcpy(s->arena + s->guidance, &r->guidance, sizeof(r->guidance));
    const ck_diffusion_schedule_request step = {
        .step = s->step, .arena = s->arena, .arena_bytes = s->arena_bytes, .bindings = s->step_bindings,
        .latent_elements = s->latent_elements, .token_count = s->block_rows, .vocabulary = s->vocabulary,
        .initial_latent = r->latent, .schedule = r->schedule, .schedule_count = r->schedule_count,
        .noise = r->noise, .noise_elements = r->noise_elements, .output_ids = r->staged_ids,
        .output_capacity = r->staged_capacity, .should_cancel = r->should_cancel, .cancel_context = r->cancel_context,
    };
    int status = ck_run_checked_diffusion_schedule(&step);
    if (status) return status;
    status = encode_clean(s, r->staged_ids, s->block_rows, 0, r->should_cancel, r->cancel_context);
    if (status) return status;
    memcpy(r->output_ids, r->staged_ids, s->block_rows * sizeof(int32_t));
    return 0;
}
