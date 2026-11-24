#include "checked_float_buffer.h"
#include "checked_parallel.h"
#include "checked_q4_k_activation.h"
#include "packed_q8_dot_ordered.h"
#include "packed_k_dot_ordered.h"
#include "packed_q8_activation_checked.h"
#include "ckernel_transformer_rows_checked.h"

#include <string.h>
#include <stdatomic.h>

enum expert_storage {
    EXPERT_F32,
    EXPERT_Q8_0,
    EXPERT_IQ4_NL,
    EXPERT_Q4_K,
    EXPERT_Q5_0,
};

#define IQ4_NL_BLOCK_BYTES 18

/* GGML IQ4_NL codebook and split-half nibble layout. */
static const int8_t iq4_nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static size_t packed_block_bytes(enum expert_storage storage)
{
    switch (storage) {
    case EXPERT_Q8_0: return sizeof(block_q8_0);
    case EXPERT_Q4_K: return sizeof(block_q4_K);
    case EXPERT_Q5_0: return sizeof(block_q5_0);
    case EXPERT_IQ4_NL: return IQ4_NL_BLOCK_BYTES;
    default: return sizeof(float);
    }
}

static size_t packed_block_elements(enum expert_storage storage)
{
    switch (storage) {
    case EXPERT_F32: return 1;
    case EXPERT_Q4_K: return QK_K;
    case EXPERT_Q5_0: return QK5_0;
    default: return QK8_0;
    }
}

/* Packed blocks are read in GGUF byte order and expanded inside the dot.
 * The FP32 activation and ordered FP64 accumulation policy is shared. */
static float expert_dot(const float *input, const unsigned char *weight,
                        size_t width, enum expert_storage storage, const unsigned char *activation)
{
    if (activation) return ck_q8_activation_dot(weight, activation, width);
    if (storage == EXPERT_Q8_0) return ck_packed_q8_dot_ordered(input, weight, width);
    if (storage == EXPERT_Q4_K) return ck_packed_q4_k_dot_ordered(input, weight, width);
    if (storage == EXPERT_Q5_0) return ck_packed_q5_0_dot_ordered(input, weight, width);
    double sum = 0.0;
    if (storage == EXPERT_F32) {
        for (size_t i = 0; i < width; ++i) {
            float value;
            memcpy(&value, weight + i * sizeof(float), sizeof(float));
            sum += (double)input[i] * (double)value;
        }
    } else {
        const size_t block_bytes = packed_block_bytes(storage);
        for (size_t i = 0; i < width; i += QK8_0, weight += block_bytes) {
            const float scale = ck_fp16_to_fp32_soft(
                (uint16_t)(weight[0] | ((uint16_t)weight[1] << 8)));
            for (size_t j = 0; j < QK8_0; ++j) {
                const unsigned pair = weight[2 + j % 16];
                const float expanded = scale * (float)iq4_nl_values[j < 16 ? pair & 15 : pair >> 4];
                sum += (double)input[i + j] * (double)expanded;
            }
        }
    }
    return (float)sum;
}

static int contains_overlap(const void *p, size_t pn, const void *q, size_t qn)
{
    return (uintptr_t)p < (uintptr_t)q + qn && (uintptr_t)q < (uintptr_t)p + pn;
}

static int region(const void *p, size_t bytes)
{
    return p && bytes && (uintptr_t)p <= UINTPTR_MAX - bytes;
}

#define ROUTED_EXPERT_ARGS \
    const float *hidden, const int32_t *ids, const float *route_weights, \
    const int32_t *expert_slots, const void *gate, const void *up, const void *down, \
    float *expert_output, float *output, float *work, \
    size_t rows, size_t hidden_dim, size_t inner, size_t experts, size_t slots, \
    size_t top_k, size_t matrix_bytes, size_t hidden_elements, size_t id_elements, size_t route_elements, \
    size_t expert_slot_elements, size_t expert_output_elements, size_t output_capacity, \
    size_t work_elements, float gate_cap, float up_cap

struct q4_hidden_ops {
    int (*pack)(const float *, unsigned char *, size_t);
    void (*matvec)(float *, const void *, const void *, int, int);
    void (*down_q5)(float *, const void *, const void *, int, int);
    void (*matvec_pair)(float *, float *, const void *, const void *, const void *, int, int);
    int group_rows;
    int dispatch_pairs;
};

static const struct q4_hidden_ops q4_hidden = {
    .pack = ck_pack_q4_input_q8_k,
    .matvec = gemv_q4_k_q8_k_avx2,
};

void gemv_q5_0_q8_0_checked_cpu(float *, const void *, const void *, int, int);
static const struct q4_hidden_ops q4_hidden_and_down = {
    .pack = ck_pack_q4_input_q8_k,
    .matvec = gemv_q4_k_q8_k_avx2,
    .down_q5 = gemv_q5_0_q8_0_checked_cpu,
};

static const struct q4_hidden_ops q4_grouped = {
    .pack = ck_pack_q4_input_q8_k,
    .matvec = gemv_q4_k_q8_k_avx2,
    .down_q5 = gemv_q5_0_q8_0_checked_cpu,
    .group_rows = 1,
};

static const struct q4_hidden_ops q4_grouped_pairs = {
    .pack = ck_pack_q4_input_q8_k,
    .matvec = gemv_q4_k_q8_k_avx2,
    .down_q5 = gemv_q5_0_q8_0_checked_cpu,
    .matvec_pair = gemv_q4_k_q8_k_pair_checked_cpu,
    .group_rows = 1,
};

static const struct q4_hidden_ops q4_pair_jobs = {
    .pack = ck_pack_q4_input_q8_k,
    .matvec = gemv_q4_k_q8_k_avx2,
    .down_q5 = gemv_q5_0_q8_0_checked_cpu,
    .matvec_pair = gemv_q4_k_q8_k_pair_checked_cpu,
    .group_rows = 1,
    .dispatch_pairs = 1,
};

struct routed_projection {
    const unsigned char *gate, *up, *down;
    size_t width, inner, gate_row, down_row, gate_expert_bytes, down_expert_bytes;
    float gate_cap, up_cap, gate_reciprocal, up_reciprocal;
    enum expert_storage storage, down_storage;
    int quantize, reference;
    const struct q4_hidden_ops *packed_q4;
};

static int finish_expert(const struct routed_projection *p, size_t slot, float *scratch, float *output)
{
    const unsigned char *down = p->down + slot * p->down_expert_bytes;
    float *gate_values = scratch, *up_values = gate_values + p->inner, *activation = up_values + p->inner;
    unsigned char *packed_activation = p->quantize ? (unsigned char *)(activation + p->inner) + p->gate_row : NULL;
    if (p->packed_q4 && p->packed_q4->down_q5)
        packed_activation = (unsigned char *)(activation + p->inner) + p->width / QK_K * sizeof(block_q8_K);

    if (p->reference) {
        for (size_t i = 0; i < p->inner; ++i) {
            const float g = gate_values[i], u = up_values[i];
            if (!isfinite(g) || !isfinite(u)) return -3;
            const float capped_gate = p->gate_cap * tanhf(g * p->gate_reciprocal);
            const float capped_up = p->up_cap * tanhf(u * p->up_reciprocal);
            const float sigmoid = 1.0f / (1.0f + expf(-g));
            activation[i] = (capped_gate * sigmoid) * capped_up;
            if (!isfinite(activation[i])) return -3;
        }
    } else if (softcapped_swiglu_rows_checked_f32(gate_values, p->inner, up_values, p->inner,
                   activation, p->inner, p->inner, p->gate_cap, p->up_cap)) return -3;
    if (packed_activation && ck_pack_q8_activation_rne(activation, packed_activation, p->inner)) return -3;
    if (p->packed_q4 && p->packed_q4->down_q5 && p->down_storage == EXPERT_Q5_0) {
        p->packed_q4->down_q5(output, down, packed_activation, (int)p->width, (int)p->inner);
        if (!ck_checked_finite(output, p->width)) return -3;
    } else {
        for (size_t i = 0; i < p->width; ++i) {
            output[i] = expert_dot(activation, down + i * p->down_row, p->inner, p->down_storage, packed_activation);
            if (!isfinite(output[i])) return -3;
        }
    }
    return 0;
}

static int project_expert(const struct routed_projection *p, const float *input,
                          const unsigned char *packed_hidden, size_t slot, float *scratch,
                          float *output)
{
    const size_t offset = slot * p->gate_expert_bytes;
    const unsigned char *gate = p->gate + offset, *up = p->up + offset;
    float *gate_values = scratch, *up_values = scratch + p->inner;
    if (p->packed_q4) {
        p->packed_q4->matvec(gate_values, gate, packed_hidden, (int)p->inner, (int)p->width);
        p->packed_q4->matvec(up_values, up, packed_hidden, (int)p->inner, (int)p->width);
    } else {
        for (size_t i = 0; i < p->inner; ++i) {
            if (packed_hidden) {
                ck_q8_activation_dot_pair(gate + i * p->gate_row, up + i * p->gate_row, packed_hidden,
                                          p->width, gate_values + i, up_values + i);
            } else {
                gate_values[i] = expert_dot(input, gate + i * p->gate_row, p->width, p->storage, NULL);
                up_values[i] = expert_dot(input, up + i * p->gate_row, p->width, p->storage, NULL);
            }
        }
    }
    return finish_expert(p, slot, scratch, output);
}

static int project_expert_pair(const struct routed_projection *p, size_t slot,
                               const unsigned char *x0, const unsigned char *x1,
                               float *work0, float *work1, float *out0, float *out1)
{
    const size_t offset = slot * p->gate_expert_bytes;
    p->packed_q4->matvec_pair(work0, work1, p->gate + offset, x0, x1, (int)p->inner, (int)p->width);
    p->packed_q4->matvec_pair(work0 + p->inner, work1 + p->inner, p->up + offset,
                            x0, x1, (int)p->inner, (int)p->width);
    if (finish_expert(p, slot, work0, out0)) return -3;
    return finish_expert(p, slot, work1, out1);
}

static inline int combine_expert(const float *expert, float *mixture, size_t width,
                                 float weight, size_t index, int reference)
{
    for (size_t i = 0; i < width; ++i) {
        if (reference) {
            const volatile float weighted = expert[i] * weight;
            mixture[i] = index == 0 ? weighted : mixture[i] + weighted;
        } else {
            mixture[i] += expert[i] * weight;
        }
        if (!isfinite(mixture[i])) return -3;
    }
    return 0;
}

static void order_slots(const int32_t *ids, size_t experts, size_t top_k, size_t *ordered_slots)
{
    size_t next = 0;
    for (size_t id = 0; id < experts; ++id)
        for (size_t slot = 0; slot < top_k; ++slot)
            if ((size_t)ids[slot] == id) ordered_slots[next++] = slot;
}

struct expert_pair_job {
    uint32_t first, second;
};

_Static_assert(sizeof(struct expert_pair_job) == 2 * sizeof(float) &&
               _Alignof(struct expert_pair_job) <= _Alignof(float),
               "expert job workspace layout changed");

struct routed_work {
    const struct routed_projection *projection;
    const float *hidden, *route_weights;
    const int32_t *ids, *slots;
    float *row_work, *staged_experts, *staged_output;
    const unsigned char *packed_rows;
    const struct expert_pair_job *jobs;
    const ck_threadpool_t *pool;
    size_t packed_row_bytes;
    size_t row_scratch, scratch_rows, rows, experts, top_k;
    atomic_int failed;
};

static unsigned char *hidden_workspace(const struct routed_projection *p, float *scratch)
{
    return p->quantize || p->packed_q4 ? (unsigned char *)(scratch + 3 * p->inner) : NULL;
}

static int pack_hidden(const struct routed_projection *p, const float *input, unsigned char *packed)
{
    if (!packed) return 0;
    if (p->packed_q4) return p->packed_q4->pack(input, packed, p->width);
    return ck_pack_q8_activation_rne(input, packed, p->width);
}

static void project_routes(size_t begin, size_t end, void *data)
{
    struct routed_work *work = data;
    const struct routed_projection *p = work->projection;
    for (size_t route = begin; route < end; ++route) {
        const unsigned char *packed_hidden = hidden_workspace(p, work->row_work);
        if (project_expert(p, work->hidden, packed_hidden, work->slots[work->ids[route]],
                           work->row_work + route * work->row_scratch,
                           work->staged_experts + route * p->width)) {
            atomic_store_explicit(&work->failed, 1, memory_order_relaxed);
            return;
        }
    }
}

static void project_rows(size_t begin, size_t end, void *data)
{
    struct routed_work *work = data;
    const struct routed_projection *p = work->projection;
    for (size_t row = begin; row < end; ++row) {
        float *scratch = work->row_work + (work->scratch_rows > 1 ? row * work->row_scratch : 0);
        unsigned char *packed_hidden = hidden_workspace(p, scratch);
        const float *input = work->hidden + row * p->width;
        if (pack_hidden(p, input, packed_hidden)) {
            atomic_store_explicit(&work->failed, 1, memory_order_relaxed);
            return;
        }
        size_t ordered_slots[1024];
        if (!p->reference) order_slots(work->ids + row * work->top_k, work->experts, work->top_k, ordered_slots);
        for (size_t index = 0; index < work->top_k; ++index) {
            const size_t route = row * work->top_k + (p->reference ? index : ordered_slots[index]);
            float *expert = work->staged_experts + route * p->width;
            if (project_expert(p, input, packed_hidden, work->slots[work->ids[route]], scratch, expert) ||
                combine_expert(expert, work->staged_output + row * p->width, p->width,
                               work->route_weights[route], index, p->reference)) {
                atomic_store_explicit(&work->failed, 1, memory_order_relaxed);
                return;
            }
        }
    }
}

/* Keep each expert's weights hot across routed rows. Route results remain in
 * their original slots and are mixed only after all expert work completes. */
static void project_grouped_experts(size_t begin, size_t end, void *opaque)
{
    struct routed_work *work = opaque;
    const struct routed_projection *p = work->projection;
    const size_t scratch_per_expert = work->row_scratch * (p->packed_q4->matvec_pair ? 2 : 1);
    for (size_t id = begin; id < end; ++id) {
        if (work->slots[id] < 0) continue;
        float *scratch = work->row_work + id * scratch_per_expert;
        size_t pending = SIZE_MAX;
        for (size_t row = 0; row < work->rows; ++row) {
            const unsigned char *packed = work->packed_rows + row * work->packed_row_bytes;
            for (size_t i = 0; i < work->top_k; ++i) {
                const size_t route = row * work->top_k + i;
                if ((size_t)work->ids[route] != id) continue;
                if (p->packed_q4->matvec_pair && pending == SIZE_MAX) {
                    pending = route;
                    break;
                }
                const int status = pending == SIZE_MAX
                    ? project_expert(p, work->hidden + row * p->width, packed, work->slots[id], scratch,
                                     work->staged_experts + route * p->width)
                    : project_expert_pair(p, work->slots[id],
                        work->packed_rows + pending / work->top_k * work->packed_row_bytes, packed,
                        scratch, scratch + work->row_scratch,
                        work->staged_experts + pending * p->width, work->staged_experts + route * p->width);
                if (status) {
                    atomic_store_explicit(&work->failed, 1, memory_order_relaxed);
                    return;
                }
                pending = SIZE_MAX;
                break;
            }
        }
        if (pending != SIZE_MAX) {
            const size_t row = pending / work->top_k;
            if (project_expert(p, work->hidden + row * p->width,
                    work->packed_rows + row * work->packed_row_bytes, work->slots[id], scratch,
                    work->staged_experts + pending * p->width)) {
                atomic_store_explicit(&work->failed, 1, memory_order_relaxed);
                return;
            }
        }
    }
}

static size_t collect_expert_pair_jobs(const struct routed_work *work, struct expert_pair_job *jobs)
{
    size_t count = 0;
    for (size_t id = 0; id < work->experts; ++id) {
        uint32_t pending = UINT32_MAX;
        for (size_t route = 0; route < work->rows * work->top_k; ++route) {
            if ((size_t)work->ids[route] != id) continue;
            if (pending == UINT32_MAX) {
                pending = (uint32_t)route;
            } else {
                jobs[count++] = (struct expert_pair_job){pending, (uint32_t)route};
                pending = UINT32_MAX;
            }
        }
        if (pending != UINT32_MAX) jobs[count++] = (struct expert_pair_job){pending, UINT32_MAX};
    }
    return count;
}

static void project_expert_pair_jobs(size_t begin, size_t end, void *opaque)
{
    struct routed_work *work = opaque;
    const struct routed_projection *p = work->projection;
    const int worker = ck_threadpool_thread_id(work->pool);
    const size_t scratch_slot = worker < 0 ? 0 : (size_t)worker;
    float *scratch = work->row_work + scratch_slot * 2 * work->row_scratch;
    for (size_t i = begin; i < end; ++i) {
        const uint32_t first = work->jobs[i].first, second = work->jobs[i].second;
        const size_t row = first / work->top_k;
        const size_t slot = work->slots[work->ids[first]];
        const unsigned char *packed = work->packed_rows + row * work->packed_row_bytes;
        float *out = work->staged_experts + (size_t)first * p->width;
        const int status = second == UINT32_MAX
            ? project_expert(p, work->hidden + row * p->width, packed, slot, scratch, out)
            : project_expert_pair(p, slot, packed,
                work->packed_rows + second / work->top_k * work->packed_row_bytes,
                scratch, scratch + work->row_scratch, out, work->staged_experts + (size_t)second * p->width);
        if (status) {
            atomic_store_explicit(&work->failed, 1, memory_order_relaxed);
            return;
        }
    }
}

static int routed_softcapped_experts(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes,
                                     enum expert_storage storage, enum expert_storage down_storage,
                                     int quantize, int reference_moe, const struct q4_hidden_ops *packed_q4)
{
    /* Bounds keep workspace arithmetic valid on supported size_t widths. */
    if (!rows || rows > CHECKED_ROUTED_EXPERT_MAX_ROWS || !hidden_dim || hidden_dim > 8192 ||
        !inner || inner > 8192 || !experts || experts > 1024 || !slots ||
        slots > experts || !top_k || top_k > experts ||
        !isfinite(gate_cap) || gate_cap <= 0.0f || !isfinite(up_cap) || up_cap <= 0.0f)
        return -1;
    const size_t gate_block = packed_block_elements(storage);
    const size_t down_block = packed_block_elements(down_storage);
    if (hidden_dim % gate_block || inner % down_block) return -1;
    const size_t gate_row = hidden_dim / gate_block * packed_block_bytes(storage);
    const size_t down_row = inner / down_block * packed_block_bytes(down_storage);
    const size_t gate_expert_bytes = inner * gate_row;
    const size_t down_expert_bytes = hidden_dim * down_row;
    const size_t route_count = rows * top_k;
    const size_t output_elements = rows * hidden_dim;
    const size_t row_outputs = route_count * hidden_dim;
    if (packed_q4 && ((uintptr_t)gate % _Alignof(block_q4_K) ||
                     (uintptr_t)up % _Alignof(block_q4_K))) return -1;
    if (packed_q4 && packed_q4->down_q5 && down_storage == EXPERT_Q5_0 &&
        (uintptr_t)down % _Alignof(block_q5_0)) return -1;
    size_t packed_bytes = packed_q4 ? hidden_dim / QK_K * sizeof(block_q8_K) :
                          quantize ? gate_row + down_row : 0;
    if (packed_q4 && packed_q4->down_q5) packed_bytes += inner / QK8_0 * sizeof(block_q8_0);
    const size_t packed_elements = (packed_bytes + sizeof(float) - 1) / sizeof(float);
    const size_t row_scratch = 3 * inner + packed_elements;
    const size_t staged_elements = row_outputs + output_elements;
    const size_t scratch_rows = work_elements >= staged_elements + rows * row_scratch ? rows : 1;
    const size_t route_scratch = rows == 1 && top_k > 1 &&
        work_elements >= staged_elements + top_k * row_scratch ? top_k : 0;
    const size_t packed_row_bytes = hidden_dim / QK_K * sizeof(block_q8_K);
    const int grouped = packed_q4 && packed_q4->group_rows && rows > 1;
    const size_t grouped_pack_elements = grouped ? rows * packed_row_bytes / sizeof(float) : 0;
    const int dispatch_pairs = grouped && packed_q4->dispatch_pairs;
    const size_t jobs_elements = dispatch_pairs ? 2 * route_count : 0;
    const size_t scratch_count = dispatch_pairs ? 2 * CK_THREADPOOL_MAX_THREADS :
        grouped ? experts * (packed_q4->matvec_pair ? 2 : 1) : route_scratch ? route_scratch : scratch_rows;
    const size_t scratch_elements = staged_elements + grouped_pack_elements + jobs_elements + scratch_count * row_scratch;
    if (hidden_elements < output_elements || id_elements < route_count || route_elements < route_count ||
        expert_slot_elements < experts || expert_output_elements < row_outputs ||
        output_capacity < output_elements || work_elements < scratch_elements ||
        scratch_elements > SIZE_MAX / sizeof(float)) return -1;
    if (gate_expert_bytes > SIZE_MAX / slots || down_expert_bytes > SIZE_MAX / slots ||
        matrix_bytes != gate_expert_bytes * slots ||
        down_matrix_bytes != down_expert_bytes * slots) return -1;
    const void *buffers[] = {hidden, ids, route_weights, expert_slots, gate, up, down,
                             expert_output, output, work};
    const size_t sizes[] = {output_elements * sizeof(float), route_count * sizeof(int32_t),
                            route_count * sizeof(float), experts * sizeof(int32_t),
                            matrix_bytes, matrix_bytes, down_matrix_bytes, row_outputs * sizeof(float),
                            output_elements * sizeof(float),
                            scratch_elements * sizeof(float)};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
        if (!region(buffers[i], sizes[i])) return -1;
        if (i < 4 || i >= 7) {
            if ((uintptr_t)buffers[i] % _Alignof(float)) return -1;
        }
        if (i < 7) continue;
        for (size_t j = 0; j < i; ++j)
            if (contains_overlap(buffers[i], sizes[i], buffers[j], sizes[j])) return -1;
    }
    if (!ck_checked_finite(hidden, output_elements) ||
        !ck_checked_finite(route_weights, route_count)) return -2;

    unsigned char assigned[1024] = {0};
    for (size_t id = 0; id < experts; ++id) {
        const int32_t slot = expert_slots[id];
        if (slot == -1) continue;
        if (slot < 0 || (size_t)slot >= slots || assigned[slot]) return -2;
        assigned[slot] = 1;
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t k = 0; k < top_k; ++k) {
            const size_t route = row * top_k + k;
            const int32_t id = ids[route];
            if (id < 0 || (size_t)id >= experts || expert_slots[id] < 0 ||
                (size_t)expert_slots[id] >= slots || route_weights[route] < 0.0f) return -2;
            for (size_t prev = 0; prev < k; ++prev)
                if (id == ids[row * top_k + prev]) return -2;
        }
    }

    const float gate_reciprocal = reference_moe ? 1.0f / gate_cap : 0.0f;
    const float up_reciprocal = reference_moe ? 1.0f / up_cap : 0.0f;
    if (!isfinite(gate_reciprocal) || !isfinite(up_reciprocal)) return -3;

    float *staged_experts = work;
    float *staged_output = staged_experts + row_outputs;
    float *packed_rows = staged_output + output_elements;
    float *jobs_buffer = packed_rows + grouped_pack_elements;
    float *row_work = jobs_buffer + jobs_elements;
    memset(staged_output, 0, output_elements * sizeof(float));
    const struct routed_projection projection = {
        .gate = gate, .up = up, .down = down, .width = hidden_dim, .inner = inner,
        .gate_row = gate_row, .down_row = down_row,
        .gate_expert_bytes = gate_expert_bytes, .down_expert_bytes = down_expert_bytes,
        .gate_cap = gate_cap, .up_cap = up_cap, .gate_reciprocal = gate_reciprocal, .up_reciprocal = up_reciprocal,
        .storage = storage, .down_storage = down_storage, .quantize = quantize, .reference = reference_moe, .packed_q4 = packed_q4,
    };
    struct routed_work task = {
        .projection = &projection, .hidden = hidden, .route_weights = route_weights,
        .ids = ids, .slots = expert_slots, .row_work = row_work,
        .staged_experts = staged_experts, .staged_output = staged_output,
        .packed_rows = (const unsigned char *)packed_rows, .packed_row_bytes = packed_row_bytes,
        .row_scratch = row_scratch, .scratch_rows = scratch_rows, .rows = rows, .experts = experts, .top_k = top_k,
    };
    atomic_init(&task.failed, 0);
    if (grouped) {
        unsigned char *packs = (unsigned char *)packed_rows;
        for (size_t row = 0; row < rows; ++row) {
            if (pack_hidden(&projection, hidden + row * hidden_dim, packs + row * packed_row_bytes)) return -3;
        }
        if (dispatch_pairs) {
            struct expert_pair_job *jobs = (struct expert_pair_job *)jobs_buffer;
            const size_t count = collect_expert_pair_jobs(&task, jobs);
            task.jobs = jobs;
            task.pool = ck_threadpool_global();
            ck_checked_parallel_for(count, 1, project_expert_pair_jobs, &task);
        } else {
            ck_checked_parallel_for(experts, 1, project_grouped_experts, &task);
        }
        if (atomic_load_explicit(&task.failed, memory_order_relaxed)) return -3;
        for (size_t row = 0; row < rows; ++row) {
            size_t ordered_slots[1024];
            order_slots(ids + row * top_k, experts, top_k, ordered_slots);
            for (size_t index = 0; index < top_k; ++index) {
                const size_t route = row * top_k + ordered_slots[index];
                if (combine_expert(staged_experts + route * hidden_dim, staged_output + row * hidden_dim,
                                   hidden_dim, route_weights[route], index, 0)) return -3;
            }
        }
    } else if (route_scratch) {
        unsigned char *packed_hidden = hidden_workspace(&projection, row_work);
        if (pack_hidden(&projection, hidden, packed_hidden)) return -3;
        /* Each task writes its declared slot. Mixing preserves the FP32 order. */
        ck_checked_parallel_for(top_k, 1, project_routes, &task);
        if (atomic_load_explicit(&task.failed, memory_order_relaxed)) return -3;
        size_t ordered_slots[1024];
        if (!reference_moe) order_slots(ids, experts, top_k, ordered_slots);
        for (size_t index = 0; index < top_k; ++index) {
            const size_t route = reference_moe ? index : ordered_slots[index];
            if (combine_expert(staged_experts + route * hidden_dim, staged_output, hidden_dim,
                               route_weights[route], index, reference_moe)) return -3;
        }
    } else {
        ck_checked_parallel_for(rows, scratch_rows > 1 ? 1 : (int)rows, project_rows, &task);
        if (atomic_load_explicit(&task.failed, memory_order_relaxed)) return -3;
    }
    memcpy(expert_output, staged_experts, row_outputs * sizeof(float));
    memcpy(output, staged_output, output_elements * sizeof(float));
    return 0;
}

#define ROUTED_EXPERT_FORWARD(storage, down_storage, down_bytes, quantize, reference_moe, packed_q4) \
    routed_softcapped_experts(hidden, ids, route_weights, expert_slots, gate, up, down, \
                             expert_output, output, work, rows, hidden_dim, inner, experts, \
                             slots, top_k, matrix_bytes, hidden_elements, id_elements, route_elements, \
                             expert_slot_elements, expert_output_elements, output_capacity, \
                             work_elements, gate_cap, up_cap, down_bytes, storage, down_storage, quantize, reference_moe, packed_q4)

int routed_softcapped_experts_checked_f32(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_F32, EXPERT_F32, matrix_bytes, 0, 0, NULL);
}

int routed_softcapped_experts_checked_q8_0(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q8_0, EXPERT_Q8_0, matrix_bytes, 0, 0, NULL);
}

int routed_softcapped_experts_checked_iq4_nl(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_IQ4_NL, EXPERT_IQ4_NL, matrix_bytes, 0, 0, NULL);
}


int routed_softcapped_experts_checked_q8_0_q8_activation(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q8_0, EXPERT_Q8_0, matrix_bytes, 1, 0, NULL);
}

int routed_softcapped_experts_checked_q8_0_q8_activation_scaled_situ(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q8_0, EXPERT_Q8_0, matrix_bytes, 1, 1, NULL);
}


int routed_softcapped_experts_checked_q4_k_q5_0(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q5_0, down_matrix_bytes, 0, 0, NULL);
}

int routed_softcapped_experts_checked_q4_k_q8_0(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q8_0, down_matrix_bytes, 0, 0, NULL);
}

int routed_softcapped_experts_checked_q4_k_q5_0_q8_k_hidden(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q5_0, down_matrix_bytes, 0, 0, &q4_hidden);
}

int routed_softcapped_experts_checked_q4_k_q8_0_q8_k_hidden(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q8_0, down_matrix_bytes, 0, 0, &q4_hidden);
}

int routed_softcapped_experts_checked_q4_k_q5_0_q8_k_hidden_q8_0_down(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q5_0, down_matrix_bytes, 0, 0, &q4_hidden_and_down);
}

int routed_softcapped_experts_checked_q4_k_q8_0_q8_k_hidden_q8_0_down(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q8_0, down_matrix_bytes, 0, 0, &q4_hidden_and_down);
}

int routed_softcapped_experts_checked_q4_k_q5_0_grouped(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q5_0, down_matrix_bytes, 0, 0, &q4_grouped);
}

int routed_softcapped_experts_checked_q4_k_q8_0_grouped(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q8_0, down_matrix_bytes, 0, 0, &q4_grouped);
}

int routed_softcapped_experts_checked_q4_k_q5_0_grouped_pair(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q5_0, down_matrix_bytes, 0, 0, &q4_grouped_pairs);
}

int routed_softcapped_experts_checked_q4_k_q8_0_grouped_pair(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q8_0, down_matrix_bytes, 0, 0, &q4_grouped_pairs);
}

int routed_softcapped_experts_checked_q4_k_q5_0_pair_jobs(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q5_0, down_matrix_bytes, 0, 0, &q4_pair_jobs);
}

int routed_softcapped_experts_checked_q4_k_q8_0_pair_jobs(ROUTED_EXPERT_ARGS, size_t down_matrix_bytes)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q4_K, EXPERT_Q8_0, down_matrix_bytes, 0, 0, &q4_pair_jobs);
}
