#include "checked_float_buffer.h"
#include "packed_q8_dot_ordered.h"
#include "packed_q8_activation_checked.h"
#include "ckernel_transformer_rows_checked.h"

#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

enum expert_storage {
    EXPERT_F32,
    EXPERT_Q8_0,
    EXPERT_IQ4_NL,
};

#define IQ4_NL_BLOCK_BYTES 18
#define CHECKED_ROUTED_EXPERT_MAX_ROWS 64

/* GGML IQ4_NL codebook and split-half nibble layout. */
static const int8_t iq4_nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static size_t packed_block_bytes(enum expert_storage storage)
{
    return storage == EXPERT_Q8_0 ? sizeof(block_q8_0) : IQ4_NL_BLOCK_BYTES;
}

/* Packed blocks are read in GGUF byte order and expanded inside the dot.
 * The FP32 activation and ordered FP64 accumulation policy is shared. */
static float expert_dot(const float *input, const unsigned char *weight,
                        size_t width, enum expert_storage storage, const unsigned char *activation)
{
    if (activation) return ck_q8_activation_dot(weight, activation, width);
    if (storage == EXPERT_Q8_0) return ck_packed_q8_dot_ordered(input, weight, width);
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

struct routed_projection {
    const unsigned char *gate, *up, *down;
    size_t width, inner, gate_row, down_row, expert_bytes;
    float gate_cap, up_cap, gate_reciprocal, up_reciprocal;
    enum expert_storage storage;
    int quantize, reference;
};

static int project_expert(const struct routed_projection *p, const float *input,
                          const unsigned char *packed_hidden, size_t slot, float *scratch,
                          float *output)
{
    const size_t offset = slot * p->expert_bytes;
    const unsigned char *gate = p->gate + offset, *up = p->up + offset, *down = p->down + offset;
    float *gate_values = scratch, *up_values = gate_values + p->inner, *activation = up_values + p->inner;
    unsigned char *packed_activation = p->quantize ? (unsigned char *)(activation + p->inner) + p->gate_row : NULL;

    for (size_t i = 0; i < p->inner; ++i) {
        if (packed_hidden) {
            ck_q8_activation_dot_pair(gate + i * p->gate_row, up + i * p->gate_row, packed_hidden,
                                      p->width, gate_values + i, up_values + i);
        } else {
            gate_values[i] = expert_dot(input, gate + i * p->gate_row, p->width, p->storage, NULL);
            up_values[i] = expert_dot(input, up + i * p->gate_row, p->width, p->storage, NULL);
        }
    }
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
    for (size_t i = 0; i < p->width; ++i) {
        output[i] = expert_dot(activation, down + i * p->down_row, p->inner, p->storage, packed_activation);
        if (!isfinite(output[i])) return -3;
    }
    return 0;
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

static int routed_softcapped_experts(ROUTED_EXPERT_ARGS, enum expert_storage storage, int quantize, int reference_moe)
{
    /* Bounds keep workspace arithmetic valid on supported size_t widths. */
    if (!rows || rows > CHECKED_ROUTED_EXPERT_MAX_ROWS || !hidden_dim || hidden_dim > 8192 ||
        !inner || inner > 8192 || !experts || experts > 1024 || !slots ||
        slots > experts || !top_k || top_k > experts ||
        !isfinite(gate_cap) || gate_cap <= 0.0f || !isfinite(up_cap) || up_cap <= 0.0f)
        return -1;
    const int packed = storage != EXPERT_F32;
    if (packed && (hidden_dim % QK8_0 || inner % QK8_0)) return -1;
    const size_t block_bytes = packed ? packed_block_bytes(storage) : sizeof(float);
    const size_t block_elements = packed ? QK8_0 : 1;
    const size_t gate_row = hidden_dim / block_elements * block_bytes;
    const size_t down_row = inner / block_elements * block_bytes;
    const size_t expert_bytes = inner * gate_row;
    const size_t route_count = rows * top_k;
    const size_t output_elements = rows * hidden_dim;
    const size_t row_outputs = route_count * hidden_dim;
    const size_t packed_elements = quantize ? (gate_row + down_row + sizeof(float) - 1) / sizeof(float) : 0;
    const size_t row_scratch = 3 * inner + packed_elements;
    const size_t staged_elements = row_outputs + output_elements;
    size_t scratch_rows = 1;
#ifdef _OPENMP
    if (work_elements >= staged_elements + rows * row_scratch) scratch_rows = rows;
#endif
    size_t route_scratch = 0;
#ifdef _OPENMP
    if (quantize && rows == 1 && top_k > 1 && omp_get_max_threads() > 1 &&
        work_elements >= staged_elements + top_k * row_scratch)
        route_scratch = top_k;
#endif
    const size_t scratch_elements = staged_elements + (route_scratch ? route_scratch : scratch_rows) * row_scratch;
    if (hidden_elements < output_elements || id_elements < route_count || route_elements < route_count ||
        expert_slot_elements < experts || expert_output_elements < row_outputs ||
        output_capacity < output_elements || work_elements < scratch_elements) return -1;
    if (expert_bytes != hidden_dim * down_row || expert_bytes > SIZE_MAX / slots ||
        matrix_bytes != expert_bytes * slots) return -1;
    const void *buffers[] = {hidden, ids, route_weights, expert_slots, gate, up, down,
                             expert_output, output, work};
    const size_t sizes[] = {output_elements * sizeof(float), route_count * sizeof(int32_t),
                            route_count * sizeof(float), experts * sizeof(int32_t),
                            matrix_bytes, matrix_bytes, matrix_bytes, row_outputs * sizeof(float),
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
    float *row_work = staged_output + output_elements;
    memset(staged_output, 0, output_elements * sizeof(float));
    const struct routed_projection projection = {
        .gate = gate, .up = up, .down = down, .width = hidden_dim, .inner = inner,
        .gate_row = gate_row, .down_row = down_row, .expert_bytes = expert_bytes,
        .gate_cap = gate_cap, .up_cap = up_cap, .gate_reciprocal = gate_reciprocal, .up_reciprocal = up_reciprocal,
        .storage = storage, .quantize = quantize, .reference = reference_moe,
    };
    int failed = 0;
#ifdef _OPENMP
    if (route_scratch) {
        unsigned char *packed_hidden = (unsigned char *)(row_work + 3 * inner);
        if (ck_pack_q8_activation_rne(hidden, packed_hidden, hidden_dim)) return -3;
        /* Each task writes its declared slot. Mixing below preserves the
         * original expert-ID or router-slot FP32 order. */
        const int max_threads = omp_get_max_threads();
        const int threads = (size_t)max_threads < top_k ? max_threads : (int)top_k;
#pragma omp parallel for num_threads(threads) reduction(|:failed)
        for (size_t route = 0; route < top_k; ++route) {
            failed |= project_expert(&projection, hidden, packed_hidden, expert_slots[ids[route]],
                                     row_work + route * row_scratch, staged_experts + route * hidden_dim) != 0;
        }
        if (failed) return -3;
        size_t ordered_slots[1024];
        if (!reference_moe) order_slots(ids, experts, top_k, ordered_slots);
        for (size_t index = 0; index < top_k; ++index) {
            const size_t route = reference_moe ? index : ordered_slots[index];
            if (combine_expert(staged_experts + route * hidden_dim, staged_output, hidden_dim,
                               route_weights[route], index, reference_moe)) return -3;
        }
    } else
#endif
    {
#ifdef _OPENMP
#pragma omp parallel for if(scratch_rows > 1) reduction(|:failed)
#endif
        for (size_t row = 0; row < rows; ++row) {
            float *scratch = row_work + (scratch_rows > 1 ? row * row_scratch : 0);
            unsigned char *packed_hidden = quantize ? (unsigned char *)(scratch + 3 * inner) : NULL;
            const float *input = hidden + row * hidden_dim;
            if (packed_hidden && ck_pack_q8_activation_rne(input, packed_hidden, hidden_dim)) {
                failed = 1;
                continue;
            }
            size_t ordered_slots[1024];
            if (!reference_moe) order_slots(ids + row * top_k, experts, top_k, ordered_slots);
            for (size_t index = 0; index < top_k; ++index) {
                const size_t route = row * top_k + (reference_moe ? index : ordered_slots[index]);
                float *expert = staged_experts + route * hidden_dim;
                if (project_expert(&projection, input, packed_hidden, expert_slots[ids[route]], scratch, expert) ||
                    combine_expert(expert, staged_output + row * hidden_dim, hidden_dim,
                                   route_weights[route], index, reference_moe)) {
                    failed = 1;
                    break;
                }
            }
        }
        if (failed) return -3;
    }
    memcpy(expert_output, staged_experts, row_outputs * sizeof(float));
    memcpy(output, staged_output, output_elements * sizeof(float));
    return 0;
}

#define ROUTED_EXPERT_FORWARD(storage, quantize, reference_moe) \
    routed_softcapped_experts(hidden, ids, route_weights, expert_slots, gate, up, down, \
                             expert_output, output, work, rows, hidden_dim, inner, experts, \
                             slots, top_k, matrix_bytes, hidden_elements, id_elements, route_elements, \
                             expert_slot_elements, expert_output_elements, output_capacity, \
                             work_elements, gate_cap, up_cap, storage, quantize, reference_moe)

int routed_softcapped_experts_checked_f32(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_F32, 0, 0);
}

int routed_softcapped_experts_checked_q8_0(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q8_0, 0, 0);
}

int routed_softcapped_experts_checked_iq4_nl(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_IQ4_NL, 0, 0);
}


int routed_softcapped_experts_checked_q8_0_q8_activation(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q8_0, 1, 0);
}

int routed_softcapped_experts_checked_q8_0_q8_activation_scaled_situ(ROUTED_EXPERT_ARGS)
{
    return ROUTED_EXPERT_FORWARD(EXPERT_Q8_0, 1, 1);
}
