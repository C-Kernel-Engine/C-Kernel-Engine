#include "checked_float_buffer.h"
#include "checked_parallel.h"
#include "checked_q4_k_activation.h"
#include "packed_k_dot_ordered.h"

#include <stdatomic.h>

#define LINEAR_ARGS \
    const float *input, size_t input_elements, const void *weight, size_t weight_bytes, \
    float *output, size_t output_elements, size_t rows, size_t width, size_t channels

struct packed_linear_work {
    const float *input;
    const unsigned char *weights;
    float *output;
    size_t width, channels, row_bytes;
    float (*dot)(const float *, const unsigned char *, size_t);
    atomic_int failed;
    int publish;
};

static void packed_linear_outputs(size_t begin, size_t end, void *opaque)
{
    struct packed_linear_work *work = opaque;
    size_t row = begin / work->channels;
    size_t channel = begin % work->channels;
    for (size_t i = begin; i < end; ++i) {
        const unsigned char *packed = work->weights + channel * work->row_bytes;
        const float value = work->dot(work->input + row * work->width, packed, work->width);
        if (!isfinite(value)) {
            atomic_store_explicit(&work->failed, 1, memory_order_relaxed);
            return;
        }
        if (work->publish) work->output[i] = value;
        if (++channel == work->channels) {
            channel = 0;
            ++row;
        }
    }
}

static int linear_rows_packed_k_checked(LINEAR_ARGS, size_t block_bytes,
                                        float (*dot)(const float *, const unsigned char *, size_t))
{
    size_t in, out;
    if (!width || width % QK_K || width / QK_K > SIZE_MAX / block_bytes ||
        !ck_checked_span(rows, width, width, &in) ||
        !ck_checked_span(rows, channels, channels, &out) ||
        input_elements < in || output_elements < out) return -2;
    const size_t row_bytes = width / QK_K * block_bytes;
    if (channels > SIZE_MAX / row_bytes || weight_bytes != channels * row_bytes) return -2;
    if (!weight || (uintptr_t)weight > UINTPTR_MAX - weight_bytes ||
        !ck_checked_region(input, in) || !ck_checked_region(output, out) ||
        ck_checked_overlap(input, in, output, out)) return -1;
    const uintptr_t dst = (uintptr_t)output, weights = (uintptr_t)weight;
    if (dst < weights + weight_bytes && weights < dst + out * sizeof(float)) return -1;
    if (!ck_checked_finite(input, in)) return -3;
    struct packed_linear_work work = {
        .input = input, .weights = weight, .output = output, .width = width,
        .channels = channels, .row_bytes = row_bytes, .dot = dot,
    };
    atomic_init(&work.failed, 0);
    for (work.publish = 0; work.publish < 2; ++work.publish) {
        ck_checked_parallel_for(out, 32, packed_linear_outputs, &work);
        if (atomic_load_explicit(&work.failed, memory_order_relaxed)) return -3;
    }
    return 0;
}

int linear_rows_q4_k_checked_f32(LINEAR_ARGS)
{
    return linear_rows_packed_k_checked(input, input_elements, weight, weight_bytes, output,
                                        output_elements, rows, width, channels, sizeof(block_q4_K),
                                        ck_packed_q4_k_dot_ordered);
}

int linear_rows_q6_k_checked_f32(LINEAR_ARGS)
{
    return linear_rows_packed_k_checked(input, input_elements, weight, weight_bytes, output,
                                        output_elements, rows, width, channels, sizeof(block_q6_K),
                                        ck_packed_q6_k_dot_ordered);
}

void gemv_q6_k_q8_k_checked_cpu(float *, const void *, const void *, int, int);

typedef void (*packed_matvec_fn)(float *, const void *, const void *, int, int);
struct q8_k_linear_work {
    const unsigned char *weights;
    const block_q8_K *activations;
    float *staged;
    size_t width, channels, weight_row_bytes;
    packed_matvec_fn matvec;
};

static void packed_k_matvec_ranges(size_t begin, size_t end, void *opaque)
{
    const struct q8_k_linear_work *work = opaque;
    while (begin < end) {
        const size_t row = begin / work->channels, channel = begin % work->channels;
        const size_t left = work->channels - channel, remaining = end - begin;
        const size_t count = left < remaining ? left : remaining;
        work->matvec(work->staged + begin, work->weights + channel * work->weight_row_bytes,
                            work->activations + row * (work->width / QK_K),
                            (int)count, (int)work->width);
        begin += count;
    }
}

static int linear_rows_packed_k_q8_k_checked(
    LINEAR_ARGS, float *work, size_t work_elements, size_t block_bytes, size_t weight_alignment,
    packed_matvec_fn matvec)
{
    size_t in, out;
    if (width > INT_MAX || channels > INT_MAX || !width || width % QK_K ||
        !ck_checked_span(rows, width, width, &in) ||
        !ck_checked_span(rows, channels, channels, &out) ||
        input_elements < in || output_elements < out) return -2;
    const size_t blocks = width / QK_K;
    const size_t row_bytes = blocks * block_bytes;
    if (channels > SIZE_MAX / row_bytes || weight_bytes != channels * row_bytes ||
        rows > SIZE_MAX / sizeof(block_q8_K) / blocks) return -2;
    const size_t packed_bytes = rows * blocks * sizeof(block_q8_K);
    const size_t packed_elements = packed_bytes / sizeof(float);
    if (packed_elements > work_elements || out > work_elements - packed_elements) return -2;
    const size_t needed = packed_elements + out;
    if (!weight || (uintptr_t)weight % weight_alignment ||
        (uintptr_t)weight > UINTPTR_MAX - weight_bytes || !ck_checked_region(input, in) ||
        !ck_checked_region(output, out) || !ck_checked_region(work, needed) ||
        ck_checked_overlap(input, in, output, out) || ck_checked_overlap(work, needed, input, in) ||
        ck_checked_overlap(work, needed, output, out)) return -1;
    const uintptr_t dst = (uintptr_t)output, weights = (uintptr_t)weight, scratch = (uintptr_t)work;
    if ((dst < weights + weight_bytes && weights < dst + out * sizeof(float)) ||
        (scratch < weights + weight_bytes && weights < scratch + needed * sizeof(float))) return -1;
    if (!ck_checked_finite(input, in)) return -3;
    unsigned char *packed = (unsigned char *)work;
    for (size_t row = 0; row < rows; ++row) {
        if (ck_pack_q4_input_q8_k(input + row * width, packed + row * blocks * sizeof(block_q8_K), width))
            return -3;
    }
    struct q8_k_linear_work task = {
        .weights = weight, .activations = (const block_q8_K *)work, .staged = work + packed_elements,
        .width = width, .channels = channels, .weight_row_bytes = row_bytes, .matvec = matvec,
    };
    ck_checked_parallel_for(out, 128, packed_k_matvec_ranges, &task);
    if (!ck_checked_finite(task.staged, out)) return -3;
    memcpy(output, task.staged, out * sizeof(float));
    return 0;
}

int linear_rows_q4_k_q8_k_checked_f32(LINEAR_ARGS, float *work, size_t work_elements)
{
    return linear_rows_packed_k_q8_k_checked(input, input_elements, weight, weight_bytes, output,
        output_elements, rows, width, channels, work, work_elements, sizeof(block_q4_K),
        _Alignof(block_q4_K), gemv_q4_k_q8_k_avx2);
}

int linear_rows_q6_k_q8_k_checked_f32(LINEAR_ARGS, float *work, size_t work_elements)
{
    return linear_rows_packed_k_q8_k_checked(input, input_elements, weight, weight_bytes, output,
        output_elements, rows, width, channels, work, work_elements, sizeof(block_q6_K),
        _Alignof(block_q6_K), gemv_q6_k_q8_k_checked_cpu);
}
