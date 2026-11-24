#include "checked_float_buffer.h"

#if defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
#include "checked_parallel.h"
#define CK_DEPTH_OPTIONAL_POOL 1
#pragma weak ck_get_num_threads
#pragma weak ck_threadpool_global
#pragma weak ck_threadpool_in_dispatch
#pragma weak ck_threadpool_n_threads
#pragma weak ck_threadpool_parallel_for_n
#endif

static double rms_scale(const float *x, size_t width, double epsilon)
{
    double sum = 0.0;
    for (size_t c = 0; c < width; ++c) sum += (double)x[c] * x[c];
    return 1.0 / sqrt(sum / (double)width + epsilon);
}

struct depth_mix_rows {
    const float *bank, *current, *query;
    float *output;
    size_t states, width;
    float epsilon;
};

static void mix_depth_rows(size_t begin, size_t end, void *opaque)
{
    const struct depth_mix_rows *work = opaque;
    const float *bank = work->bank, *current = work->current, *query = work->query;
    float *output = work->output;
    const size_t states = work->states, width = work->width;
    const float epsilon = work->epsilon;
    const double temperature = 1.0 / sqrt((double)width);
    for (size_t r = begin; r < end; ++r) {
        double scale[65], logits[65], max_logit = -INFINITY;
        for (size_t s = 0; s <= states; ++s) {
            const float *x = s == states ? current + r * width : bank + (r * states + s) * width;
            scale[s] = rms_scale(x, width, epsilon);
            double dot = 0.0;
            for (size_t c = 0; c < width; ++c) {
                const float normalized = (float)(x[c] * scale[s]);
                dot += (double)query[c] * normalized;
            }
            logits[s] = dot * temperature;
            max_logit = fmax(max_logit, logits[s]);
        }
        double denominator = 0.0;
        for (size_t s = 0; s <= states; ++s) {
            logits[s] = exp(logits[s] - max_logit);
            denominator += logits[s];
        }
        for (size_t c = 0; c < width; ++c) {
            double sum = 0.0;
            for (size_t s = 0; s <= states; ++s) {
                const float *x = s == states ? current + r * width : bank + (r * states + s) * width;
                const float normalized = (float)(x[c] * scale[s]);
                sum += logits[s] * normalized;
            }
            output[r * width + c] = (float)(sum / denominator);
        }
    }
}

int depth_residual_mix_checked_f32(const float *bank, size_t bank_elements,
                                    const float *current, size_t current_elements,
                                    const float *query, size_t query_elements,
                                    float *output, size_t output_elements,
                                    size_t rows, size_t states, size_t width, float epsilon)
{
    size_t present, prior;
    if (!states || states > 64 || !isfinite(epsilon) || epsilon <= 0.0f ||
        !ck_checked_span(rows, width, width, &present) ||
        present > SIZE_MAX / states ||
        (prior = present * states) > SIZE_MAX / sizeof(float) ||
        bank_elements < prior || current_elements < present ||
        output_elements < present || query_elements < width) return -2;
    if (!ck_checked_region(bank, prior) || !ck_checked_region(current, present) ||
        !ck_checked_region(query, width) || !ck_checked_region(output, present) ||
        ck_checked_overlap(output, present, bank, prior) ||
        ck_checked_overlap(output, present, current, present) ||
        ck_checked_overlap(output, present, query, width)) return -1;
    if (!ck_checked_finite(bank, prior) || !ck_checked_finite(current, present) ||
        !ck_checked_finite(query, width)) return -3;

    struct depth_mix_rows work = {
        .bank = bank, .current = current, .query = query, .output = output,
        .states = states, .width = width, .epsilon = epsilon,
    };
#ifdef CK_DEPTH_OPTIONAL_POOL
    if (rows >= 16 && width >= 128 && ck_get_num_threads && ck_threadpool_global &&
        ck_threadpool_in_dispatch && ck_threadpool_n_threads && ck_threadpool_parallel_for_n)
        ck_checked_parallel_for(rows, 4, mix_depth_rows, &work);
    else
#endif
        mix_depth_rows(0, rows, &work);
    return 0;
}
