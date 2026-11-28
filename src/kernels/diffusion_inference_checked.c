#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include "bf16_utils.h"
#include "diffusion_vp_common.h"

typedef struct {
    const void *ptr;
    size_t elements;
    size_t width;
} tensor_span;

static int valid_span(tensor_span s)
{
    return s.ptr && (uintptr_t)s.ptr % s.width == 0 &&
           s.elements <= (UINTPTR_MAX - (uintptr_t)s.ptr) / s.width;
}

static int overlap(tensor_span a, tensor_span b)
{
    return (uintptr_t)a.ptr < (uintptr_t)b.ptr + b.elements * b.width &&
           (uintptr_t)b.ptr < (uintptr_t)a.ptr + a.elements * a.width;
}

static int checked_spans(const tensor_span *spans, size_t reads, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (!valid_span(spans[i])) return 0;
        if (i >= reads)
            for (size_t j = 0; j < i; ++j)
                if (overlap(spans[i], spans[j])) return 0;
    }
    return 1;
}

int ck_diffusion_vp_condition_inputs_checked(
    const double *latent, const double *gamma, const float *self_condition,
    const float *frequencies, float *normalized, float *prior_input,
    float *scaled_self, float *time_features, size_t rows, size_t width,
    size_t bands, size_t elements, size_t time_elements, size_t gamma_elements)
{
    if (!rows || !width || !bands || bands > SIZE_MAX / 2 || rows > SIZE_MAX / width ||
        rows > SIZE_MAX / (2 * bands) || elements != rows * width ||
        time_elements != rows * 2 * bands || gamma_elements != 2) return -1;
    const tensor_span spans[] = {{latent,elements,8}, {gamma,2,8}, {self_condition,elements,4},
        {frequencies,bands,4}, {normalized,elements,4}, {prior_input,elements,4},
        {scaled_self,elements,4}, {time_features,time_elements,4}};
    if (!checked_spans(spans, 4, sizeof spans / sizeof spans[0]) ||
        !isfinite(gamma[0]) || !isfinite(gamma[1]) || gamma[1] > gamma[0]) return -1;
    const double alpha_sq = vp_sigmoid(-gamma[0]), sigma_sq = vp_sigmoid(gamma[0]);
    const double denominator = sqrt(alpha_sq / (double)width + sigma_sq);
    const double ratio = sqrt(alpha_sq) / sigma_sq;
    if (!isfinite(ratio) || ratio > FLT_MAX || fabs(gamma[0]) > FLT_MAX || !(denominator > 0.0))
        return -1;
    const float prior_gain = (float)ratio;
    const float self_gain = sqrtf((float)width);
    for (size_t i = 0; i < elements; ++i) {
        if (!isfinite(latent[i]) || fabs(latent[i]) > FLT_MAX ||
            fabs(latent[i] / denominator) > FLT_MAX ||
            !isfinite((float)latent[i] * prior_gain) || !isfinite(self_condition[i] * self_gain))
            return -1;
    }
    for (size_t i = 0; i < bands; ++i)
        if (!isfinite((float)gamma[0] * frequencies[i])) return -1;

    for (size_t i = 0; i < elements; ++i) {
        normalized[i] = (float)(latent[i] / denominator);
        prior_input[i] = (float)latent[i] * prior_gain;
        scaled_self[i] = self_condition[i] * self_gain;
    }
    for (size_t i = 0; i < bands; ++i) {
        const float angle = (float)gamma[0] * frequencies[i];
        const float sine = sinf(angle), cosine = cosf(angle);
        for (size_t r = 0; r < rows; ++r) {
            time_features[r * 2 * bands + i] = sine;
            time_features[r * 2 * bands + bands + i] = cosine;
        }
    }
    return 0;
}

int ck_diffusion_embedding_posterior_checked(
    const uint16_t *model_scores, const float *prior_scores, const float *embedding,
    float *logits, float *mean, int32_t *ids, float *probabilities,
    size_t rows, size_t vocabulary, size_t width, size_t scores_elements,
    size_t embedding_elements, size_t mean_elements)
{
    if (!rows || !vocabulary || !width || vocabulary > INT32_MAX ||
        rows > SIZE_MAX / vocabulary || vocabulary > SIZE_MAX / width || rows > SIZE_MAX / width ||
        scores_elements != rows * vocabulary || embedding_elements != vocabulary * width ||
        mean_elements != rows * width) return -1;
    const tensor_span spans[] = {{model_scores,scores_elements,2}, {prior_scores,scores_elements,4},
        {embedding,embedding_elements,4}, {logits,scores_elements,4}, {mean,mean_elements,4},
        {ids,rows,4}, {probabilities,vocabulary,4}};
    if (!checked_spans(spans, 3, sizeof spans / sizeof spans[0])) return -1;
    for (size_t i = 0; i < embedding_elements; ++i) {
        /* The table contains prepared unit-vector coordinates. */
        if (!isfinite(embedding[i]) || fabsf(embedding[i]) > 1.0f) return -1;
    }
    for (size_t i = 0; i < scores_elements; ++i)
        if (!isfinite(bf16_to_float(model_scores[i]) + prior_scores[i])) return -1;

    for (size_t r = 0; r < rows; ++r) {
        const size_t start = r * vocabulary;
        float largest = -INFINITY;
        size_t best = 0;
        for (size_t i = 0; i < vocabulary; ++i) {
            const float value = bf16_to_float(model_scores[start + i]) + prior_scores[start + i];
            logits[start + i] = value;
            if (value > largest) {
                largest = value;
                best = i;
            }
        }
        double denominator = 0.0;
        for (size_t i = 0; i < vocabulary; ++i) {
            probabilities[i] = expf(logits[start + i] - largest);
            denominator += probabilities[i];
        }
        for (size_t i = 0; i < vocabulary; ++i)
            probabilities[i] = (float)(probabilities[i] / denominator);
        for (size_t d = 0; d < width; ++d) {
            double sum = 0.0;
            for (size_t i = 0; i < vocabulary; ++i) {
                /* Product of two FP32 values is exact in FP64. */
                sum += (double)probabilities[i] * embedding[i * width + d];
            }
            mean[r * width + d] = (float)sum;
        }
        ids[r] = (int32_t)best;
    }
    return 0;
}

static float guidance_value(float conditional, float unconditional, float scale)
{
    if (scale == 1.0f) return conditional;
    const volatile float delta = conditional - unconditional;
    const volatile float weighted = scale * delta;
    return unconditional + weighted;
}

int ck_diffusion_guidance_checked(const float *conditional, const float *unconditional,
                                  const float *scale, double *output, size_t elements,
                                  size_t scale_elements)
{
    if (!elements || scale_elements != 1) return -1;
    const tensor_span spans[] = {{conditional,elements,4}, {unconditional,elements,4},
        {scale,1,4}, {output,elements,8}};
    if (!checked_spans(spans, 3, sizeof spans / sizeof spans[0]) || !isfinite(*scale)) return -1;
    for (size_t i = 0; i < elements; ++i)
        if (!isfinite(conditional[i]) || !isfinite(unconditional[i]) ||
            !isfinite(guidance_value(conditional[i], unconditional[i], *scale))) return -1;
    for (size_t i = 0; i < elements; ++i)
        output[i] = guidance_value(conditional[i], unconditional[i], *scale);
    return 0;
}
