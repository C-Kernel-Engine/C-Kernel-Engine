#include "ck_decoder_features.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int span(const void *p, size_t n, uintptr_t *lo, uintptr_t *hi) {
    if (!p || !n) return 0;
    *lo = (uintptr_t)p;
    if (n > UINTPTR_MAX - *lo) return 0;
    *hi = *lo + n;
    return 1;
}

int decoder_features_sequence_checked_f32(
    const void *opaque, size_t context_bytes,
    const int32_t *ids, size_t id_elements, size_t tokens, size_t channels,
    float *output, size_t output_elements,
    float *scratch, size_t scratch_elements) {
    if (!opaque || context_bytes != sizeof(CKDecoderFeatures) ||
        (uintptr_t)opaque % _Alignof(CKDecoderFeatures) ||
        !tokens || !channels || tokens > SIZE_MAX / channels)
        return -1;
    const size_t elements = tokens * channels;
    if (elements > SIZE_MAX / sizeof(float) || tokens > SIZE_MAX / sizeof(int32_t) ||
        id_elements < tokens || output_elements < elements || scratch_elements < elements)
        return -2;
    uintptr_t lo[5], hi[5];
    if (!span(opaque, sizeof(CKDecoderFeatures), &lo[0], &hi[0]) ||
        !span(ids, tokens * sizeof(int32_t), &lo[1], &hi[1]) ||
        !span(output, elements * sizeof(float), &lo[2], &hi[2]) ||
        !span(scratch, elements * sizeof(float), &lo[3], &hi[3]) ||
        (uintptr_t)ids % _Alignof(int32_t) ||
        (uintptr_t)output % _Alignof(float) || (uintptr_t)scratch % _Alignof(float))
        return -2;
    const CKDecoderFeatures *ctx = (const CKDecoderFeatures *)opaque;
    if (ctx->abi_version != 1 || ctx->channels != channels ||
        ctx->feature_elements < channels || !ctx->decode || !ctx->reset ||
        !ctx->active_tokens || !ctx->context_window || !ctx->vocabulary_size ||
        !span(ctx->features, channels * sizeof(float), &lo[4], &hi[4]) ||
        (uintptr_t)ctx->features % _Alignof(float))
        return -1;
    /* The live feature row is writable by decode, so it must not overlap IDs
       or descriptor storage either. Keep all five declared regions disjoint. */
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = i + 1; j < 5; ++j)
            if (lo[i] < hi[j] && lo[j] < hi[i]) return -2;
    const int capacity = ctx->context_window();
    const int vocabulary = ctx->vocabulary_size();
    if (capacity <= 0 || vocabulary <= 0 || tokens > (size_t)capacity) return -2;
    for (size_t row = 0; row < tokens; ++row)
        if (ids[row] < 0 || ids[row] >= vocabulary) return -1;
    ctx->reset();
    int status = ctx->active_tokens() == 0 ? 0 : -3;
    for (size_t row = 0; status == 0 && row < tokens; ++row) {
        if (ctx->should_cancel && ctx->should_cancel(ctx->cancel_context)) {
            status = -5;
            break;
        }
        if (ctx->decode(ids[row], NULL) != 0 ||
            ctx->active_tokens() != (int)(row + 1)) {
            status = -3;
            break;
        }
        for (size_t col = 0; col < channels; ++col) {
            const float value = ctx->features[col];
            if (!isfinite(value)) {
                status = -4;
                break;
            }
            scratch[row * channels + col] = value;
        }
    }
    ctx->reset();
    if (ctx->active_tokens() != 0) status = -3;
    if (!status && ctx->should_cancel && ctx->should_cancel(ctx->cancel_context)) status = -5;
    if (status) return status;
    memcpy(output, scratch, elements * sizeof(float));
    return 0;
}
