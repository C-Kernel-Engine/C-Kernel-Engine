#ifndef CK_DECODER_FEATURES_H
#define CK_DECODER_FEATURES_H
#include <stddef.h>
#include <stdint.h>

/* ABI 1 describes an exclusively owned, already initialized generated decoder.
 * The callbacks have the ordinary generated decoder ABI; they schedule no layers
 * in the caller. Feature storage must remain valid until this call returns.
 * This descriptor is runtime state, never a weight or checkpoint tensor. */
typedef struct CKDecoderFeatures {
    uint32_t abi_version;
    size_t channels;
    const float *features;
    size_t feature_elements;
    int (*decode)(int32_t token, float *logits);
    void (*reset)(void);
    int (*active_tokens)(void);
    int (*context_window)(void);
    int (*vocabulary_size)(void);
    int (*should_cancel)(void *context); /* Optional; independent of decoder reset. */
    void *cancel_context;
} CKDecoderFeatures;

/* IDs and all buffers are caller-owned. Rejection before execution changes
 * neither decoder state nor output. Execution failures attempt decoder reset
 * and leave output untouched; scratch may change. A failed reset invalidates
 * the decoder until its owner reinitializes it. Only valid rows are published.
 * The caller must exclude concurrent access to this decoder instance. */
int decoder_features_sequence_checked_f32(
    const void *context, size_t context_bytes,
    const int32_t *ids, size_t id_elements, size_t tokens, size_t channels,
    float *output, size_t output_elements,
    float *scratch, size_t scratch_elements);
#endif
