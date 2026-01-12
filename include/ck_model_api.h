/**
 * @file ck_model_api.h
 * @brief Generic Model API - Model-agnostic interface for CK-Engine
 *
 * All generated models emit the same function names, making the inference
 * engine completely model-agnostic. To use different models, compile
 * separate binaries.
 *
 * Usage:
 *   void *model = ck_model_create();
 *   ck_model_load_weights(model, "weights.bump");
 *   ck_model_forward(model, tokens, num_tokens);  // prefill
 *   ck_model_decode(model, &token, token_index);  // decode
 *   float *logits = ck_model_get_logits(model);
 *   ck_model_free(model);
 */

#ifndef CK_MODEL_API_H
#define CK_MODEL_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * MODEL CONFIGURATION (read-only, set by generated code)
 * ============================================================================ */

typedef struct {
    int embed_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int num_layers;
    int vocab_size;
    int max_seq_len;
    size_t total_bytes;
    size_t weight_bytes;
    size_t activation_bytes;
    const char *model_name;
    const char *model_family;  /* "qwen2", "llama", "mistral", etc. */
} CKModelConfig;

/* ============================================================================
 * GENERIC MODEL API - Same names for ALL models
 * ============================================================================ */

/**
 * Get model configuration (dimensions, sizes, etc.)
 * This is available before allocation.
 */
const CKModelConfig *ck_model_get_config(void);

/**
 * Create and allocate model memory.
 * Returns opaque model pointer, or NULL on failure.
 */
void *ck_model_create(void);

/**
 * Free model memory.
 */
void ck_model_free(void *model);

/**
 * Precompute RoPE cos/sin caches.
 * Call once after allocation, before inference.
 */
void ck_model_precompute_rope(void *model);

/**
 * Load weights from BUMP file into model.
 * Returns 0 on success, -1 on failure.
 */
int ck_model_load_weights(void *model, const char *bump_path);

/**
 * Forward pass (prefill) - process multiple tokens.
 * Used for initial prompt processing.
 */
void ck_model_forward(void *model, const int *tokens, int num_tokens);

/**
 * Decode single token at position token_index.
 * Used for autoregressive generation.
 */
void ck_model_decode(void *model, const int *token, int token_index);

/**
 * Get pointer to output logits buffer.
 * Size is vocab_size floats.
 */
float *ck_model_get_logits(void *model);

/**
 * Verify memory canaries (debug).
 * Returns number of corrupted canaries (0 = OK).
 */
int ck_model_verify_canaries(void *model);

/**
 * Get model base pointer (for weight loading).
 */
void *ck_model_get_base(void *model);

/**
 * Get total model size in bytes.
 */
size_t ck_model_get_total_bytes(void *model);

#ifdef __cplusplus
}
#endif

#endif /* CK_MODEL_API_H */
