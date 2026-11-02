#ifndef CK_MODEL_ABI_V8_H
#define CK_MODEL_ABI_V8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CK_MODEL_ABI_V8_VERSION UINT32_C(1)

enum CKModelCapabilityV8 {
    CK_MODEL_CAP_INIT                    = UINT64_C(1) << 0,
    CK_MODEL_CAP_AUTOREGRESSIVE_DECODE   = UINT64_C(1) << 1,
    CK_MODEL_CAP_TEXT_ENCODE             = UINT64_C(1) << 2,
    CK_MODEL_CAP_TOKEN_DECODE            = UINT64_C(1) << 3,
    CK_MODEL_CAP_CHAT_FORMAT             = UINT64_C(1) << 4,
    CK_MODEL_CAP_STOP_TOKENS             = UINT64_C(1) << 5,
    CK_MODEL_CAP_MIXED_EMBEDDING_PREFILL = UINT64_C(1) << 6,
    CK_MODEL_CAP_AUDIO_WAV_ENCODER       = UINT64_C(1) << 7,
    CK_MODEL_CAP_IMAGE_TENSOR_ENCODER    = UINT64_C(1) << 8,
    CK_MODEL_CAP_RAW_IMAGE_ENCODER       = UINT64_C(1) << 9,
    CK_MODEL_CAP_ENCODER_OUTPUT          = UINT64_C(1) << 10,
    CK_MODEL_CAP_ENCODER_MEMORY          = UINT64_C(1) << 11,
    CK_MODEL_CAP_NAMED_ACTIVATIONS       = UINT64_C(1) << 12,
    CK_MODEL_CAP_PROFILE                 = UINT64_C(1) << 13,
    CK_MODEL_CAP_XRAY_KV                 = UINT64_C(1) << 14,
    CK_MODEL_CAP_GENERATION_POLICY       = UINT64_C(1) << 15,
    /* Serialized state switching only; this does not imply batched arithmetic. */
    CK_MODEL_CAP_SEQUENCE_STATE_SWITCH   = UINT64_C(1) << 16,
    /* Generated M=2 projection work with sequence-local attention/KV. */
    CK_MODEL_CAP_BATCH_DECODE_TWO_ROWS   = UINT64_C(1) << 17,
    /* Process-local state save/restore for a checked sequence-state layout. */
    CK_MODEL_CAP_SEQUENCE_CHECKPOINT     = UINT64_C(1) << 18
};

#define CK_MODEL_CAP_V8_KNOWN_MASK ((UINT64_C(1) << 19) - UINT64_C(1))

enum CKGenerationFlagsV8 {
    CK_GENERATION_FLAG_TIMESTAMPS = 1u << 0
};

enum CKModelArtifactRoleV8 {
    CK_MODEL_ROLE_UNKNOWN = 0,
    CK_MODEL_ROLE_DECODER = 1,
    CK_MODEL_ROLE_ENCODER = 2,
    CK_MODEL_ROLE_COMBINED = 3
};

/* Fixed-layout descriptor. New fields consume reserved slots and require an
 * ABI version bump when their interpretation changes. Hosts must check both
 * struct_size and abi_version before using it. */
typedef struct CKModelRuntimeDescriptorV8 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t capabilities;
    uint32_t artifact_role;
    uint32_t reserved0;
    int32_t context_length;
    int32_t vocab_size;
    int32_t encoder_memory_tokens;
    int32_t encoder_memory_dim;
    int32_t primary_input_tokens;
    int32_t primary_input_dim;
    uint64_t reserved[8];
} CKModelRuntimeDescriptorV8;

typedef uint32_t (*ck_model_get_abi_version_v8_fn)(void);
typedef uint64_t (*ck_model_get_capabilities_v8_fn)(void);
typedef int (*ck_model_get_runtime_descriptor_v8_fn)(
    CKModelRuntimeDescriptorV8 *descriptor,
    size_t descriptor_size);

/* Optional when CK_MODEL_CAP_SEQUENCE_STATE_SWITCH is set. All calls, including
 * decode and reset, remain serialized. Nonzero handles are generation-tagged
 * IDs; retired handles and handles from an earlier model load are rejected.
 * The default state uses the model arena. Each extra sequence needs a caller-
 * owned aligned KV arena of the reported size; it must remain alive through
 * destroy or model_free. Admission is bounded to two extra sequences. This
 * ABI provides isolation, not simultaneous or computationally batched decode.
 */
typedef int (*ck_model_sequence_state_requirements_v8_fn)(size_t *bytes, size_t *alignment);
typedef uint64_t (*ck_model_sequence_state_default_v8_fn)(void);
typedef int (*ck_model_sequence_state_create_v8_fn)(void *arena, size_t arena_bytes,
                                                    uint64_t *handle_out);
typedef int (*ck_model_sequence_state_activate_v8_fn)(uint64_t handle);
typedef int (*ck_model_sequence_state_destroy_v8_fn)(uint64_t handle);

/* Optional with CK_MODEL_CAP_SEQUENCE_CHECKPOINT. These operations are
 * serialized with all model execution and used only at a successful forward
 * boundary; a failed or cancelled forward requires reset. The caller owns aligned snapshot
 * storage and must bind its exact token prefix and loaded artifact identity;
 * generated snapshots validate a fresh host-issued 128-bit instance nonce and state
 * layout. They are not a portable disk or cross-process format. Invalid or
 * corrupt snapshots must leave the destination sequence unchanged. Logits are
 * not saved; a caller must execute a new token before sampling from restored
 * state, or separately retain validated logits for an exact-prefix hit. */
typedef int (*ck_model_sequence_checkpoint_bind_instance_v8_fn)(uint64_t nonce_hi,
                                                               uint64_t nonce_lo);
typedef int (*ck_model_sequence_checkpoint_requirements_v8_fn)(size_t *bytes,
                                                               size_t *alignment);
typedef int (*ck_model_sequence_checkpoint_save_v8_fn)(uint64_t handle,
                                                       void *out, size_t out_bytes);
typedef int (*ck_model_sequence_checkpoint_restore_v8_fn)(uint64_t handle,
                                                          const void *snapshot,
                                                          size_t snapshot_bytes);

/* Optional when CK_MODEL_CAP_BATCH_DECODE_TWO_ROWS is set. The first version
 * accepts two distinct sequence handles with one token each. row_offset and
 * token_count reserve explicit packed-row geometry for future mixed steps;
 * unsupported values fail before either sequence advances. Caller owns
 * distinct output vectors and one 64-byte-aligned workspace for the call.
 * Calls remain serialized with all other model operations. */
typedef struct CKModelBatchDecodeRowV8 {
    uint64_t sequence_handle;
    int32_t token;
    int32_t position;
    uint32_t row_offset;
    uint32_t token_count;
    float *logits;
} CKModelBatchDecodeRowV8;

typedef int (*ck_model_batch_decode_workspace_v8_fn)(size_t *bytes, size_t *alignment);
/* Optional diagnostics on generated batch entries; absent on older bundles. */
typedef int (*ck_model_batch_decode_projection_groups_v8_fn)(void);
typedef int (*ck_model_batch_decode_shared_layers_v8_fn)(void);
typedef int (*ck_model_decode_batch2_v8_fn)(const CKModelBatchDecodeRowV8 *rows,
                                             size_t count, void *workspace,
                                             size_t workspace_bytes);
/* Optional phase-aware variant. The selected prefill row uses map-resolved
 * prefill continuation arithmetic in local stages; shared projections retain
 * the same checked two-row provider. Unsupported graphs omit this symbol. */
typedef int (*ck_model_decode_batch2_mixed_prefill_v8_fn)(
    const CKModelBatchDecodeRowV8 *rows, size_t count, void *workspace,
    size_t workspace_bytes, uint32_t prefill_row);
/* Optional standalone staged append. The selected map-resolved continuation
 * phase must be paired with a row-group value of one before external callers
 * use it for arbitrary token chunk boundaries. */
typedef int (*ck_model_decode_prefill_phase_v8_fn)(int32_t token, float *logits);
typedef int (*ck_model_prefill_append_row_group_v8_fn)(void);

#ifdef __cplusplus
}
#endif

#endif
