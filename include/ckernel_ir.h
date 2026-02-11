#ifndef CKERNEL_IR_H
#define CKERNEL_IR_H

#include <stdint.h>
#include <stdio.h>

/**
 * Minimal HF-style model config extracted from config.json.
 *
 * Design note:
 *  - The long-term IR structure follows a "header / block / footer" pattern,
 *    inspired by the Antsand HMVC website generator:
 *      - header: one-time ops before the decoder stack (embeddings, pos-enc, etc.)
 *      - block : per-layer ops (RMSNorm, attention, MLP/SwiGLU, residual adds)
 *      - footer: one-time ops after the stack (final norm, LM head / weight-tying, loss)
 *  - In code this will eventually become a double-array layout (layers x ops)
 *    for the block section, plus separate header/footer op lists. The current
 *    CKIRGraph is the primitive building block for that higher-level IR.
 *
 * This header focuses on the basic decoder-only transformer parameters and
 * node-level IR that higher layers (codegen, memory planner, etc.) will build on.
 */
typedef struct {
    int num_layers;         // num_hidden_layers
    int hidden_size;        // hidden_size / d_model
    int intermediate_size;  // intermediate_size (MLP)
    int num_heads;          // num_attention_heads
    int num_kv_heads;       // num_key_value_heads (GQA)
    int vocab_size;         // vocab_size (if available)
    int context_window;     // max positions / context length (if available)
    float rms_norm_eps;     // RMSNorm epsilon (if available)
    float rope_theta;       // RoPE base (0 disables RoPE)
    int rotary_dim;         // RoPE rotary dimensions (0 = use head_dim)
    float rope_scaling_factor; // RoPE scaling factor (1.0 = no scaling)
    char rope_scaling_type[16]; // RoPE scaling type: "none", "linear", "dynamic", "yarn"
} CKModelConfig;

typedef enum {
    CK_OP_RMSNORM = 0,
    CK_OP_LINEAR_QKV,
    CK_OP_ATTENTION,
    CK_OP_ADD,
    CK_OP_LINEAR,
    CK_OP_SPLIT,
    CK_OP_SWIGLU,
    /* Backward ops (mirrors forward ops) */
    CK_OP_RMSNORM_BWD,
    CK_OP_LINEAR_QKV_BWD,
    CK_OP_ATTENTION_BWD,
    CK_OP_ADD_BWD,
    CK_OP_LINEAR_BWD,
    CK_OP_SPLIT_BWD,
    CK_OP_SWIGLU_BWD
} CKOpType;

typedef struct {
    uint16_t layer;  // decoder layer index
    uint16_t node;   // kernel index within that layer
} CKKernelId;

typedef struct {
    CKKernelId producer;  // which kernel produced this input
    uint8_t out_index;    // which output slot (0 for most kernels)
} CKInputRef;

typedef struct {
    CKKernelId id;
    CKOpType op;
    CKInputRef inputs[4];
    uint8_t n_inputs;
    uint8_t n_outputs;
    // For attention we may record heads; for now keep it simple.
} CKIRNode;

typedef struct {
    CKModelConfig config;
    int num_nodes;
    CKIRNode *nodes;  // heap-allocated array of length num_nodes
} CKIRGraph;

/**
 * Parse a HuggingFace-style config.json into CKModelConfig.
 *
 * This is a very small, dependency-free parser that looks for a handful
 * of integer keys. If a "text_config" object is present, it will parse
 * the fields from that object (so vision_config does not override text).
 *
 * Keys recognized (with common fallbacks):
 *   - "num_hidden_layers" ("n_layer")
 *   - "hidden_size" ("n_embd", "d_model")
 *   - "intermediate_size" ("n_inner", "ffn_dim", "mlp_dim")
 *   - "num_attention_heads" ("n_head", "num_heads")
 *   - "num_key_value_heads" ("num_kv_heads") (optional; defaults to num_attention_heads)
 *   - "vocab_size" ("n_vocab") (optional)
 *   - "max_position_embeddings" ("n_positions", "context_length", "seq_len") (optional)
 *   - "rms_norm_eps" ("layer_norm_eps") (optional; defaults to 1e-5)
 *   - "rope_theta" (optional; 0 disables RoPE)
 *
 * Returns 0 on success, non-zero on failure.
 */
int ck_model_config_from_hf_json(const char *path, CKModelConfig *cfg);

/**
 * Build a simple decoder-only IR graph for the given config.
 *
 * For now this constructs a canonical pattern per layer:
 *   RMSNorm -> QKV Linear -> Attention -> Add (residual)
 *   RMSNorm -> W1 Linear -> Split -> SwiGLU -> W2 Linear -> Add
 *
 * The IR is suitable as a starting point for later fusion and codegen.
 * Returns 0 on success, non-zero on failure.
 */
int ck_build_decoder_ir(const CKModelConfig *cfg, CKIRGraph *graph);

/**
 * Build a naive backward IR graph from a forward decoder IR.
 *
 * This reverses the node order and maps each forward op to a corresponding
 * *_BWD op type. This is an early skeleton useful for planning backprop
 * codegen; it does not yet encode full gradient wiring.
 */
int ck_build_decoder_backward_ir(const CKIRGraph *forward, CKIRGraph *backward);

/**
 * Free any heap-allocated memory owned by the graph.
 */
void ck_ir_free(CKIRGraph *graph);

/**
 * Dump a human-readable view of the IR to the given stream.
 */
void ck_ir_dump(const CKIRGraph *graph, FILE *out);

/**
 * Serialize a CKIRGraph to a simple JSON IR map file.
 *
 * This is an explicit per-node, per-layer view of the decoder stack. It is
 * intentionally low-level and mirrors ck_ir_dump(), but in a machine-readable
 * format. Higher-level tooling can later reorganize this into header/block/
 * footer sections.
 *
 * Returns 0 on success, non-zero on failure.
 */
int ck_ir_serialize_json(const CKIRGraph *graph, const char *path);

/**
 * Parse a JSON IR map file (as produced by ck_ir_serialize_json) back into
 * a CKIRGraph. This enables a two-stage pipeline:
 *
 *   config.json -> CKIRGraph -> ir.json
 *   ir.json     -> CKIRGraph -> codegen / analysis
 *
 * Returns 0 on success, non-zero on failure.
 */
int ck_ir_parse_json(const char *path, CKIRGraph *graph);

#endif /* CKERNEL_IR_H */
