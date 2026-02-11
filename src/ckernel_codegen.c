/**
 * @file ckernel_codegen.c
 *
 * ===========================================================================
 * LEGACY CODE - NOT USED IN v6.6
 * ===========================================================================
 *
 * This file contains the older C-side kernel/codegen orchestration path.
 * It is kept for reference and potential fallback use but is not part of the
 * v6.6 generated-model inference flow.
 *
 * v6.6 Architecture:
 *   - IR Lower 1/2/3 drives execution planning
 *   - version/v6.6/scripts/codegen_v6_6.py emits model_v6_6.c directly
 *   - Kernel dispatch is selected from kernel_maps + lowered call IR
 *
 * To remove completely:
 *   1. Delete this file
 *   2. Remove ckernel_codegen.* references from legacy build paths
 *
 * Last used: v6.5
 * Deprecated: v6.6 (2026-02)
 * ===========================================================================
 */

#include "ckernel_codegen.h"
#include "ckernel_registry.h"
#include "ckernel_kernel_specs.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *op_name(CKOpType op)
{
    switch (op) {
    case CK_OP_RMSNORM:        return "RMSNORM";
    case CK_OP_LINEAR_QKV:     return "LINEAR_QKV";
    case CK_OP_ATTENTION:      return "ATTENTION";
    case CK_OP_ADD:            return "ADD";
    case CK_OP_LINEAR:         return "LINEAR";
    case CK_OP_SPLIT:          return "SPLIT";
    case CK_OP_SWIGLU:         return "SWIGLU";
    case CK_OP_RMSNORM_BWD:    return "RMSNORM_BWD";
    case CK_OP_LINEAR_QKV_BWD: return "LINEAR_QKV_BWD";
    case CK_OP_ATTENTION_BWD:  return "ATTENTION_BWD";
    case CK_OP_ADD_BWD:        return "ADD_BWD";
    case CK_OP_LINEAR_BWD:     return "LINEAR_BWD";
    case CK_OP_SPLIT_BWD:      return "SPLIT_BWD";
    case CK_OP_SWIGLU_BWD:     return "SWIGLU_BWD";
    default:                   return "UNKNOWN";
    }
}

static const CKBufferSpec *ck_find_buffer_spec(const char *name)
{
    if (!name) {
        return NULL;
    }
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        if (strcmp(ck_decoder_buffers[i].name, name) == 0) {
            return &ck_decoder_buffers[i];
        }
    }
    return NULL;
}

static const CKKernelSpec *ck_find_kernel_spec(const char *name)
{
    if (!name) {
        return NULL;
    }
    for (size_t i = 0; i < ck_kernel_spec_count; ++i) {
        if (strcmp(ck_kernel_specs[i].name, name) == 0) {
            return &ck_kernel_specs[i];
        }
    }
    return NULL;
}

static int ck_buffer_should_alloc(const CKBufferSpec *spec)
{
    return spec && spec->role != CK_ROLE_INPUT;
}

static int ck_buffer_uses_weight_dtype(const CKBufferSpec *spec)
{
    if (!spec || spec->role != CK_ROLE_WEIGHT || !spec->name) {
        return 0;
    }

    /* Weight-only quantization targets: big GEMM weights and tied embeddings.
     * Small vectors (norm scales, biases) remain fp32 for simplicity. */
    return (strcmp(spec->name, "token_emb") == 0) ||
           (strcmp(spec->name, "wq") == 0) ||
           (strcmp(spec->name, "wk") == 0) ||
           (strcmp(spec->name, "wv") == 0) ||
           (strcmp(spec->name, "wo") == 0) ||
           (strcmp(spec->name, "w1") == 0) ||
           (strcmp(spec->name, "w2") == 0) ||
           (strcmp(spec->name, "lm_head_weight") == 0);
}

static const char *ck_weight_dtype_expr(const CKBufferSpec *spec)
{
    if (!spec || spec->role != CK_ROLE_WEIGHT || !spec->name) {
        return NULL;
    }
    if (spec->scope == CK_SCOPE_GLOBAL) {
        if (strcmp(spec->name, "token_emb") == 0) {
            return "m->token_emb_dtype";
        }
        if (strcmp(spec->name, "lm_head_weight") == 0) {
            return "m->lm_head_weight_dtype";
        }
        return NULL;
    }
    if (spec->scope == CK_SCOPE_LAYER) {
        if (strcmp(spec->name, "wq") == 0) return "L->wq_dtype";
        if (strcmp(spec->name, "wk") == 0) return "L->wk_dtype";
        if (strcmp(spec->name, "wv") == 0) return "L->wv_dtype";
        if (strcmp(spec->name, "wo") == 0) return "L->wo_dtype";
        if (strcmp(spec->name, "w1") == 0) return "L->w1_dtype";
        if (strcmp(spec->name, "w2") == 0) return "L->w2_dtype";
        return NULL;
    }
    return NULL;
}

static void emit_offset_field(FILE *out, const char *name)
{
    fprintf(out, "    size_t %s_offset;\n", name);
}

static void emit_layer_offsets_struct(FILE *out)
{
    fprintf(out, "typedef struct {\n");
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->scope != CK_SCOPE_LAYER) {
            continue;
        }
        if (!ck_buffer_should_alloc(spec)) {
            continue;
        }
        emit_offset_field(out, spec->name);
    }
    fprintf(out,
            "    CKDataType wq_dtype;\n"
            "    CKDataType wk_dtype;\n"
            "    CKDataType wv_dtype;\n"
            "    CKDataType wo_dtype;\n"
            "    CKDataType w1_dtype;\n"
            "    CKDataType w2_dtype;\n");
    fprintf(out, "} LayerOffsets;\n\n");
}

static void emit_global_offset_fields(FILE *out)
{
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        if (!ck_buffer_should_alloc(spec)) {
            continue;
        }
        emit_offset_field(out, spec->name);
    }
}

static void emit_model_struct(FILE *out)
{
    fprintf(out,
            "typedef LayerOffsets TrulyOptimalLayer;\n\n"
            "typedef struct {\n"
            "    char magic[8];\n"
            "    uint32_t version;\n"
            "    uint32_t model_type;\n"
            "\n"
            "    int num_layers;\n"
            "    int vocab_size;\n"
            "    int embed_dim;\n"
            "    int context_window;\n"
            "    int intermediate_size;\n"
            "\n"
            "    size_t aligned_embed_dim;\n"
            "    size_t aligned_head_dim;\n"
            "    size_t aligned_attn_context_window;\n"
            "\n"
            "    int num_cores;\n"
            "    int tokens_per_core;\n"
            "    int num_attention_heads;\n"
            "    int num_kv_heads;\n"
            "    int head_dim;\n"
            "    float rms_norm_eps;\n"
            "    float rope_theta;\n"
            "\n"
            "    uint8_t *memory_base;\n"
            "    size_t total_bytes;\n"
            "    size_t elem_bytes;\n"
            "    CKDataType weight_dtype;\n"
            "    CKDataType token_emb_dtype;\n"
            "    CKDataType pos_emb_dtype;\n"
            "    CKDataType lm_head_weight_dtype;\n"
            "    bool weights_mixed;\n"
            "    bool weights_quantized;\n"
            "    size_t layer_stride;\n"
            "\n"
            "    size_t layers_start_offset;\n");

    emit_global_offset_fields(out);

    fprintf(out,
            "\n"
            "    TrulyOptimalLayer *layers;\n"
            "\n"
            "    GradientStorage gradients;\n"
            "    bool training_enabled;\n"
            "    float learning_rate;\n"
            "    int lr_warmup_steps;\n"
            "    float lr_warmup_init;\n"
            "    float grad_clip;\n"
            "    size_t training_cache_samples;\n"
            "    int active_tokens;\n"
            "    TaskType task_type;\n"
            "    OptimizerType optimizer;\n"
            "    uint64_t optimizer_step;\n"
            "    float adam_beta1;\n"
            "    float adam_beta2;\n"
            "    float adam_eps;\n"
            "    float weight_decay;\n"
            "    bool ema_enabled;\n"
            "    float ema_decay;\n"
            "    bool optimizer_state_initialized;\n"
            "\n"
            "    bool seq_cls_enabled;\n"
            "    int seq_cls_num_classes;\n"
            "    int seq_cls_pooling;\n"
            "    size_t seq_cls_weight_offset;\n"
            "    size_t seq_cls_bias_offset;\n"
            "\n"
            "    bool kv_cache_enabled;\n"
            "    int kv_cache_capacity;\n"
            "    int kv_cache_tokens;\n"
            "\n"
            "    long *training_data_buffer;\n"
            "    long num_training_tokens;\n"
            "\n"
            "    uint8_t checksum[32];\n"
            "    uint8_t reserved[32];\n"
            "} TransformerModel;\n\n");
}

static void emit_dim_expr(FILE *out, CKDimKind dim)
{
    switch (dim) {
    case CK_DIM_TOKENS:              fprintf(out, "(size_t)m->context_window"); break;
    case CK_DIM_EMBED:               fprintf(out, "(size_t)m->embed_dim"); break;
    case CK_DIM_ALIGNED_EMBED:       fprintf(out, "m->aligned_embed_dim"); break;
    case CK_DIM_HEAD_DIM:            fprintf(out, "(size_t)m->head_dim"); break;
    case CK_DIM_ALIGNED_HEAD:        fprintf(out, "m->aligned_head_dim"); break;
    case CK_DIM_NUM_HEADS:           fprintf(out, "(size_t)m->num_attention_heads"); break;
    case CK_DIM_NUM_KV_HEADS:        fprintf(out, "(size_t)m->num_kv_heads"); break;
    case CK_DIM_ALIGNED_CTX:         fprintf(out, "m->aligned_attn_context_window"); break;
    case CK_DIM_INTERMEDIATE:        fprintf(out, "(size_t)m->intermediate_size"); break;
    case CK_DIM_ALIGNED_INTERMEDIATE:fprintf(out, "aligned_intermediate_dim"); break;
    case CK_DIM_VOCAB:               fprintf(out, "(size_t)m->vocab_size"); break;
    case CK_DIM_END:                 fprintf(out, "0"); break;
    }
}

static void emit_shape_expr(FILE *out, const CKDimToken *shape)
{
    int first = 1;
    for (int i = 0; i < 4; ++i) {
        if (shape[i].dim == CK_DIM_END) {
            break;
        }
        if (!first) {
            fprintf(out, " * ");
        }
        fprintf(out, "(");
        emit_dim_expr(out, shape[i].dim);
        if (shape[i].mult != 1) {
            fprintf(out, " * %d", shape[i].mult);
        }
        if (shape[i].div != 1) {
            fprintf(out, " / %d", shape[i].div);
        }
        fprintf(out, ")");
        first = 0;
    }
    if (first) {
        fprintf(out, "0");
    }
}

static void emit_bump_bytes_assignment(FILE *out,
                                      const char *indent,
                                      const char *struct_prefix,
                                      const char *name,
                                      const CKDimToken *shape)
{
    fprintf(out, "%s%s%s_offset = bump_bytes(&off, (", indent, struct_prefix, name);
    emit_shape_expr(out, shape);
    fprintf(out, ") * elem_bytes, CACHELINE_BYTES);\n");
}

static void emit_bump_bytes_assignment_weight_dtype(FILE *out,
                                                    const char *indent,
                                                    const char *struct_prefix,
                                                    const char *name,
                                                    const CKDimToken *shape,
                                                    const char *dtype_expr)
{
    fprintf(out, "%s%s%s_offset = bump_bytes(&off, ck_dtype_row_bytes(%s, (",
            indent, struct_prefix, name, dtype_expr);
    emit_shape_expr(out, shape);
    fprintf(out, ")), CACHELINE_BYTES);\n");
}

static void emit_training_conditional_assignment(FILE *out,
                                                 const char *indent,
                                                 const char *struct_prefix,
                                                 const char *name,
                                                 const CKDimToken *shape)
{
    /* Allocate gradient buffers only if training is enabled at init time */
    fprintf(out, "%s%s%s_offset = m->training_enabled ? bump_bytes(&off, (", indent, struct_prefix, name);
    emit_shape_expr(out, shape);
    fprintf(out, ") * elem_bytes, CACHELINE_BYTES) : 0;\n");
}

static void emit_global_allocations(FILE *out)
{
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        if (!ck_buffer_should_alloc(spec)) {
            continue;
        }
        if (spec->role == CK_ROLE_GRAD) {
            emit_training_conditional_assignment(out, "    ", "m->", spec->name, spec->shape);
            continue;
        }
        if (spec->alias_of) {
            const CKBufferSpec *alias = ck_find_buffer_spec(spec->alias_of);
            if (alias && alias->scope == CK_SCOPE_GLOBAL) {
                fprintf(out, "    m->%s_offset = m->%s_offset;\n", spec->name, spec->alias_of);
            }
            continue;
        }
        if (spec->condition && strcmp(spec->condition, "rope_theta") == 0) {
            fprintf(out, "    if (m->rope_theta > 0.0f) {\n");
            fprintf(out, "        m->%s_offset = bump_bytes(&off, (", spec->name);
            emit_shape_expr(out, spec->shape);
            fprintf(out, ") * elem_bytes, CACHELINE_BYTES);\n");
            fprintf(out, "    } else {\n");
            fprintf(out, "        m->%s_offset = 0;\n", spec->name);
            fprintf(out, "    }\n");
            continue;
        }
        if (spec->condition && strcmp(spec->condition, "training_enabled") == 0) {
            fprintf(out, "    if (m->training_enabled) {\n");
            fprintf(out, "        m->%s_offset = bump_bytes(&off, (", spec->name);
            emit_shape_expr(out, spec->shape);
            fprintf(out, ") * elem_bytes, CACHELINE_BYTES);\n");
            fprintf(out, "    } else {\n");
            fprintf(out, "        m->%s_offset = 0;\n", spec->name);
            fprintf(out, "    }\n");
            continue;
        }
        if (ck_buffer_uses_weight_dtype(spec)) {
            const char *dtype_expr = ck_weight_dtype_expr(spec);
            if (dtype_expr) {
                emit_bump_bytes_assignment_weight_dtype(out, "    ", "m->", spec->name, spec->shape, dtype_expr);
                continue;
            }
        }
        emit_bump_bytes_assignment(out, "    ", "m->", spec->name, spec->shape);
    }
}

static void emit_layer_allocations(FILE *out)
{
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->scope != CK_SCOPE_LAYER) {
            continue;
        }
        if (!ck_buffer_should_alloc(spec)) {
            continue;
        }
        if (spec->role == CK_ROLE_GRAD) {
            emit_training_conditional_assignment(out, "        ", "L->", spec->name, spec->shape);
            continue;
        }
        if (spec->condition && strcmp(spec->condition, "training_enabled") == 0) {
            fprintf(out, "        if (m->training_enabled) {\n");
            fprintf(out, "            L->%s_offset = bump_bytes(&off, (", spec->name);
            emit_shape_expr(out, spec->shape);
            fprintf(out, ") * elem_bytes, CACHELINE_BYTES);\n");
            fprintf(out, "        } else {\n");
            fprintf(out, "            L->%s_offset = 0;\n", spec->name);
            fprintf(out, "        }\n");
            continue;
        }
        if (ck_buffer_uses_weight_dtype(spec)) {
            const char *dtype_expr = ck_weight_dtype_expr(spec);
            if (dtype_expr) {
                emit_bump_bytes_assignment_weight_dtype(out, "        ", "L->", spec->name, spec->shape, dtype_expr);
                continue;
            }
        }
        emit_bump_bytes_assignment(out, "        ", "L->", spec->name, spec->shape);
    }
}

static void emit_global_aliases_to_layer(FILE *out)
{
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->scope != CK_SCOPE_GLOBAL || !spec->alias_of) {
            continue;
        }
        const CKBufferSpec *alias = ck_find_buffer_spec(spec->alias_of);
        if (!alias || alias->scope != CK_SCOPE_LAYER) {
            continue;
        }
        fprintf(out,
                "    if (m->num_layers > 0) {\n"
                "        m->%s_offset = m->layers[m->num_layers - 1].%s_offset;\n"
                "    } else {\n"
                "        m->%s_offset = 0;\n"
                "    }\n",
                spec->name, spec->alias_of, spec->name);
    }
}

static void emit_zero_grad(FILE *out)
{
    fprintf(out,
            "static void zero_grad(TransformerModel *m)\n"
            "{\n"
            "    if (!m || !m->training_enabled) return;\n"
            "    uint8_t *base = m->memory_base;\n"
            "    size_t aligned_intermediate_dim = align_up_elems((size_t)m->intermediate_size, m->elem_bytes, CACHELINE_BYTES);\n");

    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->role != CK_ROLE_GRAD || spec->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        fprintf(out, "    if (m->%s_offset) {\n", spec->name);
        fprintf(out, "        memset(base + m->%s_offset, 0, (", spec->name);
        emit_shape_expr(out, spec->shape);
        fprintf(out, ") * m->elem_bytes);\n");
        fprintf(out, "    }\n");
    }

    fprintf(out,
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n");
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->role != CK_ROLE_GRAD || spec->scope != CK_SCOPE_LAYER) {
            continue;
        }
        fprintf(out, "        if (L->%s_offset) {\n", spec->name);
        fprintf(out, "            memset(base + L->%s_offset, 0, (", spec->name);
        emit_shape_expr(out, spec->shape);
        fprintf(out, ") * m->elem_bytes);\n");
        fprintf(out, "        }\n");
    }
    fprintf(out,
            "    }\n"
            "}\n\n");
}

static void emit_sgd_update(FILE *out)
{
    fprintf(out,
            "static void sgd_update(TransformerModel *m, float lr)\n"
            "{\n"
            "    if (!m || !m->training_enabled || lr == 0.0f) return;\n"
            "    uint8_t *base = m->memory_base;\n"
            "    size_t aligned_intermediate_dim = align_up_elems((size_t)m->intermediate_size, m->elem_bytes, CACHELINE_BYTES);\n");

    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->role != CK_ROLE_WEIGHT || spec->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        if (spec->alias_of) {
            continue;
        }
        char grad_name[128];
        snprintf(grad_name, sizeof(grad_name), "d_%s", spec->name);
        const CKBufferSpec *grad = ck_find_buffer_spec(grad_name);
        if (!grad || grad->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        fprintf(out,
                "    if (m->%s_offset && m->%s_offset) {\n"
                "        float *w = ptr_f32(base, m->%s_offset);\n"
                "        float *g = ptr_f32(base, m->%s_offset);\n"
                "        size_t count = (",
                spec->name, grad_name, spec->name, grad_name);
        emit_shape_expr(out, spec->shape);
        fprintf(out,
                ");\n"
                "        for (size_t i = 0; i < count; ++i) {\n"
                "            w[i] -= lr * g[i];\n"
                "        }\n"
                "    }\n");
    }

    fprintf(out,
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n");
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->role != CK_ROLE_WEIGHT || spec->scope != CK_SCOPE_LAYER) {
            continue;
        }
        char grad_name[128];
        snprintf(grad_name, sizeof(grad_name), "d_%s", spec->name);
        const CKBufferSpec *grad = ck_find_buffer_spec(grad_name);
        if (!grad || grad->scope != CK_SCOPE_LAYER) {
            continue;
        }
        fprintf(out,
                "        if (L->%s_offset && L->%s_offset) {\n"
                "            float *w = ptr_f32(base, L->%s_offset);\n"
                "            float *g = ptr_f32(base, L->%s_offset);\n"
                "            size_t count = (",
                spec->name, grad_name, spec->name, grad_name);
        emit_shape_expr(out, spec->shape);
        fprintf(out,
                ");\n"
                "            for (size_t i = 0; i < count; ++i) {\n"
                "                w[i] -= lr * g[i];\n"
                "            }\n"
                "        }\n");
    }
    fprintf(out,
            "    }\n"
            "}\n\n");
}

static int emit_unique_source(FILE *f,
                              const char *path,
                              const char **seen,
                              size_t *seen_count,
                              size_t seen_cap)
{
    if (!path || !path[0]) {
        return 0;
    }
    for (size_t i = 0; i < *seen_count; ++i) {
        if (strcmp(seen[i], path) == 0) {
            return 0;
        }
    }
    if (*seen_count >= seen_cap) {
        return -1;
    }
    fputs(path, f);
    fputc('\n', f);
    seen[*seen_count] = path;
    (*seen_count)++;
    return 0;
}

static int ck_plan_step_enabled(const CKPlanStep *step, const CKIRGraph *cfg)
{
    if (!step || !step->condition || !cfg) {
        return 1;
    }
    if (strcmp(step->condition, "rope_theta") == 0) {
        return cfg->config.rope_theta > 0.0f;
    }
    if (strcmp(step->condition, "rope_theta>0") == 0) {
        return cfg->config.rope_theta > 0.0f;
    }
    return 1;
}

static int emit_plan_sources(FILE *f,
                             const CKPlanStep *plan,
                             size_t plan_count,
                             const CKIRGraph *cfg,
                             const char **seen,
                             size_t *seen_count,
                             size_t seen_cap)
{
    for (size_t i = 0; i < plan_count; ++i) {
        const CKPlanStep *step = &plan[i];
        if (!ck_plan_step_enabled(step, cfg)) {
            continue;
        }
        const CKKernelSpec *spec = ck_find_kernel_spec(step->kernel);
        if (!spec) {
            continue;
        }
        for (size_t s = 0; s < CKERNEL_MAX_KERNEL_SOURCES; ++s) {
            const char *src = spec->sources[s];
            if (!src) {
                continue;
            }
            if (emit_unique_source(f, src, seen, seen_count, seen_cap) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static const char *ck_first_layer_buffer_name(void)
{
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        const CKBufferSpec *spec = &ck_decoder_buffers[i];
        if (spec->scope != CK_SCOPE_LAYER) {
            continue;
        }
        if (!ck_buffer_should_alloc(spec)) {
            continue;
        }
        return spec->name;
    }
    return "ln1_gamma";
}

void ck_codegen_c_skeleton(const CKIRGraph *forward,
                           const CKIRGraph *backward,
                           FILE *out)
{
    if (!forward || !out) {
        return;
    }

    fprintf(out,
            "/* Auto-generated skeleton from CKIRGraph.\n"
            " * This file sketches the structure of the forward and backward\n"
            " * execution for a decoder-only transformer. It is NOT yet a\n"
            " * complete, runnable implementation. You can use it as a\n"
            " * starting point to wire buffers, kernel calls, and memory layout.\n"
            " */\n\n");

    fprintf(out, "#include \"ckernel_engine.h\"\n");
    fprintf(out, "#include \"ckernel_model.h\"\n");
    fprintf(out, "#include \"ckernel_alloc.h\"\n\n");

    /* Forward function */
    fprintf(out,
            "void run_decoder_forward(TransformerModel *model /*, inputs, etc. */)\n"
            "{\n"
            "    for (int layer = 0; layer < model->cfg.num_layers; ++layer) {\n"
            "        /* Forward pass for layer */\n");

    int nodes_per_layer = 0;
    if (forward->num_nodes > 0) {
        int l0 = forward->nodes[0].id.layer;
        for (int i = 0; i < forward->num_nodes; ++i) {
            if (forward->nodes[i].id.layer != l0) {
                break;
            }
            nodes_per_layer++;
        }
    }

    if (nodes_per_layer <= 0) {
        nodes_per_layer = forward->num_nodes;
    }

    fprintf(out, "        /* This layer has %d IR nodes */\n", nodes_per_layer);

    for (int i = 0; i < nodes_per_layer; ++i) {
        const CKIRNode *n = &forward->nodes[i];
        fprintf(out, "        // L%%d: %s\n", op_name(n->op));
        fprintf(out,
                "        //   outputs: [");
        for (int o = 0; o < n->n_outputs; ++o) {
            if (o > 0) fprintf(out, ", ");
            fprintf(out, "L%%d:N%d:%d", n->id.node, o);
        }
        fprintf(out, "]\n");
        fprintf(out, "        //   inputs : [");
        for (int j = 0; j < n->n_inputs; ++j) {
            const CKInputRef *inp = &n->inputs[j];
            if (j > 0) fprintf(out, ", ");
            if (inp->producer.node == 0xFFFFu) {
                fprintf(out, "IN");
            } else {
                fprintf(out, "L%%d:N%u:%u",
                        (unsigned)inp->producer.node,
                        (unsigned)inp->out_index);
            }
        }
        fprintf(out, "]\n");
        fprintf(out,
                "        //   TODO: bind buffers/weights and call %s kernel here\n\n",
                op_name(n->op));
    }

    fprintf(out,
            "    } /* end for layer */\n"
            "}\n\n");

    /* Backward skeleton */
    if (backward && backward->nodes && backward->num_nodes > 0) {
        fprintf(out,
            "void run_decoder_backward(TransformerModel *model /*, grads, etc. */)\n"
            "{\n"
            "    for (int layer = model->cfg.num_layers - 1; layer >= 0; --layer) {\n"
            "        /* Backward pass for layer */\n");

        int bwd_per_layer = 0;
        int l0 = backward->nodes[0].id.layer;
        for (int i = 0; i < backward->num_nodes; ++i) {
            if (backward->nodes[i].id.layer != l0) break;
            bwd_per_layer++;
        }
        if (bwd_per_layer <= 0) bwd_per_layer = backward->num_nodes;

        fprintf(out, "        /* This layer has %d backward IR nodes */\n", bwd_per_layer);

        for (int i = 0; i < bwd_per_layer; ++i) {
            const CKIRNode *n = &backward->nodes[i];
            fprintf(out, "        // L%%d: %s\n", op_name(n->op));
            fprintf(out,
                    "        //   TODO: wire gradient tensors and call %s kernel here\n\n",
                    op_name(n->op));
        }

        fprintf(out,
                "    } /* end for layer */\n"
                "}\n\n");
    }

    fprintf(out,
            "int main(int argc, char **argv)\n"
            "{\n"
            "    (void)argc; (void)argv;\n"
            "    TransformerModel model = {0};\n"
            "    model.cfg.num_layers        = %d;\n"
            "    model.cfg.hidden_size       = %d;\n"
            "    model.cfg.intermediate_size = %d;\n"
            "    model.cfg.num_heads         = %d;\n"
            "    model.cfg.num_kv_heads      = %d;\n"
            "    model.cfg.vocab_size        = %d;\n"
            "    model.cfg.context_window    = %d;\n"
            "    model.cfg.rms_norm_eps      = %.9g;\n"
            "    model.cfg.rope_theta        = %.9g;\n"
            "    layout_transformer_from_ir(&model, NULL); /* TODO: pass IR if needed */\n"
            "    size_t bytes = model.total_bytes;\n"
            "    model.memory_base = (uint8_t *)ck_huge_alloc(bytes);\n"
            "    if (!model.memory_base) {\n"
            "        fprintf(stderr, \"Failed to allocate %%zu bytes for model\\n\", bytes);\n"
            "        return 1;\n"
            "    }\n"
            "    // TODO: load weights into model.memory_base based on offsets\n"
            "    run_decoder_forward(&model);\n"
            "    // TODO: run_decoder_backward(&model) when training\n"
            "    ck_huge_free(model.memory_base, bytes);\n"
            "    return 0;\n"
            "}\n",
            forward->config.num_layers,
            forward->config.hidden_size,
            forward->config.intermediate_size,
            forward->config.num_heads,
            forward->config.num_kv_heads,
            forward->config.vocab_size,
            forward->config.context_window,
            forward->config.rms_norm_eps,
            forward->config.rope_theta);
}

static int emit_runtime_preamble(FILE *out)
{
    fprintf(out,
            "/* Auto-generated runtime from CKIRGraph.\n"
            " * This file wires the existing C-Kernel-Engine kernels into a\n"
            " * decoder-only transformer forward pass.\n"
            " *\n"
            " * Compile (scalar): gcc -O2 generated_model.c $(cat generated_model.c.kernels) -Iinclude -lm -o generated_model\n"
            " * Compile (AVX-512): gcc -O3 -mavx512f -mfma generated_model.c $(cat generated_model.c.kernels) -Iinclude -lm -o generated_model\n"
            " */\n\n");

    fprintf(out,
            "#define _GNU_SOURCE\n"
            "#include <stddef.h>\n"
            "#include <stdint.h>\n"
            "#include <stdbool.h>\n"
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "#include <string.h>\n"
            "#include <math.h>\n"
            "#include <errno.h>\n"
            "#include <sys/types.h>\n"
            "#include <unistd.h>\n"
            "#include \"ckernel_engine.h\"\n"
            "#include \"ckernel_dtype.h\"\n"
            "#include \"ckernel_orchestration.h\"\n"
            "#include \"ckernel_alloc.h\"\n\n");

    fprintf(out,
            "#define CACHELINE_BYTES 64\n"
            "static size_t align_up_bytes(size_t n, size_t align) {\n"
            "    if (align == 0) return n;\n"
            "    return (n + align - 1) & ~(align - 1);\n"
            "}\n\n"
            "static size_t align_up_elems(size_t elems, size_t elem_bytes, size_t align) {\n"
            "    size_t bytes = elems * elem_bytes;\n"
            "    bytes = align_up_bytes(bytes, align);\n"
            "    return bytes / elem_bytes;\n"
            "}\n\n"
            "static size_t bump_bytes(size_t *off, size_t bytes, size_t align) {\n"
            "    size_t start = align_up_bytes(*off, align);\n"
            "    *off = start + bytes;\n"
            "    return start;\n"
            "}\n\n"
            "static inline float *ptr_f32(uint8_t *base, size_t offset) {\n"
            "    return (float *)(base + offset);\n"
            "}\n"
            "static inline const float *cptr_f32(const uint8_t *base, size_t offset) {\n"
            "    return (const float *)(base + offset);\n"
            "}\n\n");

    fprintf(out,
            "static inline uint8_t *ptr_u8(uint8_t *base, size_t offset) {\n"
            "    return base + offset;\n"
            "}\n"
            "static inline const void *cptr_void(const uint8_t *base, size_t offset) {\n"
            "    return (const void *)(base + offset);\n"
            "}\n\n");

    return 0;
}

static int emit_kernel_manifest(const CKIRGraph *forward, const char *runtime_path)
{
    if (!forward || !runtime_path) {
        return -1;
    }

    const char *suffix = ".kernels";
    size_t len = strlen(runtime_path) + strlen(suffix) + 1;
    char *path = (char *)malloc(len);
    if (!path) {
        return -1;
    }
    snprintf(path, len, "%s%s", runtime_path, suffix);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ck_codegen_emit_runtime: failed to open %s: %s\n",
                path, strerror(errno));
        free(path);
        return -1;
    }

    size_t seen_cap = ck_kernel_spec_count * CKERNEL_MAX_KERNEL_SOURCES + 8;
    const char **seen = (const char **)calloc(seen_cap, sizeof(char *));
    if (!seen) {
        fclose(f);
        free(path);
        return -1;
    }
    size_t seen_count = 0;

    emit_unique_source(f, "src/ckernel_alloc.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/ckernel_strict.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/cpu_features.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/embedding_kernels.c", seen, &seen_count, seen_cap);
    /* Quantized inference support (K-quants: Q4_K_M / Q4_K / Q6_K). */
    emit_unique_source(f, "src/kernels/dequant_kernels.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q4k.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q4k_q8k.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q6k.c", seen, &seen_count, seen_cap);
    /* Legacy quant support (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0). */
    emit_unique_source(f, "src/kernels/gemm_kernels_q4_0.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q4_1.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q5_0.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q5_1.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q8_0.c", seen, &seen_count, seen_cap);
    /* SSE/AVX/AVX2/VNNI fallback implementations for quantized kernels. */
    emit_unique_source(f, "src/kernels/gemm_kernels_q4k_sse.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q4k_avx.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q4k_q8k_avx2.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q4k_q8k_vnni.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q5_0_sse.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q5_0_sse_v2.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_kernels_q6k_sse.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/quantize_row_q8_k_sse.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/rope_kernels.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/loss_kernels.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/kv_cache_kernels.c", seen, &seen_count, seen_cap);
    /* Fused kernels used by orchestration layer. */
    emit_unique_source(f, "src/kernels/gemm_fused_kernels.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/mlp_fused_decode.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/gemm_microkernel.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/attention_decode_fused.c", seen, &seen_count, seen_cap);
    emit_unique_source(f, "src/kernels/attention_flash_true.c", seen, &seen_count, seen_cap);
    if (emit_plan_sources(f,
                          ck_decoder_forward_plan,
                          ck_decoder_forward_plan_count,
                          forward,
                          seen,
                          &seen_count,
                          seen_cap) != 0) {
        free(seen);
        fclose(f);
        free(path);
        return -1;
    }
    if (emit_plan_sources(f,
                          ck_decoder_backward_plan,
                          ck_decoder_backward_plan_count,
                          forward,
                          seen,
                          &seen_count,
                          seen_cap) != 0) {
        free(seen);
        fclose(f);
        free(path);
        return -1;
    }
    free(seen);

    fclose(f);
    fprintf(stderr, "[ck_codegen] kernels manifest written to %s\n", path);
    free(path);
    return 0;
}

/* Library API functions emitted when mode == CK_EMIT_LIBRARY */
static void emit_library_api(FILE *out, const CKIRGraph *forward)
{
    fprintf(out,
            "\n/* ═══════════════════════════════════════════════════════════════\n"
            " * C-Kernel-Engine Library API (for dlopen)\n"
            " * ═══════════════════════════════════════════════════════════════ */\n\n"
            "#ifdef _WIN32\n"
            "#define CK_EXPORT __declspec(dllexport)\n"
            "#else\n"
            "#define CK_EXPORT __attribute__((visibility(\"default\")))\n"
            "#endif\n\n"
            "typedef struct {\n"
            "    int num_layers;\n"
            "    int hidden_size;\n"
            "    int intermediate_size;\n"
            "    int num_heads;\n"
            "    int num_kv_heads;\n"
            "    int vocab_size;\n"
            "    int context_window;\n"
            "    float rms_norm_eps;\n"
            "    float rope_theta;\n"
            "} CKModelInfo;\n\n"
            "static TransformerModel g_model = {0};\n"
            "static int g_initialized = 0;\n"
            "static int g_fuse_swiglu_decode = -2;\n"
            "static int g_fuse_attn_decode = -2;\n\n"
            "static int ck_fuse_swiglu_decode_mode(void)\n"
            "{\n"
            "    if (g_fuse_swiglu_decode != -2) return g_fuse_swiglu_decode;\n"
            "    const char *env = getenv(\"CK_FUSE_SWIGLU_DECODE\");\n"
            "    if (!env || !env[0]) {\n"
            "        g_fuse_swiglu_decode = -1; /* auto */\n"
            "        return g_fuse_swiglu_decode;\n"
            "    }\n"
            "    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' || env[0] == 'F') {\n"
            "        g_fuse_swiglu_decode = 0;\n"
            "    } else {\n"
            "        g_fuse_swiglu_decode = 1;\n"
            "    }\n"
            "    return g_fuse_swiglu_decode;\n"
            "}\n\n");

    fprintf(out,
            "static int ck_fuse_attn_decode_mode(void)\n"
            "{\n"
            "    if (g_fuse_attn_decode != -2) return g_fuse_attn_decode;\n"
            "    const char *env = getenv(\"CK_FUSE_ATTN_DECODE\");\n"
            "    if (!env || !env[0]) {\n"
            "        g_fuse_attn_decode = -1; /* auto */\n"
            "        return g_fuse_attn_decode;\n"
            "    }\n"
            "    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' || env[0] == 'F') {\n"
            "        g_fuse_attn_decode = 0;\n"
            "    } else {\n"
            "        g_fuse_attn_decode = 1;\n"
            "    }\n"
            "    return g_fuse_attn_decode;\n"
            "}\n\n");

    fprintf(out,
            "static int run_model_decode(TransformerModel *m, int32_t token)\n"
            "{\n"
            "    if (!m || !m->memory_base) return -1;\n"
            "    /* KV-cache decode is an inference-only fast path; training uses the full forward/backward graph. */\n"
            "    if (m->training_enabled) return -4;\n"
            "    if (!m->kv_cache_enabled) return -2;\n"
            "\n"
            "    int cache_cap = m->kv_cache_capacity > 0 ? m->kv_cache_capacity : m->context_window;\n"
            "    if (cache_cap > m->context_window) cache_cap = m->context_window;\n"
            "    int t = m->kv_cache_tokens;\n"
            "    if (t < 0) t = 0;\n"
            "    if (t >= cache_cap) return -3;\n"
            "\n"
            "    embed_token_at(m, token, t);\n"
            "\n"
            "    uint8_t *base = m->memory_base;\n"
            "    float *current = ptr_f32(base, m->embedded_input_offset);\n"
            "    int aligned_intermediate_dim = (int)align_up_elems((size_t)m->intermediate_size, m->elem_bytes, CACHELINE_BYTES);\n"
            "    int fuse_swiglu_mode = ck_fuse_swiglu_decode_mode();\n"
            "    int use_fused_swiglu = 0;\n"
            "    if (fuse_swiglu_mode > 0) {\n"
            "        use_fused_swiglu = 1;\n"
            "    } else if (fuse_swiglu_mode == 0) {\n"
            "        use_fused_swiglu = 0;\n"
            "    } else if (!ck_strict_parity_enabled()) {\n"
            "        use_fused_swiglu = 1;\n"
            "    }\n"
            "    int fuse_attn_mode = ck_fuse_attn_decode_mode();\n"
            "    int use_fused_attn = 0;\n"
            "    if (fuse_attn_mode > 0) {\n"
            "        use_fused_attn = 1;\n"
            "    } else if (fuse_attn_mode == 0) {\n"
            "        use_fused_attn = 0;\n"
            "    } else if (!ck_strict_parity_enabled()) {\n"
            "        use_fused_attn = 1;\n"
            "    }\n"
            "    if (m->weights_quantized) {\n"
            "        use_fused_swiglu = 0;\n"
            "        use_fused_attn = 0;\n"
            "    }\n"
            "\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
                "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        if (!m->weights_mixed && m->weight_dtype == CK_DT_Q4_K) {\n"
            "            CKLayerForwardParamsQ4K p = {0};\n"
            "            p.tokens = cache_cap;\n"
            "            p.embed_dim = m->embed_dim;\n"
            "            p.aligned_embed_dim = (int)m->aligned_embed_dim;\n"
            "            p.num_heads = m->num_attention_heads;\n"
            "            p.num_kv_heads = m->num_kv_heads;\n"
            "            p.head_dim = m->head_dim;\n"
            "            p.aligned_head_dim = (int)m->aligned_head_dim;\n"
            "            p.aligned_context_window = (int)m->aligned_attn_context_window;\n"
            "            p.intermediate_dim = m->intermediate_size;\n"
            "            p.aligned_intermediate_dim = aligned_intermediate_dim;\n"
            "            p.eps = m->rms_norm_eps;\n"
            "            p.rope_pos_offset = t;\n"
            "            p.rope_cos = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_cos_cache_offset) : NULL;\n"
            "            p.rope_sin = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_sin_cache_offset) : NULL;\n"
            "            p.input = current;\n"
            "            p.ln1_gamma = cptr_f32(base, L->ln1_gamma_offset);\n"
            "            p.ln2_gamma = cptr_f32(base, L->ln2_gamma_offset);\n"
            "            p.wq = cptr_void(base, L->wq_offset);\n"
            "            p.bq = cptr_f32(base, L->bq_offset);\n"
            "            p.wk = cptr_void(base, L->wk_offset);\n"
            "            p.bk = cptr_f32(base, L->bk_offset);\n"
            "            p.wv = cptr_void(base, L->wv_offset);\n"
            "            p.bv = cptr_f32(base, L->bv_offset);\n"
            "            p.wo = cptr_void(base, L->wo_offset);\n"
            "            p.bo = cptr_f32(base, L->bo_offset);\n"
            "            p.w1 = cptr_void(base, L->w1_offset);\n"
            "            p.b1 = cptr_f32(base, L->b1_offset);\n"
            "            p.w2 = cptr_void(base, L->w2_offset);\n"
            "            p.b2 = cptr_f32(base, L->b2_offset);\n"
            "            p.ln1_out = ptr_f32(base, L->ln1_out_offset);\n"
            "            p.ln1_rstd = ptr_f32(base, L->ln1_rstd_offset);\n"
            "            p.k = ptr_f32(base, L->k_offset);\n"
            "            p.v = ptr_f32(base, L->v_offset);\n"
            "            p.proj_tmp = ptr_f32(base, L->proj_tmp_offset);\n"
            "            p.proj_scratch = ptr_f32(base, L->proj_scratch_offset);\n"
            "            p.residual1 = ptr_f32(base, L->residual1_offset);\n"
            "            p.ln2_out = ptr_f32(base, L->ln2_out_offset);\n"
            "            p.ln2_rstd = ptr_f32(base, L->ln2_rstd_offset);\n"
            "            p.fc1_out = ptr_f32(base, L->fc1_out_offset);\n"
            "            p.swiglu_out = ptr_f32(base, L->swiglu_out_offset);\n"
            "            p.mlp_out = ptr_f32(base, L->mlp_out_offset);\n"
            "            p.output = ptr_f32(base, L->output_offset);\n"
            "\n"
            "            ck_layer_forward_rmsnorm_swiglu_decode_q4_k(&p, t, cache_cap);\n"
            "        } else if (m->weights_quantized) {\n"
            "            CKLayerForwardParamsQ4K p = {0};\n"
            "            p.tokens = cache_cap;\n"
            "            p.embed_dim = m->embed_dim;\n"
            "            p.aligned_embed_dim = (int)m->aligned_embed_dim;\n"
            "            p.num_heads = m->num_attention_heads;\n"
            "            p.num_kv_heads = m->num_kv_heads;\n"
            "            p.head_dim = m->head_dim;\n"
            "            p.aligned_head_dim = (int)m->aligned_head_dim;\n"
            "            p.aligned_context_window = (int)m->aligned_attn_context_window;\n"
            "            p.intermediate_dim = m->intermediate_size;\n"
            "            p.aligned_intermediate_dim = aligned_intermediate_dim;\n"
            "            p.eps = m->rms_norm_eps;\n"
            "            p.rope_pos_offset = t;\n"
            "            p.rope_cos = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_cos_cache_offset) : NULL;\n"
            "            p.rope_sin = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_sin_cache_offset) : NULL;\n"
            "            p.input = current;\n"
            "            p.ln1_gamma = cptr_f32(base, L->ln1_gamma_offset);\n"
            "            p.ln2_gamma = cptr_f32(base, L->ln2_gamma_offset);\n"
            "            p.wq = cptr_void(base, L->wq_offset);\n"
            "            p.bq = cptr_f32(base, L->bq_offset);\n"
            "            p.wk = cptr_void(base, L->wk_offset);\n"
            "            p.bk = cptr_f32(base, L->bk_offset);\n"
            "            p.wv = cptr_void(base, L->wv_offset);\n"
            "            p.bv = cptr_f32(base, L->bv_offset);\n"
            "            p.wo = cptr_void(base, L->wo_offset);\n"
            "            p.bo = cptr_f32(base, L->bo_offset);\n"
            "            p.w1 = cptr_void(base, L->w1_offset);\n"
            "            p.b1 = cptr_f32(base, L->b1_offset);\n"
            "            p.w2 = cptr_void(base, L->w2_offset);\n"
            "            p.b2 = cptr_f32(base, L->b2_offset);\n"
            "            p.ln1_out = ptr_f32(base, L->ln1_out_offset);\n"
            "            p.ln1_rstd = ptr_f32(base, L->ln1_rstd_offset);\n"
            "            p.k = ptr_f32(base, L->k_offset);\n"
            "            p.v = ptr_f32(base, L->v_offset);\n"
            "            p.proj_tmp = ptr_f32(base, L->proj_tmp_offset);\n"
            "            p.proj_scratch = ptr_f32(base, L->proj_scratch_offset);\n"
            "            p.residual1 = ptr_f32(base, L->residual1_offset);\n"
            "            p.ln2_out = ptr_f32(base, L->ln2_out_offset);\n"
            "            p.ln2_rstd = ptr_f32(base, L->ln2_rstd_offset);\n"
            "            p.fc1_out = ptr_f32(base, L->fc1_out_offset);\n"
            "            p.swiglu_out = ptr_f32(base, L->swiglu_out_offset);\n"
            "            p.mlp_out = ptr_f32(base, L->mlp_out_offset);\n"
            "            p.output = ptr_f32(base, L->output_offset);\n"
            "            p.wq_dtype = L->wq_dtype;\n"
            "            p.wk_dtype = L->wk_dtype;\n"
            "            p.wv_dtype = L->wv_dtype;\n"
            "            p.wo_dtype = L->wo_dtype;\n"
            "            p.w1_dtype = L->w1_dtype;\n"
            "            p.w2_dtype = L->w2_dtype;\n"
            "\n"
            "            ck_layer_forward_rmsnorm_swiglu_decode_quant(&p, t, cache_cap);\n"
            "        } else {\n"
            "            CKLayerForwardParams p = {0};\n"
            "            p.tokens = cache_cap;\n"
            "            p.embed_dim = m->embed_dim;\n"
            "            p.aligned_embed_dim = (int)m->aligned_embed_dim;\n"
            "            p.num_heads = m->num_attention_heads;\n"
            "            p.num_kv_heads = m->num_kv_heads;\n"
            "            p.head_dim = m->head_dim;\n"
            "            p.aligned_head_dim = (int)m->aligned_head_dim;\n"
            "            p.aligned_context_window = (int)m->aligned_attn_context_window;\n"
            "            p.intermediate_dim = m->intermediate_size;\n"
            "            p.aligned_intermediate_dim = aligned_intermediate_dim;\n"
            "            p.eps = m->rms_norm_eps;\n"
            "            p.rope_pos_offset = t;\n"
            "            p.rope_cos = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_cos_cache_offset) : NULL;\n"
            "            p.rope_sin = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_sin_cache_offset) : NULL;\n"
            "            p.input = current;\n"
            "            p.ln1_gamma = cptr_f32(base, L->ln1_gamma_offset);\n"
            "            p.ln2_gamma = cptr_f32(base, L->ln2_gamma_offset);\n"
            "            p.wq = cptr_f32(base, L->wq_offset);\n"
            "            p.bq = cptr_f32(base, L->bq_offset);\n"
            "            p.wk = cptr_f32(base, L->wk_offset);\n"
            "            p.bk = cptr_f32(base, L->bk_offset);\n"
            "            p.wv = cptr_f32(base, L->wv_offset);\n"
            "            p.bv = cptr_f32(base, L->bv_offset);\n"
            "            p.wo = cptr_f32(base, L->wo_offset);\n"
            "            p.bo = cptr_f32(base, L->bo_offset);\n"
            "            p.w1 = cptr_f32(base, L->w1_offset);\n"
            "            p.b1 = cptr_f32(base, L->b1_offset);\n"
            "            p.w2 = cptr_f32(base, L->w2_offset);\n"
            "            p.b2 = cptr_f32(base, L->b2_offset);\n"
            "            p.ln1_out = ptr_f32(base, L->ln1_out_offset);\n"
            "            p.ln1_rstd = ptr_f32(base, L->ln1_rstd_offset);\n"
            "            p.k = ptr_f32(base, L->k_offset);\n"
            "            p.v = ptr_f32(base, L->v_offset);\n"
            "            p.proj_tmp = ptr_f32(base, L->proj_tmp_offset);\n"
            "            p.proj_scratch = ptr_f32(base, L->proj_scratch_offset);\n"
            "            p.residual1 = ptr_f32(base, L->residual1_offset);\n"
            "            p.ln2_out = ptr_f32(base, L->ln2_out_offset);\n"
            "            p.ln2_rstd = ptr_f32(base, L->ln2_rstd_offset);\n"
            "            p.fc1_out = ptr_f32(base, L->fc1_out_offset);\n"
            "            p.swiglu_out = ptr_f32(base, L->swiglu_out_offset);\n"
            "            p.mlp_out = ptr_f32(base, L->mlp_out_offset);\n"
            "            p.output = ptr_f32(base, L->output_offset);\n"
            "\n"
            "            if (use_fused_attn) {\n"
            "                if (use_fused_swiglu) {\n"
            "                    ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_mlp(&p, t, cache_cap);\n"
            "                } else {\n"
            "                    ck_layer_forward_rmsnorm_swiglu_decode_fused_attn(&p, t, cache_cap);\n"
            "                }\n"
            "            } else if (use_fused_swiglu) {\n"
            "                ck_layer_forward_rmsnorm_swiglu_decode_fused(&p, t, cache_cap);\n"
            "            } else {\n"
            "                ck_layer_forward_rmsnorm_swiglu_decode(&p, t, cache_cap);\n"
            "            }\n"
            "        }\n"
            "        current = ptr_f32(base, L->output_offset);\n"
            "    }\n"
            "\n"
            "    int V = m->vocab_size;\n"
            "    int D = m->embed_dim;\n"
            "    int aligned_D = (int)m->aligned_embed_dim;\n"
            "    float *final_in = current + (size_t)t * aligned_D;\n"
            "    float *final_out = ptr_f32(base, m->final_output_offset) + (size_t)t * aligned_D;\n"
            "    float *final_rstd = ptr_f32(base, m->final_ln_rstd_offset) + (size_t)t;\n"
            "\n"
            "    rmsnorm_forward(final_in,\n"
            "                    cptr_f32(base, m->final_ln_weight_offset),\n"
            "                    final_out,\n"
            "                    final_rstd,\n"
            "                    1,\n"
            "                    D,\n"
            "                    aligned_D,\n"
            "                    m->rms_norm_eps);\n"
            "    if (V > 0) {\n"
            "        float *logits_row = ptr_f32(base, m->logits_offset) + (size_t)t * (size_t)V;\n"
            "        if (m->lm_head_weight_dtype == CK_DT_Q4_K) {\n"
            "            gemm_nt_q4_k(final_out,\n"
            "                         cptr_void(base, m->lm_head_weight_offset),\n"
            "                         NULL,\n"
            "                         logits_row,\n"
            "                         1,\n"
            "                         V,\n"
            "                         aligned_D);\n"
            "        } else if (m->lm_head_weight_dtype == CK_DT_Q6_K) {\n"
            "            gemm_nt_q6_k(final_out,\n"
            "                         cptr_void(base, m->lm_head_weight_offset),\n"
            "                         NULL,\n"
            "                         logits_row,\n"
            "                         1,\n"
            "                         V,\n"
            "                         aligned_D);\n"
            "        } else {\n"
            "            lm_head_forward(final_out,\n"
            "                            cptr_f32(base, m->lm_head_weight_offset),\n"
            "                            logits_row,\n"
            "                            1,\n"
            "                            V,\n"
            "                            D,\n"
            "                            aligned_D);\n"
            "        }\n"
            "    }\n"
            "\n"
            "    m->kv_cache_tokens = t + 1;\n"
            "    m->active_tokens = m->kv_cache_tokens;\n"
            "    return 0;\n"
            "}\n\n");

    /* ck_model_init */
    fprintf(out,
            "CK_EXPORT int ck_model_init(const char *weights_path)\n"
            "{\n"
            "    if (g_initialized) return 0;\n"
            "    memcpy(g_model.magic, \"BUMPWGT3\", 8);\n"
            "    g_model.version = 3;\n"
            "    g_model.model_type = 0;\n"
            "    g_model.num_layers = %d;\n"
            "    g_model.embed_dim = %d;\n"
            "    g_model.intermediate_size = %d;\n"
            "    g_model.num_attention_heads = %d;\n"
            "    g_model.num_kv_heads = %d;\n"
            "    g_model.vocab_size = %d;\n"
            "    g_model.context_window = %d;\n"
            "    g_model.rms_norm_eps = (float)%.9g;\n"
            "    g_model.rope_theta = (float)%.9g;\n"
            "    g_model.num_cores = 1;\n"
            "    g_model.task_type = TASK_LM;\n"
            "    g_model.weight_dtype = CK_DT_FP32;\n"
            "    const char *wd = getenv(\"CK_WEIGHT_DTYPE\");\n"
            "    if (wd) {\n"
            "        if (strcmp(wd, \"q4_k\") == 0 || strcmp(wd, \"q4_k_m\") == 0 ||\n"
            "            strcmp(wd, \"Q4_K\") == 0 || strcmp(wd, \"Q4_K_M\") == 0) {\n"
            "            g_model.weight_dtype = CK_DT_Q4_K;\n"
            "        } else if (strcmp(wd, \"q6_k\") == 0 || strcmp(wd, \"q6_k_l\") == 0 ||\n"
            "                   strcmp(wd, \"Q6_K\") == 0 || strcmp(wd, \"Q6_K_L\") == 0) {\n"
            "            g_model.weight_dtype = CK_DT_Q6_K;\n"
            "        }\n"
            "    }\n"
            "    init_weight_dtypes_uniform(&g_model, g_model.weight_dtype);\n"
            "    refresh_weight_flags(&g_model);\n"
            "    /* Check env var to pre-allocate gradient buffers for training */\n"
            "    const char *train_env = getenv(\"CK_ENABLE_TRAINING\");\n"
            "    if (train_env && (train_env[0] == '1' || train_env[0] == 'y' || train_env[0] == 'Y')) {\n"
            "        g_model.training_enabled = true;\n"
            "        g_model.learning_rate = 1e-4f;\n"
            "    }\n"
            "    if (weights_path) {\n"
            "        int dtype_rc = load_weight_dtypes(weights_path, &g_model);\n"
            "        if (dtype_rc < 0) {\n"
            "            fprintf(stderr, \"Failed to read weight dtype table from %%s\\n\", weights_path);\n"
            "            return -6;\n"
            "        }\n"
            "    }\n"
            "    if (g_model.training_enabled && g_model.weights_quantized) {\n"
            "        fprintf(stderr, \"Quantized weights are inference-only; disable training for this model\\n\");\n"
            "        return -5;\n"
            "    }\n"
            "    g_model.kv_cache_enabled = false;\n"
            "    g_model.kv_cache_capacity = g_model.context_window;\n"
            "    g_model.kv_cache_tokens = 0;\n"
            "    if (layout_model(&g_model) != 0) return -1;\n"
            "    if (weights_path) {\n"
            "        if (load_model_weights(weights_path, &g_model) != 0) return -2;\n"
            "    }\n"
            "    g_initialized = 1;\n"
            "    return 0;\n"
            "}\n\n",
            forward->config.num_layers,
            forward->config.hidden_size,
            forward->config.intermediate_size,
            forward->config.num_heads,
            forward->config.num_kv_heads,
            forward->config.vocab_size,
            forward->config.context_window,
            forward->config.rms_norm_eps,
            forward->config.rope_theta);

    /* ck_model_get_info */
    fprintf(out,
            "CK_EXPORT void ck_model_get_info(CKModelInfo *info)\n"
            "{\n"
            "    if (!info) return;\n"
            "    info->num_layers = g_model.num_layers;\n"
            "    info->hidden_size = g_model.embed_dim;\n"
            "    info->intermediate_size = g_model.intermediate_size;\n"
            "    info->num_heads = g_model.num_attention_heads;\n"
            "    info->num_kv_heads = g_model.num_kv_heads;\n"
            "    info->vocab_size = g_model.vocab_size;\n"
            "    info->context_window = g_model.context_window;\n"
            "    info->rms_norm_eps = g_model.rms_norm_eps;\n"
            "    info->rope_theta = g_model.rope_theta;\n"
            "}\n\n");

    /* ck_model_embed_tokens */
    fprintf(out,
            "CK_EXPORT int ck_model_embed_tokens(const int32_t *tokens, int num_tokens)\n"
            "{\n"
            "    if (!g_initialized) return -1;\n"
            "    int cap = g_model.context_window;\n"
            "    if (g_model.kv_cache_enabled && g_model.kv_cache_capacity > 0 && g_model.kv_cache_capacity < cap) {\n"
            "        cap = g_model.kv_cache_capacity;\n"
            "    }\n"
            "    if (num_tokens > cap) num_tokens = cap;\n"
            "    if (num_tokens < 1) num_tokens = 1;\n"
            "    g_model.active_tokens = num_tokens;\n"
            "    if (g_model.kv_cache_enabled && !g_model.training_enabled) {\n"
            "        g_model.kv_cache_tokens = 0;\n"
            "    }\n"
            "    embed_tokens(&g_model, tokens, num_tokens);\n"
            "    return 0;\n"
            "}\n\n");

    /* ck_model_forward */
    fprintf(out,
            "CK_EXPORT int ck_model_forward(float *logits_out)\n"
            "{\n"
            "    if (!g_initialized) return -1;\n"
            "    run_model_forward(&g_model);\n"
            "    if (g_model.kv_cache_enabled && !g_model.training_enabled) {\n"
            "        g_model.kv_cache_tokens = g_model.active_tokens;\n"
            "    }\n"
            "    if (logits_out && g_model.vocab_size > 0) {\n"
            "        size_t n = (size_t)g_model.active_tokens * (size_t)g_model.vocab_size;\n"
            "        memcpy(logits_out, ptr_f32(g_model.memory_base, g_model.logits_offset), n * sizeof(float));\n"
            "    }\n"
            "    return 0;\n"
            "}\n\n");

    /* KV-cache helpers + decode API */
    fprintf(out,
            "CK_EXPORT int ck_model_kv_cache_enable(int capacity)\n"
            "{\n"
            "    if (!g_initialized) return -1;\n"
            "    if (g_model.training_enabled) return -4;\n"
            "    g_model.kv_cache_enabled = true;\n"
            "    int cap = capacity;\n"
            "    if (cap <= 0 || cap > g_model.context_window) cap = g_model.context_window;\n"
            "    g_model.kv_cache_capacity = cap;\n"
            "    g_model.kv_cache_tokens = 0;\n"
            "    g_model.active_tokens = 0;\n"
            "    return 0;\n"
            "}\n\n"
            "CK_EXPORT void ck_model_kv_cache_reset(void)\n"
            "{\n"
            "    if (!g_initialized) return;\n"
            "    g_model.kv_cache_tokens = 0;\n"
            "    g_model.active_tokens = 0;\n"
            "}\n\n"
            "CK_EXPORT int ck_model_kv_cache_get_tokens(void)\n"
            "{\n"
            "    return g_initialized ? g_model.kv_cache_tokens : 0;\n"
            "}\n\n"
            "CK_EXPORT int ck_model_decode(int32_t token, float *logits_out)\n"
            "{\n"
            "    if (!g_initialized) return -1;\n"
            "    if (g_model.training_enabled) return -4;\n"
            "    int ret = run_model_decode(&g_model, token);\n"
            "    if (ret != 0) return ret;\n"
            "    if (logits_out && g_model.vocab_size > 0) {\n"
            "        int t = g_model.active_tokens - 1;\n"
            "        memcpy(logits_out,\n"
            "               ptr_f32(g_model.memory_base, g_model.logits_offset) + (size_t)t * (size_t)g_model.vocab_size,\n"
            "               (size_t)g_model.vocab_size * sizeof(float));\n"
            "    }\n"
            "    return 0;\n"
            "}\n\n");

    /* ck_model_get_logits - get pointer to internal logits buffer */
    fprintf(out,
            "CK_EXPORT float* ck_model_get_logits(void)\n"
            "{\n"
            "    if (!g_initialized) return NULL;\n"
            "    return ptr_f32(g_model.memory_base, g_model.logits_offset);\n"
            "}\n\n");

    /* ck_model_backward */
    fprintf(out,
            "CK_EXPORT int ck_model_backward(const int32_t *tokens, const int32_t *targets, float *loss_out)\n"
            "{\n"
            "    if (!g_initialized) return -1;\n"
            "    return run_model_backward(&g_model, tokens, targets, loss_out);\n"
            "}\n\n");

    /* ck_model_free */
    fprintf(out,
            "CK_EXPORT void ck_model_free(void)\n"
            "{\n"
            "    if (!g_initialized) return;\n"
            "    if (g_model.memory_base) ck_huge_free(g_model.memory_base, g_model.total_bytes);\n"
            "    if (g_model.layers) free(g_model.layers);\n"
            "    memset(&g_model, 0, sizeof(g_model));\n"
            "    g_initialized = 0;\n"
            "}\n\n");

    /* ck_model_get_context_window */
    fprintf(out,
            "CK_EXPORT int ck_model_get_context_window(void) { return g_initialized ? g_model.context_window : 0; }\n"
            "CK_EXPORT int ck_model_get_vocab_size(void) { return g_initialized ? g_model.vocab_size : 0; }\n"
            "CK_EXPORT int ck_model_get_hidden_size(void) { return g_initialized ? g_model.embed_dim : 0; }\n"
            "CK_EXPORT int ck_model_get_active_tokens(void) { return g_initialized ? g_model.active_tokens : 0; }\n"
            "CK_EXPORT int ck_model_is_training_enabled(void) { return g_initialized ? g_model.training_enabled : 0; }\n"
            "CK_EXPORT void ck_model_set_learning_rate(float lr) { if (g_initialized) g_model.learning_rate = lr; }\n"
            "CK_EXPORT float ck_model_get_learning_rate(void) { return g_initialized ? g_model.learning_rate : 0.0f; }\n\n"
            "CK_EXPORT int ck_model_enable_training(float learning_rate)\n"
            "{\n"
            "    if (!g_initialized) return -1;\n"
            "    g_model.training_enabled = true;\n"
            "    g_model.learning_rate = learning_rate;\n"
            "    return 0;\n"
            "}\n\n"
            "CK_EXPORT void ck_model_disable_training(void)\n"
            "{\n"
            "    if (g_initialized) g_model.training_enabled = false;\n"
            "}\n\n"
            "CK_EXPORT void ck_model_optimizer_step(void)\n"
            "{\n"
            "    if (!g_initialized || !g_model.training_enabled) return;\n"
            "    sgd_update(&g_model, g_model.learning_rate);\n"
            "}\n\n");
}

int ck_codegen_emit_runtime(const CKIRGraph *forward, const char *path, CKEmitMode mode)
{
    if (!forward || !path) {
        return -1;
    }
    if (ck_ir_validate_supported(forward) != 0) {
        return -1;
    }

    FILE *out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ck_codegen_emit_runtime: failed to open %s: %s\n",
                path, strerror(errno));
        return -1;
    }

    if (emit_runtime_preamble(out) != 0) {
        fclose(out);
        return -1;
    }

    fprintf(out,
            "typedef enum {\n"
            "    TASK_LM = 0,\n"
            "    TASK_SEQ_CLS = 1\n"
            "} TaskType;\n\n"
            "typedef enum {\n"
            "    OPTIMIZER_SGD = 0,\n"
            "    OPTIMIZER_ADAM = 1\n"
            "} OptimizerType;\n\n"
            "typedef struct {\n"
            "    size_t total_gradient_floats;\n"
            "} GradientStorage;\n\n");

    emit_layer_offsets_struct(out);
    emit_model_struct(out);

    fprintf(out,
            "static int ensure_layers_allocated(TransformerModel *m)\n"
            "{\n"
            "    if (!m) return -1;\n"
            "    if (!m->layers && m->num_layers > 0) {\n"
            "        m->layers = (TrulyOptimalLayer *)calloc((size_t)m->num_layers, sizeof(TrulyOptimalLayer));\n"
            "        if (!m->layers) return -1;\n"
            "    }\n"
            "    return 0;\n"
            "}\n\n"
            "static void init_weight_dtypes_uniform(TransformerModel *m, CKDataType dt)\n"
            "{\n"
            "    if (!m) return;\n"
            "    m->token_emb_dtype = dt;\n"
            "    m->lm_head_weight_dtype = dt;\n"
            "    m->pos_emb_dtype = CK_DT_FP32;\n"
            "    if (ensure_layers_allocated(m) != 0) return;\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        L->wq_dtype = dt;\n"
            "        L->wk_dtype = dt;\n"
            "        L->wv_dtype = dt;\n"
            "        L->wo_dtype = dt;\n"
            "        L->w1_dtype = dt;\n"
            "        L->w2_dtype = dt;\n"
            "    }\n"
            "}\n\n"
            "static void refresh_weight_flags(TransformerModel *m)\n"
            "{\n"
            "    if (!m) return;\n"
            "    CKDataType base = m->token_emb_dtype;\n"
            "    int mixed = 0;\n"
            "    int quant = ck_dtype_is_quantized(base);\n"
            "    if (m->lm_head_weight_dtype != base) mixed = 1;\n"
            "    if (ck_dtype_is_quantized(m->lm_head_weight_dtype)) quant = 1;\n"
            "    if (m->layers) {\n"
            "        for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "            TrulyOptimalLayer *L = &m->layers[layer];\n"
            "            if (L->wq_dtype != base || L->wk_dtype != base || L->wv_dtype != base ||\n"
            "                L->wo_dtype != base || L->w1_dtype != base || L->w2_dtype != base) {\n"
            "                mixed = 1;\n"
            "            }\n"
            "            if (ck_dtype_is_quantized(L->wq_dtype) || ck_dtype_is_quantized(L->wk_dtype) ||\n"
            "                ck_dtype_is_quantized(L->wv_dtype) || ck_dtype_is_quantized(L->wo_dtype) ||\n"
            "                ck_dtype_is_quantized(L->w1_dtype) || ck_dtype_is_quantized(L->w2_dtype)) {\n"
            "                quant = 1;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    m->weights_mixed = mixed ? true : false;\n"
            "    m->weights_quantized = quant ? true : false;\n"
            "    if (!mixed) {\n"
            "        m->weight_dtype = base;\n"
            "    }\n"
            "}\n\n"
            "static int load_weight_dtypes(const char *path, TransformerModel *m)\n"
            "{\n"
            "    if (!path || !m) return -1;\n"
            "    FILE *f = fopen(path, \"rb\");\n"
            "    if (!f) return -1;\n"
            "    char magic[8];\n"
            "    if (fread(magic, 1, 8, f) != 8) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    if (memcmp(magic, \"BUMPWGT3\", 8) != 0) {\n"
            "        fclose(f);\n"
            "        return 0;\n"
            "    }\n"
            "    uint32_t version = 0;\n"
            "    if (fread(&version, sizeof(uint32_t), 1, f) != 1) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    if (version < 3) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    if (fseek(f, 128, SEEK_SET) != 0) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    uint32_t dtype_len = 0;\n"
            "    if (fread(&dtype_len, sizeof(uint32_t), 1, f) != 1) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    if (dtype_len == 0) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    uint8_t *dtype_buf = (uint8_t *)malloc(dtype_len);\n"
            "    if (!dtype_buf) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    if (fread(dtype_buf, 1, dtype_len, f) != dtype_len) {\n"
            "        free(dtype_buf);\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    fclose(f);\n"
            "\n"
            "    size_t expected = (size_t)m->num_layers * 14u + 4u;\n"
            "    if (dtype_len != expected) {\n"
            "        free(dtype_buf);\n"
            "        return -1;\n"
            "    }\n"
            "    if (ensure_layers_allocated(m) != 0) {\n"
            "        free(dtype_buf);\n"
            "        return -1;\n"
            "    }\n"
            "\n"
            "    size_t idx = 0;\n"
            "    CKDataType token_dt = (CKDataType)dtype_buf[idx++];\n"
            "    CKDataType pos_dt = (CKDataType)dtype_buf[idx++];\n"
            "    if (pos_dt != CK_DT_FP32) {\n"
            "        free(dtype_buf);\n"
            "        return -1;\n"
            "    }\n"
            "    if (token_dt != CK_DT_FP32 && token_dt != CK_DT_Q4_K && token_dt != CK_DT_Q6_K) {\n"
            "        free(dtype_buf);\n"
            "        return -1;\n"
            "    }\n"
            "    m->token_emb_dtype = token_dt;\n"
            "    m->lm_head_weight_dtype = token_dt;\n"
            "    m->pos_emb_dtype = pos_dt;\n"
            "\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        CKDataType ln1_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType ln2_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType wq_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType bq_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType wk_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType bk_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType wv_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType bv_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType wo_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType bo_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType w1_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType b1_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType w2_dt = (CKDataType)dtype_buf[idx++];\n"
            "        CKDataType b2_dt = (CKDataType)dtype_buf[idx++];\n"
            "\n"
            "        if (ln1_dt != CK_DT_FP32 || ln2_dt != CK_DT_FP32 ||\n"
            "            bq_dt != CK_DT_FP32 || bk_dt != CK_DT_FP32 ||\n"
            "            bv_dt != CK_DT_FP32 || bo_dt != CK_DT_FP32 ||\n"
            "            b1_dt != CK_DT_FP32 || b2_dt != CK_DT_FP32) {\n"
            "            free(dtype_buf);\n"
            "            return -1;\n"
            "        }\n"
            "        if ((wq_dt != CK_DT_FP32 && wq_dt != CK_DT_Q4_K && wq_dt != CK_DT_Q6_K) ||\n"
            "            (wk_dt != CK_DT_FP32 && wk_dt != CK_DT_Q4_K && wk_dt != CK_DT_Q6_K) ||\n"
            "            (wv_dt != CK_DT_FP32 && wv_dt != CK_DT_Q4_K && wv_dt != CK_DT_Q6_K) ||\n"
            "            (wo_dt != CK_DT_FP32 && wo_dt != CK_DT_Q4_K && wo_dt != CK_DT_Q6_K) ||\n"
            "            (w1_dt != CK_DT_FP32 && w1_dt != CK_DT_Q4_K && w1_dt != CK_DT_Q6_K) ||\n"
            "            (w2_dt != CK_DT_FP32 && w2_dt != CK_DT_Q4_K && w2_dt != CK_DT_Q6_K)) {\n"
            "            free(dtype_buf);\n"
            "            return -1;\n"
            "        }\n"
            "\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        L->wq_dtype = wq_dt;\n"
            "        L->wk_dtype = wk_dt;\n"
            "        L->wv_dtype = wv_dt;\n"
            "        L->wo_dtype = wo_dt;\n"
            "        L->w1_dtype = w1_dt;\n"
            "        L->w2_dtype = w2_dt;\n"
            "    }\n"
            "\n"
            "    CKDataType final_norm_dt = (CKDataType)dtype_buf[idx++];\n"
            "    CKDataType final_bias_dt = (CKDataType)dtype_buf[idx++];\n"
            "    free(dtype_buf);\n"
            "    if (final_norm_dt != CK_DT_FP32 || final_bias_dt != CK_DT_FP32) {\n"
            "        return -1;\n"
            "    }\n"
            "\n"
            "    refresh_weight_flags(m);\n"
            "    return 1;\n"
            "}\n\n"
            "\n"
            "static int layout_model(TransformerModel *m)\n"
            "{\n"
            "    if (!m) return -1;\n"
            "    if (m->num_attention_heads <= 0 || m->embed_dim <= 0) return -1;\n"
            "    if (m->num_kv_heads <= 0) m->num_kv_heads = m->num_attention_heads;\n"
            "    if (m->num_attention_heads %% m->num_kv_heads != 0) return -1;\n"
            "    if (m->context_window <= 0) m->context_window = 1;\n"
            "    if (m->vocab_size <= 0) m->vocab_size = 1;\n"
            "    if (m->intermediate_size <= 0) return -1;\n"
            "    m->head_dim = m->embed_dim / m->num_attention_heads;\n"
            "    if (m->rms_norm_eps <= 0.0f) m->rms_norm_eps = 1e-5f;\n"
            "    if (m->rope_theta < 0.0f) m->rope_theta = 0.0f;\n"
            "    if (m->rope_theta > 0.0f && (m->head_dim %% 2 != 0)) return -1;\n"
            "    if (m->elem_bytes == 0) m->elem_bytes = sizeof(float);\n"
            "    size_t elem_bytes = m->elem_bytes;\n"
            "    m->aligned_embed_dim = align_up_elems((size_t)m->embed_dim, elem_bytes, CACHELINE_BYTES);\n"
            "    m->aligned_head_dim = align_up_elems((size_t)m->head_dim, elem_bytes, CACHELINE_BYTES);\n"
            "    m->aligned_attn_context_window = align_up_elems((size_t)m->context_window, elem_bytes, CACHELINE_BYTES);\n"
            "    size_t aligned_intermediate_dim = align_up_elems((size_t)m->intermediate_size, elem_bytes, CACHELINE_BYTES);\n"
            "    if (ensure_layers_allocated(m) != 0) return -1;\n"
            "    if (m->weights_quantized) {\n"
            "        /* K-quant weights require K dimension to be a multiple of 256. */\n"
            "        if ((m->aligned_embed_dim %% 256) != 0) return -1;\n"
            "        if ((aligned_intermediate_dim %% 256) != 0) return -1;\n"
            "        int wo_quant = 0;\n"
            "        if (m->layers) {\n"
            "            for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "                if (ck_dtype_is_quantized(m->layers[layer].wo_dtype)) {\n"
            "                    wo_quant = 1;\n"
            "                    break;\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        if (wo_quant && (size_t)m->num_attention_heads * m->aligned_head_dim != m->aligned_embed_dim) return -1;\n"
            "    }\n"
            "\n"
            "    if (m->num_cores <= 0) m->num_cores = 1;\n"
            "    m->tokens_per_core = (m->context_window + m->num_cores - 1) / m->num_cores;\n"
            "\n"
            "    size_t off = 0;\n");
    emit_global_allocations(out);
    fprintf(out,
            "    m->layers_start_offset = off;\n"
            "\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n");
    emit_layer_allocations(out);
    fprintf(out,
            "    }\n"
            "\n");
    {
        const char *stride_field = ck_first_layer_buffer_name();
        fprintf(out,
                "    if (m->num_layers > 1) {\n"
                "        m->layer_stride = m->layers[1].%s_offset - m->layers[0].%s_offset;\n"
                "    } else {\n"
                "        m->layer_stride = 0;\n"
                "    }\n",
                stride_field, stride_field);
    }
    emit_global_aliases_to_layer(out);
    fprintf(out,
            "    m->total_bytes = align_up_bytes(off, CACHELINE_BYTES);\n"
            "    m->memory_base = (uint8_t *)ck_huge_alloc(m->total_bytes);\n"
            "    if (!m->memory_base) return -1;\n"
            "    if (m->rope_theta > 0.0f) {\n"
            "        rope_precompute_cache(ptr_f32(m->memory_base, m->rope_cos_cache_offset),\n"
            "                             ptr_f32(m->memory_base, m->rope_sin_cache_offset),\n"
            "                             m->context_window,\n"
            "                             m->head_dim,\n"
            "                             m->rope_theta,\n"
            "                             m->head_dim,\n"
            "                             \"none\",\n"
            "                             1.0f);\n"
            "    }\n"
            "    return 0;\n"
            "}\n\n");

    fprintf(out,
            "static void lm_head_forward(const float *hidden,\n"
            "                            const float *weights,\n"
            "                            float *logits,\n"
            "                            int T, int V, int D, int aligned_D);\n"
            "static void lm_head_backward(const float *hidden,\n"
            "                             const float *weights,\n"
            "                             const float *d_logits,\n"
            "                             float *d_hidden,\n"
            "                             float *d_weights,\n"
            "                             int T, int V, int D, int aligned_D);\n"
            "static void softmax_cross_entropy(const float *logits,\n"
            "                                  const int32_t *targets,\n"
            "                                  int T, int V,\n"
            "                                  float *d_logits,\n"
            "                                  float *loss_out);\n\n");

    fprintf(out,
            "static void run_model_forward(TransformerModel *m)\n"
            "{\n"
            "    uint8_t *base = m->memory_base;\n"
            "    float *current = ptr_f32(base, m->embedded_input_offset);\n"
            "    int aligned_intermediate_dim = (int)align_up_elems((size_t)m->intermediate_size, m->elem_bytes, CACHELINE_BYTES);\n"
            "    int T = m->active_tokens > 0 ? m->active_tokens : m->context_window;\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        if (!m->weights_mixed && m->weight_dtype == CK_DT_Q4_K) {\n"
            "            CKLayerForwardParamsQ4K p = {0};\n"
            "            p.tokens = T;\n"
            "            p.embed_dim = m->embed_dim;\n"
            "            p.aligned_embed_dim = (int)m->aligned_embed_dim;\n"
            "            p.num_heads = m->num_attention_heads;\n"
            "            p.num_kv_heads = m->num_kv_heads;\n"
            "            p.head_dim = m->head_dim;\n"
            "            p.aligned_head_dim = (int)m->aligned_head_dim;\n"
            "            p.aligned_context_window = (int)m->aligned_attn_context_window;\n"
            "            p.intermediate_dim = m->intermediate_size;\n"
            "            p.aligned_intermediate_dim = aligned_intermediate_dim;\n"
            "            p.eps = m->rms_norm_eps;\n"
            "            p.rope_pos_offset = 0;\n"
            "            p.rope_cos = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_cos_cache_offset) : NULL;\n"
            "            p.rope_sin = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_sin_cache_offset) : NULL;\n"
            "            p.input = current;\n"
            "            p.ln1_gamma = cptr_f32(base, L->ln1_gamma_offset);\n"
            "            p.ln2_gamma = cptr_f32(base, L->ln2_gamma_offset);\n"
            "            p.wq = cptr_void(base, L->wq_offset);\n"
            "            p.bq = cptr_f32(base, L->bq_offset);\n"
            "            p.wk = cptr_void(base, L->wk_offset);\n"
            "            p.bk = cptr_f32(base, L->bk_offset);\n"
            "            p.wv = cptr_void(base, L->wv_offset);\n"
            "            p.bv = cptr_f32(base, L->bv_offset);\n"
            "            p.wo = cptr_void(base, L->wo_offset);\n"
            "            p.bo = cptr_f32(base, L->bo_offset);\n"
            "            p.w1 = cptr_void(base, L->w1_offset);\n"
            "            p.b1 = cptr_f32(base, L->b1_offset);\n"
            "            p.w2 = cptr_void(base, L->w2_offset);\n"
            "            p.b2 = cptr_f32(base, L->b2_offset);\n"
            "            p.ln1_out = ptr_f32(base, L->ln1_out_offset);\n"
            "            p.ln1_rstd = ptr_f32(base, L->ln1_rstd_offset);\n"
            "            p.q = ptr_f32(base, L->q_offset);\n"
            "            p.k = ptr_f32(base, L->k_offset);\n"
            "            p.v = ptr_f32(base, L->v_offset);\n"
            "            p.scores = L->scores_offset ? ptr_f32(base, L->scores_offset) : NULL;\n"
            "            p.attn_out = ptr_f32(base, L->attn_out_offset);\n"
            "            p.proj_tmp = ptr_f32(base, L->proj_tmp_offset);\n"
            "            p.proj_scratch = ptr_f32(base, L->proj_scratch_offset);\n"
            "            p.residual1 = ptr_f32(base, L->residual1_offset);\n"
            "            p.ln2_out = ptr_f32(base, L->ln2_out_offset);\n"
            "            p.ln2_rstd = ptr_f32(base, L->ln2_rstd_offset);\n"
            "            p.fc1_out = ptr_f32(base, L->fc1_out_offset);\n"
            "            p.swiglu_out = ptr_f32(base, L->swiglu_out_offset);\n"
            "            p.mlp_out = ptr_f32(base, L->mlp_out_offset);\n"
            "            p.output = ptr_f32(base, L->output_offset);\n"
            "            ck_layer_forward_rmsnorm_swiglu_q4_k(&p);\n"
            "            if (m->kv_cache_enabled && !m->training_enabled) {\n"
            "                kv_cache_repack_head_major_inplace(p.k,\n"
            "                                                  p.num_kv_heads,\n"
            "                                                  T,\n"
            "                                                  m->kv_cache_capacity,\n"
            "                                                  p.aligned_head_dim);\n"
            "                kv_cache_repack_head_major_inplace(p.v,\n"
            "                                                  p.num_kv_heads,\n"
            "                                                  T,\n"
            "                                                  m->kv_cache_capacity,\n"
            "                                                  p.aligned_head_dim);\n"
            "            }\n"
            "            current = p.output;\n"
            "        } else if (m->weights_quantized) {\n"
            "            CKLayerForwardParamsQ4K p = {0};\n"
            "            p.tokens = T;\n"
            "            p.embed_dim = m->embed_dim;\n"
            "            p.aligned_embed_dim = (int)m->aligned_embed_dim;\n"
            "            p.num_heads = m->num_attention_heads;\n"
            "            p.num_kv_heads = m->num_kv_heads;\n"
            "            p.head_dim = m->head_dim;\n"
            "            p.aligned_head_dim = (int)m->aligned_head_dim;\n"
            "            p.aligned_context_window = (int)m->aligned_attn_context_window;\n"
            "            p.intermediate_dim = m->intermediate_size;\n"
            "            p.aligned_intermediate_dim = aligned_intermediate_dim;\n"
            "            p.eps = m->rms_norm_eps;\n"
            "            p.rope_pos_offset = 0;\n"
            "            p.rope_cos = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_cos_cache_offset) : NULL;\n"
            "            p.rope_sin = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_sin_cache_offset) : NULL;\n"
            "            p.input = current;\n"
            "            p.ln1_gamma = cptr_f32(base, L->ln1_gamma_offset);\n"
            "            p.ln2_gamma = cptr_f32(base, L->ln2_gamma_offset);\n"
            "            p.wq = cptr_void(base, L->wq_offset);\n"
            "            p.bq = cptr_f32(base, L->bq_offset);\n"
            "            p.wk = cptr_void(base, L->wk_offset);\n"
            "            p.bk = cptr_f32(base, L->bk_offset);\n"
            "            p.wv = cptr_void(base, L->wv_offset);\n"
            "            p.bv = cptr_f32(base, L->bv_offset);\n"
            "            p.wo = cptr_void(base, L->wo_offset);\n"
            "            p.bo = cptr_f32(base, L->bo_offset);\n"
            "            p.w1 = cptr_void(base, L->w1_offset);\n"
            "            p.b1 = cptr_f32(base, L->b1_offset);\n"
            "            p.w2 = cptr_void(base, L->w2_offset);\n"
            "            p.b2 = cptr_f32(base, L->b2_offset);\n"
            "            p.ln1_out = ptr_f32(base, L->ln1_out_offset);\n"
            "            p.ln1_rstd = ptr_f32(base, L->ln1_rstd_offset);\n"
            "            p.q = ptr_f32(base, L->q_offset);\n"
            "            p.k = ptr_f32(base, L->k_offset);\n"
            "            p.v = ptr_f32(base, L->v_offset);\n"
            "            p.scores = L->scores_offset ? ptr_f32(base, L->scores_offset) : NULL;\n"
            "            p.attn_out = ptr_f32(base, L->attn_out_offset);\n"
            "            p.proj_tmp = ptr_f32(base, L->proj_tmp_offset);\n"
            "            p.proj_scratch = ptr_f32(base, L->proj_scratch_offset);\n"
            "            p.residual1 = ptr_f32(base, L->residual1_offset);\n"
            "            p.ln2_out = ptr_f32(base, L->ln2_out_offset);\n"
            "            p.ln2_rstd = ptr_f32(base, L->ln2_rstd_offset);\n"
            "            p.fc1_out = ptr_f32(base, L->fc1_out_offset);\n"
            "            p.swiglu_out = ptr_f32(base, L->swiglu_out_offset);\n"
            "            p.mlp_out = ptr_f32(base, L->mlp_out_offset);\n"
            "            p.output = ptr_f32(base, L->output_offset);\n"
            "            p.wq_dtype = L->wq_dtype;\n"
            "            p.wk_dtype = L->wk_dtype;\n"
            "            p.wv_dtype = L->wv_dtype;\n"
            "            p.wo_dtype = L->wo_dtype;\n"
            "            p.w1_dtype = L->w1_dtype;\n"
            "            p.w2_dtype = L->w2_dtype;\n"
            "            ck_layer_forward_rmsnorm_swiglu_quant(&p);\n"
            "            if (m->kv_cache_enabled && !m->training_enabled) {\n"
            "                kv_cache_repack_head_major_inplace(p.k,\n"
            "                                                  p.num_kv_heads,\n"
            "                                                  T,\n"
            "                                                  m->kv_cache_capacity,\n"
            "                                                  p.aligned_head_dim);\n"
            "                kv_cache_repack_head_major_inplace(p.v,\n"
            "                                                  p.num_kv_heads,\n"
            "                                                  T,\n"
            "                                                  m->kv_cache_capacity,\n"
            "                                                  p.aligned_head_dim);\n"
            "            }\n"
            "            current = p.output;\n"
            "        } else {\n"
            "            CKLayerForwardParams p = {0};\n"
            "            p.tokens = T;\n"
            "            p.embed_dim = m->embed_dim;\n"
            "            p.aligned_embed_dim = (int)m->aligned_embed_dim;\n"
            "            p.num_heads = m->num_attention_heads;\n"
            "            p.num_kv_heads = m->num_kv_heads;\n"
            "            p.head_dim = m->head_dim;\n"
            "            p.aligned_head_dim = (int)m->aligned_head_dim;\n"
            "            p.aligned_context_window = (int)m->aligned_attn_context_window;\n"
            "            p.intermediate_dim = m->intermediate_size;\n"
            "            p.aligned_intermediate_dim = aligned_intermediate_dim;\n"
            "            p.eps = m->rms_norm_eps;\n"
            "            p.rope_pos_offset = 0;\n"
            "            p.rope_cos = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_cos_cache_offset) : NULL;\n"
            "            p.rope_sin = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_sin_cache_offset) : NULL;\n"
            "            p.input = current;\n"
            "            p.ln1_gamma = cptr_f32(base, L->ln1_gamma_offset);\n"
            "            p.ln2_gamma = cptr_f32(base, L->ln2_gamma_offset);\n"
            "            p.wq = cptr_f32(base, L->wq_offset);\n"
            "            p.bq = cptr_f32(base, L->bq_offset);\n"
            "            p.wk = cptr_f32(base, L->wk_offset);\n"
            "            p.bk = cptr_f32(base, L->bk_offset);\n"
            "            p.wv = cptr_f32(base, L->wv_offset);\n"
            "            p.bv = cptr_f32(base, L->bv_offset);\n"
            "            p.wo = cptr_f32(base, L->wo_offset);\n"
            "            p.bo = cptr_f32(base, L->bo_offset);\n"
            "            p.w1 = cptr_f32(base, L->w1_offset);\n"
            "            p.b1 = cptr_f32(base, L->b1_offset);\n"
            "            p.w2 = cptr_f32(base, L->w2_offset);\n"
            "            p.b2 = cptr_f32(base, L->b2_offset);\n"
            "            p.ln1_out = ptr_f32(base, L->ln1_out_offset);\n"
            "            p.ln1_rstd = ptr_f32(base, L->ln1_rstd_offset);\n"
            "            p.q = ptr_f32(base, L->q_offset);\n"
            "            p.k = ptr_f32(base, L->k_offset);\n"
            "            p.v = ptr_f32(base, L->v_offset);\n"
            "            p.scores = L->scores_offset ? ptr_f32(base, L->scores_offset) : NULL;\n"
            "            p.attn_out = ptr_f32(base, L->attn_out_offset);\n"
            "            p.proj_tmp = ptr_f32(base, L->proj_tmp_offset);\n"
            "            p.proj_scratch = ptr_f32(base, L->proj_scratch_offset);\n"
            "            p.residual1 = ptr_f32(base, L->residual1_offset);\n"
            "            p.ln2_out = ptr_f32(base, L->ln2_out_offset);\n"
            "            p.ln2_rstd = ptr_f32(base, L->ln2_rstd_offset);\n"
            "            p.fc1_out = ptr_f32(base, L->fc1_out_offset);\n"
            "            p.swiglu_out = ptr_f32(base, L->swiglu_out_offset);\n"
            "            p.mlp_out = ptr_f32(base, L->mlp_out_offset);\n"
            "            p.output = ptr_f32(base, L->output_offset);\n"
            "            ck_layer_forward_rmsnorm_swiglu(&p);\n"
            "            if (m->kv_cache_enabled && !m->training_enabled) {\n"
            "                kv_cache_repack_head_major_inplace(p.k,\n"
            "                                                  p.num_kv_heads,\n"
            "                                                  T,\n"
            "                                                  m->kv_cache_capacity,\n"
            "                                                  p.aligned_head_dim);\n"
            "                kv_cache_repack_head_major_inplace(p.v,\n"
            "                                                  p.num_kv_heads,\n"
            "                                                  T,\n"
            "                                                  m->kv_cache_capacity,\n"
            "                                                  p.aligned_head_dim);\n"
            "            }\n"
            "            current = p.output;\n"
            "        }\n"
            "    }\n"
            "    float *final_out = ptr_f32(base, m->final_output_offset);\n"
            "    rmsnorm_forward(current,\n"
            "                    cptr_f32(base, m->final_ln_weight_offset),\n"
            "                    final_out,\n"
            "                    ptr_f32(base, m->final_ln_rstd_offset),\n"
            "                    T,\n"
            "                    m->embed_dim,\n"
            "                    (int)m->aligned_embed_dim,\n"
            "                    m->rms_norm_eps);\n"
            "    if (m->vocab_size > 0) {\n"
            "        if (m->lm_head_weight_dtype == CK_DT_Q4_K) {\n"
            "            gemm_nt_q4_k(final_out,\n"
            "                         cptr_void(base, m->lm_head_weight_offset),\n"
            "                         NULL,\n"
            "                         ptr_f32(base, m->logits_offset),\n"
            "                         T,\n"
            "                         m->vocab_size,\n"
            "                         (int)m->aligned_embed_dim);\n"
            "        } else if (m->lm_head_weight_dtype == CK_DT_Q6_K) {\n"
            "            gemm_nt_q6_k(final_out,\n"
            "                         cptr_void(base, m->lm_head_weight_offset),\n"
            "                         NULL,\n"
            "                         ptr_f32(base, m->logits_offset),\n"
            "                         T,\n"
            "                         m->vocab_size,\n"
            "                         (int)m->aligned_embed_dim);\n"
            "        } else {\n"
            "            lm_head_forward(final_out,\n"
            "                            cptr_f32(base, m->lm_head_weight_offset),\n"
            "                            ptr_f32(base, m->logits_offset),\n"
            "                            T,\n"
            "                            m->vocab_size,\n"
            "                            m->embed_dim,\n"
            "                            (int)m->aligned_embed_dim);\n"
            "        }\n"
            "    }\n"
            "}\n\n");

    emit_zero_grad(out);
    emit_sgd_update(out);

    fprintf(out,
            "static int run_model_backward(TransformerModel *m,\n"
            "                              const int32_t *tokens,\n"
            "                              const int32_t *targets,\n"
            "                              float *loss_out)\n"
            "{\n"
            "    if (!m || !m->training_enabled) return 0;\n"
            "    if (!tokens || !targets) return -1;\n"
            "    if (m->num_layers <= 0) return -1;\n"
            "    int T = m->active_tokens > 0 ? m->active_tokens : m->context_window;\n"
            "    int V = m->vocab_size;\n"
            "    int D = m->embed_dim;\n"
            "    int aligned_D = (int)m->aligned_embed_dim;\n"
            "    uint8_t *base = m->memory_base;\n"
            "\n"
            "    zero_grad(m);\n"
            "\n"
            "    float *final_out = ptr_f32(base, m->final_output_offset);\n"
            "    float *logits = ptr_f32(base, m->logits_offset);\n"
            "    float *d_logits = ptr_f32(base, m->d_logits_offset);\n"
            "    float *d_final_out = ptr_f32(base, m->d_final_output_offset);\n"
            "    float *d_final_in = ptr_f32(base, m->d_final_input_offset);\n"
            "\n"
            "    float loss = 0.0f;\n"
            "    softmax_cross_entropy(logits, targets, T, V, d_logits, &loss);\n"
            "    if (loss_out) {\n"
            "        *loss_out = loss;\n"
            "    }\n"
            "    lm_head_backward(final_out,\n"
            "                     cptr_f32(base, m->lm_head_weight_offset),\n"
            "                     d_logits,\n"
            "                     d_final_out,\n"
            "                     ptr_f32(base, m->d_token_emb_offset),\n"
            "                     T, V, D, aligned_D);\n"
            "    rmsnorm_backward(d_final_out,\n"
            "                     ptr_f32(base, m->layers[m->num_layers - 1].output_offset),\n"
            "                     cptr_f32(base, m->final_ln_weight_offset),\n"
            "                     ptr_f32(base, m->final_ln_rstd_offset),\n"
            "                     d_final_in,\n"
            "                     ptr_f32(base, m->d_final_ln_weight_offset),\n"
            "                     T, D, aligned_D);\n"
            "\n"
            "    for (int layer = m->num_layers - 1; layer >= 0; --layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        CKLayerBackwardParams p = {0};\n"
            "        p.tokens = T;\n"
            "        p.embed_dim = m->embed_dim;\n"
            "        p.aligned_embed_dim = (int)m->aligned_embed_dim;\n"
            "        p.num_heads = m->num_attention_heads;\n"
            "        p.num_kv_heads = m->num_kv_heads;\n"
            "        p.head_dim = m->head_dim;\n"
            "        p.aligned_head_dim = (int)m->aligned_head_dim;\n"
            "        p.aligned_context_window = (int)m->aligned_attn_context_window;\n"
            "        p.intermediate_dim = m->intermediate_size;\n"
            "        p.aligned_intermediate_dim = (int)align_up_elems((size_t)m->intermediate_size, m->elem_bytes, CACHELINE_BYTES);\n"
            "        p.eps = m->rms_norm_eps;\n"
            "        p.rope_pos_offset = 0;\n"
            "        p.rope_cos = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_cos_cache_offset) : NULL;\n"
            "        p.rope_sin = (m->rope_theta > 0.0f) ? cptr_f32(base, m->rope_sin_cache_offset) : NULL;\n"
            "        p.input = (layer == 0) ? ptr_f32(base, m->embedded_input_offset)\n"
            "                             : ptr_f32(base, m->layers[layer - 1].output_offset);\n"
            "        p.ln1_gamma = cptr_f32(base, L->ln1_gamma_offset);\n"
            "        p.ln2_gamma = cptr_f32(base, L->ln2_gamma_offset);\n"
            "        p.ln1_out = cptr_f32(base, L->ln1_out_offset);\n"
            "        p.ln1_rstd = cptr_f32(base, L->ln1_rstd_offset);\n"
            "        p.ln2_out = cptr_f32(base, L->ln2_out_offset);\n"
            "        p.ln2_rstd = cptr_f32(base, L->ln2_rstd_offset);\n"
            "        p.wq = cptr_f32(base, L->wq_offset);\n"
            "        p.bq = cptr_f32(base, L->bq_offset);\n"
            "        p.wk = cptr_f32(base, L->wk_offset);\n"
            "        p.bk = cptr_f32(base, L->bk_offset);\n"
            "        p.wv = cptr_f32(base, L->wv_offset);\n"
            "        p.bv = cptr_f32(base, L->bv_offset);\n"
            "        p.wo = cptr_f32(base, L->wo_offset);\n"
            "        p.bo = cptr_f32(base, L->bo_offset);\n"
            "        p.w1 = cptr_f32(base, L->w1_offset);\n"
            "        p.b1 = cptr_f32(base, L->b1_offset);\n"
            "        p.w2 = cptr_f32(base, L->w2_offset);\n"
            "        p.b2 = cptr_f32(base, L->b2_offset);\n"
            "        p.q = cptr_f32(base, L->q_offset);\n"
            "        p.k = cptr_f32(base, L->k_offset);\n"
            "        p.v = cptr_f32(base, L->v_offset);\n"
            "        p.scores = L->scores_offset ? cptr_f32(base, L->scores_offset) : NULL;\n"
            "        p.attn_out = cptr_f32(base, L->attn_out_offset);\n"
            "        p.residual1 = cptr_f32(base, L->residual1_offset);\n"
            "        p.fc1_out = cptr_f32(base, L->fc1_out_offset);\n"
            "        p.swiglu_out = cptr_f32(base, L->swiglu_out_offset);\n"
            "        p.d_output = ptr_f32(base, L->d_output_offset);\n"
            "        p.d_input = ptr_f32(base, L->d_input_offset);\n"
            "        p.d_ln1_gamma = ptr_f32(base, L->d_ln1_gamma_offset);\n"
            "        p.d_ln2_gamma = ptr_f32(base, L->d_ln2_gamma_offset);\n"
            "        p.d_wq = ptr_f32(base, L->d_wq_offset);\n"
            "        p.d_bq = ptr_f32(base, L->d_bq_offset);\n"
            "        p.d_wk = ptr_f32(base, L->d_wk_offset);\n"
            "        p.d_bk = ptr_f32(base, L->d_bk_offset);\n"
            "        p.d_wv = ptr_f32(base, L->d_wv_offset);\n"
            "        p.d_bv = ptr_f32(base, L->d_bv_offset);\n"
            "        p.d_wo = ptr_f32(base, L->d_wo_offset);\n"
            "        p.d_bo = ptr_f32(base, L->d_bo_offset);\n"
            "        p.d_w1 = ptr_f32(base, L->d_w1_offset);\n"
            "        p.d_b1 = ptr_f32(base, L->d_b1_offset);\n"
            "        p.d_w2 = ptr_f32(base, L->d_w2_offset);\n"
            "        p.d_b2 = ptr_f32(base, L->d_b2_offset);\n"
            "        p.d_ln1_out = ptr_f32(base, L->d_ln1_out_offset);\n"
            "        p.d_q = ptr_f32(base, L->d_q_offset);\n"
            "        p.d_k = ptr_f32(base, L->d_k_offset);\n"
            "        p.d_v = ptr_f32(base, L->d_v_offset);\n"
            "        p.d_scores = ptr_f32(base, L->d_scores_offset);\n"
            "        p.d_attn_out = ptr_f32(base, L->d_attn_out_offset);\n"
            "        p.d_proj_tmp = ptr_f32(base, L->d_proj_tmp_offset);\n"
            "        p.d_residual1 = ptr_f32(base, L->d_residual1_offset);\n"
            "        p.d_ln2_out = ptr_f32(base, L->d_ln2_out_offset);\n"
            "        p.d_fc1_out = ptr_f32(base, L->d_fc1_out_offset);\n"
            "        p.d_swiglu_out = ptr_f32(base, L->d_swiglu_out_offset);\n"
            "        p.d_mlp_out = ptr_f32(base, L->d_mlp_out_offset);\n"
            "\n"
            "        const float *src = (layer == m->num_layers - 1)\n"
            "            ? d_final_in\n"
            "            : ptr_f32(base, m->layers[layer + 1].d_input_offset);\n"
            "        memcpy(p.d_output, src, (size_t)T * (size_t)aligned_D * sizeof(float));\n"
            "\n"
            "        ck_layer_backward_rmsnorm_swiglu(&p);\n"
            "    }\n"
            "\n"
            "    {\n"
            "        TrulyOptimalLayer *L0 = &m->layers[0];\n"
            "        embedding_backward(tokens,\n"
            "                           T,\n"
            "                           ptr_f32(base, L0->d_input_offset),\n"
            "                           ptr_f32(base, m->d_token_emb_offset),\n"
            "                           ptr_f32(base, m->d_pos_emb_offset),\n"
            "                           m->vocab_size,\n"
            "                           m->embed_dim,\n"
            "                           aligned_D,\n"
            "                           m->context_window,\n"
            "                           m->rope_theta <= 0.0f);\n"
            "    }\n"
            "\n"
            "    /* SGD update is now called separately via optimizer_step() */\n"
            "    return 0;\n"
            "}\n\n");

    fprintf(out,
            "static int parse_int_arg(const char *s, int *out)\n"
            "{\n"
            "    if (!s || !out) return 0;\n"
            "    char *end = NULL;\n"
            "    long v = strtol(s, &end, 10);\n"
            "    if (!end || *end != '\\0') return 0;\n"
            "    *out = (int)v;\n"
            "    return 1;\n"
            "}\n\n"
            "static int parse_float_arg(const char *s, float *out)\n"
            "{\n"
            "    if (!s || !out) return 0;\n"
            "    char *end = NULL;\n"
            "    double v = strtod(s, &end);\n"
            "    if (!end || *end != '\\0') return 0;\n"
            "    *out = (float)v;\n"
            "    return 1;\n"
            "}\n\n"
            "static void print_usage(const char *prog)\n"
            "{\n"
            "    printf(\"Usage: %%s [options]\\n\", prog);\n"
            "    printf(\"  --dump             Print layout summary (layer 0 only)\\n\");\n"
            "    printf(\"  --dump-all         Print layout summary for all layers\\n\");\n"
            "    printf(\"  --no-forward       Skip forward pass (layout + alloc only)\\n\");\n"
            "    printf(\"  --layers N         Override num_layers\\n\");\n"
            "    printf(\"  --embed N          Override embed_dim\\n\");\n"
            "    printf(\"  --intermediate N   Override intermediate_size\\n\");\n"
            "    printf(\"  --heads N          Override num_attention_heads\\n\");\n"
            "    printf(\"  --kv-heads N       Override num_kv_heads\\n\");\n"
            "    printf(\"  --vocab N          Override vocab_size\\n\");\n"
            "    printf(\"  --ctx N            Override context_window\\n\");\n"
            "    printf(\"  --cores N          Override num_cores\\n\");\n"
            "    printf(\"  --litmus           Run LM head + CE + backward litmus\\n\");\n"
            "    printf(\"  --backward         Run backward pass + SGD update (requires --tokens/--targets)\\n\");\n"
            "    printf(\"  --lr F             SGD learning rate (default: 1e-3 when --backward)\\n\");\n"
            "    printf(\"  --steps N          Training steps (default: 1)\\n\");\n"
            "    printf(\"  --log-steps       Print loss per step during training\\n\");\n"
            "    printf(\"  --strict          Enable strict parity mode (single-thread + double GEMM)\\n\");\n"
            "    printf(\"  --hidden PATH      Load hidden activations [T x aligned_D] f32\\n\");\n"
            "    printf(\"  --weights PATH     Load LM head weights [V x aligned_D] f32 (litmus)\\n\");\n"
            "    printf(\"  --targets PATH     Load target tokens [T] int32\\n\");\n"
            "    printf(\"  --model-weights PATH  Load full model weights (bump format)\\n\");\n"
            "    printf(\"  --tokens PATH      Load token IDs [T] int32 and build embeddings\\n\");\n"
            "    printf(\"  --out-logits PATH  Write logits [T x V] f32\\n\");\n"
            "    printf(\"  --out-dlogits PATH Write d_logits [T x V] f32\\n\");\n"
            "    printf(\"  --out-dhidden PATH Write d_hidden [T x aligned_D] f32\\n\");\n"
            "    printf(\"  --out-dweights PATH Write d_weights [V x aligned_D] f32\\n\");\n"
            "    printf(\"  --out-loss PATH    Write loss (single f32)\\n\");\n"
            "    printf(\"  --out-weights PATH Write model weights (flat, no header)\\n\");\n"
            "    printf(\"  --help             Show this help\\n\");\n"
            "}\n\n"
            "static int read_floats(const char *path, float *dst, size_t count)\n"
            "{\n"
            "    if (!path || !dst) return -1;\n"
            "    FILE *f = fopen(path, \"rb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    size_t got = fread(dst, sizeof(float), count, f);\n"
            "    fclose(f);\n"
            "    return got == count ? 0 : -1;\n"
            "}\n\n"
            "static int read_ints(const char *path, int32_t *dst, size_t count)\n"
            "{\n"
            "    if (!path || !dst) return -1;\n"
            "    FILE *f = fopen(path, \"rb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    size_t got = fread(dst, sizeof(int32_t), count, f);\n"
            "    fclose(f);\n"
            "    return got == count ? 0 : -1;\n"
            "}\n\n"
            "static int read_floats_file(FILE *f, float *dst, size_t count)\n"
            "{\n"
            "    if (!f || !dst) return -1;\n"
            "    size_t got = fread(dst, sizeof(float), count, f);\n"
            "    return got == count ? 0 : -1;\n"
            "}\n\n"
            "static int read_bytes_file(FILE *f, void *dst, size_t bytes)\n"
            "{\n"
            "    if (!f || !dst) return -1;\n"
            "    size_t got = fread(dst, 1, bytes, f);\n"
            "    return got == bytes ? 0 : -1;\n"
            "}\n\n"
            "static int write_floats_file(FILE *f, const float *src, size_t count)\n"
            "{\n"
            "    if (!f || !src) return -1;\n"
            "    size_t wrote = fwrite(src, sizeof(float), count, f);\n"
            "    return wrote == count ? 0 : -1;\n"
            "}\n\n"
            "static int write_bytes_file(FILE *f, const void *src, size_t bytes)\n"
            "{\n"
            "    if (!f || !src) return -1;\n"
            "    size_t wrote = fwrite(src, 1, bytes, f);\n"
            "    return wrote == bytes ? 0 : -1;\n"
            "}\n\n"
            "static int read_weight_file(FILE *f, CKDataType dtype, void *dst, size_t n_elements)\n"
            "{\n"
            "    if (!f || !dst) return -1;\n"
            "    if (dtype == CK_DT_FP32) {\n"
            "        return read_floats_file(f, (float *)dst, n_elements);\n"
            "    }\n"
            "    return read_bytes_file(f, dst, ck_dtype_row_bytes(dtype, n_elements));\n"
            "}\n\n"
            "static int write_weight_file(FILE *f, CKDataType dtype, const void *src, size_t n_elements)\n"
            "{\n"
            "    if (!f || !src) return -1;\n"
            "    if (dtype == CK_DT_FP32) {\n"
            "        return write_floats_file(f, (const float *)src, n_elements);\n"
            "    }\n"
            "    return write_bytes_file(f, src, ck_dtype_row_bytes(dtype, n_elements));\n"
            "}\n\n"
            "static int skip_bump_header(FILE *f)\n"
            "{\n"
            "    if (!f) return -1;\n"
            "    char magic[8];\n"
            "    if (fread(magic, 1, 8, f) != 8) return -1;\n"
            "    if (memcmp(magic, \"BUMPWGT3\", 8) == 0) {\n"
            "        if (fseek(f, 128, SEEK_SET) != 0) return -1;\n"
            "        uint32_t dtype_len = 0;\n"
            "        if (fread(&dtype_len, sizeof(uint32_t), 1, f) != 1) return -1;\n"
            "        if (fseek(f, (long)dtype_len, SEEK_CUR) != 0) return -1;\n"
            "        return 1;\n"
            "    }\n"
            "    if (memcmp(magic, \"BUMPWGT2\", 8) == 0) {\n"
            "        if (fseek(f, 128, SEEK_SET) != 0) return -1;\n"
            "        return 1;\n"
            "    }\n"
            "    if (fseek(f, 0, SEEK_SET) != 0) return -1;\n"
            "    return 0;\n"
            "}\n\n"
            "static int load_model_weights(const char *path, TransformerModel *m)\n"
            "{\n"
            "    if (!path || !m || !m->memory_base) return -1;\n"
            "    FILE *f = fopen(path, \"rb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    if (skip_bump_header(f) < 0) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    uint8_t *base = m->memory_base;\n"
            "    size_t aligned_intermediate = align_up_elems((size_t)m->intermediate_size, m->elem_bytes, CACHELINE_BYTES);\n"
            "    size_t tok_elems = (size_t)m->vocab_size * m->aligned_embed_dim;\n"
            "    if (read_weight_file(f, m->token_emb_dtype, ptr_u8(base, m->token_emb_offset), tok_elems) != 0) goto fail;\n"
            "    if (read_floats_file(f, ptr_f32(base, m->pos_emb_offset),\n"
            "                        (size_t)m->context_window * m->aligned_embed_dim) != 0) goto fail;\n"
            "\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        size_t head_w_stride = m->aligned_head_dim * m->aligned_embed_dim;\n"
            "        size_t q_w = (size_t)m->num_attention_heads * head_w_stride;\n"
            "        size_t kv_w = (size_t)m->num_kv_heads * head_w_stride;\n"
            "        size_t q_b = (size_t)m->num_attention_heads * m->aligned_head_dim;\n"
            "        size_t kv_b = (size_t)m->num_kv_heads * m->aligned_head_dim;\n"
            "        size_t wo_w = (size_t)m->num_attention_heads * m->aligned_embed_dim * m->aligned_head_dim;\n"
            "        size_t w1_w = (size_t)(2 * aligned_intermediate) * m->aligned_embed_dim;\n"
            "        size_t w2_w = m->aligned_embed_dim * aligned_intermediate;\n"
            "\n"
            "        if (read_floats_file(f, ptr_f32(base, L->ln1_gamma_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (read_floats_file(f, ptr_f32(base, L->ln2_gamma_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (read_weight_file(f, L->wq_dtype, ptr_u8(base, L->wq_offset), q_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, ptr_f32(base, L->bq_offset), q_b) != 0) goto fail;\n"
            "        if (read_weight_file(f, L->wk_dtype, ptr_u8(base, L->wk_offset), kv_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, ptr_f32(base, L->bk_offset), kv_b) != 0) goto fail;\n"
            "        if (read_weight_file(f, L->wv_dtype, ptr_u8(base, L->wv_offset), kv_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, ptr_f32(base, L->bv_offset), kv_b) != 0) goto fail;\n"
            "        if (read_weight_file(f, L->wo_dtype, ptr_u8(base, L->wo_offset), wo_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, ptr_f32(base, L->bo_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (read_weight_file(f, L->w1_dtype, ptr_u8(base, L->w1_offset), w1_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, ptr_f32(base, L->b1_offset), (size_t)(2 * aligned_intermediate)) != 0) goto fail;\n"
            "        if (read_weight_file(f, L->w2_dtype, ptr_u8(base, L->w2_offset), w2_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, ptr_f32(base, L->b2_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "    }\n"
            "\n"
            "    if (read_floats_file(f, ptr_f32(base, m->final_ln_weight_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "    if (read_floats_file(f, ptr_f32(base, m->final_ln_bias_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "\n"
            "    fclose(f);\n"
            "    return 0;\n"
            "fail:\n"
            "    fclose(f);\n"
            "    return -1;\n"
            "}\n\n"
            "static int save_model_weights(const char *path, const TransformerModel *m)\n"
            "{\n"
            "    if (!path || !m || !m->memory_base) return -1;\n"
            "    FILE *f = fopen(path, \"wb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    uint8_t *base = m->memory_base;\n"
            "    size_t aligned_intermediate = align_up_elems((size_t)m->intermediate_size, m->elem_bytes, CACHELINE_BYTES);\n"
            "    size_t tok_elems = (size_t)m->vocab_size * m->aligned_embed_dim;\n"
            "    if (write_weight_file(f, m->token_emb_dtype, cptr_void(base, m->token_emb_offset), tok_elems) != 0) goto fail;\n"
            "    if (write_floats_file(f, ptr_f32(base, m->pos_emb_offset),\n"
            "                         (size_t)m->context_window * m->aligned_embed_dim) != 0) goto fail;\n"
            "\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        const TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        size_t head_w_stride = m->aligned_head_dim * m->aligned_embed_dim;\n"
            "        size_t q_w = (size_t)m->num_attention_heads * head_w_stride;\n"
            "        size_t kv_w = (size_t)m->num_kv_heads * head_w_stride;\n"
            "        size_t q_b = (size_t)m->num_attention_heads * m->aligned_head_dim;\n"
            "        size_t kv_b = (size_t)m->num_kv_heads * m->aligned_head_dim;\n"
            "        size_t wo_w = (size_t)m->num_attention_heads * m->aligned_embed_dim * m->aligned_head_dim;\n"
            "        size_t w1_w = (size_t)(2 * aligned_intermediate) * m->aligned_embed_dim;\n"
            "        size_t w2_w = m->aligned_embed_dim * aligned_intermediate;\n"
            "\n"
            "        if (write_floats_file(f, cptr_f32(base, L->ln1_gamma_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (write_floats_file(f, cptr_f32(base, L->ln2_gamma_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (write_weight_file(f, L->wq_dtype, cptr_void(base, L->wq_offset), q_w) != 0) goto fail;\n"
            "        if (write_floats_file(f, cptr_f32(base, L->bq_offset), q_b) != 0) goto fail;\n"
            "        if (write_weight_file(f, L->wk_dtype, cptr_void(base, L->wk_offset), kv_w) != 0) goto fail;\n"
            "        if (write_floats_file(f, cptr_f32(base, L->bk_offset), kv_b) != 0) goto fail;\n"
            "        if (write_weight_file(f, L->wv_dtype, cptr_void(base, L->wv_offset), kv_w) != 0) goto fail;\n"
            "        if (write_floats_file(f, cptr_f32(base, L->bv_offset), kv_b) != 0) goto fail;\n"
            "        if (write_weight_file(f, L->wo_dtype, cptr_void(base, L->wo_offset), wo_w) != 0) goto fail;\n"
            "        if (write_floats_file(f, cptr_f32(base, L->bo_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (write_weight_file(f, L->w1_dtype, cptr_void(base, L->w1_offset), w1_w) != 0) goto fail;\n"
            "        if (write_floats_file(f, cptr_f32(base, L->b1_offset), (size_t)(2 * aligned_intermediate)) != 0) goto fail;\n"
            "        if (write_weight_file(f, L->w2_dtype, cptr_void(base, L->w2_offset), w2_w) != 0) goto fail;\n"
            "        if (write_floats_file(f, cptr_f32(base, L->b2_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "    }\n"
            "\n"
            "    if (write_floats_file(f, cptr_f32(base, m->final_ln_weight_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "    if (write_floats_file(f, cptr_f32(base, m->final_ln_bias_offset), m->aligned_embed_dim) != 0) goto fail;\n"
            "\n"
            "    fclose(f);\n"
            "    return 0;\n"
            "fail:\n"
            "    fclose(f);\n"
            "    return -1;\n"
            "}\n\n"
            "static void embed_tokens(const TransformerModel *m, const int32_t *tokens, int token_count)\n"
            "{\n"
            "    if (!m || !m->memory_base || !tokens) return;\n"
            "    const uint8_t *base = m->memory_base;\n"
            "    float *out = ptr_f32((uint8_t *)base, m->embedded_input_offset);\n"
            "    const float *tok_f32 = cptr_f32(base, m->token_emb_offset);\n"
            "    const uint8_t *tok_q = (const uint8_t *)cptr_void(base, m->token_emb_offset);\n"
            "    const float *pos = cptr_f32(base, m->pos_emb_offset);\n"
            "    int T = m->context_window;\n"
            "    int D = m->embed_dim;\n"
            "    int aligned_D = (int)m->aligned_embed_dim;\n"
            "    for (int t = 0; t < T; ++t) {\n"
            "        float *dst = out + (size_t)t * aligned_D;\n"
            "        if (t < token_count) {\n"
            "            int id = tokens[t];\n"
            "            if (id < 0 || id >= m->vocab_size) id = 0;\n"
            "            if (m->token_emb_dtype == CK_DT_Q4_K) {\n"
            "                size_t row_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, (size_t)aligned_D);\n"
            "                const void *row = tok_q + (size_t)id * row_bytes;\n"
            "                dequant_q4_k_row(row, dst, (size_t)aligned_D);\n"
            "            } else if (m->token_emb_dtype == CK_DT_Q6_K) {\n"
            "                size_t row_bytes = ck_dtype_row_bytes(CK_DT_Q6_K, (size_t)aligned_D);\n"
            "                const void *row = tok_q + (size_t)id * row_bytes;\n"
            "                dequant_q6_k_row(row, dst, (size_t)aligned_D);\n"
            "            } else {\n"
            "                const float *src = tok_f32 + (size_t)id * aligned_D;\n"
            "                memcpy(dst, src, (size_t)D * sizeof(float));\n"
            "            }\n"
            "            if (aligned_D > D) {\n"
            "                memset(dst + D, 0, (size_t)(aligned_D - D) * sizeof(float));\n"
            "            }\n"
            "            if (m->rope_theta <= 0.0f) {\n"
            "                const float *p = pos + (size_t)t * aligned_D;\n"
            "                for (int d = 0; d < D; ++d) {\n"
            "                    dst[d] += p[d];\n"
            "                }\n"
            "            }\n"
            "        } else {\n"
            "            memset(dst, 0, (size_t)aligned_D * sizeof(float));\n"
            "        }\n"
            "    }\n"
            "}\n\n"
            "static void embed_token_at(const TransformerModel *m, int32_t token, int t)\n"
            "{\n"
            "    if (!m || !m->memory_base) return;\n"
            "    if (t < 0 || t >= m->context_window) return;\n"
            "    const uint8_t *base = m->memory_base;\n"
            "    float *out = ptr_f32((uint8_t *)base, m->embedded_input_offset);\n"
            "    const float *tok_f32 = cptr_f32(base, m->token_emb_offset);\n"
            "    const uint8_t *tok_q = (const uint8_t *)cptr_void(base, m->token_emb_offset);\n"
            "    const float *pos = cptr_f32(base, m->pos_emb_offset);\n"
            "    int D = m->embed_dim;\n"
            "    int aligned_D = (int)m->aligned_embed_dim;\n"
            "    int id = (int)token;\n"
            "    if (id < 0 || id >= m->vocab_size) id = 0;\n"
            "    float *dst = out + (size_t)t * aligned_D;\n"
            "    if (m->token_emb_dtype == CK_DT_Q4_K) {\n"
            "        size_t row_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, (size_t)aligned_D);\n"
            "        const void *row = tok_q + (size_t)id * row_bytes;\n"
            "        dequant_q4_k_row(row, dst, (size_t)aligned_D);\n"
            "    } else if (m->token_emb_dtype == CK_DT_Q6_K) {\n"
            "        size_t row_bytes = ck_dtype_row_bytes(CK_DT_Q6_K, (size_t)aligned_D);\n"
            "        const void *row = tok_q + (size_t)id * row_bytes;\n"
            "        dequant_q6_k_row(row, dst, (size_t)aligned_D);\n"
            "    } else {\n"
            "        const float *src = tok_f32 + (size_t)id * aligned_D;\n"
            "        memcpy(dst, src, (size_t)D * sizeof(float));\n"
            "    }\n"
            "    if (aligned_D > D) {\n"
            "        memset(dst + D, 0, (size_t)(aligned_D - D) * sizeof(float));\n"
            "    }\n"
            "    if (m->rope_theta <= 0.0f) {\n"
            "        const float *p = pos + (size_t)t * aligned_D;\n"
            "        for (int d = 0; d < D; ++d) {\n"
            "            dst[d] += p[d];\n"
            "        }\n"
            "    }\n"
            "}\n\n"
            "static int write_floats(const char *path, const float *src, size_t count)\n"
            "{\n"
            "    if (!path || !src) return -1;\n"
            "    FILE *f = fopen(path, \"wb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    size_t wrote = fwrite(src, sizeof(float), count, f);\n"
            "    fclose(f);\n"
            "    return wrote == count ? 0 : -1;\n"
            "}\n\n"
            "static int write_float_scalar(const char *path, float v)\n"
            "{\n"
            "    if (!path) return -1;\n"
            "    FILE *f = fopen(path, \"wb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    size_t wrote = fwrite(&v, sizeof(float), 1, f);\n"
            "    fclose(f);\n"
            "    return wrote == 1 ? 0 : -1;\n"
            "}\n\n"
            "static void lm_head_forward(const float *hidden,\n"
            "                            const float *weights,\n"
            "                            float *logits,\n"
            "                            int T, int V, int D, int aligned_D)\n"
            "{\n"
            "    for (int t = 0; t < T; ++t) {\n"
            "        const float *h = hidden + (size_t)t * aligned_D;\n"
            "        float *out = logits + (size_t)t * V;\n"
            "        for (int v = 0; v < V; ++v) {\n"
            "            const float *w = weights + (size_t)v * aligned_D;\n"
            "            float sum = 0.0f;\n"
            "            for (int d = 0; d < D; ++d) {\n"
            "                sum += h[d] * w[d];\n"
            "            }\n"
            "            out[v] = sum;\n"
            "        }\n"
            "    }\n"
            "}\n\n"
            "static void softmax_cross_entropy(const float *logits,\n"
            "                                  const int32_t *targets,\n"
            "                                  int T, int V,\n"
            "                                  float *d_logits,\n"
            "                                  float *loss_out)\n"
            "{\n"
            "    double total = 0.0;\n"
            "    for (int t = 0; t < T; ++t) {\n"
            "        const float *row = logits + (size_t)t * V;\n"
            "        float *drow = d_logits + (size_t)t * V;\n"
            "        int target = targets[t];\n"
            "        float max_logit = row[0];\n"
            "        for (int v = 1; v < V; ++v) {\n"
            "            if (row[v] > max_logit) max_logit = row[v];\n"
            "        }\n"
            "        double sum_exp = 0.0;\n"
            "        for (int v = 0; v < V; ++v) {\n"
            "            drow[v] = expf(row[v] - max_logit);\n"
            "            sum_exp += drow[v];\n"
            "        }\n"
            "        float inv_sum = 1.0f / (float)sum_exp;\n"
            "        for (int v = 0; v < V; ++v) {\n"
            "            drow[v] *= inv_sum;\n"
            "        }\n"
            "        double logsum = (double)max_logit + log(sum_exp);\n"
            "        total += logsum - (double)row[target];\n"
            "        drow[target] -= 1.0f;\n"
            "        float scale = 1.0f / (float)T;\n"
            "        for (int v = 0; v < V; ++v) {\n"
            "            drow[v] *= scale;\n"
            "        }\n"
            "    }\n"
            "    if (loss_out) {\n"
            "        *loss_out = (float)(total / (double)T);\n"
            "    }\n"
            "}\n\n"
            "static void lm_head_backward(const float *hidden,\n"
            "                             const float *weights,\n"
            "                             const float *d_logits,\n"
            "                             float *d_hidden,\n"
            "                             float *d_weights,\n"
            "                             int T, int V, int D, int aligned_D)\n"
            "{\n"
            "    size_t dh_count = (size_t)T * aligned_D;\n"
            "    size_t dw_count = (size_t)V * aligned_D;\n"
            "    for (size_t i = 0; i < dh_count; ++i) d_hidden[i] = 0.0f;\n"
            "    for (size_t i = 0; i < dw_count; ++i) d_weights[i] = 0.0f;\n"
            "    for (int t = 0; t < T; ++t) {\n"
            "        const float *dlog = d_logits + (size_t)t * V;\n"
            "        for (int d = 0; d < D; ++d) {\n"
            "            double sum = 0.0;\n"
            "            for (int v = 0; v < V; ++v) {\n"
            "                sum += (double)dlog[v] * (double)weights[(size_t)v * aligned_D + d];\n"
            "            }\n"
            "            d_hidden[(size_t)t * aligned_D + d] = (float)sum;\n"
            "        }\n"
            "    }\n"
            "    for (int v = 0; v < V; ++v) {\n"
            "        float *dw = d_weights + (size_t)v * aligned_D;\n"
            "        for (int d = 0; d < D; ++d) {\n"
            "            double sum = 0.0;\n"
            "            for (int t = 0; t < T; ++t) {\n"
            "                sum += (double)d_logits[(size_t)t * V + v] * (double)hidden[(size_t)t * aligned_D + d];\n"
            "            }\n"
            "            dw[d] = (float)sum;\n"
            "        }\n"
            "    }\n"
            "}\n\n");

    fprintf(out,
            "static void dump_layer_offsets(const TransformerModel *m, int layer)\n"
            "{\n"
            "    const TrulyOptimalLayer *L = &m->layers[layer];\n"
            "    printf(\"Layer %%d offsets (bytes):\\n\", layer);\n"
            "    printf(\"  ln1_gamma=%%zu ln2_gamma=%%zu wq=%%zu wk=%%zu wv=%%zu wo=%%zu w1=%%zu w2=%%zu\\n\",\n"
            "           L->ln1_gamma_offset, L->ln2_gamma_offset, L->wq_offset, L->wk_offset,\n"
            "           L->wv_offset, L->wo_offset, L->w1_offset, L->w2_offset);\n"
            "    printf(\"  ln1_out=%%zu q=%%zu k=%%zu v=%%zu scores=%%zu attn_out=%%zu\\n\",\n"
            "           L->ln1_out_offset, L->q_offset, L->k_offset, L->v_offset,\n"
            "           L->scores_offset, L->attn_out_offset);\n"
            "    printf(\"  proj_tmp=%%zu residual1=%%zu ln2_out=%%zu fc1_out=%%zu swiglu_out=%%zu mlp_out=%%zu output=%%zu\\n\",\n"
            "           L->proj_tmp_offset, L->residual1_offset, L->ln2_out_offset,\n"
            "           L->fc1_out_offset, L->swiglu_out_offset, L->mlp_out_offset, L->output_offset);\n"
            "}\n\n"
            "static void dump_layout(const TransformerModel *m, int dump_all)\n"
            "{\n"
            "    size_t bytes = m->total_bytes;\n"
            "    printf(\"Model config:\\n\");\n"
            "    printf(\"  layers=%%d embed=%%d intermediate=%%d heads=%%d kv_heads=%%d\\n\",\n"
            "           m->num_layers, m->embed_dim, m->intermediate_size, m->num_attention_heads, m->num_kv_heads);\n"
            "    printf(\"  head_dim=%%d vocab=%%d ctx=%%d cores=%%d\\n\",\n"
            "           m->head_dim, m->vocab_size, m->context_window, m->num_cores);\n"
            "    printf(\"  eps=%%.6g rope_theta=%%.6g\\n\", m->rms_norm_eps, m->rope_theta);\n"
            "    printf(\"Aligned dims (elements): embed=%%zu head=%%zu ctx=%%zu\\n\",\n"
            "           m->aligned_embed_dim, m->aligned_head_dim, m->aligned_attn_context_window);\n"
            "    printf(\"Memory: total_bytes=%%zu\\n\", bytes);\n"
            "    printf(\"Global offsets (bytes): token=%%zu pos=%%zu embedded=%%zu layers_start=%%zu\\n\",\n"
            "           m->token_emb_offset, m->pos_emb_offset, m->embedded_input_offset, m->layers_start_offset);\n"
            "    printf(\"Final offsets (bytes): final_ln_w=%%zu final_ln_b=%%zu final_ln_mean=%%zu final_ln_rstd=%%zu\\n\",\n"
            "           m->final_ln_weight_offset, m->final_ln_bias_offset,\n"
            "           m->final_ln_mean_offset, m->final_ln_rstd_offset);\n"
            "    printf(\"LM/logits offsets (bytes): lm_head=%%zu logits=%%zu\\n\",\n"
            "           m->lm_head_weight_offset, m->logits_offset);\n"
            "    if (m->num_layers > 0) {\n"
            "        dump_layer_offsets(m, 0);\n"
            "        if (dump_all) {\n"
            "            for (int i = 1; i < m->num_layers; ++i) {\n"
            "                dump_layer_offsets(m, i);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n\n");

    /* Emit either main() for standalone or API for library mode */
    if (mode == CK_EMIT_STANDALONE) {
    fprintf(out,
            "int main(int argc, char **argv)\n"
            "{\n"
            "    int dump = 0;\n"
            "    int dump_all = 0;\n"
            "    int no_forward = 0;\n"
            "    int run_litmus = 0;\n"
            "    int run_backward = 0;\n"
            "    const char *litmus_hidden = NULL;\n"
            "    const char *litmus_weights = NULL;\n"
            "    const char *litmus_targets = NULL;\n"
            "    const char *model_weights = NULL;\n"
            "    const char *tokens_path = NULL;\n"
            "    const char *out_logits = NULL;\n"
            "    const char *out_dlogits = NULL;\n"
            "    const char *out_dhidden = NULL;\n"
            "    const char *out_dweights = NULL;\n"
            "    const char *out_loss = NULL;\n"
            "    const char *out_weights = NULL;\n"
            "    int steps = 1;\n"
            "    int log_steps = 0;\n"
            "    int strict = 0;\n"
            "    int32_t *tokens = NULL;\n"
            "    int32_t *targets = NULL;\n"
            "    TransformerModel m = {0};\n"
            "    memcpy(m.magic, \"BUMPWGT3\", 8);\n"
            "    m.version = 3;\n"
            "    m.model_type = 0;\n"
            "    m.num_layers = %d;\n"
            "    m.embed_dim = %d;\n"
            "    m.intermediate_size = %d;\n"
            "    m.num_attention_heads = %d;\n"
            "    m.num_kv_heads = %d;\n"
            "    m.vocab_size = %d;\n"
            "    m.context_window = %d;\n"
            "    m.rms_norm_eps = %.9g;\n"
            "    m.rope_theta = %.9g;\n"
            "    m.num_cores = 1;\n"
            "    m.task_type = TASK_LM;\n"
            "    m.optimizer = OPTIMIZER_SGD;\n"
            "    m.learning_rate = 0.0f;\n"
            "    for (int i = 1; i < argc; ++i) {\n"
            "        if (strcmp(argv[i], \"--dump\") == 0) {\n"
            "            dump = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--dump-all\") == 0) {\n"
            "            dump = 1;\n"
            "            dump_all = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--no-forward\") == 0) {\n"
            "            no_forward = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--strict\") == 0) {\n"
            "            strict = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--litmus\") == 0) {\n"
            "            run_litmus = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--backward\") == 0) {\n"
            "            run_backward = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--lr\") == 0 && i + 1 < argc) {\n"
            "            parse_float_arg(argv[++i], &m.learning_rate);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--help\") == 0) {\n"
            "            print_usage(argv[0]);\n"
            "            return 0;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--hidden\") == 0 && i + 1 < argc) {\n"
            "            litmus_hidden = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--weights\") == 0 && i + 1 < argc) {\n"
            "            litmus_weights = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--targets\") == 0 && i + 1 < argc) {\n"
            "            litmus_targets = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--model-weights\") == 0 && i + 1 < argc) {\n"
            "            model_weights = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--tokens\") == 0 && i + 1 < argc) {\n"
            "            tokens_path = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-logits\") == 0 && i + 1 < argc) {\n"
            "            out_logits = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-dlogits\") == 0 && i + 1 < argc) {\n"
            "            out_dlogits = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-dhidden\") == 0 && i + 1 < argc) {\n"
            "            out_dhidden = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-dweights\") == 0 && i + 1 < argc) {\n"
            "            out_dweights = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-loss\") == 0 && i + 1 < argc) {\n"
            "            out_loss = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-weights\") == 0 && i + 1 < argc) {\n"
            "            out_weights = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--steps\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &steps);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--log-steps\") == 0) {\n"
            "            log_steps = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--layers\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.num_layers);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--embed\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.embed_dim);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--intermediate\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.intermediate_size);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--heads\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.num_attention_heads);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--kv-heads\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.num_kv_heads);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--vocab\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.vocab_size);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--ctx\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.context_window);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--cores\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.num_cores);\n"
            "            continue;\n"
            "        }\n"
            "        fprintf(stderr, \"Unknown or invalid arg: %%s\\n\", argv[i]);\n"
            "        print_usage(argv[0]);\n"
            "        return 1;\n"
            "    }\n"
            "    if (strict) {\n"
            "        ck_set_strict_parity(1);\n"
            "    }\n"
            "    if (run_backward && m.learning_rate == 0.0f) {\n"
            "        m.learning_rate = 1e-3f;\n"
            "    }\n"
            "    m.training_enabled = run_backward;\n"
            "    m.weight_dtype = CK_DT_FP32;\n"
            "    {\n"
            "        const char *wd = getenv(\"CK_WEIGHT_DTYPE\");\n"
            "        if (wd) {\n"
            "            if (strcmp(wd, \"q4_k\") == 0 || strcmp(wd, \"q4_k_m\") == 0 ||\n"
            "                strcmp(wd, \"Q4_K\") == 0 || strcmp(wd, \"Q4_K_M\") == 0) {\n"
            "                m.weight_dtype = CK_DT_Q4_K;\n"
            "            } else if (strcmp(wd, \"q6_k\") == 0 || strcmp(wd, \"q6_k_l\") == 0 ||\n"
            "                       strcmp(wd, \"Q6_K\") == 0 || strcmp(wd, \"Q6_K_L\") == 0) {\n"
            "                m.weight_dtype = CK_DT_Q6_K;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    init_weight_dtypes_uniform(&m, m.weight_dtype);\n"
            "    refresh_weight_flags(&m);\n"
            "    if (model_weights) {\n"
            "        int dtype_rc = load_weight_dtypes(model_weights, &m);\n"
            "        if (dtype_rc < 0) {\n"
            "            fprintf(stderr, \"failed to read weight dtype table\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "    }\n"
            "    if (m.training_enabled && m.weights_quantized) {\n"
            "        fprintf(stderr, \"Quantized weights are inference-only; disable training\\n\");\n"
            "        return 1;\n"
            "    }\n"
            "    if (layout_model(&m) != 0) {\n"
            "        fprintf(stderr, \"layout_model failed\\n\");\n"
            "        return 1;\n"
            "    }\n"
            "    if (model_weights) {\n"
            "        if (load_model_weights(model_weights, &m) != 0) {\n"
            "            fprintf(stderr, \"failed to load model weights\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "    }\n"
            "    if (tokens_path) {\n"
            "        int T = m.context_window;\n"
            "        tokens = (int32_t *)malloc((size_t)T * sizeof(int32_t));\n"
            "        if (!tokens) {\n"
            "            fprintf(stderr, \"failed to alloc tokens\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        if (read_ints(tokens_path, tokens, (size_t)T) != 0) {\n"
            "            fprintf(stderr, \"failed to read tokens\\n\");\n"
            "            free(tokens);\n"
            "            tokens = NULL;\n"
            "            return 1;\n"
            "        }\n"
            "        if (!run_backward) {\n"
            "            embed_tokens(&m, tokens, T);\n"
            "            free(tokens);\n"
            "            tokens = NULL;\n"
            "        }\n"
            "    }\n"
            "    if (run_backward) {\n"
            "        if (!litmus_targets) {\n"
            "            fprintf(stderr, \"backward requires --targets\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        int T = m.context_window;\n"
            "        targets = (int32_t *)malloc((size_t)T * sizeof(int32_t));\n"
            "        if (!targets) {\n"
            "            fprintf(stderr, \"failed to alloc targets\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        if (read_ints(litmus_targets, targets, (size_t)T) != 0) {\n"
            "            fprintf(stderr, \"failed to read targets\\n\");\n"
            "            free(targets);\n"
            "            targets = NULL;\n"
            "            return 1;\n"
            "        }\n"
            "    }\n"
            "    if (dump) {\n"
            "        dump_layout(&m, dump_all);\n"
            "    }\n"
            "    if (run_litmus) {\n"
            "        if (!litmus_hidden || !litmus_weights || !litmus_targets) {\n"
            "            fprintf(stderr, \"litmus requires --hidden, --weights, and --targets\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        int T = m.context_window;\n"
            "        int V = m.vocab_size;\n"
            "        int D = m.embed_dim;\n"
            "        int aligned_D = (int)m.aligned_embed_dim;\n"
            "        float *hidden = ptr_f32(m.memory_base, m.final_output_offset);\n"
            "        float *weights = ptr_f32(m.memory_base, m.lm_head_weight_offset);\n"
            "        float *logits = ptr_f32(m.memory_base, m.logits_offset);\n"
            "        if (read_floats(litmus_hidden, hidden, (size_t)T * aligned_D) != 0) {\n"
            "            fprintf(stderr, \"failed to read hidden\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        if (read_floats(litmus_weights, weights, (size_t)V * aligned_D) != 0) {\n"
            "            fprintf(stderr, \"failed to read weights\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        int32_t *targets = (int32_t *)malloc((size_t)T * sizeof(int32_t));\n"
            "        if (!targets) {\n"
            "            fprintf(stderr, \"failed to alloc targets\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        if (read_ints(litmus_targets, targets, (size_t)T) != 0) {\n"
            "            fprintf(stderr, \"failed to read targets\\n\");\n"
            "            free(targets);\n"
            "            return 1;\n"
            "        }\n"
            "        float *d_logits = (float *)calloc((size_t)T * V, sizeof(float));\n"
            "        float *d_hidden = (float *)calloc((size_t)T * aligned_D, sizeof(float));\n"
            "        float *d_weights = (float *)calloc((size_t)V * aligned_D, sizeof(float));\n"
            "        if (!d_logits || !d_hidden || !d_weights) {\n"
            "            fprintf(stderr, \"failed to alloc grads\\n\");\n"
            "            free(targets);\n"
            "            free(d_logits);\n"
            "            free(d_hidden);\n"
            "            free(d_weights);\n"
            "            return 1;\n"
            "        }\n"
            "        lm_head_forward(hidden, weights, logits, T, V, D, aligned_D);\n"
            "        float loss = 0.0f;\n"
            "        softmax_cross_entropy(logits, targets, T, V, d_logits, &loss);\n"
            "        lm_head_backward(hidden, weights, d_logits, d_hidden, d_weights, T, V, D, aligned_D);\n"
            "        if (out_logits) write_floats(out_logits, logits, (size_t)T * V);\n"
            "        if (out_dlogits) write_floats(out_dlogits, d_logits, (size_t)T * V);\n"
            "        if (out_dhidden) write_floats(out_dhidden, d_hidden, (size_t)T * aligned_D);\n"
            "        if (out_dweights) write_floats(out_dweights, d_weights, (size_t)V * aligned_D);\n"
            "        if (out_loss) write_float_scalar(out_loss, loss);\n"
            "        if (!out_loss) printf(\"loss=%%.6f\\n\", loss);\n"
            "        free(targets);\n"
            "        free(d_logits);\n"
            "        free(d_hidden);\n"
            "        free(d_weights);\n"
            "        ck_huge_free(m.memory_base, m.total_bytes);\n"
            "        free(m.layers);\n"
            "        return 0;\n"
            "    }\n"
            "    // TODO: load weights into m.memory_base using the offsets above.\n"
            "    // TODO: write token/pos embeddings into embedded_input_offset.\n"
            "    if (!run_backward) {\n"
            "        if (!no_forward) {\n"
            "            run_model_forward(&m);\n"
            "        }\n"
            "    } else {\n"
            "        if (!tokens || !targets) {\n"
            "            fprintf(stderr, \"backward requires --tokens and --targets\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        if (steps < 1) steps = 1;\n"
            "        float loss = 0.0f;\n"
            "        for (int step = 0; step < steps; ++step) {\n"
            "            embed_tokens(&m, tokens, m.context_window);\n"
            "            run_model_forward(&m);\n"
            "            if (run_model_backward(&m, tokens, targets, &loss) != 0) {\n"
            "                fprintf(stderr, \"backward failed\\n\");\n"
            "                return 1;\n"
            "            }\n"
            "            if (log_steps) {\n"
            "                printf(\"step %%d loss=%%.6f\\n\", step, loss);\n"
            "            }\n"
            "        }\n"
            "        if (out_loss) {\n"
            "            write_float_scalar(out_loss, loss);\n"
            "        }\n"
            "    }\n"
            "    if (out_logits) {\n"
            "        write_floats(out_logits, ptr_f32(m.memory_base, m.logits_offset),\n"
            "                     (size_t)m.context_window * (size_t)m.vocab_size);\n"
            "    }\n"
            "    if (out_weights) {\n"
            "        if (save_model_weights(out_weights, &m) != 0) {\n"
            "            fprintf(stderr, \"failed to save model weights\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "    }\n"
            "    ck_huge_free(m.memory_base, m.total_bytes);\n"
            "    free(m.layers);\n"
            "    free(tokens);\n"
            "    free(targets);\n"
            "    return 0;\n"
            "}\n",
            forward->config.num_layers,
            forward->config.hidden_size,
            forward->config.intermediate_size,
            forward->config.num_heads,
            forward->config.num_kv_heads,
            forward->config.vocab_size,
            forward->config.context_window,
            forward->config.rms_norm_eps,
            forward->config.rope_theta);
    } else {
        /* Library mode - emit API functions instead of main() */
        emit_library_api(out, forward);
    }

    fclose(out);
    if (emit_kernel_manifest(forward, path) != 0) {
        return -1;
    }
    return 0;
}
