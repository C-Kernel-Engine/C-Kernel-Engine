#include "ckernel_codegen_v2_emit.h"

#include "ckernel_dtype.h"

#include <stdio.h>
#include <string.h>

#define CK_V2_SECTION_ALIGN 64

typedef struct {
    size_t aligned_embed;
    size_t aligned_head;
    size_t aligned_intermediate;
    size_t aligned_context;
} CKV2AlignInfo;

static size_t align_up_bytes(size_t n, size_t align)
{
    if (align == 0) {
        return n;
    }
    return (n + align - 1) & ~(align - 1);
}

static size_t align_up_elems(size_t elems, size_t elem_bytes, size_t align_bytes)
{
    size_t bytes = elems * elem_bytes;
    bytes = align_up_bytes(bytes, align_bytes);
    return bytes / elem_bytes;
}

static CKV2AlignInfo compute_align(const CKModelConfig *cfg)
{
    CKV2AlignInfo info = {0};
    if (!cfg) {
        return info;
    }
    size_t elem_bytes = sizeof(float);
    size_t head_dim = (cfg->num_heads > 0) ? (size_t)(cfg->hidden_size / cfg->num_heads) : 0;
    info.aligned_embed = align_up_elems((size_t)cfg->hidden_size, elem_bytes, CK_V2_SECTION_ALIGN);
    info.aligned_head = align_up_elems(head_dim, elem_bytes, CK_V2_SECTION_ALIGN);
    info.aligned_intermediate =
        align_up_elems((size_t)cfg->intermediate_size, elem_bytes, CK_V2_SECTION_ALIGN);
    info.aligned_context =
        align_up_elems((size_t)cfg->context_window, elem_bytes, CK_V2_SECTION_ALIGN);
    return info;
}

static size_t resolve_dim(const CKModelConfig *cfg,
                          const CKV2AlignInfo *align,
                          CKDimKind kind)
{
    switch (kind) {
    case CK_DIM_TOKENS:
        return cfg ? (size_t)cfg->context_window : 0;
    case CK_DIM_EMBED:
        return cfg ? (size_t)cfg->hidden_size : 0;
    case CK_DIM_ALIGNED_EMBED:
        return align ? align->aligned_embed : 0;
    case CK_DIM_HEAD_DIM:
        return (cfg && cfg->num_heads > 0) ? (size_t)(cfg->hidden_size / cfg->num_heads) : 0;
    case CK_DIM_ALIGNED_HEAD:
        return align ? align->aligned_head : 0;
    case CK_DIM_NUM_HEADS:
        return cfg ? (size_t)cfg->num_heads : 0;
    case CK_DIM_NUM_KV_HEADS:
        return cfg ? (size_t)cfg->num_kv_heads : 0;
    case CK_DIM_ALIGNED_CTX:
        return align ? align->aligned_context : 0;
    case CK_DIM_INTERMEDIATE:
        return cfg ? (size_t)cfg->intermediate_size : 0;
    case CK_DIM_ALIGNED_INTERMEDIATE:
        return align ? align->aligned_intermediate : 0;
    case CK_DIM_VOCAB:
        return cfg ? (size_t)cfg->vocab_size : 0;
    case CK_DIM_END:
    default:
        return 0;
    }
}

static size_t resolve_shape_elems(const CKModelConfig *cfg,
                                  const CKV2AlignInfo *align,
                                  const CKDimToken *shape)
{
    size_t total = 1;
    for (int i = 0; i < CK_IR_V2_MAX_DIMS; ++i) {
        if (shape[i].dim == CK_DIM_END) {
            break;
        }
        size_t dim = resolve_dim(cfg, align, shape[i].dim);
        size_t mult = (size_t)(shape[i].mult > 0 ? shape[i].mult : 1);
        size_t div = (size_t)(shape[i].div > 0 ? shape[i].div : 1);
        if (div == 0) div = 1;
        total = total * dim * mult / div;
    }
    return total;
}

static size_t buffer_bytes(const CKIRV2Buffer *buf,
                           const CKModelConfig *cfg,
                           const CKV2AlignInfo *align)
{
    size_t elems = resolve_shape_elems(cfg, align, buf->shape);
    return ck_dtype_row_bytes(buf->dtype, elems);
}

static int is_footer_global(const char *name)
{
    if (!name) {
        return 0;
    }
    if (strncmp(name, "final_", 6) == 0) {
        return 1;
    }
    if (strcmp(name, "lm_head_weight") == 0) {
        return 1;
    }
    if (strcmp(name, "logits") == 0) {
        return 1;
    }
    if (strncmp(name, "d_final_", 8) == 0) {
        return 1;
    }
    if (strcmp(name, "d_logits") == 0) {
        return 1;
    }
    return 0;
}

static void emit_span_field(FILE *out, const char *label)
{
    fprintf(out, "    CKV2Span %s;\n", label);
}

static void emit_span_value(FILE *out, const char *label, size_t offset, size_t size, int comma)
{
    fprintf(out, "        .%s = { %zu, %zu }%s\n", label, offset, size, comma ? "," : "");
}

static int is_activation_role(CKBufferRole role)
{
    return role == CK_ROLE_INPUT || role == CK_ROLE_OUTPUT ||
           role == CK_ROLE_ACTIVATION || role == CK_ROLE_SCRATCH;
}

static void emit_header_fields(FILE *out,
                               const CKIRV2Graph *graph,
                               CKBufferRole role_filter,
                               int activation_group)
{
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        if (buf->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        if (activation_group) {
            if (!is_activation_role(buf->role)) {
                continue;
            }
        } else if (buf->role != role_filter) {
            continue;
        }
        if (is_footer_global(buf->name)) {
            continue;
        }
        emit_span_field(out, buf->name);
    }
}

static void emit_body_fields(FILE *out,
                             const CKIRV2Graph *graph,
                             CKBufferRole role_filter,
                             int activation_group)
{
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        if (buf->scope != CK_SCOPE_LAYER) {
            continue;
        }
        if (activation_group) {
            if (!is_activation_role(buf->role)) {
                continue;
            }
        } else if (buf->role != role_filter) {
            continue;
        }
        emit_span_field(out, buf->name);
    }
}

static void emit_footer_fields(FILE *out,
                               const CKIRV2Graph *graph,
                               CKBufferRole role_filter,
                               int activation_group)
{
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        if (buf->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        if (activation_group) {
            if (!is_activation_role(buf->role)) {
                continue;
            }
        } else if (buf->role != role_filter) {
            continue;
        }
        if (!is_footer_global(buf->name)) {
            continue;
        }
        emit_span_field(out, buf->name);
    }
}

static size_t plan_size(const CKMemPlan *plan, int idx)
{
    if (!plan || !plan->spans || idx < 0 || idx >= plan->num_spans) {
        return 0;
    }
    return plan->spans[idx].size_bytes;
}

static void emit_header_values(FILE *out,
                               const CKIRV2Graph *graph,
                               const CKMemPlan *plan,
                               CKBufferRole role_filter,
                               int activation_group,
                               size_t *offset)
{
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        if (buf->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        if (activation_group) {
            if (!is_activation_role(buf->role)) {
                continue;
            }
        } else if (buf->role != role_filter) {
            continue;
        }
        if (is_footer_global(buf->name)) {
            continue;
        }
        size_t bytes = plan_size(plan, i);
        *offset = align_up_bytes(*offset, CK_V2_SECTION_ALIGN);
        emit_span_value(out, buf->name, *offset, bytes, 1);
        *offset += bytes;
    }
}

static size_t emit_body_values(FILE *out,
                               const CKIRV2Graph *graph,
                               const CKMemPlan *plan,
                               CKBufferRole role_filter,
                               int activation_group)
{
    size_t layer_offset = 0;
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        if (buf->scope != CK_SCOPE_LAYER) {
            continue;
        }
        if (activation_group) {
            if (!is_activation_role(buf->role)) {
                continue;
            }
        } else if (buf->role != role_filter) {
            continue;
        }
        size_t bytes = plan_size(plan, i);
        layer_offset = align_up_bytes(layer_offset, CK_V2_SECTION_ALIGN);
        emit_span_value(out, buf->name, layer_offset, bytes, 1);
        layer_offset += bytes;
    }
    return layer_offset;
}

static void emit_footer_values(FILE *out,
                               const CKIRV2Graph *graph,
                               const CKMemPlan *plan,
                               CKBufferRole role_filter,
                               int activation_group,
                               size_t *offset)
{
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        if (buf->scope != CK_SCOPE_GLOBAL) {
            continue;
        }
        if (activation_group) {
            if (!is_activation_role(buf->role)) {
                continue;
            }
        } else if (buf->role != role_filter) {
            continue;
        }
        if (!is_footer_global(buf->name)) {
            continue;
        }
        size_t bytes = plan_size(plan, i);
        *offset = align_up_bytes(*offset, CK_V2_SECTION_ALIGN);
        emit_span_value(out, buf->name, *offset, bytes, 1);
        *offset += bytes;
    }
}

void ck_codegen_v2_emit_sections(FILE *out,
                                 const CKIRV2Graph *graph,
                                 const CKMemPlan *prefill_plan,
                                 const CKMemPlan *decode_plan,
                                 const CKMemPlan *backward_plan)
{
    if (!out || !graph) {
        return;
    }

    const int L = graph->config.num_layers;

    fprintf(out,
            "typedef struct {\n"
            "    size_t offset;\n"
            "    size_t size;\n"
            "} CKV2Span;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_header_fields(out, graph, CK_ROLE_WEIGHT, 0);
    fprintf(out, "} CKV2HeaderWeights;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_body_fields(out, graph, CK_ROLE_WEIGHT, 0);
    fprintf(out, "} CKV2LayerWeights;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_footer_fields(out, graph, CK_ROLE_WEIGHT, 0);
    fprintf(out, "} CKV2FooterWeights;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_header_fields(out, graph, CK_ROLE_ACTIVATION, 1);
    fprintf(out, "} CKV2HeaderActivations;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_body_fields(out, graph, CK_ROLE_ACTIVATION, 1);
    fprintf(out, "} CKV2LayerActivations;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_footer_fields(out, graph, CK_ROLE_ACTIVATION, 1);
    fprintf(out, "} CKV2FooterActivations;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_header_fields(out, graph, CK_ROLE_GRAD, 0);
    fprintf(out, "} CKV2HeaderGrads;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_body_fields(out, graph, CK_ROLE_GRAD, 0);
    fprintf(out, "} CKV2LayerGrads;\n\n");

    fprintf(out, "typedef struct {\n");
    emit_footer_fields(out, graph, CK_ROLE_GRAD, 0);
    fprintf(out, "} CKV2FooterGrads;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    CKV2HeaderWeights header;\n"
            "    CKV2LayerWeights body;\n"
            "    CKV2FooterWeights footer;\n"
            "    size_t layer_stride_bytes;\n"
            "    size_t total_bytes;\n"
            "} CKV2WeightLayout;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    CKV2HeaderActivations header;\n"
            "    CKV2LayerActivations body;\n"
            "    CKV2FooterActivations footer;\n"
            "    size_t layer_stride_bytes;\n"
            "    size_t total_bytes;\n"
            "} CKV2ActivationLayout;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    CKV2HeaderGrads header;\n"
            "    CKV2LayerGrads body;\n"
            "    CKV2FooterGrads footer;\n"
            "    size_t layer_stride_bytes;\n"
            "    size_t total_bytes;\n"
            "} CKV2GradLayout;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    int num_layers;\n"
            "    int context_window;\n"
            "    int hidden_size;\n"
            "    int intermediate_size;\n"
            "    int num_heads;\n"
            "    int num_kv_heads;\n"
            "    int head_dim;\n"
            "    int vocab_size;\n"
            "} CKV2ModelConfig;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    CKV2WeightLayout weights;\n"
            "    CKV2ActivationLayout activations;\n"
            "    CKV2GradLayout grads;\n"
            "} CKV2SectionLayout;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    CKV2SectionLayout prefill;\n"
            "    CKV2SectionLayout decode;\n"
            "    CKV2SectionLayout backward;\n"
            "} CKV2SectionLayouts;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    CKV2ModelConfig config;\n"
            "    CKV2SectionLayouts decoder;\n"
            "} CKV2RuntimeLayout;\n\n");

    fprintf(out, "static const CKV2RuntimeLayout ck_v2_layout = {\n");
    fprintf(out, "    .config = {\n");
    fprintf(out, "        .num_layers = %d,\n", graph->config.num_layers);
    fprintf(out, "        .context_window = %d,\n", graph->config.context_window);
    fprintf(out, "        .hidden_size = %d,\n", graph->config.hidden_size);
    fprintf(out, "        .intermediate_size = %d,\n", graph->config.intermediate_size);
    fprintf(out, "        .num_heads = %d,\n", graph->config.num_heads);
    fprintf(out, "        .num_kv_heads = %d,\n", graph->config.num_kv_heads);
    fprintf(out, "        .head_dim = %d,\n",
            graph->config.num_heads > 0 ? graph->config.hidden_size / graph->config.num_heads : 0);
    fprintf(out, "        .vocab_size = %d\n", graph->config.vocab_size);
    fprintf(out, "    },\n");
    fprintf(out, "    .decoder = {\n");

    const CKMemPlan *plans[3] = { prefill_plan, decode_plan, backward_plan };
    const char *modes[3] = { "prefill", "decode", "backward" };

    for (int mode = 0; mode < 3; ++mode) {
        const CKMemPlan *plan = plans[mode];
        fprintf(out, "        .%s = {\n", modes[mode]);

        size_t offset = 0;
        size_t header_end = 0;
        size_t layer_stride = 0;
        size_t footer_base = 0;

        fprintf(out, "            .weights = {\n");
        fprintf(out, "                .header = {\n");
        emit_header_values(out, graph, prefill_plan, CK_ROLE_WEIGHT, 0, &offset);
        header_end = offset;
        fprintf(out, "                },\n");

        fprintf(out, "                .body = {\n");
        layer_stride = emit_body_values(out, graph, prefill_plan, CK_ROLE_WEIGHT, 0);
        fprintf(out, "                },\n");

        footer_base = header_end + layer_stride * (size_t)L;
        offset = footer_base;
        fprintf(out, "                .footer = {\n");
        emit_footer_values(out, graph, prefill_plan, CK_ROLE_WEIGHT, 0, &offset);
        fprintf(out, "                },\n");

        fprintf(out, "                .layer_stride_bytes = %zu,\n", layer_stride);
        fprintf(out, "                .total_bytes = %zu\n", offset);
        fprintf(out, "            },\n");

        offset = 0;
        header_end = 0;
        layer_stride = 0;
        footer_base = 0;

        fprintf(out, "            .activations = {\n");
        fprintf(out, "                .header = {\n");
        emit_header_values(out, graph, plan, CK_ROLE_ACTIVATION, 1, &offset);
        header_end = offset;
        fprintf(out, "                },\n");

        fprintf(out, "                .body = {\n");
        layer_stride = emit_body_values(out, graph, plan, CK_ROLE_ACTIVATION, 1);
        fprintf(out, "                },\n");

        footer_base = header_end + layer_stride * (size_t)L;
        offset = footer_base;
        fprintf(out, "                .footer = {\n");
        emit_footer_values(out, graph, plan, CK_ROLE_ACTIVATION, 1, &offset);
        fprintf(out, "                },\n");

        fprintf(out, "                .layer_stride_bytes = %zu,\n", layer_stride);
        fprintf(out, "                .total_bytes = %zu\n", offset);
        fprintf(out, "            },\n");

        offset = 0;
        header_end = 0;
        layer_stride = 0;
        footer_base = 0;

        fprintf(out, "            .grads = {\n");
        fprintf(out, "                .header = {\n");
        emit_header_values(out, graph, plan, CK_ROLE_GRAD, 0, &offset);
        header_end = offset;
        fprintf(out, "                },\n");

        fprintf(out, "                .body = {\n");
        layer_stride = emit_body_values(out, graph, plan, CK_ROLE_GRAD, 0);
        fprintf(out, "                },\n");

        footer_base = header_end + layer_stride * (size_t)L;
        offset = footer_base;
        fprintf(out, "                .footer = {\n");
        emit_footer_values(out, graph, plan, CK_ROLE_GRAD, 0, &offset);
        fprintf(out, "                },\n");

        fprintf(out, "                .layer_stride_bytes = %zu,\n", layer_stride);
        fprintf(out, "                .total_bytes = %zu\n", offset);
        fprintf(out, "            }\n");

        fprintf(out, "        }%s\n", mode == 2 ? "" : ",");
    }

    fprintf(out, "    }\n");
    fprintf(out, "};\n\n");
}
