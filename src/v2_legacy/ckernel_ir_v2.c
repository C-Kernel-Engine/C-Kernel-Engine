#include "ckernel_ir_v2.h"
#include "ckernel_mem_plan.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ck_ir_v2_free_buffer(CKIRV2Buffer *buf)
{
    if (!buf) {
        return;
    }
    free(buf->name);
    free(buf->alias_of);
    free(buf->condition);
    memset(buf, 0, sizeof(*buf));
}

static void ck_ir_v2_free_node(CKIRV2Node *node)
{
    if (!node) {
        return;
    }
    for (int i = 0; i < node->n_bindings; ++i) {
        free(node->bindings[i].arg);
    }
    free(node->op);
    free(node->kernel);
    free(node->condition);
    memset(node, 0, sizeof(*node));
}

void ck_ir_v2_free(CKIRV2Graph *graph)
{
    if (!graph) {
        return;
    }
    if (graph->buffers) {
        for (int i = 0; i < graph->num_buffers; ++i) {
            ck_ir_v2_free_buffer(&graph->buffers[i]);
        }
        free(graph->buffers);
    }
    if (graph->nodes) {
        for (int i = 0; i < graph->num_nodes; ++i) {
            ck_ir_v2_free_node(&graph->nodes[i]);
        }
        free(graph->nodes);
    }
    memset(graph, 0, sizeof(*graph));
}

static const char *ck_ir_v2_scope_name(CKBufferScope scope)
{
    switch (scope) {
    case CK_SCOPE_LAYER:
        return "layer";
    case CK_SCOPE_GLOBAL:
        return "global";
    default:
        return "unknown";
    }
}

static const char *ck_ir_v2_role_name(CKBufferRole role)
{
    switch (role) {
    case CK_ROLE_INPUT:
        return "input";
    case CK_ROLE_OUTPUT:
        return "output";
    case CK_ROLE_ACTIVATION:
        return "activation";
    case CK_ROLE_WEIGHT:
        return "weight";
    case CK_ROLE_SCRATCH:
        return "scratch";
    case CK_ROLE_GRAD:
        return "grad";
    default:
        return "unknown";
    }
}

static const char *ck_ir_v2_dtype_name(CKDataType dtype)
{
    switch (dtype) {
    case CK_DT_FP32:
        return "fp32";
    case CK_DT_BF16:
        return "bf16";
    case CK_DT_FP16:
        return "fp16";
    case CK_DT_Q4_0:
        return "q4_0";
    case CK_DT_Q4_K:
        return "q4_k";
    case CK_DT_Q6_K:
        return "q6_k";
    case CK_DT_Q8_0:
        return "q8_0";
    default:
        return "unknown";
    }
}

static const char *ck_ir_v2_dim_name(CKDimKind dim)
{
    switch (dim) {
    case CK_DIM_TOKENS:
        return "tokens";
    case CK_DIM_EMBED:
        return "embed";
    case CK_DIM_ALIGNED_EMBED:
        return "aligned_embed";
    case CK_DIM_HEAD_DIM:
        return "head_dim";
    case CK_DIM_ALIGNED_HEAD:
        return "aligned_head";
    case CK_DIM_NUM_HEADS:
        return "num_heads";
    case CK_DIM_NUM_KV_HEADS:
        return "num_kv_heads";
    case CK_DIM_ALIGNED_CTX:
        return "aligned_ctx";
    case CK_DIM_INTERMEDIATE:
        return "intermediate";
    case CK_DIM_ALIGNED_INTERMEDIATE:
        return "aligned_intermediate";
    case CK_DIM_VOCAB:
        return "vocab";
    default:
        return "unknown";
    }
}

static CKDimKind ck_ir_v2_dim_kind_from_name(const char *name)
{
    if (!name) {
        return CK_DIM_END;
    }
    if (strcmp(name, "tokens") == 0) return CK_DIM_TOKENS;
    if (strcmp(name, "embed") == 0) return CK_DIM_EMBED;
    if (strcmp(name, "aligned_embed") == 0) return CK_DIM_ALIGNED_EMBED;
    if (strcmp(name, "head_dim") == 0) return CK_DIM_HEAD_DIM;
    if (strcmp(name, "aligned_head") == 0) return CK_DIM_ALIGNED_HEAD;
    if (strcmp(name, "num_heads") == 0) return CK_DIM_NUM_HEADS;
    if (strcmp(name, "num_kv_heads") == 0) return CK_DIM_NUM_KV_HEADS;
    if (strcmp(name, "aligned_ctx") == 0) return CK_DIM_ALIGNED_CTX;
    if (strcmp(name, "intermediate") == 0) return CK_DIM_INTERMEDIATE;
    if (strcmp(name, "aligned_intermediate") == 0) return CK_DIM_ALIGNED_INTERMEDIATE;
    if (strcmp(name, "vocab") == 0) return CK_DIM_VOCAB;
    return CK_DIM_END;
}

typedef struct {
    size_t aligned_embed;
    size_t aligned_head;
    size_t aligned_intermediate;
    size_t aligned_context;
} CKIRV2AlignInfo;

static size_t ck_ir_v2_align_up_bytes(size_t n, size_t align)
{
    if (align == 0) {
        return n;
    }
    return (n + align - 1) & ~(align - 1);
}

static size_t ck_ir_v2_align_up_elems(size_t elems, size_t elem_bytes, size_t align_bytes)
{
    size_t bytes = elems * elem_bytes;
    bytes = ck_ir_v2_align_up_bytes(bytes, align_bytes);
    return bytes / elem_bytes;
}

static void ck_ir_v2_resolve_align(const CKModelConfig *cfg,
                                   size_t alignment_bytes,
                                   CKIRV2AlignInfo *align)
{
    if (!align) {
        return;
    }
    memset(align, 0, sizeof(*align));
    if (!cfg) {
        return;
    }
    if (alignment_bytes == 0) {
        alignment_bytes = CK_MEM_PLAN_DEFAULT_ALIGN;
    }
    size_t elem_bytes = sizeof(float);
    size_t head_dim = (cfg->num_heads > 0) ? (size_t)(cfg->hidden_size / cfg->num_heads) : 0;
    align->aligned_embed = ck_ir_v2_align_up_elems((size_t)cfg->hidden_size, elem_bytes, alignment_bytes);
    align->aligned_head = ck_ir_v2_align_up_elems(head_dim, elem_bytes, alignment_bytes);
    align->aligned_intermediate = ck_ir_v2_align_up_elems((size_t)cfg->intermediate_size,
                                                          elem_bytes, alignment_bytes);
    align->aligned_context = ck_ir_v2_align_up_elems((size_t)cfg->context_window, elem_bytes, alignment_bytes);
}

static size_t ck_ir_v2_resolve_dim_value(const CKModelConfig *cfg,
                                         const CKIRV2AlignInfo *align,
                                         CKDimKind dim,
                                         int tokens_override)
{
    switch (dim) {
    case CK_DIM_TOKENS:
        if (tokens_override >= 0) {
            return (size_t)tokens_override;
        }
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

static int ck_ir_v2_emit_shape(FILE *out, const CKDimToken *shape)
{
    fprintf(out, "[");
    int first = 1;
    for (int i = 0; i < CK_IR_V2_MAX_DIMS; ++i) {
        if (shape[i].dim == CK_DIM_END) {
            break;
        }
        if (!first) {
            fprintf(out, ", ");
        }
        fprintf(out, "{\"dim\":\"%s\",\"dim_id\":%d,\"mult\":%d,\"div\":%d}",
                ck_ir_v2_dim_name(shape[i].dim),
                (int)shape[i].dim,
                shape[i].mult,
                shape[i].div);
        first = 0;
    }
    fprintf(out, "]");
    return 0;
}

static void ck_ir_v2_emit_resolved_shape(FILE *out,
                                         const CKModelConfig *cfg,
                                         const CKIRV2AlignInfo *align,
                                         const CKDimToken *shape,
                                         int tokens_override)
{
    fprintf(out, "[");
    int first = 1;
    for (int i = 0; i < CK_IR_V2_MAX_DIMS; ++i) {
        if (shape[i].dim == CK_DIM_END) {
            break;
        }
        size_t dim = ck_ir_v2_resolve_dim_value(cfg, align, shape[i].dim, tokens_override);
        size_t mult = (size_t)(shape[i].mult > 0 ? shape[i].mult : 1);
        size_t div = (size_t)(shape[i].div > 0 ? shape[i].div : 1);
        size_t resolved = (div == 0) ? 0 : (dim * mult / div);
        if (!first) {
            fprintf(out, ", ");
        }
        fprintf(out, "%zu", resolved);
        first = 0;
    }
    fprintf(out, "]");
}

static void ck_ir_v2_emit_dimensions(FILE *out,
                                     const CKModelConfig *cfg,
                                     const CKIRV2AlignInfo *align,
                                     int tokens_override)
{
    fprintf(out, "  \"dimensions\": [\n");
    CKDimKind dims[] = {
        CK_DIM_TOKENS,
        CK_DIM_EMBED,
        CK_DIM_ALIGNED_EMBED,
        CK_DIM_HEAD_DIM,
        CK_DIM_ALIGNED_HEAD,
        CK_DIM_NUM_HEADS,
        CK_DIM_NUM_KV_HEADS,
        CK_DIM_ALIGNED_CTX,
        CK_DIM_INTERMEDIATE,
        CK_DIM_ALIGNED_INTERMEDIATE,
        CK_DIM_VOCAB
    };
    size_t dim_count = sizeof(dims) / sizeof(dims[0]);
    for (size_t i = 0; i < dim_count; ++i) {
        CKDimKind dim = dims[i];
        size_t value = ck_ir_v2_resolve_dim_value(cfg, align, dim, tokens_override);
        fprintf(out,
                "    {\"id\": %d, \"name\": \"%s\", \"value\": %zu}%s\n",
                (int)dim,
                ck_ir_v2_dim_name(dim),
                value,
                (i + 1 == dim_count) ? "" : ",");
    }
    fprintf(out, "  ],\n");
}

static const char *ck_ir_v2_mem_arena_name(CKMemArenaKind arena)
{
    switch (arena) {
    case CK_MEM_ARENA_WEIGHTS:
        return "weights";
    case CK_MEM_ARENA_ACTIVATIONS:
        return "activations";
    case CK_MEM_ARENA_GRADS:
        return "grads";
    default:
        return "unknown";
    }
}

static void ck_ir_v2_emit_memory_plan(FILE *out,
                                      const CKIRV2Graph *graph,
                                      const CKMemPlan *plan)
{
    if (!plan) {
        return;
    }
    fprintf(out, "  \"memory_plan\": {\n");
    fprintf(out, "    \"alignment_bytes\": %zu,\n", plan->alignment_bytes);
    fprintf(out, "    \"total_bytes\": {\n");
    fprintf(out, "      \"weights\": %zu,\n", plan->total_bytes[CK_MEM_ARENA_WEIGHTS]);
    fprintf(out, "      \"activations\": %zu,\n", plan->total_bytes[CK_MEM_ARENA_ACTIVATIONS]);
    fprintf(out, "      \"grads\": %zu\n", plan->total_bytes[CK_MEM_ARENA_GRADS]);
    fprintf(out, "    },\n");
    fprintf(out, "    \"buffers\": [\n");
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKMemSpan *span = &plan->spans[i];
        const char *name = graph->buffers[i].name ? graph->buffers[i].name : "";
        int enabled = span->size_bytes > 0;
        fprintf(out,
                "      {\"name\": \"%s\", \"arena\": \"%s\", \"offset_bytes\": %zu, \"size_bytes\": %zu, \"enabled\": %s}%s\n",
                name,
                ck_ir_v2_mem_arena_name(span->arena),
                span->offset_bytes,
                span->size_bytes,
                enabled ? "true" : "false",
                (i + 1 == graph->num_buffers) ? "" : ",");
    }
    fprintf(out, "    ]\n");
    fprintf(out, "  },\n");
}

static int ck_ir_v2_serialize_json_internal(const CKIRV2Graph *graph,
                                            const CKMemPlan *plan,
                                            const char *mode,
                                            int tokens_override,
                                            int base_context_window,
                                            const char *path)
{
    if (!graph || !path) {
        return -1;
    }
    FILE *out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ck_ir_v2_serialize_json: failed to open %s: %s\n",
                path, strerror(errno));
        return -1;
    }

    CKIRV2AlignInfo align = {0};
    ck_ir_v2_resolve_align(&graph->config, CK_MEM_PLAN_DEFAULT_ALIGN, &align);

    fprintf(out, "{\n");
    fprintf(out, "  \"version\": 2,\n");
    fprintf(out, "  \"notes\": [\n");
    fprintf(out, "    \"shape.dim uses symbolic names; see dimensions for resolved values\",\n");
    fprintf(out,
            "    \"resolved_shape applies mult/div using alignment_bytes=%d and elem_bytes=4\",\n",
            CK_MEM_PLAN_DEFAULT_ALIGN);
    fprintf(out, "    \"kernel is the selected impl; kernel_dtype records dtype selection\"\n");
    fprintf(out, "  ],\n");
    fprintf(out, "  \"config\": {\n");
    fprintf(out, "    \"num_layers\": %d,\n", graph->config.num_layers);
    fprintf(out, "    \"hidden_size\": %d,\n", graph->config.hidden_size);
    fprintf(out, "    \"intermediate_size\": %d,\n", graph->config.intermediate_size);
    fprintf(out, "    \"num_attention_heads\": %d,\n", graph->config.num_heads);
    fprintf(out, "    \"num_kv_heads\": %d,\n", graph->config.num_kv_heads);
    fprintf(out, "    \"vocab_size\": %d,\n", graph->config.vocab_size);
    fprintf(out, "    \"context_window\": %d,\n", graph->config.context_window);
    fprintf(out, "    \"rms_norm_eps\": %.9g,\n", graph->config.rms_norm_eps);
    fprintf(out, "    \"rope_theta\": %.9g\n", graph->config.rope_theta);
    fprintf(out, "  },\n");

    ck_ir_v2_emit_dimensions(out, &graph->config, &align, tokens_override);

    fprintf(out, "  \"meta\": {\n");
    fprintf(out, "    \"has_pos_emb\": %s,\n", graph->has_pos_emb ? "true" : "false");
    if (graph->tie_word_embeddings < 0) {
        fprintf(out, "    \"tie_word_embeddings\": null,\n");
    } else {
        fprintf(out, "    \"tie_word_embeddings\": %s,\n", graph->tie_word_embeddings ? "true" : "false");
    }
    if (graph->fused_qkv < 0) {
        fprintf(out, "    \"fused_qkv\": null,\n");
    } else {
        fprintf(out, "    \"fused_qkv\": %s,\n", graph->fused_qkv ? "true" : "false");
    }
    if (graph->gated_mlp < 0) {
        fprintf(out, "    \"gated_mlp\": null\n");
    } else {
        fprintf(out, "    \"gated_mlp\": %s\n", graph->gated_mlp ? "true" : "false");
    }
    fprintf(out, "  },\n");

    if (plan) {
        int training = (mode && strcmp(mode, "backward") == 0);
        int tokens = (tokens_override >= 0) ? tokens_override : graph->config.context_window;
        fprintf(out, "  \"lowering\": {\n");
        fprintf(out, "    \"mode\": \"%s\",\n", mode ? mode : "unknown");
        fprintf(out, "    \"training\": %s,\n", training ? "true" : "false");
        fprintf(out, "    \"tokens\": %d", tokens);
        if (base_context_window >= 0) {
            fprintf(out, ",\n    \"base_context_window\": %d\n", base_context_window);
        } else {
            fprintf(out, "\n");
        }
        fprintf(out, "  },\n");
        ck_ir_v2_emit_memory_plan(out, graph, plan);
    }

    fprintf(out, "  \"buffers\": [\n");
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        fprintf(out, "    {\n");
        fprintf(out, "      \"name\": \"%s\",\n", buf->name ? buf->name : "");
        fprintf(out, "      \"scope\": \"%s\",\n", ck_ir_v2_scope_name(buf->scope));
        fprintf(out, "      \"role\": \"%s\",\n", ck_ir_v2_role_name(buf->role));
        fprintf(out, "      \"dtype\": \"%s\",\n", ck_ir_v2_dtype_name(buf->dtype));
        fprintf(out, "      \"optional\": %d,\n", buf->optional ? 1 : 0);
        fprintf(out, "      \"shape\": ");
        ck_ir_v2_emit_shape(out, buf->shape);
        fprintf(out, ",\n");
        fprintf(out, "      \"resolved_shape\": ");
        ck_ir_v2_emit_resolved_shape(out, &graph->config, &align, buf->shape,
                                     tokens_override);
        fprintf(out, ",\n");
        if (buf->alias_of) {
            fprintf(out, "      \"alias_of\": \"%s\",\n", buf->alias_of);
        } else {
            fprintf(out, "      \"alias_of\": null,\n");
        }
        if (buf->condition) {
            fprintf(out, "      \"condition\": \"%s\"\n", buf->condition);
        } else {
            fprintf(out, "      \"condition\": null\n");
        }
        fprintf(out, "    }%s\n", (i + 1 == graph->num_buffers) ? "" : ",");
    }
    fprintf(out, "  ],\n");

    fprintf(out, "  \"nodes\": [\n");
    for (int i = 0; i < graph->num_nodes; ++i) {
        const CKIRV2Node *node = &graph->nodes[i];
        fprintf(out, "    {\n");
        fprintf(out, "      \"layer\": %d,\n", (int)node->layer);
        fprintf(out, "      \"op\": \"%s\",\n", node->op ? node->op : "");
        fprintf(out, "      \"kernel\": \"%s\",\n", node->kernel ? node->kernel : "");
        fprintf(out, "      \"kernel_variant\": \"%s\",\n", node->kernel ? node->kernel : "");
        fprintf(out, "      \"kernel_dtype\": \"%s\",\n", ck_ir_v2_dtype_name(node->kernel_dtype));
        fprintf(out, "      \"flags\": %u,\n", (unsigned)node->flags);
        if (node->condition) {
            fprintf(out, "      \"condition\": \"%s\",\n", node->condition);
        } else {
            fprintf(out, "      \"condition\": null,\n");
        }
        fprintf(out, "      \"bindings\": [\n");
        for (int b = 0; b < node->n_bindings; ++b) {
            const CKIRV2Binding *bind = &node->bindings[b];
            const char *buf_name = "";
            if (bind->buffer >= 0 && bind->buffer < graph->num_buffers) {
                buf_name = graph->buffers[bind->buffer].name ? graph->buffers[bind->buffer].name : "";
            }
            fprintf(out,
                    "        {\"arg\": \"%s\", \"buffer\": \"%s\"}%s\n",
                    bind->arg ? bind->arg : "",
                    buf_name,
                    (b + 1 == node->n_bindings) ? "" : ",");
        }
        fprintf(out, "      ]\n");
        fprintf(out, "    }%s\n", (i + 1 == graph->num_nodes) ? "" : ",");
    }
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");

    fclose(out);
    return 0;
}

int ck_ir_v2_serialize_json(const CKIRV2Graph *graph, const char *path)
{
    return ck_ir_v2_serialize_json_internal(graph, NULL, NULL, -1, -1, path);
}

int ck_ir_v2_serialize_json_with_plan(const CKIRV2Graph *graph,
                                      const CKMemPlan *plan,
                                      const char *mode,
                                      int tokens_override,
                                      int base_context_window,
                                      const char *path)
{
    return ck_ir_v2_serialize_json_internal(graph, plan, mode,
                                            tokens_override,
                                            base_context_window,
                                            path);
}

static int ck_ir_v2_parse_string(const char *start,
                                 const char *end,
                                 char **out_str)
{
    if (!start || !out_str || start >= end || *start != '"') {
        return -1;
    }
    const char *cur = start + 1;
    while (cur < end && *cur != '"') {
        if (*cur == '\\' && (cur + 1) < end) {
            cur += 2;
            continue;
        }
        cur++;
    }
    if (cur >= end || *cur != '"') {
        return -1;
    }
    size_t len = (size_t)(cur - (start + 1));
    char *buf = (char *)malloc(len + 1);
    if (!buf) {
        return -1;
    }
    memcpy(buf, start + 1, len);
    buf[len] = '\0';
    *out_str = buf;
    return 0;
}

static const char *ck_ir_v2_skip_string(const char *cur, const char *end)
{
    if (!cur || cur >= end || *cur != '"') {
        return cur;
    }
    cur++;
    while (cur < end) {
        if (*cur == '\\' && (cur + 1) < end) {
            cur += 2;
            continue;
        }
        if (*cur == '"') {
            return cur + 1;
        }
        cur++;
    }
    return end;
}

static const char *ck_ir_v2_next_object(const char *cur,
                                        const char *end,
                                        const char **obj_start,
                                        const char **obj_end)
{
    int depth = 0;
    const char *start = NULL;
    for (const char *p = cur; p < end; ++p) {
        if (*p == '"') {
            p = ck_ir_v2_skip_string(p, end) - 1;
            continue;
        }
        if (*p == '{') {
            if (depth == 0) {
                start = p;
            }
            depth++;
            continue;
        }
        if (*p == '}') {
            if (depth > 0) {
                depth--;
                if (depth == 0 && start) {
                    *obj_start = start;
                    *obj_end = p;
                    return p + 1;
                }
            }
        }
    }
    return NULL;
}

static const char *ck_ir_v2_find_array_end(const char *open, const char *end)
{
    if (!open || open >= end || *open != '[') {
        return NULL;
    }
    int depth = 0;
    for (const char *p = open; p < end; ++p) {
        if (*p == '"') {
            p = ck_ir_v2_skip_string(p, end) - 1;
            continue;
        }
        if (*p == '[') {
            depth++;
            continue;
        }
        if (*p == ']') {
            if (depth > 0) {
                depth--;
                if (depth == 0) {
                    return p;
                }
            }
        }
    }
    return NULL;
}

static const char *ck_ir_v2_find_key(const char *json,
                                     const char *key,
                                     const char *end)
{
    size_t key_len = strlen(key);
    const char *cur = json;
    while (cur + key_len < end) {
        if (memcmp(cur, key, key_len) == 0) {
            return cur;
        }
        cur++;
    }
    return NULL;
}

static int ck_ir_v2_parse_int(const char *json,
                              const char *key,
                              const char *end,
                              int *out_val)
{
    const char *p = ck_ir_v2_find_key(json, key, end);
    if (!p) {
        return -1;
    }
    const char *colon = strchr(p, ':');
    if (!colon || colon >= end) {
        return -1;
    }
    int value = 0;
    if (sscanf(colon + 1, "%d", &value) != 1) {
        return -1;
    }
    *out_val = value;
    return 0;
}

static int ck_ir_v2_parse_bool(const char *json,
                               const char *key,
                               const char *end,
                               int *out_val)
{
    const char *p = ck_ir_v2_find_key(json, key, end);
    if (!p) {
        return -1;
    }
    const char *colon = strchr(p, ':');
    if (!colon || colon >= end) {
        return -1;
    }
    const char *cur = colon + 1;
    while (cur < end && (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r')) {
        cur++;
    }
    if (cur + 4 <= end && memcmp(cur, "true", 4) == 0) {
        *out_val = 1;
        return 0;
    }
    if (cur + 5 <= end && memcmp(cur, "false", 5) == 0) {
        *out_val = 0;
        return 0;
    }
    return -1;
}

static int ck_ir_v2_parse_float(const char *json,
                                const char *key,
                                const char *end,
                                float *out_val)
{
    const char *p = ck_ir_v2_find_key(json, key, end);
    if (!p) {
        return -1;
    }
    const char *colon = strchr(p, ':');
    if (!colon || colon >= end) {
        return -1;
    }
    float value = 0.0f;
    if (sscanf(colon + 1, "%f", &value) != 1) {
        return -1;
    }
    *out_val = value;
    return 0;
}

static int ck_ir_v2_parse_string_field(const char *json,
                                       const char *key,
                                       const char *end,
                                       char **out_str)
{
    const char *p = ck_ir_v2_find_key(json, key, end);
    if (!p) {
        return -1;
    }
    const char *colon = strchr(p, ':');
    if (!colon || colon >= end) {
        return -1;
    }
    const char *cur = colon + 1;
    while (cur < end && (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r')) {
        cur++;
    }
    if (cur >= end) {
        return -1;
    }
    if (strncmp(cur, "null", 4) == 0) {
        *out_str = NULL;
        return 0;
    }
    if (*cur != '"') {
        return -1;
    }
    return ck_ir_v2_parse_string(cur, end, out_str);
}

static CKBufferScope ck_ir_v2_parse_scope(const char *s)
{
    if (!s) return CK_SCOPE_GLOBAL;
    if (strcmp(s, "layer") == 0) return CK_SCOPE_LAYER;
    if (strcmp(s, "global") == 0) return CK_SCOPE_GLOBAL;
    return CK_SCOPE_GLOBAL;
}

static CKBufferRole ck_ir_v2_parse_role(const char *s)
{
    if (!s) return CK_ROLE_ACTIVATION;
    if (strcmp(s, "input") == 0) return CK_ROLE_INPUT;
    if (strcmp(s, "output") == 0) return CK_ROLE_OUTPUT;
    if (strcmp(s, "activation") == 0) return CK_ROLE_ACTIVATION;
    if (strcmp(s, "weight") == 0) return CK_ROLE_WEIGHT;
    if (strcmp(s, "scratch") == 0) return CK_ROLE_SCRATCH;
    if (strcmp(s, "grad") == 0) return CK_ROLE_GRAD;
    return CK_ROLE_ACTIVATION;
}

static CKDataType ck_ir_v2_parse_dtype(const char *s)
{
    if (!s) return CK_DT_FP32;
    if (strcmp(s, "fp32") == 0) return CK_DT_FP32;
    if (strcmp(s, "bf16") == 0) return CK_DT_BF16;
    if (strcmp(s, "fp16") == 0) return CK_DT_FP16;
    if (strcmp(s, "q4_0") == 0) return CK_DT_Q4_0;
    if (strcmp(s, "q4_k") == 0) return CK_DT_Q4_K;
    if (strcmp(s, "q6_k") == 0) return CK_DT_Q6_K;
    if (strcmp(s, "q8_0") == 0) return CK_DT_Q8_0;
    return CK_DT_FP32;
}

static CKDimKind ck_ir_v2_parse_dim_kind(const char *obj_start,
                                         const char *obj_end)
{
    char *dim_str = NULL;
    CKDimKind kind = CK_DIM_END;
    if (ck_ir_v2_parse_string_field(obj_start, "\"dim\"", obj_end, &dim_str) == 0 &&
        dim_str) {
        kind = ck_ir_v2_dim_kind_from_name(dim_str);
    }
    if (dim_str) {
        free(dim_str);
    }
    if (kind != CK_DIM_END) {
        return kind;
    }
    int dim_id = -1;
    if (ck_ir_v2_parse_int(obj_start, "\"dim_id\"", obj_end, &dim_id) == 0) {
        return (CKDimKind)dim_id;
    }
    if (ck_ir_v2_parse_int(obj_start, "\"dim\"", obj_end, &dim_id) == 0) {
        return (CKDimKind)dim_id;
    }
    return CK_DIM_END;
}

static int ck_ir_v2_parse_shape(const char *obj_start,
                                const char *obj_end,
                                CKDimToken *shape_out)
{
    if (!shape_out) {
        return -1;
    }
    for (int i = 0; i < CK_IR_V2_MAX_DIMS; ++i) {
        shape_out[i].dim = CK_DIM_END;
        shape_out[i].mult = 0;
        shape_out[i].div = 0;
    }

    const char *start = ck_ir_v2_find_key(obj_start, "\"shape\"", obj_end);
    if (!start) {
        return -1;
    }
    const char *open = strchr(start, '[');
    const char *close = ck_ir_v2_find_array_end(open, obj_end);
    if (!open || !close) {
        return -1;
    }

    const char *cur = open;
    const char *sstart = NULL;
    const char *send = NULL;
    int idx = 0;
    while (idx < CK_IR_V2_MAX_DIMS &&
           (cur = ck_ir_v2_next_object(cur, close, &sstart, &send)) != NULL) {
        CKDimKind dim = ck_ir_v2_parse_dim_kind(sstart, send);
        if (dim == CK_DIM_END) {
            continue;
        }
        int mult = 1;
        int div = 1;
        ck_ir_v2_parse_int(sstart, "\"mult\"", send, &mult);
        ck_ir_v2_parse_int(sstart, "\"div\"", send, &div);
        shape_out[idx].dim = dim;
        shape_out[idx].mult = mult;
        shape_out[idx].div = div;
        idx++;
    }
    return (idx > 0) ? 0 : -1;
}

static const CKBufferSpec *ck_ir_v2_find_buffer_spec(const char *name)
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

static int ck_ir_v2_find_buffer_index(const CKIRV2Graph *graph, const char *name)
{
    if (!graph || !name) {
        return -1;
    }
    for (int i = 0; i < graph->num_buffers; ++i) {
        if (graph->buffers[i].name && strcmp(graph->buffers[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static int ck_ir_v2_parse_buffers(const char *json,
                                  const char *end,
                                  CKIRV2Graph *graph)
{
    const char *start = ck_ir_v2_find_key(json, "\"buffers\"", end);
    if (!start) {
        return -1;
    }
    const char *open = strchr(start, '[');
    const char *close = ck_ir_v2_find_array_end(open, end);
    if (!open || !close) {
        return -1;
    }

    int count = 0;
    const char *cur = open;
    const char *obj_start = NULL;
    const char *obj_end = NULL;
    while ((cur = ck_ir_v2_next_object(cur, close, &obj_start, &obj_end)) != NULL) {
        count++;
    }
    if (count <= 0) {
        return -1;
    }

    CKIRV2Buffer *buffers = (CKIRV2Buffer *)calloc((size_t)count, sizeof(CKIRV2Buffer));
    if (!buffers) {
        return -1;
    }
    int idx = 0;
    cur = open;
    while (idx < count &&
           (cur = ck_ir_v2_next_object(cur, close, &obj_start, &obj_end)) != NULL) {

        CKIRV2Buffer buf = {0};
        char *name = NULL;
        char *scope = NULL;
        char *role = NULL;
        char *dtype = NULL;
        char *alias = NULL;
        char *cond = NULL;
        int optional = 0;

        ck_ir_v2_parse_string_field(obj_start, "\"name\"", obj_end, &name);
        ck_ir_v2_parse_string_field(obj_start, "\"scope\"", obj_end, &scope);
        ck_ir_v2_parse_string_field(obj_start, "\"role\"", obj_end, &role);
        ck_ir_v2_parse_string_field(obj_start, "\"dtype\"", obj_end, &dtype);
        ck_ir_v2_parse_string_field(obj_start, "\"alias_of\"", obj_end, &alias);
        ck_ir_v2_parse_string_field(obj_start, "\"condition\"", obj_end, &cond);
        ck_ir_v2_parse_int(obj_start, "\"optional\"", obj_end, &optional);

        buf.name = name;
        buf.scope = ck_ir_v2_parse_scope(scope);
        buf.role = ck_ir_v2_parse_role(role);
        buf.dtype = ck_ir_v2_parse_dtype(dtype);
        buf.optional = optional;
        buf.alias_of = alias;
        buf.condition = cond;

        if (ck_ir_v2_parse_shape(obj_start, obj_end, buf.shape) != 0) {
            const CKBufferSpec *spec = ck_ir_v2_find_buffer_spec(name);
            if (spec) {
                memcpy(buf.shape, spec->shape, sizeof(buf.shape));
            } else {
                for (int i = 0; i < CK_IR_V2_MAX_DIMS; ++i) {
                    buf.shape[i].dim = CK_DIM_END;
                    buf.shape[i].mult = 0;
                    buf.shape[i].div = 0;
                }
            }
        }

        free(scope);
        free(role);
        free(dtype);

        buffers[idx++] = buf;
    }

    graph->buffers = buffers;
    graph->num_buffers = idx;
    return 0;
}

static int ck_ir_v2_parse_bindings(const char *obj_start,
                                   const char *obj_end,
                                   CKIRV2Graph *graph,
                                   CKIRV2Node *node)
{
    const char *start = ck_ir_v2_find_key(obj_start, "\"bindings\"", obj_end);
    if (!start) {
        return 0;
    }
    const char *open = strchr(start, '[');
    const char *close = ck_ir_v2_find_array_end(open, obj_end);
    if (!open || !close || close > obj_end) {
        return -1;
    }

    const char *cur = open;
    const char *bstart = NULL;
    const char *bend = NULL;
    while (node->n_bindings < CK_IR_V2_MAX_BINDINGS &&
           (cur = ck_ir_v2_next_object(cur, close, &bstart, &bend)) != NULL) {

        char *arg = NULL;
        char *buf = NULL;
        ck_ir_v2_parse_string_field(bstart, "\"arg\"", bend, &arg);
        ck_ir_v2_parse_string_field(bstart, "\"buffer\"", bend, &buf);

        if (arg) {
            int buf_idx = ck_ir_v2_find_buffer_index(graph, buf);
            node->bindings[node->n_bindings].arg = arg;
            node->bindings[node->n_bindings].buffer = buf_idx;
            node->n_bindings++;
        } else {
            free(arg);
        }
        free(buf);
    }

    return 0;
}

static int ck_ir_v2_parse_nodes(const char *json,
                                const char *end,
                                CKIRV2Graph *graph)
{
    const char *start = ck_ir_v2_find_key(json, "\"nodes\"", end);
    if (!start) {
        return -1;
    }
    const char *open = strchr(start, '[');
    const char *close = ck_ir_v2_find_array_end(open, end);
    if (!open || !close) {
        return -1;
    }

    int count = 0;
    const char *cur = open;
    const char *obj_start = NULL;
    const char *obj_end = NULL;
    while ((cur = ck_ir_v2_next_object(cur, close, &obj_start, &obj_end)) != NULL) {
        count++;
    }
    if (count <= 0) {
        return -1;
    }

    CKIRV2Node *nodes = (CKIRV2Node *)calloc((size_t)count, sizeof(CKIRV2Node));
    if (!nodes) {
        return -1;
    }
    int idx = 0;
    cur = open;
    while (idx < count &&
           (cur = ck_ir_v2_next_object(cur, close, &obj_start, &obj_end)) != NULL) {

        CKIRV2Node node = {0};
        char *op = NULL;
        char *kernel = NULL;
        char *kernel_variant = NULL;
        char *kernel_dtype = NULL;
        char *cond = NULL;
        int layer = 0;
        int flags = 0;

        ck_ir_v2_parse_string_field(obj_start, "\"op\"", obj_end, &op);
        ck_ir_v2_parse_string_field(obj_start, "\"kernel\"", obj_end, &kernel);
        ck_ir_v2_parse_string_field(obj_start, "\"kernel_variant\"", obj_end, &kernel_variant);
        ck_ir_v2_parse_string_field(obj_start, "\"kernel_dtype\"", obj_end, &kernel_dtype);
        ck_ir_v2_parse_string_field(obj_start, "\"condition\"", obj_end, &cond);
        ck_ir_v2_parse_int(obj_start, "\"layer\"", obj_end, &layer);
        ck_ir_v2_parse_int(obj_start, "\"flags\"", obj_end, &flags);

        if (!kernel && kernel_variant) {
            kernel = kernel_variant;
            kernel_variant = NULL;
        }

        node.op = op;
        node.kernel = kernel;
        node.kernel_dtype = ck_ir_v2_parse_dtype(kernel_dtype);
        node.condition = cond;
        node.layer = (uint16_t)layer;
        node.flags = (uint8_t)flags;
        node.n_bindings = 0;
        node.n_inputs = 0;
        node.n_outputs = 0;

        if (kernel_variant) {
            free(kernel_variant);
        }
        if (kernel_dtype) {
            free(kernel_dtype);
        }

        if (ck_ir_v2_parse_bindings(obj_start, obj_end, graph, &node) != 0) {
            free(node.op);
            free(node.kernel);
            free(node.condition);
            for (int b = 0; b < node.n_bindings; ++b) {
                free(node.bindings[b].arg);
            }
            for (int j = 0; j < idx; ++j) {
                ck_ir_v2_free_node(&nodes[j]);
            }
            free(nodes);
            return -1;
        }

        nodes[idx++] = node;
    }

    graph->nodes = nodes;
    graph->num_nodes = idx;
    return 0;
}

int ck_ir_v2_parse_json(const char *path, CKIRV2Graph *graph)
{
    if (!path || !graph) {
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("ck_ir_v2_parse_json: fopen");
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long len = ftell(f);
    if (len < 0) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }
    size_t nread = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[nread] = '\0';

    CKIRV2Graph tmp = {0};
    const char *end = buf + nread;

    tmp.has_pos_emb = 1;
    tmp.tie_word_embeddings = -1;
    tmp.fused_qkv = -1;
    tmp.gated_mlp = -1;

    if (ck_ir_v2_parse_int(buf, "\"num_layers\"", end, &tmp.config.num_layers) != 0 ||
        ck_ir_v2_parse_int(buf, "\"hidden_size\"", end, &tmp.config.hidden_size) != 0 ||
        ck_ir_v2_parse_int(buf, "\"intermediate_size\"", end, &tmp.config.intermediate_size) != 0 ||
        ck_ir_v2_parse_int(buf, "\"num_attention_heads\"", end, &tmp.config.num_heads) != 0 ||
        ck_ir_v2_parse_int(buf, "\"num_kv_heads\"", end, &tmp.config.num_kv_heads) != 0) {
        free(buf);
        return -1;
    }

    ck_ir_v2_parse_int(buf, "\"vocab_size\"", end, &tmp.config.vocab_size);
    ck_ir_v2_parse_int(buf, "\"context_window\"", end, &tmp.config.context_window);
    ck_ir_v2_parse_float(buf, "\"rms_norm_eps\"", end, &tmp.config.rms_norm_eps);
    ck_ir_v2_parse_float(buf, "\"rope_theta\"", end, &tmp.config.rope_theta);
    const char *meta_key = ck_ir_v2_find_key(buf, "\"meta\"", end);
    if (meta_key) {
        const char *brace = strchr(meta_key, '{');
        const char *obj_start = NULL;
        const char *obj_end = NULL;
        if (brace && ck_ir_v2_next_object(brace, end, &obj_start, &obj_end)) {
            ck_ir_v2_parse_bool(obj_start, "\"has_pos_emb\"", obj_end, &tmp.has_pos_emb);
            ck_ir_v2_parse_bool(obj_start, "\"tie_word_embeddings\"", obj_end, &tmp.tie_word_embeddings);
            ck_ir_v2_parse_bool(obj_start, "\"fused_qkv\"", obj_end, &tmp.fused_qkv);
            ck_ir_v2_parse_bool(obj_start, "\"gated_mlp\"", obj_end, &tmp.gated_mlp);
        }
    }

    if (ck_ir_v2_parse_buffers(buf, end, &tmp) != 0 ||
        ck_ir_v2_parse_nodes(buf, end, &tmp) != 0) {
        ck_ir_v2_free(&tmp);
        free(buf);
        return -1;
    }

    *graph = tmp;
    free(buf);
    return 0;
}
