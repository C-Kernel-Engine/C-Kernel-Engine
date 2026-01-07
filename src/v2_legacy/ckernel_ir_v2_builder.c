#include "ckernel_ir_v2.h"
#include "ckernel_kernel_specs.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static char *ck_ir_v2_strdup(const char *s)
{
    if (!s) {
        return NULL;
    }
    size_t len = strlen(s);
    char *out = (char *)malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len + 1);
    return out;
}

static void ck_ir_v2_copy_shape(CKDimToken *dst, const CKDimToken *src)
{
    memcpy(dst, src, sizeof(CKDimToken) * CK_IR_V2_MAX_DIMS);
}

static int ck_ir_v2_copy_buffer_spec(const CKBufferSpec *spec, CKIRV2Buffer *out)
{
    if (!spec || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->name = ck_ir_v2_strdup(spec->name);
    out->scope = spec->scope;
    out->role = spec->role;
    out->dtype = spec->dtype;
    out->optional = spec->optional;
    out->alias_of = ck_ir_v2_strdup(spec->alias_of);
    out->condition = ck_ir_v2_strdup(spec->condition);
    ck_ir_v2_copy_shape(out->shape, spec->shape);
    return 0;
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

static const CKKernelSpec *ck_ir_v2_find_kernel_spec(const char *name)
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

static const char *ck_ir_v2_select_kernel(const CKKernelSpec *spec, CKDataType dtype, int backward)
{
    if (!spec) {
        return NULL;
    }
    if (dtype < 0 || dtype >= CK_DT_COUNT) {
        dtype = spec->default_dtype;
    }
    const char *name = backward ? spec->backward[dtype] : spec->forward[dtype];
    if (name && name[0]) {
        return name;
    }
    for (int i = 0; i < CK_DT_COUNT; ++i) {
        name = backward ? spec->backward[i] : spec->forward[i];
        if (name && name[0]) {
            return name;
        }
    }
    return spec->name;
}

static const char *ck_ir_v2_find_key(const char *json, const char *key, const char *end)
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

static const char *ck_ir_v2_skip_ws(const char *cur, const char *end)
{
    while (cur < end && (*cur == ' ' || *cur == '\n' || *cur == '\r' || *cur == '\t')) {
        cur++;
    }
    return cur;
}

static int ck_ir_v2_parse_string(const char *start, const char *end, char **out)
{
    if (!start || start >= end || *start != '"') {
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
    *out = buf;
    return 0;
}

static int ck_ir_v2_parse_bool(const char *json, const char *key, const char *end, int *out)
{
    const char *p = ck_ir_v2_find_key(json, key, end);
    if (!p) {
        return -1;
    }
    p = strchr(p, ':');
    if (!p || p >= end) {
        return -1;
    }
    p = ck_ir_v2_skip_ws(p + 1, end);
    if (p + 4 <= end && memcmp(p, "true", 4) == 0) {
        *out = 1;
        return 0;
    }
    if (p + 5 <= end && memcmp(p, "false", 5) == 0) {
        *out = 0;
        return 0;
    }
    return -1;
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

static int ck_ir_v2_apply_weight_dtypes(const char *json, const char *end, CKIRV2Graph *graph)
{
    const char *start = ck_ir_v2_find_key(json, "\"weight_dtypes\"", end);
    if (!start) {
        return 0;
    }
    const char *open = strchr(start, '{');
    if (!open || open >= end) {
        return -1;
    }
    const char *cur = open + 1;
    while (cur < end) {
        cur = ck_ir_v2_skip_ws(cur, end);
        if (cur >= end || *cur == '}') {
            break;
        }
        char *key = NULL;
        if (ck_ir_v2_parse_string(cur, end, &key) != 0) {
            break;
        }
        cur = strchr(cur, ':');
        if (!cur) {
            free(key);
            break;
        }
        cur = ck_ir_v2_skip_ws(cur + 1, end);
        char *val = NULL;
        if (ck_ir_v2_parse_string(cur, end, &val) != 0) {
            free(key);
            break;
        }
        int idx = ck_ir_v2_find_buffer_index(graph, key);
        if (idx >= 0) {
            graph->buffers[idx].dtype = ck_ir_v2_parse_dtype(val);
        }
        free(key);
        free(val);
        cur = strchr(cur, ',');
        if (!cur) {
            break;
        }
        cur++;
    }
    return 0;
}

int ck_ir_v2_apply_meta(const char *path, CKIRV2Graph *graph)
{
    if (!path || !graph) {
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("ck_ir_v2_apply_meta: fopen");
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
    const char *end = buf + nread;

    int val = 0;
    if (ck_ir_v2_parse_bool(buf, "\"has_pos_emb\"", end, &val) == 0) {
        graph->has_pos_emb = val ? 1 : 0;
    }
    if (ck_ir_v2_parse_bool(buf, "\"tie_word_embeddings\"", end, &val) == 0) {
        graph->tie_word_embeddings = val ? 1 : 0;
        if (graph->tie_word_embeddings == 0) {
            int idx = ck_ir_v2_find_buffer_index(graph, "lm_head_weight");
            if (idx >= 0) {
                free(graph->buffers[idx].alias_of);
                graph->buffers[idx].alias_of = NULL;
            }
        } else {
            int idx = ck_ir_v2_find_buffer_index(graph, "lm_head_weight");
            if (idx >= 0) {
                free(graph->buffers[idx].alias_of);
                graph->buffers[idx].alias_of = ck_ir_v2_strdup("token_emb");
            }
        }
    }
    if (ck_ir_v2_parse_bool(buf, "\"fused_qkv\"", end, &val) == 0) {
        graph->fused_qkv = val ? 1 : 0;
    }
    if (ck_ir_v2_parse_bool(buf, "\"gated_mlp\"", end, &val) == 0) {
        graph->gated_mlp = val ? 1 : 0;
    }
    (void)ck_ir_v2_apply_weight_dtypes(buf, end, graph);

    free(buf);
    return 0;
}

int ck_ir_v2_build_decoder(const CKModelConfig *cfg, CKIRV2Graph *graph)
{
    if (!cfg || !graph) {
        return -1;
    }
    memset(graph, 0, sizeof(*graph));
    graph->config = *cfg;
    graph->has_pos_emb = 1;
    graph->tie_word_embeddings = -1;
    graph->fused_qkv = -1;
    graph->gated_mlp = -1;

    graph->num_buffers = (int)ck_decoder_buffer_count;
    graph->buffers = (CKIRV2Buffer *)calloc((size_t)graph->num_buffers, sizeof(CKIRV2Buffer));
    if (!graph->buffers) {
        return -1;
    }
    for (int i = 0; i < graph->num_buffers; ++i) {
        if (ck_ir_v2_copy_buffer_spec(&ck_decoder_buffers[i], &graph->buffers[i]) != 0) {
            ck_ir_v2_free(graph);
            return -1;
        }
        if (graph->buffers[i].name && strcmp(graph->buffers[i].name, "pos_emb") == 0) {
            free(graph->buffers[i].condition);
            graph->buffers[i].condition = ck_ir_v2_strdup("has_pos_emb");
        }
    }

    int plan_count = (int)ck_decoder_forward_plan_v2_count;
    graph->num_nodes = cfg->num_layers * plan_count;
    graph->nodes = (CKIRV2Node *)calloc((size_t)graph->num_nodes, sizeof(CKIRV2Node));
    if (!graph->nodes) {
        ck_ir_v2_free(graph);
        return -1;
    }

    int idx = 0;
    for (int layer = 0; layer < cfg->num_layers; ++layer) {
        for (int p = 0; p < plan_count; ++p) {
            const CKPlanStepV2 *step = &ck_decoder_forward_plan_v2[p];
            const CKKernelSpec *spec = ck_ir_v2_find_kernel_spec(step->kernel);
            CKDataType dtype = spec ? spec->default_dtype : CK_DT_FP32;
            const char *impl = ck_ir_v2_select_kernel(spec, dtype, 0);
            CKIRV2Node *node = &graph->nodes[idx++];
            node->layer = (uint16_t)layer;
            node->op = ck_ir_v2_strdup(step->kernel);
            node->kernel = ck_ir_v2_strdup(impl ? impl : step->kernel);
            node->kernel_dtype = dtype;
            node->condition = ck_ir_v2_strdup(step->condition);
            node->flags = 0;
            node->n_bindings = 0;
            if (step->bindings && step->num_bindings > 0) {
                int limit = (int)step->num_bindings;
                if (limit > CK_IR_V2_MAX_BINDINGS) {
                    limit = CK_IR_V2_MAX_BINDINGS;
                }
                for (int b = 0; b < limit; ++b) {
                    const CKPlanBinding *binding = &step->bindings[b];
                    node->bindings[node->n_bindings].arg = ck_ir_v2_strdup(binding->arg);
                    node->bindings[node->n_bindings].buffer =
                        ck_ir_v2_find_buffer_index(graph, binding->buffer);
                    node->n_bindings++;
                }
            }
            node->n_inputs = 0;
            node->n_outputs = 0;
        }
    }
    return 0;
}

int ck_ir_v2_build_decoder_backward(const CKIRV2Graph *forward, CKIRV2Graph *backward)
{
    if (!forward || !backward) {
        return -1;
    }
    memset(backward, 0, sizeof(*backward));
    backward->config = forward->config;

    backward->num_buffers = forward->num_buffers;
    backward->buffers = (CKIRV2Buffer *)calloc((size_t)backward->num_buffers, sizeof(CKIRV2Buffer));
    if (!backward->buffers) {
        return -1;
    }
    for (int i = 0; i < backward->num_buffers; ++i) {
        CKBufferSpec spec = {0};
        const CKIRV2Buffer *src = &forward->buffers[i];
        spec.name = src->name;
        spec.scope = src->scope;
        spec.role = src->role;
        spec.dtype = src->dtype;
        spec.optional = src->optional;
        spec.alias_of = src->alias_of;
        spec.condition = src->condition;
        memcpy(spec.shape, src->shape, sizeof(spec.shape));
        if (ck_ir_v2_copy_buffer_spec(&spec, &backward->buffers[i]) != 0) {
            ck_ir_v2_free(backward);
            return -1;
        }
    }

    int plan_count = (int)ck_decoder_backward_plan_v2_count;
    backward->num_nodes = forward->config.num_layers * plan_count;
    backward->nodes = (CKIRV2Node *)calloc((size_t)backward->num_nodes, sizeof(CKIRV2Node));
    if (!backward->nodes) {
        ck_ir_v2_free(backward);
        return -1;
    }

    int idx = 0;
    for (int layer = 0; layer < forward->config.num_layers; ++layer) {
        for (int p = 0; p < plan_count; ++p) {
            const CKPlanStepV2 *step = &ck_decoder_backward_plan_v2[p];
            const CKKernelSpec *spec = ck_ir_v2_find_kernel_spec(step->kernel);
            CKDataType dtype = spec ? spec->default_dtype : CK_DT_FP32;
            const char *impl = ck_ir_v2_select_kernel(spec, dtype, 1);
            CKIRV2Node *node = &backward->nodes[idx++];
            node->layer = (uint16_t)layer;
            node->op = ck_ir_v2_strdup(step->kernel);
            node->kernel = ck_ir_v2_strdup(impl ? impl : step->kernel);
            node->kernel_dtype = dtype;
            node->condition = ck_ir_v2_strdup(step->condition);
            node->flags = 0;
            node->n_bindings = 0;
            if (step->bindings && step->num_bindings > 0) {
                int limit = (int)step->num_bindings;
                if (limit > CK_IR_V2_MAX_BINDINGS) {
                    limit = CK_IR_V2_MAX_BINDINGS;
                }
                for (int b = 0; b < limit; ++b) {
                    const CKPlanBinding *binding = &step->bindings[b];
                    node->bindings[node->n_bindings].arg = ck_ir_v2_strdup(binding->arg);
                    node->bindings[node->n_bindings].buffer =
                        ck_ir_v2_find_buffer_index(backward, binding->buffer);
                    node->n_bindings++;
                }
            }
            node->n_inputs = 0;
            node->n_outputs = 0;
        }
    }
    return 0;
}
