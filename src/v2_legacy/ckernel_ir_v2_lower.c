#include "ckernel_ir_v2_lower.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *ck_ir_v2_lower_strdup(const char *s)
{
    if (!s) {
        return NULL;
    }
    size_t len = strlen(s);
    char *out = (char *)malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len);
    out[len] = '\0';
    return out;
}

const char *ck_ir_v2_lower_mode_name(CKIRV2LowerMode mode)
{
    switch (mode) {
    case CK_IR_V2_LOWER_PREFILL:
        return "prefill";
    case CK_IR_V2_LOWER_DECODE:
        return "decode";
    case CK_IR_V2_LOWER_BACKWARD:
        return "backward";
    default:
        return "unknown";
    }
}

int ck_ir_v2_lower_mode_from_string(const char *name, CKIRV2LowerMode *out_mode)
{
    if (!name || !out_mode) {
        return -1;
    }
    if (strcmp(name, "prefill") == 0) {
        *out_mode = CK_IR_V2_LOWER_PREFILL;
        return 0;
    }
    if (strcmp(name, "decode") == 0) {
        *out_mode = CK_IR_V2_LOWER_DECODE;
        return 0;
    }
    if (strcmp(name, "backward") == 0) {
        *out_mode = CK_IR_V2_LOWER_BACKWARD;
        return 0;
    }
    return -1;
}

static int ck_ir_v2_lower_node_enabled(const CKIRV2Node *node, CKIRV2LowerMode mode)
{
    if (!node) {
        return 0;
    }
    if (node->condition) {
        const char *cond = node->condition;
        const char *cur = cond;
        while (*cur == ' ' || *cur == '\t') {
            cur++;
        }
        if (strncmp(cur, "mode", 4) == 0) {
            cur += 4;
            while (*cur == ' ' || *cur == '\t') {
                cur++;
            }
            if (cur[0] == '=' && cur[1] == '=') {
                cur += 2;
                while (*cur == ' ' || *cur == '\t') {
                    cur++;
                }
                char mode_buf[16] = {0};
                int idx = 0;
                while (*cur && idx < (int)(sizeof(mode_buf) - 1) &&
                       ((*cur >= 'a' && *cur <= 'z') || *cur == '_' || *cur == '-')) {
                    mode_buf[idx++] = *cur++;
                }
                mode_buf[idx] = '\0';
                CKIRV2LowerMode target;
                if (ck_ir_v2_lower_mode_from_string(mode_buf, &target) == 0) {
                    return mode == target;
                }
            }
        }
        if (strcmp(node->condition, "training_enabled") == 0 ||
            strcmp(node->condition, "backward_only") == 0) {
            return mode == CK_IR_V2_LOWER_BACKWARD;
        }
        if (strcmp(node->condition, "inference_only") == 0) {
            return mode != CK_IR_V2_LOWER_BACKWARD;
        }
        if (strcmp(node->condition, "prefill_only") == 0) {
            return mode == CK_IR_V2_LOWER_PREFILL;
        }
        if (strcmp(node->condition, "decode_only") == 0) {
            return mode == CK_IR_V2_LOWER_DECODE;
        }
    }
    return 1;
}

static int ck_ir_v2_lower_copy_buffers(const CKIRV2Graph *input, CKIRV2Graph *output)
{
    output->num_buffers = input->num_buffers;
    output->buffers = (CKIRV2Buffer *)calloc((size_t)output->num_buffers, sizeof(CKIRV2Buffer));
    if (!output->buffers) {
        return -1;
    }
    for (int i = 0; i < input->num_buffers; ++i) {
        const CKIRV2Buffer *src = &input->buffers[i];
        CKIRV2Buffer *dst = &output->buffers[i];
        dst->name = ck_ir_v2_lower_strdup(src->name);
        dst->scope = src->scope;
        dst->role = src->role;
        dst->dtype = src->dtype;
        memcpy(dst->shape, src->shape, sizeof(dst->shape));
        dst->optional = src->optional;
        dst->alias_of = ck_ir_v2_lower_strdup(src->alias_of);
        dst->condition = ck_ir_v2_lower_strdup(src->condition);
    }
    return 0;
}

static int ck_ir_v2_lower_copy_nodes(const CKIRV2Graph *input,
                                     CKIRV2LowerMode mode,
                                     CKIRV2Graph *output)
{
    int count = 0;
    for (int i = 0; i < input->num_nodes; ++i) {
        if (ck_ir_v2_lower_node_enabled(&input->nodes[i], mode)) {
            count++;
        }
    }
    output->num_nodes = count;
    output->nodes = (CKIRV2Node *)calloc((size_t)count, sizeof(CKIRV2Node));
    if (!output->nodes) {
        return -1;
    }
    int idx = 0;
    for (int i = 0; i < input->num_nodes; ++i) {
        const CKIRV2Node *src = &input->nodes[i];
        if (!ck_ir_v2_lower_node_enabled(src, mode)) {
            continue;
        }
        CKIRV2Node *dst = &output->nodes[idx++];
        dst->op = ck_ir_v2_lower_strdup(src->op);
        dst->kernel = ck_ir_v2_lower_strdup(src->kernel);
        dst->kernel_dtype = src->kernel_dtype;
        dst->condition = ck_ir_v2_lower_strdup(src->condition);
        dst->layer = src->layer;
        dst->flags = src->flags;
        if (mode != CK_IR_V2_LOWER_BACKWARD) {
            dst->flags = (uint8_t)(dst->flags | CK_IR_V2_NODE_INFERENCE_ONLY);
        }
        dst->n_bindings = src->n_bindings;
        for (int b = 0; b < src->n_bindings; ++b) {
            dst->bindings[b].arg = ck_ir_v2_lower_strdup(src->bindings[b].arg);
            dst->bindings[b].buffer = src->bindings[b].buffer;
        }
        dst->n_inputs = src->n_inputs;
        dst->n_outputs = src->n_outputs;
        for (int j = 0; j < dst->n_inputs; ++j) {
            dst->inputs[j] = src->inputs[j];
        }
        for (int j = 0; j < dst->n_outputs; ++j) {
            dst->outputs[j] = src->outputs[j];
        }
    }
    return 0;
}

int ck_ir_v2_lower_graph(const CKIRV2Graph *input,
                         CKIRV2LowerMode mode,
                         CKIRV2Graph *output,
                         CKMemPlan *plan)
{
    if (!input || !output || !plan) {
        return -1;
    }
    memset(output, 0, sizeof(*output));
    memset(plan, 0, sizeof(*plan));

    output->config = input->config;
    output->has_pos_emb = input->has_pos_emb;
    output->tie_word_embeddings = input->tie_word_embeddings;
    output->fused_qkv = input->fused_qkv;
    output->gated_mlp = input->gated_mlp;

    if (ck_ir_v2_lower_copy_buffers(input, output) != 0 ||
        ck_ir_v2_lower_copy_nodes(input, mode, output) != 0) {
        ck_ir_v2_free(output);
        return -1;
    }

    int tokens_override = (mode == CK_IR_V2_LOWER_DECODE) ? 1 : -1;
    int rc = 0;
    if (mode == CK_IR_V2_LOWER_BACKWARD) {
        rc = ck_mem_plan_build_training_with_tokens(output, plan,
                                                    CK_MEM_PLAN_DEFAULT_ALIGN,
                                                    tokens_override);
    } else {
        rc = ck_mem_plan_build_inference_with_tokens(output, plan,
                                                     CK_MEM_PLAN_DEFAULT_ALIGN,
                                                     tokens_override);
    }
    if (rc != 0) {
        ck_ir_v2_free(output);
        ck_mem_plan_free(plan);
        return -1;
    }
    return 0;
}

int ck_ir_v2_lower_emit_json(const CKIRV2Graph *input,
                             CKIRV2LowerMode mode,
                             const char *path)
{
    if (!input || !path) {
        return -1;
    }
    CKIRV2Graph lowered = {0};
    CKMemPlan plan = {0};
    if (ck_ir_v2_lower_graph(input, mode, &lowered, &plan) != 0) {
        return -1;
    }
    int tokens_override = (mode == CK_IR_V2_LOWER_DECODE) ? 1 : -1;
    int base_context = (tokens_override >= 0) ? input->config.context_window : -1;
    int rc = ck_ir_v2_serialize_json_with_plan(&lowered, &plan,
                                               ck_ir_v2_lower_mode_name(mode),
                                               tokens_override,
                                               base_context,
                                               path);
    ck_ir_v2_free(&lowered);
    ck_mem_plan_free(&plan);
    return rc;
}
