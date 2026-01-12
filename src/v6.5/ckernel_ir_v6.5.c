#include "ckernel_ir.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_int_field(const char *json,
                           const char *key,
                           int *out_value)
{
    const char *p = strstr(json, key);
    if (!p) {
        return -1;
    }

    // Move to after the key, look for the first digit or minus sign.
    p = strchr(p, ':');
    if (!p) {
        return -1;
    }
    while (*p && (*p == ':' || *p == ' ' || *p == '\t')) {
        ++p;
    }

    int value = 0;
    if (sscanf(p, "%d", &value) != 1) {
        return -1;
    }

    *out_value = value;
    return 0;
}

static int parse_int_field_in_range(const char *json,
                                    size_t len,
                                    const char *key,
                                    int *out_value)
{
    if (!json || !key || !out_value) {
        return -1;
    }

    size_t key_len = strlen(key);
    const char *end = json + len;
    for (const char *p = json; p + key_len <= end; ++p) {
        if (memcmp(p, key, key_len) != 0) {
            continue;
        }

        const char *colon = memchr(p + key_len, ':', (size_t)(end - (p + key_len)));
        if (!colon) {
            return -1;
        }
        const char *v = colon + 1;
        while (v < end && (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r')) {
            ++v;
        }

        int value = 0;
        if (v < end && sscanf(v, "%d", &value) == 1) {
            *out_value = value;
            return 0;
        }
        return -1;
    }

    return -1;
}

static int parse_int_field_any(const char *json,
                               size_t len,
                               const char *const *keys,
                               int *out_value)
{
    if (!keys) {
        return -1;
    }
    for (int i = 0; keys[i]; ++i) {
        if (parse_int_field_in_range(json, len, keys[i], out_value) == 0) {
            return 0;
        }
    }
    return -1;
}

static int parse_float_field_in_range(const char *json,
                                      size_t len,
                                      const char *key,
                                      float *out_value)
{
    if (!json || !key || !out_value) {
        return -1;
    }

    size_t key_len = strlen(key);
    const char *end = json + len;
    for (const char *p = json; p + key_len <= end; ++p) {
        if (memcmp(p, key, key_len) != 0) {
            continue;
        }

        const char *colon = memchr(p + key_len, ':', (size_t)(end - (p + key_len)));
        if (!colon) {
            return -1;
        }
        const char *v = colon + 1;
        while (v < end && (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r')) {
            ++v;
        }

        float value = 0.0f;
        if (v < end && sscanf(v, "%f", &value) == 1) {
            *out_value = value;
            return 0;
        }
        return -1;
    }

    return -1;
}

static int parse_float_field_any(const char *json,
                                 size_t len,
                                 const char *const *keys,
                                 float *out_value)
{
    if (!keys) {
        return -1;
    }
    for (int i = 0; keys[i]; ++i) {
        if (parse_float_field_in_range(json, len, keys[i], out_value) == 0) {
            return 0;
        }
    }
    return -1;
}

static int find_object_range(const char *json,
                             const char *key,
                             const char **out_start,
                             size_t *out_len)
{
    if (!json || !key || !out_start || !out_len) {
        return -1;
    }

    const char *p = strstr(json, key);
    if (!p) {
        return -1;
    }

    const char *colon = strchr(p, ':');
    if (!colon) {
        return -1;
    }

    const char *brace = strchr(colon, '{');
    if (!brace) {
        return -1;
    }

    bool in_string = false;
    bool escape = false;
    int depth = 0;
    const char *start = NULL;

    for (const char *cur = brace; *cur; ++cur) {
        char c = *cur;
        if (in_string) {
            if (escape) {
                escape = false;
                continue;
            }
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            if (depth == 0) {
                start = cur;
            }
            depth++;
            continue;
        }
        if (c == '}') {
            depth--;
            if (depth == 0) {
                *out_start = start;
                *out_len = (size_t)(cur - start + 1);
                return 0;
            }
        }
    }

    return -1;
}

int ck_model_config_from_hf_json(const char *path, CKModelConfig *cfg)
{
    if (!path || !cfg) {
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("ck_model_config_from_hf_json: fopen");
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

    CKModelConfig tmp;
    memset(&tmp, 0, sizeof(tmp));
    tmp.rms_norm_eps = 1e-5f;
    tmp.rope_theta = 0.0f;

    const char *scope = buf;
    size_t scope_len = nread;
    if (find_object_range(buf, "\"text_config\"", &scope, &scope_len) != 0) {
        scope = buf;
        scope_len = nread;
    }

    const char *num_layers_keys[] = { "\"num_hidden_layers\"", "\"n_layer\"", NULL };
    const char *hidden_size_keys[] = { "\"hidden_size\"", "\"n_embd\"", "\"d_model\"", NULL };
    const char *intermediate_keys[] = { "\"intermediate_size\"", "\"n_inner\"", "\"ffn_dim\"", "\"mlp_dim\"", NULL };
    const char *num_heads_keys[] = { "\"num_attention_heads\"", "\"n_head\"", "\"num_heads\"", NULL };
    const char *num_kv_heads_keys[] = { "\"num_key_value_heads\"", "\"num_kv_heads\"", NULL };
    const char *vocab_keys[] = { "\"vocab_size\"", "\"n_vocab\"", NULL };
    const char *context_keys[] = { "\"max_position_embeddings\"", "\"n_positions\"", "\"context_length\"", "\"seq_len\"", NULL };
    const char *rms_eps_keys[] = { "\"rms_norm_eps\"", "\"layer_norm_eps\"", NULL };
    const char *rope_theta_keys[] = { "\"rope_theta\"", "\"rope_base\"", NULL };

    if (parse_int_field_any(scope, scope_len, num_layers_keys, &tmp.num_layers) != 0) {
        fprintf(stderr, "Warning: num_hidden_layers not found in %s\n", path);
    }
    if (parse_int_field_any(scope, scope_len, hidden_size_keys, &tmp.hidden_size) != 0) {
        fprintf(stderr, "Warning: hidden_size not found in %s\n", path);
    }
    if (parse_int_field_any(scope, scope_len, intermediate_keys, &tmp.intermediate_size) != 0) {
        fprintf(stderr, "Warning: intermediate_size not found in %s\n", path);
    }
    if (parse_int_field_any(scope, scope_len, num_heads_keys, &tmp.num_heads) != 0) {
        fprintf(stderr, "Warning: num_attention_heads not found in %s\n", path);
    }

    // num_key_value_heads is optional; default to num_heads if missing.
    if (parse_int_field_any(scope, scope_len, num_kv_heads_keys, &tmp.num_kv_heads) != 0) {
        tmp.num_kv_heads = tmp.num_heads;
    }

    // Optional: vocab_size
    if (parse_int_field_any(scope, scope_len, vocab_keys, &tmp.vocab_size) != 0) {
        tmp.vocab_size = 0;
    }

    // Optional: context length (try max_position_embeddings, then n_positions)
    if (parse_int_field_any(scope, scope_len, context_keys, &tmp.context_window) != 0) {
        tmp.context_window = 0;
    }
    if (parse_float_field_any(scope, scope_len, rms_eps_keys, &tmp.rms_norm_eps) != 0) {
        tmp.rms_norm_eps = 1e-5f;
    }
    if (parse_float_field_any(scope, scope_len, rope_theta_keys, &tmp.rope_theta) != 0) {
        tmp.rope_theta = 0.0f;
    }

    free(buf);
    *cfg = tmp;
    return 0;
}

int ck_build_decoder_ir(const CKModelConfig *cfg, CKIRGraph *graph)
{
    if (!cfg || !graph) {
        return -1;
    }

    const int L = cfg->num_layers > 0 ? cfg->num_layers : 1;
    const int nodes_per_layer = 10; // LN1, QKV, ATT, ADD, LN2, W1, SPLIT, SWIGLU, W2, ADD
    const int total_nodes = L * nodes_per_layer;

    CKIRNode *nodes = (CKIRNode *)calloc((size_t)total_nodes, sizeof(CKIRNode));
    if (!nodes) {
        return -1;
    }

    // Sentinel producer for block inputs (e.g., h_in, past_kv) per layer.
    const uint16_t INPUT_NODE_SENTINEL = 0xFFFF;

    for (int layer = 0; layer < L; ++layer) {
        const int base = layer * nodes_per_layer;

        // Node 0: LN1 = RMSNorm(h_in)
        nodes[base + 0].id.layer = (uint16_t)layer;
        nodes[base + 0].id.node  = 0;
        nodes[base + 0].op       = CK_OP_RMSNORM;
        nodes[base + 0].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 0].inputs[0].producer.node  = INPUT_NODE_SENTINEL; // h_in
        nodes[base + 0].inputs[0].out_index      = 0;
        nodes[base + 0].n_inputs  = 1;
        nodes[base + 0].n_outputs = 1;

        // Node 1: QKV Linear
        nodes[base + 1].id.layer = (uint16_t)layer;
        nodes[base + 1].id.node  = 1;
        nodes[base + 1].op       = CK_OP_LINEAR_QKV;
        nodes[base + 1].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 1].inputs[0].producer.node  = 0;
        nodes[base + 1].inputs[0].out_index      = 0;
        nodes[base + 1].n_inputs  = 1;
        nodes[base + 1].n_outputs = 1;

        // Node 2: Attention
        nodes[base + 2].id.layer = (uint16_t)layer;
        nodes[base + 2].id.node  = 2;
        nodes[base + 2].op       = CK_OP_ATTENTION;
        nodes[base + 2].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 2].inputs[0].producer.node  = 1; // qkv
        nodes[base + 2].inputs[0].out_index      = 0;
        nodes[base + 2].n_inputs  = 1;  // past_kv omitted for now
        nodes[base + 2].n_outputs = 1;

        // Node 3: Add residual (h_in + attn_out)
        nodes[base + 3].id.layer = (uint16_t)layer;
        nodes[base + 3].id.node  = 3;
        nodes[base + 3].op       = CK_OP_ADD;
        nodes[base + 3].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 3].inputs[0].producer.node  = INPUT_NODE_SENTINEL; // h_in
        nodes[base + 3].inputs[0].out_index      = 0;
        nodes[base + 3].inputs[1].producer.layer = (uint16_t)layer;
        nodes[base + 3].inputs[1].producer.node  = 2; // attn_out
        nodes[base + 3].inputs[1].out_index      = 0;
        nodes[base + 3].n_inputs  = 2;
        nodes[base + 3].n_outputs = 1;

        // Node 4: LN2 = RMSNorm(residual)
        nodes[base + 4].id.layer = (uint16_t)layer;
        nodes[base + 4].id.node  = 4;
        nodes[base + 4].op       = CK_OP_RMSNORM;
        nodes[base + 4].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 4].inputs[0].producer.node  = 3;
        nodes[base + 4].inputs[0].out_index      = 0;
        nodes[base + 4].n_inputs  = 1;
        nodes[base + 4].n_outputs = 1;

        // Node 5: W1 Linear
        nodes[base + 5].id.layer = (uint16_t)layer;
        nodes[base + 5].id.node  = 5;
        nodes[base + 5].op       = CK_OP_LINEAR;
        nodes[base + 5].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 5].inputs[0].producer.node  = 4;
        nodes[base + 5].inputs[0].out_index      = 0;
        nodes[base + 5].n_inputs  = 1;
        nodes[base + 5].n_outputs = 1;

        // Node 6: Split into (a, b)
        nodes[base + 6].id.layer = (uint16_t)layer;
        nodes[base + 6].id.node  = 6;
        nodes[base + 6].op       = CK_OP_SPLIT;
        nodes[base + 6].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 6].inputs[0].producer.node  = 5;
        nodes[base + 6].inputs[0].out_index      = 0;
        nodes[base + 6].n_inputs  = 1;
        nodes[base + 6].n_outputs = 2;

        // Node 7: SwiGLU(a,b)
        nodes[base + 7].id.layer = (uint16_t)layer;
        nodes[base + 7].id.node  = 7;
        nodes[base + 7].op       = CK_OP_SWIGLU;
        nodes[base + 7].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 7].inputs[0].producer.node  = 6;
        nodes[base + 7].inputs[0].out_index      = 0; // a
        nodes[base + 7].inputs[1].producer.layer = (uint16_t)layer;
        nodes[base + 7].inputs[1].producer.node  = 6;
        nodes[base + 7].inputs[1].out_index      = 1; // b
        nodes[base + 7].n_inputs  = 2;
        nodes[base + 7].n_outputs = 1;

        // Node 8: W2 Linear
        nodes[base + 8].id.layer = (uint16_t)layer;
        nodes[base + 8].id.node  = 8;
        nodes[base + 8].op       = CK_OP_LINEAR;
        nodes[base + 8].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 8].inputs[0].producer.node  = 7;
        nodes[base + 8].inputs[0].out_index      = 0;
        nodes[base + 8].n_inputs  = 1;
        nodes[base + 8].n_outputs = 1;

        // Node 9: Add residual: out = residual + mlp_out
        nodes[base + 9].id.layer = (uint16_t)layer;
        nodes[base + 9].id.node  = 9;
        nodes[base + 9].op       = CK_OP_ADD;
        nodes[base + 9].inputs[0].producer.layer = (uint16_t)layer;
        nodes[base + 9].inputs[0].producer.node  = 3; // first residual output
        nodes[base + 9].inputs[0].out_index      = 0;
        nodes[base + 9].inputs[1].producer.layer = (uint16_t)layer;
        nodes[base + 9].inputs[1].producer.node  = 8; // mlp_out
        nodes[base + 9].inputs[1].out_index      = 0;
        nodes[base + 9].n_inputs  = 2;
        nodes[base + 9].n_outputs = 1;
    }

    graph->config   = *cfg;
    graph->num_nodes = total_nodes;
    graph->nodes     = nodes;
    return 0;
}

static CKOpType map_forward_to_backward(CKOpType op)
{
    switch (op) {
    case CK_OP_RMSNORM:    return CK_OP_RMSNORM_BWD;
    case CK_OP_LINEAR_QKV: return CK_OP_LINEAR_QKV_BWD;
    case CK_OP_ATTENTION:  return CK_OP_ATTENTION_BWD;
    case CK_OP_ADD:        return CK_OP_ADD_BWD;
    case CK_OP_LINEAR:     return CK_OP_LINEAR_BWD;
    case CK_OP_SPLIT:      return CK_OP_SPLIT_BWD;
    case CK_OP_SWIGLU:     return CK_OP_SWIGLU_BWD;
    default:               return op;
    }
}

int ck_build_decoder_backward_ir(const CKIRGraph *forward, CKIRGraph *backward)
{
    if (!forward || !backward) {
        return -1;
    }
    if (forward->num_nodes <= 0 || !forward->nodes) {
        return -1;
    }

    const int N = forward->num_nodes;
    CKIRNode *nodes = (CKIRNode *)calloc((size_t)N, sizeof(CKIRNode));
    if (!nodes) {
        return -1;
    }

    for (int i = 0; i < N; ++i) {
        const CKIRNode *f = &forward->nodes[N - 1 - i]; // reverse order
        CKIRNode *b = &nodes[i];

        b->id       = f->id;                         // same layer/node id
        b->op       = map_forward_to_backward(f->op);
        b->n_inputs = f->n_outputs;                  // placeholder
        b->n_outputs= f->n_inputs;                   // placeholder

        // For now we simply copy the forward inputs to keep a reference
        // to which activations this backward op relates to.
        for (int j = 0; j < f->n_inputs; ++j) {
            b->inputs[j] = f->inputs[j];
        }
    }

    backward->config    = forward->config;
    backward->num_nodes = N;
    backward->nodes     = nodes;
    return 0;
}

void ck_ir_free(CKIRGraph *graph)
{
    if (!graph) {
        return;
    }
    free(graph->nodes);
    graph->nodes = NULL;
    graph->num_nodes = 0;
}

static const char *op_name(CKOpType op)
{
    switch (op) {
    case CK_OP_RMSNORM:     return "RMSNORM";
    case CK_OP_LINEAR_QKV:  return "LINEAR_QKV";
    case CK_OP_ATTENTION:   return "ATTENTION";
    case CK_OP_ADD:         return "ADD";
    case CK_OP_LINEAR:      return "LINEAR";
    case CK_OP_SPLIT:       return "SPLIT";
    case CK_OP_SWIGLU:      return "SWIGLU";
    case CK_OP_RMSNORM_BWD:    return "RMSNORM_BWD";
    case CK_OP_LINEAR_QKV_BWD: return "LINEAR_QKV_BWD";
    case CK_OP_ATTENTION_BWD:  return "ATTENTION_BWD";
    case CK_OP_ADD_BWD:        return "ADD_BWD";
    case CK_OP_LINEAR_BWD:     return "LINEAR_BWD";
    case CK_OP_SPLIT_BWD:      return "SPLIT_BWD";
    case CK_OP_SWIGLU_BWD:     return "SWIGLU_BWD";
    default:                return "UNKNOWN";
    }
}

void ck_ir_dump(const CKIRGraph *graph, FILE *out)
{
    if (!graph || !out) {
        return;
    }

    fprintf(out,
            "CKIRGraph: layers=%d, hidden_size=%d, intermediate_size=%d, heads=%d, kv_heads=%d, vocab=%d, ctx=%d, eps=%.6g, rope_theta=%.6g\n",
            graph->config.num_layers,
            graph->config.hidden_size,
            graph->config.intermediate_size,
            graph->config.num_heads,
            graph->config.num_kv_heads,
            graph->config.vocab_size,
            graph->config.context_window,
            graph->config.rms_norm_eps,
            graph->config.rope_theta);

    for (int i = 0; i < graph->num_nodes; ++i) {
        const CKIRNode *n = &graph->nodes[i];
        fprintf(out, "  L%u N%u %-14s outputs=[",
                (unsigned)n->id.layer,
                (unsigned)n->id.node,
                op_name(n->op));
        for (int o = 0; o < n->n_outputs; ++o) {
            if (o > 0) {
                fputc(',', out);
            }
            fprintf(out, "L%u:N%u:%d",
                    (unsigned)n->id.layer,
                    (unsigned)n->id.node,
                    o);
        }
        fprintf(out, "] inputs=[");
        for (int j = 0; j < n->n_inputs; ++j) {
            const CKInputRef *inp = &n->inputs[j];
            if (j > 0) {
                fputc(',', out);
            }
            if (inp->producer.node == 0xFFFFu) {
                fprintf(out, "IN");
            } else {
                fprintf(out, "L%u:N%u",
                        (unsigned)inp->producer.layer,
                        (unsigned)inp->producer.node);
            }
        }
        fprintf(out, "]\n");
    }
}

int ck_ir_serialize_json(const CKIRGraph *graph, const char *path)
{
    if (!graph || !path) {
        return -1;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        perror("ck_ir_serialize_json: fopen");
        return -1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"config\": {\n");
    fprintf(f, "    \"num_layers\": %d,\n", graph->config.num_layers);
    fprintf(f, "    \"hidden_size\": %d,\n", graph->config.hidden_size);
    fprintf(f, "    \"intermediate_size\": %d,\n", graph->config.intermediate_size);
    fprintf(f, "    \"num_attention_heads\": %d,\n", graph->config.num_heads);
    fprintf(f, "    \"num_key_value_heads\": %d,\n", graph->config.num_kv_heads);
    fprintf(f, "    \"vocab_size\": %d,\n", graph->config.vocab_size);
    fprintf(f, "    \"context_window\": %d,\n", graph->config.context_window);
    fprintf(f, "    \"rms_norm_eps\": %.9g,\n", graph->config.rms_norm_eps);
    fprintf(f, "    \"rope_theta\": %.9g\n", graph->config.rope_theta);
    fprintf(f, "  },\n");

    // For now we only emit a flat "nodes" array. Higher-level tools can
    // reorganize this into header/block/footer with per-layer arrays.
    fprintf(f, "  \"nodes\": [\n");
    for (int i = 0; i < graph->num_nodes; ++i) {
        const CKIRNode *n = &graph->nodes[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"layer\": %u,\n", (unsigned)n->id.layer);
        fprintf(f, "      \"node\": %u,\n", (unsigned)n->id.node);
        fprintf(f, "      \"op\": \"%s\",\n", op_name(n->op));

        // Outputs: derive labels L<layer>:N<node>:slot
        fprintf(f, "      \"outputs\": [");
        for (int o = 0; o < n->n_outputs; ++o) {
            if (o > 0) fprintf(f, ", ");
            fprintf(f, "\"L%u:N%u:%d\"",
                    (unsigned)n->id.layer,
                    (unsigned)n->id.node,
                    o);
        }
        fprintf(f, "],\n");

        // Inputs: either "IN" or L<layer>:N<node>:slot
        fprintf(f, "      \"inputs\": [");
        for (int j = 0; j < n->n_inputs; ++j) {
            const CKInputRef *inp = &n->inputs[j];
            if (j > 0) fprintf(f, ", ");
            if (inp->producer.node == 0xFFFFu) {
                fprintf(f, "\"IN\"");
            } else {
                fprintf(f, "\"L%u:N%u:%u\"",
                        (unsigned)inp->producer.layer,
                        (unsigned)inp->producer.node,
                        (unsigned)inp->out_index);
            }
        }
        fprintf(f, "]\n");

        fprintf(f, "    }%s\n", (i + 1 < graph->num_nodes) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

static CKOpType parse_op(const char *s)
{
    if (strcmp(s, "RMSNORM") == 0)        return CK_OP_RMSNORM;
    if (strcmp(s, "LINEAR_QKV") == 0)     return CK_OP_LINEAR_QKV;
    if (strcmp(s, "ATTENTION") == 0)      return CK_OP_ATTENTION;
    if (strcmp(s, "ADD") == 0)            return CK_OP_ADD;
    if (strcmp(s, "LINEAR") == 0)         return CK_OP_LINEAR;
    if (strcmp(s, "SPLIT") == 0)          return CK_OP_SPLIT;
    if (strcmp(s, "SWIGLU") == 0)         return CK_OP_SWIGLU;
    if (strcmp(s, "RMSNORM_BWD") == 0)    return CK_OP_RMSNORM_BWD;
    if (strcmp(s, "LINEAR_QKV_BWD") == 0) return CK_OP_LINEAR_QKV_BWD;
    if (strcmp(s, "ATTENTION_BWD") == 0)  return CK_OP_ATTENTION_BWD;
    if (strcmp(s, "ADD_BWD") == 0)        return CK_OP_ADD_BWD;
    if (strcmp(s, "LINEAR_BWD") == 0)     return CK_OP_LINEAR_BWD;
    if (strcmp(s, "SPLIT_BWD") == 0)      return CK_OP_SPLIT_BWD;
    if (strcmp(s, "SWIGLU_BWD") == 0)     return CK_OP_SWIGLU_BWD;
    return CK_OP_RMSNORM; // default fallback
}

int ck_ir_parse_json(const char *path, CKIRGraph *graph)
{
    if (!path || !graph) {
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("ck_ir_parse_json: fopen");
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

    CKIRGraph tmp;
    memset(&tmp, 0, sizeof(tmp));

    // Parse config using the same helper as HF-style JSON.
    if (parse_int_field(buf, "\"num_layers\"", &tmp.config.num_layers) != 0) {
        fprintf(stderr, "ck_ir_parse_json: missing num_layers\n");
    }
    if (parse_int_field(buf, "\"hidden_size\"", &tmp.config.hidden_size) != 0) {
        fprintf(stderr, "ck_ir_parse_json: missing hidden_size\n");
    }
    if (parse_int_field(buf, "\"intermediate_size\"", &tmp.config.intermediate_size) != 0) {
        fprintf(stderr, "ck_ir_parse_json: missing intermediate_size\n");
    }
    if (parse_int_field(buf, "\"num_attention_heads\"", &tmp.config.num_heads) != 0) {
        fprintf(stderr, "ck_ir_parse_json: missing num_attention_heads\n");
    }
    if (parse_int_field(buf, "\"num_key_value_heads\"", &tmp.config.num_kv_heads) != 0) {
        tmp.config.num_kv_heads = tmp.config.num_heads;
    }

    // Optional: vocab_size / context_window may be present
    if (parse_int_field(buf, "\"vocab_size\"", &tmp.config.vocab_size) != 0) {
        tmp.config.vocab_size = 0;
    }
    if (parse_int_field(buf, "\"context_window\"", &tmp.config.context_window) != 0) {
        tmp.config.context_window = 0;
    }
    if (parse_float_field_in_range(buf, nread, "\"rms_norm_eps\"", &tmp.config.rms_norm_eps) != 0) {
        tmp.config.rms_norm_eps = 1e-5f;
    }
    if (parse_float_field_in_range(buf, nread, "\"rope_theta\"", &tmp.config.rope_theta) != 0) {
        tmp.config.rope_theta = 0.0f;
    }

    // Count nodes by scanning for "layer" keys in the nodes array.
    char *nodes_begin = strstr(buf, "\"nodes\"");
    if (!nodes_begin) {
        free(buf);
        return -1;
    }
    char *p = nodes_begin;
    int count = 0;
    while ((p = strstr(p, "\"layer\"")) != NULL) {
        count++;
        p += 7;
    }
    if (count <= 0) {
        free(buf);
        return -1;
    }

    CKIRNode *nodes = (CKIRNode *)calloc((size_t)count, sizeof(CKIRNode));
    if (!nodes) {
        free(buf);
        return -1;
    }

    // Parse each node sequentially.
    p = nodes_begin;
    for (int i = 0; i < count; ++i) {
        // layer
        char *pl = strstr(p, "\"layer\"");
        if (!pl) { free(nodes); free(buf); return -1; }
        int layer = 0;
        if (sscanf(strchr(pl, ':'), " : %d", &layer) != 1) {
            free(nodes); free(buf); return -1;
        }

        // node
        char *pn = strstr(pl, "\"node\"");
        if (!pn) { free(nodes); free(buf); return -1; }
        int node = 0;
        if (sscanf(strchr(pn, ':'), " : %d", &node) != 1) {
            free(nodes); free(buf); return -1;
        }

        // op string
        char *po = strstr(pn, "\"op\"");
        if (!po) { free(nodes); free(buf); return -1; }
        char op_str[64] = {0};
        if (sscanf(strchr(po, ':'), " : \"%63[^\"]\"", op_str) != 1) {
            free(nodes); free(buf); return -1;
        }

        CKIRNode *n = &nodes[i];
        n->id.layer = (uint16_t)layer;
        n->id.node  = (uint16_t)node;
        n->op       = parse_op(op_str);

        // outputs: count entries between [ ... ]
        char *pout = strstr(po, "\"outputs\"");
        if (!pout) { free(nodes); free(buf); return -1; }
        char *bo = strchr(pout, '[');
        char *eo = strchr(pout, ']');
        int out_count = 0;
        if (bo && eo && eo > bo) {
            char *q = bo;
            while ((q = strchr(q, '"')) && q < eo) {
                out_count++;
                q = strchr(q + 1, '"');
                if (!q || q >= eo) break;
                // Skip closing quote
                q++;
            }
        }
        n->n_outputs = (uint8_t)out_count;

        // inputs
        char *pin = strstr(po, "\"inputs\"");
        if (!pin) { free(nodes); free(buf); return -1; }
        char *bi = strchr(pin, '[');
        char *ei = strchr(pin, ']');
        int in_count = 0;
        if (bi && ei && ei > bi) {
            // Simple scan for tokens "IN" or "Lx:Nx:s"
            char *q = bi;
            while ((q = strchr(q, '"')) && q < ei) {
                char tok[64] = {0};
                if (sscanf(q, "\"%63[^\"]\"", tok) != 1) {
                    break;
                }
                if (strcmp(tok, "IN") == 0) {
                    n->inputs[in_count].producer.layer = (uint16_t)layer;
                    n->inputs[in_count].producer.node  = 0xFFFFu;
                    n->inputs[in_count].out_index      = 0;
                } else {
                    unsigned plh = 0, pnn = 0, slot = 0;
                    if (sscanf(tok, "L%u:N%u:%u", &plh, &pnn, &slot) == 3) {
                        n->inputs[in_count].producer.layer = (uint16_t)plh;
                        n->inputs[in_count].producer.node  = (uint16_t)pnn;
                        n->inputs[in_count].out_index      = (uint8_t)slot;
                    }
                }
                in_count++;
                // Move q past this token
                q = strchr(q + 1, '"');
                if (!q || q >= ei) break;
                q++;
            }
        }
        n->n_inputs = (uint8_t)in_count;

        // Move p forward for the next iteration
        p = pin + 7;
    }

    free(buf);
    graph->config    = tmp.config;
    graph->num_nodes = count;
    graph->nodes     = nodes;
    return 0;
}
