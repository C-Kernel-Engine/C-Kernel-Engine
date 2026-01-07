#include "ckernel_codegen_v2_emit.h"

#include <stdio.h>

static const char *ck_codegen_v2_dtype_name(CKDataType dtype)
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

static void emit_schedule_block(FILE *out,
                                const CKIRV2Graph *graph,
                                const char *func_name,
                                const char *label,
                                const char *runtime_sym)
{
    fprintf(out, "void %s(void) {\n", func_name);
    fprintf(out, "    /* %s schedule (no-op placeholders). */\n", label);
    for (int i = 0; i < graph->num_nodes; ++i) {
        const CKIRV2Node *node = &graph->nodes[i];
        fprintf(out,
                "    /* node %d: layer=%d op=%s kernel=%s dtype=%s */\n",
                i,
                (int)node->layer,
                node->op ? node->op : "none",
                node->kernel ? node->kernel : "none",
                ck_codegen_v2_dtype_name(node->kernel_dtype));
        for (int b = 0; b < node->n_bindings; ++b) {
            const CKIRV2Binding *bind = &node->bindings[b];
            const char *buf_name = "unknown";
            if (bind->buffer >= 0 && bind->buffer < graph->num_buffers) {
                buf_name = graph->buffers[bind->buffer].name ? graph->buffers[bind->buffer].name : "unnamed";
            }
            fprintf(out,
                    "    /*   bind: %s -> %s */\n",
                    bind->arg ? bind->arg : "arg",
                    buf_name);
        }
        fprintf(out, "    ck_v2_dispatch_node(%d);\n", i);
    }
    fprintf(out, "    (void)%s;\n", runtime_sym ? runtime_sym : "ck_v2_runtime");
    fprintf(out, "}\n\n");
}

void ck_codegen_v2_emit_schedule(FILE *out,
                                 const CKIRV2Graph *graph,
                                 const char *prefill_runtime,
                                 const char *decode_runtime,
                                 const char *backward_runtime)
{
    emit_schedule_block(out, graph, "ck_v2_run_prefill", "prefill",
                        prefill_runtime ? prefill_runtime : "ck_v2_prefill_runtime");
    emit_schedule_block(out, graph, "ck_v2_run_decode", "decode",
                        decode_runtime ? decode_runtime : "ck_v2_decode_runtime");
    emit_schedule_block(out, graph, "ck_v2_run_backward", "backward",
                        backward_runtime ? backward_runtime : "ck_v2_backward_runtime");

    fprintf(out,
            "void ck_v2_run_forward(void) {\n"
            "    ck_v2_run_prefill();\n"
            "}\n\n");
}
