#include "ckernel_codegen_v2_emit.h"

#include <stdio.h>

int ck_codegen_v2_emit_preamble(FILE *out)
{
    fprintf(out,
            "/* Auto-generated runtime from CKIRV2Graph.\n"
            " * This v2 emitter currently writes the schedule and buffer layout\n"
            " * stubs. Kernel wiring will be layered on as IR v2 grows.\n"
            " */\n\n");
    fprintf(out,
            "#include <stddef.h>\n"
            "#include <stdint.h>\n"
            "#include <stdio.h>\n"
            "#include <string.h>\n\n");
    return 0;
}

void ck_codegen_v2_emit_struct(FILE *out,
                               const CKIRV2Graph *graph,
                               const CKMemPlan *plan,
                               const char *tag)
{
    const char *suffix = tag ? tag : "prefill";
    fprintf(out, "typedef struct {\n"
                 "    size_t offset_bytes;\n"
                 "    size_t size_bytes;\n"
                 "    int arena;\n"
                 "} CKV2BufferLayout;\n\n");

    fprintf(out, "static const CKV2BufferLayout ck_v2_%s_buffers[] = {\n", suffix);
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKMemSpan *span = &plan->spans[i];
        fprintf(out,
                "    { %zu, %zu, %d }, /* %s */\n",
                span->offset_bytes,
                span->size_bytes,
                (int)span->arena,
                graph->buffers[i].name ? graph->buffers[i].name : "unnamed");
    }
    fprintf(out, "};\n\n");

    fprintf(out, "static const size_t ck_v2_%s_total_bytes[] = {\n", suffix);
    for (int i = 0; i < CK_MEM_ARENA_COUNT; ++i) {
        fprintf(out, "    %zu%s\n",
                plan->total_bytes[i],
                (i + 1 == CK_MEM_ARENA_COUNT) ? "" : ",");
    }
    fprintf(out, "};\n\n");

    fprintf(out, "typedef struct {\n"
                 "    const CKV2BufferLayout *buffers;\n"
                 "    int num_buffers;\n"
                 "    const size_t *total_bytes;\n"
                 "    size_t alignment_bytes;\n"
                 "} CKV2Runtime;\n\n");

    fprintf(out, "static const CKV2Runtime ck_v2_%s_runtime = {\n"
                 "    ck_v2_%s_buffers,\n"
                 "    %d,\n"
                 "    ck_v2_%s_total_bytes,\n"
                 "    %d\n"
                 "};\n\n",
            suffix,
            suffix,
            graph->num_buffers,
            suffix,
            CK_MEM_PLAN_DEFAULT_ALIGN);
}
