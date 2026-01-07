#include "ckernel_codegen_v2.h"
#include "ckernel_codegen_v2_emit.h"
#include "ckernel_mem_plan.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ck_codegen_v2_emit_runtime(const CKIRV2Graph *graph,
                               const char *path,
                               CKEmitMode mode)
{
    if (!graph || !path) {
        return -1;
    }

    FILE *out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ck_codegen_v2_emit_runtime: failed to open %s: %s\n",
                path, strerror(errno));
        return -1;
    }

    CKMemPlan prefill_plan = {0};
    CKMemPlan decode_plan = {0};
    CKMemPlan backward_plan = {0};

    if (ck_mem_plan_build_inference_with_tokens(graph, &prefill_plan,
                                                CK_MEM_PLAN_DEFAULT_ALIGN, -1) != 0 ||
        ck_mem_plan_build_inference_with_tokens(graph, &decode_plan,
                                                CK_MEM_PLAN_DEFAULT_ALIGN, 1) != 0 ||
        ck_mem_plan_build_training_with_tokens(graph, &backward_plan,
                                               CK_MEM_PLAN_DEFAULT_ALIGN, -1) != 0) {
        fclose(out);
        ck_mem_plan_free(&prefill_plan);
        ck_mem_plan_free(&decode_plan);
        ck_mem_plan_free(&backward_plan);
        return -1;
    }

    ck_codegen_v2_emit_preamble(out);
    ck_codegen_v2_emit_struct(out, graph, &prefill_plan, "prefill");
    ck_codegen_v2_emit_struct(out, graph, &decode_plan, "decode");
    ck_codegen_v2_emit_struct(out, graph, &backward_plan, "backward");
    ck_codegen_v2_emit_sections(out, graph,
                                &prefill_plan,
                                &decode_plan,
                                &backward_plan);
    ck_codegen_v2_emit_dispatch(out, graph);
    ck_codegen_v2_emit_schedule(out, graph,
                                "ck_v2_prefill_runtime",
                                "ck_v2_decode_runtime",
                                "ck_v2_backward_runtime");

    if (mode == CK_EMIT_STANDALONE) {
        fprintf(out,
                "int main(void) {\n"
                "    ck_v2_run_forward();\n"
                "    return 0;\n"
                "}\n");
    }

    ck_mem_plan_free(&prefill_plan);
    ck_mem_plan_free(&decode_plan);
    ck_mem_plan_free(&backward_plan);
    fclose(out);
    return 0;
}
