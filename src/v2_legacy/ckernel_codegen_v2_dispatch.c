#include "ckernel_codegen_v2_emit.h"

#include <stdio.h>

void ck_codegen_v2_emit_dispatch(FILE *out, const CKIRV2Graph *graph)
{
    (void)graph;
    fprintf(out,
            "static void ck_v2_dispatch_node(int node_id) {\n"
            "    /* TODO: wire kernel dispatch using node metadata. */\n"
            "    (void)node_id;\n"
            "}\n\n");
}
