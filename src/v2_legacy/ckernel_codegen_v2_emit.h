#ifndef CKERNEL_CODEGEN_V2_EMIT_H
#define CKERNEL_CODEGEN_V2_EMIT_H

#include "ckernel_ir_v2.h"
#include "ckernel_mem_plan.h"

#include <stdio.h>

int ck_codegen_v2_emit_preamble(FILE *out);
void ck_codegen_v2_emit_struct(FILE *out,
                               const CKIRV2Graph *graph,
                               const CKMemPlan *plan,
                               const char *tag);
void ck_codegen_v2_emit_dispatch(FILE *out, const CKIRV2Graph *graph);
void ck_codegen_v2_emit_schedule(FILE *out,
                                 const CKIRV2Graph *graph,
                                 const char *prefill_runtime,
                                 const char *decode_runtime,
                                 const char *backward_runtime);
void ck_codegen_v2_emit_sections(FILE *out,
                                 const CKIRV2Graph *graph,
                                 const CKMemPlan *prefill_plan,
                                 const CKMemPlan *decode_plan,
                                 const CKMemPlan *backward_plan);

#endif /* CKERNEL_CODEGEN_V2_EMIT_H */
