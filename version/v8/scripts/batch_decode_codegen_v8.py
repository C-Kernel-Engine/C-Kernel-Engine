"""Emit a narrow, generated two-sequence decode step from an admitted IR split."""

from __future__ import annotations

from typing import Any


def emit_full_layer_batch_api(plan: dict[str, Any], *,
                              prefill_phase: bool = False) -> str:
    """Emit checked two-row projection stages across admitted decoder layers."""
    offsets: dict[str, int] = {}
    total = 0
    for ref, size in sorted(plan["extents"].items()):
        total = (total + 63) & ~63
        offsets[ref] = total
        total += 2 * size
    total = (total + 63) & ~63
    projections = [projection for stage in plan["stages"]
                   for projection in stage.get("projections", [])]
    compact_input_bytes = max(
        (projection["input_bytes"] for projection in projections
         if plan["extents"][projection["input"]] != projection["input_bytes"]),
        default=0,
    )
    compact_output_bytes = max(
        (projection["output_bytes"] for projection in projections
         if plan["extents"][projection["output"]] != projection["output_bytes"]),
        default=0,
    )
    compact_input_offset = total
    total = (total + 2 * compact_input_bytes + 63) & ~63
    compact_output_offset = total
    total = (total + 2 * compact_output_bytes + 63) & ~63
    recurrent_overlap = ""
    if plan.get("recurrent_state_bytes"):
        state_bytes = plan["recurrent_state_bytes"]
        recurrent_overlap = "\n".join(
            f"""        if (ck_batch_ranges_overlap(workspace, {total}u, live->{name},
                                    (size_t){state_bytes[name]}u) != 0 ||
            ck_batch_ranges_overlap(rows, 2 * sizeof(*rows), live->{name},
                                    (size_t){state_bytes[name]}u) != 0 ||
            ck_batch_ranges_overlap(rows[0].logits, output_bytes, live->{name},
                                    (size_t){state_bytes[name]}u) != 0 ||
            ck_batch_ranges_overlap(rows[1].logits, output_bytes, live->{name},
                                    (size_t){state_bytes[name]}u) != 0) return -2;"""
            for name in ("conv", "ssm")
        )
    work: list[str] = []
    previous_live: list[str] = []
    snapshot_copy_bytes = 0
    packing_copy_bytes = 0
    for number, stage in enumerate(plan["stages"]):
        if stage["kind"] == "local":
            stage_call = f"ck_batch_stage_{number}(g_model, rows[i].token);"
            if prefill_phase:
                stage_call = (
                    f"if (prefill_row == (int)i) "
                    f"ck_batch_prefill_stage_{number}(g_model, rows[i].token); "
                    f"else {stage_call}"
                )
            lines = ["    for (size_t i = 0; i < 2; ++i) {",
                     "        /* All validation is complete before either row advances. */",
                     "        if (ck_model_sequence_state_activate(rows[i].sequence_handle) != 0) return -2;"]
            for ref in previous_live:
                size, macro = plan["extents"][ref], plan["buffers"][ref]
                lines.append(f"        memcpy(g_model->bump + {macro}, base + {offsets[ref]}u + i * {size}u, {size}u);")
                snapshot_copy_bytes += 2 * size
            if stage is plan["stages"][-1]:
                lines.extend(["        g_ck_skip_decode_logits = 0;",
                              f"        {stage_call}",
                              "        memcpy(rows[i].logits, g_model->logits, (size_t)VOCAB_SIZE * sizeof(float));"])
            else:
                lines.append(f"        {stage_call}")
                for ref in stage["live_after"]:
                    size, macro = plan["extents"][ref], plan["buffers"][ref]
                    lines.append(f"        memcpy(base + {offsets[ref]}u + i * {size}u, g_model->bump + {macro}, {size}u);")
                    snapshot_copy_bytes += 2 * size
            lines.append("    }")
            work.extend(lines)
        else:
            for projection in stage["projections"]:
                src, dst = projection["input"], projection["output"]
                source = f"base + {offsets[src]}u"
                output = f"base + {offsets[dst]}u"
                if plan["extents"][src] != projection["input_bytes"]:
                    size = projection["input_bytes"]
                    work.append(
                        f"    for (size_t i = 0; i < 2; ++i) "
                        f"memcpy(base + {compact_input_offset}u + i * {size}u, "
                        f"base + {offsets[src]}u + i * {plan['extents'][src]}u, {size}u);"
                    )
                    source = f"base + {compact_input_offset}u"
                    packing_copy_bytes += 2 * size
                if plan["extents"][dst] != projection["output_bytes"]:
                    output = f"base + {compact_output_offset}u"
                work.append(
                    f"    {projection['function']}((const {'void' if projection['dtype'] == 'q8_k' else 'float'} *)({source}), "
                    f"g_model->bump + {projection['weight']}, NULL, "
                    f"(float *)({output}), 2, {projection['output_dim']}, {projection['input_dim']});"
                )
                if plan["extents"][dst] != projection["output_bytes"]:
                    size = projection["output_bytes"]
                    work.append(
                        f"    for (size_t i = 0; i < 2; ++i) "
                        f"memcpy(base + {offsets[dst]}u + i * {plan['extents'][dst]}u, "
                        f"base + {compact_output_offset}u + i * {size}u, {size}u);"
                    )
                    packing_copy_bytes += 2 * size
        previous_live = stage.get("live_after", [])
    execution = "\n".join(work).replace("memcpy(", "ck_batch_memcpy(")
    entry_name = ("ck_model_decode_batch2_impl" if prefill_phase else
                  "ck_model_decode_batch2")
    entry_export = "static " if prefill_phase else "CK_EXPORT "
    phase_argument = ", int prefill_row" if prefill_phase else ""
    wrappers = ""
    if prefill_phase:
        wrappers = '''
CK_EXPORT int ck_model_decode_batch2(const CKModelBatchDecodeRowV8 *rows,
                                      size_t count, void *workspace,
                                      size_t workspace_bytes) {
    return ck_model_decode_batch2_impl(rows, count, workspace, workspace_bytes, -1);
}

CK_EXPORT int ck_model_decode_batch2_mixed_prefill(
    const CKModelBatchDecodeRowV8 *rows, size_t count, void *workspace,
    size_t workspace_bytes, uint32_t prefill_row) {
    if (prefill_row >= 2) return -2;
    return ck_model_decode_batch2_impl(rows, count, workspace,
                                       workspace_bytes, (int)prefill_row);
}
'''
    return f'''
#include <time.h>
/* Generated from checked IR cuts. {plan['projection_groups']} shared projection
 * groups span {len(plan['shared_layers'])} layer(s); attention and KV updates remain local. */
static uint64_t g_ck_batch_copy_ns;
static int g_ck_batch_profile_copies;
static uint64_t ck_batch_clock_ns(void) {{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * 1000000000u + (uint64_t)now.tv_nsec;
}}
static void ck_batch_memcpy(void *dst, const void *src, size_t bytes) {{
    if (!g_ck_batch_profile_copies) {{ memcpy(dst, src, bytes); return; }}
    uint64_t begin = ck_batch_clock_ns();
    memcpy(dst, src, bytes);
    uint64_t end = ck_batch_clock_ns();
    if (end >= begin) g_ck_batch_copy_ns += end - begin;
}}
CK_EXPORT uint64_t ck_model_batch_decode_last_copy_ns(void) {{
    return g_ck_batch_copy_ns;
}}
static int ck_batch_ranges_overlap(const void *left, size_t left_bytes,
                                   const void *right, size_t right_bytes) {{
    uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
    if (!a || !b || a > UINTPTR_MAX - left_bytes ||
        b > UINTPTR_MAX - right_bytes) return -1;
    return a < b + right_bytes && b < a + left_bytes;
}}

CK_EXPORT int ck_model_batch_decode_workspace(size_t *bytes, size_t *alignment) {{
    if (!bytes || !alignment) return -1;
    *bytes = {total}u;
    *alignment = 64u;
    return 0;
}}

CK_EXPORT int ck_model_batch_decode_projection_groups(void) {{
    return {plan['projection_groups']};
}}

CK_EXPORT int ck_model_batch_decode_shared_layers(void) {{
    return {len(plan['shared_layers'])};
}}

CK_EXPORT size_t ck_model_batch_decode_snapshot_copy_bytes(void) {{
    return {snapshot_copy_bytes}u;
}}

CK_EXPORT size_t ck_model_batch_decode_packing_copy_bytes(void) {{
    return {packing_copy_bytes}u;
}}

{entry_export}int {entry_name}(const CKModelBatchDecodeRowV8 *rows,
                                      size_t count, void *workspace,
                                      size_t workspace_bytes{phase_argument}) {{
    if (!g_model || !rows || count != 2 || !workspace) return -1;
    if (((uintptr_t)workspace & 63u) || workspace_bytes < {total}u) return -3;
    if (ck_model_cancel_requested() || getenv("CK_STOP_OP") ||
        getenv("CK_DEBUG_IMPORT_HIDDEN") || getenv("CK_V8_DEBUG_ATTN_PROJ_FP32_LAYER") ||
        getenv("CK_V8_DEBUG_ATTN_PROJ_FP32_OP") ||
        getenv("CK_V7_DEBUG_OUTPROJ_FP32") || getenv("CK_V7_DEBUG_OUTPROJ_FP32_LAYER") ||
        getenv("CK_V7_DEBUG_MLP_DOWN_FP32") || getenv("CK_V7_DEBUG_MLP_DOWN_FP32_LAYER") ||
        getenv("CK_V8_DEBUG_MLP_GATE_UP_FP32") || getenv("CK_V8_DEBUG_MLP_GATE_UP_FP32_LAYER") ||
        getenv("CK_V8_DEBUG_MLP_GATE_UP_Q8_CONTRACT") || getenv("CK_V8_DEBUG_MLP_GATE_UP_Q8_CONTRACT_LAYER")) return -2;
    CKSequenceStateV8 *states[2];
    for (size_t i = 0; i < 2; ++i) {{
        const CKModelBatchDecodeRowV8 *row = &rows[i];
        states[i] = ck_sequence_find(row->sequence_handle);
        if (!states[i] || !row->logits || row->row_offset != i ||
            row->token_count != 1 || row->token < 0 || row->token >= VOCAB_SIZE ||
            row->position < 0 || row->position >= MAX_SEQ_LEN ||
            row->position != (states[i] == g_active_sequence ? g_model->pos : states[i]->pos))
            return -2;
    }}
    if (states[0] == states[1] || rows[0].logits == rows[1].logits) return -2;
    const size_t output_bytes = (size_t)VOCAB_SIZE * sizeof(float);
    if (ck_batch_ranges_overlap(workspace, {total}u, rows, 2 * sizeof(*rows)) != 0 ||
        ck_batch_ranges_overlap(workspace, {total}u, g_model->bump, g_model->bump_size) != 0 ||
        ck_batch_ranges_overlap(rows, 2 * sizeof(*rows),
                                g_model->bump, g_model->bump_size) != 0 ||
        ck_batch_ranges_overlap(rows[0].logits, output_bytes,
                                rows[1].logits, output_bytes) != 0) return -2;
    for (size_t j = 0; j < 3; ++j) {{
        CKSequenceStateV8 *live = &g_sequence_states[j];
        if (!live->occupied) continue;
        if (ck_batch_ranges_overlap(workspace, {total}u, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows, 2 * sizeof(*rows), live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows[0].logits, output_bytes, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows[1].logits, output_bytes, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0) return -2;
{recurrent_overlap}
    }}
    for (size_t i = 0; i < 2; ++i) {{
        if (ck_batch_ranges_overlap(rows[i].logits, output_bytes, rows,
                                    2 * sizeof(*rows)) != 0 ||
            ck_batch_ranges_overlap(rows[i].logits, output_bytes, workspace, {total}u) != 0 ||
            ck_batch_ranges_overlap(rows[i].logits, output_bytes, g_model->bump,
                                    g_model->bump_size) != 0) return -2;
    }}
    uint8_t *base = (uint8_t *)workspace;
    const char *copy_profile = getenv("CK_BATCH_PROFILE_COPIES");
    g_ck_batch_profile_copies = copy_profile && copy_profile[0] && copy_profile[0] != '0';
    g_ck_batch_copy_ns = 0;
{execution}
    return 0;
}}
{wrappers}
'''


def emit_two_row_batch_api(contract: dict[str, Any]) -> str:
    b = contract["buffers"]
    size = contract["sizes"]
    weight_q, weight_k = contract["weights"]
    gemm = contract["gemm_function"]
    # Each section is packed by row. The input section is the GEMM's M=2 A
    # matrix; Q/K sections are its two independently addressed output rows.
    input_bytes = size["input"]
    residual_bytes = size["residual"]
    q_bytes = size["q"]
    k_bytes = size["k"]
    total = 2 * (input_bytes + residual_bytes + q_bytes + k_bytes)
    extension = contract.get("layer_extension")
    extra_buffers = ""
    extra_work = ""
    finish = ""
    debug_rejection = ""
    if extension:
        mid_input = extension["input_bytes"]
        mid_residual = extension["residual_bytes"]
        gateup_bytes = extension["output_bytes"]
        extra_start = total
        total += 2 * (mid_input + mid_residual + gateup_bytes)
        extra_buffers = f'''
    float *gateup_inputs = (float *)(base + {extra_start}u);
    uint8_t *gateup_residual = base + {extra_start + 2 * mid_input}u;
    float *gateup_output = (float *)(base + {extra_start + 2 * (mid_input + mid_residual)}u);'''
        extra_work = f'''
        ck_batch_decode_before_gateup(g_model, rows[i].token);
        memcpy((uint8_t *)gateup_inputs + i * {mid_input}u,
               g_model->bump + {extension['input_define']}, {mid_input}u);
        memcpy(gateup_residual + i * {mid_residual}u,
               g_model->bump + {extension['residual_define']}, {mid_residual}u);'''
        finish = f'''
    }}

    /* The attention/KV interval above is sequence-local. The first layer's
     * MLP gate/up projection now shares a second selected weight traversal. */
    {extension['gemm_function']}(gateup_inputs,
           g_model->bump + {extension['weight']}, NULL,
           gateup_output, 2, {extension['output_dim']}, {contract['input_dim']});

    for (size_t i = 0; i < 2; ++i) {{
        if (ck_model_sequence_state_activate(rows[i].sequence_handle) != 0) return -2;
        memcpy(g_model->bump + {extension['residual_define']},
               gateup_residual + i * {mid_residual}u, {mid_residual}u);
        memcpy(g_model->bump + {extension['output_define']},
               (uint8_t *)gateup_output + i * {gateup_bytes}u, {gateup_bytes}u);
        g_ck_skip_decode_logits = 0;
        ck_batch_decode_after_gateup(g_model, rows[i].token);
        memcpy(rows[i].logits, g_model->logits, VOCAB_SIZE * sizeof(float));'''
        debug_rejection = (' || getenv("CK_V8_DEBUG_MLP_GATE_UP_FP32")'
                           ' || getenv("CK_V8_DEBUG_MLP_GATE_UP_FP32_LAYER")'
                           ' || getenv("CK_V8_DEBUG_MLP_GATE_UP_Q8_CONTRACT")'
                           ' || getenv("CK_V8_DEBUG_MLP_GATE_UP_Q8_CONTRACT_LAYER")')
    else:
        finish = '''
        g_ck_skip_decode_logits = 0;
        ck_batch_decode_suffix(g_model, rows[i].token);
        memcpy(rows[i].logits, g_model->logits, VOCAB_SIZE * sizeof(float));'''
    return f'''
/* Generated two-row decode: map-selected projections, sequence-local state. */
static int ck_batch_ranges_overlap(const void *left, size_t left_bytes,
                                   const void *right, size_t right_bytes) {{
    uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
    if (!a || !b || a > UINTPTR_MAX - left_bytes ||
        b > UINTPTR_MAX - right_bytes) return -1;
    return a < b + right_bytes && b < a + left_bytes;
}}

CK_EXPORT int ck_model_batch_decode_workspace(size_t *bytes, size_t *alignment) {{
    if (!bytes || !alignment) return -1;
    *bytes = {total}u;
    *alignment = 64u;
    return 0;
}}

CK_EXPORT int ck_model_batch_decode_projection_groups(void) {{
    return {2 if extension else 1};
}}

CK_EXPORT int ck_model_decode_batch2(const CKModelBatchDecodeRowV8 *rows,
                                      size_t count, void *workspace,
                                      size_t workspace_bytes) {{
    if (!g_model || !rows || count != 2 || !workspace) return -1;
    if (((uintptr_t)workspace & 63u) || workspace_bytes < {total}u) return -3;
    if (ck_model_cancel_requested() || getenv("CK_STOP_OP") ||
        getenv("CK_DEBUG_IMPORT_HIDDEN") || getenv("CK_V8_DEBUG_ATTN_PROJ_FP32_LAYER"){debug_rejection}) return -2;
    CKSequenceStateV8 *states[2];
    for (size_t i = 0; i < 2; ++i) {{
        const CKModelBatchDecodeRowV8 *row = &rows[i];
        states[i] = ck_sequence_find(row->sequence_handle);
        if (!states[i] || !row->logits || row->row_offset != i ||
            row->token_count != 1 || row->token < 0 || row->token >= VOCAB_SIZE ||
            row->position < 0 || row->position >= MAX_SEQ_LEN ||
            row->position != (states[i] == g_active_sequence ? g_model->pos : states[i]->pos))
            return -2;
    }}
    if (states[0] == states[1] || rows[0].logits == rows[1].logits) return -2;
    const size_t output_bytes = (size_t)VOCAB_SIZE * sizeof(float);
    if (ck_batch_ranges_overlap(workspace, {total}u, rows, 2 * sizeof(*rows)) != 0 ||
        ck_batch_ranges_overlap(workspace, {total}u, g_model->bump, g_model->bump_size) != 0 ||
        ck_batch_ranges_overlap(rows, 2 * sizeof(*rows),
                                g_model->bump, g_model->bump_size) != 0 ||
        ck_batch_ranges_overlap(rows[0].logits, output_bytes,
                                rows[1].logits, output_bytes) != 0) return -2;
    for (size_t j = 0; j < 3; ++j) {{
        CKSequenceStateV8 *live = &g_sequence_states[j];
        if (!live->occupied) continue;
        if (ck_batch_ranges_overlap(workspace, {total}u, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows, 2 * sizeof(*rows), live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows[0].logits, output_bytes, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0 ||
            ck_batch_ranges_overlap(rows[1].logits, output_bytes, live->kv,
                                    (size_t)KV_CACHE_SIZE) != 0) return -2;
    }}
    for (size_t i = 0; i < 2; ++i) {{
        if (
            ck_batch_ranges_overlap(rows[i].logits, output_bytes, rows,
                                    2 * sizeof(*rows)) != 0 ||
            ck_batch_ranges_overlap(rows[i].logits, output_bytes, workspace, {total}u) != 0 ||
            ck_batch_ranges_overlap(rows[i].logits, output_bytes, g_model->bump,
                                    g_model->bump_size) != 0) return -2;
    }}
    uint8_t *base = (uint8_t *)workspace;
    float *inputs = (float *)base;
    uint8_t *residual = base + {2 * input_bytes}u;
    float *q = (float *)(residual + {2 * residual_bytes}u);
    float *k = (float *)((uint8_t *)q + {2 * q_bytes}u);
{extra_buffers}

    /* Neither prefix updates position or KV. Snapshot the only call-local
     * values the suffix consumes before admitting the second prefix. */
    for (size_t i = 0; i < 2; ++i) {{
        if (ck_model_sequence_state_activate(rows[i].sequence_handle) != 0) return -2;
        ck_batch_decode_prefix(g_model, rows[i].token);
        memcpy((uint8_t *)inputs + i * {input_bytes}u,
               g_model->bump + {b['input']}, {input_bytes}u);
        memcpy(residual + i * {residual_bytes}u,
               g_model->bump + {b['residual']}, {residual_bytes}u);
    }}

    /* Both calls use map-declared Q5_1/Q8_1 M=2 arithmetic. Each packed
     * weight block is unpacked once and used for both independent rows. */
    {gemm}(inputs, g_model->bump + {weight_q}, NULL,
           q, 2, {contract['q_dim']}, {contract['input_dim']});
    {gemm}(inputs, g_model->bump + {weight_k}, NULL,
           k, 2, {contract['k_dim']}, {contract['input_dim']});

    for (size_t i = 0; i < 2; ++i) {{
        if (ck_model_sequence_state_activate(rows[i].sequence_handle) != 0) return -2;
        memcpy(g_model->bump + {b['input']},
               (uint8_t *)inputs + i * {input_bytes}u, {input_bytes}u);
        memcpy(g_model->bump + {b['residual']},
               residual + i * {residual_bytes}u, {residual_bytes}u);
        memcpy(g_model->bump + {b['q']},
               (uint8_t *)q + i * {q_bytes}u, {q_bytes}u);
        memcpy(g_model->bump + {b['k']},
               (uint8_t *)k + i * {k_bytes}u, {k_bytes}u);
{extra_work}{finish}
    }}
    return 0;
}}
'''
