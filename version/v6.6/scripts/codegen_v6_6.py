#!/usr/bin/env python3
"""
codegen_v6_6.py - Generate C code from lowered IR.

RESPONSIBILITIES:
1. Create memory layout from layout.json (structs, offsets, allocations)
2. Parse lowered IR and emit function calls (unrolled, one after another)
3. Pass pointers cleanly to all functions

If there are memory issues → fix the memory layout builder, not codegen.
If there are kernel issues → fix the IR lower, not codegen.

===============================================================================
HARDCODED VALUES & MODEL-SPECIFIC ASSUMPTIONS - TECHNICAL DEBT TRACKER
===============================================================================

This section documents values that are hardcoded in codegen but should come from
IR config or dedicated kernels. These WILL BREAK for non-Qwen2 models.

Delete entries from this list as they are properly fixed.

NOTE: Init ops (rope_init, etc.) now use init_call.json pattern:
  manifest.config → init.json → init_call.json → codegen emits calls
  This is the correct pattern for model-specific initialization.

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ROPE SCALING TYPE - MEDIUM                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: rope_precompute_cache kernel                                      │
│ Current: Standard RoPE only (no scaling)                                    │
│ Should be: Support for rope_scaling_type from config:                       │
│   - "linear": freq *= 1/scaling_factor                                      │
│   - "dynamic": NTK-aware dynamic scaling                                    │
│   - "yarn": YaRN (Yet another RoPE extensioN)                               │
│                                                                             │
│ Impact: Context extension won't work for models using scaled RoPE           │
│   - Llama 3.1 uses scaled RoPE for 128K context                             │
│   - Code Llama uses linear scaling                                          │
│                                                                             │
│ Fix: Extend rope_precompute_cache kernel to accept scaling_type param       │
│      init.json already has rope_scaling_type field ready to use             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. ROPE MEMORY LAYOUT - MEDIUM                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: init_call.json args for rope_precompute_cache                     │
│ Current: Half-dimension interleaved layout                                  │
│   sin_cache = cos_cache + MAX_SEQ_LEN * HEAD_DIM / 2                        │
│                                                                             │
│ Should be: Layout from config.get("rope_layout") or rotary_dim              │
│   - Some models use full HEAD_DIM (not HEAD_DIM/2)                          │
│   - Some use [cos, sin] interleaved per position                            │
│   - rotary_dim may be < head_dim (partial rotation)                         │
│                                                                             │
│ Impact: Wrong RoPE application for models with different layouts            │
│                                                                             │
│ Fix: generate_init_ir_lower_3() should read rotary_dim from config          │
│      and adjust buffer expressions accordingly                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. ACTIVATION FUNCTION - CRITICAL                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: emit_model_and_api() kernel declarations, line ~413               │
│ Hardcoded: swiglu_forward() only                                            │
│ Should be: Based on config.get("hidden_act")                                │
│                                                                             │
│ Impact: WRONG MLP activation for non-SwiGLU models                          │
│   - GPT-2, GPT-Neo, OPT: GELU                                               │
│   - Older GPT: ReLU                                                         │
│   - Some models: SiLU (without gating)                                      │
│   - Qwen2, Llama, Mistral: SwiGLU (current hardcoded)                       │
│                                                                             │
│ Fix: IR should specify activation kernel, codegen emits what IR says        │
│ Note: IR does map silu_mul -> swiglu, but codegen doesn't check hidden_act  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. ATTENTION SOFTMAX SCALE - LOW                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: Implicit in attention kernels                                     │
│ Hardcoded: 1/sqrt(head_dim) in kernel implementation                        │
│ Should be: config.get("attention_scale") or computed from head_dim          │
│                                                                             │
│ Impact: Minor - most models use 1/sqrt(d), but some override                │
│   - Falcon uses different scaling                                           │
│   - Some models use learned scale parameter                                 │
│                                                                             │
│ Fix: Attention kernel should accept scale parameter from IR                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. KV CACHE LAYOUT - MEDIUM                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: do_init() line ~440, kv_cache_store kernel calls                  │
│ Hardcoded: Head-major layout [num_layers, 2, num_kv_heads, seq_len, head_d] │
│ Should be: config.get("kv_layout") or IR memory spec                        │
│                                                                             │
│ Impact: Incompatible with paged attention or different cache layouts        │
│   - vLLM uses paged KV cache                                                │
│   - Some implementations use [batch, layers, heads, seq, dim]               │
│                                                                             │
│ Fix: KV cache layout should be defined in memory_planner, not codegen       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. DEFAULT CONFIG VALUES (Qwen2-0.5B specific) - LOW                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: emit_memory_layout() lines ~96-103                                │
│ Hardcoded defaults:                                                         │
│   embed_dim: 896          (Qwen2-0.5B, Llama-7B uses 4096)                  │
│   num_heads: 14           (Qwen2-0.5B, Llama-7B uses 32)                    │
│   num_kv_heads: 2         (Qwen2-0.5B GQA 7:1, Llama-2 uses MHA)            │
│   head_dim: 64            (common, but some use 128)                        │
│   intermediate_size: 4864 (Qwen2-0.5B, Llama-7B uses 11008)                 │
│   vocab_size: 151936      (Qwen family, Llama uses 32000-128256)            │
│   context_length: 32768   (Qwen2, varies widely)                            │
│                                                                             │
│ Impact: LOW if config is properly provided (defaults only used as fallback) │
│ These are just defaults - real values should always come from manifest      │
│                                                                             │
│ Fix: Change defaults to more conservative/common values, or remove defaults │
│ and fail explicitly if config missing                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. BUMP FILE LAYOUT DEFAULTS - LOW                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: emit_memory_layout() lines ~68-72                                 │
│ Hardcoded defaults:                                                         │
│   header_size: 128                                                          │
│   ext_metadata_size: 24                                                     │
│   data_start: 152                                                           │
│                                                                             │
│ Impact: LOW - converter should always provide these in layout.json          │
│ These should NEVER be used; if they are, converter is broken                │
│                                                                             │
│ Fix: Remove defaults and fail explicitly if bump_layout missing             │
└─────────────────────────────────────────────────────────────────────────────┘

===============================================================================
WHAT CODEGEN DOES CORRECTLY (keep these patterns)
===============================================================================

✓ Reads ops directly from IR - emit_op() emits exactly what IR Lower provides
✓ Uses layout.json for memory offsets - no offset calculations in codegen
✓ Kernel function names from IR - doesn't hardcode which kernel to call
✓ Layer count from config - unrolls based on num_layers from manifest
✓ Token offset from layout - gets from token_ids buffer in layout.json
✓ Weight offsets from layout - all W_* macros come from layout

===============================================================================
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# =============================================================================
# PART 1: MEMORY LAYOUT IN C
# =============================================================================

def _sanitize_macro(name: str) -> str:
    out = []
    prev_us = False
    for ch in name:
        if ch.isalnum():
            out.append(ch.upper())
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s = "".join(out).strip("_")
    if not s:
        s = "UNNAMED"
    if s[0].isdigit():
        s = f"N_{s}"
    return s


def emit_memory_layout(layout: Dict, config: Dict) -> str:
    """Emit C code for memory layout from layout.json."""

    weights = layout.get("memory", {}).get("weights", {})
    activations = layout.get("memory", {}).get("activations", {})

    weights_size = weights.get("size", 0)

    # Compute activations size from buffers
    act_size = 0
    for buf in activations.get("buffers", []):
        end = buf.get("offset", 0) + buf.get("size", 0)
        act_size = max(act_size, end)

    kv_cache_size = None
    for buf in activations.get("buffers", []):
        if buf.get("name") == "kv_cache":
            kv_cache_size = int(buf.get("size", 0))
            break

    num_layers = config.get("num_layers", 24)

    # Get bump_layout from layout (passed through from manifest via build_ir)
    bump_layout = layout.get("bump_layout", {
        "header_size": 128,
        "ext_metadata_size": 24,
        "data_start": 152,
    })

    lines = []

    # Bump file layout constants (from converter via manifest → build_ir → layout.json)
    lines.append(f'''
/* ============================================================================
 * BUMP FILE LAYOUT (from converter)
 * ============================================================================
 * These constants define the .bump file structure:
 *   [0..BUMP_HEADER_SIZE)           : Header
 *   [BUMP_HEADER_SIZE..BUMP_DATA_START) : Extended metadata
 *   [BUMP_DATA_START..]             : dtype_table + weights
 * ============================================================================ */
#define BUMP_HEADER_SIZE {bump_layout.get("header_size", 128)}
#define BUMP_EXT_METADATA_SIZE {bump_layout.get("ext_metadata_size", 24)}
#define BUMP_DATA_START {bump_layout.get("data_start", 152)}
''')

    # Model dimensions
    lines.append(f'''
/* ============================================================================
 * MODEL CONFIGURATION
 * ============================================================================ */
#define EMBED_DIM {config.get("embed_dim", 896)}
#define NUM_HEADS {config.get("num_heads", 14)}
#define NUM_KV_HEADS {config.get("num_kv_heads", 2)}
#define HEAD_DIM {config.get("head_dim", 64)}
#define INTERMEDIATE_SIZE {config.get("intermediate_size", 4864)}
#define NUM_LAYERS {num_layers}
#define VOCAB_SIZE {config.get("vocab_size", 151936)}
#define MAX_SEQ_LEN {config.get("context_length", 32768)}

/* Memory sizes */
#define WEIGHTS_SIZE {weights_size}ULL
#define ACTIVATIONS_SIZE {act_size}ULL
''')
    if kv_cache_size:
        lines.append(f"#define KV_CACHE_SIZE {kv_cache_size}ULL")
    else:
        lines.append("#define KV_CACHE_SIZE (NUM_LAYERS * 2 * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * sizeof(float))")
    lines.append("")

    # Header offsets struct
    header_entries = [e for e in weights.get("entries", []) if "layer" not in e.get("name", "")]
    vocab_merges_size = 0
    for e in header_entries:
        if e.get("name") == "vocab_merges":
            vocab_merges_size = int(e.get("size", 0))
            break
    lines.append("/* Header weight offsets */")
    lines.append("typedef struct {")
    for e in header_entries:
        lines.append(f"    size_t {e['name']};  /* {e.get('dtype', 'unknown')}, {e.get('size', 0)} bytes */")
    lines.append("} HeaderOffsets;")
    lines.append("")
    lines.append("static const HeaderOffsets L_HEADER = {")
    for e in header_entries:
        lines.append(f"    .{e['name']} = {e.get('offset', 0)},")
    lines.append("};")
    lines.append("")
    if vocab_merges_size:
        lines.append(f"#define VOCAB_MERGES_COUNT {vocab_merges_size // 4}")
    else:
        lines.append("#define VOCAB_MERGES_COUNT 0")
    lines.append("")

    # Layer offsets struct - get fields from layer 0
    layer0_entries = [e for e in weights.get("entries", []) if e.get("name", "").startswith("layer.0.")]
    field_names = sorted(set(e["name"].replace("layer.0.", "") for e in layer0_entries))

    lines.append("/* Per-layer weight offsets */")
    lines.append("typedef struct {")
    for field in field_names:
        lines.append(f"    size_t {field};")
    lines.append("} LayerOffsets;")
    lines.append("")

    # Layer offset values
    lines.append(f"static const LayerOffsets L_LAYERS[{num_layers}] = {{")
    for layer_idx in range(num_layers):
        layer_entries = [e for e in weights.get("entries", [])
                        if e.get("name", "").startswith(f"layer.{layer_idx}.")]
        lines.append(f"    [{layer_idx}] = {{")
        for e in layer_entries:
            field = e["name"].replace(f"layer.{layer_idx}.", "")
            lines.append(f"        .{field} = {e.get('offset', 0)},")
        lines.append("    },")
    lines.append("};")
    lines.append("")

    # Memory allocation offsets
    weights_base = weights.get('base_offset', 0)
    arena = layout.get("memory", {}).get("arena", {})
    act_base = int(arena.get("activations_base", weights_base + weights_size))
    total_size = int(arena.get("total_size", act_base + act_size))
    layout_mode = arena.get("mode", "region")
    lines.append(f'''
/* Memory layout offsets (single contiguous allocation) */
#define BUMP_WEIGHTS_OFFSET {weights_base}
#define BUMP_ACT_OFFSET {act_base}
#define BUMP_TOTAL_SIZE {total_size}ULL
#define LAYOUT_MODE_PACKED {1 if layout_mode == "packed" else 0}
''')

    # Absolute offsets (single bump base)
    lines.append("/* Absolute offsets (bump base) */")
    for e in weights.get("entries", []):
        macro = e.get("define") or f"W_{_sanitize_macro(e.get('name', 'weight'))}"
        abs_off = e.get("abs_offset", weights_base + e.get("offset", 0))
        lines.append(f"#define {macro} {abs_off}")
    for buf in activations.get("buffers", []):
        macro = buf.get("define") or f"A_{_sanitize_macro(buf.get('name', 'buffer'))}"
        abs_off = buf.get("abs_offset", act_base + buf.get("offset", 0))
        lines.append(f"#define {macro} {abs_off}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# PART 2: EMIT FUNCTION CALLS FROM IR LOWER
# =============================================================================

def _collect_layout_defines(layout: Dict) -> tuple[dict, int]:
    """Collect macro -> absolute offset mapping and total size from layout."""
    memory = layout.get("memory", {})
    weights = memory.get("weights", {})
    activations = memory.get("activations", {})
    arena = memory.get("arena", {})
    weights_base = int(weights.get("base_offset", 0))
    act_base = int(arena.get("activations_base", weights_base + int(weights.get("size", 0))))
    total_size = int(arena.get("total_size", act_base + int(activations.get("size", 0))))

    defines = {}
    for e in weights.get("entries", []):
        macro = e.get("define") or f"W_{_sanitize_macro(e.get('name', 'weight'))}"
        abs_off = int(e.get("abs_offset", weights_base + int(e.get("offset", 0))))
        defines[macro] = abs_off
    for buf in activations.get("buffers", []):
        macro = buf.get("define") or f"A_{_sanitize_macro(buf.get('name', 'buffer'))}"
        abs_off = int(buf.get("abs_offset", act_base + int(buf.get("offset", 0))))
        defines[macro] = abs_off

    defines["BUMP_WEIGHTS_OFFSET"] = weights_base
    defines["BUMP_ACT_OFFSET"] = act_base
    return defines, total_size


def _max_bump_offset_from_ir(ir: Dict, defines: dict) -> tuple[int, set]:
    """Return max offset used with model->bump and any unknown macros."""
    ops = ir.get("ops", ir.get("operations", []))
    max_off = 0
    unknown = set()
    bump_pattern = re.compile(r"model->bump\\s*\\+\\s*([A-Za-z_][A-Za-z0-9_]*|\\d+)")

    for op in ops:
        for arg in op.get("args", []):
            expr = arg.get("expr", "")
            if "model->bump" not in expr:
                continue
            for m in bump_pattern.finditer(expr):
                tok = m.group(1)
                if tok.isdigit():
                    off = int(tok)
                else:
                    off = defines.get(tok)
                    if off is None:
                        unknown.add(tok)
                        continue
                if off > max_off:
                    max_off = off
    return max_off, unknown


def _guard_bump_offsets(layout: Dict, ir_list: List[Dict]) -> None:
    """Fail fast if any IR uses model->bump + offset beyond BUMP_TOTAL_SIZE."""
    defines, total_size = _collect_layout_defines(layout)
    max_off = 0
    unknown = set()
    for ir in ir_list:
        off, unk = _max_bump_offset_from_ir(ir, defines)
        if off > max_off:
            max_off = off
        unknown |= unk
    if max_off >= total_size:
        raise RuntimeError(
            f"Codegen guard failed: max bump offset {max_off} >= BUMP_TOTAL_SIZE {total_size}. "
            f"Check that decode and prefill use a shared layout."
        )
    if unknown:
        print(f"Warning: Unresolved bump macros in IR: {sorted(unknown)[:5]}")

def emit_op(op: Dict, seq_idx: int | None = None, debug: bool = False) -> str:
    """Emit a single function call from an IR operation.

    Just read the op and emit the call. No special cases.
    IR Lower 3 provides call-ready args with exact expressions.

    If debug=True, emit printf statements to dump output buffer values.
    """
    function = op.get("function", op.get("kernel", "unknown"))
    idx = op.get("idx", 0)
    layer = op.get("layer", -1)
    section = op.get("section", "")
    op_name = op.get("op", "unknown")
    args = op.get("args", [])

    lines = []
    lines.append(f"    /* Op {idx}: {function} ({op_name}) layer={layer} section={section} */")
    if not args:
        lines.append(f"    {function}();")
        return "\n".join(lines)

    lines.append(f"    {function}(")
    for i, arg in enumerate(args):
        expr = arg.get("expr", "0")
        comma = "," if i < len(args) - 1 else ""
        lines.append(f"        {expr}{comma}")
    lines.append("    );")

    if seq_idx is not None:
        lines.append(f"    if (stop_seq == {seq_idx}) return;")

    # Add debug output if enabled
    if debug:
        # Find output buffer (usually the arg with "output" or "C" in name, or casted to non-const float*)
        output_expr = None
        for arg in args:
            name = arg.get("name", "").lower()
            source = arg.get("source", "").lower()
            expr = arg.get("expr", "")
            # Output args are typically non-const float pointers
            if "(float*)" in expr and "const" not in expr:
                output_expr = expr
                break
            if "output" in name or "output" in source or name == "c":
                output_expr = expr
                break

        if output_expr:
            # Remove cast to get raw pointer
            raw_expr = output_expr.replace("(float*)", "").replace("(void*)", "").strip()
            lines.append(f'    {{ float *_dbg = (float*){raw_expr}; '
                        f'printf("[Op {idx} {op_name} L{layer}] out[0..4]: %f %f %f %f %f\\n", '
                        f'_dbg[0], _dbg[1], _dbg[2], _dbg[3], _dbg[4]); }}')

    return "\n".join(lines)


def emit_decode_function(ops: List[Dict], token_offset: int, token_base: str, debug: bool = False) -> str:
    """Emit the decode function with all ops unrolled."""
    lines = []
    lines.append("""
/* ============================================================================
 * DECODE - Unrolled from IR Lower
 * ============================================================================ */
static void ck_decode(CKModel *model, int32_t token) {
    uint8_t *MEM = (uint8_t*)model->bump;
    uint8_t *ACT = (uint8_t*)model->activations;
    (void)ACT;
    const char *stop_env = getenv("CK_STOP_OP");
    int stop_seq = stop_env ? atoi(stop_env) : -1;
""")
    lines.append(f"    /* Store token at offset {token_offset} (from layout) */")
    lines.append(f"    *(int32_t*)({token_base} + {token_offset}) = token;")
    lines.append("")

    for seq_idx, op in enumerate(ops):
        lines.append(emit_op(op, seq_idx, debug=debug))
        lines.append("")

    lines.append("    model->pos++;")
    lines.append("}")
    return "\n".join(lines)


# =============================================================================
# PART 3: CLEAN API WITH POINTER PASSING
# =============================================================================

def emit_init_call(op: Dict) -> str:
    """Emit a single init op call from lowered init IR."""
    func = op.get("function", "unknown")
    args = op.get("args", [])
    arg_exprs = [arg.get("expr", "0") for arg in args]
    return f"    {func}({', '.join(arg_exprs)});"


def emit_model_and_api(init_call: Dict = None) -> str:
    """Emit model struct and clean API functions.

    Args:
        init_call: Lowered init IR (init_call.json) with call-ready ops.
                   If provided, emits init ops from IR (no hardcoding).
                   If None, emits empty init (no rope precompute).
    """
    # Generate init ops code from lowered IR
    init_ops_code = ""
    if init_call:
        ops = init_call.get("operations", [])
        if ops:
            init_lines = ["    /* Init ops from init_call.json */"]
            for op in ops:
                if not op.get("errors"):
                    init_lines.append(emit_init_call(op))
                else:
                    init_lines.append(f"    /* ERROR: {op.get('function')} - {op.get('errors')} */")
            init_ops_code = "\n".join(init_lines)

    # If no init_call provided, leave a placeholder comment
    if not init_ops_code:
        init_ops_code = "    /* No init ops (init_call.json not provided) */"

    return f'''
/* ============================================================================
 * MODEL STRUCT
 * ============================================================================ */
typedef struct {{
    uint8_t *bump;           /* Single contiguous allocation */
    size_t bump_size;
    uint8_t *bump_weights;   /* Weights section */
    float *activations;      /* Activations section */
    float *kv_cache;         /* KV cache section */
    float *rope_cos;         /* RoPE cos table */
    float *rope_sin;         /* RoPE sin table */
    float *logits;           /* Output logits */
    int pos;                 /* Current position */
}} CKModel;

static CKModel *g_model = NULL;
static ck_manifest_map_t *g_manifest = NULL;

/* Weight pointer macros */
#define W_PTR(off) ((void*)(g_model->bump_weights + (off)))
#define W_FLOAT(off) ((float*)(g_model->bump_weights + (off)))
/* Manifest runtime offsets are absolute (bump base) */
#define MANIFEST_OFFSETS_ABSOLUTE 1

/* ============================================================================
 * KERNEL DECLARATIONS
 * ============================================================================ */
void embedding_forward_q8_0(const int32_t *tokens, int count, int vocab, const void *emb, const float *pos, float *out, int dim, int adim, int ctx, int add_pos);
void rmsnorm_forward(const float *in, const float *gamma, float *out, float *rstd, int T, int D, int AD, float eps);
void gemv_q5_0(float *y, const void *W, const float *x, int M, int K);
void gemv_q8_0(float *y, const void *W, const float *x, int M, int K);
void gemv_q6_k(float *y, const void *W, const float *x, int M, int K);
void gemv_q4_k(float *y, const void *W, const float *x, int M, int K);
void gemm_nt_q5_0(const float *a, const void *W, float *bias, float *out, int M, int N, int K);
void gemm_nt_q8_0(const float *a, const void *W, float *bias, float *out, int M, int N, int K);
void gemm_nt_q6_k(const float *a, const void *W, float *bias, float *out, int M, int N, int K);
void gemm_nt_q4_k(const float *a, const void *W, float *bias, float *out, int M, int N, int K);
void rope_forward_qk(float *q, float *k, const float *cos, const float *sin, int H, int Hkv, int T, int D, int AD, int pos);
void rope_precompute_cache(float *cos_cache, float *sin_cache, int max_seq_len, int head_dim, float base);
void attention_forward_causal_head_major_gqa_flash_strided(const float *q, const float *k, const float *v, float *out, int H, int Hkv, int T, int D, int AD, int stride);
void attention_forward_decode_head_major_gqa_flash(const float *q, const float *k, const float *v, float *out, int H, int Hkv, int T, int D, int AD, int stride);
void kv_cache_store(float *k_cache, float *v_cache, const float *k_new, const float *v_new, int layer, int pos, int Hkv, int D, int stride);
void ck_qkv_project_head_major_quant(const float *input,
                                     const void *wq, const float *bq, int wq_dtype,
                                     const void *wk, const float *bk, int wk_dtype,
                                     const void *wv, const float *bv, int wv_dtype,
                                     float *q, float *k, float *v,
                                     int tokens, int kv_stride_tokens,
                                     int aligned_embed_dim, int num_heads,
                                     int num_kv_heads, int aligned_head_dim);
void ck_attention_project_head_major_quant(const float *attn_out,
                                           const void *wo, const float *bo,
                                           float *out, float *scratch,
                                           int tokens, int aligned_embed_dim,
                                           int num_heads, int aligned_head_dim,
                                           int wo_dtype);
void ck_residual_add_token_major(const float *a, const float *b, float *out, int T, int D);
void add_inplace_f32(float *a, const float *b, size_t n);
void swiglu_forward(const float *in, float *out, int T, int D);

/* ============================================================================
 * INIT / LOAD / FREE
 * ============================================================================ */
#ifdef _WIN32
#define CK_EXPORT __declspec(dllexport)
#else
#define CK_EXPORT __attribute__((visibility("default")))
#endif

/* Forward declarations */
static void ck_decode(CKModel *model, int32_t token);
static void ck_prefill(CKModel *model, const int32_t *tokens, int count);

static int do_init(void) {{
    if (g_model) return 0;
    g_model = calloc(1, sizeof(CKModel));
    if (!g_model) return -1;

    g_model->bump_size = BUMP_TOTAL_SIZE;
    g_model->bump = aligned_alloc(64, g_model->bump_size);
    if (!g_model->bump) {{ free(g_model); g_model = NULL; return -1; }}
    memset(g_model->bump, 0, g_model->bump_size);

    g_model->bump_weights = g_model->bump + BUMP_WEIGHTS_OFFSET;
    g_model->activations = (float*)(g_model->bump + BUMP_ACT_OFFSET);
    g_model->kv_cache = (float*)(g_model->bump + A_KV_CACHE);
    g_model->rope_cos = (float*)(g_model->bump + A_ROPE_CACHE);
    g_model->rope_sin = g_model->rope_cos + MAX_SEQ_LEN * HEAD_DIM / 2;
    g_model->logits = (float*)(g_model->bump + A_LOGITS);
    g_model->pos = 0;

{init_ops_code}

    return 0;
}}
''' + '''
static int build_manifest_path(const char *weights_path, char *out, size_t out_sz) {
    const char *slash = strrchr(weights_path, '/');
#ifdef _WIN32
    const char *bslash = strrchr(weights_path, '\\\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
#endif
    if (slash) {
        size_t dir_len = (size_t)(slash - weights_path + 1);
        const char *fname = "weights_manifest.map";
        size_t need = dir_len + strlen(fname) + 1;
        if (need > out_sz) return -1;
        memcpy(out, weights_path, dir_len);
        memcpy(out + dir_len, fname, strlen(fname) + 1);
        return 0;
    }
    if (strlen("weights_manifest.map") + 1 > out_sz) return -1;
    strcpy(out, "weights_manifest.map");
    return 0;
}

static int do_load_manifest(const char *weights_path, const char *manifest_path) {
    #if MANIFEST_OFFSETS_ABSOLUTE
    g_manifest = ck_load_weights_manifest_v66(g_model->bump, weights_path, manifest_path);
    #else
    g_manifest = ck_load_weights_manifest_v66(g_model->bump_weights, weights_path, manifest_path);
    #endif
    return g_manifest ? 0 : -1;
}

/* Combined init + load (ck_chat.py expects this signature) */
CK_EXPORT int ck_model_init(const char *weights_path) {
    char manifest_path[4096];
    if (do_init() != 0) return -1;
    if (build_manifest_path(weights_path, manifest_path, sizeof(manifest_path)) != 0) return -1;
    return do_load_manifest(weights_path, manifest_path);
}

/* Explicit manifest path (preferred) */
CK_EXPORT int ck_model_init_with_manifest(const char *weights_path, const char *manifest_path) {
    if (do_init() != 0) return -1;
    return do_load_manifest(weights_path, manifest_path);
}

CK_EXPORT void ck_model_free(void) {
    if (!g_model) return;
    if (g_manifest) {
        ck_unload_manifest_map(g_manifest);
        g_manifest = NULL;
    }
    if (g_model->bump) free(g_model->bump);
    free(g_model);
    g_model = NULL;
}

CK_EXPORT void ck_model_kv_cache_reset(void) {
    if (!g_model) return;
    memset(g_model->kv_cache, 0, KV_CACHE_SIZE);
    g_model->pos = 0;
}

CK_EXPORT int ck_model_kv_cache_enable(int capacity) {
    /* KV cache is always enabled in v6.6 */
    (void)capacity;
    return 0;
}

/* ============================================================================
 * PUBLIC API (compatible with ck_chat.py)
 * ============================================================================ */

/* Embed tokens (prefill) - stores embeddings in activation buffer */
CK_EXPORT int ck_model_embed_tokens(const int32_t *tokens, int count) {
    if (!g_model || !tokens || count <= 0) return -1;

#ifdef CK_HAS_PREFILL
    /* Use batched prefill for multiple tokens */
    if (count > 1) {
        ck_prefill(g_model, tokens, count);
        return 0;
    }
#endif

    /* Single token or no prefill: process one by one via decode */
    for (int i = 0; i < count; i++) {
        ck_decode(g_model, tokens[i]);
    }
    return 0;
}

/* Forward pass (after embed_tokens) */
CK_EXPORT int ck_model_forward(float *output) {
    if (!g_model) return -1;
    /* Logits are already computed by embed_tokens/decode */
    if (output) memcpy(output, g_model->logits, VOCAB_SIZE * sizeof(float));
    return 0;
}

/* Decode single token */
CK_EXPORT int ck_model_decode(int32_t token, float *output) {
    if (!g_model) return -1;
    /* Capture position before decode (ck_decode increments pos at end) */
    int token_pos = g_model->pos;
    ck_decode(g_model, token);
    /* Copy logits from position 0 to position token_pos in the logits buffer.
     * This makes the logits buffer match what ck_chat.py expects:
     * logits[token_pos * VOCAB_SIZE .. (token_pos+1) * VOCAB_SIZE] */
    if (token_pos > 0) {
        memmove(
            g_model->logits + (size_t)token_pos * VOCAB_SIZE,
            g_model->logits,
            VOCAB_SIZE * sizeof(float)
        );
    }
    if (output) memcpy(output, g_model->logits + (size_t)token_pos * VOCAB_SIZE, VOCAB_SIZE * sizeof(float));
    return 0;
}

/* Getters */
CK_EXPORT int ck_model_get_vocab_size(void) { return VOCAB_SIZE; }
CK_EXPORT const int32_t* ck_model_get_vocab_offsets(void) {
    return g_model ? (const int32_t*)(g_model->bump + W_VOCAB_OFFSETS) : NULL;
}
CK_EXPORT const uint8_t* ck_model_get_vocab_strings(void) {
    return g_model ? (const uint8_t*)(g_model->bump + W_VOCAB_STRINGS) : NULL;
}
CK_EXPORT int ck_model_get_num_merges(void) { return VOCAB_MERGES_COUNT; }
CK_EXPORT int ck_model_get_context_window(void) { return MAX_SEQ_LEN; }
CK_EXPORT int ck_model_get_active_tokens(void) { return g_model ? g_model->pos : 0; }
CK_EXPORT float* ck_model_get_logits(void) { return g_model ? g_model->logits : NULL; }
CK_EXPORT uintptr_t ck_model_get_base_ptr(void) { return (uintptr_t)(g_model ? g_model->bump : NULL); }
'''


# =============================================================================
# MAIN
# =============================================================================

def generate(ir_path: Path, layout_path: Path, debug: bool = False, init_call: Dict = None) -> str:
    """Generate complete C code.

    Args:
        ir_path: Path to lowered IR JSON
        layout_path: Path to layout JSON
        debug: If True, emit printf statements to dump tensor values after each op
        init_call: Lowered init IR dict (from init_call.json) with call-ready ops

    If debug=True, emit printf statements to dump tensor values after each op.
    """
    with open(ir_path) as f:
        ir = json.load(f)
    with open(layout_path) as f:
        layout = json.load(f)

    config = ir.get("config", {})
    # Prefer layout config for global dimensions (context length, etc.)
    layout_config = layout.get("config", {})
    if layout_config:
        merged = dict(config)
        merged.update(layout_config)
        config = merged
    ops = ir.get("ops", ir.get("operations", []))  # Support both "ops" and "operations"
    memory = layout.get("memory", ir.get("memory", {}))  # Use layout memory for offsets

    # Extract rope_theta from init_call config for header comment
    rope_theta = 10000.0  # Safe default (Llama 2)
    if init_call:
        init_config = init_call.get("config", {})
        rope_theta = init_config.get("rope_theta", rope_theta)
    elif "rope_theta" in config:
        rope_theta = config.get("rope_theta", rope_theta)

    # Fail fast if IR Lower 3 has errors or missing args
    ir_errors = ir.get("errors", [])
    op_errors = [op for op in ops if op.get("errors")]
    missing_args = [op for op in ops if "args" not in op]
    if ir_errors or op_errors or missing_args:
        raise RuntimeError(
            f"IR Lower 3 invalid: {len(ir_errors)} issues, "
            f"{len(op_errors)} ops with errors, {len(missing_args)} ops missing args"
        )

    # Get token offset from layout
    token_offset = 16
    token_base = "ACT"
    for buf in memory.get("activations", {}).get("buffers", []):
        if buf.get("name") == "token_ids":
            abs_off = buf.get("abs_offset")
            if abs_off is not None:
                token_offset = abs_off
                token_base = "MEM"
            else:
                token_offset = buf.get("offset", 16)
                token_base = "ACT"
            break

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    parts = []
    parts.append(f'''/*
 * Auto-generated by codegen_v6_6.py
 * Generated: {now}
 * Model: {config.get("model", "unknown")}
 * Mode: {ir.get("mode", "decode")}
 * Layers: {config.get("num_layers", 0)} (unrolled)
 * RoPE theta: {rope_theta}
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ckernel_model_load_v6.6.h"
#include "ckernel_engine.h"  /* Kernel declarations */
''')

    parts.append(emit_memory_layout(layout, config))
    parts.append(emit_model_and_api(init_call=init_call))  # Pass lowered init IR for init ops
    parts.append(emit_decode_function(ops, token_offset, token_base, debug=debug))

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate C code from IR")
    # New API (single mode)
    parser.add_argument("--ir", help="Lowered IR JSON (single mode)")
    parser.add_argument("--layout", help="Layout JSON (single mode)")
    # Legacy API (decode + prefill combined)
    parser.add_argument("--decode", help="Lowered decode IR JSON")
    parser.add_argument("--prefill", help="Lowered prefill IR JSON (optional)")
    # Init IR (rope_init, etc.)
    parser.add_argument("--init", help="Init IR JSON (rope_init, etc.) - uses rope_theta from here")
    # Output
    parser.add_argument("-o", "--output", required=True, help="Output C file")
    # Debug
    parser.add_argument("--debug", action="store_true", help="Emit debug printf for tensor values")
    args = parser.parse_args()

    # Handle legacy API (--decode/--prefill)
    if args.decode:
        decode_ir = Path(args.decode)
        # Infer layout path from decode IR path
        layout_decode = decode_ir.parent / "layout_decode.json"
        if not layout_decode.exists():
            print(f"Error: Could not find layout at {layout_decode}")
            sys.exit(1)

        # Load lowered init IR (init_call.json) if available
        init_call_obj = None
        init_call_path = Path(args.init) if args.init else decode_ir.parent / "init_call.json"
        if init_call_path.exists():
            with open(init_call_path) as f:
                init_call_obj = json.load(f)
            print(f"  Loaded init_call IR from {init_call_path}")
            init_ops = init_call_obj.get("operations", [])
            if init_ops:
                rope_theta = init_call_obj.get("config", {}).get("rope_theta", 10000.0)
                print(f"  - {len(init_ops)} init ops (rope_theta={rope_theta})")
        else:
            # Try legacy init.json path
            init_path = decode_ir.parent / "init.json"
            if init_path.exists():
                print(f"  Warning: Found init.json but not init_call.json - run build_ir with --init-output to generate")
            else:
                print(f"  Warning: No init_call.json found at {init_call_path}, init ops will be empty")

        # Load IRs for guard
        with open(decode_ir) as f:
            decode_obj = json.load(f)
        prefill_obj = None
        if args.prefill:
            prefill_ir = Path(args.prefill)
            if prefill_ir.exists():
                with open(prefill_ir) as f:
                    prefill_obj = json.load(f)
        with open(layout_decode) as f:
            layout_obj = json.load(f)
        ir_list = [decode_obj] + ([prefill_obj] if prefill_obj else [])
        _guard_bump_offsets(layout_obj, ir_list)

        # Generate decode code
        code = generate(decode_ir, layout_decode, debug=args.debug, init_call=init_call_obj)

        # Generate prefill code if provided
        prefill_code = ""
        if args.prefill:
            prefill_ir = Path(args.prefill)
            if prefill_ir.exists():
                try:
                    from codegen_prefill_v6_6 import generate_prefill
                    prefill_code = generate_prefill(prefill_ir)
                    print(f"  + Prefill code generated")
                except ImportError as e:
                    print(f"Warning: Could not import prefill codegen: {e}")
                except Exception as e:
                    print(f"Warning: Prefill codegen failed: {e}")

        # Combine decode + prefill
        if prefill_code:
            # Add CK_HAS_PREFILL define at the top (after initial includes)
            # Find where to insert (after the first #include block)
            insert_marker = "#include <math.h>"
            if insert_marker in code:
                code = code.replace(
                    insert_marker,
                    insert_marker + "\n\n/* Prefill support enabled */\n#define CK_HAS_PREFILL 1"
                )
            # Append prefill function at the end
            code = code + "\n\n" + prefill_code

        with open(args.output, 'w') as f:
            f.write(code)
        print(f"Generated: {args.output}")

    # Handle new API (--ir/--layout)
    elif args.ir and args.layout:
        # Guard for single IR + layout
        with open(args.ir) as f:
            ir_obj = json.load(f)
        with open(args.layout) as f:
            layout_obj = json.load(f)
        _guard_bump_offsets(layout_obj, [ir_obj])
        code = generate(Path(args.ir), Path(args.layout), debug=args.debug)
        with open(args.output, 'w') as f:
            f.write(code)
        print(f"Generated: {args.output}")

    else:
        print("Error: Must provide either --ir/--layout or --decode")
        sys.exit(1)


if __name__ == "__main__":
    main()
