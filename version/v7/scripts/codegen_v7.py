#!nusr/bin/env python3
from __future__ import annotations
"""
codegen_v7.py - Generate C code from lowered IR.

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
│ 3. ACTIVATION FUNCTION - IR LOWER ISSUE (NOT CODEGEN)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Status: Codegen is CORRECT - emits whatever kernel IR specifies             │
│ Fix needed in: build_ir_v7.py (IR Lower)                                  │
│                                                                             │
│ Issue: IR Lower hardcodes silu_mul -> swiglu mapping                        │
│ Should: Read hidden_act from config and map to correct kernel               │
│   - hidden_act: "silu" -> swiglu_forward                                    │
│   - hidden_act: "gelu" -> gelu_forward                                      │
│   - hidden_act: "relu" -> relu_forward                                      │
│                                                                             │
│ Impact: Non-SwiGLU models get wrong activation                              │
│   - GPT-2, GPT-Neo, OPT: GELU                                               │
│   - Qwen2, Llama, Mistral: SwiGLU (works today)                             │
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
✓ Kernel declarations from ckernel_engine.h - no inline declarations
✓ Init ops from init_call.json - RoPE theta, etc. come from IR
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


def _pick_first(config: Dict, keys: List[str], default=None):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _normalize_model_config(config: Dict) -> Dict:
    """Normalize common model config aliases to canonical keys used by codegen."""
    out = dict(config)

    embed_dim = _pick_first(out, ["embed_dim", "hidden_size", "n_embd", "d_model"])
    num_heads = _pick_first(out, ["num_heads", "num_attention_heads", "n_head"])
    num_kv_heads = _pick_first(out, ["num_kv_heads", "num_key_value_heads", "n_kv_head"], default=num_heads)
    head_dim = _pick_first(out, ["head_dim"])
    intermediate = _pick_first(out, ["intermediate_size", "ffn_dim", "n_inner"])
    vocab_size = _pick_first(out, ["vocab_size", "n_vocab"])
    context_length = _pick_first(out, ["context_length", "max_position_embeddings", "context_window", "max_seq_len"])
    num_layers = _pick_first(out, ["num_layers", "n_layer"])

    missing = []
    if embed_dim is None:
        missing.append("embed_dim/hidden_size")
    if num_heads is None:
        missing.append("num_heads/num_attention_heads")
    if intermediate is None:
        missing.append("intermediate_size/ffn_dim")
    if vocab_size is None:
        missing.append("vocab_size")
    if context_length is None:
        missing.append("context_length/max_position_embeddings")
    if num_layers is None:
        missing.append("num_layers")
    if missing:
        raise ValueError(f"Missing required model config keys: {', '.join(missing)}")

    embed_dim = int(embed_dim)
    num_heads = int(num_heads)
    num_kv_heads = int(num_kv_heads)
    intermediate = int(intermediate)
    vocab_size = int(vocab_size)
    context_length = int(context_length)
    num_layers = int(num_layers)

    if head_dim is None:
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(f"Cannot derive head_dim from embed_dim={embed_dim}, num_heads={num_heads}")
        head_dim = embed_dim // num_heads
    head_dim = int(head_dim)

    rotary_dim = int(_pick_first(out, ["rotary_dim"], default=head_dim))
    rope_theta = float(_pick_first(out, ["rope_theta", "rope_base", "theta"], default=10000.0))
    rope_scaling_type = str(_pick_first(out, ["rope_scaling_type"], default="none"))
    rope_scaling_factor = float(_pick_first(out, ["rope_scaling_factor"], default=1.0))

    out.update({
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "num_layers": num_layers,
        "rotary_dim": rotary_dim,
        "rope_theta": rope_theta,
        "rope_scaling_type": rope_scaling_type,
        "rope_scaling_factor": rope_scaling_factor,
    })
    return out


def emit_memory_layout(layout: Dict, config: Dict) -> str:
    """Emit C code for memory layout from layout.json."""
    config = _normalize_model_config(config)

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

    num_layers = config["num_layers"]

    # Get bump_layout from layout (passed through from manifest via build_ir)
    bump_layout = layout.get("bump_layout", {})

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
    # Extract RoPE params from init_call or config
    rotary_dim = config["rotary_dim"]
    rope_scaling_type = config["rope_scaling_type"]
    rope_scaling_factor = config["rope_scaling_factor"]

    lines.append(f'''
/* ============================================================================
 * MODEL CONFIGURATION
 * ============================================================================ */
#define EMBED_DIM {config["embed_dim"]}
#define NUM_HEADS {config["num_heads"]}
#define NUM_KV_HEADS {config["num_kv_heads"]}
#define HEAD_DIM {config["head_dim"]}
#define ROTARY_DIM {rotary_dim}
#define INTERMEDIATE_SIZE {config["intermediate_size"]}
#define NUM_LAYERS {num_layers}
#define VOCAB_SIZE {config["vocab_size"]}
#define MAX_SEQ_LEN {config["context_length"]}
/* RoPE scaling: type={rope_scaling_type}, factor={rope_scaling_factor} */

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
    vocab_strings_size = 0
    for e in header_entries:
        if e.get("name") == "vocab_merges":
            vocab_merges_size = int(e.get("size", 0))
        elif e.get("name") == "vocab_strings":
            vocab_strings_size = int(e.get("size", 0))
            break
    lines.append("/* Header weight offsets */")
    header_field_by_name: Dict[str, str] = {}
    used_header_fields = set()
    for e in header_entries:
        base = _sanitize_macro(e.get("name", "header_weight")).lower()
        field = base
        suffix = 2
        while field in used_header_fields:
            field = f"{base}_{suffix}"
            suffix += 1
        used_header_fields.add(field)
        header_field_by_name[e.get("name", "")] = field

    lines.append("typedef struct {")
    for e in header_entries:
        field = header_field_by_name.get(e.get("name", ""), _sanitize_macro(e.get("name", "header_weight")).lower())
        lines.append(f"    size_t {field};  /* {e.get('dtype', 'unknown')}, {e.get('size', 0)} bytes */")
    lines.append("} HeaderOffsets;")
    lines.append("")
    lines.append("static const HeaderOffsets L_HEADER = {")
    for e in header_entries:
        field = header_field_by_name.get(e.get("name", ""), _sanitize_macro(e.get("name", "header_weight")).lower())
        lines.append(f"    .{field} = {e.get('offset', 0)},")
    lines.append("};")
    lines.append("")
    if vocab_merges_size:
        lines.append(f"#define VOCAB_MERGES_COUNT {vocab_merges_size // 4}")
    else:
        lines.append("#define VOCAB_MERGES_COUNT 0")
    if vocab_strings_size:
        lines.append(f"#define VOCAB_STRINGS_SIZE {vocab_strings_size}")
    else:
        lines.append("#define VOCAB_STRINGS_SIZE 0")
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


def _infer_logits_layout(config: Dict, layout: Dict) -> str:
    """Infer logits layout ('last' or 'full') from config/layout."""
    layout_cfg = str(config.get("logits_layout", "auto")).lower()
    if layout_cfg in {"last", "full"}:
        return layout_cfg
    vocab = int(config.get("vocab_size", 0))
    if vocab > 0:
        for buf in layout.get("memory", {}).get("activations", {}).get("buffers", []):
            if buf.get("name") == "logits":
                size = int(buf.get("size", 0))
                return "full" if size > vocab * 4 else "last"
    return "last"

def emit_op(op: Dict, seq_idx: int | None = None, debug: bool = False, profile: bool = False, dump: bool = False) -> str:
    """Emit a single function call from an IR operation.

    Just read the op and emit the call. No special cases.
    IR Lower 3 provides call-ready args with exact expressions.

    If debug=True, emit printf statements to dump output buffer values.
    If profile=True, emit CK_PROFILE_BEGIN/END timing wrappers.
    """
    function = op.get("function", op.get("kernel", "unknown"))
    idx = op.get("idx", 0)
    layer = op.get("layer", -1)
    section = op.get("section", "")
    op_name = op.get("op", "unknown")
    args = op.get("args", [])

    lines = []
    lines.append(f"    /* Op {idx}: {function} ({op_name}) layer={layer} section={section} */")
    if profile:
        lines.append(f"    CK_PROFILE_BEGIN();")
    if not args:
        lines.append(f"    {function}();")
        if profile:
            lines.append(f'    CK_PROFILE_END("decode", "{function}", "{op_name}", {layer});')
        return "\n".join(lines)

    lines.append(f"    {function}(")
    for i, arg in enumerate(args):
        expr = arg.get("expr", "0")
        comma = "," if i < len(args) - 1 else ""
        lines.append(f"        {expr}{comma}")
    lines.append("    );")
    if profile:
        lines.append(f'    CK_PROFILE_END("decode", "{function}", "{op_name}", {layer});')

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

    # Add parity dump if enabled
    if dump:
        # Normalize op names to the same family used by parity_test.py.
        dump_op_map = {
            "dense_embedding_lookup": "token_embedding",
            "attn_norm": "attn_norm",
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "attn_sliding": "attn_output",
            "out_proj": "attn_output",
            "post_attention_norm": "attn_post_norm",
            "ffn_norm": "ffn_norm",
            "mlp_gate_up": "ffn_gate_up",
            "geglu": "ffn_gate_par",
            "mlp_down": "down_proj",
            "post_ffn_norm": "ffn_post_norm",
            "final_rmsnorm": "final_norm",
            "logits": "logits",
        }
        dump_name = dump_op_map.get(op_name)

        arg_expr_by_name = {}
        for arg in args:
            nm = str(arg.get("name", "")).lower()
            ex = str(arg.get("expr", ""))
            if nm and ex and nm not in arg_expr_by_name:
                arg_expr_by_name[nm] = ex

        def _get_arg(*names: str) -> str | None:
            for nm in names:
                ex = arg_expr_by_name.get(nm.lower())
                if ex:
                    return ex
            return None

        def _mul_expr(*terms: str | None) -> str | None:
            used = [f"({t})" for t in terms if t]
            if not used:
                return None
            return " * ".join(used)

        def _emit_dump(expr: str | None, name: str, size_expr: str | None) -> None:
            if not expr or not size_expr:
                return
            raw_expr = expr.replace("(float*)", "").replace("(void*)", "").strip()
            lines.append("    #ifdef CK_PARITY_DUMP")
            lines.append(f'    ck_dump_tensor((float*){raw_expr}, {layer}, "{name}", {size_expr});')
            lines.append("    #endif")

        tokens = _get_arg("tokens", "num_tokens", "token_count") or "1"
        embed_dim = _get_arg("aligned_embed_dim", "d_model", "embed_dim") or "EMBED_DIM"
        m_dim = _get_arg("m")
        n_dim = _get_arg("n")
        num_heads = _get_arg("num_heads") or "NUM_HEADS"
        num_kv_heads = _get_arg("num_kv_heads") or "NUM_KV_HEADS"
        head_dim = _get_arg("aligned_head_dim", "head_dim") or "HEAD_DIM"

        if op_name == "qk_norm":
            q_expr = _get_arg("q")
            k_expr = _get_arg("k")
            q_size = _mul_expr(_get_arg("num_tokens", "tokens") or tokens, num_heads, head_dim)
            k_size = _mul_expr(_get_arg("num_tokens", "tokens") or tokens, num_kv_heads, head_dim)
            _emit_dump(q_expr, "qcur_normed", q_size)
            _emit_dump(k_expr, "kcur_normed", k_size)
        elif dump_name:
            out_expr = _get_arg("output", "out", "c", "y", "out_token")
            size_expr = None

            if op_name in ("dense_embedding_lookup", "attn_norm", "post_attention_norm", "ffn_norm", "post_ffn_norm", "residual_add"):
                size_expr = _mul_expr(tokens, embed_dim)
            elif op_name in ("q_proj", "k_proj", "v_proj", "out_proj", "mlp_gate_up", "mlp_down", "logits"):
                if function.startswith("gemm_") and m_dim and n_dim:
                    size_expr = _mul_expr(m_dim, n_dim)
                else:
                    size_expr = m_dim or n_dim
            elif op_name == "attn_sliding":
                size_expr = _mul_expr(tokens, num_heads, head_dim)
            elif op_name == "geglu":
                dim = _get_arg("dim")
                size_expr = _mul_expr(tokens, dim) if dim else None

            _emit_dump(out_expr, dump_name, size_expr)

    return "\n".join(lines)


def emit_decode_function(
    ops: List[Dict],
    token_offset: int,
    token_base: str,
    debug: bool = False,
    profile: bool = False,
    dump: bool = False,
    scale_embeddings_sqrt_dim: bool = False,
) -> str:
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
    if profile:
        lines.append("    CK_PROFILE_VARS();")
    lines.append(f"    /* Store token at offset {token_offset} (from layout) */")
    lines.append(f"    *(int32_t*)({token_base} + {token_offset}) = token;")
    lines.append("")

    embed_scale_emitted = False
    for seq_idx, op in enumerate(ops):
        lines.append(emit_op(op, seq_idx, debug=debug, profile=profile, dump=dump))
        if (scale_embeddings_sqrt_dim
                and not embed_scale_emitted
                and op.get("op") == "dense_embedding_lookup"
                and int(op.get("layer", -1)) == -1):
            lines.append("""    /* Gemma embedding contract:
     * llama.cpp applies inp_scaled = inp_embd * sqrt(n_embd) before layer-0.
     * Keep this in generated decode path so every new token follows the same rule.
     */
    {
        const float emb_scale = sqrtf((float)EMBED_DIM);
        float *emb = (float*)(model->bump + A_EMBEDDED_INPUT);
        for (int i = 0; i < EMBED_DIM; ++i) {
            emb[i] *= emb_scale;
        }
    }
    #ifdef CK_PARITY_DUMP
    ck_dump_tensor((float*)(model->bump + A_EMBEDDED_INPUT), -1, "inp_scaled", EMBED_DIM);
    #endif""")
            embed_scale_emitted = True
        lines.append("")

    lines.append("    model->pos++;")
    lines.append("}")
    return "\n".join(lines)


# =============================================================================
# PART 3: CLEAN API WITH POINTER PASSING
# =============================================================================

def emit_init_call(op: Dict) -> str:
    """Emit a single init op call from lowered init IR.

    Handles two cases:
    1. Ops with c_code field - emit the C code directly (tokenizer_init, etc.)
    2. Ops with function/args - emit as function call (rope_init, etc.)
    """
    # Case 1: Direct C code (tokenizer_init, etc.)
    c_code = op.get("c_code")
    if c_code:
        if isinstance(c_code, dict):
            # Return the init portion of the c_code
            return c_code.get("init", "    /* No init code */")
        else:
            return c_code

    # Case 2: Function call (rope_init, etc.)
    func = op.get("function", "unknown")
    args = op.get("args", [])
    arg_exprs = [arg.get("expr", "0") for arg in args]
    return f"    {func}({', '.join(arg_exprs)});"


def get_init_free_code(init_call: Dict) -> str:
    """Get cleanup code for init ops that have free code."""
    if not init_call:
        return ""

    ops = init_call.get("operations", [])
    free_lines = []
    for op in ops:
        c_code = op.get("c_code")
        if isinstance(c_code, dict) and c_code.get("free"):
            free_lines.append(c_code["free"])

    return "\n".join(free_lines)


def emit_model_and_api(init_call: Dict = None, profile: bool = False, logits_stride: int = 0, dump: bool = False) -> str:
    """Emit model struct and clean API functions.

    Args:
        init_call: Lowered init IR (init_call.json) with call-ready ops.
                   If provided, emits init ops from IR (no hardcoding).
                   If None, emits empty init (no rope precompute).

    IMPORTANT: Codegen is DUMB. All tokenizer-specific code comes from IR.
    - Tokenizer struct field: from op["c_code"]["struct_field"]
    - Tokenizer init code: from op["c_code"]["init"]
    - Tokenizer free code: from op["c_code"]["free"]
    - Tokenizer API functions: from op["c_code"]["api_functions"]

    INIT ORDERING:
    - Pre-weights init (rope_init): Runs in do_init() BEFORE weights loaded.
      These ops have function/args but no c_code.
    - Post-weights init (tokenizer_init): Runs AFTER do_load_manifest().
      These ops have c_code that reads from bump memory (needs loaded weights).
    """
    logits_stride = int(logits_stride)
    if logits_stride > 0:
        decode_logits_copy = f"""
    /* Copy logits from position 0 to position token_pos in the logits buffer.
     * This makes the logits buffer match what ck_chat.py expects:
     * logits[token_pos * LOGITS_STRIDE .. (token_pos+1) * LOGITS_STRIDE] */
    if (token_pos > 0) {{
        memmove(
            g_model->logits + (size_t)token_pos * {logits_stride},
            g_model->logits,
            VOCAB_SIZE * sizeof(float)
        );
    }}
    if (output) memcpy(output, g_model->logits + (size_t)token_pos * {logits_stride}, VOCAB_SIZE * sizeof(float));"""
    else:
        decode_logits_copy = """
    /* Last-only logits layout */
    if (output) memcpy(output, g_model->logits, VOCAB_SIZE * sizeof(float));"""

    # Generate init ops code from lowered IR
    # Separate into pre-weights (rope, etc.) and post-weights (tokenizer, etc.)
    pre_weights_init_code = ""
    post_weights_init_code = ""
    free_ops_code = ""
    tokenizer_struct_field = ""
    tokenizer_api_functions = ""
    tokenizer_include = ""
    stop_tokens_api = ""

    if init_call:
        # Generate stop tokens API from special_tokens in init_call
        # These come from GGUF metadata (tokenizer.ggml.eos_token_id, etc.)
        special_tokens = init_call.get("special_tokens", {})
        if special_tokens:
            eos_id = special_tokens.get("eos_token_id", -1)
            bos_id = special_tokens.get("bos_token_id", -1)
            # Build stop tokens array (for now just EOS, could add more)
            stop_ids = []
            if eos_id is not None and eos_id >= 0:
                stop_ids.append(eos_id)

            stop_tokens_api = f"""
/* ============================================================================
 * STOP TOKENS API - Extracted from GGUF metadata
 * ============================================================================
 * These token IDs signal end-of-generation. Orchestrator should check if
 * sampled token matches any stop token and terminate the autoregressive loop.
 */
static const int32_t g_stop_tokens[] = {{ {', '.join(str(x) for x in stop_ids) if stop_ids else '-1'} }};
static const int g_num_stop_tokens = {len(stop_ids)};
static const int32_t g_eos_token_id = {eos_id if eos_id is not None else -1};
static const int32_t g_bos_token_id = {bos_id if bos_id is not None else -1};

/* Get number of stop tokens */
CK_EXPORT int ck_model_get_num_stop_tokens(void) {{
    return g_num_stop_tokens;
}}

/* Get stop tokens array (read-only) */
CK_EXPORT const int32_t* ck_model_get_stop_tokens(void) {{
    return g_stop_tokens;
}}

/* Check if a token is a stop token */
CK_EXPORT int ck_model_is_stop_token(int32_t token_id) {{
    for (int i = 0; i < g_num_stop_tokens; i++) {{
        if (g_stop_tokens[i] == token_id) return 1;
    }}
    return 0;
}}

/* Get EOS token ID (-1 if not set) */
CK_EXPORT int32_t ck_model_get_eos_token_id(void) {{
    return g_eos_token_id;
}}

/* Get BOS token ID (-1 if not set) */
CK_EXPORT int32_t ck_model_get_bos_token_id(void) {{
    return g_bos_token_id;
}}
"""
        ops = init_call.get("operations", [])
        if ops:
            pre_weights_lines = ["    /* Pre-weights init ops (do not depend on loaded weights) */"]
            post_weights_lines = ["    /* Post-weights init ops (depend on loaded weights) */"]
            free_lines = []
            for op in ops:
                if not op.get("errors"):
                    c_code = op.get("c_code")
                    if isinstance(c_code, dict):
                        # Post-weights init: ops with c_code (tokenizer_init reads from bump)
                        post_weights_lines.append(emit_init_call(op))
                        if c_code.get("free"):
                            free_lines.append(c_code["free"])
                        if c_code.get("struct_field"):
                            tokenizer_struct_field = c_code["struct_field"]
                        if c_code.get("api_functions"):
                            tokenizer_api_functions = c_code["api_functions"]
                        if c_code.get("include"):
                            tokenizer_include = c_code["include"]
                    else:
                        # Pre-weights init: ops without c_code (rope_init, etc.)
                        pre_weights_lines.append(emit_init_call(op))
                else:
                    pre_weights_lines.append(f"    /* ERROR: {op.get('function')} - {op.get('errors')} */")

            # Only include if we have actual ops
            if len(pre_weights_lines) > 1:
                pre_weights_init_code = "\n".join(pre_weights_lines)
            if len(post_weights_lines) > 1:
                post_weights_init_code = "\n".join(post_weights_lines)
            if free_lines:
                free_ops_code = "\n".join(free_lines)

    # Provide placeholder comments if no ops
    init_ops_code = pre_weights_init_code if pre_weights_init_code else "    /* No pre-weights init ops */"
    if not post_weights_init_code:
        post_weights_init_code = "    /* No post-weights init ops (no tokenizer) */"

    # Build optional tokenizer struct field (comes from IR, not hardcoded)
    tokenizer_field_line = ""
    if tokenizer_struct_field:
        tokenizer_field_line = f"    {tokenizer_struct_field}"

    # Build optional tokenizer include (comes from IR, not hardcoded)
    tokenizer_include_line = ""
    if tokenizer_include:
        tokenizer_include_line = tokenizer_include

    # Profile dump calls (guarded by #ifdef CK_PROFILE)
    profile_dump_after_decode = ""
    profile_dump_after_prefill = ""
    profile_dump_api = ""
    if profile:
        profile_dump_after_decode = """
#ifdef CK_PROFILE
    _ck_profile_dump();
#endif"""
        profile_dump_after_prefill = """
#ifdef CK_PROFILE
    _ck_profile_dump();
#endif"""
        profile_dump_api = """
/* Profile dump API */
CK_EXPORT void ck_model_profile_dump(void) {
#ifdef CK_PROFILE
    _ck_profile_dump();
#endif
}
"""

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
{tokenizer_field_line}
}} CKModel;

static CKModel *g_model = NULL;
static ck_manifest_map_t *g_manifest = NULL;

/* Weight pointer macros */
#define W_PTR(off) ((void*)(g_model->bump_weights + (off)))
#define W_FLOAT(off) ((float*)(g_model->bump_weights + (off)))
/* Manifest runtime offsets are absolute (bump base) */
#define MANIFEST_OFFSETS_ABSOLUTE 1

/* Kernel declarations from ckernel_engine.h (included above) */

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
    g_model->rope_sin = g_model->rope_cos + MAX_SEQ_LEN * ROTARY_DIM / 2;
    g_model->logits = (float*)(g_model->bump + A_LOGITS);
    g_model->pos = 0;

{init_ops_code}

    return 0;
}}
''' + f'''
static int build_manifest_path(const char *weights_path, char *out, size_t out_sz) {{
    const char *slash = strrchr(weights_path, '/');
#ifdef _WIN32
    const char *bslash = strrchr(weights_path, '\\\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
#endif
    if (slash) {{
        size_t dir_len = (size_t)(slash - weights_path + 1);
        const char *fname = "weights_manifest.map";
        size_t need = dir_len + strlen(fname) + 1;
        if (need > out_sz) return -1;
        memcpy(out, weights_path, dir_len);
        memcpy(out + dir_len, fname, strlen(fname) + 1);
        return 0;
    }}
    if (strlen("weights_manifest.map") + 1 > out_sz) return -1;
    strcpy(out, "weights_manifest.map");
    return 0;
}}

static int do_load_manifest(const char *weights_path, const char *manifest_path) {{
    #if MANIFEST_OFFSETS_ABSOLUTE
    g_manifest = ck_load_weights_manifest_v7(g_model->bump, weights_path, manifest_path);
    #else
    g_manifest = ck_load_weights_manifest_v7(g_model->bump_weights, weights_path, manifest_path);
    #endif
    return g_manifest ? 0 : -1;
}}

/* Post-weights initialization (tokenizer, etc.)
 * MUST be called AFTER do_load_manifest() because these ops read from bump memory */
static void do_post_weights_init(void) {{
{post_weights_init_code}
}}

static void print_omp_info(void) {{
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    const char *env = getenv("OMP_NUM_THREADS");
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    fprintf(stderr, "[OpenMP]   Threads: %d (OMP_NUM_THREADS=%s) | CPU cores: %ld\\n",
            max_threads, env ? env : "auto", ncpu);
#else
    fprintf(stderr, "[OpenMP]   Disabled (compiled without -fopenmp)\\n");
#endif
}}

/* Combined init + load (ck_chat.py expects this signature) */
CK_EXPORT int ck_model_init(const char *weights_path) {{
    char manifest_path[4096];
    if (do_init() != 0) return -1;
    if (build_manifest_path(weights_path, manifest_path, sizeof(manifest_path)) != 0) return -1;
    if (do_load_manifest(weights_path, manifest_path) != 0) return -1;
    do_post_weights_init();  /* Initialize tokenizer AFTER weights are loaded */
    print_omp_info();
#ifdef CK_PARALLEL_DECODE
    ck_parallel_decode_init();
#endif
#ifdef CK_PARALLEL_PREFILL
    ck_parallel_prefill_init();
#endif
#ifdef CK_PARITY_DUMP
    ck_dump_init(NULL);  /* Initialize parity dumping to ck_parity_dumps/dump.bin */
#endif
    return 0;
}}

/* Explicit manifest path (preferred) */
CK_EXPORT int ck_model_init_with_manifest(const char *weights_path, const char *manifest_path) {{
    if (do_init() != 0) return -1;
    if (do_load_manifest(weights_path, manifest_path) != 0) return -1;
    do_post_weights_init();  /* Initialize tokenizer AFTER weights are loaded */
    print_omp_info();
#ifdef CK_PARALLEL_DECODE
    ck_parallel_decode_init();
#endif
#ifdef CK_PARALLEL_PREFILL
    ck_parallel_prefill_init();
#endif
#ifdef CK_PARITY_DUMP
    ck_dump_init(NULL);
#endif
    return 0;
}}

CK_EXPORT void ck_model_free(void) {{
    if (!g_model) return;
#ifdef CK_PARALLEL_DECODE
    ck_parallel_decode_shutdown();
#endif
#ifdef CK_PARALLEL_PREFILL
    ck_parallel_prefill_shutdown();
#endif
#ifdef CK_PARITY_DUMP
    ck_dump_close();  /* Close parity dump file */
#endif
{free_ops_code}
    if (g_manifest) {{
        ck_unload_manifest_map(g_manifest);
        g_manifest = NULL;
    }}
    if (g_model->bump) free(g_model->bump);
    free(g_model);
    g_model = NULL;
}}

CK_EXPORT void ck_model_kv_cache_reset(void) {{
    if (!g_model) return;
    memset(g_model->kv_cache, 0, KV_CACHE_SIZE);
    g_model->pos = 0;
}}

CK_EXPORT int ck_model_kv_cache_enable(int capacity) {{
    /* KV cache is always enabled in v7 */
    (void)capacity;
    return 0;
}}

/* ============================================================================
 * PUBLIC API (compatible with ck_chat.py)
 * ============================================================================ */

/* Embed tokens (prefill) - stores embeddings in activation buffer */
CK_EXPORT int ck_model_embed_tokens(const int32_t *tokens, int count) {{
    if (!g_model || !tokens || count <= 0) return -1;

#ifdef CK_HAS_PREFILL
    /* Use batched prefill for multiple tokens */
    if (count > 1) {{
        ck_prefill(g_model, tokens, count);{profile_dump_after_prefill}
        return 0;
    }}
#endif

    /* Single token or no prefill: process one by one via decode */
    for (int i = 0; i < count; i++) {{
        ck_decode(g_model, tokens[i]);
    }}{profile_dump_after_decode}
    return 0;
}}

/* Forward pass (after embed_tokens) */
CK_EXPORT int ck_model_forward(float *output) {{
    if (!g_model) return -1;
    /* Logits are already computed by embed_tokens/decode */
    if (output) memcpy(output, g_model->logits, VOCAB_SIZE * sizeof(float));
    return 0;
}}

/* Decode single token */
CK_EXPORT int ck_model_decode(int32_t token, float *output) {{
    if (!g_model) return -1;
    /* Capture position before decode (ck_decode increments pos at end) */
    int token_pos = g_model->pos;
    ck_decode(g_model, token);{profile_dump_after_decode}{decode_logits_copy}
    return 0;
}}

/* Getters */
CK_EXPORT int ck_model_get_vocab_size(void) {{ return VOCAB_SIZE; }}
CK_EXPORT int ck_model_get_vocab_strings_size(void) {{
#ifdef VOCAB_STRINGS_SIZE
    return VOCAB_STRINGS_SIZE;
#else
    return 0;
#endif
}}
CK_EXPORT const int32_t* ck_model_get_vocab_offsets(void) {{
#ifdef W_VOCAB_OFFSETS
    return g_model ? (const int32_t*)(g_model->bump + W_VOCAB_OFFSETS) : NULL;
#else
    return NULL;
#endif
}}
CK_EXPORT const uint8_t* ck_model_get_vocab_strings(void) {{
#ifdef W_VOCAB_STRINGS
    return g_model ? (const uint8_t*)(g_model->bump + W_VOCAB_STRINGS) : NULL;
#else
    return NULL;
#endif
}}
CK_EXPORT int ck_model_get_num_merges(void) {{
#ifdef VOCAB_MERGES_COUNT
    return VOCAB_MERGES_COUNT;
#else
    return 0;
#endif
}}
CK_EXPORT int ck_model_get_context_window(void) {{ return MAX_SEQ_LEN; }}
CK_EXPORT int ck_model_get_active_tokens(void) {{ return g_model ? g_model->pos : 0; }}
CK_EXPORT int ck_model_get_logits_stride(void) {{ return {logits_stride}; }}
CK_EXPORT float* ck_model_get_logits(void) {{ return g_model ? g_model->logits : NULL; }}
CK_EXPORT uintptr_t ck_model_get_base_ptr(void) {{ return (uintptr_t)(g_model ? g_model->bump : NULL); }}

{tokenizer_api_functions}
{stop_tokens_api}
{profile_dump_api}
'''

# =============================================================================
# MAIN
# =============================================================================

def generate(ir_path: Path, layout_path: Path, debug: bool = False, init_call: Dict = None, profile: bool = False, dump: bool = False) -> str:
    """Generate complete C code.

    Args:
        ir_path: Path to lowered IR JSON
        layout_path: Path to layout JSON
        debug: If True, emit printf statements to dump tensor values after each op
        init_call: Lowered init IR dict (from init_call.json) with call-ready ops
        profile: If True, emit CK_PROFILE timing wrappers around each kernel call
        dump: If True, emit parity dump calls for llama.cpp comparison

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
    config = _normalize_model_config(config)
    logits_layout = _infer_logits_layout(config, layout)
    vocab_size = int(config.get("vocab_size", 0))
    logits_stride = vocab_size if logits_layout == "full" else 0
    ops = ir.get("ops", ir.get("operations", []))  # Support both "ops" and "operations"
    scale_embeddings_sqrt_dim = bool(config.get("scale_embeddings_sqrt_dim", False))
    memory = layout.get("memory", ir.get("memory", {}))  # Use layout memory for offsets

    # Extract RoPE params from init_call config for header comment
    rope_theta = config["rope_theta"]
    rotary_dim = config["rotary_dim"]
    rope_scaling_type = config["rope_scaling_type"]
    rope_scaling_factor = config["rope_scaling_factor"]
    rope_init_kernel = "rope_precompute_cache"
    rope_qk_kernel = "rope_forward_qk"

    if init_call:
        init_config = init_call.get("config", {})
        rope_theta = init_config.get("rope_theta", rope_theta)
        rotary_dim = init_config.get("rotary_dim", rotary_dim)
        rope_scaling_type = init_config.get("rope_scaling_type", rope_scaling_type)
        rope_scaling_factor = init_config.get("rope_scaling_factor", rope_scaling_factor)
        # Infer init kernel from lowered init ops (if present)
        for op in init_call.get("operations", []):
            if op.get("op") == "rope_init":
                rope_init_kernel = op.get("function", op.get("kernel", rope_init_kernel))
                break
    elif "rope_theta" in config:
        rope_theta = config.get("rope_theta", rope_theta)
        rotary_dim = config.get("rotary_dim", rotary_dim)
        rope_scaling_type = config.get("rope_scaling_type", rope_scaling_type)
        rope_scaling_factor = config.get("rope_scaling_factor", rope_scaling_factor)

    # Infer rope_qk kernel from IR ops (if present)
    for op in ops:
        if op.get("op") == "rope_qk":
            rope_qk_kernel = op.get("function", op.get("kernel", rope_qk_kernel))
            break

    rope_cache_layout = "head_dim/2" if rope_init_kernel == "rope_precompute_cache_split" else "rotary_dim/2"

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

    # Extract tokenizer include from init_call (if present)
    # Codegen is DUMB - we read the include from IR, not hardcode it
    tokenizer_include = ""
    if init_call:
        for op in init_call.get("operations", []):
            c_code = op.get("c_code", {})
            if isinstance(c_code, dict) and c_code.get("include"):
                tokenizer_include = c_code["include"]
                break

    parts = []
    parts.append(f'''/*
 * Auto-generated by codegen_v7.py
 * Generated: {now}
 * Model: {config.get("model", "unknown")}
 * Mode: {ir.get("mode", "decode")}
 * Layers: {config.get("num_layers", 0)} (unrolled)
 * RoPE: theta={rope_theta}, rotary={rotary_dim}, scaling={rope_scaling_type}/{rope_scaling_factor}
 * RoPE kernels: init={rope_init_kernel}, qk={rope_qk_kernel}, cache={rope_cache_layout}
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "ckernel_model_load_v7.h"
#include "ckernel_engine.h"  /* Kernel declarations */
{tokenizer_include}

/* Parity dump instrumentation for llama.cpp comparison */
{"#define CK_PARITY_DUMP 1" if dump else "/* undef CK_PARITY_DUMP */"}
#ifdef CK_PARITY_DUMP
#include "ck_parity_dump.h"
#endif

/* Thread-pool parallel GEMV dispatch (persistent pthread workers) */
#define CK_PARALLEL_DECODE 1
#include "ck_parallel_decode.h"

/* Thread-pool parallel GEMM dispatch for prefill (row-splitting) */
#define CK_PARALLEL_PREFILL 1
#include "ck_parallel_prefill.h"
''')

    # Emit profiling infrastructure (all #ifdef CK_PROFILE guarded)
    if profile:
        parts.append('''
/* ============================================================================
 * PROFILING INFRASTRUCTURE (CK_PROFILE)
 * ============================================================================ */
#ifdef CK_PROFILE
#include <time.h>

typedef struct {
    const char *mode;       /* "prefill" or "decode" */
    const char *kernel;     /* e.g. "gemv_q5_0_q8_0" */
    const char *op;         /* e.g. "q_proj", "mlp_gate_up" */
    int layer;
    double time_us;
} CKProfileEntry;

#define CK_PROFILE_MAX_ENTRIES 16384
static CKProfileEntry _ck_profile_entries[CK_PROFILE_MAX_ENTRIES];
static int _ck_profile_count = 0;
static int _ck_profile_token_id = 0;

static inline void _ck_profile_log(const char *mode, const char *kernel,
                                    const char *op, int layer,
                                    struct timespec t0, struct timespec t1) {
    if (_ck_profile_count >= CK_PROFILE_MAX_ENTRIES) return;
    double us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
    _ck_profile_entries[_ck_profile_count++] = (CKProfileEntry){
        mode, kernel, op, layer, us
    };
}

static void _ck_profile_dump_json(FILE *f) {
    fprintf(f, "{\\n  \\"entries\\": [\\n");
    for (int i = 0; i < _ck_profile_count; i++) {
        CKProfileEntry *e = &_ck_profile_entries[i];
        fprintf(f, "    {\\"mode\\":\\"%s\\",\\"kernel\\":\\"%s\\",\\"op\\":\\"%s\\","
                "\\"layer\\":%d,\\"time_us\\":%.1f,\\"token_id\\":%d}%s\\n",
                e->mode, e->kernel, e->op, e->layer, e->time_us,
                _ck_profile_token_id,
                i < _ck_profile_count - 1 ? "," : "");
    }
    fprintf(f, "  ],\\n  \\"total_entries\\": %d,\\n  \\"token_id\\": %d\\n}\\n",
            _ck_profile_count, _ck_profile_token_id);
}

static void _ck_profile_dump(void) {
    const char *csv_path = getenv("CK_PROFILE_CSV");
    const char *json_path = getenv("CK_PROFILE_JSON");
    if (csv_path) {
        FILE *f = fopen(csv_path, _ck_profile_token_id == 0 ? "w" : "a");
        if (f) {
            if (_ck_profile_token_id == 0) {
                fprintf(f, "mode,kernel,op,layer,time_us,token_id\\n");
            }
            for (int i = 0; i < _ck_profile_count; i++) {
                CKProfileEntry *e = &_ck_profile_entries[i];
                fprintf(f, "%s,%s,%s,%d,%.1f,%d\\n",
                        e->mode, e->kernel, e->op, e->layer, e->time_us,
                        _ck_profile_token_id);
            }
            fclose(f);
        }
    }
    if (json_path) {
        FILE *f = fopen(json_path, "w");
        if (f) { _ck_profile_dump_json(f); fclose(f); }
    }
    if (!csv_path && !json_path) {
        fprintf(stderr, "mode,kernel,op,layer,time_us,token_id\\n");
        for (int i = 0; i < _ck_profile_count; i++) {
            CKProfileEntry *e = &_ck_profile_entries[i];
            fprintf(stderr, "%s,%s,%s,%d,%.1f,%d\\n",
                    e->mode, e->kernel, e->op, e->layer, e->time_us,
                    _ck_profile_token_id);
        }
    }
    _ck_profile_count = 0;
    _ck_profile_token_id++;
}

#define CK_PROFILE_VARS() struct timespec _pt0, _pt1;
#define CK_PROFILE_BEGIN() clock_gettime(CLOCK_MONOTONIC, &_pt0);
#define CK_PROFILE_END(mode, kernel, op, layer) \\
    clock_gettime(CLOCK_MONOTONIC, &_pt1); \\
    _ck_profile_log(mode, kernel, op, layer, _pt0, _pt1);
#else
#define CK_PROFILE_VARS()
#define CK_PROFILE_BEGIN()
#define CK_PROFILE_END(mode, kernel, op, layer)
#endif
''')

    parts.append(emit_memory_layout(layout, config))
    parts.append(emit_model_and_api(init_call=init_call, profile=profile, logits_stride=logits_stride, dump=dump))  # Pass lowered init IR for init ops
    parts.append(
        emit_decode_function(
            ops,
            token_offset,
            token_base,
            debug=debug,
            profile=profile,
            dump=dump,
            scale_embeddings_sqrt_dim=scale_embeddings_sqrt_dim,
        )
    )

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
    # Profile
    parser.add_argument("--profile", action="store_true", help="Emit CK_PROFILE timing wrappers")
    # Parity Dump
    parser.add_argument("--parity-dump", action="store_true",
                       help="Emit parity dump calls for llama.cpp comparison")
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
                init_config = init_call_obj.get("config", {})
                rope_theta = init_config.get("rope_theta")
                rotary_dim = init_config.get("rotary_dim", init_config.get("head_dim"))
                rope_scaling_type = init_config.get("rope_scaling_type")
                rope_scaling_factor = init_config.get("rope_scaling_factor")
                print(f"  - {len(init_ops)} init ops (rope_theta={rope_theta}, rotary={rotary_dim}, scaling={rope_scaling_type}/{rope_scaling_factor})")
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
        code = generate(decode_ir, layout_decode, debug=args.debug, init_call=init_call_obj, profile=args.profile, dump=args.parity_dump)

        # Generate prefill code if provided
        prefill_code = ""
        if args.prefill:
            prefill_ir = Path(args.prefill)
            if prefill_ir.exists():
                try:
                    from codegen_prefill_v7 import generate_prefill
                    prefill_code = generate_prefill(prefill_ir, profile=args.profile)
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

        # Load init_call.json for stop tokens API (if available)
        init_call_obj = None
        init_call_path = Path(args.init) if args.init else Path(args.ir).parent / "init_call.json"
        if init_call_path.exists():
            with open(init_call_path) as f:
                init_call_obj = json.load(f)

        code = generate(Path(args.ir), Path(args.layout), debug=args.debug, init_call=init_call_obj, profile=args.profile, dump=args.parity_dump)
        with open(args.output, 'w') as f:
            f.write(code)
        print(f"Generated: {args.output}")

    else:
        print("Error: Must provide either --ir/--layout or --decode")
        sys.exit(1)


if __name__ == "__main__":
    main()
