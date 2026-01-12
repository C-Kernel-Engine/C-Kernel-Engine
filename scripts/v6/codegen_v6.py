#!/usr/bin/env python3
"""
codegen_v6.py - Explicit unrolled C source emitter for IR v6 layouts.

Key differences from v4:
- Each layer is unrolled explicitly (no loop over layers)
- Explicit kernel names (gemm_nt_q5_0 not ck_gemm_nt_quant)
- Per-layer weight dtypes are visible in generated code
- Designed for easy debugging and PyTorch comparison

Generated code shows the exact quant type for each operation, e.g.:
    // Layer 0: wq=Q5_0, wv=Q8_0, w2=Q6_K
    gemm_nt_q5_0(L0_input, L0_WQ, NULL, L0_q, ...);
    gemm_nt_q8_0(L0_input, L0_WV, NULL, L0_v, ...);
"""

from datetime import datetime
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_DIR = _SCRIPT_DIR / "v3"
if _V3_DIR.is_dir():
    sys.path.insert(0, str(_V3_DIR))

import build_ir_v3 as v3


def _layer_buffer_names(section) -> List[str]:
    if not section.layers:
        return []
    return [buf.name for buf in section.layers[0].buffers]


def _layer_has_field(layer_names: List[str], field: str) -> bool:
    suffix = f".{field}"
    for name in layer_names:
        if name.endswith(suffix):
            return True
    return False


# Map CK_DT_* types to explicit kernel function names
DTYPE_TO_GEMM_NT_KERNEL = {
    "q4_0": "gemm_nt_q4_0",
    "q4_1": "gemm_nt_q4_1",
    "q5_0": "gemm_nt_q5_0",
    "q5_1": "gemm_nt_q5_1",
    "q8_0": "gemm_nt_q8_0",
    "q4_k": "gemm_nt_q4_k",
    "q6_k": "gemm_nt_q6_k",
    "fp32": "gemm_blocked_serial",  # FP32 fallback
}

DTYPE_TO_EMBEDDING_KERNEL = {
    "q4_0": "embedding_forward_q4_0",
    "q4_1": "embedding_forward_q4_1",
    "q5_0": "embedding_forward_q5_0",
    "q5_1": "embedding_forward_q5_1",
    "q8_0": "embedding_forward_q8_0",
    "q4_k": "embedding_forward_q4_k",
    "q6_k": "embedding_forward_q6_k",
    "fp32": "embedding_forward",
}


def get_quant_type(dtype: str) -> str:
    """Return normalized quant type string (e.g., 'q5_0', 'q4_k', 'fp32')."""
    dtype = dtype.lower()
    for qt in ["q4_k", "q6_k", "q8_k", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1"]:
        if dtype.startswith(qt):
            return qt
    return "fp32"


def dtype_const(dtype: str) -> str:
    """Convert dtype string to CK_DT_* constant."""
    qt = get_quant_type(dtype)
    return {
        "q4_k": "CK_DT_Q4_K",
        "q6_k": "CK_DT_Q6_K",
        "q4_0": "CK_DT_Q4_0",
        "q4_1": "CK_DT_Q4_1",
        "q5_0": "CK_DT_Q5_0",
        "q5_1": "CK_DT_Q5_1",
        "q8_0": "CK_DT_Q8_0",
        "q8_k": "CK_DT_Q8_K",
        "fp32": "CK_DT_FP32",
    }.get(qt, "CK_DT_FP32")


def validate_layout_vs_manifest(layout: v3.ModelLayout, manifest: Dict) -> List[str]:
    """Validate layout dtypes against manifest dtypes.

    Returns list of error messages (empty if all OK).
    """
    errors = []
    warnings = []

    if not manifest or "entries" not in manifest:
        return ["No manifest entries found"]

    # Build manifest lookup
    manifest_lookup = {e["name"]: e for e in manifest["entries"]}

    section = layout.sections[0]
    config = layout.config
    num_layers = config.get("num_hidden_layers", len(section.layers))

    # Check token_emb
    for buf in section.header_buffers:
        if buf.name == "token_emb":
            mentry = manifest_lookup.get("token_emb")
            if mentry:
                layout_dtype = get_quant_type(str(buf.dtype).lower())
                manifest_dtype = get_quant_type(mentry["dtype"])
                if layout_dtype != manifest_dtype:
                    errors.append(f"token_emb: layout={layout_dtype} vs manifest={manifest_dtype}")

    # Check per-layer weights
    for layer_id, layer in enumerate(section.layers):
        for buf in layer.buffers:
            parts = buf.name.split(".", 2)
            if len(parts) == 3 and parts[0] == "layer":
                weight_suffix = parts[2]
                manifest_name = f"layer.{layer_id}.{weight_suffix}"
                mentry = manifest_lookup.get(manifest_name)
                if mentry:
                    layout_dtype = get_quant_type(str(buf.dtype).lower())
                    manifest_dtype = get_quant_type(mentry["dtype"])
                    if layout_dtype != manifest_dtype:
                        errors.append(f"{manifest_name}: layout={layout_dtype} vs manifest={manifest_dtype}")

    return errors


def print_manifest_validation_table(manifest: Dict, num_layers: int) -> List[str]:
    """Generate a validation table showing manifest entries for code comments."""
    lines = []

    if not manifest or "entries" not in manifest:
        return lines

    # Build per-layer summary
    lines.append(" * ")
    lines.append(" * ═══════════════════════════════════════════════════════════════════════════")
    lines.append(" * MANIFEST VALIDATION (from weights_manifest.json)")
    lines.append(" * ═══════════════════════════════════════════════════════════════════════════")
    lines.append(" * ")
    lines.append(" * Layer | WQ    | WK    | WV    | WO    | W1    | W2    | BQ | BK | BV | BO")
    lines.append(" * ------|-------|-------|-------|-------|-------|-------|----|----|----|----|")

    manifest_lookup = {e["name"]: e for e in manifest["entries"]}

    for layer_id in range(num_layers):
        wq = manifest_lookup.get(f"layer.{layer_id}.wq", {}).get("dtype", "-")
        wk = manifest_lookup.get(f"layer.{layer_id}.wk", {}).get("dtype", "-")
        wv = manifest_lookup.get(f"layer.{layer_id}.wv", {}).get("dtype", "-")
        wo = manifest_lookup.get(f"layer.{layer_id}.wo", {}).get("dtype", "-")
        w1 = manifest_lookup.get(f"layer.{layer_id}.w1", {}).get("dtype", "-")
        w2 = manifest_lookup.get(f"layer.{layer_id}.w2", {}).get("dtype", "-")

        # Check biases - ✓ if from GGUF (size > 0 and not zeros), ○ if zeros
        bq_entry = manifest_lookup.get(f"layer.{layer_id}.bq", {})
        bk_entry = manifest_lookup.get(f"layer.{layer_id}.bk", {})
        bv_entry = manifest_lookup.get(f"layer.{layer_id}.bv", {})
        bo_entry = manifest_lookup.get(f"layer.{layer_id}.bo", {})

        # For now, assume any bias entry exists
        bq = "✓" if bq_entry.get("size", 0) > 0 else "○"
        bk = "✓" if bk_entry.get("size", 0) > 0 else "○"
        bv = "✓" if bv_entry.get("size", 0) > 0 else "○"
        bo = "○"  # bo is always zeros for Qwen2

        lines.append(f" * {layer_id:5d} | {wq:5s} | {wk:5s} | {wv:5s} | {wo:5s} | {w1:5s} | {w2:5s} | {bq:2s} | {bk:2s} | {bv:2s} | {bo:2s}")

    lines.append(" * ")
    lines.append(f" * Total manifest entries: {len(manifest['entries'])}")
    if manifest.get("has_attention_biases"):
        lines.append(" * Attention biases: PRESENT (Qwen2-style)")
    else:
        lines.append(" * Attention biases: NONE (LLaMA-style)")
    lines.append(" * ═══════════════════════════════════════════════════════════════════════════")
    lines.append(" * ")

    return lines


def emit_c_source_v6(layout: v3.ModelLayout,
                     output_path: str,
                     header_name: str,
                     mode: str,
                     emit_main: bool = False,
                     emit_debug: bool = False,
                     emit_parity: bool = False,
                     weights_manifest: Optional[Dict] = None) -> None:
    """Emit generated_model.c with explicit per-layer unrolled kernel calls.

    Args:
        emit_debug: If True, insert debug prints after each layer to detect NaN/zero outputs.
        emit_parity: If True, save intermediate buffers to files for comparison with PyTorch.
        weights_manifest: If provided, validate layout against manifest and embed validation table.
    """
    if mode not in ("prefill", "decode"):
        raise ValueError(f"v6 codegen only supports prefill/decode (got: {mode})")

    # ═══════════════════════════════════════════════════════════════════════════
    # Manifest validation - catch dtype mismatches at codegen time
    # ═══════════════════════════════════════════════════════════════════════════
    if weights_manifest:
        print(f"[CODEGEN] Validating layout against manifest...")
        validation_errors = validate_layout_vs_manifest(layout, weights_manifest)
        if validation_errors:
            print(f"[CODEGEN] ⚠ VALIDATION ERRORS:")
            for err in validation_errors:
                print(f"[CODEGEN]   - {err}")
            raise ValueError(f"Layout/manifest dtype mismatch: {len(validation_errors)} errors. Fix conversion or regenerate IR.")
        print(f"[CODEGEN] ✓ Layout matches manifest ({len(weights_manifest.get('entries', []))} entries)")

    config = layout.config
    section = layout.sections[0]

    safe_name = layout.name.upper().replace("-", "_").replace(".", "_")
    safe_name_lower = safe_name.lower()

    # Build per-layer dtype maps for mixed quant support
    layer_dtype_maps: List[Dict[str, str]] = []
    if section.layers:
        for layer in section.layers:
            layer_map = {}
            for buf in layer.buffers:
                parts = buf.name.split(".", 2)
                if len(parts) == 3 and parts[0] == "layer":
                    layer_map[parts[2]] = str(buf.dtype).lower()
            layer_dtype_maps.append(layer_map)

    def layer_weight_dtype(name: str, layer_id: int = 0) -> str:
        if layer_id < len(layer_dtype_maps):
            return layer_dtype_maps[layer_id].get(name, "")
        return ""

    def buffer_dtype(buffers, name: str) -> str:
        for buf in buffers:
            if buf.name == name:
                return str(buf.dtype).lower()
        return ""

    token_emb_dtype = buffer_dtype(section.header_buffers, "token_emb")
    lm_head_dtype = buffer_dtype(section.footer_buffers, "lm_head_weight")
    # When embeddings are tied, lm_head uses same weights as token_emb
    if config.get("tie_word_embeddings", True) and token_emb_dtype:
        lm_head_dtype = token_emb_dtype

    embed_quant_type = get_quant_type(token_emb_dtype)
    lm_head_quant_type = get_quant_type(lm_head_dtype)

    embed_use_q4_k = embed_quant_type == "q4_k"
    lm_head_use_q4_k = lm_head_quant_type == "q4_k"

    # Mixed-quant detection (per-layer dtypes)
    def has_mixed_layer_dtypes() -> bool:
        if len(layer_dtype_maps) <= 1:
            return False
        ref = layer_dtype_maps[0]
        for layer_map in layer_dtype_maps[1:]:
            for key in ["wq", "wk", "wv", "wo", "w1", "w2"]:
                if get_quant_type(layer_map.get(key, "")) != get_quant_type(ref.get(key, "")):
                    return True
        return False

    mixed_quant = has_mixed_layer_dtypes()

    wq_dtype = layer_weight_dtype("wq")
    wk_dtype = layer_weight_dtype("wk")
    wv_dtype = layer_weight_dtype("wv")
    wo_dtype = layer_weight_dtype("wo")
    w1_dtype = layer_weight_dtype("w1")
    w2_dtype = layer_weight_dtype("w2")

    weight_dtypes = [wq_dtype, wk_dtype, wv_dtype, wo_dtype, w1_dtype, w2_dtype]
    weight_quant_types = [get_quant_type(d) for d in weight_dtypes]
    has_quant = any(qt != "fp32" for qt in weight_quant_types)
    all_q4_k = has_quant and all(qt == "q4_k" for qt in weight_quant_types if qt != "fp32")
    use_fast_q4 = has_quant and all_q4_k
    use_quant_path = has_quant and not use_fast_q4

    num_layers = config.get("num_hidden_layers", len(section.layers))

    aligned_embed = int(config.get("aligned_embed") or 0)
    aligned_head = int(config.get("aligned_head") or 0)
    aligned_intermediate = int(config.get("aligned_intermediate") or 0)
    aligned_context = int(config.get("aligned_context") or 0)

    def aligned_expr(value: int, fallback: str) -> str:
        return str(value) if value > 0 else fallback

    layer_names = _layer_buffer_names(section)
    has_rope = any(buf.name in {"rope_cos_cache", "rope_sin_cache"} for buf in section.globals)
    has_proj_scratch = _layer_has_field(layer_names, "proj_scratch")
    has_scores = _layer_has_field(layer_names, "scores")
    has_output_bias = _layer_has_field(layer_names, "bo")
    has_mlp_bias1 = _layer_has_field(layer_names, "b1")
    has_mlp_bias2 = _layer_has_field(layer_names, "b2")
    has_attention_biases = weights_manifest.get("has_attention_biases", False) if weights_manifest else False

    lines = []
    def add(s=""):
        lines.append(s)

    # File header
    add("/**")
    add(f" * @file {os.path.basename(output_path)}")
    add(f" * @brief AUTO-GENERATED: {layout.name} Implementation (IR v6 - Explicit Unrolled)")
    add(f" *")
    add(f" * Generated: {datetime.utcnow().isoformat()} UTC")
    add(f" * Total Memory: {layout.total_bytes / 1e9:.2f} GB")
    add(f" * Mode: {mode}")
    add(f" * Layers: {num_layers} (fully unrolled)")
    add(f" *")
    add(f" * Per-layer quant types:")
    for layer_id in range(min(num_layers, 3)):  # Show first 3 layers in header
        dtypes = layer_dtype_maps[layer_id] if layer_id < len(layer_dtype_maps) else {}
        wq_dt = get_quant_type(dtypes.get("wq", "fp32"))
        wk_dt = get_quant_type(dtypes.get("wk", "fp32"))
        wv_dt = get_quant_type(dtypes.get("wv", "fp32"))
        wo_dt = get_quant_type(dtypes.get("wo", "fp32"))
        w1_dt = get_quant_type(dtypes.get("w1", "fp32"))
        w2_dt = get_quant_type(dtypes.get("w2", "fp32"))
        add(f" *   Layer {layer_id}: wq={wq_dt} wk={wk_dt} wv={wv_dt} wo={wo_dt} w1={w1_dt} w2={w2_dt}")
    if num_layers > 3:
        add(f" *   ... ({num_layers - 3} more layers)")

    # Add manifest validation table if available
    if weights_manifest:
        manifest_lines = print_manifest_validation_table(weights_manifest, num_layers)
        for line in manifest_lines:
            add(line)

    add(f" *")
    add(f" * DO NOT EDIT - Regenerate with build_ir_v6.py or codegen_v6.py")
    add(f" */")
    add()
    add("#define _GNU_SOURCE  /* For MAP_ANONYMOUS, MAP_HUGETLB */")
    add()
    add(f'#include "{header_name}"')
    add()
    add('#include "ckernel_engine.h"')
    add()
    add("#include <stdio.h>")
    add("#include <stdlib.h>")
    add("#include <string.h>")
    add("#include <stdint.h>")
    add("#include <math.h>")
    add()
    add("#ifdef __linux__")
    add("#include <sys/mman.h>")
    add("#endif")
    add()
    add(f"#if {safe_name}_DTYPE_BYTES != 4")
    add(f'#error "{layout.name}: v6 codegen currently supports fp32 only. Use --dtype=fp32."')
    add("#endif")
    add()

    add("/* ============================================================================")
    add(" * LOCAL HELPERS (no orchestration dependency)")
    add(" * ============================================================================ */")
    add()
    add(f"static void {safe_name_lower}_residual_add_token_major(")
    add("    const float *a,")
    add("    const float *b,")
    add("    float *out,")
    add("    int tokens,")
    add("    int aligned_embed_dim")
    add(") {")
    add("    if (!a || !b || !out) {")
    add("        return;")
    add("    }")
    add("    for (int t = 0; t < tokens; ++t) {")
    add("        const float *pa = a + (size_t)t * (size_t)aligned_embed_dim;")
    add("        const float *pb = b + (size_t)t * (size_t)aligned_embed_dim;")
    add("        float *pc = out + (size_t)t * (size_t)aligned_embed_dim;")
    add("        for (int d = 0; d < aligned_embed_dim; ++d) {")
    add("            pc[d] = pa[d] + pb[d];")
    add("        }")
    add("    }")
    add("}")
    add()

    # Debug helpers
    if emit_debug:
        add("/* ============================================================================")
        add(" * DEBUG HELPERS")
        add(" * ============================================================================ */")
        add()
        add("static void debug_check_buffer(const char *name, const float *buf, int size) {")
        add("    int nan_count = 0, inf_count = 0, zero_count = 0;")
        add("    float min_val = 1e38f, max_val = -1e38f, sum = 0.0f;")
        add("    for (int i = 0; i < size; ++i) {")
        add("        float v = buf[i];")
        add("        if (isnan(v)) { nan_count++; continue; }")
        add("        if (isinf(v)) { inf_count++; continue; }")
        add("        if (v == 0.0f) zero_count++;")
        add("        if (v < min_val) min_val = v;")
        add("        if (v > max_val) max_val = v;")
        add("        sum += v;")
        add("    }")
        add("    float mean = (size - nan_count - inf_count > 0) ? sum / (size - nan_count - inf_count) : 0.0f;")
        add('    fprintf(stderr, "[DEBUG] %-30s size=%6d  nan=%d inf=%d zero=%d  range=[%.3e, %.3e] mean=%.3e\\n",')
        add("            name, size, nan_count, inf_count, zero_count, min_val, max_val, mean);")
        add("    if (nan_count > 0 || inf_count > 0) {")
        add('        fprintf(stderr, "[DEBUG] *** WARNING: %s has %d NaN and %d Inf values! ***\\n", name, nan_count, inf_count);')
        add("    }")
        add("}")
        add()

    # Parity helpers - only define in decode mode, use extern in prefill
    if emit_parity:
        add("/* ============================================================================")
        add(" * PARITY HELPERS (save buffers for PyTorch comparison)")
        add(" * ============================================================================ */")
        add()
        if mode == "decode":
            # Define the globals and functions only in decode
            add("const char *g_parity_dir = NULL;")
            add("int g_parity_token_idx = 0;")
            add()
            add("void parity_set_output_dir(const char *dir) {")
            add("    g_parity_dir = dir;")
            add("}")
            add()
            add("void parity_set_token_index(int idx) {")
            add("    g_parity_token_idx = idx;")
            add("}")
            add()
            add("void parity_save_buffer(const char *name, const float *buf, int size) {")
            add("    if (!g_parity_dir) return;")
            add("    char path[512];")
            add('    snprintf(path, sizeof(path), "%s/%s_tok%d.bin", g_parity_dir, name, g_parity_token_idx);')
            add('    FILE *f = fopen(path, "wb");')
            add('    if (!f) { fprintf(stderr, "[PARITY] Failed to open %s\\n", path); return; }')
            add("    fwrite(buf, sizeof(float), size, f);")
            add("    fclose(f);")
            add('    fprintf(stderr, "[PARITY] Saved %s (%d floats)\\n", path, size);')
            add("}")
        else:
            # Prefill: use extern declarations to reference decode's definitions
            add("extern const char *g_parity_dir;")
            add("extern int g_parity_token_idx;")
            add("extern void parity_set_output_dir(const char *dir);")
            add("extern void parity_set_token_index(int idx);")
            add("extern void parity_save_buffer(const char *name, const float *buf, int size);")
        add()

    # Magic header
    add("/* ============================================================================")
    add(" * MAGIC HEADER")
    add(" * ============================================================================ */")
    add()
    add("typedef struct __attribute__((packed)) {")
    add(f"    uint32_t magic;           /* 0x{v3.MAGIC_PREFIX:08X} */")
    add("    uint32_t version;          /* IR version */")
    add("    uint64_t total_bytes;")
    add("    uint64_t weight_bytes;")
    add("    uint64_t activation_bytes;")
    add("    uint32_t num_layers;")
    add("    uint32_t embed_dim;")
    add("    uint32_t num_heads;")
    add("    uint32_t vocab_size;")
    add("    uint32_t max_seq_len;")
    add("    uint32_t canary_count;")
    add("    uint8_t  reserved[8];       /* Pad to 64 bytes */")
    add("} MagicHeader;")
    add()
    add("_Static_assert(sizeof(MagicHeader) == 64, \"MagicHeader must be 64 bytes\");")
    add()

    # Allocation
    add("/* ============================================================================")
    add(" * ALLOCATION")
    add(" * ============================================================================ */")
    add()
    add(f"int {safe_name_lower}_model_allocate({safe_name}Model *model) {{")
    add(f"    size_t total = {safe_name}_TOTAL_BYTES;")
    add()
    add("#ifdef __linux__")
    add("    model->base = mmap(NULL, total,")
    add("                       PROT_READ | PROT_WRITE,")
    add("                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,")
    add("                       -1, 0);")
    add("    if (model->base == MAP_FAILED) {")
    add("        model->base = mmap(NULL, total,")
    add("                           PROT_READ | PROT_WRITE,")
    add("                           MAP_PRIVATE | MAP_ANONYMOUS,")
    add("                           -1, 0);")
    add("    }")
    add("    if (model->base == MAP_FAILED) {")
    add('        perror("mmap failed");')
    add("        return -1;")
    add("    }")
    add("#else")
    add("    model->base = aligned_alloc(64, total);")
    add("    if (!model->base) {")
    add('        perror("aligned_alloc failed");')
    add("        return -1;")
    add("    }")
    add("#endif")
    add()
    add("    model->total_bytes = total;")
    add()
    add("    /* Initialize magic header */")
    add("    MagicHeader *header = (MagicHeader *)model->base;")
    add(f"    header->magic = {safe_name}_MAGIC;")
    add("    header->version = 5;")
    add(f"    header->total_bytes = {safe_name}_TOTAL_BYTES;")
    add(f"    header->weight_bytes = {safe_name}_WEIGHT_BYTES;")
    add(f"    header->activation_bytes = {safe_name}_ACTIVATION_BYTES;")
    add(f"    header->num_layers = {safe_name}_NUM_LAYERS;")
    add(f"    header->embed_dim = {safe_name}_EMBED_DIM;")
    add(f"    header->num_heads = {safe_name}_NUM_HEADS;")
    add(f"    header->vocab_size = {safe_name}_VOCAB_SIZE;")
    add(f"    header->max_seq_len = {safe_name}_MAX_SEQ_LEN;")
    add(f"    header->canary_count = {safe_name}_CANARY_COUNT;")
    add()
    add("    /* Initialize canary guards */")
    add(f"    for (int i = 0; i < {safe_name}_CANARY_COUNT; i++) {{")
    add(f"        uint32_t *ptr = (uint32_t*)((char*)model->base + {safe_name}_CANARIES[i].offset);")
    add(f"        for (int j = 0; j < ({safe_name}_CANARY_SIZE / 4); j++) {{")
    add(f"            ptr[j] = {safe_name}_CANARY_VALUE;")
    add("        }")
    add("    }")
    add()
    add("    return 0;")
    add("}")
    add()

    # Free
    add(f"void {safe_name_lower}_model_free({safe_name}Model *model) {{")
    add("    if (!model || !model->base) return;")
    add("#ifdef __linux__")
    add("    munmap(model->base, model->total_bytes);")
    add("#else")
    add("    free(model->base);")
    add("#endif")
    add("    model->base = NULL;")
    add("    model->total_bytes = 0;")
    add("}")
    add()

    # Canary verify
    add(f"int {safe_name_lower}_verify_canaries({safe_name}Model *model) {{")
    add("    int errors = 0;")
    add("    uint32_t *ptr;")
    add()
    add(f"    for (int i = 0; i < {safe_name}_CANARY_COUNT; i++) {{")
    add(f"        ptr = (uint32_t*)((char*)model->base + {safe_name}_CANARIES[i].offset);")
    add(f"        for (int j = 0; j < 4; j++) {{")  # CANARY_SIZE / 4 = 16 / 4 = 4
    add(f"            if (ptr[j] != {safe_name}_CANARY_VALUE) {{")
    add(f'                fprintf(stderr, "CANARY CORRUPTION: %s at offset 0x%lX\\n",')
    add(f"                        {safe_name}_CANARIES[i].name,")
    add(f"                        {safe_name}_CANARIES[i].offset);")
    add("                errors++;")
    add("                break;")
    add("            }")
    add("        }")
    add("    }")
    add()
    add("    return errors;")
    add("}")
    add()

    # Alignment helper (used by prefill)
    add("/* ============================================================================")
    add(" * ALIGNMENT HELPERS")
    add(" * ============================================================================ */")
    add()
    add(f"static int {safe_name_lower}_align_elems(int elems, int elem_bytes, int align_bytes) {{")
    add("    int bytes = elems * elem_bytes;")
    add("    int aligned = (bytes + align_bytes - 1) / align_bytes * align_bytes;")
    add("    return aligned / elem_bytes;")
    add("}")
    add()

    # RoPE precompute
    add("/* ============================================================================")
    add(" * ROPE PRECOMPUTE")
    add(" * ============================================================================ */")
    add()
    add(f"void {safe_name_lower}_precompute_rope({safe_name}Model *model) {{")
    if has_rope:
        add(f"    const int T = {safe_name}_MAX_SEQ_LEN;")
        add(f"    const int D = {safe_name}_HEAD_DIM / 2;")
        add(f"    const float theta = {config.get('rope_theta', 10000.0)}f;")
        add()
        add(f"    float *cos_ptr = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_cos_cache);")
        add(f"    float *sin_ptr = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_sin_cache);")
        add()
        add("    for (int pos = 0; pos < T; pos++) {")
        add("        for (int i = 0; i < D; i++) {")
        add("            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)(D * 2));")
        add("            float angle = (float)pos * freq;")
        add("            cos_ptr[pos * D + i] = cosf(angle);")
        add("            sin_ptr[pos * D + i] = sinf(angle);")
        add("        }")
        add("    }")
    else:
        add("    (void)model;")
    add("}")
    add()

    # PREFILL helpers (shared by prefill + decode builds)


    def emit_prefill_impl() -> None:
        add("/* ============================================================================")
        add(" * EXPLICIT PER-LAYER PREFILL FUNCTIONS (v6 unrolled)")
        add(" * ============================================================================ */")
        add()

        for layer_id in range(num_layers):
            dtypes = layer_dtype_maps[layer_id] if layer_id < len(layer_dtype_maps) else {}
            wq_dt = get_quant_type(dtypes.get("wq", "fp32"))
            wk_dt = get_quant_type(dtypes.get("wk", "fp32"))
            wv_dt = get_quant_type(dtypes.get("wv", "fp32"))
            wo_dt = get_quant_type(dtypes.get("wo", "fp32"))
            w1_dt = get_quant_type(dtypes.get("w1", "fp32"))
            w2_dt = get_quant_type(dtypes.get("w2", "fp32"))

            wq_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(wq_dt, "gemm_blocked_serial")
            wk_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(wk_dt, "gemm_blocked_serial")
            wv_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(wv_dt, "gemm_blocked_serial")
            wo_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(wo_dt, "gemm_blocked_serial")
            w1_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(w1_dt, "gemm_blocked_serial")
            w2_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(w2_dt, "gemm_blocked_serial")

            add("/*")
            add(f" * Layer {layer_id}: wq={wq_dt} wk={wk_dt} wv={wv_dt} wo={wo_dt} w1={w1_dt} w2={w2_dt}")
            add(" */")
            add(f"static void {safe_name_lower}_layer_{layer_id}_prefill(")
            add(f"    {safe_name}Model *model,")
            add("    int num_tokens,")
            add("    int aligned_embed_dim,")
            add("    int aligned_head_dim,")
            add("    int aligned_intermediate_dim,")
            add("    int aligned_context_window")
            add(") {")
            add(f"    const {safe_name}LayerOffsets *L = &{safe_name}_LAYERS[{layer_id}];")
            add()

            if layer_id == 0:
                add(f"    float *input = {safe_name}_PTR(model, {safe_name}_HEADER.embedded_input);")
            else:
                add(f"    float *input = {safe_name}_PTR(model, {safe_name}_LAYERS[{layer_id - 1}].output);")

            add(f"    float *ln1_gamma = {safe_name}_PTR(model, L->ln1_gamma);")
            add(f"    float *ln1_out = {safe_name}_PTR(model, L->ln1_out);")
            add(f"    float *ln2_gamma = {safe_name}_PTR(model, L->ln2_gamma);")
            add(f"    float *ln2_out = {safe_name}_PTR(model, L->ln2_out);")
            add(f"    float *q = {safe_name}_PTR(model, L->q);")
            add(f"    float *k = {safe_name}_PTR(model, L->k);")
            add(f"    float *v = {safe_name}_PTR(model, L->v);")
            add(f"    float *attn_out = {safe_name}_PTR(model, L->attn_out);")
            add(f"    float *proj_tmp = {safe_name}_PTR(model, L->proj_tmp);")
            if has_proj_scratch:
                add(f"    float *proj_scratch = {safe_name}_PTR(model, L->proj_scratch);")
            else:
                add("    float *proj_scratch = NULL;")
            add(f"    float *residual1 = {safe_name}_PTR(model, L->residual1);")
            add(f"    float *fc1_out = {safe_name}_PTR(model, L->fc1_out);")
            add(f"    float *swiglu_out = {safe_name}_PTR(model, L->swiglu_out);")
            add(f"    float *mlp_out = {safe_name}_PTR(model, L->mlp_out);")
            add(f"    float *output = {safe_name}_PTR(model, L->output);")
            add()

            add(f"    const void *WQ = (const void *){safe_name}_PTR(model, L->wq);")
            add(f"    const void *WK = (const void *){safe_name}_PTR(model, L->wk);")
            add(f"    const void *WV = (const void *){safe_name}_PTR(model, L->wv);")
            add(f"    const void *WO = (const void *){safe_name}_PTR(model, L->wo);")
            add(f"    const void *W1 = (const void *){safe_name}_PTR(model, L->w1);")
            add(f"    const void *W2 = (const void *){safe_name}_PTR(model, L->w2);")

            if has_attention_biases:
                add(f"    const float *BQ = (const float *){safe_name}_PTR(model, L->bq);")
                add(f"    const float *BK = (const float *){safe_name}_PTR(model, L->bk);")
                add(f"    const float *BV = (const float *){safe_name}_PTR(model, L->bv);")
            else:
                add("    const float *BQ = NULL;")
                add("    const float *BK = NULL;")
                add("    const float *BV = NULL;")

            if has_output_bias:
                add(f"    const float *BO = (const float *){safe_name}_PTR(model, L->bo);")
            else:
                add("    const float *BO = NULL;")
            if has_mlp_bias1:
                add(f"    const float *B1 = (const float *){safe_name}_PTR(model, L->b1);")
            else:
                add("    const float *B1 = NULL;")
            if has_mlp_bias2:
                add(f"    const float *B2 = (const float *){safe_name}_PTR(model, L->b2);")
            else:
                add("    const float *B2 = NULL;")
            add()

            if has_rope:
                add(f"    float *rope_cos = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_cos_cache);")
                add(f"    float *rope_sin = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_sin_cache);")

            add()
            add(f"    const int H = {safe_name}_NUM_HEADS;")
            add(f"    const int H_kv = {safe_name}_NUM_KV_HEADS;")
            add(f"    const int head_dim = {safe_name}_HEAD_DIM;")
            add("    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;")
            add("    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;")
            add("    const size_t kv_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;")
            add()

            add("    /* RMSNorm before attention */")
            add("    rmsnorm_forward(input,")
            add("                    ln1_gamma,")
            add("                    ln1_out,")
            add("                    NULL,")
            add("                    num_tokens,")
            add(f"                    {safe_name}_EMBED_DIM,")
            add("                    aligned_embed_dim,")
            add(f"                    {config.get('rms_norm_eps', 1e-6)}f);")
            add()

            add("    /* Q projection (head-major) */")
            if wq_dt == "fp32":
                add("    const float *WQ_f = (const float *)WQ;")
                add("    for (int h = 0; h < H; ++h) {")
                add("        const float *wq_h = WQ_f + (size_t)h * head_w_elems;")
                add("        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;")
                add("        float *q_h = q + (size_t)h * q_head_stride;")
                add(f"        {wq_kernel}(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);")
                add("    }")
            else:
                add(f"    const size_t wq_head_bytes = ck_dtype_row_bytes({dtype_const(wq_dt)}, head_w_elems);")
                add("    const uint8_t *WQ_bytes = (const uint8_t *)WQ;")
                add("    for (int h = 0; h < H; ++h) {")
                add("        const void *wq_h = (const void *)(WQ_bytes + (size_t)h * wq_head_bytes);")
                add("        const float *bq_h = BQ ? (BQ + (size_t)h * (size_t)aligned_head_dim) : NULL;")
                add("        float *q_h = q + (size_t)h * q_head_stride;")
                add(f"        {wq_kernel}(ln1_out, wq_h, bq_h, q_h, num_tokens, aligned_head_dim, aligned_embed_dim);")
                add("    }")
            add()

            add("    /* K projection (head-major) */")
            if wk_dt == "fp32":
                add("    const float *WK_f = (const float *)WK;")
                add("    for (int h = 0; h < H_kv; ++h) {")
                add("        const float *wk_h = WK_f + (size_t)h * head_w_elems;")
                add("        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;")
                add("        float *k_h = k + (size_t)h * kv_head_stride;")
                add(f"        {wk_kernel}(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);")
                add("    }")
            else:
                add(f"    const size_t wk_head_bytes = ck_dtype_row_bytes({dtype_const(wk_dt)}, head_w_elems);")
                add("    const uint8_t *WK_bytes = (const uint8_t *)WK;")
                add("    for (int h = 0; h < H_kv; ++h) {")
                add("        const void *wk_h = (const void *)(WK_bytes + (size_t)h * wk_head_bytes);")
                add("        const float *bk_h = BK ? (BK + (size_t)h * (size_t)aligned_head_dim) : NULL;")
                add("        float *k_h = k + (size_t)h * kv_head_stride;")
                add(f"        {wk_kernel}(ln1_out, wk_h, bk_h, k_h, num_tokens, aligned_head_dim, aligned_embed_dim);")
                add("    }")
            add()

            add("    /* V projection (head-major) */")
            if wv_dt == "fp32":
                add("    const float *WV_f = (const float *)WV;")
                add("    for (int h = 0; h < H_kv; ++h) {")
                add("        const float *wv_h = WV_f + (size_t)h * head_w_elems;")
                add("        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;")
                add("        float *v_h = v + (size_t)h * kv_head_stride;")
                add(f"        {wv_kernel}(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);")
                add("    }")
            else:
                add(f"    const size_t wv_head_bytes = ck_dtype_row_bytes({dtype_const(wv_dt)}, head_w_elems);")
                add("    const uint8_t *WV_bytes = (const uint8_t *)WV;")
                add("    for (int h = 0; h < H_kv; ++h) {")
                add("        const void *wv_h = (const void *)(WV_bytes + (size_t)h * wv_head_bytes);")
                add("        const float *bv_h = BV ? (BV + (size_t)h * (size_t)aligned_head_dim) : NULL;")
                add("        float *v_h = v + (size_t)h * kv_head_stride;")
                add(f"        {wv_kernel}(ln1_out, wv_h, bv_h, v_h, num_tokens, aligned_head_dim, aligned_embed_dim);")
                add("    }")
            add()

            if has_rope:
                add("    /* RoPE */")
                add("    rope_forward_qk(q,")
                add("                    k,")
                add("                    rope_cos,")
                add("                    rope_sin,")
                add("                    H,")
                add("                    H_kv,")
                add("                    num_tokens,")
                add("                    head_dim,")
                add("                    aligned_head_dim,")
                add("                    0);")
                add()

            add("    /* Attention (prefill, causal) */")
            add("    attention_forward_causal_head_major_gqa_flash(q,")
            add("                                                   k,")
            add("                                                   v,")
            add("                                                   attn_out,")
            add("                                                   H,")
            add("                                                   H_kv,")
            add("                                                   num_tokens,")
            add("                                                   head_dim,")
            add("                                                   aligned_head_dim);")
            add()

            add("    /* Output projection (flatten head-major to token-major) */")
            add("    const int K = H * aligned_head_dim;")
            add("    if (K != aligned_embed_dim) {")
            add("        return;")
            add("    }")
            add("    const float *proj_in = attn_out;")
            add("    if (H > 1) {")
            add("        if (!proj_scratch) {")
            add("            return;")
            add("        }")
            add("        for (int t = 0; t < num_tokens; ++t) {")
            add("            float *dst = proj_scratch + (size_t)t * (size_t)aligned_embed_dim;")
            add("            for (int h = 0; h < H; ++h) {")
            add("                const float *src = attn_out + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;")
            add("                memcpy(dst + (size_t)h * (size_t)aligned_head_dim,")
            add("                       src,")
            add("                       (size_t)aligned_head_dim * sizeof(float));")
            add("            }")
            add("        }")
            add("        proj_in = proj_scratch;")
            add("    }")

            add(f"    {wo_kernel}(proj_in, WO, BO, proj_tmp, num_tokens, aligned_embed_dim, K);")
            add()

            add("    /* Residual add */")
            add(f"    {safe_name_lower}_residual_add_token_major(input, proj_tmp, residual1, num_tokens, aligned_embed_dim);")
            add()

            add("    /* RMSNorm before MLP */")
            add("    rmsnorm_forward(residual1,")
            add("                    ln2_gamma,")
            add("                    ln2_out,")
            add("                    NULL,")
            add("                    num_tokens,")
            add(f"                    {safe_name}_EMBED_DIM,")
            add("                    aligned_embed_dim,")
            add(f"                    {config.get('rms_norm_eps', 1e-6)}f);")
            add()

            add("    /* MLP (SwiGLU) */")
            add(f"    {w1_kernel}(ln2_out, W1, B1, fc1_out, num_tokens, 2 * aligned_intermediate_dim, aligned_embed_dim);")
            add("    swiglu_forward(fc1_out, swiglu_out, num_tokens, aligned_intermediate_dim);")
            add(f"    {w2_kernel}(swiglu_out, W2, B2, mlp_out, num_tokens, aligned_embed_dim, aligned_intermediate_dim);")
            add()

            add("    /* Final residual add */")
            add(f"    {safe_name_lower}_residual_add_token_major(residual1, mlp_out, output, num_tokens, aligned_embed_dim);")
            add("}")
            add()

        add("/* ============================================================================")
        add(" * FORWARD PASS (PREFILL)")
        add(" * ============================================================================ */")
        add()
        add(f"static void {safe_name_lower}_forward_prefill_impl(")
        add(f"    {safe_name}Model *model,")
        add("    const int *tokens,")
        add("    int num_tokens")
        add(") {")
        add("    if (!model || !tokens || num_tokens <= 0) {")
        add("        return;")
        add("    }")
        add()
        add(f"    const int elem_bytes = {safe_name}_DTYPE_BYTES;")
        add(f"    const int aligned_embed_dim = {aligned_expr(aligned_embed, f'{safe_name_lower}_align_elems({safe_name}_EMBED_DIM, elem_bytes, 64)')};")
        add(f"    const int aligned_head_dim = {aligned_expr(aligned_head, f'{safe_name_lower}_align_elems({safe_name}_HEAD_DIM, elem_bytes, 64)')};")
        add(f"    const int aligned_intermediate_dim = {aligned_expr(aligned_intermediate, f'{safe_name_lower}_align_elems({safe_name}_INTERMEDIATE, elem_bytes, 64)')};")
        add(f"    const int aligned_context_window = {aligned_expr(aligned_context, f'{safe_name_lower}_align_elems({safe_name}_MAX_SEQ_LEN, elem_bytes, 64)')};")
        add()
        add(f"    float *embed_out = {safe_name}_PTR(model, {safe_name}_HEADER.embedded_input);")
        embed_kernel = DTYPE_TO_EMBEDDING_KERNEL.get(embed_quant_type, "embedding_forward")
        if embed_quant_type != "fp32":
            add(f"    const void *embed_weight = (const void *){safe_name}_PTR(model, {safe_name}_HEADER.token_emb);")
            add(f"    {embed_kernel}((const int32_t *)tokens,")
            add("                          num_tokens,")
            add(f"                          {safe_name}_VOCAB_SIZE,")
            add("                          embed_weight,")
            add("                          NULL,")
            add("                          embed_out,")
            add(f"                          {safe_name}_EMBED_DIM,")
            add("                          aligned_embed_dim,")
            add("                          num_tokens,")
            add("                          0);")
        else:
            add(f"    float *embed_weight = {safe_name}_PTR(model, {safe_name}_HEADER.token_emb);")
            add("    embedding_forward((const int32_t *)tokens,")
            add("                      num_tokens,")
            add(f"                      {safe_name}_VOCAB_SIZE,")
            add("                      embed_weight,")
            add("                      NULL,")
            add("                      embed_out,")
            add(f"                      {safe_name}_EMBED_DIM,")
            add("                      aligned_embed_dim,")
            add("                      num_tokens,")
            add("                      0);")
        add()

        for layer_id in range(num_layers):
            add(f"    {safe_name_lower}_layer_{layer_id}_prefill(")
            add("        model,")
            add("        num_tokens,")
            add("        aligned_embed_dim,")
            add("        aligned_head_dim,")
            add("        aligned_intermediate_dim,")
            add("        aligned_context_window);")
            add()
            add("    kv_cache_repack_head_major_inplace(")
            add(f"        {safe_name}_PTR(model, {safe_name}_LAYERS[{layer_id}].k),")
            add(f"        {safe_name}_NUM_KV_HEADS,")
            add("        num_tokens,")
            add("        aligned_context_window,")
            add("        aligned_head_dim);")
            add("    kv_cache_repack_head_major_inplace(")
            add(f"        {safe_name}_PTR(model, {safe_name}_LAYERS[{layer_id}].v),")
            add(f"        {safe_name}_NUM_KV_HEADS,")
            add("        num_tokens,")
            add("        aligned_context_window,")
            add("        aligned_head_dim);")
            add()

        add(f"    float *last_hidden = {safe_name}_PTR(model, {safe_name}_LAYERS[{safe_name}_NUM_LAYERS - 1].output);")
        add(f"    float *final_ln_weight = {safe_name}_PTR(model, {safe_name}_FOOTER.final_ln_weight);")
        add(f"    float *final_out = {safe_name}_PTR(model, {safe_name}_FOOTER.final_output);")
        add("    rmsnorm_forward(last_hidden,")
        add("                   final_ln_weight,")
        add("                   final_out,")
        add("                   NULL,")
        add("                   num_tokens,")
        add(f"                   {safe_name}_EMBED_DIM,")
        add("                   aligned_embed_dim,")
        add(f"                   {config.get('rms_norm_eps', 1e-6)}f);")
        add()
        add(f"    float *logits = {safe_name}_PTR(model, {safe_name}_FOOTER.logits);")
        lm_head_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(lm_head_quant_type, "gemm_blocked_serial")
        if lm_head_quant_type != "fp32":
            add(f"    const void *lm_head = (const void *){safe_name}_PTR(model, {safe_name}_FOOTER.lm_head_weight);")
            if lm_head_use_q4_k:
                add("    const size_t q8_bytes = ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)aligned_embed_dim);")
                add("    for (int t = 0; t < num_tokens; ++t) {")
                add("        uint8_t q8_buf[q8_bytes];")
                add("        const float *row = final_out + (size_t)t * (size_t)aligned_embed_dim;")
                add(f"        float *logits_row = logits + (size_t)t * (size_t){safe_name}_VOCAB_SIZE;")
                add("        quantize_row_q8_k(row, q8_buf, aligned_embed_dim);")
                add("        gemm_nt_q4_k_q8_k(q8_buf,")
                add("                          lm_head,")
                add("                          NULL,")
                add("                          logits_row,")
                add("                          1,")
                add(f"                          {safe_name}_VOCAB_SIZE,")
                add("                          aligned_embed_dim);")
                add("    }")
            else:
                add("    for (int t = 0; t < num_tokens; ++t) {")
                add("        const float *row = final_out + (size_t)t * (size_t)aligned_embed_dim;")
                add(f"        float *logits_row = logits + (size_t)t * (size_t){safe_name}_VOCAB_SIZE;")
                add(f"        {lm_head_kernel}(row,")
                add("                       lm_head,")
                add("                       NULL,")
                add("                       logits_row,")
                add("                       1,")
                add(f"                       {safe_name}_VOCAB_SIZE,")
                add("                       aligned_embed_dim);")
                add("    }")
        else:
            add(f"    float *lm_head = {safe_name}_PTR(model, {safe_name}_FOOTER.lm_head_weight);")
            add("    gemm_blocked_serial(final_out,")
            add("                        lm_head,")
            add("                        NULL,")
            add("                        logits,")
            add("                        num_tokens,")
            add(f"                        {safe_name}_VOCAB_SIZE,")
            add("                        aligned_embed_dim);")
        add("}")
        add()
    emit_prefill_impl()


    # DECODE MODE - Generate explicit per-layer functions
    if mode == "decode":
        add("/* ============================================================================")
        add(" * EXPLICIT PER-LAYER DECODE FUNCTIONS (v6 unrolled)")
        add(" * ============================================================================ */")
        add()

        # Generate a separate function for each layer
        for layer_id in range(num_layers):
            dtypes = layer_dtype_maps[layer_id] if layer_id < len(layer_dtype_maps) else {}
            wq_dt = get_quant_type(dtypes.get("wq", "fp32"))
            wk_dt = get_quant_type(dtypes.get("wk", "fp32"))
            wv_dt = get_quant_type(dtypes.get("wv", "fp32"))
            wo_dt = get_quant_type(dtypes.get("wo", "fp32"))
            w1_dt = get_quant_type(dtypes.get("w1", "fp32"))
            w2_dt = get_quant_type(dtypes.get("w2", "fp32"))

            add(f"/*")
            add(f" * Layer {layer_id}: wq={wq_dt} wk={wk_dt} wv={wv_dt} wo={wo_dt} w1={w1_dt} w2={w2_dt}")
            add(f" */")
            add(f"static void {safe_name_lower}_layer_{layer_id}_decode(")
            add(f"    {safe_name}Model *model,")
            add("    int token_index,")
            add("    int aligned_embed_dim,")
            add("    int aligned_head_dim,")
            add("    int aligned_intermediate_dim,")
            add("    int aligned_context_window")
            add(") {")
            add(f"    const {safe_name}LayerOffsets *L = &{safe_name}_LAYERS[{layer_id}];")
            add()

            # Input pointer - layer 0 uses embedded_input, others use previous layer output
            if layer_id == 0:
                add(f"    float *input = {safe_name}_PTR(model, {safe_name}_HEADER.embedded_input);")
            else:
                add(f"    float *input = {safe_name}_PTR(model, {safe_name}_LAYERS[{layer_id - 1}].output);")
            add()

            # Get all buffer pointers
            add(f"    float *ln1_gamma = {safe_name}_PTR(model, L->ln1_gamma);")
            add(f"    float *ln1_out = {safe_name}_PTR(model, L->ln1_out);")
            add(f"    float *ln2_gamma = {safe_name}_PTR(model, L->ln2_gamma);")
            add(f"    float *ln2_out = {safe_name}_PTR(model, L->ln2_out);")
            add(f"    float *k_cache = {safe_name}_PTR(model, L->k);")
            add(f"    float *v_cache = {safe_name}_PTR(model, L->v);")
            add(f"    float *proj_tmp = {safe_name}_PTR(model, L->proj_tmp);")
            if has_proj_scratch:
                add(f"    float *proj_scratch = {safe_name}_PTR(model, L->proj_scratch);")
            add(f"    float *residual1 = {safe_name}_PTR(model, L->residual1);")
            add(f"    float *mlp_out = {safe_name}_PTR(model, L->mlp_out);")
            add(f"    float *output = {safe_name}_PTR(model, L->output);")
            add()

            # Weight pointers with explicit types
            add(f"    /* Weights (explicit types for layer {layer_id}) */")
            add(f"    const void *WQ = (const void *){safe_name}_PTR(model, L->wq);  /* {wq_dt.upper()} */")
            add(f"    const void *WK = (const void *){safe_name}_PTR(model, L->wk);  /* {wk_dt.upper()} */")
            add(f"    const void *WV = (const void *){safe_name}_PTR(model, L->wv);  /* {wv_dt.upper()} */")
            add(f"    const void *WO = (const void *){safe_name}_PTR(model, L->wo);  /* {wo_dt.upper()} */")
            add(f"    const void *W1 = (const void *){safe_name}_PTR(model, L->w1);  /* {w1_dt.upper()} (gate+up) */")
            add(f"    const void *W2 = (const void *){safe_name}_PTR(model, L->w2);  /* {w2_dt.upper()} (down) */")
            add()

            # Attention biases (Qwen2-style)
            if has_attention_biases:
                add(f"    /* Attention biases (Qwen2-style) */")
                add(f"    const float *BQ = (const float *){safe_name}_PTR(model, L->bq);")
                add(f"    const float *BK = (const float *){safe_name}_PTR(model, L->bk);")
                add(f"    const float *BV = (const float *){safe_name}_PTR(model, L->bv);")
                add()

            # RoPE pointers
            if has_rope:
                add(f"    float *rope_cos = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_cos_cache);")
                add(f"    float *rope_sin = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_sin_cache);")
                add()

            # Local buffers for attention
            add(f"    const int H = {safe_name}_NUM_HEADS;")
            add(f"    const int H_kv = {safe_name}_NUM_KV_HEADS;")
            add(f"    const int head_dim = {safe_name}_HEAD_DIM;")
            add()
            add("    float q_token[H * aligned_head_dim];")
            add("    float k_token[H_kv * aligned_head_dim];")
            add("    float v_token[H_kv * aligned_head_dim];")
            add("    float attn_token[H * aligned_head_dim];")
            add()
            # Local buffers for MLP (fc1_out = 2x intermediate for gate+up, swiglu_out = intermediate)
            add("    /* Local MLP buffers (avoid layout dependencies for intermediate values) */")
            add("    float fc1_out[2 * aligned_intermediate_dim];")
            add("    float swiglu_out[aligned_intermediate_dim];")
            add()

            # Step 1: RMSNorm
            add("    /* Step 1: RMSNorm before attention */")
            add("    rmsnorm_forward(input,")
            add("                    ln1_gamma,")
            add("                    ln1_out,")
            add("                    NULL,")
            add("                    1,")
            add(f"                    {safe_name}_EMBED_DIM,")
            add("                    aligned_embed_dim,")
            add(f"                    {config.get('rms_norm_eps', 1e-6)}f);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_ln1_out", ln1_out, aligned_embed_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_ln1_out", ln1_out, aligned_embed_dim);')
            add()

            # Step 2: QKV projection with explicit kernel names
            add("    /* Step 2: QKV projection */")
            wq_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(wq_dt, "gemm_blocked_serial")
            wk_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(wk_dt, "gemm_blocked_serial")
            wv_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(wv_dt, "gemm_blocked_serial")

            # Use actual biases if model has attention biases, otherwise NULL
            bq_arg = "BQ" if has_attention_biases else "NULL"
            bk_arg = "BK" if has_attention_biases else "NULL"
            bv_arg = "BV" if has_attention_biases else "NULL"

            add(f"    /* Q projection: {wq_dt.upper()} -> {wq_kernel} */")
            add(f"    {wq_kernel}(ln1_out, WQ, {bq_arg}, q_token, 1, H * head_dim, aligned_embed_dim);")
            add()
            add(f"    /* K projection: {wk_dt.upper()} -> {wk_kernel} */")
            add(f"    {wk_kernel}(ln1_out, WK, {bk_arg}, k_token, 1, H_kv * head_dim, aligned_embed_dim);")
            add()
            add(f"    /* V projection: {wv_dt.upper()} -> {wv_kernel} */")
            add(f"    {wv_kernel}(ln1_out, WV, {bv_arg}, v_token, 1, H_kv * head_dim, aligned_embed_dim);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_q_token", q_token, H * aligned_head_dim);')
                add(f'    debug_check_buffer("layer{layer_id}_k_token", k_token, H_kv * aligned_head_dim);')
                add(f'    debug_check_buffer("layer{layer_id}_v_token", v_token, H_kv * aligned_head_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_q_proj", q_token, H * aligned_head_dim);')
                add(f'    parity_save_buffer("layer_{layer_id}_k_proj", k_token, H_kv * aligned_head_dim);')
                add(f'    parity_save_buffer("layer_{layer_id}_v_proj", v_token, H_kv * aligned_head_dim);')
            add()

            # Step 3: RoPE
            if has_rope:
                add("    /* Step 3: RoPE */")
                add("    rope_forward_qk(q_token,")
                add("                    k_token,")
                add("                    rope_cos,")
                add("                    rope_sin,")
                add("                    H,")
                add("                    H_kv,")
                add("                    1,")
                add("                    head_dim,")
                add("                    aligned_head_dim,")
                add("                    token_index);")
                if emit_parity:
                    add(f'    parity_save_buffer("layer_{layer_id}_q_rope", q_token, H * aligned_head_dim);')
                    add(f'    parity_save_buffer("layer_{layer_id}_k_rope", k_token, H_kv * aligned_head_dim);')
                add()

            # Step 4: KV cache write
            add("    /* Step 4: KV cache write */")
            add("    kv_cache_write_head_major(k_token,")
            add("                              v_token,")
            add("                              k_cache,")
            add("                              v_cache,")
            add("                              H_kv,")
            add("                              token_index,")
            add("                              aligned_context_window,")
            add("                              head_dim,")
            add("                              aligned_head_dim);")
            add()

            # Step 5: Attention
            add("    /* Step 5: Attention (decode) */")
            add("    attention_forward_decode_head_major_gqa_regular(q_token,")
            add("                                                   k_cache,")
            add("                                                   v_cache,")
            add("                                                   attn_token,")
            add("                                                   H,")
            add("                                                   H_kv,")
            add("                                                   token_index + 1,")
            add("                                                   aligned_context_window,")
            add("                                                   head_dim,")
            add("                                                   aligned_head_dim);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_attn_token", attn_token, H * aligned_head_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_attn", attn_token, H * aligned_head_dim);')
            add()

            # Step 6: Output projection
            add("    /* Step 6: Output projection */")
            wo_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(wo_dt, "gemm_blocked_serial")
            add(f"    /* WO projection: {wo_dt.upper()} -> {wo_kernel} */")
            add(f"    {wo_kernel}(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_proj_tmp", proj_tmp, aligned_embed_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_attn_proj", proj_tmp, aligned_embed_dim);')
            add()

            # Step 7: Residual add
            add("    /* Step 7: Residual add */")
            add(f"    {safe_name_lower}_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_residual1", residual1, aligned_embed_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_residual1", residual1, aligned_embed_dim);')
            add()

            # Step 8: RMSNorm before MLP
            add("    /* Step 8: RMSNorm before MLP */")
            add("    rmsnorm_forward(residual1,")
            add("                    ln2_gamma,")
            add("                    ln2_out,")
            add("                    NULL,")
            add("                    1,")
            add(f"                    {safe_name}_EMBED_DIM,")
            add("                    aligned_embed_dim,")
            add(f"                    {config.get('rms_norm_eps', 1e-6)}f);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_ln2_out", ln2_out, aligned_embed_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_ln2_out", ln2_out, aligned_embed_dim);')
            add()

            # Step 9: MLP (gate + up + SwiGLU + down)
            add("    /* Step 9: MLP (SwiGLU) */")
            w1_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(w1_dt, "gemm_blocked_serial")
            w2_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(w2_dt, "gemm_blocked_serial")

            add(f"    /* Gate+Up projection: {w1_dt.upper()} -> {w1_kernel} */")
            add(f"    {w1_kernel}(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_fc1_out", fc1_out, 2 * aligned_intermediate_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_fc1", fc1_out, 2 * aligned_intermediate_dim);')
            add()
            add("    /* SwiGLU activation */")
            add(f"    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_swiglu_out", swiglu_out, aligned_intermediate_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_swiglu", swiglu_out, aligned_intermediate_dim);')
            add()
            add(f"    /* Down projection: {w2_dt.upper()} -> {w2_kernel} */")
            add(f"    {w2_kernel}(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_mlp_out", mlp_out, aligned_embed_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_mlp", mlp_out, aligned_embed_dim);')
            add()

            # Step 10: Final residual add
            add("    /* Step 10: Final residual add */")
            add(f"    {safe_name_lower}_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);")
            if emit_debug:
                add(f'    debug_check_buffer("layer{layer_id}_output", output, aligned_embed_dim);')
            if emit_parity:
                add(f'    parity_save_buffer("layer_{layer_id}_output", output, aligned_embed_dim);')
            add("}")
            add()

        # Main decode function that calls each layer explicitly
        add("/* ============================================================================")
        add(" * DECODE TOKEN (calls each layer explicitly)")
        add(" * ============================================================================ */")
        add()
        add(f"static void {safe_name_lower}_decode_token(")
        add(f"    {safe_name}Model *model,")
        add("    const int *token,")
        add("    int token_index")
        add(") {")
        add("    if (!model || !token) return;")
        add()
        add(f"    const int aligned_embed_dim = {aligned_expr(aligned_embed, str(config.get('hidden_size', 896)))};")
        add(f"    const int aligned_head_dim = {aligned_expr(aligned_head, str(config.get('hidden_size', 896) // config.get('num_attention_heads', 14)))};")
        add(f"    const int aligned_intermediate_dim = {aligned_expr(aligned_intermediate, str(config.get('intermediate_size', 4864)))};")
        add(f"    const int aligned_context_window = {aligned_expr(aligned_context, str(config.get('max_position_embeddings', 32768)))};")
        add()
        add("    if (token_index < 0 || token_index >= aligned_context_window) return;")
        add()

        # Embedding
        add("    /* Embedding lookup */")
        add(f"    float *embed_out = {safe_name}_PTR(model, {safe_name}_HEADER.embedded_input);")
        embed_kernel = DTYPE_TO_EMBEDDING_KERNEL.get(embed_quant_type, "embedding_forward")
        if embed_quant_type != "fp32":
            add(f"    const void *embed_weight = (const void *){safe_name}_PTR(model, {safe_name}_HEADER.token_emb);")
            add(f"    /* Embedding: {embed_quant_type.upper()} -> {embed_kernel} */")
            add(f"    {embed_kernel}((const int32_t *)token,")
            add("                          1,")
            add(f"                          {safe_name}_VOCAB_SIZE,")
            add("                          embed_weight,")
            add("                          NULL,")
            add("                          embed_out,")
            add(f"                          {safe_name}_EMBED_DIM,")
            add("                          aligned_embed_dim,")
            add("                          1,")
            add("                          0);")
        else:
            add(f"    float *embed_weight = {safe_name}_PTR(model, {safe_name}_HEADER.token_emb);")
            add("    embedding_forward((const int32_t *)token,")
            add("                      1,")
            add(f"                      {safe_name}_VOCAB_SIZE,")
            add("                      embed_weight,")
            add("                      NULL,")
            add("                      embed_out,")
            add(f"                      {safe_name}_EMBED_DIM,")
            add("                      aligned_embed_dim,")
            add("                      1,")
            add("                      0);")
        if emit_debug:
            add('    debug_check_buffer("embed_out", embed_out, aligned_embed_dim);')
        if emit_parity:
            add('    parity_save_buffer("embed_out", embed_out, aligned_embed_dim);')
        add()

        # Call each layer explicitly
        add("    /* Process each layer explicitly */")
        for layer_id in range(num_layers):
            add(f"    {safe_name_lower}_layer_{layer_id}_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);")
        add()

        # Final RMSNorm
        add("    /* Final RMSNorm */")
        add(f"    float *last_hidden = {safe_name}_PTR(model, {safe_name}_LAYERS[{num_layers - 1}].output);")
        add(f"    float *final_ln_weight = {safe_name}_PTR(model, {safe_name}_FOOTER.final_ln_weight);")
        add(f"    float *final_out = {safe_name}_PTR(model, {safe_name}_FOOTER.final_output);")
        add("    rmsnorm_forward(last_hidden,")
        add("                    final_ln_weight,")
        add("                    final_out,")
        add("                    NULL,")
        add("                    1,")
        add(f"                    {safe_name}_EMBED_DIM,")
        add("                    aligned_embed_dim,")
        add(f"                    {config.get('rms_norm_eps', 1e-6)}f);")
        if emit_debug:
            add('    debug_check_buffer("final_out", final_out, aligned_embed_dim);')
        if emit_parity:
            add('    parity_save_buffer("final_out", final_out, aligned_embed_dim);')
        add()

        # LM head
        add("    /* LM head projection */")
        add(f"    float *logits = {safe_name}_PTR(model, {safe_name}_FOOTER.logits);")
        lm_head_kernel = DTYPE_TO_GEMM_NT_KERNEL.get(lm_head_quant_type, "gemm_blocked_serial")
        if lm_head_quant_type != "fp32":
            add(f"    const void *lm_head = (const void *){safe_name}_PTR(model, {safe_name}_FOOTER.lm_head_weight);")
            add(f"    /* LM head: {lm_head_quant_type.upper()} -> {lm_head_kernel} */")
            add(f"    {lm_head_kernel}(final_out, lm_head, NULL, logits, 1, {safe_name}_VOCAB_SIZE, aligned_embed_dim);")
        else:
            add(f"    float *lm_head = {safe_name}_PTR(model, {safe_name}_FOOTER.lm_head_weight);")
            add(f"    gemm_blocked_serial(final_out, lm_head, NULL, logits, 1, {safe_name}_VOCAB_SIZE, aligned_embed_dim);")
        if emit_debug:
            add(f'    debug_check_buffer("logits", logits, {safe_name}_VOCAB_SIZE);')
        if emit_parity:
            add(f'    parity_save_buffer("logits", logits, {safe_name}_VOCAB_SIZE);')
        add("}")
        add()

        # Public API
        add("/* ============================================================================")
        add(" * PUBLIC API")
        add(" * ============================================================================ */")
        add()
        add(f"void {safe_name_lower}_forward(")
        add(f"    {safe_name}Model *model,")
        add("    const int *tokens,")
        add("    int num_tokens")
        add(") {")
        add("    if (!model || !tokens || num_tokens <= 0) return;")
        add(f"    {safe_name_lower}_forward_prefill_impl(model, tokens, num_tokens);")
        add("}")
        add()
        add(f"void {safe_name_lower}_decode({safe_name}Model *model, const int *token, int token_index) {{")
        add(f"    {safe_name_lower}_decode_token(model, token, token_index);")
        add("}")
        add()

    else:  # prefill mode
        add("/* ============================================================================")
        add(" * PREFILL (v6)")
        add(" * ============================================================================ */")
        add()
        add(f"void {safe_name_lower}_forward(")
        add(f"    {safe_name}Model *model,")
        add("    const int *tokens,")
        add("    int num_tokens")
        add(") {")
        add("    if (!model || !tokens || num_tokens <= 0) return;")
        add(f"    {safe_name_lower}_forward_prefill_impl(model, tokens, num_tokens);")
        add("}")
        add()

    # Write output
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[v6.c] Written: {output_path}")
    print(f"[v6.c] Mode: {mode}, Layers: {num_layers} (explicit unrolled)")


if __name__ == "__main__":
    # This module is imported by build_ir_v6.py
    print("codegen_v6.py - Use build_ir_v6.py to generate code")
