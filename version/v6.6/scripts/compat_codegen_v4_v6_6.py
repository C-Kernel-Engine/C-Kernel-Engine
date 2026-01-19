# ============================================================================
# ORIGIN: Copied from v4/codegen_v4.py for v6.6 independence
# PURPOSE: Legacy codegen fallback for v4-style code generation
# NOTE: Compatibility layer - prefer codegen_v6_6.py for new development
# ============================================================================

#!/usr/bin/env python3
"""
codegen_v4.py - C source emitter for IR v4 layouts.

Emits forward implementations that call real kernels (fp32 + Q4_K paths).
"""

from datetime import datetime
import os
import sys
from pathlib import Path
from typing import List

_SCRIPT_DIR = Path(__file__).resolve().parent
# Try v3 first (for compatibility), then fall back to v6
_V3_DIR = _SCRIPT_DIR.parent.parent / "scripts" / "v3"
if _V3_DIR.is_dir():
    sys.path.insert(0, str(_V3_DIR))
    import build_ir_v3 as v3
else:
    # Fall back to v6
    _V6_DIR = _SCRIPT_DIR.parent.parent / "scripts" / "v6"
    if _V6_DIR.is_dir():
        sys.path.insert(0, str(_V6_DIR))
        import build_ir_v6 as v3
    else:
        raise ImportError("Neither v3 nor v6 build_ir module found")


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


def emit_c_source_v4(layout: v3.ModelLayout,
                     output_path: str,
                     header_name: str,
                     mode: str,
                     emit_main: bool = False,
                     emit_debug: bool = False,
                     emit_parity: bool = False) -> None:
    """Emit generated_model.c with real kernel calls (fp32-only).

    Args:
        emit_debug: If True, insert debug prints after each layer to detect NaN/zero outputs.
        emit_parity: If True, save intermediate buffers to files for comparison with PyTorch.
    """
    if mode not in ("prefill", "decode"):
        raise ValueError(f"v4 codegen only supports prefill/decode (got: {mode})")
    config = layout.config
    section = layout.sections[0]

    safe_name = layout.name.upper().replace("-", "_").replace(".", "_")
    safe_name_lower = safe_name.lower()

    weight_dtype = str(config.get("weight_dtype", "")).lower()
    use_q4_k = weight_dtype.startswith("q4_k")

    def buffer_dtype(buffers, name: str) -> str:
        for buf in buffers:
            if buf.name == name:
                return str(buf.dtype).lower()
        return ""

    token_emb_dtype = buffer_dtype(section.header_buffers, "token_emb")
    lm_head_dtype = buffer_dtype(section.footer_buffers, "lm_head_weight")
    # When embeddings are tied, lm_head uses same weights as token_emb
    # so it inherits the token_emb dtype (even if layout says fp32)
    if config.get("tie_word_embeddings", True) and token_emb_dtype:
        lm_head_dtype = token_emb_dtype

    # Determine embedding quantization type (None = FP32, or specific quant type)
    def get_quant_type(dtype: str) -> str:
        """Return quant type string or empty string for FP32."""
        if dtype.startswith("q4_k"):
            return "q4_k"
        if dtype.startswith("q6_k"):
            return "q6_k"
        if dtype.startswith("q8_0"):
            return "q8_0"
        if dtype.startswith("q5_0"):
            return "q5_0"
        if dtype.startswith("q5_1"):
            return "q5_1"
        if dtype.startswith("q4_0"):
            return "q4_0"
        if dtype.startswith("q4_1"):
            return "q4_1"
        return ""

    embed_quant_type = get_quant_type(token_emb_dtype)
    lm_head_quant_type = get_quant_type(lm_head_dtype)

    # Backwards compatibility
    embed_use_q4_k = embed_quant_type == "q4_k"
    lm_head_use_q4_k = lm_head_quant_type == "q4_k"

    # Build per-layer dtype maps for mixed quant support
    layer_dtype_maps = []  # List of {weight_name: dtype} per layer
    if section.layers:
        for layer in section.layers:
            layer_map = {}
            for buf in layer.buffers:
                # Parse "layer.N.weight_name"
                parts = buf.name.split(".", 2)
                if len(parts) == 3 and parts[0] == "layer":
                    layer_map[parts[2]] = str(buf.dtype).lower()
            layer_dtype_maps.append(layer_map)

    # Legacy: use layer 0 for backwards compat checks
    layer_dtype_map = layer_dtype_maps[0] if layer_dtype_maps else {}

    def layer_weight_dtype(name: str, layer_id: int = 0) -> str:
        if layer_id < len(layer_dtype_maps):
            return layer_dtype_maps[layer_id].get(name, "")
        return layer_dtype_map.get(name, "")

    # Check if any layer differs from layer 0 (mixed quant)
    def has_mixed_layer_dtypes() -> bool:
        if len(layer_dtype_maps) <= 1:
            return False
        ref = layer_dtype_maps[0]
        for layer_map in layer_dtype_maps[1:]:
            for key in ["wq", "wk", "wv", "wo", "w1", "w2"]:
                if layer_map.get(key, "") != ref.get(key, ""):
                    return True
        return False

    mixed_quant = has_mixed_layer_dtypes()

    def dtype_const(dtype: str) -> str:
        """Convert dtype string to CK_DT_* constant."""
        if dtype.startswith("q4_k"):
            return "CK_DT_Q4_K"
        if dtype.startswith("q6_k"):
            return "CK_DT_Q6_K"
        if dtype.startswith("q4_0"):
            return "CK_DT_Q4_0"
        if dtype.startswith("q4_1"):
            return "CK_DT_Q4_1"
        if dtype.startswith("q5_0"):
            return "CK_DT_Q5_0"
        if dtype.startswith("q5_1"):
            return "CK_DT_Q5_1"
        if dtype.startswith("q8_0"):
            return "CK_DT_Q8_0"
        if dtype.startswith("q8_k"):
            return "CK_DT_Q8_K"
        return "CK_DT_FP32"

    wq_dtype = layer_weight_dtype("wq")
    wk_dtype = layer_weight_dtype("wk")
    wv_dtype = layer_weight_dtype("wv")
    wo_dtype = layer_weight_dtype("wo")
    w1_dtype = layer_weight_dtype("w1")
    w2_dtype = layer_weight_dtype("w2")

    weight_dtypes = [wq_dtype, wk_dtype, wv_dtype, wo_dtype, w1_dtype, w2_dtype]
    has_quant = any(d.startswith(("q4_k", "q6_k", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_k")) for d in weight_dtypes if d)
    all_q4_k = all(d.startswith("q4_k") for d in weight_dtypes if d)
    use_fast_q4 = use_q4_k and all_q4_k
    use_quant_path = has_quant and not use_fast_q4

    aligned_embed = int(config.get("aligned_embed") or 0)
    aligned_head = int(config.get("aligned_head") or 0)
    aligned_intermediate = int(config.get("aligned_intermediate") or 0)
    aligned_context = int(config.get("aligned_context") or 0)

    def aligned_expr(value: int, fallback: str) -> str:
        return str(value) if value > 0 else fallback

    layer_names = _layer_buffer_names(section)

    required_prefill = [
        "ln1_gamma", "ln1_out", "wq", "wk", "wv",
        "q", "k", "v", "attn_out", "wo", "proj_tmp",
        "residual1", "ln2_gamma", "ln2_out",
        "w1", "fc1_out", "swiglu_out", "w2",
        "mlp_out", "output",
    ]
    if config["num_heads"] > 1:
        required_prefill.append("proj_scratch")

    required_decode = [
        "ln1_gamma", "ln1_out", "wq", "wk", "wv",
        "k", "v", "wo", "proj_tmp", "residual1",
        "ln2_gamma", "ln2_out", "w1", "fc1_out",
        "swiglu_out", "w2", "mlp_out", "output",
    ]

    required = required_prefill if mode == "prefill" else required_decode
    missing = [field for field in required if not _layer_has_field(layer_names, field)]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"v4 codegen needs unfused buffers ({missing_list}). "
            "Re-run build_ir_v4.py with --fusion=off."
        )

    has_scores = _layer_has_field(layer_names, "scores")
    has_proj_scratch = _layer_has_field(layer_names, "proj_scratch")
    has_rope = any(buf.name in {"rope_cos_cache", "rope_sin_cache"} for buf in section.globals)

    lines = []
    def add(s=""):
        lines.append(s)

    add("/**")
    add(f" * @file {os.path.basename(output_path)}")
    add(f" * @brief AUTO-GENERATED: {layout.name} Implementation (IR v4)")
    add(f" *")
    add(f" * Generated: {datetime.utcnow().isoformat()} UTC")
    add(f" * Total Memory: {layout.total_bytes / 1e9:.2f} GB")
    add(f" *")
    add(f" * DO NOT EDIT - Regenerate with build_ir_v4.py")
    add(f" */")
    add()
    add("#define _GNU_SOURCE  /* For MAP_ANONYMOUS, MAP_HUGETLB */")
    add()
    add(f'#include "{header_name}"')
    add()
    add('#include "ckernel_engine.h"')
    add('#include "ckernel_orchestration.h"')
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
    add(f'#error "{layout.name}: v4 codegen currently supports fp32 only. Use --dtype=fp32."')
    add("#endif")
    add()

    add("/* Forward declarations */")
    add(f"static void {safe_name_lower}_init_canaries({safe_name}Model *model);")
    if mode == "decode":
        add(f"static void {safe_name_lower}_decode_token({safe_name}Model *model, const int *token, int token_index);")
    add()

    # Debug helpers (always emitted, but calls are conditional on emit_debug)
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
        add("    fprintf(stderr, \"[DEBUG] %-30s size=%6d  nan=%d inf=%d zero=%d  range=[%.3e, %.3e] mean=%.3e\\n\",")
        add("            name, size, nan_count, inf_count, zero_count, min_val, max_val, mean);")
        add("    if (nan_count > 0 || inf_count > 0) {")
        add("        fprintf(stderr, \"[DEBUG] *** WARNING: %s has %d NaN and %d Inf values! ***\\n\", name, nan_count, inf_count);")
        add("    }")
        add("}")
        add()
        add("static void debug_check_q4k_weights(const char *name, const void *w, int M, int K) {")
        add("    /* Check Q4_K block scales for zeros and NaN */")
        add("    const uint8_t *bytes = (const uint8_t *)w;")
        add("    int blocks_per_row = K / 256;")
        add("    int zero_d = 0, zero_dmin = 0, nan_d = 0, nan_dmin = 0;")
        add("    for (int row = 0; row < M && row < 16; ++row) {  /* Sample first 16 rows */")
        add("        for (int b = 0; b < blocks_per_row; ++b) {")
        add("            /* block_q4_K: first 2 bytes are d (fp16), next 2 are dmin (fp16) */")
        add("            int offset = (row * blocks_per_row + b) * 144;  /* Q4_K block size = 144 bytes */")
        add("            uint16_t d_bits = *(uint16_t *)&bytes[offset];")
        add("            uint16_t dmin_bits = *(uint16_t *)&bytes[offset + 2];")
        add("            if (d_bits == 0) zero_d++;")
        add("            if (dmin_bits == 0) zero_dmin++;")
        add("            /* FP16 NaN: exponent=31 (0x7C00), mantissa!=0 */")
        add("            if ((d_bits & 0x7C00) == 0x7C00 && (d_bits & 0x03FF) != 0) nan_d++;")
        add("            if ((dmin_bits & 0x7C00) == 0x7C00 && (dmin_bits & 0x03FF) != 0) nan_dmin++;")
        add("        }")
        add("    }")
        add("    fprintf(stderr, \"[DEBUG] %-30s M=%d K=%d zero_d=%d zero_dmin=%d nan_d=%d nan_dmin=%d\\n\",")
        add("            name, M, K, zero_d, zero_dmin, nan_d, nan_dmin);")
        add("    if (nan_d > 0 || nan_dmin > 0) {")
        add("        fprintf(stderr, \"[DEBUG] *** WARNING: Q4_K weights have NaN scales! ***\\n\");")
        add("    }")
        add("}")
        add()

    # Parity helpers (save buffers to files for comparison with PyTorch)
    if emit_parity:
        add("/* ============================================================================")
        add(" * PARITY HELPERS (save buffers for PyTorch comparison)")
        add(" * ============================================================================ */")
        add()
        add("static const char *g_parity_dir = NULL;")
        add("static int g_parity_token_idx = 0;")
        add()
        add("void parity_set_output_dir(const char *dir) {")
        add("    g_parity_dir = dir;")
        add("}")
        add()
        add("void parity_set_token_index(int idx) {")
        add("    g_parity_token_idx = idx;")
        add("}")
        add()
        add("static void parity_save_buffer(const char *name, const float *buf, int size) {")
        add("    if (!g_parity_dir) return;")
        add("    char path[512];")
        add("    snprintf(path, sizeof(path), \"%s/%s_tok%d.bin\", g_parity_dir, name, g_parity_token_idx);")
        add("    FILE *f = fopen(path, \"wb\");")
        add("    if (!f) { fprintf(stderr, \"[PARITY] Failed to open %s\\n\", path); return; }")
        add("    fwrite(buf, sizeof(float), size, f);")
        add("    fclose(f);")
        add("    fprintf(stderr, \"[PARITY] Saved %s (%d floats)\\n\", path, size);")
        add("}")
        add()
        add("static void parity_save_q4k_dequant(const char *name, const void *q4k, int M, int K) {")
        add("    if (!g_parity_dir) return;")
        add("    /* Dequantize Q4_K weights to FP32 for comparison */")
        add("    float *dequant = (float *)malloc((size_t)M * K * sizeof(float));")
        add("    if (!dequant) return;")
        add("    for (int row = 0; row < M; ++row) {")
        add("        const uint8_t *row_ptr = (const uint8_t *)q4k + row * (K / 256) * 144;")
        add("        dequant_q4_k_row(row_ptr, dequant + row * K, (size_t)K);")
        add("    }")
        add("    char path[512];")
        add("    snprintf(path, sizeof(path), \"%s/%s_dequant.bin\", g_parity_dir, name);")
        add("    FILE *f = fopen(path, \"wb\");")
        add("    if (f) {")
        add("        fwrite(dequant, sizeof(float), (size_t)M * K, f);")
        add("        fclose(f);")
        add("        fprintf(stderr, \"[PARITY] Saved %s (%d x %d floats)\\n\", path, M, K);")
        add("    }")
        add("    free(dequant);")
        add("}")
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
    add("    /* Try hugepages first */")
    add("    model->base = mmap(NULL, total,")
    add("                       PROT_READ | PROT_WRITE,")
    add("                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,")
    add("                       -1, 0);")
    add("    if (model->base == MAP_FAILED) {")
    add("        /* Fall back to regular pages */")
    add("        model->base = mmap(NULL, total,")
    add("                           PROT_READ | PROT_WRITE,")
    add("                           MAP_PRIVATE | MAP_ANONYMOUS,")
    add("                           -1, 0);")
    add("    }")
    add("    if (model->base == MAP_FAILED) {")
    add("        perror(\"mmap failed\");")
    add("        return -1;")
    add("    }")
    add("#else")
    add("    model->base = aligned_alloc(64, total);")
    add("    if (!model->base) {")
    add("        perror(\"aligned_alloc failed\");")
    add("        return -1;")
    add("    }")
    add("#endif")
    add()
    add("    model->total_bytes = total;")
    add()
    add("    /* Initialize magic header */")
    add("    MagicHeader *header = (MagicHeader *)model->base;")
    add(f"    header->magic = {safe_name}_MAGIC;")
    add("    header->version = 4;")
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
    add("    /* Initialize canaries */")
    add(f"    {safe_name_lower}_init_canaries(model);")
    add()
    add("    return 0;")
    add("}")
    add()

    # Free function
    add(f"void {safe_name_lower}_model_free({safe_name}Model *model) {{")
    add("    if (!model || !model->base) return;")
    add()
    add("#ifdef __linux__")
    add("    munmap(model->base, model->total_bytes);")
    add("#else")
    add("    free(model->base);")
    add("#endif")
    add()
    add("    model->base = NULL;")
    add("    model->total_bytes = 0;")
    add("}")
    add()

    # Canary init
    add("/* ============================================================================")
    add(" * CANARY SYSTEM")
    add(" * ============================================================================ */")
    add()
    add(f"static void {safe_name_lower}_init_canaries({safe_name}Model *model) {{")
    add("    uint32_t *ptr;")
    add()
    add(f"    /* Write 0x{v3.CANARY_VALUE:08X} to each canary slot */")
    add(f"    for (int i = 0; i < {safe_name}_CANARY_COUNT; i++) {{")
    add(f"        ptr = (uint32_t*)((char*)model->base + {safe_name}_CANARIES[i].offset);")
    add(f"        for (int j = 0; j < {v3.CANARY_SIZE // 4}; j++) {{")
    add(f"            ptr[j] = {safe_name}_CANARY_VALUE;")
    add("        }")
    add("    }")
    add("}")
    add()

    # Canary verify
    add(f"int {safe_name_lower}_verify_canaries({safe_name}Model *model) {{")
    add("    int errors = 0;")
    add("    uint32_t *ptr;")
    add()
    add(f"    for (int i = 0; i < {safe_name}_CANARY_COUNT; i++) {{")
    add(f"        ptr = (uint32_t*)((char*)model->base + {safe_name}_CANARIES[i].offset);")
    add(f"        for (int j = 0; j < {v3.CANARY_SIZE // 4}; j++) {{")
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

    # Alignment helper
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
        add("    /* No RoPE globals defined */")
        add("    (void)model;")
    add("}")
    add()

    if mode == "prefill":
        add("/* ============================================================================")
        add(" * LAYER FORWARD (PREFILL)")
        add(" * ============================================================================ */")
        add()
        add(f"static void {safe_name_lower}_layer_forward_prefill(")
        add(f"    {safe_name}Model *model,")
        add("    int layer_id,")
        add("    int num_tokens,")
        add("    int aligned_embed_dim,")
        add("    int aligned_head_dim,")
        add("    int aligned_intermediate_dim,")
        add("    int aligned_context_window")
        add(") {")
        add(f"    const {safe_name}LayerOffsets *L = &{safe_name}_LAYERS[layer_id];")
        if has_quant:
            add("    CKLayerForwardParamsQ4K p = {0};")
        else:
            add("    CKLayerForwardParams p = {0};")
        add()
        add("    p.tokens = num_tokens;")
        add(f"    p.embed_dim = {safe_name}_EMBED_DIM;")
        add("    p.aligned_embed_dim = aligned_embed_dim;")
        add(f"    p.num_heads = {safe_name}_NUM_HEADS;")
        add(f"    p.num_kv_heads = {safe_name}_NUM_KV_HEADS;")
        add(f"    p.head_dim = {safe_name}_HEAD_DIM;")
        add("    p.aligned_head_dim = aligned_head_dim;")
        add("    p.aligned_context_window = aligned_context_window;")
        add(f"    p.intermediate_dim = {safe_name}_INTERMEDIATE;")
        add("    p.aligned_intermediate_dim = aligned_intermediate_dim;")
        add(f"    p.eps = {config.get('rms_norm_eps', 1e-6)}f;")
        add("    p.rope_pos_offset = 0;")
        add()
        add("    if (layer_id == 0) {")
        add(f"        p.input = {safe_name}_PTR(model, {safe_name}_HEADER.embedded_input);")
        add("    } else {")
        add(f"        p.input = {safe_name}_PTR(model, {safe_name}_LAYERS[layer_id - 1].output);")
        add("    }")
        add()
        add(f"    p.ln1_gamma = {safe_name}_PTR(model, L->ln1_gamma);")
        add(f"    p.ln2_gamma = {safe_name}_PTR(model, L->ln2_gamma);")
        add()
        if has_rope:
            add(f"    p.rope_cos = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_cos_cache);")
            add(f"    p.rope_sin = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_sin_cache);")
        else:
            add("    p.rope_cos = NULL;")
            add("    p.rope_sin = NULL;")
        add()
        if has_quant:
            add(f"    p.wq = (const void *){safe_name}_PTR(model, L->wq);")
            add("    p.bq = NULL;")
            add(f"    p.wk = (const void *){safe_name}_PTR(model, L->wk);")
            add("    p.bk = NULL;")
            add(f"    p.wv = (const void *){safe_name}_PTR(model, L->wv);")
            add("    p.bv = NULL;")
            add(f"    p.wo = (const void *){safe_name}_PTR(model, L->wo);")
            add("    p.bo = NULL;")
            add(f"    p.w1 = (const void *){safe_name}_PTR(model, L->w1);")
            add("    p.b1 = NULL;")
            add(f"    p.w2 = (const void *){safe_name}_PTR(model, L->w2);")
            add("    p.b2 = NULL;")
        else:
            add(f"    p.wq = {safe_name}_PTR(model, L->wq);")
            add("    p.bq = NULL;")
            add(f"    p.wk = {safe_name}_PTR(model, L->wk);")
            add("    p.bk = NULL;")
            add(f"    p.wv = {safe_name}_PTR(model, L->wv);")
            add("    p.bv = NULL;")
            add(f"    p.wo = {safe_name}_PTR(model, L->wo);")
            add("    p.bo = NULL;")
            add(f"    p.w1 = {safe_name}_PTR(model, L->w1);")
            add("    p.b1 = NULL;")
            add(f"    p.w2 = {safe_name}_PTR(model, L->w2);")
            add("    p.b2 = NULL;")
        add()
        if has_quant:
            if mixed_quant:
                # Per-layer dtypes for mixed quant models
                add(f"    p.wq_dtype = {safe_name}_LAYER_WQ_DTYPE[layer_id];")
                add(f"    p.wk_dtype = {safe_name}_LAYER_WK_DTYPE[layer_id];")
                add(f"    p.wv_dtype = {safe_name}_LAYER_WV_DTYPE[layer_id];")
                add(f"    p.wo_dtype = {safe_name}_LAYER_WO_DTYPE[layer_id];")
                add(f"    p.w1_dtype = {safe_name}_LAYER_W1_DTYPE[layer_id];")
                add(f"    p.w2_dtype = {safe_name}_LAYER_W2_DTYPE[layer_id];")
            else:
                # All layers use same dtype
                add(f"    p.wq_dtype = {dtype_const(wq_dtype)};")
                add(f"    p.wk_dtype = {dtype_const(wk_dtype)};")
                add(f"    p.wv_dtype = {dtype_const(wv_dtype)};")
                add(f"    p.wo_dtype = {dtype_const(wo_dtype)};")
                add(f"    p.w1_dtype = {dtype_const(w1_dtype)};")
                add(f"    p.w2_dtype = {dtype_const(w2_dtype)};")
            add()
        add(f"    p.ln1_out = {safe_name}_PTR(model, L->ln1_out);")
        add("    p.ln1_rstd = NULL;")
        add(f"    p.q = {safe_name}_PTR(model, L->q);")
        add(f"    p.k = {safe_name}_PTR(model, L->k);")
        add(f"    p.v = {safe_name}_PTR(model, L->v);")
        if has_scores:
            add(f"    p.scores = {safe_name}_PTR(model, L->scores);")
        else:
            add("    p.scores = NULL;")
        add(f"    p.attn_out = {safe_name}_PTR(model, L->attn_out);")
        add(f"    p.proj_tmp = {safe_name}_PTR(model, L->proj_tmp);")
        if has_proj_scratch:
            add(f"    p.proj_scratch = {safe_name}_PTR(model, L->proj_scratch);")
        else:
            add("    p.proj_scratch = NULL;")
        add(f"    p.residual1 = {safe_name}_PTR(model, L->residual1);")
        add(f"    p.ln2_out = {safe_name}_PTR(model, L->ln2_out);")
        add("    p.ln2_rstd = NULL;")
        add(f"    p.fc1_out = {safe_name}_PTR(model, L->fc1_out);")
        add(f"    p.swiglu_out = {safe_name}_PTR(model, L->swiglu_out);")
        add(f"    p.mlp_out = {safe_name}_PTR(model, L->mlp_out);")
        add(f"    p.output = {safe_name}_PTR(model, L->output);")
        add()
        if has_quant:
            if use_fast_q4:
                add("    ck_layer_forward_rmsnorm_swiglu_q4_k(&p);")
            else:
                add("    ck_layer_forward_rmsnorm_swiglu_quant(&p);")
        else:
            add("    ck_layer_forward_rmsnorm_swiglu(&p);")
        add("}")
        add()

        add("/* ============================================================================")
        add(" * FORWARD PASS (PREFILL)")
        add(" * ============================================================================ */")
        add()
        add(f"void {safe_name_lower}_forward(")
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
        if embed_quant_type:
            # Quantized embedding - use appropriate dequant kernel
            embed_func = f"embedding_forward_{embed_quant_type}"
            add(f"    const void *embed_weight = (const void *){safe_name}_PTR(model, {safe_name}_HEADER.token_emb);")
            add(f"    {embed_func}((const int32_t *)tokens,")
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
        add(f"    for (int layer_id = 0; layer_id < {safe_name}_NUM_LAYERS; ++layer_id) {{")
        add(f"        {safe_name_lower}_layer_forward_prefill(")
        add("            model,")
        add("            layer_id,")
        add("            num_tokens,")
        add("            aligned_embed_dim,")
        add("            aligned_head_dim,")
        add("            aligned_intermediate_dim,")
        add("            aligned_context_window);")
        add()
        add("        kv_cache_repack_head_major_inplace(")
        add(f"            {safe_name}_PTR(model, {safe_name}_LAYERS[layer_id].k),")
        add(f"            {safe_name}_NUM_KV_HEADS,")
        add("            num_tokens,")
        add("            aligned_context_window,")
        add("            aligned_head_dim);")
        add("        kv_cache_repack_head_major_inplace(")
        add(f"            {safe_name}_PTR(model, {safe_name}_LAYERS[layer_id].v),")
        add(f"            {safe_name}_NUM_KV_HEADS,")
        add("            num_tokens,")
        add("            aligned_context_window,")
        add("            aligned_head_dim);")
        add("    }")
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
        if lm_head_quant_type:
            # Quantized lm_head - use appropriate gemm kernel
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
                # Other quant types (Q8_0, Q6_K, Q5_0, etc.) - process row by row
                add("    for (int t = 0; t < num_tokens; ++t) {")
                add("        const float *row = final_out + (size_t)t * (size_t)aligned_embed_dim;")
                add(f"        float *logits_row = logits + (size_t)t * (size_t){safe_name}_VOCAB_SIZE;")
                add("        ck_gemm_nt_quant(row,")
                add("                         lm_head,")
                add("                         NULL,")
                add("                         logits_row,")
                add("                         1,")
                add(f"                         {safe_name}_VOCAB_SIZE,")
                add("                         aligned_embed_dim,")
                add(f"                         {dtype_const(lm_head_dtype)});")
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

    else:
        add("/* ============================================================================")
        add(" * LAYER FORWARD (DECODE, SINGLE TOKEN)")
        add(" * ============================================================================ */")
        add()
        if has_quant:
            add(f"static void {safe_name_lower}_layer_forward_decode_token(")
            add(f"    {safe_name}Model *model,")
            add("    int layer_id,")
            add("    int token_index,")
            add("    int aligned_embed_dim,")
            add("    int aligned_head_dim,")
            add("    int aligned_intermediate_dim,")
            add("    int aligned_context_window")
            add(") {")
            add(f"    const {safe_name}LayerOffsets *L = &{safe_name}_LAYERS[layer_id];")
            add("    CKLayerForwardParamsQ4K p = {0};")
            add()
            add("    p.tokens = token_index + 1;")
            add(f"    p.embed_dim = {safe_name}_EMBED_DIM;")
            add("    p.aligned_embed_dim = aligned_embed_dim;")
            add(f"    p.num_heads = {safe_name}_NUM_HEADS;")
            add(f"    p.num_kv_heads = {safe_name}_NUM_KV_HEADS;")
            add(f"    p.head_dim = {safe_name}_HEAD_DIM;")
            add("    p.aligned_head_dim = aligned_head_dim;")
            add("    p.aligned_context_window = aligned_context_window;")
            add(f"    p.intermediate_dim = {safe_name}_INTERMEDIATE;")
            add("    p.aligned_intermediate_dim = aligned_intermediate_dim;")
            add(f"    p.eps = {config.get('rms_norm_eps', 1e-6)}f;")
            add("    p.rope_pos_offset = token_index;")
            add()
            add("    if (layer_id == 0) {")
            add(f"        p.input = {safe_name}_PTR(model, {safe_name}_HEADER.embedded_input);")
            add("    } else {")
            add(f"        p.input = {safe_name}_PTR(model, {safe_name}_LAYERS[layer_id - 1].output);")
            add("    }")
            add()
            add(f"    p.ln1_gamma = {safe_name}_PTR(model, L->ln1_gamma);")
            add(f"    p.ln2_gamma = {safe_name}_PTR(model, L->ln2_gamma);")
            add()
            if has_rope:
                add(f"    p.rope_cos = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_cos_cache);")
                add(f"    p.rope_sin = {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_sin_cache);")
            else:
                add("    p.rope_cos = NULL;")
                add("    p.rope_sin = NULL;")
            add()
            add(f"    p.wq = (const void *){safe_name}_PTR(model, L->wq);")
            add("    p.bq = NULL;")
            add(f"    p.wk = (const void *){safe_name}_PTR(model, L->wk);")
            add("    p.bk = NULL;")
            add(f"    p.wv = (const void *){safe_name}_PTR(model, L->wv);")
            add("    p.bv = NULL;")
            add(f"    p.wo = (const void *){safe_name}_PTR(model, L->wo);")
            add("    p.bo = NULL;")
            add(f"    p.w1 = (const void *){safe_name}_PTR(model, L->w1);")
            add("    p.b1 = NULL;")
            add(f"    p.w2 = (const void *){safe_name}_PTR(model, L->w2);")
            add("    p.b2 = NULL;")
            add()
            if mixed_quant:
                # Per-layer dtypes for mixed quant models
                add(f"    p.wq_dtype = {safe_name}_LAYER_WQ_DTYPE[layer_id];")
                add(f"    p.wk_dtype = {safe_name}_LAYER_WK_DTYPE[layer_id];")
                add(f"    p.wv_dtype = {safe_name}_LAYER_WV_DTYPE[layer_id];")
                add(f"    p.wo_dtype = {safe_name}_LAYER_WO_DTYPE[layer_id];")
                add(f"    p.w1_dtype = {safe_name}_LAYER_W1_DTYPE[layer_id];")
                add(f"    p.w2_dtype = {safe_name}_LAYER_W2_DTYPE[layer_id];")
            else:
                add(f"    p.wq_dtype = {dtype_const(wq_dtype)};")
                add(f"    p.wk_dtype = {dtype_const(wk_dtype)};")
                add(f"    p.wv_dtype = {dtype_const(wv_dtype)};")
                add(f"    p.wo_dtype = {dtype_const(wo_dtype)};")
                add(f"    p.w1_dtype = {dtype_const(w1_dtype)};")
                add(f"    p.w2_dtype = {dtype_const(w2_dtype)};")
            add()
            add(f"    p.ln1_out = {safe_name}_PTR(model, L->ln1_out);")
            add("    p.ln1_rstd = NULL;")
            add("    p.q = NULL;")
            add(f"    p.k = {safe_name}_PTR(model, L->k);")
            add(f"    p.v = {safe_name}_PTR(model, L->v);")
            add("    p.scores = NULL;")
            add("    p.attn_out = NULL;")
            add(f"    p.proj_tmp = {safe_name}_PTR(model, L->proj_tmp);")
            if has_proj_scratch:
                add(f"    p.proj_scratch = {safe_name}_PTR(model, L->proj_scratch);")
            else:
                add("    p.proj_scratch = NULL;")
            add(f"    p.residual1 = {safe_name}_PTR(model, L->residual1);")
            add(f"    p.ln2_out = {safe_name}_PTR(model, L->ln2_out);")
            add("    p.ln2_rstd = NULL;")
            add(f"    p.fc1_out = {safe_name}_PTR(model, L->fc1_out);")
            add(f"    p.swiglu_out = {safe_name}_PTR(model, L->swiglu_out);")
            add(f"    p.mlp_out = {safe_name}_PTR(model, L->mlp_out);")
            add(f"    p.output = {safe_name}_PTR(model, L->output);")
            add()
            # Add granular debug for Q4_K decode path
            if emit_debug:
                add("    /* Granular debug before layer forward */")
                add("    if (layer_id == 0) {")
                add("        debug_check_buffer(\"layer0_input\", p.input, aligned_embed_dim);")
                add("        if (p.wq_dtype == CK_DT_Q4_K) {")
                add("            debug_check_q4k_weights(\"layer0_wq\", p.wq, p.num_heads * aligned_head_dim, aligned_embed_dim);")
                add("        }")
                add("        if (p.wk_dtype == CK_DT_Q4_K) {")
                add("            debug_check_q4k_weights(\"layer0_wk\", p.wk, p.num_kv_heads * aligned_head_dim, aligned_embed_dim);")
                add("        }")
                add("        if (p.wv_dtype == CK_DT_Q4_K) {")
                add("            debug_check_q4k_weights(\"layer0_wv\", p.wv, p.num_kv_heads * aligned_head_dim, aligned_embed_dim);")
                add("        }")
                add("    }")
            if use_fast_q4:
                add("    ck_layer_forward_rmsnorm_swiglu_decode_q4_k(&p, token_index, aligned_context_window);")
            else:
                add("    ck_layer_forward_rmsnorm_swiglu_decode_quant(&p, token_index, aligned_context_window);")
            if emit_debug:
                add("    /* Granular debug after layer forward */")
                add("    if (layer_id == 0) {")
                add("        debug_check_buffer(\"layer0_output\", p.output, aligned_embed_dim);")
                add("        debug_check_buffer(\"layer0_ln1_out\", p.ln1_out, aligned_embed_dim);")
                add("        debug_check_buffer(\"layer0_proj_tmp\", p.proj_tmp, aligned_embed_dim);")
                add("    }")
            add("}")
            add()
        else:
            add(f"static void {safe_name_lower}_layer_forward_decode_token(")
            add(f"    {safe_name}Model *model,")
            add("    int layer_id,")
            add("    int token_index,")
            add("    int aligned_embed_dim,")
            add("    int aligned_head_dim,")
            add("    int aligned_intermediate_dim,")
            add("    int aligned_context_window")
            add(") {")
            add(f"    const {safe_name}LayerOffsets *L = &{safe_name}_LAYERS[layer_id];")
            add()
            add(f"    const int H = {safe_name}_NUM_HEADS;")
            add(f"    const int H_kv = {safe_name}_NUM_KV_HEADS;")
            add(f"    const int head_dim = {safe_name}_HEAD_DIM;")
            add()
            add("    float *input_row;")
            add("    if (layer_id == 0) {")
            add(f"        input_row = {safe_name}_PTR(model, {safe_name}_HEADER.embedded_input);")
            add("    } else {")
            add(f"        input_row = {safe_name}_PTR(model, {safe_name}_LAYERS[layer_id - 1].output);")
            add("    }")
            add()
            add(f"    float *ln1_out = {safe_name}_PTR(model, L->ln1_out);")
            add(f"    float *ln2_out = {safe_name}_PTR(model, L->ln2_out);")
            add(f"    float *proj_tmp = {safe_name}_PTR(model, L->proj_tmp);")
            add(f"    float *residual1 = {safe_name}_PTR(model, L->residual1);")
            add(f"    float *fc1_out = {safe_name}_PTR(model, L->fc1_out);")
            add(f"    float *swiglu_out = {safe_name}_PTR(model, L->swiglu_out);")
            add(f"    float *mlp_out = {safe_name}_PTR(model, L->mlp_out);")
            add(f"    float *output = {safe_name}_PTR(model, L->output);")
            add()
            add(f"    float *k_cache = {safe_name}_PTR(model, L->k);")
            add(f"    float *v_cache = {safe_name}_PTR(model, L->v);")
            add()
            add("    float q_token[H * aligned_head_dim];")
            add("    float k_token[H_kv * aligned_head_dim];")
            add("    float v_token[H_kv * aligned_head_dim];")
            add("    float attn_token[H * aligned_head_dim];")
            add()
            add("    rmsnorm_forward(input_row,")
            add(f"                   {safe_name}_PTR(model, L->ln1_gamma),")
            add("                   ln1_out,")
            add("                   NULL,")
            add("                   1,")
            add(f"                   {safe_name}_EMBED_DIM,")
            add("                   aligned_embed_dim,")
            add(f"                   {config.get('rms_norm_eps', 1e-6)}f);")
            add()
            add("    ck_qkv_project_head_major_token(ln1_out,")
            add(f"                                    {safe_name}_PTR(model, L->wq),")
            add("                                    NULL,")
            add(f"                                    {safe_name}_PTR(model, L->wk),")
            add("                                    NULL,")
            add(f"                                    {safe_name}_PTR(model, L->wv),")
            add("                                    NULL,")
            add("                                    q_token,")
            add("                                    k_token,")
            add("                                    v_token,")
            add("                                    aligned_embed_dim,")
            add("                                    H,")
            add("                                    H_kv,")
            add("                                    aligned_head_dim);")
            add()
            if has_rope:
                add("    rope_forward_qk(q_token,")
                add("                    k_token,")
                add(f"                    {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_cos_cache),")
                add(f"                    {safe_name}_PTR(model, {safe_name}_GLOBALS.rope_sin_cache),")
                add("                    H,")
                add("                    H_kv,")
                add("                    1,")
                add("                    head_dim,")
                add("                    aligned_head_dim,")
                add("                    token_index);")
                add()
            add("    kv_cache_write_head_major(k_token,")
            add("                             v_token,")
            add("                             k_cache,")
            add("                             v_cache,")
            add("                             H_kv,")
            add("                             token_index,")
            add("                             aligned_context_window,")
            add("                             head_dim,")
            add("                             aligned_head_dim);")
            add()
            add("    attention_forward_decode_head_major_gqa_flash(q_token,")
            add("                                                  k_cache,")
            add("                                                  v_cache,")
            add("                                                  attn_token,")
            add("                                                  H,")
            add("                                                  H_kv,")
            add("                                                  token_index + 1,")
            add("                                                  aligned_context_window,")
            add("                                                  head_dim,")
            add("                                                  aligned_head_dim);")
            add()
            add("    ck_attention_project_head_major_decode_token(attn_token,")
            add(f"                                                 {safe_name}_PTR(model, L->wo),")
            add("                                                 NULL,")
            add("                                                 proj_tmp,")
            add(f"                                                 {safe_name}_EMBED_DIM,")
            add("                                                 aligned_embed_dim,")
            add("                                                 H,")
            add("                                                 aligned_head_dim);")
            add()
            add("    ck_residual_add_token_major(input_row,")
            add("                                proj_tmp,")
            add("                                residual1,")
            add("                                1,")
            add("                                aligned_embed_dim);")
            add()
            add("    rmsnorm_forward(residual1,")
            add(f"                   {safe_name}_PTR(model, L->ln2_gamma),")
            add("                   ln2_out,")
            add("                   NULL,")
            add("                   1,")
            add(f"                   {safe_name}_EMBED_DIM,")
            add("                   aligned_embed_dim,")
            add(f"                   {config.get('rms_norm_eps', 1e-6)}f);")
            add()
            add("    ck_mlp_swiglu_forward(ln2_out,")
            add(f"                          {safe_name}_PTR(model, L->w1),")
            add("                          NULL,")
            add(f"                          {safe_name}_PTR(model, L->w2),")
            add("                          NULL,")
            add("                          fc1_out,")
            add("                          swiglu_out,")
            add("                          mlp_out,")
            add("                          1,")
            add("                          aligned_embed_dim,")
            add("                          aligned_intermediate_dim);")
            add()
            add("    ck_residual_add_token_major(residual1,")
            add("                                mlp_out,")
            add("                                output,")
            add("                                1,")
            add("                                aligned_embed_dim);")
            add("}")
            add()

        add("/* ============================================================================")
        add(" * DECODE HELPERS")
        add(" * ============================================================================ */")
        add()
        add(f"static void {safe_name_lower}_decode_token(")
        add(f"    {safe_name}Model *model,")
        add("    const int *token,")
        add("    int token_index")
        add(") {")
        add("    if (!model || !token) {")
        add("        return;")
        add("    }")
        add()
        add(f"    const int elem_bytes = {safe_name}_DTYPE_BYTES;")
        add(f"    const int aligned_embed_dim = {aligned_expr(aligned_embed, f'{safe_name_lower}_align_elems({safe_name}_EMBED_DIM, elem_bytes, 64)')};")
        add(f"    const int aligned_head_dim = {aligned_expr(aligned_head, f'{safe_name_lower}_align_elems({safe_name}_HEAD_DIM, elem_bytes, 64)')};")
        add(f"    const int aligned_intermediate_dim = {aligned_expr(aligned_intermediate, f'{safe_name_lower}_align_elems({safe_name}_INTERMEDIATE, elem_bytes, 64)')};")
        add(f"    const int aligned_context_window = {aligned_expr(aligned_context, f'{safe_name_lower}_align_elems({safe_name}_MAX_SEQ_LEN, elem_bytes, 64)')};")
        add("    if (token_index < 0 || token_index >= aligned_context_window) {")
        add("        return;")
        add("    }")
        add()
        add(f"    float *embed_out = {safe_name}_PTR(model, {safe_name}_HEADER.embedded_input);")
        if embed_quant_type:
            # Quantized embedding - use appropriate dequant kernel
            embed_func = f"embedding_forward_{embed_quant_type}"
            add(f"    const void *embed_weight = (const void *){safe_name}_PTR(model, {safe_name}_HEADER.token_emb);")
            add(f"    {embed_func}((const int32_t *)token,")
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
            add()
            add("    /* DEBUG: Check embedding output */")
            add(f"    debug_check_buffer(\"embed_out\", embed_out, aligned_embed_dim);")
        if emit_parity:
            add()
            add("    /* PARITY: Save embedding output */")
            add("    parity_save_buffer(\"embed_out\", embed_out, aligned_embed_dim);")
        add()
        add(f"    for (int layer_id = 0; layer_id < {safe_name}_NUM_LAYERS; ++layer_id) {{")
        add(f"        {safe_name_lower}_layer_forward_decode_token(")
        add("            model,")
        add("            layer_id,")
        add("            token_index,")
        add("            aligned_embed_dim,")
        add("            aligned_head_dim,")
        add("            aligned_intermediate_dim,")
        add("            aligned_context_window);")
        if emit_debug:
            add()
            add("        /* DEBUG: Check layer output */")
            add("        char layer_name[64];")
            add("        snprintf(layer_name, sizeof(layer_name), \"layer[%d].output\", layer_id);")
            add(f"        float *layer_out = {safe_name}_PTR(model, {safe_name}_LAYERS[layer_id].output);")
            add("        debug_check_buffer(layer_name, layer_out, aligned_embed_dim);")
        if emit_parity:
            add()
            add("        /* PARITY: Save layer output */")
            add("        char parity_name[64];")
            add("        snprintf(parity_name, sizeof(parity_name), \"layer_%d_output\", layer_id);")
            add(f"        float *parity_out = {safe_name}_PTR(model, {safe_name}_LAYERS[layer_id].output);")
            add("        parity_save_buffer(parity_name, parity_out, aligned_embed_dim);")
        add("    }")
        add()
        add(f"    float *last_hidden = {safe_name}_PTR(model, {safe_name}_LAYERS[{safe_name}_NUM_LAYERS - 1].output);")
        add(f"    float *final_ln_weight = {safe_name}_PTR(model, {safe_name}_FOOTER.final_ln_weight);")
        add(f"    float *final_out = {safe_name}_PTR(model, {safe_name}_FOOTER.final_output);")
        add("    rmsnorm_forward(last_hidden,")
        add("                   final_ln_weight,")
        add("                   final_out,")
        add("                   NULL,")
        add("                   1,")
        add(f"                   {safe_name}_EMBED_DIM,")
        add("                   aligned_embed_dim,")
        add(f"                   {config.get('rms_norm_eps', 1e-6)}f);")
        if emit_debug:
            add()
            add("    /* DEBUG: Check final rmsnorm output */")
            add("    debug_check_buffer(\"final_out (after rmsnorm)\", final_out, aligned_embed_dim);")
        if emit_parity:
            add()
            add("    /* PARITY: Save final rmsnorm output */")
            add("    parity_save_buffer(\"final_rmsnorm_out\", final_out, aligned_embed_dim);")
        add()
        add(f"    float *logits = {safe_name}_PTR(model, {safe_name}_FOOTER.logits);")
        if lm_head_quant_type:
            # Quantized lm_head - use appropriate gemm kernel
            add(f"    const void *lm_head = (const void *){safe_name}_PTR(model, {safe_name}_FOOTER.lm_head_weight);")
            if lm_head_use_q4_k:
                if emit_debug:
                    add()
                    add("    /* DEBUG: Check Q4K lm_head weights */")
                    add(f"    debug_check_q4k_weights(\"lm_head (q4k)\", lm_head, {safe_name}_VOCAB_SIZE, aligned_embed_dim);")
                add("    const size_t q8_bytes = ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)aligned_embed_dim);")
                add("    uint8_t q8_buf[q8_bytes];")
                add("    quantize_row_q8_k(final_out, q8_buf, aligned_embed_dim);")
                add("    gemm_nt_q4_k_q8_k(q8_buf,")
                add("                      lm_head,")
                add("                      NULL,")
                add("                      logits,")
                add("                      1,")
                add(f"                      {safe_name}_VOCAB_SIZE,")
                add("                      aligned_embed_dim);")
            else:
                # Other quant types (Q8_0, Q6_K, Q5_0, etc.) - use generic quant gemm
                add(f"    ck_gemm_nt_quant(final_out,")
                add("                      lm_head,")
                add("                      NULL,")
                add("                      logits,")
                add("                      1,")
                add(f"                      {safe_name}_VOCAB_SIZE,")
                add("                      aligned_embed_dim,")
                add(f"                      {dtype_const(lm_head_dtype)});")
        else:
            add(f"    float *lm_head = {safe_name}_PTR(model, {safe_name}_FOOTER.lm_head_weight);")
            add("    gemm_blocked_serial(final_out,")
            add("                        lm_head,")
            add("                        NULL,")
            add("                        logits,")
            add("                        1,")
            add(f"                        {safe_name}_VOCAB_SIZE,")
            add("                        aligned_embed_dim);")
        if emit_debug:
            add()
            add("    /* DEBUG: Check logits output */")
            add(f"    debug_check_buffer(\"logits\", logits, {safe_name}_VOCAB_SIZE);")
        if emit_parity:
            add()
            add("    /* PARITY: Save logits */")
            add(f"    parity_save_buffer(\"logits\", logits, {safe_name}_VOCAB_SIZE);")
        add("}")
        add()

        add("/* ============================================================================")
        add(" * FORWARD PASS (DECODE)")
        add(" * ============================================================================ */")
        add()
        add(f"void {safe_name_lower}_forward(")
        add(f"    {safe_name}Model *model,")
        add("    const int *tokens,")
        add("    int num_tokens")
        add(") {")
        add("    if (!model || !tokens || num_tokens <= 0) {")
        add("        return;")
        add("    }")
        add()
        add("    for (int i = 0; i < num_tokens; ++i) {")
        add(f"        {safe_name_lower}_decode_token(model, tokens + i, i);")
        add("    }")
        add("}")
        add()

        add("/* Explicit decode API for cache-aware generation */")
        add(f"void {safe_name_lower}_decode({safe_name}Model *model, const int *token, int token_index) {{")
        add(f"    {safe_name_lower}_decode_token(model, token, token_index);")
        add("}")
        add()

    if emit_main:
        add("/* ============================================================================")
        add(" * STANDALONE MAIN (STUB)")
        add(" * ============================================================================ */")
        add()
        add("int main(int argc, char **argv) {")
        add("    printf(\"%s: generated runtime (%s)\\n\", argv[0], \"" + mode + "\");")
        add(f"    printf(\"total_bytes=%zu weight_bytes=%zu activation_bytes=%zu\\n\",")
        add(f"           (size_t){safe_name}_TOTAL_BYTES,")
        add(f"           (size_t){safe_name}_WEIGHT_BYTES,")
        add(f"           (size_t){safe_name}_ACTIVATION_BYTES);")
        add("    if (argc > 1 && strcmp(argv[1], \"--alloc\") == 0) {")
        add(f"        {safe_name}Model model = {{0}};")
        add(f"        if ({safe_name_lower}_model_allocate(&model) != 0) {{")
        add("            return 1;")
        add("        }")
        add(f"        {safe_name_lower}_precompute_rope(&model);")
        add(f"        {safe_name_lower}_model_free(&model);")
        add("        return 0;")
        add("    }")
        add("    fprintf(stderr, \"Run with --alloc to allocate buffers.\\n\");")
        add("    fprintf(stderr, \"Load weights and call *_forward() from your host app.\\n\");")
        add("    return 0;")
        add("}")
        add()

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[.c] Written: {output_path}")
