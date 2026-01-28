#!/usr/bin/env python3
"""
=============================================================================
LEGACY FILE - NOT USED BY CURRENT v6.6 PIPELINE
=============================================================================
This file has been superseded by: codegen_v6_6.py

This was a temporary/experimental codegen variant created during development.
The current v6.6 pipeline uses codegen_v6_6.py.

To use the current pipeline: python ck_run_v6_6.py run <model>
=============================================================================

codegen_v6_6.py - Generate unrolled C code from lowered IR.

DATA-DRIVEN APPROACH:
- Kernel signatures come from kernel maps (not hardcoded)
- Parameter bindings come from lowered IR (not hardcoded)
- Memory offsets come from bump_layout (precomputed structs)

The lowered IR operations contain:
  - function: kernel function name
  - weights: dict of weight name -> {bump_offset, dtype, ...}
  - activations: dict of activation name -> {activation_offset, dtype, ...}
  - outputs: dict of output name -> {activation_offset, dtype, ...}
  - params: dict of param name -> value

Generated code is fully unrolled (no loops over layers).
Uses struct-based offsets from bump_layout_v6_6.py generated header.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Path to kernel registry and bindings
SCRIPTS_DIR = Path(__file__).parent
KERNEL_MAPS_DIR = SCRIPTS_DIR.parent / "kernel_maps"
KERNEL_REGISTRY_PATH = KERNEL_MAPS_DIR / "KERNEL_REGISTRY.json"
KERNEL_BINDINGS_PATH = KERNEL_MAPS_DIR / "kernel_bindings.json"


def load_kernel_registry() -> Dict:
    """Load the kernel registry with c_declarations."""
    if KERNEL_REGISTRY_PATH.exists():
        with open(KERNEL_REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return {"kernels": []}


def load_kernel_bindings() -> Dict:
    """Load kernel parameter bindings for data-driven codegen."""
    if KERNEL_BINDINGS_PATH.exists():
        with open(KERNEL_BINDINGS_PATH, 'r') as f:
            data = json.load(f)
            return data.get("bindings", {})
    return {}


def load_kernel_map(kernel_name: str) -> Optional[Dict]:
    """Load individual kernel map JSON for detailed parameter info."""
    path = KERNEL_MAPS_DIR / f"{kernel_name}.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def parse_c_declaration(c_decl: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Parse C declaration to extract function name and parameter list.

    Returns: (function_name, [(param_type, param_name), ...])

    Example:
      "void rope_forward_qk(float *q, float *k, const float *cos_cache, ...);"
      -> ("rope_forward_qk", [("float *", "q"), ("float *", "k"), ...])
    """
    # Remove 'void ' prefix and trailing ';'
    c_decl = c_decl.strip()
    if c_decl.startswith("void "):
        c_decl = c_decl[5:]
    if c_decl.endswith(";"):
        c_decl = c_decl[:-1]

    # Extract function name and params
    match = re.match(r'(\w+)\s*\((.*)\)', c_decl)
    if not match:
        return "", []

    func_name = match.group(1)
    params_str = match.group(2)

    # Parse each parameter
    params = []
    for param in params_str.split(','):
        param = param.strip()
        if not param:
            continue
        # Split type and name - last word is name, rest is type
        parts = param.rsplit(None, 1)
        if len(parts) == 2:
            # Handle pointer in name like "float *q" -> type="float *", name="q"
            ptype, pname = parts
            if pname.startswith('*'):
                ptype = ptype + ' *'
                pname = pname[1:]
            params.append((ptype.strip(), pname.strip()))
        elif len(parts) == 1:
            params.append(("", parts[0]))

    return func_name, params


def extract_unique_kernels(operations: List[Dict]) -> List[str]:
    """Extract unique kernel function names from lowered IR operations."""
    kernels = set()
    for op in operations:
        kernel = op.get("function", op.get("kernel", ""))
        if kernel:
            kernels.add(kernel)
    return sorted(kernels)


def generate_kernel_declarations(kernels: List[str], registry: Dict) -> str:
    """Generate kernel declarations from registry."""
    # Build lookup from kernel name to c_declaration
    decl_map = {}
    for k in registry.get("kernels", []):
        func_name = k.get("impl", {}).get("function", k.get("id", ""))
        c_decl = k.get("impl", {}).get("c_declaration", "")
        if func_name and c_decl:
            decl_map[func_name] = c_decl

    lines = [
        "/* ============================================================================",
        " * KERNEL FUNCTION DECLARATIONS (from kernel registry)",
        " * ============================================================================ */",
        "",
        "/* CKDataType enum - MUST match ckernel_dtype.h order exactly */",
        "typedef enum {",
        "    CK_DT_FP32 = 0,  /* 0 */",
        "    CK_DT_BF16,      /* 1 */",
        "    CK_DT_FP16,      /* 2 */",
        "    CK_DT_INT8,      /* 3 */",
        "    CK_DT_INT4,      /* 4 */",
        "    CK_DT_Q4_0,      /* 5 */",
        "    CK_DT_Q4_1,      /* 6 */",
        "    CK_DT_Q4_K,      /* 7 */",
        "    CK_DT_Q6_K,      /* 8 */",
        "    CK_DT_Q8_0,      /* 9 */",
        "    CK_DT_Q8_K,      /* 10 */",
        "    CK_DT_Q5_0,      /* 11 */",
        "    CK_DT_Q5_1,      /* 12 */",
        "    CK_DT_COUNT",
        "} CKDataType;",
        "",
    ]

    for kernel in kernels:
        if kernel in decl_map:
            lines.append(decl_map[kernel])
        else:
            # Fallback to generic declaration
            lines.append(f"/* WARNING: No c_declaration in registry for {kernel} */")
            lines.append(f"extern void {kernel}();")

    lines.append("")
    return "\n".join(lines)


def generate_header(config: Dict, operations: List[Dict], registry: Dict, layout_header: str) -> str:
    """Generate C file header with includes, layout, and declarations."""
    kernels = extract_unique_kernels(operations)
    model_name = config.get("model", "model").upper().replace("-", "_")

    header = f'''/*
 * Auto-generated by codegen_v6_6.py
 * Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
 *
 * Model: {config.get("model", "unknown")}
 * Layers: {config.get("num_layers", 0)}, Embed: {config.get("embed_dim", 0)}, Heads: {config.get("num_heads", 0)}
 *
 * Unique kernels ({len(kernels)}): {", ".join(kernels)}
 *
 * Memory layout: Struct-based precomputed offsets (no runtime pointer math)
 * Codegen: Data-driven from kernel maps + lowered IR
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <errno.h>
#include <sys/mman.h>

/* ============================================================================
 * HUGE PAGE ALLOCATION (from v6.5/v6.6)
 * 1. First try explicit huge pages via mmap + MAP_HUGETLB
 * 2. Fallback to aligned_alloc with MADV_HUGEPAGE hint
 * ============================================================================ */

#ifndef HUGE_PAGE_SIZE
#define HUGE_PAGE_SIZE (2UL * 1024UL * 1024UL)
#endif

static size_t ck_align_up(size_t n, size_t align) {{
    if (align == 0) return n;
    return (n + align - 1) & ~(align - 1);
}}

static int g_was_mmap = 0;  /* Track if we used mmap for cleanup */

static void* ck_huge_alloc(size_t bytes) {{
    size_t len = ck_align_up(bytes, HUGE_PAGE_SIZE);

    /* First, try explicit huge pages via mmap + MAP_HUGETLB */
    void *p = mmap(NULL, len,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                   -1, 0);
    if (p != MAP_FAILED) {{
        g_was_mmap = 1;
        fprintf(stderr, "[ck_huge_alloc] Using explicit huge pages for %zu MB\\n", len / (1024*1024));
        return p;
    }}

    /* Fallback: aligned_alloc with transparent hugepage hint */
    void *q = aligned_alloc(HUGE_PAGE_SIZE, len);
    if (!q) {{
        fprintf(stderr, "ck_huge_alloc: aligned_alloc failed for %zu bytes: %s\\n",
                len, strerror(errno));
        return NULL;
    }}

    /* Best-effort hint for transparent huge pages; ignore errors */
    (void)madvise(q, len, MADV_HUGEPAGE);
    g_was_mmap = 0;
    fprintf(stderr, "[ck_huge_alloc] Using aligned_alloc + MADV_HUGEPAGE for %zu MB\\n", len / (1024*1024));
    return q;
}}

static void ck_huge_free(void *ptr, size_t bytes) {{
    if (!ptr || bytes == 0) return;
    size_t len = ck_align_up(bytes, HUGE_PAGE_SIZE);
    if (g_was_mmap) {{
        munmap(ptr, len);
    }} else {{
        free(ptr);
    }}
}}

/* ============================================================================
 * MEMORY LAYOUT (precomputed offsets from bump_layout_v6_6.py)
 * ============================================================================ */

{layout_header}

{generate_kernel_declarations(kernels, registry)}

/* KV cache dimensions */
#define KV_CACHE_SIZE ({model_name}_DECODE_NUM_LAYERS * 2 * {model_name}_DECODE_MAX_SEQ_LEN * {model_name}_DECODE_NUM_KV_HEADS * {model_name}_DECODE_HEAD_DIM * sizeof(float))

/* Convenience aliases */
#define EMBED_DIM {model_name}_DECODE_EMBED_DIM
#define NUM_HEADS {model_name}_DECODE_NUM_HEADS
#define NUM_KV_HEADS {model_name}_DECODE_NUM_KV_HEADS
#define HEAD_DIM {model_name}_DECODE_HEAD_DIM
#define NUM_LAYERS {model_name}_DECODE_NUM_LAYERS
#define INTERMEDIATE_SIZE {model_name}_DECODE_INTERMEDIATE
#define VOCAB_SIZE {model_name}_DECODE_VOCAB_SIZE
#define MAX_SEQ_LEN {model_name}_DECODE_MAX_SEQ_LEN
#define WEIGHTS_SIZE {model_name}_DECODE_WEIGHT_BYTES
#define ACTIVATIONS_SIZE {model_name}_DECODE_ACTIVATION_BYTES
#define ROPE_THETA {model_name}_DECODE_ROPE_THETA

/* Struct-based offset access */
#define L_LAYERS {model_name}_DECODE_LAYERS
#define L_HEADER {model_name}_DECODE_HEADER
#define L_FOOTER {model_name}_DECODE_FOOTER
#define LayerOffsets {model_name}_DECODELayerOffsets

/* Weight pointer macros */
#define W_PTR(offset) ((void*)((uint8_t*)(model->bump_weights) + (offset)))
#define W_FLOAT(offset) ((float*)((uint8_t*)(model->bump_weights) + (offset)))
#define A_PTR(offset) ((float*)((uint8_t*)(model->activations) + (offset)))

'''
    return header


def generate_memory_struct(memory: Dict, config: Dict) -> str:
    """Generate memory structure definitions with single contiguous bump."""
    return '''
/* ============================================================================
 * EXPORT MACRO (for shared library visibility)
 * ============================================================================ */

#ifdef _WIN32
#define CK_EXPORT __declspec(dllexport)
#else
#define CK_EXPORT __attribute__((visibility("default")))
#endif

/* ============================================================================
 * MODEL STRUCTURE - Single Contiguous Bump Allocation
 * ============================================================================ */

/*
 * Memory Layout (single contiguous allocation):
 *   [0 .. WEIGHTS_SIZE)         : Weight data (from bump file)
 *   [WEIGHTS_SIZE .. +ACT)      : Activation buffers
 *   [+ACT .. +KV)               : KV cache
 *   [+KV .. +ROPE)              : RoPE cos/sin tables
 *   [+ROPE .. +LOGITS)          : Output logits
 *
 * This enables:
 *   1. Single mmap for entire model
 *   2. Zero-copy loading
 *   3. Distributed computing (known offsets)
 */

/* Computed offsets within the bump */
#define BUMP_WEIGHTS_OFFSET     0
#define BUMP_ACT_OFFSET         WEIGHTS_SIZE
#define BUMP_KV_OFFSET          (BUMP_ACT_OFFSET + ACTIVATIONS_SIZE)
#define BUMP_ROPE_COS_OFFSET    (BUMP_KV_OFFSET + KV_CACHE_SIZE)
#define BUMP_ROPE_SIN_OFFSET    (BUMP_ROPE_COS_OFFSET + (MAX_SEQ_LEN * HEAD_DIM * sizeof(float)))
#define BUMP_LOGITS_OFFSET      (BUMP_ROPE_SIN_OFFSET + (MAX_SEQ_LEN * HEAD_DIM * sizeof(float)))
#define BUMP_TOTAL_SIZE         (BUMP_LOGITS_OFFSET + (VOCAB_SIZE * sizeof(float)))

typedef struct {
    /* Single contiguous bump allocation */
    uint8_t* bump;
    size_t bump_size;

    /* Pointers into bump (computed at init) */
    uint8_t* bump_weights;   /* Weights section */
    size_t bump_weights_size;
    float* activations;       /* Activation buffers */
    size_t activations_size;
    float* kv_cache;          /* KV cache */
    size_t kv_cache_size;
    float* rope_cos;          /* RoPE cos table */
    float* rope_sin;          /* RoPE sin table */
    float* logits;            /* Output logits */

    /* Vocab data (from bump header) */
    int32_t* vocab_offsets;   /* Offset into vocab_strings for each token */
    char* vocab_strings;      /* Packed vocabulary token strings */
    int32_t* vocab_merges;    /* BPE merge rules [token_a, token_b, merged_token] */
    int vocab_size;           /* Number of tokens in vocabulary */
    int vocab_strings_size;   /* Total bytes of vocab strings */
    int num_merges;           /* Number of BPE merge rules */

    /* Current position in sequence */
    int pos;

} CKModel;

'''


def generate_init_function(config: Dict) -> str:
    """Generate model initialization function with SINGLE contiguous allocation.

    Uses ck_huge_alloc() for huge page support:
    1. First tries mmap + MAP_HUGETLB for explicit huge pages
    2. Falls back to aligned_alloc + MADV_HUGEPAGE for transparent huge pages

    All memory regions (weights, activations, KV cache, RoPE, logits) are in ONE
    contiguous buffer. Offsets are computed at compile time via BUMP_*_OFFSET macros.
    """
    return f'''
/* Global model instance (singleton for ck_chat.py compatibility) */
static CKModel* g_model = NULL;

/* BUMPWGT5 header structure (matches convert_gguf_to_bump_v6_6.py) */
typedef struct __attribute__((packed)) {{
    char magic[8];           /* "BUMPWGT5" or "BUMPWGT4" */
    uint32_t version;        /* 5 */
    uint32_t model_type;     /* 1 = legacy */
    uint32_t num_layers;
    uint32_t vocab_size;
    uint32_t embed_dim;
    uint32_t intermediate;
    uint32_t context_len;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint64_t aligned_embed_dim;
    uint64_t aligned_head_dim;
    uint64_t aligned_intermediate;
    uint64_t aligned_context;
    uint32_t num_merges;
    uint32_t total_vocab_bytes;
    uint8_t checksum[16];
}} BumpHeader;

#define BUMP_HEADER_SIZE 104  /* Size of BumpHeader (verified against converter) */

/* Internal: Allocate model with SINGLE contiguous buffer using huge pages */
static CKModel* ck_model_alloc(void) {{
    CKModel* model = (CKModel*)calloc(1, sizeof(CKModel));
    if (!model) return NULL;

    /* ═══════════════════════════════════════════════════════════════════
     * SINGLE CONTIGUOUS ALLOCATION for entire model
     * Layout: [weights | activations | kv_cache | rope_cos | rope_sin | logits]
     * Uses huge pages (mmap + MAP_HUGETLB) with madvise fallback
     * ═══════════════════════════════════════════════════════════════════ */
    model->bump_size = BUMP_TOTAL_SIZE;
    model->bump = (uint8_t*)ck_huge_alloc(BUMP_TOTAL_SIZE);
    if (!model->bump) {{
        fprintf(stderr, "Failed to allocate model buffer (%zu MB)\\n", (size_t)BUMP_TOTAL_SIZE / (1024*1024));
        free(model);
        return NULL;
    }}

    /* Zero-initialize entire bump (weights will be overwritten on load) */
    memset(model->bump, 0, BUMP_TOTAL_SIZE);

    /* ═══════════════════════════════════════════════════════════════════
     * Set up convenience pointers into the single contiguous buffer
     * All offsets are precomputed at compile time (BUMP_*_OFFSET macros)
     * ═══════════════════════════════════════════════════════════════════ */

    /* Weights section */
    model->bump_weights = model->bump + BUMP_WEIGHTS_OFFSET;
    model->bump_weights_size = WEIGHTS_SIZE;

    /* Activations section */
    model->activations = (float*)(model->bump + BUMP_ACT_OFFSET);
    model->activations_size = ACTIVATIONS_SIZE;

    /* KV cache section */
    model->kv_cache = (float*)(model->bump + BUMP_KV_OFFSET);
    model->kv_cache_size = KV_CACHE_SIZE;

    /* RoPE tables section */
    model->rope_cos = (float*)(model->bump + BUMP_ROPE_COS_OFFSET);
    model->rope_sin = (float*)(model->bump + BUMP_ROPE_SIN_OFFSET);

    /* Logits buffer section */
    model->logits = (float*)(model->bump + BUMP_LOGITS_OFFSET);

    /* Precompute RoPE tables */
    for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {{
        for (int i = 0; i < HEAD_DIM / 2; i++) {{
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / HEAD_DIM);
            float theta = pos * freq;
            model->rope_cos[pos * HEAD_DIM + i] = cosf(theta);
            model->rope_cos[pos * HEAD_DIM + HEAD_DIM/2 + i] = cosf(theta);
            model->rope_sin[pos * HEAD_DIM + i] = sinf(theta);
            model->rope_sin[pos * HEAD_DIM + HEAD_DIM/2 + i] = sinf(theta);
        }}
    }}

    /* Initialize vocab to defaults (will be loaded from bump) */
    model->vocab_size = VOCAB_SIZE;
    model->vocab_strings_size = 0;
    model->num_merges = 0;
    model->vocab_offsets = NULL;
    model->vocab_strings = NULL;
    model->vocab_merges = NULL;

    model->pos = 0;

    fprintf(stderr, "Allocated SINGLE contiguous buffer: %zu MB\\n",
            (size_t)BUMP_TOTAL_SIZE / (1024*1024));
    fprintf(stderr, "  weights=%zu MB @ offset 0\\n", (size_t)WEIGHTS_SIZE / (1024*1024));
    fprintf(stderr, "  activations=%zu MB @ offset %zu\\n", (size_t)ACTIVATIONS_SIZE / (1024*1024), (size_t)BUMP_ACT_OFFSET);
    fprintf(stderr, "  kv_cache=%zu MB @ offset %zu\\n", (size_t)KV_CACHE_SIZE / (1024*1024), (size_t)BUMP_KV_OFFSET);
    fprintf(stderr, "  rope=%zu KB @ offset %zu\\n", (size_t)(MAX_SEQ_LEN * HEAD_DIM * 2 * sizeof(float)) / 1024, (size_t)BUMP_ROPE_COS_OFFSET);
    fprintf(stderr, "  logits=%zu KB @ offset %zu\\n", (size_t)(VOCAB_SIZE * sizeof(float)) / 1024, (size_t)BUMP_LOGITS_OFFSET);

    return model;
}}

/* Initialize model from weights file (ck_chat.py API) */
CK_EXPORT int ck_model_init(const char* weights_path) {{
    if (g_model) {{
        ck_model_free();
    }}

    g_model = ck_model_alloc();
    if (!g_model) {{
        fprintf(stderr, "Failed to allocate model\\n");
        return -1;
    }}

    /* Load weights */
    FILE* f = fopen(weights_path, "rb");
    if (!f) {{
        fprintf(stderr, "Failed to open weights file: %s\\n", weights_path);
        return -1;
    }}

    /* Read BUMPWGT5/BUMPWGT4 header */
    BumpHeader header;
    size_t weights_offset = 0;

    if (fread(&header, sizeof(header), 1, f) == 1) {{
        if (memcmp(header.magic, "BUMPWGT5", 8) == 0) {{
            /* BUMPWGT5 format has extended metadata after header:
             *   [0..104)   : BumpHeader (104 bytes)
             *   [104..128) : Extended metadata (24 bytes)
             *   [128..132) : dtype_table_len (4 bytes, uint32)
             *   [132..132+dtype_table_len) : dtype table
             *   [132+dtype_table_len..) : Weights data
             */
            g_model->vocab_size = header.vocab_size > 0 ? header.vocab_size : VOCAB_SIZE;
            g_model->num_merges = header.num_merges;
            g_model->vocab_strings_size = header.total_vocab_bytes;

            /* Read dtype_table_len from offset 128 */
            fseek(f, 128, SEEK_SET);
            uint32_t dtype_table_len = 0;
            if (fread(&dtype_table_len, sizeof(dtype_table_len), 1, f) == 1) {{
                weights_offset = 132 + dtype_table_len;
            }} else {{
                /* Fallback if can't read dtype_table_len */
                weights_offset = BUMP_HEADER_SIZE;
            }}
        }} else if (memcmp(header.magic, "BUMPWGT4", 8) == 0) {{
            /* BUMPWGT4 format - weights start right after header */
            g_model->vocab_size = header.vocab_size > 0 ? header.vocab_size : VOCAB_SIZE;
            g_model->num_merges = header.num_merges;
            g_model->vocab_strings_size = header.total_vocab_bytes;
            weights_offset = BUMP_HEADER_SIZE;
        }} else {{
            /* Not a BUMP file - assume raw weights starting at offset 0 */
            weights_offset = 0;
        }}
    }} else {{
        /* Couldn't read header - assume raw weights */
        weights_offset = 0;
    }}

    /* Seek to weights data */
    fseek(f, weights_offset, SEEK_SET);

    /* Read weights */
    size_t read = fread(g_model->bump_weights, 1, g_model->bump_weights_size, f);
    fclose(f);

    if (read < g_model->bump_weights_size * 0.9) {{  /* Allow some tolerance */
        fprintf(stderr, "Weight file size mismatch: expected %zu, got %zu (offset=%zu)\\n",
                g_model->bump_weights_size, read, weights_offset);
        return -1;
    }}

    return 0;
}}

/* Free model - SINGLE allocation to free! (ck_chat.py API) */
CK_EXPORT void ck_model_free(void) {{
    if (!g_model) return;

    /* Free the SINGLE contiguous buffer using ck_huge_free */
    if (g_model->bump) {{
        ck_huge_free(g_model->bump, g_model->bump_size);
    }}

    /* Note: bump_weights, activations, kv_cache, rope_cos, rope_sin, logits
     * are ALL pointers INTO the single bump allocation, not separate allocations */

    /* Vocab data might be separately allocated (from header parsing) */
    if (g_model->vocab_offsets) free(g_model->vocab_offsets);
    if (g_model->vocab_strings) free(g_model->vocab_strings);
    if (g_model->vocab_merges) free(g_model->vocab_merges);

    free(g_model);
    g_model = NULL;
}}

'''


def resolve_binding_source(param: Dict, op: Dict, layer: int) -> str:
    """Resolve a binding parameter to its C expression.

    param: {"name": "wq", "source": "weight:wq", "cast": "int32_t*"} or similar
    op: The operation dict with weights, activations, params
    layer: The layer index
    """
    source = param.get("source", "")
    if not source:
        return "NULL"

    parts = source.split(":", 1)
    source_type = parts[0]
    source_ref = parts[1] if len(parts) > 1 else ""

    weights = op.get("weights", {})
    cast = param.get("cast", "")

    # Helper to apply cast if specified
    def apply_cast(expr: str) -> str:
        if cast:
            return f"({cast})({expr})"
        return expr

    if source_type == "weight":
        # Weight via struct offset
        if source_ref == "_first_weight":
            # Find first weight in op's weights dict
            source_ref = list(weights.keys())[0] if weights else "unknown"
        if source_ref in ["token_emb", "final_ln_weight", "final_ln_bias"]:
            return f"W_PTR(L_HEADER.{source_ref})"
        return f"W_PTR(L->{source_ref})"

    elif source_type == "weight_f":
        # Weight as float*
        actual_ref = source_ref
        section = op.get("section", "body")

        # Special case: _gamma means find the actual gamma weight
        if source_ref == "_gamma" or source_ref not in weights:
            # Check alt sources
            alts = param.get("alt", [])
            for alt in alts:
                if alt in weights:
                    actual_ref = alt
                    break
            else:
                # If no alt found and section is footer, use final_ln_weight
                if section == "footer" and "final_ln_weight" in alts:
                    actual_ref = "final_ln_weight"
                elif layer >= 0:
                    # Default to first alt for layer ops
                    actual_ref = alts[0] if alts else source_ref

        # Header weights are accessed via L_HEADER
        header_weights = ["token_emb", "final_ln_weight", "final_ln_bias"]
        if actual_ref in header_weights:
            expr = f"W_FLOAT(L_HEADER.{actual_ref})"
        else:
            expr = f"W_FLOAT(L->{actual_ref})"

        # Apply cast if specified (e.g., const float*)
        if cast and cast != "float*":
            return f"({cast})({expr})"
        return expr

    elif source_type == "output":
        # Direct reference to output field: "output:y" looks up op["outputs"]["y"]
        outputs = op.get("outputs", {})
        actual_offset = 0

        # Try exact match first
        if source_ref in outputs:
            info = outputs[source_ref]
            if isinstance(info, dict) and "activation_offset" in info:
                actual_offset = info["activation_offset"]
        else:
            # Fallback: use first output found
            for name, info in outputs.items():
                if isinstance(info, dict) and "activation_offset" in info:
                    actual_offset = info["activation_offset"]
                    break

        if cast:
            return f"(({cast})((uint8_t*)model->activations + {actual_offset}))"
        return f"A_PTR({actual_offset})"

    elif source_type == "activation":
        # Direct reference to activation field: "activation:x" looks up op["activations"]["x"]
        activations = op.get("activations", {})
        actual_offset = 0

        # Try exact match first
        if source_ref in activations:
            info = activations[source_ref]
            if isinstance(info, dict) and "activation_offset" in info:
                actual_offset = info["activation_offset"]
        else:
            # Fallback: use first activation found (or numeric offset if source_ref is a number)
            if source_ref.isdigit():
                actual_offset = int(source_ref)
            else:
                for name, info in activations.items():
                    if isinstance(info, dict) and "activation_offset" in info:
                        actual_offset = info["activation_offset"]
                        break

        if cast:
            return f"(({cast})((uint8_t*)model->activations + {actual_offset}))"
        return f"A_PTR({actual_offset})"

    elif source_type == "dim":
        dim_map = {
            "embed_dim": "EMBED_DIM",
            "num_heads": "NUM_HEADS",
            "num_kv_heads": "NUM_KV_HEADS",
            "head_dim": "HEAD_DIM",
            "num_layers": "NUM_LAYERS",
            "intermediate_size": "INTERMEDIATE_SIZE",
            "vocab_size": "VOCAB_SIZE",
            "max_seq_len": "MAX_SEQ_LEN",
            "_output_dim": "EMBED_DIM",  # Default, can be specialized
            "_input_dim": "EMBED_DIM",
        }
        # Special case for gemv dims based on weight name
        if source_ref in ["_output_dim", "_input_dim"]:
            w_key = list(weights.keys())[0] if weights else ""
            if w_key in ["w1", "w3"]:
                if source_ref == "_output_dim":
                    return "INTERMEDIATE_SIZE"
            elif w_key == "w2":
                if source_ref == "_input_dim":
                    return "INTERMEDIATE_SIZE"
            return "EMBED_DIM"
        return dim_map.get(source_ref, source_ref.upper())

    elif source_type == "runtime":
        # KV cache layout: [num_layers, 2, num_kv_heads, max_seq_len, head_dim]
        # K slice for layer L: offset = L * 2 * num_kv_heads * max_seq_len * head_dim
        # V slice for layer L: offset = (L * 2 + 1) * num_kv_heads * max_seq_len * head_dim
        kv_layer_stride = "NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM"
        kv_pos_stride = "NUM_KV_HEADS * HEAD_DIM"
        runtime_map = {
            "kv_k": f"model->kv_cache + {layer} * 2 * {kv_layer_stride}",
            "kv_v": f"model->kv_cache + ({layer} * 2 + 1) * {kv_layer_stride}",
            # Per-layer KV cache base pointers (for kv_cache_store and attention)
            "kv_cache_k_layer": f"model->kv_cache + {layer} * 2 * {kv_layer_stride}",
            "kv_cache_v_layer": f"model->kv_cache + ({layer} * 2 + 1) * {kv_layer_stride}",
            # RoPE caches
            "rope_cos": "model->rope_cos + model->pos * HEAD_DIM",
            "rope_sin": "model->rope_sin + model->pos * HEAD_DIM",
            # Runtime values
            "pos": "model->pos",
            "layer": str(layer),
            "seq_len": "(model->pos + 1)",  # For attention: attend to all positions 0..pos
        }
        expr = runtime_map.get(source_ref, source_ref)
        if cast:
            return f"({cast})({expr})"
        return expr

    elif source_type == "dtype":
        dtype_map = {
            "fp32": "CK_DT_FP32",
            "q8_0": "CK_DT_Q8_0",
            "q5_0": "CK_DT_Q5_0",
            "q4_k": "CK_DT_Q4_K",
            "q6_k": "CK_DT_Q6_K",
        }
        return dtype_map.get(source_ref, "CK_DT_FP32")

    elif source_type == "const":
        return source_ref

    elif source_type == "null":
        return "NULL"

    elif source_type == "scratch":
        # Scratch buffer from lowered IR's scratch list
        scratch_list = op.get("scratch", [])
        scratch_offset = 0
        if scratch_list:
            # Find the matching scratch buffer by name or use first one
            for scratch in scratch_list:
                if source_ref and scratch.get("name", "") == source_ref:
                    scratch_offset = scratch.get("scratch_offset", 0)
                    break
                elif not source_ref:
                    scratch_offset = scratch.get("scratch_offset", 0)
                    break
            else:
                # Use first scratch buffer if no match
                scratch_offset = scratch_list[0].get("scratch_offset", 0)
        if cast:
            return f"(({cast})((uint8_t*)model->activations + {scratch_offset}))"
        return f"((void*)((uint8_t*)model->activations + {scratch_offset}))"

    return source


# Global bindings cache
_KERNEL_BINDINGS = None

def get_kernel_bindings() -> Dict:
    """Load and cache kernel bindings."""
    global _KERNEL_BINDINGS
    if _KERNEL_BINDINGS is None:
        _KERNEL_BINDINGS = load_kernel_bindings()
    return _KERNEL_BINDINGS


def generate_op_call_from_ir(op: Dict, model_prefix: str) -> str:
    """Generate C code for an operation DIRECTLY from lowered IR data.

    THIS IS THE TRUE DATA-DRIVEN APPROACH:
    1. Read function name from op["function"]
    2. Read input/output offsets directly from op["activations"]/op["outputs"]
    3. Read weight refs directly from op["weights"]
    4. NO special cases, NO hardcoded offsets
    """
    function = op.get("function", op.get("kernel", "unknown"))
    idx = op.get("idx", 0)
    layer = op.get("layer", -1)
    section = op.get("section", "body")
    op_name = op.get("op", "unknown")

    lines = []
    lines.append(f"    /* Op {idx}: {function} ({op_name}, {section}, L{layer}) */")

    # Get all data from IR
    weights = op.get("weights", {})
    activations = op.get("activations", {})
    outputs = op.get("outputs", {})
    params = op.get("params", {})
    scratch_list = op.get("scratch", [])

    # Build lookup for scratch offsets
    scratch_offsets = {}
    for s in scratch_list:
        scratch_offsets[s.get("name", "")] = s.get("scratch_offset", 0)

    # Helper to get activation offset by name
    def get_activation_offset(name):
        info = activations.get(name, {})
        if isinstance(info, dict):
            return info.get("activation_offset", 0)
        return 0

    # Helper to get output offset by name
    def get_output_offset(name):
        info = outputs.get(name, {})
        if isinstance(info, dict):
            return info.get("activation_offset", 0)
        return 0

    # Helper to get first input offset
    def get_first_input_offset():
        for name, info in activations.items():
            if isinstance(info, dict) and "activation_offset" in info:
                return info["activation_offset"]
        return 0

    # Helper to get first output offset
    def get_first_output_offset():
        for name, info in outputs.items():
            if isinstance(info, dict) and "activation_offset" in info:
                return info["activation_offset"]
        return 0

    # Helper to get weight reference (generates L->name or L_HEADER.name)
    def get_weight_ref(name, as_float=False):
        header_weights = ["token_emb", "final_ln_weight", "final_ln_bias"]
        if name in header_weights:
            return f"W_FLOAT(L_HEADER.{name})" if as_float else f"W_PTR(L_HEADER.{name})"
        return f"W_FLOAT(L->{name})" if as_float else f"W_PTR(L->{name})"

    # Generate based on kernel type - ALL using IR data
    if layer >= 0:
        lines.append(f"    {{")
        lines.append(f"        const LayerOffsets *L = &L_LAYERS[{layer}];")
        indent = "        "
    else:
        indent = "    "

    # EMBEDDING
    if function == "embedding_forward_q8_0":
        # Get offsets directly from IR
        token_offset = get_activation_offset("tokens")
        output_offset = get_first_output_offset()
        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    ((int32_t*)((uint8_t*)model->activations + {token_offset})),")
        lines.append(f"{indent}    1,")
        lines.append(f"{indent}    VOCAB_SIZE,")
        lines.append(f"{indent}    W_PTR(L_HEADER.token_emb),")
        lines.append(f"{indent}    NULL,")
        lines.append(f"{indent}    ((float*)((uint8_t*)model->activations + {output_offset})),")
        lines.append(f"{indent}    EMBED_DIM,")
        lines.append(f"{indent}    EMBED_DIM,")
        lines.append(f"{indent}    MAX_SEQ_LEN,")
        lines.append(f"{indent}    0")
        lines.append(f"{indent});")

    # RMSNORM
    elif function == "rmsnorm_forward":
        input_offset = get_first_input_offset()
        output_offset = get_first_output_offset()
        gamma_ref = get_weight_ref("ln1_gamma", as_float=True)
        # Check if we have final_ln_weight for footer
        if section == "footer" and "final_ln_weight" in weights:
            gamma_ref = get_weight_ref("final_ln_weight", as_float=True)
        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    ((const float*)((uint8_t*)model->activations + {input_offset})),")
        lines.append(f"{indent}    {gamma_ref},")
        lines.append(f"{indent}    ((float*)((uint8_t*)model->activations + {output_offset})),")
        lines.append(f"{indent}    NULL,")
        lines.append(f"{indent}    1,")
        lines.append(f"{indent}    EMBED_DIM,")
        lines.append(f"{indent}    EMBED_DIM,")
        lines.append(f"{indent}    1e-6f")
        lines.append(f"{indent});")

    # GEMV variants (q_proj, k_proj, v_proj, out_proj, mlp_gate_up, mlp_down)
    elif function.startswith("gemv_"):
        input_offset = get_first_input_offset()
        output_offset = get_first_output_offset()
        # Get weight name from weights dict
        weight_name = list(weights.keys())[0] if weights else "wq"
        # Get bias name if present
        bias_name = None
        for w in weights:
            if w.startswith("b"):
                bias_name = w
                break
        # Determine output dim based on op type
        if op_name in ["q_proj"]:
            out_dim = "NUM_HEADS * HEAD_DIM"
        elif op_name in ["k_proj", "v_proj"]:
            out_dim = "NUM_KV_HEADS * HEAD_DIM"
        elif op_name in ["mlp_gate_up"]:
            out_dim = "INTERMEDIATE_SIZE * 2"
        elif op_name in ["mlp_down", "out_proj"]:
            out_dim = "EMBED_DIM"
        else:
            out_dim = "EMBED_DIM"
        # Input dim
        if op_name in ["mlp_down"]:
            in_dim = "INTERMEDIATE_SIZE"
        else:
            in_dim = "EMBED_DIM"

        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    ((float*)((uint8_t*)model->activations + {output_offset})),")
        lines.append(f"{indent}    {get_weight_ref(weight_name)},")
        lines.append(f"{indent}    ((const float*)((uint8_t*)model->activations + {input_offset})),")
        lines.append(f"{indent}    {out_dim},")
        lines.append(f"{indent}    {in_dim}")
        lines.append(f"{indent});")
        # Add bias if present
        if bias_name:
            lines.append(f"{indent}/* Add bias */")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}    float *out = (float*)((uint8_t*)model->activations + {output_offset});")
            lines.append(f"{indent}    const float *bias = {get_weight_ref(bias_name, as_float=True)};")
            lines.append(f"{indent}    for (int i = 0; i < {out_dim}; i++) out[i] += bias[i];")
            lines.append(f"{indent}}}")

    # ROPE
    elif function == "rope_forward_qk":
        q_offset = get_activation_offset("q") or get_first_input_offset()
        k_offset = get_activation_offset("k") or scratch_offsets.get("k_scratch", 0)
        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    ((float*)((uint8_t*)model->activations + {q_offset})),")
        lines.append(f"{indent}    ((float*)((uint8_t*)model->activations + {k_offset})),")
        lines.append(f"{indent}    model->rope_cos + model->pos * HEAD_DIM / 2,")
        lines.append(f"{indent}    model->rope_sin + model->pos * HEAD_DIM / 2,")
        lines.append(f"{indent}    NUM_HEADS,")
        lines.append(f"{indent}    NUM_KV_HEADS,")
        lines.append(f"{indent}    1,")
        lines.append(f"{indent}    HEAD_DIM,")
        lines.append(f"{indent}    HEAD_DIM,")
        lines.append(f"{indent}    model->pos")
        lines.append(f"{indent});")

    # KV CACHE STORE
    elif function == "kv_cache_store":
        k_offset = scratch_offsets.get("k_scratch", get_activation_offset("k"))
        v_offset = scratch_offsets.get("v_scratch", get_activation_offset("v"))
        lines.append(f"{indent}/* Store K/V to KV cache at layer {layer}, pos */")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    const size_t head_stride = MAX_SEQ_LEN * HEAD_DIM;")
        lines.append(f"{indent}    float *kv_k = model->kv_cache + {layer} * 2 * NUM_KV_HEADS * head_stride;")
        lines.append(f"{indent}    float *kv_v = model->kv_cache + ({layer} * 2 + 1) * NUM_KV_HEADS * head_stride;")
        lines.append(f"{indent}    const float *k = (const float*)((uint8_t*)model->activations + {k_offset});")
        lines.append(f"{indent}    const float *v = (const float*)((uint8_t*)model->activations + {v_offset});")
        lines.append(f"{indent}    for (int h = 0; h < NUM_KV_HEADS; h++) {{")
        lines.append(f"{indent}        memcpy(kv_k + h * head_stride + model->pos * HEAD_DIM,")
        lines.append(f"{indent}               k + h * HEAD_DIM, HEAD_DIM * sizeof(float));")
        lines.append(f"{indent}        memcpy(kv_v + h * head_stride + model->pos * HEAD_DIM,")
        lines.append(f"{indent}               v + h * HEAD_DIM, HEAD_DIM * sizeof(float));")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")

    # ATTENTION (decode mode - single token attending to KV cache)
    elif "attention" in function and "flash" in function:
        q_offset = get_activation_offset("q") or get_first_input_offset()
        output_offset = get_first_output_offset()
        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    ((const float*)((uint8_t*)model->activations + {q_offset})),")
        lines.append(f"{indent}    model->kv_cache + {layer} * 2 * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM,")
        lines.append(f"{indent}    model->kv_cache + ({layer} * 2 + 1) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM,")
        lines.append(f"{indent}    ((float*)((uint8_t*)model->activations + {output_offset})),")
        lines.append(f"{indent}    NUM_HEADS,")
        lines.append(f"{indent}    NUM_KV_HEADS,")
        lines.append(f"{indent}    model->pos + 1,")
        lines.append(f"{indent}    HEAD_DIM,")
        lines.append(f"{indent}    HEAD_DIM,")
        lines.append(f"{indent}    MAX_SEQ_LEN")
        lines.append(f"{indent});")

    # RESIDUAL ADD
    elif function == "ck_residual_add_token_major":
        a_offset = get_activation_offset("a") or get_first_input_offset()
        b_offset = get_activation_offset("b") or get_activation_offset("residual") or 0
        output_offset = get_first_output_offset()
        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    ((const float*)((uint8_t*)model->activations + {a_offset})),")
        lines.append(f"{indent}    ((const float*)((uint8_t*)model->activations + {b_offset})),")
        lines.append(f"{indent}    ((float*)((uint8_t*)model->activations + {output_offset})),")
        lines.append(f"{indent}    1,")
        lines.append(f"{indent}    EMBED_DIM")
        lines.append(f"{indent});")

    # SWIGLU
    elif function == "swiglu_forward":
        input_offset = get_first_input_offset()
        output_offset = get_first_output_offset()
        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    ((const float*)((uint8_t*)model->activations + {input_offset})),")
        lines.append(f"{indent}    ((float*)((uint8_t*)model->activations + {output_offset})),")
        lines.append(f"{indent}    1,")
        lines.append(f"{indent}    INTERMEDIATE_SIZE")
        lines.append(f"{indent});")

    # LOGITS (weight tying with embedding)
    elif op_name == "logits":
        input_offset = get_first_input_offset()
        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    model->logits,")
        lines.append(f"{indent}    W_PTR(L_HEADER.token_emb),")
        lines.append(f"{indent}    ((const float*)((uint8_t*)model->activations + {input_offset})),")
        lines.append(f"{indent}    VOCAB_SIZE,")
        lines.append(f"{indent}    EMBED_DIM")
        lines.append(f"{indent});")

    else:
        # Unknown kernel - generate placeholder
        lines.append(f"{indent}/* TODO: {function} not yet implemented in data-driven codegen */")

    if layer >= 0:
        lines.append(f"    }}")

    lines.append("")
    return "\n".join(lines)


def generate_op_call_from_bindings(op: Dict, model_prefix: str) -> str:
    """Generate C code for an operation using kernel_bindings.json.

    This is the data-driven approach:
    1. Look up function in kernel_bindings.json
    2. Iterate through param bindings
    3. Resolve each source to C expression

    Fallback: If kernel not in bindings, use legacy generation.

    NOTE: This function is DEPRECATED - use generate_op_call_from_ir instead.
    Keeping for backward compatibility but routing to new implementation.
    """
    # Route to new data-driven implementation
    return generate_op_call_from_ir(op, model_prefix)
            for param in params:
                c_expr = resolve_binding_source(param, op, layer)
                args.append(c_expr)

            lines.append(f"        {function}(")
            lines.append(f"            " + f",\n            ".join(args))
            lines.append(f"        );")

            # Add bias if layer has bias weights
            lines.append(f"        /* Add {op_name} bias */")
            lines.append(f"        {{")
            lines.append(f"            float *out = (float*)((uint8_t*)model->activations + {out_offset});")
            lines.append(f"            const float *bias = W_FLOAT(L->{bias_name});")
            lines.append(f"            for (int i = 0; i < {out_dim}; i++) out[i] += bias[i];")
            lines.append(f"        }}")
            lines.append(f"    }}")
            lines.append("")
            return "\n".join(lines)

    # Look up bindings from kernel_bindings.json
    bindings = get_kernel_bindings()
    kernel_binding = bindings.get(function, {})

    if kernel_binding:
        # Data-driven: generate from bindings file
        params = kernel_binding.get("params", [])

        if layer >= 0:
            lines.append(f"    {{")
            lines.append(f"        const LayerOffsets *L = &L_LAYERS[{layer}];")

        # Generate function call
        args = []
        for param in params:
            c_expr = resolve_binding_source(param, op, layer)
            args.append(c_expr)

        indent = "        " if layer >= 0 else "    "
        lines.append(f"{indent}{function}(")
        lines.append(f"{indent}    " + f",\n{indent}    ".join(args))
        lines.append(f"{indent});")

        if layer >= 0:
            lines.append(f"    }}")
    else:
        # Legacy: explicit generation (fallback for ops without bindings)
        lines.append(generate_op_call_legacy(op, model_prefix))

    lines.append("")
    return "\n".join(lines)


def generate_op_call_legacy(op: Dict, model_prefix: str) -> str:
    """Legacy explicit op generation for ops without bindings in IR."""
    kernel = op.get("kernel", "")
    function = op.get("function", kernel)
    layer = op.get("layer", -1)

    weights = op.get("weights", {})

    # For ops without bindings, generate based on kernel type
    # This is the fallback - ideally all ops should have bindings

    if "embedding" in kernel:
        return f"""    embedding_forward_q8_0(
        (int32_t*)(model->activations),
        1, VOCAB_SIZE,
        W_PTR(L_HEADER.token_emb),
        NULL,
        A_PTR(0),
        EMBED_DIM, EMBED_DIM, MAX_SEQ_LEN, 0
    );"""

    elif "mega_fused_attention_decode" in kernel:
        return f"""    {{
        const LayerOffsets *L = &L_LAYERS[{layer}];
        mega_fused_attention_decode(
            A_PTR(0), A_PTR(0), A_PTR(0),
            W_FLOAT(L->ln1_gamma),
            W_PTR(L->wq), W_FLOAT(L->bq),
            W_PTR(L->wk), W_FLOAT(L->bk),
            W_PTR(L->wv), W_FLOAT(L->bv),
            W_PTR(L->wo), W_FLOAT(L->bo),
            model->kv_cache + {layer} * 2 * MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM,
            model->kv_cache + ({layer} * 2 + 1) * MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM,
            model->rope_cos + model->pos * HEAD_DIM,
            model->rope_sin + model->pos * HEAD_DIM,
            model->pos,
            EMBED_DIM, EMBED_DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, HEAD_DIM,
            MAX_SEQ_LEN, 1e-5f
        );
    }}"""

    elif "mega_fused_outproj_mlp" in kernel:
        return f"""    {{
        const LayerOffsets *L = &L_LAYERS[{layer}];
        mega_fused_outproj_mlp_prefill(
            A_PTR(0), A_PTR(0), A_PTR(0),
            W_FLOAT(L->ln2_gamma),
            W_PTR(L->wo), W_FLOAT(L->bo), CK_DT_Q5_0,
            W_PTR(L->w1), W_FLOAT(L->b1), CK_DT_Q5_0,
            W_PTR(L->w2), W_FLOAT(L->b2), CK_DT_Q6_K,
            1, EMBED_DIM, EMBED_DIM, NUM_HEADS, HEAD_DIM,
            INTERMEDIATE_SIZE, INTERMEDIATE_SIZE, 1e-5f, NULL
        );
    }}"""

    elif "rmsnorm" in kernel:
        if "final_ln_weight" in weights:
            gamma_ref = "L_HEADER.final_ln_weight"
        elif "ln2_gamma" in weights:
            gamma_ref = f"L_LAYERS[{layer}].ln2_gamma"
        else:
            gamma_ref = f"L_LAYERS[{layer}].ln1_gamma"
        return f"""    rmsnorm_forward(
        A_PTR(0), W_FLOAT({gamma_ref}), A_PTR(0),
        NULL, 1, EMBED_DIM, EMBED_DIM, 1e-5f
    );"""

    elif "gemv" in kernel:
        w_key = None
        for wkey in ["wo", "w1", "w2", "w3", "wq", "wk", "wv"]:
            if wkey in weights:
                w_key = wkey
                break
        if w_key:
            M = "INTERMEDIATE_SIZE" if w_key in ["w1", "w3"] else "EMBED_DIM"
            K = "INTERMEDIATE_SIZE" if w_key == "w2" else "EMBED_DIM"
            return f"""    {{
        const LayerOffsets *L = &L_LAYERS[{layer}];
        {function}(
            A_PTR(0), W_PTR(L->{w_key}), (void*)(model->activations),
            {M}, {K}
        );
    }}"""

    elif "swiglu" in kernel:
        return f"    swiglu_forward(A_PTR(0), A_PTR(0), 1, INTERMEDIATE_SIZE);"

    elif "residual" in kernel:
        return f"    ck_residual_add_token_major(A_PTR(0), A_PTR(0), A_PTR(0), 1, EMBED_DIM);"

    elif "rope" in kernel:
        return f"""    rope_forward_qk(
        A_PTR(0), A_PTR(0),
        model->rope_cos + model->pos * HEAD_DIM,
        model->rope_sin + model->pos * HEAD_DIM,
        NUM_HEADS, NUM_KV_HEADS, 1, HEAD_DIM, HEAD_DIM, model->pos
    );"""

    elif "attention_forward" in kernel:
        return f"""    attention_forward_causal_head_major_gqa_flash_strided(
        A_PTR(0),
        model->kv_cache + {layer} * 2 * MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM,
        model->kv_cache + ({layer} * 2 + 1) * MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM,
        A_PTR(0),
        NUM_HEADS, NUM_KV_HEADS, 1, HEAD_DIM, HEAD_DIM, MAX_SEQ_LEN
    );"""

    return f"    /* TODO: {function} */"


def generate_decode_function(operations: List[Dict], config: Dict, model_prefix: str, token_offset: int = 16, memory: Dict = None) -> str:
    """Generate decode function with struct-based offsets.

    Args:
        operations: List of operations from lowered IR
        config: Model configuration
        model_prefix: Prefix for model-specific defines
        token_offset: Offset in activation buffer for token_ids (from memory.activations.buffers)
        memory: Memory layout with buffer offsets
    """
    lines = []

    # Get buffer offsets from memory layout
    residual_offset = 7188  # default
    layer_output_offset = 814072340  # default
    if memory:
        for buf in memory.get("activations", {}).get("buffers", []):
            if buf.get("name") == "residual":
                residual_offset = buf.get("offset", residual_offset)
            elif buf.get("name") == "layer_output":
                layer_output_offset = buf.get("offset", layer_output_offset)

    lines.append(f'''
/* ============================================================================
 * DECODE FUNCTION (one token at a time, autoregressive)
 * ============================================================================ */

/* Internal decode function - takes model pointer */
static void ck_model_decode_internal(CKModel* model, int32_t token) {{
    /* Store input token in activation buffer at offset {token_offset} (from IR) */
    *((int32_t*)((uint8_t*)model->activations + {token_offset})) = token;

''')

    # Track current layer to emit residual saves
    # Pre-LN transformer: residual flow is:
    #   x_in -> rmsnorm -> attn -> add(attn_out, x_in) -> rmsnorm -> mlp -> add(mlp_out, post_attn_sum)
    # So we need TWO residual saves per layer:
    #   1. Before first rmsnorm: save layer input
    #   2. After first residual_add: save post-attention sum for MLP residual
    prev_layer = -2
    residual_add_count = {}  # Track residual_adds per layer

    for i, op in enumerate(operations):
        layer = op.get("layer", -1)
        section = op.get("section", "")
        op_type = op.get("op", "")

        # Before each layer's first rmsnorm, save current buffer to residual
        if section == "body" and layer >= 0 and layer != prev_layer:
            if op_type == "rmsnorm":
                # Save from layer_output (embedding output for L0, previous layer output for L1+)
                # to residual buffer for later residual add
                lines.append(f'''
    /* Save layer {layer} input to residual buffer */
    memcpy((uint8_t*)model->activations + {residual_offset},
           (uint8_t*)model->activations + {layer_output_offset},
           EMBED_DIM * sizeof(float));
''')
                prev_layer = layer
                residual_add_count[layer] = 0  # Reset count for new layer

        lines.append(generate_op_call_from_bindings(op, model_prefix))

        # After first residual_add in a layer, update residual buffer for MLP block
        # The first residual_add outputs to scratch buffer, which becomes the new residual
        if section == "body" and layer >= 0 and op_type == "residual_add":
            residual_add_count[layer] = residual_add_count.get(layer, 0) + 1
            if residual_add_count[layer] == 1:
                # First residual_add: output is post-attention sum
                # Find the output buffer of this residual_add
                outputs = op.get("outputs", {})
                out_offset = None
                for out_name, out_info in outputs.items():
                    if isinstance(out_info, dict) and "activation_offset" in out_info:
                        out_offset = out_info["activation_offset"]
                        break
                if out_offset is not None:
                    lines.append(f'''
    /* Update residual buffer with post-attention sum for MLP residual */
    memcpy((uint8_t*)model->activations + {residual_offset},
           (uint8_t*)model->activations + {out_offset},
           EMBED_DIM * sizeof(float));
''')

    lines.append('''
    /* Increment position */
    model->pos++;
}
''')

    return "".join(lines)


def generate_prefill_function(operations: List[Dict], config: Dict, model_prefix: str) -> str:
    """Generate prefill function."""
    return '''
/* ============================================================================
 * PREFILL FUNCTION (process multiple tokens at once)
 * ============================================================================ */

static void ck_model_prefill_internal(CKModel* model, int32_t* tokens, int num_tokens) {
    /* For now, decode token by token (correct but slower) */
    for (int i = 0; i < num_tokens; i++) {
        ck_model_decode_internal(model, tokens[i]);
    }
}
'''


def generate_api_functions(token_offset: int = 0) -> str:
    """Generate the ck_model_* API functions matching ck_chat.py."""
    return f'''
/* ============================================================================
 * API FUNCTIONS (ck_chat.py / ck_cli_v6.6 compatible)
 * ============================================================================ */

/* Token input offset in activation buffer (from IR) */
#define TOKEN_INPUT_OFFSET {token_offset}

/* Embed tokens - store in activation buffer for forward pass */
CK_EXPORT int ck_model_embed_tokens(const int32_t* tokens, int num_tokens) {{
    if (!g_model) return -1;
    /* Store tokens in activation buffer at correct offset */
    memcpy((uint8_t*)g_model->activations + TOKEN_INPUT_OFFSET, tokens, num_tokens * sizeof(int32_t));
    return 0;
}}

/* Forward pass - run one decode step */
CK_EXPORT int ck_model_forward(float* output) {{
    if (!g_model) return -1;
    /* Get the token from activations (stored by embed_tokens) */
    int32_t token = *((int32_t*)((uint8_t*)g_model->activations + TOKEN_INPUT_OFFSET));
    ck_model_decode_internal(g_model, token);
    return 0;
}}

/* Get logits pointer */
CK_EXPORT float* ck_model_get_logits(void) {{
    if (!g_model) return NULL;
    return g_model->logits;
}}

/* Get vocab size */
CK_EXPORT int ck_model_get_vocab_size(void) {{
    if (!g_model) return VOCAB_SIZE;
    return g_model->vocab_size;
}}

/* Get context window */
CK_EXPORT int ck_model_get_context_window(void) {{
    return MAX_SEQ_LEN;
}}

/* Get active tokens (current position) */
CK_EXPORT int ck_model_get_active_tokens(void) {{
    if (!g_model) return 0;
    return g_model->pos;
}}

/* Enable KV cache with capacity */
CK_EXPORT int ck_model_kv_cache_enable(int capacity) {{
    /* KV cache is always enabled, capacity is fixed at MAX_SEQ_LEN */
    (void)capacity;
    return 0;
}}

/* Reset KV cache */
CK_EXPORT void ck_model_kv_cache_reset(void) {{
    if (!g_model) return;
    memset(g_model->kv_cache, 0, g_model->kv_cache_size);
    g_model->pos = 0;
}}

/* Decode single token (alternative API) */
CK_EXPORT int ck_model_decode(int32_t token, float* output) {{
    if (!g_model) return -1;
    ck_model_decode_internal(g_model, token);
    if (output) {{
        memcpy(output, g_model->logits, VOCAB_SIZE * sizeof(float));
    }}
    return 0;
}}

/* Sample next token (argmax of logits) */
CK_EXPORT int32_t ck_model_sample_argmax(void) {{
    if (!g_model) return -1;
    float* logits = g_model->logits;
    int vocab_size = g_model->vocab_size;
    int32_t max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {{
        if (logits[i] > max_val) {{
            max_val = logits[i];
            max_idx = i;
        }}
    }}
    return max_idx;
}}

/* ============================================================================
 * VOCAB ACCESSOR FUNCTIONS (for tokenizer initialization)
 * ============================================================================ */

/* Get number of BPE merge rules */
CK_EXPORT int ck_model_get_num_merges(void) {{
    if (!g_model) return 0;
    return g_model->num_merges;
}}

/* Get total vocab strings size in bytes */
CK_EXPORT int ck_model_get_vocab_strings_size(void) {{
    if (!g_model) return 0;
    return g_model->vocab_strings_size;
}}

/* Get vocab offsets array pointer */
CK_EXPORT const int32_t* ck_model_get_vocab_offsets(void) {{
    if (!g_model) return NULL;
    return g_model->vocab_offsets;
}}

/* Get vocab strings data pointer */
CK_EXPORT const char* ck_model_get_vocab_strings(void) {{
    if (!g_model) return NULL;
    return g_model->vocab_strings;
}}

/* Get vocab merges array pointer */
CK_EXPORT const int32_t* ck_model_get_vocab_merges(void) {{
    if (!g_model) return NULL;
    return g_model->vocab_merges;
}}
'''


def generate_code(decode_ir: Dict, prefill_ir: Optional[Dict] = None) -> str:
    """Generate complete C code from lowered IR with struct-based memory layout."""
    config = decode_ir.get("config", {})
    memory = decode_ir.get("memory", {})
    decode_ops = decode_ir.get("operations", [])
    prefill_ops = prefill_ir.get("operations", []) if prefill_ir else []

    # Extract model name and create prefix
    model_name = config.get("model", "model")
    model_prefix = f"{model_name.upper().replace('-', '_')}_DECODE"

    # Generate memory layout from IR
    import sys
    sys.path.insert(0, str(SCRIPTS_DIR))
    from bump_layout_v6_6 import extract_layout_from_ir, generate_header_file as gen_layout
    layout = extract_layout_from_ir(decode_ir)
    layout_header = gen_layout(layout, "decode")

    # Load kernel registry for c_declarations
    registry = load_kernel_registry()

    parts = []

    # Header with embedded layout
    parts.append(generate_header(config, decode_ops, registry, layout_header))

    # Memory struct
    parts.append(generate_memory_struct(memory, config))

    # Forward declarations
    parts.append("/* Forward declarations */")
    parts.append("void ck_model_free(void);")
    parts.append("static void ck_model_decode_internal(CKModel* model, int32_t token);")
    parts.append("")

    # Init and load functions
    parts.append(generate_init_function(config))

    # Find token_ids offset from memory.activations.buffers (NOT from operation bindings)
    # The operation bindings point to layer_input, but we need the actual token_ids buffer
    token_offset = 16  # Default: token_ids is typically at offset 16 (after text_input[16])
    activations_memory = memory.get("activations", {})
    for buf in activations_memory.get("buffers", []):
        if buf.get("name") == "token_ids":
            token_offset = buf.get("offset", 16)
            break

    # Decode function (pass token_offset and memory layout from IR)
    parts.append(generate_decode_function(decode_ops, config, model_prefix, token_offset, memory))

    # Prefill function
    parts.append(generate_prefill_function(prefill_ops, config, model_prefix))

    # API functions (pass token_offset for correct activation buffer access)
    parts.append(generate_api_functions(token_offset))

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate C code from lowered IR")
    parser.add_argument("--decode", type=Path, required=True, help="Lowered IR for decode mode")
    parser.add_argument("--prefill", type=Path, help="Lowered IR for prefill mode")
    parser.add_argument("--output", type=Path, required=True, help="Output C file")

    args = parser.parse_args()

    # Load decode IR
    with open(args.decode, 'r') as f:
        decode_ir = json.load(f)

    # Load prefill IR if provided
    prefill_ir = None
    if args.prefill and args.prefill.exists():
        with open(args.prefill, 'r') as f:
            prefill_ir = json.load(f)

    # Generate code
    code = generate_code(decode_ir, prefill_ir)

    # Write output
    with open(args.output, 'w') as f:
        f.write(code)

    print(f"Generated {args.output}")
    print(f"  Decode ops: {len(decode_ir.get('operations', []))}")
    if prefill_ir:
        print(f"  Prefill ops: {len(prefill_ir.get('operations', []))}")


if __name__ == "__main__":
    main()
