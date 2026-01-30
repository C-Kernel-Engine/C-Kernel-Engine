#!/usr/bin/env python3
"""
build_ir_v6_6.py - Complete IR Pipeline: Template + Quant → IR1 → Fusion → Layout

PIPELINE (4 stages):
    1. IR1 Generation: Template + Quant Summary → Kernel IDs
    2. Fusion Pass: Combine consecutive kernels using registry-driven patterns
    3. Memory Layout: Plan activation buffers and weight offsets
    4. Output: IR1 JSON + Memory Layout JSON

Stage 1 - IR1 Generation (Direct mapping, no intermediate abstractions):
    1. Parse template sequence (what ops to run)
    2. Read quant summary from manifest (what dtypes for weights)
    3. Map template ops → kernel ops → concrete kernel IDs
    4. Return: List of kernel function names

Stage 2 - Fusion Pass:
    1. Scan kernel registry for kernels with "fuses" field
    2. Match consecutive kernel sequences in IR1
    3. Replace matching sequences with fused kernels
    4. Return: Optimized kernel list + fusion statistics

Stage 3 - Memory Layout:
    1. Calculate activation buffer sizes (based on mode: decode vs prefill)
    2. Plan weight memory layout with explicit offsets
    3. Generate buffer allocation map
    4. Return: Complete memory layout with offsets

REQUIREMENTS:
    1. weights_manifest.json with template and quant_summary
    2. KERNEL_REGISTRY.json

USAGE:
    # Generate IR1 only
    python build_ir_v6_6.py --manifest=/path/to/weights_manifest.json \\
        --mode=decode --output=ir1_decode.json

    # Generate full pipeline (IR1 + Fusion + Layout)
    python build_ir_v6_6.py --manifest=/path/to/weights_manifest.json \\
        --mode=decode --output=ir1_decode.json --layout-output=layout_decode.json

OUTPUTS:
    - IR1 JSON: Simple kernel sequence (before fusion)
    - Layout JSON: Fused kernels + memory layout with explicit offsets
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Import memory planner
from memory_planner_v6_6 import plan_memory, MemoryPlanner


# ═══════════════════════════════════════════════════════════════════════════════
# DATAFLOW DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Each op type defines:
#   - inputs: {input_name: slot_name} - which logical slot this input reads from
#   - outputs: {output_name: slot_name} - which logical slot this output writes to
#   - dtype: output dtype (fp32, q8_0, q8_k, etc.)
#
# Slot names are logical (not physical buffers):
#   - "main_stream"     : Primary activation stream (fp32)
#   - "main_stream_q8"  : Quantized activation stream (q8_0 or q8_k)
#   - "residual"        : Saved residual for skip connection
#   - "q_scratch"       : Q projection output
#   - "k_scratch"       : K projection output
#   - "v_scratch"       : V projection output
#   - "attn_scratch"    : Attention output
#   - "mlp_scratch"     : MLP gate_up output
#   - "kv_cache"        : KV cache (persistent across tokens)
#   - "external:X"      : External input (token_ids, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

OP_DATAFLOW = {
    # Header ops
    "dense_embedding_lookup": {
        "inputs": {"token_ids": "external:token_ids"},
        "outputs": {"out": {"slot": "main_stream", "dtype": "fp32"}},
    },

    # Attention block
    "rmsnorm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "quantize_input_0": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "quantize_input_1": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "residual_save": {
        "inputs": {"src": "main_stream"},
        "outputs": {"dst": {"slot": "residual", "dtype": "fp32"}},
    },
    "q_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "q_scratch", "dtype": "fp32"}},
    },
    "k_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "k_scratch", "dtype": "fp32"}},
    },
    "v_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "v_scratch", "dtype": "fp32"}},
    },
    "bias_add_q": {
        "inputs": {"x": "q_scratch"},
        "outputs": {"x": {"slot": "q_scratch", "dtype": "fp32"}},
    },
    "bias_add_k": {
        "inputs": {"x": "k_scratch"},
        "outputs": {"x": {"slot": "k_scratch", "dtype": "fp32"}},
    },
    "bias_add_v": {
        "inputs": {"x": "v_scratch"},
        "outputs": {"x": {"slot": "v_scratch", "dtype": "fp32"}},
    },
    "rope_qk": {
        "inputs": {"q": "q_scratch", "k": "k_scratch"},
        "outputs": {
            "q": {"slot": "q_scratch", "dtype": "fp32"},
            "k": {"slot": "k_scratch", "dtype": "fp32"},
        },
    },
    "kv_cache_store": {
        "inputs": {"k": "k_scratch", "v": "v_scratch"},
        "outputs": {
            "k_cache": {"slot": "kv_cache", "dtype": "fp32"},
            "v_cache": {"slot": "kv_cache", "dtype": "fp32"},
        },
    },
    "attn": {
        "inputs": {"q": "q_scratch", "k": "kv_cache", "v": "kv_cache"},
        "outputs": {"out": {"slot": "attn_scratch", "dtype": "fp32"}},
    },
    "quantize_out_proj_input": {
        "inputs": {"input": "attn_scratch"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "out_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "bias_add": {
        "inputs": {"x": "main_stream"},
        "outputs": {"x": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "residual_add": {
        "inputs": {
            "a": "main_stream",   # Current stream (from out_proj/bias_add)
            "b": "residual",      # Saved residual
        },
        "outputs": {"out": {"slot": "main_stream", "dtype": "fp32"}},
    },

    # MLP block
    "mlp_gate_up": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "bias_add_mlp": {
        "inputs": {"x": "mlp_scratch"},
        "outputs": {"x": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "silu_mul": {
        "inputs": {"x": "mlp_scratch"},
        "outputs": {"x": {"slot": "mlp_scratch", "dtype": "fp32"}},  # In-place
    },
    "quantize_mlp_down_input": {
        "inputs": {"input": "mlp_scratch"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_k"}},
    },
    "mlp_down": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },

    # Footer ops
    "quantize_final_output": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "logits": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "logits", "dtype": "fp32"}},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# INIT OPS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
# Init ops are run ONCE at model load time (not per-token).
# Examples: rope_init (precompute cos/sin tables), ALiBi init, etc.
#
# The init.json file is separate from decode.json and prefill.json because:
#   1. Init ops run once, inference ops run per-token
#   2. Different architectures may need different init ops
#   3. Clean separation of concerns
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_tokenizer_c_code(tokenizer_type: str, vocab_size: int, num_merges: int) -> Optional[Dict]:
    """
    Generate tokenizer-specific C code based on tokenizer type from template.

    The tokenizer type comes from template flags (e.g., "bpe", "wordpiece", "sentencepiece").
    This function generates ALL the C code - codegen just emits it blindly.

    For future tokenizer types, add a new elif branch here.
    """
    if tokenizer_type == "bpe":
        return {
            "type": "bpe",
            "include": '#include "tokenizer/true_bpe.h"',
            "struct_field": "CKTrueBPE *tokenizer;    /* BPE tokenizer */",
            "init": f"""
    /* Initialize BPE tokenizer from bump data */
    g_model->tokenizer = ck_true_bpe_create();
    if (g_model->tokenizer) {{
        ck_true_bpe_load_binary(
            g_model->tokenizer,
            {vocab_size},
            (const int32_t*)(g_model->bump + W_VOCAB_OFFSETS),
            (const char*)(g_model->bump + W_VOCAB_STRINGS),
            {num_merges},
            (const int32_t*)(g_model->bump + W_VOCAB_MERGES)
        );

        /* Register special tokens for pre-BPE matching.
         * Without this, <|im_end|> gets broken into characters by BPE.
         */
        static const char *special_tokens[] = {{
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>",
            "</s>", "<s>",
            NULL
        }};
        for (int i = 0; special_tokens[i] != NULL; i++) {{
            int32_t id = ck_true_bpe_lookup(g_model->tokenizer, special_tokens[i]);
            const char *check = ck_true_bpe_id_to_token(g_model->tokenizer, id);
            if (check && strcmp(check, special_tokens[i]) == 0) {{
                ck_true_bpe_add_special_token(g_model->tokenizer, special_tokens[i], id);
                #ifdef CK_DEBUG_TOKENIZER
                printf("[Tokenizer] Registered special: %s -> %d\\n", special_tokens[i], id);
                #endif
            }}
        }}
    }}""",
            "free": """
    if (g_model->tokenizer) {
        ck_true_bpe_free(g_model->tokenizer);
        g_model->tokenizer = NULL;
    }""",
            "api_functions": """
/* ============================================================================
 * TOKENIZER API - Encode text to tokens using C tokenizer
 * ============================================================================
 * Returns: number of tokens written to internal buffer
 * The tokens are written to the same buffer that prefill() reads from.
 * After encoding, call ck_model_prefill() with the returned count.
 */
CK_EXPORT int ck_model_encode_text(const char *text, int text_len) {
    if (!g_model || !g_model->tokenizer || !text) return 0;
    if (text_len < 0) text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    /* Encode directly into the token_ids buffer that prefill uses */
    int32_t *token_buf = (int32_t*)(g_model->bump + A_TOKEN_IDS);
    int max_tokens = MAX_SEQ_LEN;

    int num_tokens = ck_true_bpe_encode(
        g_model->tokenizer,
        text,
        text_len,
        token_buf,
        max_tokens
    );

    return num_tokens;
}

/* Decode tokens back to text */
CK_EXPORT int ck_model_decode_tokens(const int32_t *ids, int num_ids, char *text, int max_len) {
    if (!g_model || !g_model->tokenizer || !ids || !text || max_len <= 0) return 0;
    return ck_true_bpe_decode(g_model->tokenizer, ids, num_ids, text, max_len);
}

/* Check if tokenizer is available */
CK_EXPORT int ck_model_has_tokenizer(void) {
    return (g_model && g_model->tokenizer) ? 1 : 0;
}

/* Get pointer to token buffer (for reading encoded tokens) */
CK_EXPORT const int32_t* ck_model_get_token_buffer(void) {
    return g_model ? (const int32_t*)(g_model->bump + A_TOKEN_IDS) : NULL;
}

/* Lookup single token by text (returns token ID or -1 if not found)
 * Uses DIRECT vocabulary lookup, not encoding.
 * This is important for special tokens like <|im_end|> which should NOT
 * be encoded through BPE (which would break them into characters).
 */
CK_EXPORT int32_t ck_model_lookup_token(const char *text) {
    if (!g_model || !g_model->tokenizer || !text) return -1;
    /* Direct vocabulary lookup - returns token ID or unk_id if not found */
    int32_t id = ck_true_bpe_lookup(g_model->tokenizer, text);
    /* unk_id is 0 by default, but we want to return -1 for "not found" */
    /* Check if the token is actually in vocab by verifying round-trip */
    const char *token_str = ck_true_bpe_id_to_token(g_model->tokenizer, id);
    if (token_str && strcmp(token_str, text) == 0) {
        return id;  /* Found exact match */
    }
    return -1;  /* Not found or matched to different token */
}
"""
        }

    # Future: Add other tokenizer types here
    # elif tokenizer_type == "wordpiece":
    #     return { ... wordpiece c_code ... }
    # elif tokenizer_type == "sentencepiece":
    #     return { ... sentencepiece c_code ... }

    # Unknown tokenizer type
    return None


def generate_init_ops(manifest: Dict, config: Dict) -> List[Dict]:
    """
    Generate initialization ops based on model config and template flags.

    Returns list of init ops in IR1 format:
        {
            "op_id": 0,
            "kernel": "rope_precompute_cache",
            "op": "rope_init",
            "section": "init",
            "layer": -1,
            "instance": 0,
            "params": {...},
            "outputs": {...}
        }
    """
    init_ops = []
    op_id = 0

    template = manifest.get("template", {})
    flags = template.get("flags", {})

    # ═══════════════════════════════════════════════════════════
    # ROPE INIT: Precompute cos/sin tables if model uses RoPE
    # ═══════════════════════════════════════════════════════════
    rope_type = flags.get("rope", None)
    if rope_type in ("rope", "rope_qk", True):
        # Get config values
        rope_theta = config.get("rope_theta", 10000.0)
        head_dim = config.get("head_dim", 64)
        max_seq_len = config.get("context_length", config.get("max_seq_len", 32768))

        # RoPE scaling (for extended context models like Llama 3.1)
        rope_scaling_type = config.get("rope_scaling_type", "none")
        rope_scaling_factor = config.get("rope_scaling_factor", 1.0)

        init_ops.append({
            "op_id": op_id,
            "kernel": "rope_precompute_cache",
            "op": "rope_init",
            "section": "init",
            "layer": -1,
            "instance": 0,
            "dataflow": {
                "inputs": {},  # No inputs - pure computation from config
                "outputs": {
                    "cos_cache": {"dtype": "fp32", "buffer": "rope_cache"},
                    "sin_cache": {"dtype": "fp32", "buffer": "rope_cache"},
                }
            },
            "params": {
                "max_seq_len": {"source": "dim:max_seq_len", "value": max_seq_len},
                "head_dim": {"source": "dim:head_dim", "value": head_dim},
                "base": {"source": "config:rope_theta", "value": rope_theta},
            },
            "config": {
                "rope_theta": rope_theta,
                "rope_scaling_type": rope_scaling_type,
                "rope_scaling_factor": rope_scaling_factor,
            },
            "notes": f"RoPE cache init: theta={rope_theta}, head_dim={head_dim}, max_seq={max_seq_len}"
        })
        op_id += 1

    # ═══════════════════════════════════════════════════════════
    # TOKENIZER INIT: Load tokenizer from bump data
    # ═══════════════════════════════════════════════════════════
    # Tokenizer type comes from template flags: "bpe", "wordpiece", "sentencepiece", etc.
    tokenizer_type = flags.get("tokenizer", None)

    # Check if vocab data is in manifest (entries list, not weights dict)
    entries = manifest.get("entries", [])
    entry_names = {e.get("name") for e in entries}
    has_vocab = all(k in entry_names for k in ["vocab_offsets", "vocab_strings", "vocab_merges"])

    if has_vocab and tokenizer_type:
        vocab_size = config.get("vocab_size", 151936)
        # Build entry lookup dict
        entry_by_name = {e.get("name"): e for e in entries}
        vocab_offsets_info = entry_by_name.get("vocab_offsets", {})
        vocab_strings_info = entry_by_name.get("vocab_strings", {})
        vocab_merges_info = entry_by_name.get("vocab_merges", {})

        # Calculate number of merges from size (each merge is 3 int32s = 12 bytes)
        merges_size = vocab_merges_info.get("size", 0)
        num_merges = merges_size // 12  # 3 * sizeof(int32_t)

        # Generate tokenizer-specific c_code based on type from template
        c_code = _generate_tokenizer_c_code(tokenizer_type, vocab_size, num_merges)

        if c_code:
            init_ops.append({
                "op_id": op_id,
                "kernel": f"tokenizer_{tokenizer_type}_init",
                "op": "tokenizer_init",
                "section": "init",
                "layer": -1,
                "instance": 0,
                "dataflow": {
                    "inputs": {
                        "vocab_offsets": {"dtype": "i32", "source": "weight:vocab_offsets"},
                        "vocab_strings": {"dtype": "u8", "source": "weight:vocab_strings"},
                        "vocab_merges": {"dtype": "i32", "source": "weight:vocab_merges"},
                    },
                    "outputs": {}
                },
                "params": {
                    "vocab_size": {"source": "config:vocab_size", "value": vocab_size},
                    "num_merges": {"source": "computed", "value": num_merges},
                },
                "c_code": c_code,
                "notes": f"{tokenizer_type.upper()} tokenizer init: vocab_size={vocab_size}, num_merges={num_merges}"
            })
            op_id += 1

    # ═══════════════════════════════════════════════════════════
    # FUTURE: Add other init ops here
    # ═══════════════════════════════════════════════════════════
    # Examples:
    #   - ALiBi slope computation (for models using ALiBi instead of RoPE)
    #   - Learned positional embedding init
    #   - Custom attention bias init

    return init_ops


def generate_init_ir(manifest: Dict, config: Dict) -> Dict:
    """
    Generate the complete init IR (init.json) with all initialization ops.

    Returns:
        {
            "format": "ir1-init-v6.6",
            "version": 1,
            "config": {...},
            "special_tokens": {...},  # EOS, BOS, etc. from GGUF
            "ops": [...]
        }
    """
    init_ops = generate_init_ops(manifest, config)

    # Extract special tokens from manifest for propagation to generated code
    # These come from GGUF metadata (tokenizer.ggml.eos_token_id, etc.)
    special_tokens = manifest.get("special_tokens", {})

    return {
        "format": "ir1-init-v6.6",
        "version": 1,
        "description": "Model initialization ops (run once at load time)",
        "config": {
            "model": config.get("model", "unknown"),
            "rope_theta": config.get("rope_theta", 10000.0),
            "head_dim": config.get("head_dim", 64),
            "max_seq_len": config.get("context_length", 32768),
            "num_heads": config.get("num_heads", 0),
            "num_kv_heads": config.get("num_kv_heads", 0),
        },
        # Special tokens from GGUF - used by orchestrator for EOS detection
        "special_tokens": special_tokens if special_tokens else None,
        "ops": init_ops,
        "stats": {
            "total_ops": len(init_ops),
            "has_rope_init": any(op["op"] == "rope_init" for op in init_ops),
        }
    }


class DataflowTracker:
    """
    Tracks dataflow during IR1 generation.

    Maintains a mapping of slot_name -> (op_id, output_name, dtype) for each logical slot.
    When an op is added, records its inputs (from current slot state) and outputs (updates slot state).
    """

    def __init__(self):
        # Map slot name -> {op_id, output_name, dtype}
        self.slots: Dict[str, Dict[str, Any]] = {}
        # For residual_save tracking within a layer
        self.layer_residual_sources: Dict[int, Dict[str, Any]] = {}  # layer -> slot info

    def reset_for_layer(self, layer: int):
        """Reset per-layer state (but keep residual from previous residual_save)."""
        # Clear main stream slots but keep residual
        pass  # Slots persist, residual_save will update the residual slot

    def record_op(self, op_id: int, op_type: str, layer: int, instance: int) -> Dict[str, Any]:
        """
        Record an op's dataflow and return the dataflow info to embed in IR1.

        Returns:
            {
                "inputs": {input_name: {"from_op": X, "from_output": "Y", "dtype": "Z"}},
                "outputs": {output_name: {"dtype": "Z"}}
            }
        """
        dataflow_def = OP_DATAFLOW.get(op_type, {})

        # ═══════════════════════════════════════════════════════════
        # NOTE: Residual saving is now handled by explicit residual_save ops
        # inserted before rmsnorm in IR1 generation. The residual_save op
        # updates the "residual" slot, and residual_add reads from it.
        # ═══════════════════════════════════════════════════════════

        # Build inputs from current slot state
        inputs = {}
        for input_name, slot_name in dataflow_def.get("inputs", {}).items():
            if slot_name.startswith("external:"):
                # External input (token_ids, etc.)
                inputs[input_name] = {
                    "from": slot_name,
                    "dtype": "i32" if "token" in slot_name else "fp32"
                }
            elif slot_name in self.slots:
                # Get from slot
                slot_info = self.slots[slot_name]
                inputs[input_name] = {
                    "from_op": slot_info["op_id"],
                    "from_output": slot_info["output_name"],
                    "dtype": slot_info["dtype"],
                }
            else:
                # Slot not yet written - this is a bug or first use
                inputs[input_name] = {
                    "from": f"uninitialized:{slot_name}",
                    "dtype": "unknown"
                }

        # Build outputs and update slot state
        outputs = {}
        for output_name, output_info in dataflow_def.get("outputs", {}).items():
            if isinstance(output_info, dict):
                slot_name = output_info["slot"]
                dtype = output_info["dtype"]
            else:
                # Legacy format - just slot name
                slot_name = output_info
                dtype = "fp32"

            outputs[output_name] = {"dtype": dtype}

            # Update slot state
            self.slots[slot_name] = {
                "op_id": op_id,
                "output_name": output_name,
                "dtype": dtype,
            }

            # Special handling for residual_save - track per layer
            if op_type == "residual_save":
                self.layer_residual_sources[layer] = self.slots[slot_name].copy()

        return {
            "inputs": inputs,
            "outputs": outputs,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about tracked dataflow."""
        return {
            "slots_active": list(self.slots.keys()),
            "layers_with_residual": list(self.layer_residual_sources.keys()),
        }


def _sanitize_macro(name: str) -> str:
    """Return an ASCII-safe macro suffix for a name."""
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


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def build_activation_specs(config: Dict[str, Any], mode: str, context_len: int, num_layers_override: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Return activation buffer specs keyed by name."""
    embed_dim = int(config.get("embed_dim", 896))
    num_heads = int(config.get("num_heads", 14))
    num_kv_heads = int(config.get("num_kv_heads", 2))
    head_dim = int(config.get("head_dim", 64))
    intermediate_size = int(config.get("intermediate_size", config.get("intermediate_dim", 4864)))
    vocab_size = int(config.get("vocab_size", 151936))
    num_layers = int(num_layers_override or config.get("num_layers", 24))

    max_context = int(config.get("context_length", 32768))
    if context_len is None:
        context_len = max_context
    else:
        context_len = min(context_len, max_context)

    seq_len = 1 if mode == "decode" else context_len

    specs = {}

    def add(name: str, size: int, shape: str, dtype: str = "fp32") -> None:
        specs[name] = {
            "name": name,
            "size": int(size),
            "shape": shape,
            "dtype": dtype,
        }

    # Text input (optional)
    max_input_bytes = seq_len * 16
    add("text_input", max_input_bytes, f"[{max_input_bytes}]", "u8")

    # Token IDs
    token_ids_size = seq_len * 4
    add("token_ids", token_ids_size, f"[{seq_len}]", "i32")

    # Embedding + layer buffers
    embedded_size = seq_len * embed_dim * 4
    add("embedded_input", embedded_size, f"[{seq_len}, {embed_dim}]")
    add("layer_input", embedded_size, f"[{seq_len}, {embed_dim}]")
    add("residual", embedded_size, f"[{seq_len}, {embed_dim}]")

    # KV cache + RoPE
    kv_per_layer = num_kv_heads * context_len * head_dim * 4
    total_kv_size = num_layers * 2 * kv_per_layer
    add("kv_cache", total_kv_size, f"[{num_layers}, 2, {num_kv_heads}, {context_len}, {head_dim}]")

    rope_size = context_len * (head_dim // 2) * 4 * 2
    add("rope_cache", rope_size, f"[2, {context_len}, {head_dim // 2}]")

    # Scratch buffers
    q_size = num_heads * seq_len * head_dim * 4
    k_size = num_kv_heads * seq_len * head_dim * 4
    v_size = num_kv_heads * seq_len * head_dim * 4
    attn_out_size = num_heads * seq_len * head_dim * 4
    add("q_scratch", q_size, f"[{num_heads}, {seq_len}, {head_dim}]")
    add("k_scratch", k_size, f"[{num_kv_heads}, {seq_len}, {head_dim}]")
    add("v_scratch", v_size, f"[{num_kv_heads}, {seq_len}, {head_dim}]")
    add("attn_scratch", attn_out_size, f"[{num_heads}, {seq_len}, {head_dim}]")

    mlp_size = seq_len * intermediate_size * 2 * 4
    fused_attn_scratch = max(350 * 1024, 3 * num_heads * seq_len * head_dim * 4 + embed_dim * 4 * seq_len * 4)
    scratch_size = max(mlp_size, fused_attn_scratch)
    add("mlp_scratch", scratch_size, f"[max({seq_len}*{intermediate_size*2}, fused_attn)]")

    # Layer output
    layer_out_size = seq_len * embed_dim * 4
    add("layer_output", layer_out_size, f"[{seq_len}, {embed_dim}]")

    # Logits
    logits_seq = 1 if mode == "decode" else seq_len
    logits_size = logits_seq * vocab_size * 4
    add("logits", logits_size, f"[{logits_seq}, {vocab_size}]")

    return specs

# Script directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Template Op → Kernel Op Mapping
# This is the single source of truth for how template ops map to kernel registry ops
# Note: "matmul" is a logical op that maps to gemv (decode) or gemm (prefill) based on mode
TEMPLATE_TO_KERNEL_OP = {
    # Header ops
    "tokenizer": None,  # Metadata op - no kernel (deprecated, use bpe_tokenizer)
    "bpe_tokenizer": None,  # BPE tokenizer - init handled separately
    "wordpiece_tokenizer": None,  # WordPiece tokenizer - init handled separately
    "patch_embeddings": None,  # Vision model patches - init handled separately
    "dense_embedding_lookup": "embedding",  # Token embedding lookup
    "embedding": "embedding",

    # Attention block
    "rmsnorm": "rmsnorm",
    "qkv_proj": "qkv_projection",  # Or fallback to 3x matmul
    "rope_qk": "rope",
    "kv_cache_store": "kv_cache_store",  # Store K,V to KV cache at pos
    "attn": "attention",
    "out_proj": "matmul",  # gemv (decode) or gemm (prefill)

    # Residual
    "residual_add": "residual_add",

    # MLP block
    # NOTE: mega_fused_outproj_mlp_prefill expects head-major attention output,
    # which conflicts with the current pipeline where attention is followed by OutProj.
    # Use simple matmul for mlp_gate_up to avoid the mismatch.
    "mlp_gate_up": "matmul",  # gemv (decode) or gemm (prefill) - use unfused MLP
    "silu_mul": "swiglu",
    "mlp_down": "matmul",  # gemv (decode) or gemm (prefill)

    # Footer ops
    "final_rmsnorm": "rmsnorm",
    "weight_tying": None,  # Metadata op - no kernel
    "logits": "matmul",  # gemv (decode) or gemm (prefill)
}

# Map IR1 weight keys to kernel input names
# IR1 uses: wq, wk, wv, wo, w1, w2, ln1_gamma, token_emb, etc.
# Kernel maps use: W, x, gamma, weight, etc.
WEIGHT_TO_KERNEL_INPUT = {
    # Matrix weights → W
    "wq": "W", "wk": "W", "wv": "W", "wo": "W",
    "w1": "W", "w2": "W", "w3": "W",
    # Biases → bias (if kernel has it)
    "bq": "bias", "bk": "bias", "bv": "bias", "bo": "bias",
    "b1": "bias", "b2": "bias",
    # Layer norms → gamma
    "ln1_gamma": "gamma", "ln2_gamma": "gamma",
    # Embeddings
    "token_emb": "weight",
    # Footer
    "final_ln_weight": "gamma", "final_ln_bias": "bias",
}

def compute_matmul_dims(op_name: str, config: Dict) -> Tuple[Optional[int], Optional[int]]:
    """Compute output/input dims for matmul-like ops (gemv/gemm) and quantize ops."""
    embed = config.get("embed_dim", 896)
    heads = config.get("num_heads", 14)
    kv_heads = config.get("num_kv_heads", 2)
    head_dim = config.get("head_dim", 64)
    inter = config.get("intermediate_size", config.get("intermediate_dim", 4864))
    vocab = config.get("vocab_size", 0)

    if op_name in ("q_proj",):
        return heads * head_dim, embed
    if op_name in ("k_proj", "v_proj"):
        return kv_heads * head_dim, embed
    if op_name in ("out_proj", "attn_proj"):
        return embed, embed
    if op_name in ("mlp_gate_up", "mlp_up"):
        return inter * 2, embed
    if op_name in ("mlp_gate",):
        return inter, embed
    if op_name in ("mlp_down",):
        return embed, inter
    if op_name in ("logits",):
        return vocab, embed
    # Quantize ops: _input_dim is the size to quantize
    # quantize_input_0/1: quantize embed_dim (rmsnorm output before projections)
    # quantize_out_proj_input: quantize embed_dim (attention output)
    # quantize_mlp_down_input: quantize intermediate_size (swiglu output)
    # quantize_final_output: quantize embed_dim (footer rmsnorm output before logits)
    if op_name in ("quantize_input_0", "quantize_input_1", "quantize_out_proj_input", "quantize_final_output"):
        return embed, embed  # output_dim not used, but _input_dim = embed
    if op_name in ("quantize_mlp_down_input",):
        return inter, inter  # _input_dim = intermediate_size
    return None, None


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


def load_kernel_registry() -> Dict:
    """Load kernel registry."""
    registry_path = PROJECT_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
    with open(registry_path, 'r') as f:
        return json.load(f)


def load_manifest(manifest_path: Path) -> Dict:
    """Load weights manifest with template and quant summary."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def validate_template_ops(template_ops: List[str]) -> List[str]:
    """
    Validate that all template ops have kernel mappings.
    Returns list of unmapped ops (empty if all valid).
    """
    unmapped = []
    for op in template_ops:
        if op not in TEMPLATE_TO_KERNEL_OP:
            unmapped.append(op)
    return unmapped


def validate_kernel_availability(registry: Dict, kernel_ops: List[str]) -> Dict[str, bool]:
    """
    Check which kernel ops are available in the registry.
    Returns dict: {kernel_op: is_available}
    """
    available_ops = set()
    for kernel in registry["kernels"]:
        available_ops.add(kernel["op"])

    availability = {}
    for op in kernel_ops:
        availability[op] = op in available_ops

    return availability


def find_kernel(
    registry: Dict,
    op: str,
    quant: Dict[str, str],
    mode: str = "decode",
    prefer_q8_activation: bool = True,  # V6.5 parity: use Q8_0 activation kernels
    prefer_parallel: bool = False  # Use OpenMP-parallel kernels for decode throughput
) -> Optional[str]:
    """
    Find kernel ID from registry.

    Args:
        registry: Kernel registry
        op: Operation type (e.g., "qkv_projection", "matmul", "attention")
        quant: Quantization dict (e.g., {"weight": "q5_0"})
        mode: Execution mode ("decode" or "prefill")
        prefer_q8_activation: If True, prefer Q8_0 activation kernels (V6.5 parity).
                              If False, prefer FP32 activation kernels.
        prefer_parallel: If True, prefer _parallel_omp kernel variants for decode mode.
                         These have the same signature as serial kernels but use OpenMP
                         internally — no wrapper code or IR changes needed.

    Returns:
        Kernel ID (C function name) or None if not found

    Note:
        "matmul" is a logical op that maps to:
        - gemv (matrix-vector) for decode mode (single token)
        - gemm (matrix-matrix) for prefill mode (multiple tokens)
    """
    # Map logical "matmul" to concrete gemv/gemm based on mode
    actual_op = op
    if op == "matmul":
        actual_op = "gemv" if mode == "decode" else "gemm"

    candidates = [k for k in registry["kernels"] if k["op"] == actual_op]

    # Filter and collect all matching candidates
    matches = []
    for candidate in candidates:
        k_quant = candidate.get("quant", {})

        # Match weight quantization
        if "weight" in quant:
            weight_quant = k_quant.get("weight", "")
            # Support multi-quant kernels (e.g., "q5_0|q8_0|q4_k")
            allowed_quants = weight_quant.split("|")
            if quant["weight"] not in allowed_quants and weight_quant != "none":
                continue

        # Match mode (if kernel specifies it)
        kernel_mode = candidate.get("mode", "")
        variant = candidate.get("variant", "")

        # If kernel specifies a mode, it must match
        if kernel_mode and kernel_mode != mode:
            continue

        # Also check variant name for mode hints
        if mode == "decode" and "prefill" in variant:
            continue
        if mode == "prefill" and "decode" in variant:
            continue

        # Collect match
        matches.append(candidate)

    if not matches:
        return None

    # Sort by activation type preference
    # When prefer_q8_activation=True (V6.5 parity): prefer Q8_0 activation kernels
    # When prefer_q8_activation=False: prefer FP32 activation kernels
    def activation_priority(k):
        act = k.get("quant", {}).get("activation", "fp32")
        if prefer_q8_activation:
            # V6.5 parity mode: prefer Q8_0 activation (quantized input)
            if act == "q8_0":
                return 0  # Prefer Q8_0 activation
            if act == "q8_k":
                return 1  # Q8_K is second choice
            return 2  # FP32 last
        else:
            # FP32 mode: prefer FP32 activation (original behavior)
            if act == "fp32":
                return 0  # Prefer FP32 activation
            return 1  # Quantized last

    matches.sort(key=activation_priority)

    # When prefer_parallel=True in decode mode, look for _parallel_omp variant
    # among the top-priority activation matches. These have the same signature
    # as serial kernels — the IR just swaps the function name, no wrapper needed.
    if prefer_parallel and mode == "decode":
        top_act = matches[0].get("quant", {}).get("activation", "fp32")
        same_act = [m for m in matches if m.get("quant", {}).get("activation", "fp32") == top_act]
        for m in same_act:
            if m.get("parallel", False):
                return m["id"]

    return matches[0]["id"]


def kernel_needs_q8_activation(registry: Dict, kernel_id: str) -> bool:
    """
    Check if a kernel requires Q8_0 quantized activation input.

    Args:
        registry: Kernel registry
        kernel_id: Kernel ID to check

    Returns:
        True if kernel expects Q8_0 activation input, False otherwise
    """
    for k in registry.get("kernels", []):
        if k.get("id") == kernel_id:
            act = k.get("quant", {}).get("activation", "fp32")
            return act in ("q8_0", "q8_k")
    return False


def get_quantize_kernel_for_activation(activation_dtype: str) -> Optional[str]:
    """
    Get the appropriate quantize kernel for the target activation dtype.

    Args:
        activation_dtype: Target activation dtype (e.g., "q8_0", "q8_k")

    Returns:
        Quantize kernel ID or None if no quantization needed
    """
    if activation_dtype == "q8_0":
        return "quantize_row_q8_0"
    elif activation_dtype == "q8_k":
        return "quantize_row_q8_k"
    return None


def build_ir1_direct(manifest: Dict, manifest_path: Path, mode: str = "decode",
                     prefer_parallel: bool = False) -> List[str]:
    """
    Direct mapping: Template + Quant Summary → Kernel IDs.

    This is the CORRECT approach - no intermediate abstractions!

    Algorithm:
        1. Validate template ops have kernel mappings
        2. Validate all required kernels exist in registry
        3. For each layer, map template ops → kernel IDs
        4. Return list of kernel IDs (IR1)

    Args:
        manifest: Weights manifest with template and quant_summary
        mode: Execution mode ("decode" or "prefill")
        prefer_parallel: If True, select _parallel_omp kernel variants for decode.
                         These have the same signature as serial kernels but use
                         OpenMP internally — the IR just swaps the function name.

    Returns:
        List of kernel IDs (C function names) in execution order

    Raises:
        RuntimeError: If validation fails (missing mappings or kernels)
    """
    template = manifest.get("template", {})
    quant_summary = manifest.get("quant_summary", {})
    config = manifest.get("config", {})
    registry = load_kernel_registry()

    num_layers = config.get("num_layers", 0)

    # If template is missing or in old format, auto-bump to update it
    # The converter will automatically pull the latest template from templates/ folder
    if not template or "sequence" not in template:
        print(f"\n⚠️  Template missing or outdated (no 'sequence' field)")
        print(f"   Triggering auto-bump to update manifest with latest template...")

        # Find the .bump or .gguf file (should be in same dir as manifest)
        manifest_dir = manifest_path.parent
        bump_file = None

        # Look for .bump file
        for f in manifest_dir.glob("*.bump"):
            bump_file = f
            break

        if not bump_file:
            # Try looking for original GGUF
            for f in manifest_dir.glob("*.gguf"):
                gguf_file = f
                print(f"   Found GGUF: {gguf_file}")
                print(f"   Running converter...")

                converter_script = SCRIPT_DIR / "convert_gguf_to_bump_v6_6.py"
                bump_output = manifest_dir / "weights.bump"

                cmd = [
                    sys.executable,
                    str(converter_script),
                    str(gguf_file),
                    "--output", str(bump_output),
                    "--bump-version=5"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"\n❌ HARD FAULT: Converter failed!")
                    print(result.stderr)
                    raise RuntimeError("Failed to re-bump model")

                print(f"   ✅ Converter succeeded - reloading manifest...")

                # Reload manifest
                manifest_path_new = manifest_dir / "weights_manifest.json"
                manifest = load_manifest(manifest_path_new)
                template = manifest.get("template", {})
                quant_summary = manifest.get("quant_summary", {})
                config = manifest.get("config", {})

                break
        else:
            print(f"\n❌ HARD FAULT: Cannot auto-bump - no .gguf or .bump file found")
            print(f"   Searched in: {manifest_dir}")
            raise RuntimeError("Template missing and cannot auto-bump")

    # Extract ops sequences from v2 template (header, body, footer)
    block_name = template["sequence"][0]
    block = template["block_types"][block_name]

    header_ops = block.get("header", [])
    body_ops = block["body"]["ops"] if isinstance(block["body"], dict) else block["body"]
    footer_ops = block.get("footer", [])

    # For validation, we need all ops
    all_template_ops = header_ops + body_ops + footer_ops

    print(f"\n{'='*60}")
    print("VALIDATION PHASE")
    print(f"{'='*60}")

    # VALIDATION 1: Check template ops have kernel mappings
    print(f"\n[1/2] Validating template ops...")
    print(f"  Header: {header_ops}")
    print(f"  Body: {body_ops}")
    print(f"  Footer: {footer_ops}")

    unmapped_ops = validate_template_ops(all_template_ops)
    if unmapped_ops:
        print(f"\n❌ HARD FAULT: Template ops have no kernel mapping!")
        for op in unmapped_ops:
            print(f"  - {op}")
        print(f"\nAction required:")
        print(f"  Add mappings to TEMPLATE_TO_KERNEL_OP in build_ir_v6_6.py")
        raise RuntimeError(f"Missing kernel mappings for: {unmapped_ops}")

    # Get required kernel ops (filter out None for metadata ops)
    required_kernel_ops = set()
    metadata_ops = []
    for template_op in all_template_ops:
        kernel_op = TEMPLATE_TO_KERNEL_OP[template_op]
        if kernel_op is None:
            metadata_ops.append(template_op)
        else:
            required_kernel_ops.add(kernel_op)

    print(f"  ✅ All {len(all_template_ops)} template ops have mappings")
    if metadata_ops:
        print(f"  Metadata ops (no kernel): {', '.join(metadata_ops)}")
    print(f"  Required kernel ops: {', '.join(sorted(required_kernel_ops))}")

    # VALIDATION 2: Check kernels exist in registry
    print(f"\n[2/2] Validating kernel availability...")

    # Handle "matmul" specially - it maps to gemv (decode) or gemm (prefill)
    validation_kernel_ops = []
    for op in required_kernel_ops:
        if op == "matmul":
            validation_kernel_ops.extend(["gemv", "gemm"])
        else:
            validation_kernel_ops.append(op)

    availability = validate_kernel_availability(registry, validation_kernel_ops)
    missing_kernels = [op for op, avail in availability.items() if not avail]

    if missing_kernels:
        print(f"\n❌ HARD FAULT: Required kernels not found in registry!")
        for op in missing_kernels:
            print(f"  - {op}")
        print(f"\nAction required:")
        print(f"  1. Implement missing kernels")
        print(f"  2. Add to kernel maps and regenerate KERNEL_REGISTRY.json")
        raise RuntimeError(f"Missing kernels: {missing_kernels}")

    print(f"  ✅ All required kernels available (matmul → gemv/gemm)")

    print(f"\n{'='*60}")
    print("IR1 GENERATION PHASE")
    print(f"{'='*60}")

    print(f"\nBuilding IR1 from template...")
    print(f"  Mode: {mode}")
    print(f"  Layers: {num_layers}")

    arranged_kernels = []  # Pass 1: list of {kernel, op, section, layer, op_id, instance, dataflow}
    global_op_id = 0  # Global operation ID counter

    # ═══════════════════════════════════════════════════════════
    # IR1 now includes DATAFLOW information:
    #   - Each op has "dataflow" with "inputs" and "outputs"
    #   - Inputs reference the op_id that produced them
    #   - This enables the memory planner to assign physical buffers
    # ═══════════════════════════════════════════════════════════

    # Initialize dataflow tracker
    dataflow_tracker = DataflowTracker()

    # Weight entries from manifest (for Pass 2 binding)
    entries = manifest.get("entries", [])
    weight_index = {e["name"]: e for e in entries}

    # ═══════════════════════════════════════════════════════════
    # Op → Weight mapping (which weights each op uses for quant lookup)
    # ═══════════════════════════════════════════════════════════
    OP_TO_WEIGHT_KEYS = {
        # Ops with quantized weights - look up in quant_summary
        "qkv_proj": ["wq", "wk", "wv"],  # Split into 3 matmuls if no fused kernel
        "out_proj": ["wo"],
        "mlp_gate_up": ["w1"],
        "mlp_down": ["w2"],
        "dense_embedding_lookup": [],  # Uses token_emb, usually q8_0
        "logits": [],  # Uses lm_head/token_emb, usually q8_0

        # Ops with fp32 weights (no quant lookup needed)
        "rmsnorm": None,  # gamma is always fp32

        # Ops without weights (compute-only)
        "rope_qk": None,
        "kv_cache_store": None,  # Store K,V to KV cache (no weights)
        "attn": None,
        "residual_add": None,
        "silu_mul": None,

        # Metadata ops (no kernel)
        "tokenizer": "metadata",  # Deprecated, use bpe_tokenizer
        "bpe_tokenizer": "metadata",  # BPE tokenizer init
        "wordpiece_tokenizer": "metadata",  # WordPiece tokenizer init
        "patch_embeddings": "metadata",  # Vision model patches
        "weight_tying": "metadata",
    }

    def map_op_to_kernel(op: str, layer_quant: Dict, mode: str) -> List[str]:
        """
        Map template op → kernel ID(s).

        Logic:
            1. If metadata op → return []
            2. If has weight keys → lookup quant → find gemv/gemm kernel
            3. If fp32-only → find fp32 kernel

        Note: prefer_parallel is currently DISABLED (always False).
              OpenMP fork/join overhead makes per-kernel parallelism slower than
              serial for inference workloads (tested: 3.1 tok/s parallel vs 5.9
              tok/s serial on Qwen 0.5B). Needs a persistent thread pool instead
              of OpenMP #pragma omp parallel for. See gemv_omp.c for the kernel
              implementations — they are numerically correct but need a different
              threading model.
        """
        # DISABLED: OpenMP fork/join overhead (~50-200us per call) makes parallel
        # kernels slower for inference. Each decode token calls kernels 500+ times,
        # so thread management overhead dominates. Needs persistent thread pool.
        use_parallel = False  # Was: prefer_parallel and op in PARALLEL_OPS

        kernel_op = TEMPLATE_TO_KERNEL_OP.get(op)
        if not kernel_op:
            return []

        weight_info = OP_TO_WEIGHT_KEYS.get(op)

        # Metadata ops - no kernel
        if weight_info == "metadata":
            return []

        # Ops with quantized weights
        if isinstance(weight_info, list) and weight_info:
            # NOTE: For v6.6, qkv_proj uses standard gemm_nt_* (prefill) or gemv_* (decode)
            # The head-major QKV projection kernel (ck_qkv_project_head_major_quant)
            # was from ckernel_orchestration.c which is not used in v6.6.
            # Fall through to standard matmul handling which splits into q_proj, k_proj, v_proj.

            # NOTE: For v6.6, out_proj uses standard gemm_nt_* (prefill) or gemv_* (decode)
            # The head-major attention projection kernel (ck_attention_project_head_major_quant)
            # was from ckernel_orchestration.c which is not used in v6.6.
            # Fall through to standard matmul handling below.

            # Try fused kernel first (e.g., qkv_projection)
            weight_dtype = layer_quant.get(weight_info[0], "fp32")
            kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": weight_dtype}, mode=mode,
                                    prefer_parallel=use_parallel)
            if kernel_id:
                return [kernel_id]

            # Fallback: split into individual matmuls
            # Return list of (kernel_id, split_op_name) tuples
            kernels = []
            # Map weight key to split op name
            weight_to_split_op = {
                "wq": "q_proj", "wk": "k_proj", "wv": "v_proj",
                "w1": "mlp_gate", "w3": "mlp_up", "w2": "mlp_down",
            }
            for w_key in weight_info:
                w_dtype = layer_quant.get(w_key, "fp32")
                k = find_kernel(registry, op="matmul", quant={"weight": w_dtype}, mode=mode,
                                prefer_parallel=use_parallel)
                if k:
                    split_op = weight_to_split_op.get(w_key, op)
                    kernels.append((k, split_op))
            return kernels

        # Header/footer ops with weights (embedding, logits)
        if isinstance(weight_info, list) and not weight_info:
            # These use fixed quant (typically q8_0)
            kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": "q8_0"}, mode=mode,
                                    prefer_parallel=use_parallel)
            return [kernel_id] if kernel_id else []

        # Ops with fp32 weights or no weights
        kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": "fp32"}, mode=mode,
                                prefer_parallel=use_parallel)
        if kernel_id:
            return [kernel_id]

        # Try without weight quant requirement
        kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": "none"}, mode=mode,
                                prefer_parallel=use_parallel)
        return [kernel_id] if kernel_id else []

    # ═══════════════════════════════════════════════════════════
    # Parse template → Generate IR1
    # ═══════════════════════════════════════════════════════════

    # Track op instance counts during PASS 1 for data flow lookup
    pass1_instance_counts: Dict[tuple, int] = {}  # (layer, op_type) -> count

    def get_op_info(op_type: str, section: str, layer: int) -> dict:
        """Get op_id and instance for an op. Data flow is handled in IR Lower."""
        nonlocal pass1_instance_counts, global_op_id

        # Track instance for body ops (for repeated ops like rmsnorm, residual_add)
        if section == "body":
            key = (layer, op_type)
            instance = pass1_instance_counts.get(key, 0)
            pass1_instance_counts[key] = instance + 1
        else:
            instance = 0

        # Assign global op_id
        op_id = global_op_id
        global_op_id += 1

        return {
            "op_id": op_id,
            "instance": instance,
        }

    for block_name in template["sequence"]:
        block_def = template["block_types"][block_name]
        block_sequence = block_def.get("sequence", ["header", "body", "footer"])

        print(f"\n  Block: {block_name}")

        for section_name in block_sequence:
            section_def = block_def.get(section_name)
            if section_def is None:
                continue

            # Get ops list
            if isinstance(section_def, dict):
                ops = section_def.get("ops", [])
            else:
                ops = section_def if isinstance(section_def, list) else []

            # Body: loop over layers
            if section_name == "body":
                for layer_idx in range(num_layers):
                    layer_key = f"layer.{layer_idx}"
                    layer_quant = quant_summary.get(layer_key, {})
                    # Reset instance counts for each layer
                    pass1_instance_counts = {k: v for k, v in pass1_instance_counts.items()
                                             if k[0] != layer_idx}

                    print(f"\n    Layer {layer_idx}:")

                    # Track rmsnorm instance for quantize insertion
                    rmsnorm_instance = 0

                    for op_idx, op in enumerate(ops):
                        kernels = map_op_to_kernel(op, layer_quant, mode)

                        # Check if we need to insert quantize op after rmsnorm
                        # V6.5 compatibility: quantize activation before Q8_0 activation kernels
                        if op == "rmsnorm" and op_idx + 1 < len(ops):
                            next_op = ops[op_idx + 1]
                            next_kernels = map_op_to_kernel(next_op, layer_quant, mode)
                            needs_quantize = False

                            for nk in next_kernels:
                                nk_id = nk[0] if isinstance(nk, tuple) else nk
                                if kernel_needs_q8_activation(registry, nk_id):
                                    needs_quantize = True
                                    break

                            if needs_quantize:
                                # Insert quantize op after rmsnorm (will be appended after rmsnorm below)
                                pass  # Flag is set, handled below

                        # Check if we need to insert quantize op BEFORE out_proj or mlp_down
                        # V6.5 compatibility: quantize activation output before these projections
                        if op in ("out_proj", "mlp_down") and kernels:
                            first_kernel = kernels[0]
                            fk_id = first_kernel[0] if isinstance(first_kernel, tuple) else first_kernel
                            if kernel_needs_q8_activation(registry, fk_id):
                                # Get activation dtype from kernel
                                for kreg in registry.get("kernels", []):
                                    if kreg.get("id") == fk_id:
                                        act_dtype = kreg.get("quant", {}).get("activation", "fp32")
                                        quantize_kernel = get_quantize_kernel_for_activation(act_dtype)
                                        if quantize_kernel:
                                            quant_op_name = f"quantize_{op}_input"
                                            quant_op_info = get_op_info(quant_op_name, "body", layer_idx)
                                            arranged_kernels.append({
                                                "op_id": quant_op_info["op_id"],
                                                "kernel": quantize_kernel,
                                                "op": quant_op_name,
                                                "section": "body",
                                                "layer": layer_idx,
                                                "instance": 0,
                                            })
                                            print(f"      [{quant_op_info['op_id']:3d}] {quant_op_name:20s} → {quantize_kernel}  (inst: 0) [AUTO-INSERTED]")
                                        break

                        # Insert residual_save BEFORE rmsnorm to save input for skip connection
                        if op == "rmsnorm":
                            residual_save_op_name = f"residual_save"
                            residual_save_info = get_op_info(residual_save_op_name, "body", layer_idx)
                            arranged_kernels.append({
                                "op_id": residual_save_info["op_id"],
                                "kernel": "memcpy",
                                "op": residual_save_op_name,
                                "section": "body",
                                "layer": layer_idx,
                                "instance": rmsnorm_instance,  # Same instance as rmsnorm
                                "_auto_inserted": True,
                            })
                            print(f"      [{residual_save_info['op_id']:3d}] {residual_save_op_name:20s} → memcpy  (inst: {rmsnorm_instance}) [AUTO-INSERTED before rmsnorm]")

                        for k in kernels:
                            # Handle both plain kernel ID and (kernel_id, split_op) tuples
                            if isinstance(k, tuple):
                                kernel_id, split_op = k
                            else:
                                kernel_id, split_op = k, op

                            # Get op_id and instance (data flow is handled in IR Lower)
                            op_info = get_op_info(split_op, "body", layer_idx)

                            arranged_kernels.append({
                                "op_id": op_info["op_id"],
                                "kernel": kernel_id,
                                "op": split_op,
                                "section": "body",
                                "layer": layer_idx,
                                "instance": op_info["instance"],
                            })
                            print(f"      [{op_info['op_id']:3d}] {split_op:20s} → {kernel_id}  (inst: {op_info['instance']})")

                        # Insert quantize op after rmsnorm if needed
                        if op == "rmsnorm" and op_idx + 1 < len(ops):
                            next_op = ops[op_idx + 1]
                            next_kernels = map_op_to_kernel(next_op, layer_quant, mode)

                            for nk in next_kernels:
                                nk_id = nk[0] if isinstance(nk, tuple) else nk
                                if kernel_needs_q8_activation(registry, nk_id):
                                    # Get activation dtype from kernel
                                    for kreg in registry.get("kernels", []):
                                        if kreg.get("id") == nk_id:
                                            act_dtype = kreg.get("quant", {}).get("activation", "fp32")
                                            quantize_kernel = get_quantize_kernel_for_activation(act_dtype)
                                            if quantize_kernel:
                                                quant_op_name = f"quantize_input_{rmsnorm_instance}"
                                                quant_op_info = get_op_info(quant_op_name, "body", layer_idx)
                                                arranged_kernels.append({
                                                    "op_id": quant_op_info["op_id"],
                                                    "kernel": quantize_kernel,
                                                    "op": quant_op_name,
                                                    "section": "body",
                                                    "layer": layer_idx,
                                                    "instance": rmsnorm_instance,
                                                })
                                                print(f"      [{quant_op_info['op_id']:3d}] {quant_op_name:20s} → {quantize_kernel}  (inst: {rmsnorm_instance}) [AUTO-INSERTED]")
                                            break
                                    break
                            rmsnorm_instance += 1

                        if not kernels and OP_TO_WEIGHT_KEYS.get(op) != "metadata":
                            print(f"            {op:20s} → (no kernel)")

            # Header/Footer: run once (no layer quant)
            else:
                print(f"\n    {section_name.capitalize()}:")
                footer_quantize_inserted = False  # Track if we've inserted quantize for footer
                for op_idx, op in enumerate(ops):
                    kernels = map_op_to_kernel(op, {}, mode)

                    # Footer: Insert quantize op BEFORE any op that needs Q8 activation
                    # (after rmsnorm outputs FP32, before logits needs Q8_0)
                    if section_name == "footer" and not footer_quantize_inserted:
                        for k in kernels:
                            k_id = k[0] if isinstance(k, tuple) else k
                            if kernel_needs_q8_activation(registry, k_id):
                                # Get activation dtype from kernel
                                for kreg in registry.get("kernels", []):
                                    if kreg.get("id") == k_id:
                                        act_dtype = kreg.get("quant", {}).get("activation", "fp32")
                                        quantize_kernel = get_quantize_kernel_for_activation(act_dtype)
                                        if quantize_kernel:
                                            quant_op_name = "quantize_final_output"
                                            quant_op_info = get_op_info(quant_op_name, section_name, -1)
                                            arranged_kernels.append({
                                                "op_id": quant_op_info["op_id"],
                                                "kernel": quantize_kernel,
                                                "op": quant_op_name,
                                                "section": section_name,
                                                "layer": -1,
                                                "instance": 0,
                                            })
                                            print(f"      [{quant_op_info['op_id']:3d}] {quant_op_name:20s} → {quantize_kernel}  (inst: 0) [AUTO-INSERTED before {op}]")
                                            footer_quantize_inserted = True
                                        break
                                break

                    for k in kernels:
                        # Handle both plain kernel ID and (kernel_id, split_op) tuples
                        if isinstance(k, tuple):
                            kernel_id, split_op = k
                        else:
                            kernel_id, split_op = k, op

                        # Get op_id and instance (data flow is handled in IR Lower)
                        op_info = get_op_info(split_op, section_name, -1)

                        arranged_kernels.append({
                            "op_id": op_info["op_id"],
                            "kernel": kernel_id,
                            "op": split_op,
                            "section": section_name,
                            "layer": -1,
                            "instance": op_info["instance"],
                        })
                        print(f"      [{op_info['op_id']:3d}] {split_op:20s} → {kernel_id}  (inst: {op_info['instance']})")

                    if not kernels:
                        if OP_TO_WEIGHT_KEYS.get(op) == "metadata":
                            print(f"            {op:20s} → (metadata)")
                        else:
                            print(f"            {op:20s} → (no kernel)")

    print(f"\n✓ Pass 1: Generated {len(arranged_kernels)} kernel calls")

    # ═══════════════════════════════════════════════════════════
    # PASS 1.5: Add dataflow information
    # For each op, record what it reads from and writes to
    # This enables the memory planner to assign physical buffers
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Pass 1.5: Computing dataflow graph...")

    current_layer = -1
    for ir_op in arranged_kernels:
        op_id = ir_op["op_id"]
        op_type = ir_op["op"]
        layer = ir_op["layer"]
        instance = ir_op.get("instance", 0)

        # Reset tracker for new layer
        if layer != current_layer and layer >= 0:
            dataflow_tracker.reset_for_layer(layer)
            current_layer = layer

        # Record dataflow for this op
        dataflow_info = dataflow_tracker.record_op(op_id, op_type, layer, instance)
        ir_op["dataflow"] = dataflow_info

    # Print dataflow stats
    stats = dataflow_tracker.get_stats()
    print(f"  ✓ Pass 1.5: Added dataflow to {len(arranged_kernels)} ops")
    print(f"    Active slots: {', '.join(stats['slots_active'])}")

    # ═══════════════════════════════════════════════════════════
    # PASS 2: Bind weights from sidecar entries
    # Uses instance counts from PASS 1 (stored in ir_op["instance"])
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Pass 2: Binding weights from sidecar...")

    # Mapping for repeated ops: (op_type, instance_index) -> weight_keys
    # Instance index is 0-based (first occurrence = 0)
    REPEATED_OP_WEIGHTS = {
        # rmsnorm: 1st (pre-attention) uses ln1_gamma, 2nd (pre-MLP) uses ln2_gamma
        ("rmsnorm", 0): ["ln1_gamma"],      # Pre-attention norm
        ("rmsnorm", 1): ["ln2_gamma"],      # Pre-MLP norm
        # residual_add: both instances use same weights (none), but tracked for consistency
        ("residual_add", 0): [],            # Post-attention residual
        ("residual_add", 1): [],            # Post-MLP residual
    }

    # Footer-specific weights (no instance tracking needed)
    FOOTER_OP_WEIGHTS = {
        "rmsnorm": ["final_ln_weight", "final_ln_bias"],
    }

    for ir_op in arranged_kernels:
        op = ir_op["op"]
        layer = ir_op["layer"]
        section = ir_op["section"]

        # Use instance from PASS 1 (already computed with data flow)
        instance_idx = ir_op.get("instance", 0)

        # Get weight keys for this op - check repeated op mapping first
        if section == "body" and (op, instance_idx) in REPEATED_OP_WEIGHTS:
            weight_keys = REPEATED_OP_WEIGHTS[(op, instance_idx)]
        elif section == "footer" and op in FOOTER_OP_WEIGHTS:
            weight_keys = FOOTER_OP_WEIGHTS[op]
        else:
            weight_keys = TEMPLATE_OP_WEIGHTS.get(op, [])

        ir_op["weights"] = {}

        for wkey in weight_keys:
            # Build weight name based on section/layer
            if section == "header":
                # Header weights: token_emb, etc (no layer prefix)
                weight_name = wkey
            elif section == "footer":
                # Footer weights: final_ln_weight, etc (no layer prefix)
                weight_name = wkey
            else:
                # Body weights: layer.X.wq, etc
                weight_name = f"layer.{layer}.{wkey}"

            # Look up in manifest entries
            if weight_name in weight_index:
                entry = weight_index[weight_name]
                ir_op["weights"][wkey] = {
                    "name": weight_name,
                    "offset": entry.get("file_offset", 0),
                    "size": entry.get("size", 0),
                    "dtype": entry.get("dtype", "unknown"),
                }
            else:
                # Weight not found - might be optional (biases)
                pass

    # Count weights bound
    total_weights = sum(len(op["weights"]) for op in arranged_kernels)
    print(f"  ✓ Pass 2: Bound {total_weights} weights to {len(arranged_kernels)} ops")

    return arranged_kernels


def apply_fusion_pass(ir1_ops: List[Dict], registry: Dict, mode: str, no_fusion: bool = False) -> tuple[List[Dict], Dict]:
    """
    Apply fusion pass to combine consecutive kernels where fused versions exist.

    Args:
        ir1_ops: List of IR1 ops (each is {kernel, op, section, layer, weights})
        registry: Kernel registry
        mode: Execution mode (decode/prefill)
        no_fusion: If True, skip fusion and return original ops

    Returns:
        (fused_ops, fusion_stats) - New op list after fusion and statistics

    Fusion strategy:
        1. Scan registry for kernels with "fuses" field
        2. Match consecutive kernel sequences
        3. Replace with fused kernel, merge weights
        4. Track fusion statistics
    """
    print(f"\n{'='*60}")
    print("FUSION PASS")
    print(f"{'='*60}")

    # Check for fusion disable flag (parameter only)
    if no_fusion:
        print("  ⚠️ Fusion DISABLED (--no-fusion)")
        return ir1_ops, {"total_fusions": 0, "kernels_removed": 0, "fusions_applied": [], "disabled": True}

    # Build fusion patterns from registry
    fusion_patterns = []
    for kernel in registry["kernels"]:
        if "fuses" not in kernel:
            continue

        # Check if this fused kernel matches the mode
        # NOTE: Allow prefill fused kernels in decode mode (v6.5 parity)
        # The fused prefill kernels work for tokens=1 (decode) and are more accurate
        # because they handle quantization internally.
        variant = kernel.get("variant", "")
        # Don't skip prefill kernels in decode mode - they work with tokens=1
        # if mode == "decode" and "prefill" in variant and "decode" not in variant:
        #     continue
        if mode == "prefill" and "decode" in variant and "prefill" not in variant:
            continue

        pattern = {
            "fused_kernel": kernel["id"],
            "fused_op": kernel.get("op", ""),
            "sequence": kernel.get("fuses", []),
            "variant": variant,
        }
        fusion_patterns.append(pattern)

    print(f"\nFound {len(fusion_patterns)} fusion patterns in registry for {mode} mode")

    # Apply fusion patterns
    fused_ops = [op.copy() for op in ir1_ops]  # Deep copy to avoid mutation
    for op in fused_ops:
        op["weights"] = op.get("weights", {}).copy()

    fusion_stats = {
        "total_fusions": 0,
        "kernels_removed": 0,
        "fusions_applied": [],
    }

    # Sort patterns by sequence length (longest first) for greedy matching
    fusion_patterns.sort(key=lambda p: -len(p["sequence"]))

    changed = True
    while changed:
        changed = False
        for pattern in fusion_patterns:
            sequence = pattern["sequence"]
            seq_len = len(sequence)

            # Scan for matching sequences
            i = 0
            while i <= len(fused_ops) - seq_len:
                # Check if sequence matches (compare kernel/function IDs)
                match = True
                for j in range(seq_len):
                    # Use "function" field if "kernel" not present
                    op_kernel = fused_ops[i + j].get("kernel") or fused_ops[i + j].get("function", "")
                    if op_kernel != sequence[j]:
                        match = False
                        break

                if match:
                    # Safety: if the first op is a quantize op, check it has
                    # exactly 1 consumer. Shared quantize ops (e.g. quantize_input_0
                    # feeding q/k/v projections) must NOT be fused.
                    first_op = fused_ops[i]
                    first_kernel = first_op.get("kernel", "")
                    if first_kernel.startswith("quantize_row_"):
                        first_op_id = first_op.get("op_id")
                        if first_op_id is not None:
                            consumer_count = sum(
                                1 for op in fused_ops
                                if any(
                                    inp.get("from_op") == first_op_id
                                    for inp in op.get("dataflow", {}).get("inputs", {}).values()
                                )
                            )
                            if consumer_count > 1:
                                print(f"\n  Skipping fusion at position {i}: "
                                      f"{first_kernel} (op_id={first_op_id}) has "
                                      f"{consumer_count} consumers")
                                i += 1
                                continue

                    # Found a match - replace with fused kernel
                    fused_id = pattern["fused_kernel"]
                    removed_ops = fused_ops[i:i+seq_len]
                    removed_kernels = [op.get("kernel") or op.get("function", "?") for op in removed_ops]

                    print(f"\n  Fusion opportunity at position {i}:")
                    print(f"    Replacing: {' + '.join(removed_kernels)}")
                    print(f"    With:      {fused_id}")

                    # Merge weights from all fused ops
                    merged_weights = {}
                    for op in removed_ops:
                        merged_weights.update(op.get("weights", {}))

                    # Build correct dataflow for fused op:
                    # - Input: first op's input (FP32 from rmsnorm, renamed to "x")
                    # - Output: last op's output
                    first_dataflow = removed_ops[0].get("dataflow", {})
                    last_dataflow = removed_ops[-1].get("dataflow", {})
                    # Find the "primary" op (gemv) for op name and instance
                    middle_op = removed_ops[1] if seq_len >= 3 else removed_ops[0]

                    fused_dataflow = {}
                    if first_dataflow.get("inputs"):
                        # Rename first op's input key to "x" for fused kernel
                        first_inputs = first_dataflow["inputs"]
                        # Get the first (and typically only) input
                        first_input_key = next(iter(first_inputs))
                        first_input_val = first_inputs[first_input_key]
                        fused_dataflow["inputs"] = {
                            "x": {**first_input_val, "dtype": "fp32"}
                        }
                    if last_dataflow.get("outputs"):
                        fused_dataflow["outputs"] = last_dataflow["outputs"]

                    # Create fused op preserving the primary op's identity
                    fused_op = {
                        "kernel": fused_id,
                        "op": middle_op.get("op", "fused"),
                        "section": removed_ops[0]["section"],
                        "layer": removed_ops[0]["layer"],
                        "instance": middle_op.get("instance", 0),
                        "weights": merged_weights,
                        "fused_from": removed_kernels,
                    }
                    if fused_dataflow:
                        fused_op["dataflow"] = fused_dataflow

                    # Replace sequence with fused op
                    fused_ops[i:i+seq_len] = [fused_op]

                    # Record fusion
                    fusion_stats["fusions_applied"].append({
                        "position": i,
                        "pattern": pattern["fused_op"],
                        "fused_kernel": fused_id,
                        "replaced": removed_kernels,
                    })
                    fusion_stats["total_fusions"] += 1
                    fusion_stats["kernels_removed"] += seq_len - 1

                    changed = True
                    break  # Restart scan after modification

                i += 1

            if changed:
                break  # Restart with new fusion_patterns iteration

    print(f"\n✓ Fusion complete:")
    print(f"  Total fusions: {fusion_stats['total_fusions']}")
    print(f"  Kernels removed: {fusion_stats['kernels_removed']}")
    print(f"  Final kernel count: {len(fused_ops)} (was {len(ir1_ops)})")

    return fused_ops, fusion_stats


def insert_bias_add_ops(
    ir_ops: List[Dict],
    registry: Dict,
    manifest: Dict,
    mode: str,
) -> List[Dict]:
    """
    Insert explicit bias_add ops after projections when kernels do not apply bias.

    This keeps biases visible in the lowered IR and avoids hiding them in codegen.
    """
    # Only insert if bias_add kernel exists
    if not any(k.get("id") == "bias_add" for k in registry.get("kernels", [])):
        print("  Warning: bias_add kernel not found in registry; skipping bias ops")
        return ir_ops

    kernel_maps_dir = PROJECT_ROOT / "kernel_maps"
    kernel_map_cache: Dict[str, Dict] = {}

    def load_kernel_map(kernel_id: str) -> Optional[Dict]:
        if kernel_id in kernel_map_cache:
            return kernel_map_cache[kernel_id]
        kernel_file = kernel_maps_dir / f"{kernel_id}.json"
        if kernel_file.exists():
            with open(kernel_file, "r") as f:
                kernel_map_cache[kernel_id] = json.load(f)
                return kernel_map_cache[kernel_id]
        # fallback to registry entry
        for k in registry.get("kernels", []):
            if k.get("id") == kernel_id:
                kernel_map_cache[kernel_id] = k
                return k
        kernel_map_cache[kernel_id] = {}
        return {}

    def kernel_supports_bias(kernel_id: str) -> bool:
        km = load_kernel_map(kernel_id)
        bias_inputs = {"bias", "bq", "bk", "bv", "bo", "b1", "b2"}
        for inp in km.get("inputs", []):
            name = inp.get("name", "")
            if name in bias_inputs or "bias" in name:
                return True
        for w in km.get("weights", []):
            name = w.get("name", "")
            if name == "bias" or name.startswith("b") or "bias" in name:
                return True
        return False

    bias_key_by_op = {
        "q_proj": "bq",
        "k_proj": "bk",
        "v_proj": "bv",
        "out_proj": "bo",
        "mlp_gate_up": "b1",
        "mlp_down": "b2",
    }

    config = manifest.get("config", {})
    out: List[Dict] = []
    inserted = 0

    for op in ir_ops:
        out.append(op)
        op_type = op.get("op", "")
        bias_key = bias_key_by_op.get(op_type)
        if not bias_key:
            continue
        if bias_key not in op.get("weights", {}):
            continue
        if kernel_supports_bias(op.get("kernel", "")):
            continue

        out_dim, _ = compute_matmul_dims(op_type, config)
        bias_op = {
            "kernel": "bias_add",
            "op": "bias_add",
            "layer": op.get("layer", -1),
            "section": op.get("section", ""),
            "weights": {bias_key: op["weights"][bias_key]},
            "params": {},
            "bias_for": op_type,
            "_auto_inserted": True,
        }
        if out_dim is not None:
            bias_op["params"]["_output_dim"] = out_dim
        out.append(bias_op)
        inserted += 1

    if inserted:
        print(f"  Inserted {inserted} bias_add ops (mode={mode})")

    return out


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: IR LOWER 1 (Stitch kernel maps)
# ═══════════════════════════════════════════════════════════════════════════

def generate_ir_lower_1(
    fused_ops: List[Dict],
    registry: Dict,
    manifest: Dict,
    mode: str
) -> List[Dict]:
    """
    IR Lower 1: Stitch kernel maps with IR1 ops.

    For each fused op:
      1. Load the kernel map (inputs, outputs, scratch)
      2. Map IR1 weights to kernel inputs
      3. Track activation flow between kernels

    This creates the buffer requirements that Memory Planner needs.

    Args:
        fused_ops: Fused IR1 ops from fusion pass
        registry: Kernel registry
        manifest: Model manifest
        mode: decode/prefill

    Returns:
        List of lowered ops with input/output/scratch specs
    """
    print(f"\n{'='*60}")
    print("IR LOWER 1 (Stitch kernel maps)")
    print(f"{'='*60}")

    config = manifest.get("config", {})

    # Build kernel map index by loading individual kernel map files
    # KERNEL_REGISTRY.json is only used for validation, not as source of truth
    kernel_maps_dir = PROJECT_ROOT / "kernel_maps"
    kernel_map_index = {}
    for kernel in registry.get("kernels", []):
        kernel_id = kernel["id"]
        # Try to load individual kernel map file first
        kernel_file = kernel_maps_dir / f"{kernel_id}.json"
        if kernel_file.exists():
            with open(kernel_file, 'r') as f:
                kernel_map_index[kernel_id] = json.load(f)
        else:
            # Fallback to registry entry if no individual file
            kernel_map_index[kernel_id] = kernel
    # Use module-level WEIGHT_TO_KERNEL_INPUT for name mapping

    lowered_ops = []

    # Track activation buffers for dataflow
    # The output of one kernel becomes the input of the next
    current_activation = "input_tokens"  # Start with input token IDs

    for idx, ir_op in enumerate(fused_ops):
        kernel_id = ir_op["kernel"]
        op_name = ir_op["op"]
        layer = ir_op["layer"]
        section = ir_op["section"]
        ir_weights = ir_op.get("weights", {})

        # Get kernel map
        kernel_map = kernel_map_index.get(kernel_id)
        if not kernel_map:
            print(f"  Warning: Kernel '{kernel_id}' not in registry, skipping")
            continue

        # Build lowered op - preserve ALL weights from IR1
        # Also preserve op_id and dataflow for memory planner
        lowered_op = {
            "idx": idx,
            "op_id": ir_op.get("op_id", idx),  # Preserve original op_id for memory planner
            "kernel": kernel_id,
            "op": op_name,
            "layer": layer,
            "section": section,
            "function": kernel_map.get("impl", {}).get("function", kernel_id),
            "weights": ir_weights,  # Preserve all IR1 weights
            "inputs": {},  # Activation inputs only
            "outputs": {},
            "scratch": [],
            "params": ir_op.get("params", {}),
            "bias_for": ir_op.get("bias_for"),
            "dataflow": ir_op.get("dataflow", {}),  # Preserve dataflow for memory planner
        }
        if ir_op.get("_auto_inserted"):
            lowered_op["_auto_inserted"] = True

        # Special handling for residual_save/memcpy: compute _memcpy_bytes
        if op_name == "residual_save":
            embed_dim = manifest.get("config", {}).get("embed_dim", 896)
            seq_len = 1 if mode == "decode" else manifest.get("config", {}).get("context_length", 2048)
            lowered_op["params"]["_memcpy_bytes"] = embed_dim * seq_len * 4  # FP32 = 4 bytes

        # Map kernel input activations from kernel map
        # New format has 4 sections:
        #   - 'inputs': input activations from previous layer
        #   - 'weights': static model parameters (already handled above)
        #   - 'activations': intermediate scratch tensors
        #   - 'outputs': output activations to next layer
        kernel_inputs = kernel_map.get("inputs", [])

        # Build set of weight names from kernel map to filter out weight inputs
        # (handles legacy format where weights were mixed into 'inputs')
        kernel_weight_names = set()
        for kw in kernel_map.get("weights", []):
            kernel_weight_names.add(kw["name"])
        # Also check IR1 weights mapped to kernel input names
        for wkey in ir_weights.keys():
            kernel_weight_names.add(wkey)
            mapped_name = WEIGHT_TO_KERNEL_INPUT.get(wkey)
            if mapped_name:
                kernel_weight_names.add(mapped_name)

        for kernel_input in kernel_inputs:
            input_name = kernel_input["name"]
            input_dtype = kernel_input["dtype"]
            input_shape = kernel_input.get("shape", [])

            # Skip if this is actually a weight parameter
            if input_name in kernel_weight_names:
                continue

            # Activation input (from previous kernel)
            lowered_op["inputs"][input_name] = {
                "type": "activation",
                "source": current_activation,
                "dtype": input_dtype,
                "shape": input_shape,
            }

        # Map kernel outputs
        for kernel_output in kernel_map.get("outputs", []):
            output_name = kernel_output["name"]
            output_dtype = kernel_output["dtype"]
            output_shape = kernel_output.get("shape", [])

            # Create output buffer name
            output_buffer = f"buf_{idx}_{output_name}"

            lowered_op["outputs"][output_name] = {
                "type": "activation",
                "buffer": output_buffer,
                "dtype": output_dtype,
                "shape": output_shape,
            }

            # Update current activation for next kernel
            current_activation = output_buffer

        # Map scratch buffers
        for scratch in kernel_map.get("scratch", []):
            lowered_op["scratch"].append({
                "name": scratch.get("name", f"scratch_{idx}"),
                "size": scratch.get("size", "dynamic"),
                "dtype": scratch.get("dtype", "fp32"),
            })

        lowered_ops.append(lowered_op)

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTOMATIC KV CACHE INSERTION (decode only)
    # ═══════════════════════════════════════════════════════════════════════════
    # Insert kv_cache_store ops after rope_qk to store K,V for use in subsequent decode.
    # For decode, also update attention to use the decode kernel with KV cache.
    print(f"\n  [{mode.capitalize()} mode] Inserting KV cache operations...")
    final_ops = []
    kv_store_count = 0

    for i, op in enumerate(lowered_ops):
        final_ops.append(op)

        if mode == "decode":
            # After rope_qk, insert kv_cache_store
            if op["op"] == "rope_qk":
                layer = op["layer"]
                kv_store_op = {
                    "idx": len(final_ops),  # Will be renumbered
                    "kernel": "kv_cache_store",
                    "op": "kv_cache_store",
                    "layer": layer,
                    "section": op["section"],
                    "function": "kv_cache_store",
                    "weights": {},
                    "inputs": {
                        "k": {"type": "scratch", "source": "k_scratch"},
                        "v": {"type": "scratch", "source": "v_scratch"},
                    },
                    "outputs": {
                        "kv_cache_k": {"type": "kv_cache", "buffer": f"kv_cache_k_L{layer}"},
                        "kv_cache_v": {"type": "kv_cache", "buffer": f"kv_cache_v_L{layer}"},
                    },
                    "scratch": [],
                    "_auto_inserted": True,
                }
                final_ops.append(kv_store_op)
                kv_store_count += 1

            # For decode mode, update attention ops to use decode kernel
            if op["op"] == "attn" and "attention" in op["kernel"]:
                # Switch to decode attention kernel
                op["kernel"] = "attention_forward_decode_head_major_gqa_flash"
                op["function"] = "attention_forward_decode_head_major_gqa_flash"
                # Update inputs to use KV cache instead of scratch
                op["inputs"]["k_cache"] = {"type": "kv_cache", "source": f"kv_cache_k_L{op['layer']}"}
                op["inputs"]["v_cache"] = {"type": "kv_cache", "source": f"kv_cache_v_L{op['layer']}"}
                # Remove scratch K/V references if present
                op["inputs"].pop("k", None)
                op["inputs"].pop("v", None)

        elif mode == "prefill":
            # For prefill: after q_proj/k_proj/v_proj, insert transpose from [T, H*D] to [H, T, D]
            # GEMM outputs token-major but attention expects head-major
            if op["op"] == "q_proj":
                layer = op["layer"]
                transpose_q_op = {
                    "idx": len(final_ops),
                    "kernel": "transpose_qkv_to_head_major",
                    "op": "transpose_qkv_to_head_major",
                    "layer": layer,
                    "section": op["section"],
                    "function": "transpose_inplace",
                    "weights": {},
                    "inputs": {"buf": {"type": "scratch", "source": "q_scratch"}},
                    "outputs": {"buf": {"type": "scratch", "buffer": "q_scratch"}},
                    "scratch": [],
                    "_auto_inserted": True,
                    "_qkv_type": "q",
                }
                final_ops.append(transpose_q_op)

            if op["op"] == "k_proj":
                layer = op["layer"]
                transpose_k_op = {
                    "idx": len(final_ops),
                    "kernel": "transpose_kv_to_head_major",
                    "op": "transpose_kv_to_head_major",
                    "layer": layer,
                    "section": op["section"],
                    "function": "transpose_inplace",
                    "weights": {},
                    "inputs": {"buf": {"type": "scratch", "source": "k_scratch"}},
                    "outputs": {"buf": {"type": "scratch", "buffer": "k_scratch"}},
                    "scratch": [],
                    "_auto_inserted": True,
                    "_is_k": True,
                }
                final_ops.append(transpose_k_op)

            if op["op"] == "v_proj":
                layer = op["layer"]
                transpose_v_op = {
                    "idx": len(final_ops),
                    "kernel": "transpose_kv_to_head_major",
                    "op": "transpose_kv_to_head_major",
                    "layer": layer,
                    "section": op["section"],
                    "function": "transpose_inplace",
                    "weights": {},
                    "inputs": {"buf": {"type": "scratch", "source": "v_scratch"}},
                    "outputs": {"buf": {"type": "scratch", "buffer": "v_scratch"}},
                    "scratch": [],
                    "_auto_inserted": True,
                    "_is_k": False,
                }
                final_ops.append(transpose_v_op)

            # For prefill: after attention, transpose output from head-major [H, T, D] to token-major [T, H*D]
            # Then insert kv_cache_batch_copy to copy K/V from scratch to cache
            if op["op"] == "attn":
                layer = op["layer"]
                # First: transpose attention output from head-major to token-major
                transpose_attn_out_op = {
                    "idx": len(final_ops),
                    "kernel": "transpose_attn_out_to_token_major",
                    "op": "transpose_attn_out_to_token_major",
                    "layer": layer,
                    "section": op["section"],
                    "function": "transpose_inplace",
                    "weights": {},
                    "inputs": {"buf": {"type": "scratch", "source": "attn_scratch"}},
                    "outputs": {"buf": {"type": "scratch", "buffer": "attn_scratch"}},
                    "scratch": [],
                    "_auto_inserted": True,
                }
                final_ops.append(transpose_attn_out_op)
                # Second: kv_cache_batch_copy
                kv_batch_copy_op = {
                    "idx": len(final_ops),
                    "kernel": "kv_cache_batch_copy",
                    "op": "kv_cache_batch_copy",
                    "layer": layer,
                    "section": op["section"],
                    "function": "memcpy",  # Will emit memcpy calls for K and V
                    "weights": {},
                    "inputs": {
                        "k_src": {"type": "scratch", "source": "k_scratch"},
                        "v_src": {"type": "scratch", "source": "v_scratch"},
                    },
                    "outputs": {
                        "k_dst": {"type": "kv_cache", "buffer": f"kv_cache_k_L{layer}"},
                        "v_dst": {"type": "kv_cache", "buffer": f"kv_cache_v_L{layer}"},
                    },
                    "scratch": [],
                    "_auto_inserted": True,
                }
                final_ops.append(kv_batch_copy_op)
                kv_store_count += 1

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTOMATIC LOGITS COPY FOR PREFILL
    # ═══════════════════════════════════════════════════════════════════════════
    # In prefill mode, logits are computed for ALL tokens as [num_tokens, vocab_size].
    # But ck_model_forward() expects logits at position 0 (for the LAST token).
    # Insert a copy_last_logits op to copy logits[(n-1)*V : n*V] to logits[0:V].
    copy_last_logits_inserted = False
    if mode == "prefill":
        # Insert copy_last_logits at the very end
        copy_last_logits_op = {
            "idx": len(final_ops),
            "kernel": "copy_last_logits",
            "op": "copy_last_logits",
            "layer": -1,
            "section": "footer",
            "function": "memmove",  # Use memmove for safety (overlapping memory)
            "weights": {},
            "inputs": {
                "src": {"type": "activation", "source": "logits", "offset": "(num_tokens - 1) * vocab_size"},
            },
            "outputs": {
                "dst": {"type": "activation", "buffer": "logits"},
            },
            "scratch": [],
            "_auto_inserted": True,
            "_copy_size": "vocab_size * sizeof(float)",
        }
        final_ops.append(copy_last_logits_op)
        copy_last_logits_inserted = True
        print(f"  Inserted copy_last_logits op for prefill mode")

    # Renumber ops
    for i, op in enumerate(final_ops):
        op["idx"] = i

    lowered_ops = final_ops
    print(f"  Inserted {kv_store_count} kv_cache_store operations")
    if mode == "decode":
        print(f"  Updated {kv_store_count} attention ops to use decode kernel")

    # Summary
    total_weight_refs = sum(len(op.get("weights", {})) for op in lowered_ops)
    total_activations = sum(len(op.get("inputs", {})) for op in lowered_ops)

    print(f"\n✓ IR Lower 1 complete:")
    print(f"  Lowered ops: {len(lowered_ops)}")
    print(f"  Weight references: {total_weight_refs}")
    print(f"  Activation inputs: {total_activations}")

    return lowered_ops


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: MEMORY PLANNER (with aggressive validation)
# ═══════════════════════════════════════════════════════════════════════════

# Weight name patterns for matching IR1 ops to manifest entries
# Maps: kernel weight ref → possible manifest entry patterns
WEIGHT_PATTERNS = {
    # QKV projection weights and biases
    "wq": ["layer.{L}.wq", "layers.{L}.attention.wq"],
    "wk": ["layer.{L}.wk", "layers.{L}.attention.wk"],
    "wv": ["layer.{L}.wv", "layers.{L}.attention.wv"],
    "bq": ["layer.{L}.bq", "layers.{L}.attention.bq"],
    "bk": ["layer.{L}.bk", "layers.{L}.attention.bk"],
    "bv": ["layer.{L}.bv", "layers.{L}.attention.bv"],

    # Output projection
    "wo": ["layer.{L}.wo", "layers.{L}.attention.wo"],
    "bo": ["layer.{L}.bo", "layers.{L}.attention.bo"],

    # MLP weights and biases
    "w1": ["layer.{L}.w1", "layers.{L}.feed_forward.w1"],
    "w2": ["layer.{L}.w2", "layers.{L}.feed_forward.w2"],
    "w3": ["layer.{L}.w3", "layers.{L}.feed_forward.w3"],
    "b1": ["layer.{L}.b1", "layers.{L}.feed_forward.b1"],
    "b2": ["layer.{L}.b2", "layers.{L}.feed_forward.b2"],

    # Layer norms
    "ln1_gamma": ["layer.{L}.ln1_gamma", "layers.{L}.attention_norm.weight"],
    "ln2_gamma": ["layer.{L}.ln2_gamma", "layers.{L}.ffn_norm.weight"],

    # Header weights
    "token_emb": ["token_emb", "token_embd.weight", "embed_tokens.weight"],
    "pos_emb": ["pos_emb", "pos_embd.weight", "position_embedding"],

    # Vocab/tokenizer data (not model weights, but need to track)
    "vocab_offsets": ["vocab_offsets"],
    "vocab_strings": ["vocab_strings"],
    "vocab_merges": ["vocab_merges"],

    # Footer weights
    "final_ln_weight": ["final_ln_weight", "norm.weight"],
    "final_ln_bias": ["final_ln_bias", "norm.bias"],
    "output_weight": ["output.weight", "lm_head.weight"],
}

# Template op → weight refs it uses
# This tells us which weights each template op needs
TEMPLATE_OP_WEIGHTS = {
    # Header (tokenizer is metadata, not model weights)
    "tokenizer": [],  # Deprecated, use bpe_tokenizer
    "bpe_tokenizer": [],  # BPE tokenizer data handled separately (not model weights)
    "wordpiece_tokenizer": [],  # WordPiece tokenizer data handled separately
    "patch_embeddings": [],  # Vision model patches handled separately
    "dense_embedding_lookup": ["token_emb"],  # Token embeddings only (pos_emb for non-RoPE)

    # Attention block (body + footer)
    # Body: uses ln1_gamma, ln2_gamma (per-layer)
    # Footer: uses final_ln_weight, final_ln_bias (once)
    "rmsnorm": ["ln1_gamma", "ln2_gamma", "final_ln_weight", "final_ln_bias"],
    "qkv_proj": ["wq", "wk", "wv", "bq", "bk", "bv"],  # QKV + optional biases (for fused kernel)
    "q_proj": ["wq", "bq"],  # Q projection only (when split)
    "k_proj": ["wk", "bk"],  # K projection only (when split)
    "v_proj": ["wv", "bv"],  # V projection only (when split)
    "rope_qk": [],  # No model weights (uses precomputed tables)
    "attn": [],  # No model weights
    "out_proj": ["wo", "bo"],  # Output projection + optional bias
    "residual_add": [],  # No model weights

    # MLP block
    "mlp_gate_up": ["w1", "w3", "b1"],  # Gate + up projection
    "silu_mul": [],  # No model weights
    "mlp_down": ["w2", "b2"],  # Down projection

    # Footer
    "weight_tying": [],  # Metadata only
    # For weight tying: logits uses token_emb (embedding matrix) transposed
    # output_weight is for models without weight tying
    "logits": ["token_emb"],  # Uses embedding matrix for weight tying
}


def generate_memory_layout(
    ir_lower_1_ops: List[Dict],
    manifest: Dict,
    registry: Dict,
    mode: str,
    context_len: int = None
) -> Dict:
    """
    Generate memory layout with AGGRESSIVE VALIDATION.

    Args:
        ir_lower_1_ops: List of IR Lower 1 ops (each is {kernel, op, section, layer,
                        inputs, outputs, scratch}). Inputs have type='weight' or
                        type='activation'.

    This function is a VALIDATION GATE:
    - HARD FAULT if any manifest weight is unused
    - HARD FAULT if any required weight is missing

    Steps:
    1. Build weight index from manifest entries (actual sizes)
    2. Extract weight usage from IR Lower 1 inputs (type='weight')
    3. Validate 100% weight coverage
    4. Plan activation buffers from IR Lower 1 outputs
    5. Return complete layout

    Returns:
        Layout dict with memory allocation plan and validation status

    Raises:
        RuntimeError: If validation fails (unused or missing weights)
    """
    print(f"\n{'='*60}")
    print("MEMORY PLANNER (with validation)")
    print(f"{'='*60}")

    config = manifest.get("config", {})
    entries = manifest.get("entries", [])
    template = manifest.get("template", {})
    num_layers = config.get("num_layers", 24)

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Build weight index from manifest entries
    # ═══════════════════════════════════════════════════════════

    if not entries:
        raise RuntimeError(
            "HARD FAULT: Manifest has no 'entries' field!\n"
            "  The manifest must contain weight tensor entries.\n"
            "  Re-run converter with --bump-version=5"
        )

    all_weights = {}  # name -> {dtype, size, offset, ...}
    total_weight_size = 0

    # First pass: collect all entries and find min file_offset (weights base)
    min_file_offset = None
    for entry in entries:
        fo = entry.get("file_offset", 0)
        if min_file_offset is None or fo < min_file_offset:
            min_file_offset = fo

    weights_base_offset = min_file_offset or 0

    for entry in entries:
        name = entry["name"]
        size = entry.get("size", entry.get("size_bytes", 0))
        file_offset = entry.get("file_offset", 0)

        # Compute relative offset from weights base
        relative_offset = file_offset - weights_base_offset

        all_weights[name] = {
            "name": name,
            "dtype": entry.get("dtype", "unknown"),
            "size": size,
            "file_offset": file_offset,
            "relative_offset": relative_offset,  # Offset relative to weights_base_offset
        }
        total_weight_size = max(total_weight_size, relative_offset + size)

    print(f"\n📦 Manifest weights:")
    print(f"  Total entries: {len(all_weights)}")
    print(f"  Weights base offset in file: {weights_base_offset}")
    print(f"  Total size: {total_weight_size / 1024 / 1024:.1f} MB")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Determine which weights SHOULD be used by template
    # ═══════════════════════════════════════════════════════════

    # Get template ops from template
    block_name = template.get("sequence", ["decoder"])[0]
    block = template.get("block_types", {}).get(block_name, {})

    header_ops = block.get("header", [])
    body_def = block.get("body", {})
    body_ops = body_def.get("ops", []) if isinstance(body_def, dict) else body_def
    footer_ops = block.get("footer", [])

    print(f"\n📋 Template structure:")
    print(f"  Header ops: {header_ops}")
    print(f"  Body ops: {body_ops}")
    print(f"  Footer ops: {footer_ops}")

    # Calculate expected weights based on template
    expected_weights = set()
    weight_to_op = {}  # Track which op uses each weight

    # Header weights (run once)
    for op in header_ops:
        weight_refs = TEMPLATE_OP_WEIGHTS.get(op, [])
        for ref in weight_refs:
            patterns = WEIGHT_PATTERNS.get(ref, [ref])
            for pattern in patterns:
                weight_name = pattern  # Header weights don't have layer index
                if weight_name in all_weights:
                    expected_weights.add(weight_name)
                    weight_to_op[weight_name] = f"header:{op}"

    # Body weights (run per layer)
    for layer_idx in range(num_layers):
        for op in body_ops:
            weight_refs = TEMPLATE_OP_WEIGHTS.get(op, [])
            for ref in weight_refs:
                patterns = WEIGHT_PATTERNS.get(ref, [ref])
                for pattern in patterns:
                    weight_name = pattern.replace("{L}", str(layer_idx))
                    if weight_name in all_weights:
                        expected_weights.add(weight_name)
                        weight_to_op[weight_name] = f"layer.{layer_idx}:{op}"

    # Footer weights (run once)
    for op in footer_ops:
        weight_refs = TEMPLATE_OP_WEIGHTS.get(op, [])
        for ref in weight_refs:
            patterns = WEIGHT_PATTERNS.get(ref, [ref])
            for pattern in patterns:
                weight_name = pattern
                if weight_name in all_weights:
                    expected_weights.add(weight_name)
                    weight_to_op[weight_name] = f"footer:{op}"

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Extract weights from IR Lower 1 weights field
    # ═══════════════════════════════════════════════════════════

    # IR Lower 1 ops preserve the original IR1 weights field
    ir1_used_weights = set()

    for ir_op in ir_lower_1_ops:
        weights = ir_op.get("weights", {})
        for wkey, winfo in weights.items():
            if isinstance(winfo, dict) and "name" in winfo:
                ir1_used_weights.add(winfo["name"])

    # ═══════════════════════════════════════════════════════════
    # STEP 4: VALIDATION - Check weight coverage
    # ═══════════════════════════════════════════════════════════

    all_weight_names = set(all_weights.keys())

    # Weights in manifest that are NOT model weights (tokenizer data)
    non_model_weights = {"vocab_offsets", "vocab_strings", "vocab_merges"}
    model_weights = all_weight_names - non_model_weights

    # Weights expected but not used by IR1
    unused_by_ir1 = expected_weights - ir1_used_weights

    # Weights in manifest but not used at all
    completely_unused = model_weights - expected_weights - ir1_used_weights

    coverage = len(ir1_used_weights) / len(model_weights) * 100 if model_weights else 0

    print(f"\n🔍 Weight validation:")
    print(f"  Model weights in manifest: {len(model_weights)}")
    print(f"  Expected by template: {len(expected_weights)}")
    print(f"  Used by IR1 kernels: {len(ir1_used_weights)}")
    print(f"  Coverage: {coverage:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Report unused weights (potential bugs)
    # ═══════════════════════════════════════════════════════════

    validation_errors = []

    if unused_by_ir1:
        print(f"\n⚠️  WEIGHTS EXPECTED BUT NOT USED BY IR1 ({len(unused_by_ir1)}):")

        # Categorize unused weights
        header_unused = [w for w in unused_by_ir1 if weight_to_op.get(w, "").startswith("header:")]
        footer_unused = [w for w in unused_by_ir1 if weight_to_op.get(w, "").startswith("footer:")]
        body_unused = [w for w in unused_by_ir1 if "layer." in w]

        if header_unused:
            print(f"\n   Header weights (not processed by IR1):")
            for w in sorted(header_unused)[:10]:
                print(f"     - {w} (used by {weight_to_op.get(w, 'unknown')})")
            if len(header_unused) > 10:
                print(f"     ... and {len(header_unused) - 10} more")

            validation_errors.append(
                f"Header weights not used: {len(header_unused)} weights\n"
                f"   FIX: Add header ops to IR1 generation (tokenizer, embedding)"
            )

        if footer_unused:
            print(f"\n   Footer weights (not processed by IR1):")
            for w in sorted(footer_unused)[:10]:
                print(f"     - {w} (used by {weight_to_op.get(w, 'unknown')})")
            if len(footer_unused) > 10:
                print(f"     ... and {len(footer_unused) - 10} more")

            validation_errors.append(
                f"Footer weights not used: {len(footer_unused)} weights\n"
                f"   FIX: Add footer ops to IR1 generation (final_norm, logits)"
            )

        if body_unused:
            print(f"\n   Body weights (not processed by IR1):")
            for w in sorted(body_unused)[:10]:
                print(f"     - {w} (used by {weight_to_op.get(w, 'unknown')})")
            if len(body_unused) > 10:
                print(f"     ... and {len(body_unused) - 10} more")

            validation_errors.append(
                f"Body weights not used: {len(body_unused)} weights\n"
                f"   FIX: Check body ops in IR1 are loading all required weights"
            )

    if completely_unused:
        print(f"\n⚠️  WEIGHTS IN MANIFEST BUT NOT IN TEMPLATE ({len(completely_unused)}):")
        for w in sorted(completely_unused)[:10]:
            print(f"     - {w}")
        if len(completely_unused) > 10:
            print(f"     ... and {len(completely_unused) - 10} more")

        validation_errors.append(
            f"Weights not mapped to any template op: {len(completely_unused)} weights\n"
            f"   FIX: Add TEMPLATE_OP_WEIGHTS mapping for these weight types"
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 6: HARD FAULT if validation fails
    # ═══════════════════════════════════════════════════════════

    if validation_errors:
        print(f"\n{'='*60}")
        print("❌ HARD FAULT: WEIGHT VALIDATION FAILED")
        print(f"{'='*60}")

        for i, err in enumerate(validation_errors, 1):
            print(f"\n[{i}] {err}")

        print(f"\n" + "="*60)
        print("WHY THIS MATTERS:")
        print("  - Unused weights = broken inference (wrong output)")
        print("  - For backprop: gradients will be ZERO for these weights")
        print("  - Model will not learn/work correctly")
        print(f"{'='*60}")

        raise RuntimeError(
            f"Weight validation failed: {len(validation_errors)} issues found.\n"
            f"Fix the issues above before proceeding."
        )

    print(f"\n✅ All {len(model_weights)} model weights are mapped!")

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Plan activation buffers from config + context_len
    # ═══════════════════════════════════════════════════════════

    embed_dim = config.get("embed_dim", 896)
    num_heads = config.get("num_heads", 14)
    num_kv_heads = config.get("num_kv_heads", 2)
    head_dim = config.get("head_dim", 64)
    intermediate_size = config.get("intermediate_size", 4864)
    vocab_size = config.get("vocab_size", 151936)

    # Use provided context_len or default from config
    max_context = config.get("context_length", 32768)
    if context_len is None:
        context_len = max_context
    else:
        context_len = min(context_len, max_context)

    # For decode mode, we process 1 token but need full KV cache
    if mode == "decode":
        seq_len = 1  # tokens per forward pass
    else:
        seq_len = context_len  # prefill processes all tokens

    print(f"\n📊 Activation memory planning:")
    print(f"  Mode: {mode}")
    print(f"  Context length: {context_len}")
    print(f"  Sequence length (per pass): {seq_len}")

    # Calculate buffer sizes
    activation_buffers = []
    current_offset = 0

    def add_buffer(name, size, shape_desc, dtype="fp32"):
        nonlocal current_offset
        activation_buffers.append({
            "name": name,
            "size": size,
            "offset": current_offset,
            "shape": shape_desc,
            "dtype": dtype
        })
        current_offset += size

    # ─────────────────────────────────────────────────────────────
    # HEADER buffers: tokenizer → embedding
    # ─────────────────────────────────────────────────────────────

    # Text input buffer: UTF-8 bytes (estimate 4 bytes per token avg)
    # For decode mode, only need 1 token; for prefill, need full context
    max_input_bytes = seq_len * 16  # conservative estimate (avg token ~4 bytes, pad for unicode)
    add_buffer("text_input", max_input_bytes, f"[{max_input_bytes}]", "u8")

    # Token IDs buffer: tokenizer output [seq_len] as int32
    token_ids_size = seq_len * 4  # int32
    add_buffer("token_ids", token_ids_size, f"[{seq_len}]", "i32")

    # Embedded input: embedding lookup output [seq_len, embed_dim]
    # For decode: [1, embed_dim], for prefill: [context_len, embed_dim]
    embedded_size = seq_len * embed_dim * 4
    add_buffer("embedded_input", embedded_size, f"[{seq_len}, {embed_dim}]")

    # Layer input buffer (for ping-pong)
    # Must be large enough for Q8_K quantization of MLP intermediate (n_ff elements)
    # Q8_K uses 272 bytes per 256 elements: ceil(n_ff/256) * 272 * seq_len
    q8k_blocks = (intermediate_size + 255) // 256
    q8k_size = q8k_blocks * 272 * seq_len
    layer_input_size = max(embedded_size, q8k_size)
    add_buffer("layer_input", layer_input_size, f"[{seq_len}, max({embed_dim}, Q8_K({intermediate_size}))]")

    # Residual buffer (for residual connections - stores input before layer processing)
    add_buffer("residual", embedded_size, f"[{seq_len}, {embed_dim}]")

    # ─────────────────────────────────────────────────────────────
    # BODY buffers: KV cache + RoPE (shared across all layers)
    # ─────────────────────────────────────────────────────────────

    # KV cache: [num_layers, 2, num_kv_heads, context_len, head_dim]
    # Stores K and V for all layers, indexed by position
    kv_per_layer = num_kv_heads * context_len * head_dim * 4
    total_kv_size = num_layers * 2 * kv_per_layer
    add_buffer("kv_cache", total_kv_size, f"[{num_layers}, 2, {num_kv_heads}, {context_len}, {head_dim}]")

    # RoPE tables: precomputed cos/sin [2, context_len, head_dim/2]
    rope_size = context_len * (head_dim // 2) * 4 * 2
    add_buffer("rope_cache", rope_size, f"[2, {context_len}, {head_dim // 2}]")

    # Layer scratch buffers (reused across layers)
    # Q output: [num_heads, seq_len, head_dim]
    q_size = num_heads * seq_len * head_dim * 4
    add_buffer("q_scratch", q_size, f"[{num_heads}, {seq_len}, {head_dim}]")

    # K output: [num_kv_heads, seq_len, head_dim]
    k_size = num_kv_heads * seq_len * head_dim * 4
    add_buffer("k_scratch", k_size, f"[{num_kv_heads}, {seq_len}, {head_dim}]")

    # V output: [num_kv_heads, seq_len, head_dim]
    v_size = num_kv_heads * seq_len * head_dim * 4
    add_buffer("v_scratch", v_size, f"[{num_kv_heads}, {seq_len}, {head_dim}]")

    # Attention output: [num_heads, seq_len, head_dim]
    attn_out_size = num_heads * seq_len * head_dim * 4
    add_buffer("attn_scratch", attn_out_size, f"[{num_heads}, {seq_len}, {head_dim}]")

    # MLP scratch: [seq_len, intermediate_size * 2]
    mlp_size = seq_len * intermediate_size * 2 * 4
    # Fused attention scratch needs more space (Q, attn_out, proj, qkv_scratch)
    # Formula: 3 * num_heads * seq_len * head_dim * 4 + qkv_scratch (embed_dim * 4 * tokens + overhead)
    # For safety, use at least 350KB for decode fused attention
    fused_attn_scratch = max(350 * 1024, 3 * num_heads * seq_len * head_dim * 4 + embed_dim * 4 * seq_len * 4)
    scratch_size = max(mlp_size, fused_attn_scratch)
    add_buffer("mlp_scratch", scratch_size, f"[max({seq_len}*{intermediate_size*2}, fused_attn)]")

    # Layer output: [seq_len, embed_dim]
    layer_out_size = seq_len * embed_dim * 4
    add_buffer("layer_output", layer_out_size, f"[{seq_len}, {embed_dim}]")

    # ─────────────────────────────────────────────────────────────
    # FOOTER buffers: final output
    # ─────────────────────────────────────────────────────────────

    # Logits: [seq_len, vocab_size] - for decode just 1 token
    logits_seq = 1 if mode == "decode" else seq_len
    logits_size = logits_seq * vocab_size * 4
    add_buffer("logits", logits_size, f"[{logits_seq}, {vocab_size}]")

    total_activation_size = current_offset

    print(f"\n  Buffer breakdown:")
    for buf in activation_buffers:
        size_mb = buf["size"] / (1024 * 1024)
        print(f"    {buf['name']:<20} {buf['shape']:<40} {size_mb:>8.2f} MB")
    print(f"  {'─' * 70}")
    print(f"  {'Total':<20} {'':<40} {total_activation_size / (1024 * 1024):>8.2f} MB")

    # ═══════════════════════════════════════════════════════════
    # STEP 8: Build final layout
    # ═══════════════════════════════════════════════════════════

    # Build weight layout with relative offsets (relative to weights_base_offset)
    weight_layout = []
    for name in sorted(all_weights.keys()):
        w = all_weights[name]
        rel_off = w["relative_offset"]
        abs_off = weights_base_offset + rel_off
        weight_layout.append({
            "name": name,
            "dtype": w["dtype"],
            "size": w["size"],
            "offset": rel_off,  # Offset relative to weights_base_offset
            "abs_offset": abs_off,  # Offset relative to bump base (absolute)
            "define": f"W_{_sanitize_macro(name)}",
        })

    # Add context_len to config for codegen
    layout_config = dict(config)
    if context_len is not None:
        layout_config["context_length"] = context_len
        layout_config["context_len"] = context_len
    elif "context_length" not in layout_config:
        layout_config["context_length"] = layout_config.get("max_seq_len", 32768)

    # Get bump_layout from manifest (written by converter)
    # This ensures codegen uses the same offsets as the converter
    bump_layout = manifest.get("bump_layout", {
        # Defaults if manifest doesn't have bump_layout (backward compat)
        "header_size": 128,
        "ext_metadata_size": 24,
        "data_start": 152,
        "description": "Offsets: [0..header_size) header, [header_size..data_start) ext_metadata, [data_start..] dtype_table + weights"
    })

    activations_base = weights_base_offset + total_weight_size
    for buf in activation_buffers:
        buf["define"] = f"A_{_sanitize_macro(buf.get('name', 'buffer'))}"
        buf["abs_offset"] = activations_base + buf.get("offset", 0)

    total_size = activations_base + total_activation_size
    total_size = _align_up(total_size, 64)

    layout = {
        "format": "memory-layout-v6.6",
        "version": 2,
        "mode": mode,
        "config": layout_config,
        # BUMP file layout constants - passed from converter via manifest
        # All downstream (codegen, C runtime) should use these, NOT hardcoded values
        "bump_layout": bump_layout,
        # Note: operations are NOT included here - use generate_ir_lower_2 for lowered ops with offsets
        "validation": {
            "status": "PASS",
            "total_weights": len(model_weights),
            "used_weights": len(ir1_used_weights),
            "coverage_percent": coverage,
        },
        "memory": {
            "weights": {
                "size": total_weight_size,
                "bump_size": total_weight_size,
                "base_offset": weights_base_offset,  # File offset where weights start
                "entries": weight_layout,
            },
            "activations": {
                "size": total_activation_size,
                "buffers": activation_buffers,
            },
            "arena": {
                "mode": "region",
                "weights_base": weights_base_offset,
                "activations_base": activations_base,
                "total_size": total_size,
            },
        },
    }

    print(f"\n✓ Memory layout complete")
    print(f"  Bump (weights): {total_weight_size / 1024 / 1024:.1f} MB")
    print(f"  Activations: {total_activation_size / 1024:.1f} KB")

    return layout


def generate_memory_layout_packed(
    ir_lower_1_ops: List[Dict],
    manifest: Dict,
    registry: Dict,
    mode: str,
    context_len: int = None,
    layer_limit: Optional[int] = None,
) -> Dict:
    """Generate a packed/streamed layout where weights + activations share one arena."""
    print(f"\n{'='*60}")
    print("MEMORY PLANNER (packed/streamed)")
    print(f"{'='*60}")

    config = dict(manifest.get("config", {}))
    if layer_limit:
        config["num_layers"] = int(layer_limit)

    entries = manifest.get("entries", [])
    if not entries:
        raise RuntimeError("Manifest entries missing; cannot build packed layout.")
    entry_by_name = {e["name"]: e for e in entries}

    # Validate: every used weight exists in manifest
    used_weights = set()
    for op in ir_lower_1_ops:
        for w in op.get("weights", {}).values():
            if isinstance(w, dict) and "name" in w:
                used_weights.add(w["name"])
    missing = [w for w in sorted(used_weights) if w not in entry_by_name]
    if missing:
        raise RuntimeError(f"Packed layout: missing {len(missing)} weights in manifest: {missing[:5]}")

    act_specs = build_activation_specs(config, mode, context_len, num_layers_override=layer_limit)

    weight_offset = 0
    act_offset = 0
    weight_layout = []
    weight_offsets = {}
    activation_buffers = []
    act_offsets = {}

    weight_order = []
    seen_weights = set()
    for op in ir_lower_1_ops:
        for w in op.get("weights", {}).values():
            if not isinstance(w, dict):
                continue
            name = w.get("name")
            if not name or name in seen_weights:
                continue
            seen_weights.add(name)
            weight_order.append(name)

    def alloc_weight(name: str) -> None:
        nonlocal weight_offset
        if name in weight_offsets:
            return
        entry = entry_by_name[name]
        size = int(entry.get("size", entry.get("size_bytes", 0)))
        off = _align_up(weight_offset, 64)
        weight_offsets[name] = off
        weight_layout.append({
            "name": name,
            "dtype": entry.get("dtype", "unknown"),
            "size": size,
            "offset": off,
            "abs_offset": off,
            "file_offset": entry.get("file_offset", 0),
            "define": f"W_{_sanitize_macro(name)}",
        })
        weight_offset = off + size

    def alloc_act(name: str) -> None:
        nonlocal act_offset
        if name in act_offsets:
            return
        spec = act_specs.get(name)
        if not spec:
            return
        off = _align_up(act_offset, 64)
        act_offsets[name] = off
        activation_buffers.append({
            "name": name,
            "size": spec["size"],
            "offset": off,
            "abs_offset": off,
            "shape": spec["shape"],
            "dtype": spec["dtype"],
            "define": f"A_{_sanitize_macro(name)}",
        })
        act_offset = off + spec["size"]

    # Allocate all weights first (in order of first use)
    for name in weight_order:
        alloc_weight(name)

    weights_end = _align_up(weight_offset, 64)
    act_offset = weights_end

    # Simulate op order (using same buffer naming logic as IR Lower 2)
    current_input_buffer = "token_ids"
    current_output_buffer = "embedded_input"
    qkv_input_buffer = "token_ids"

    for op in ir_lower_1_ops:
        op_type = op.get("op", "")
        kernel_type = op.get("kernel", "")

        if op_type in ("q_proj", "k_proj", "v_proj"):
            if op_type == "q_proj":
                qkv_input_buffer = current_input_buffer
            alloc_act(qkv_input_buffer)
            if op_type == "q_proj":
                alloc_act("q_scratch")
            elif op_type == "k_proj":
                alloc_act("k_scratch")
            elif op_type == "v_proj":
                alloc_act("v_scratch")
        elif op_type == "qkv_proj":
            qkv_input_buffer = current_input_buffer
            alloc_act(qkv_input_buffer)
            alloc_act("q_scratch")
            alloc_act("k_scratch")
            alloc_act("v_scratch")
        else:
            if "embedding" in kernel_type.lower():
                alloc_act("token_ids")
                alloc_act("embedded_input")
            elif op_type == "logits":
                alloc_act("logits")
            else:
                alloc_act(current_input_buffer)
                alloc_act(current_output_buffer)

            if op_type == "residual_add":
                alloc_act("residual")

        # Scratch buffers
        if op.get("scratch"):
            alloc_act("mlp_scratch")

        if op_type == "rope_qk":
            alloc_act("q_scratch")
            alloc_act("k_scratch")
            alloc_act("rope_cache")

        if op_type == "kv_cache_store":
            alloc_act("k_scratch")
            alloc_act("v_scratch")
            alloc_act("kv_cache")

        if op_type == "attn" or "attention" in kernel_type:
            alloc_act("q_scratch")
            alloc_act("k_scratch")
            alloc_act("v_scratch")
            alloc_act("attn_scratch")

        if op_type == "residual_add":
            alloc_act("residual")

        if op.get("section") == "body" and op_type == "rmsnorm":
            alloc_act("layer_input")
            alloc_act("residual")

        # Ping-pong update (same as generate_ir_lower_2)
        if "embedding" in kernel_type.lower():
            current_input_buffer = "embedded_input"
            current_output_buffer = "layer_input"
        elif op_type in ("q_proj", "k_proj", "v_proj", "qkv_proj", "rope_qk", "bias_add") or \
                (mode == "prefill" and op_type == "attn"):
            pass
        else:
            current_input_buffer, current_output_buffer = current_output_buffer, current_input_buffer

    # Ensure required buffers exist for runtime pointers
    alloc_act("kv_cache")
    alloc_act("rope_cache")
    alloc_act("logits")

    total_weight_bytes = weights_end
    total_activation_bytes = act_offset - weights_end

    layout_config = dict(config)
    if context_len is not None:
        layout_config["context_length"] = context_len
        layout_config["context_len"] = context_len
    elif "context_length" not in layout_config:
        layout_config["context_length"] = layout_config.get("max_seq_len", 32768)

    layout = {
        "format": "memory-layout-v6.6",
        "version": 3,
        "mode": mode,
        "config": layout_config,
        "bump_layout": manifest.get("bump_layout", {
            "header_size": 128,
            "ext_metadata_size": 24,
            "data_start": 152,
        }),
        "memory": {
            "weights": {
                "size": total_weight_bytes,
                "bump_size": total_weight_bytes,
                "base_offset": 0,
                "entries": weight_layout,
            },
            "activations": {
                "size": total_activation_bytes,
                "buffers": activation_buffers,
            },
            "arena": {
                "mode": "packed",
                "weights_base": 0,
                "activations_base": weights_end,
                "total_size": total_size,
            },
        },
    }

    print(f"\n✓ Packed layout complete")
    print(f"  Total arena: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Weights (used): {sum(e.get('size', 0) for e in weight_layout) / 1024 / 1024:.1f} MB")
    print(f"  Activations (allocated): {total_activation_bytes / 1024 / 1024:.1f} MB")

    return layout


def write_manifest_map(layout: Dict, manifest: Dict, output_path: Path) -> None:
    """Write weights_manifest.map with runtime offsets from layout."""
    weights = layout.get("memory", {}).get("weights", {}).get("entries", [])
    rt_by_name = {e["name"]: int(e.get("abs_offset", e.get("offset", 0))) for e in weights}
    # Preserve ordering by runtime offset (stream-friendly)
    ordered = sorted(weights, key=lambda e: int(e.get("abs_offset", e.get("offset", 0))))

    entry_by_name = {e["name"]: e for e in manifest.get("entries", [])}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("# ck-bumpwgt5-manifest-map v1\n")
        f.write("# name|dtype|file_offset|size|runtime_offset\n")
        for w in ordered:
            name = w["name"]
            m = entry_by_name.get(name)
            if not m:
                continue
            file_off = int(m.get("file_offset", 0))
            size = int(m.get("size", m.get("size_bytes", 0)))
            dtype = m.get("dtype", w.get("dtype", "unknown"))
            rt_off = rt_by_name.get(name, 0)
            f.write(f"{name}|{dtype}|0x{file_off:016X}|0x{size:016X}|0x{rt_off:016X}\n")


def generate_ir_lower_2(
    ir_lower_1_ops: List[Dict],
    layout: Dict,
    manifest: Dict,
    registry: Dict,
    mode: str
) -> Dict:
    """
    IR Lower 2: Add concrete memory offsets to IR Lower 1 ops.

    IR Lower 1 already has:
    - inputs: {type='weight', name, offset, size} or {type='activation', source}
    - outputs: {buffer, dtype, shape}
    - scratch: [{name, size, dtype}]

    This function adds:
    - Concrete bump_offset for each weight input
    - Concrete activation_offset for each activation input/output
    - Pointer expressions for codegen

    Args:
        ir_lower_1_ops: IR Lower 1 ops with inputs/outputs/scratch
        layout: Memory layout with weight offsets and activation buffers
        manifest: Model manifest
        registry: Kernel registry
        mode: Execution mode

    Returns:
        Final lowered IR with explicit pointer expressions
    """
    print(f"\n{'='*60}")
    print("IR LOWER 2 (Add memory offsets)")
    print(f"{'='*60}")

    config = manifest.get("config", {})

    # Build weight offset lookup from layout
    weight_offsets = {}
    memory = layout.get("memory", {})
    weight_entries = memory.get("weights", {}).get("entries", [])
    for entry in weight_entries:
        weight_offsets[entry["name"]] = {
            "bump_offset": entry["offset"],
            "dtype": entry["dtype"],
            "size": entry["size"],
        }

    # Activation buffer lookup
    activation_buffers = {}
    for buf in memory.get("activations", {}).get("buffers", []):
        activation_buffers[buf["name"]] = buf

    # KV cache slice helper (prefill path may write directly into KV cache)
    layout_config = layout.get("config", {}) if isinstance(layout, dict) else {}
    if layout_config:
        merged = dict(config)
        merged.update(layout_config)
        config = merged
    context_len = config.get("context_length", config.get("max_seq_len", config.get("context_len", 0)))
    num_kv_heads = config.get("num_kv_heads", 0)
    head_dim = config.get("head_dim", 0)

    def kv_layer_offsets(layer: int) -> Optional[Tuple[int, int]]:
        kv_buf = activation_buffers.get("kv_cache")
        if not kv_buf or not context_len or not num_kv_heads or not head_dim:
            return None
        kv_per_layer = num_kv_heads * context_len * head_dim * 4
        base = kv_buf["offset"] + layer * 2 * kv_per_layer
        return base, base + kv_per_layer

    lowered_ops = []

    # ═══════════════════════════════════════════════════════════════════════════════
    # MEMORY PLANNER: Pre-compute buffer assignments based on dataflow
    # This replaces the old ping-pong logic with explicit dataflow-based assignment
    # ═══════════════════════════════════════════════════════════════════════════════
    print("  Running memory planner...")
    buffer_assignments = plan_memory(ir_lower_1_ops)
    print(f"  ✓ Memory planner assigned buffers for {len(buffer_assignments)} ops")

    # Helper to get buffer info from memory planner
    def get_planned_buffer(op_id: int, io_type: str, name: str) -> Optional[Dict]:
        """Get buffer assignment from memory planner.

        Args:
            op_id: Operation ID
            io_type: 'inputs' or 'outputs'
            name: Input/output name (e.g., 'x', 'y', 'input', 'output')

        Returns:
            Buffer info dict with 'buffer' and 'dtype' keys, or None
        """
        assignment = buffer_assignments.get(op_id, {})
        io_assignments = assignment.get(io_type, {})
        return io_assignments.get(name)

    # Map buffer names to activation buffer names
    # Memory planner uses A_EMBEDDED_INPUT, A_LAYER_INPUT, etc.
    # Layout uses embedded_input, layer_input, etc.
    buffer_name_map = {
        "A_EMBEDDED_INPUT": "embedded_input",
        "A_LAYER_INPUT": "layer_input",
        "A_RESIDUAL": "residual",
        "A_ATTN_SCRATCH": "attn_scratch",
        "A_MLP_SCRATCH": "mlp_scratch",
        "A_LAYER_OUTPUT": "layer_output",
        "A_LOGITS": "logits",
        "kv_cache": "kv_cache",
    }

    # Legacy ping-pong tracking (kept for fallback, but should not be needed)
    current_input_buffer = "token_ids"
    current_output_buffer = "embedded_input"
    qkv_input_buffer = "token_ids"
    last_output_buffer: Optional[str] = None

    for ir_op in ir_lower_1_ops:
        lowered_op = {
            "idx": ir_op["idx"],
            "kernel": ir_op["kernel"],
            "op": ir_op["op"],
            "layer": ir_op["layer"],
            "section": ir_op["section"],
            "function": ir_op.get("function", ir_op["kernel"]),
            "weights": {},
            "activations": {},
            "outputs": {},
            "params": {},
        }

        # Process weights - add concrete bump offsets
        for wkey, winfo in ir_op.get("weights", {}).items():
            weight_name = winfo.get("name", "")
            weight_entry = weight_offsets.get(weight_name)

            if weight_entry:
                lowered_op["weights"][wkey] = {
                    "name": weight_name,
                    "bump_offset": weight_entry["bump_offset"],
                    "size": weight_entry["size"],
                    "dtype": weight_entry["dtype"],
                    "ptr_expr": f"bump_weights + {weight_entry['bump_offset']}",
                }
            else:
                # Weight not in layout - use file offset from IR1
                lowered_op["weights"][wkey] = {
                    "name": weight_name,
                    "bump_offset": winfo.get("offset", 0),
                    "size": winfo.get("size", 0),
                    "dtype": winfo.get("dtype", "unknown"),
                    "ptr_expr": f"bump_weights + {winfo.get('offset', 0)}",
                }

        # Special handling for Q/K/V projections: all read from same input, write to different outputs
        op_type = ir_op.get("op", "")
        if op_type == "bias_add":
            bias_for = ir_op.get("bias_for")
            target_buf = None
            if bias_for in ("q_proj", "k_proj", "v_proj"):
                target_buf = {
                    "q_proj": "q_scratch",
                    "k_proj": "k_scratch",
                    "v_proj": "v_scratch",
                }.get(bias_for)
            else:
                target_buf = last_output_buffer or current_output_buffer
            buf = activation_buffers.get(target_buf) if target_buf else None
            if buf:
                lowered_op["activations"]["y"] = {
                    "buffer": target_buf,
                    "activation_offset": buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {buf['offset']}",
                }
                lowered_op["outputs"]["y"] = {
                    "buffer": target_buf,
                    "activation_offset": buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {buf['offset']}",
                }
            if target_buf:
                last_output_buffer = target_buf
        elif op_type == "qkv_proj":
            # QKV fused projection (prefill uses head-major outputs)
            buf = activation_buffers.get(current_input_buffer)
            if buf:
                for input_name, input_info in ir_op.get("inputs", {}).items():
                    lowered_op["activations"][input_name] = {
                        "buffer": current_input_buffer,
                        "activation_offset": buf["offset"],
                        "dtype": input_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset']}",
                    }

            layer_idx = int(ir_op.get("layer", 0))
            kv_offs = kv_layer_offsets(layer_idx) if mode == "prefill" else None
            k_off = kv_offs[0] if kv_offs else None
            v_off = kv_offs[1] if kv_offs else None

            for output_name, output_info in ir_op.get("outputs", {}).items():
                if output_name.startswith("q"):
                    q_buf = activation_buffers.get("q_scratch")
                    lowered_op["outputs"][output_name] = {
                        "buffer": "q_scratch",
                        "activation_offset": q_buf["offset"] if q_buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {q_buf['offset'] if q_buf else 0}",
                    }
                    last_output_buffer = "q_scratch"
                elif output_name.startswith("k"):
                    if k_off is not None:
                        lowered_op["outputs"][output_name] = {
                            "buffer": f"kv_cache_k_L{layer_idx}",
                            "activation_offset": k_off,
                            "dtype": output_info.get("dtype", "fp32"),
                            "ptr_expr": f"activations + {k_off}",
                        }
                    else:
                        k_buf = activation_buffers.get("k_scratch")
                        lowered_op["outputs"][output_name] = {
                            "buffer": "k_scratch",
                            "activation_offset": k_buf["offset"] if k_buf else 0,
                            "dtype": output_info.get("dtype", "fp32"),
                            "ptr_expr": f"activations + {k_buf['offset'] if k_buf else 0}",
                        }
                        last_output_buffer = "k_scratch"
                elif output_name.startswith("v"):
                    if v_off is not None:
                        lowered_op["outputs"][output_name] = {
                            "buffer": f"kv_cache_v_L{layer_idx}",
                            "activation_offset": v_off,
                            "dtype": output_info.get("dtype", "fp32"),
                            "ptr_expr": f"activations + {v_off}",
                        }
                    else:
                        v_buf = activation_buffers.get("v_scratch")
                        lowered_op["outputs"][output_name] = {
                            "buffer": "v_scratch",
                            "activation_offset": v_buf["offset"] if v_buf else 0,
                            "dtype": output_info.get("dtype", "fp32"),
                            "ptr_expr": f"activations + {v_buf['offset'] if v_buf else 0}",
                        }
                        last_output_buffer = "v_scratch"
        elif op_type in ("q_proj", "k_proj", "v_proj"):
            # ═══════════════════════════════════════════════════════════════
            # USE MEMORY PLANNER for QKV input buffer assignment
            # The memory planner knows the correct buffer (main_stream_q8)
            # ═══════════════════════════════════════════════════════════════
            op_id = ir_op.get("idx", ir_op.get("op_id", -1))

            for input_name, input_info in ir_op.get("inputs", {}).items():
                # Skip weight inputs
                if input_name in ir_op.get("weights", {}):
                    continue

                # Use memory planner for input buffer
                planned = get_planned_buffer(op_id, "inputs", input_name)

                if planned:
                    planner_buf = planned.get("buffer", "layer_input")
                    buf_name = buffer_name_map.get(planner_buf, planner_buf)
                    buf = activation_buffers.get(buf_name)
                else:
                    # Fallback to layer_input (Q8 buffer) - this is where quantize_input writes
                    buf_name = "layer_input"
                    buf = activation_buffers.get("layer_input")

                if buf:
                    lowered_op["activations"][input_name] = {
                        "buffer": buf_name,
                        "activation_offset": buf["offset"],
                        "dtype": input_info.get("dtype", "q8_0"),
                        "ptr_expr": f"activations + {buf['offset']}",
                    }
            # Q writes to q_scratch
            if op_type == "q_proj":
                q_buf = activation_buffers.get("q_scratch")
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    lowered_op["outputs"][output_name] = {
                        "buffer": "q_scratch",
                        "activation_offset": q_buf["offset"] if q_buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {q_buf['offset'] if q_buf else 0}",
                    }
            # K/V write to their respective scratch buffers
            elif op_type == "k_proj":
                buf = activation_buffers.get("k_scratch")
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    lowered_op["outputs"][output_name] = {
                        "buffer": "k_scratch",
                        "activation_offset": buf["offset"] if buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset'] if buf else 0}",
                    }
            elif op_type == "v_proj":
                buf = activation_buffers.get("v_scratch")
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    lowered_op["outputs"][output_name] = {
                        "buffer": "v_scratch",
                        "activation_offset": buf["offset"] if buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset'] if buf else 0}",
                    }
            if op_type == "q_proj":
                last_output_buffer = "q_scratch"
            elif op_type == "k_proj":
                last_output_buffer = "k_scratch"
            elif op_type == "v_proj":
                last_output_buffer = "v_scratch"
        else:
            # Process activation inputs - add concrete buffer offsets
            for input_name, input_info in ir_op.get("inputs", {}).items():
                # Special case: embedding operation reads from token_ids, not layer_input
                if "embedding" in ir_op.get("kernel", "").lower() and "token" in input_name.lower():
                    buf = activation_buffers.get("token_ids")
                    if buf:
                        lowered_op["activations"][input_name] = {
                            "buffer": "token_ids",
                            "activation_offset": buf["offset"],
                            "dtype": "int32",
                            "ptr_expr": f"activations + {buf['offset']}",
                        }
                        continue

                # Skip inputs that are actually weight parameters
                # Check both direct match and mapped match via WEIGHT_TO_KERNEL_INPUT
                # e.g., gamma maps to ln1_gamma/ln2_gamma, W maps to wq/wk/wv
                is_weight_input = input_name in ir_op.get("weights", {})
                if not is_weight_input:
                    # Check if any weight key maps to this input name
                    for wkey in ir_op.get("weights", {}).keys():
                        if WEIGHT_TO_KERNEL_INPUT.get(wkey) == input_name:
                            is_weight_input = True
                            break
                if is_weight_input:
                    continue  # Weight is handled via weights dict

                # ═══════════════════════════════════════════════════════════════
                # USE MEMORY PLANNER for buffer assignment
                # ═══════════════════════════════════════════════════════════════
                op_id = ir_op.get("idx", ir_op.get("op_id", -1))

                # Map from kernel input names to dataflow names
                # Kernel maps use: A (input), B (weight), C (output)
                # Dataflow uses: x (input), y (output)
                kernel_to_dataflow_input = {
                    "A": "x",      # Matrix input for gemm/gemv
                    "x": "x",      # Direct match
                    "input": "x",  # Alternative name
                    "a": "a",      # residual_add input a
                    "b": "b",      # residual_add input b
                    "src": "src",  # memcpy source
                    "gate": "x",   # swiglu gate input -> reads from mlp_scratch
                    "up": "x",     # swiglu up input -> reads from mlp_scratch
                }
                dataflow_name = kernel_to_dataflow_input.get(input_name, input_name)
                planned = get_planned_buffer(op_id, "inputs", dataflow_name)
                # Also try the original name if mapping didn't find it
                if not planned:
                    planned = get_planned_buffer(op_id, "inputs", input_name)

                if planned:
                    # Use memory planner's assignment
                    planner_buf = planned.get("buffer", "embedded_input")
                    buf_name = buffer_name_map.get(planner_buf, planner_buf)
                    buf = activation_buffers.get(buf_name)
                else:
                    # Fallback to legacy logic for unplanned ops
                    if input_name == "attn_out":
                        buf = activation_buffers.get("attn_scratch")
                        buf_name = "attn_scratch"
                    elif input_name == "scratch":
                        buf = activation_buffers.get("mlp_scratch")
                        buf_name = "mlp_scratch"
                    else:
                        buf = activation_buffers.get(current_input_buffer)
                        buf_name = current_input_buffer

                if buf:
                    lowered_op["activations"][input_name] = {
                        "buffer": buf_name,
                        "activation_offset": buf["offset"],
                        "dtype": input_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset']}",
                    }

            # Process outputs - add concrete offsets (for non-QKV ops)
            for output_name, output_info in ir_op.get("outputs", {}).items():
                # ═══════════════════════════════════════════════════════════════
                # USE MEMORY PLANNER for output buffer assignment
                # ═══════════════════════════════════════════════════════════════
                op_id = ir_op.get("idx", ir_op.get("op_id", -1))

                # Map from kernel output names to dataflow names
                kernel_to_dataflow_output = {
                    "C": "y",       # Matrix output for gemm/gemv
                    "y": "y",       # Direct match
                    "output": "output",  # Quantize output
                    "dst": "dst",   # memcpy destination
                }
                dataflow_name = kernel_to_dataflow_output.get(output_name, output_name)
                planned = get_planned_buffer(op_id, "outputs", dataflow_name)
                # Also try the original name if mapping didn't find it
                if not planned:
                    planned = get_planned_buffer(op_id, "outputs", output_name)

                if planned:
                    # Use memory planner's assignment
                    planner_buf = planned.get("buffer", "embedded_input")
                    output_buf_name = buffer_name_map.get(planner_buf, planner_buf)
                else:
                    # Fallback to legacy logic for unplanned ops
                    if "embedding" in ir_op.get("kernel", "").lower():
                        output_buf_name = "embedded_input"
                    elif ir_op.get("op") == "attn" and mode == "prefill":
                        output_buf_name = "attn_scratch"
                    elif ir_op.get("op") == "logits":
                        output_buf_name = "logits"
                    elif ir_op.get("op") in ("mlp_gate_up", "silu_mul"):
                        output_buf_name = "mlp_scratch"
                    else:
                        output_buf_name = current_output_buffer

                buf = activation_buffers.get(output_buf_name)
                if buf:
                    lowered_op["outputs"][output_name] = {
                        "buffer": output_buf_name,
                        "activation_offset": buf["offset"],
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset']}",
                    }
                    if not last_output_buffer:
                        last_output_buffer = output_buf_name
            if lowered_op["outputs"]:
                last_output_buffer = next(iter(lowered_op["outputs"].values())).get("buffer", last_output_buffer)

        # Process scratch buffers
        lowered_op["scratch"] = []
        scratch_list = ir_op.get("scratch", [])
        if scratch_list:
            mlp_buf = activation_buffers.get("mlp_scratch")
            for i, scratch in enumerate(scratch_list):
                scratch_offset = mlp_buf["offset"] if mlp_buf else 0
                lowered_op["scratch"].append({
                    "name": scratch.get("name", f"scratch_{i}"),
                    "scratch_offset": scratch_offset,
                    "size": scratch.get("size", "dynamic"),
                    "dtype": scratch.get("dtype", "fp32"),
                    "ptr_expr": f"activations + {scratch_offset}",
                })

        # Special handling for RoPE: add q_scratch and k_scratch buffers
        # RoPE always uses the scratch buffers (where k_proj/v_proj just wrote)
        # in both decode and prefill modes
        if ir_op.get("op", "") == "rope_qk":
            q_buf = activation_buffers.get("q_scratch")
            if q_buf:
                lowered_op["scratch"].append({
                    "name": "q_scratch",
                    "scratch_offset": q_buf["offset"],
                    "size": q_buf["size"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {q_buf['offset']}",
                })

            k_buf = activation_buffers.get("k_scratch")
            if k_buf:
                lowered_op["scratch"].append({
                    "name": "k_scratch",
                    "scratch_offset": k_buf["offset"],
                    "size": k_buf["size"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {k_buf['offset']}",
                })

        # Special handling for kv_cache_store: add k_scratch and v_scratch buffers
        if ir_op.get("op", "") == "kv_cache_store":
            for scratch_name in ["k_scratch", "v_scratch"]:
                buf = activation_buffers.get(scratch_name)
                if buf:
                    lowered_op["scratch"].append({
                        "name": scratch_name,
                        "scratch_offset": buf["offset"],
                        "size": buf["size"],
                        "dtype": "fp32",
                        "ptr_expr": f"activations + {buf['offset']}",
                    })

        # Special handling for attention: add q_scratch, k_scratch, v_scratch buffers
        # Note: op type is "attn" but kernel contains "attention"
        if ir_op.get("op", "") == "attn" or "attention" in ir_op.get("kernel", ""):
            for scratch_name in ["q_scratch", "k_scratch", "v_scratch"]:
                # For DECODE mode, use KV cache offsets for K and V (they're read from cache)
                # For PREFILL mode, use scratch buffers (K/V are computed fresh each time)
                if mode == "decode" and scratch_name in ("k_scratch", "v_scratch"):
                    layer_idx = int(ir_op.get("layer", 0))
                    kv_offs = kv_layer_offsets(layer_idx)
                    if kv_offs:
                        k_off, v_off = kv_offs
                        off = k_off if scratch_name == "k_scratch" else v_off
                        lowered_op["scratch"].append({
                            "name": scratch_name,
                            "scratch_offset": off,
                            "size": activation_buffers.get(scratch_name, {}).get("size", 0),
                            "dtype": "fp32",
                            "ptr_expr": f"activations + {off}",
                            "force_offset": True,
                        })
                        continue
                buf = activation_buffers.get(scratch_name)
                if buf:
                    lowered_op["scratch"].append({
                        "name": scratch_name,
                        "scratch_offset": buf["offset"],
                        "size": buf["size"],
                        "dtype": "fp32",
                        "ptr_expr": f"activations + {buf['offset']}",
                    })

        # Special handling for residual_add: add residual buffer for the saved input
        if ir_op.get("op", "") == "residual_add":
            buf = activation_buffers.get("residual")
            if buf:
                lowered_op["scratch"].append({
                    "name": "residual",
                    "scratch_offset": buf["offset"],
                    "size": buf["size"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {buf['offset']}",
                })

        # Add model config parameters (merge with any op-specific params)
        params = dict(ir_op.get("params", {}) or {})
        params.setdefault("embed_dim", config.get("embed_dim", 896))
        params.setdefault("num_heads", config.get("num_heads", 14))
        params.setdefault("num_kv_heads", config.get("num_kv_heads", 2))
        params.setdefault("head_dim", config.get("head_dim", 64))
        params.setdefault("intermediate_size", config.get("intermediate_size", config.get("intermediate_dim", 4864)))
        params.setdefault("num_layers", config.get("num_layers", 24))
        params.setdefault("mode", mode)

        if mode == "decode":
            params.setdefault("seq_len", 1)
        else:
            params.setdefault("seq_len", config.get("max_seq_len", config.get("context_length", 2048)))

        # Add matmul dims for IR Lower 3 bindings (_input_dim/_output_dim/_m)
        out_dim, in_dim = compute_matmul_dims(op_type, config)
        if out_dim is not None and "_output_dim" not in params:
            params["_output_dim"] = out_dim
        if in_dim is not None and "_input_dim" not in params:
            params["_input_dim"] = in_dim

        if op_type == "bias_add" and "_output_dim" not in params:
            for w in lowered_op.get("weights", {}).values():
                size = int(w.get("size", 0))
                if size > 0 and size % 4 == 0:
                    params["_output_dim"] = size // 4
                    break

        params.setdefault("_m", params.get("seq_len", 1))
        lowered_op["params"] = params

        lowered_ops.append(lowered_op)

        # NOTE: residual_save ops are now explicitly in IR1 (inserted before rmsnorm)
        # The memory planner assigns buffers based on dataflow. No need to auto-insert here.

        # Ping-pong buffers for next op, UNLESS this is a Q/K/V projection
        # Q/K/V all read from the same input (RMSNorm output), so skip ping-pong for K/V
        op_type = ir_op.get("op", "")
        kernel_type = ir_op.get("kernel", "")

        if "embedding" in kernel_type.lower():
            # Embedding: reads from token_ids, outputs to embedded_input
            # Next op (RMSNorm/attention) reads from embedded_input, outputs to layer_input
            current_input_buffer = "embedded_input"
            current_output_buffer = "layer_input"
        elif op_type in ("q_proj", "k_proj", "v_proj", "qkv_proj", "rope_qk",
                         "mlp_gate_up", "silu_mul", "mlp_down", "bias_add") or \
                (mode == "prefill" and op_type == "attn"):
            # Ops that don't advance the token-major stream, don't ping-pong
            pass
        else:
            current_input_buffer, current_output_buffer = current_output_buffer, current_input_buffer

    print(f"\n✓ IR Lower 2 complete:")
    print(f"  Lowered ops: {len(lowered_ops)}")
    print(f"  Weight entries resolved: {len(weight_offsets)}")

    lowered_ir = {
        "format": "lowered-ir-v2",
        "version": 2,
        "mode": mode,
        "config": config,
        "memory": memory,
        "operations": lowered_ops,
    }

    return lowered_ir


def load_kernel_bindings() -> Dict[str, Dict]:
    """Load kernel parameter bindings for IR Lower 3."""
    bindings_path = PROJECT_ROOT / "kernel_maps" / "kernel_bindings.json"
    with open(bindings_path, "r") as f:
        data = json.load(f)
    return data.get("bindings", {})


def generate_ir_lower_3(lowered_ir: Dict, mode: str) -> Dict:
    """
    IR Lower 3: Emit call-ready ops with ordered args (function + expr list).
    This removes all semantic ambiguity for codegen.
    """
    bindings = load_kernel_bindings()
    ops = lowered_ir.get("operations", lowered_ir.get("ops", []))
    config = lowered_ir.get("config", {})
    dtype_map = {
        "fp32": "0",
        "bf16": "1",
        "fp16": "2",
        "int8": "3",
        "int4": "4",
        "q4_0": "5",
        "q4_1": "6",
        "q4_k": "7",
        "q6_k": "8",
        "q8_0": "9",
        "q8_k": "10",
        "q5_0": "11",
        "q5_1": "12",
    }

    def ptr_expr(base: str, offset: object, cast: Optional[str]) -> str:
        off = str(offset)
        expr = f"{base} + {off}"
        return f"({cast})({expr})" if cast else expr

    def select_from_dict(name: str, dct: Dict, aliases: Dict[str, List[str]]) -> Optional[Dict]:
        if name in dct:
            return dct[name]
        for alt in aliases.get(name, []):
            if alt in dct:
                return dct[alt]
        if len(dct) == 1:
            return next(iter(dct.values()))
        return None

    def select_weight(name: str, weights: Dict, alt: Optional[List[str]] = None) -> Optional[Tuple[str, Dict]]:
        if name in weights:
            return name, weights[name]
        if alt:
            for a in alt:
                if a in weights:
                    return a, weights[a]
        if name == "_first_weight":
            if weights:
                k = next(iter(weights.keys()))
                return k, weights[k]
        if name == "_bias":
            for k, v in weights.items():
                if k.startswith("b") or "bias" in k:
                    return k, v
        # Try reverse map: kernel input name -> IR weight key
        for k, v in weights.items():
            if WEIGHT_TO_KERNEL_INPUT.get(k) == name:
                return k, v
        return None

    call_ops = []
    all_errors = []

    memory = lowered_ir.get("memory", {})
    arena = memory.get("arena", {})
    weights_base = int(arena.get("weights_base", 0))
    activations_base = int(arena.get("activations_base", 0))
    layout_mode = arena.get("mode", "region")
    weight_define = {e.get("name"): e.get("define") for e in memory.get("weights", {}).get("entries", [])}
    act_define = {b.get("name"): b.get("define") for b in memory.get("activations", {}).get("buffers", [])}
    use_bump_base = bool(arena) or any(weight_define.values()) or any(act_define.values())

    for op in ops:
        func = op.get("function", op.get("kernel", "unknown"))
        binding = bindings.get(func)
        op_errors = []
        op_warnings = []

        if not binding:
            op_errors.append(f"Missing binding for function '{func}'")
            call_ops.append({
                "idx": op.get("idx", -1),
                "function": func,
                "op": op.get("op", ""),
                "layer": op.get("layer", -1),
                "section": op.get("section", ""),
                "args": [],
                "errors": op_errors,
                "warnings": op_warnings,
            })
            all_errors.append({"idx": op.get("idx", -1), "function": func, "error": op_errors[0]})
            continue

        activations = op.get("activations", {})
        outputs = op.get("outputs", {})
        weights = op.get("weights", {})
        scratch_list = op.get("scratch", [])
        scratch = {s.get("name"): s for s in scratch_list if s.get("name")}
        params = op.get("params", {})

        # Aliases for activation/output key lookups (handles case differences between bindings and IR)
        act_aliases = {
            "tokens": ["token_ids", "tokens"],
            "input": ["input", "a", "A", "x", "X"],
            "a": ["A", "a", "input"],  # GEMM uses "A" in IR, "a" in binding
            "x": ["x", "X", "input"],  # GEMV uses "x"
            "src": ["src", "input", "a", "A"],  # memcpy source
        }
        out_aliases = {
            "out": ["output", "out", "C", "c"],
            "c": ["C", "c", "output"],  # GEMM uses "C" in IR, "c" in binding
            "y": ["y", "Y", "output"],  # GEMV output
            "dst": ["dst", "output"],  # memcpy destination
        }

        args = []
        for param in binding.get("params", []):
            src = param.get("source", "")
            name = param.get("name", "")
            cast = param.get("cast")

            if src.startswith("activation:"):
                key = src.split(":", 1)[1]
                info = select_from_dict(key, activations, act_aliases)
                if not info:
                    op_errors.append(f"{func}.{name}: missing activation '{key}'")
                    expr = "NULL"
                else:
                    offset = info.get("activation_offset", 0)
                    buf_name = info.get("buffer", key)
                    macro = act_define.get(buf_name)
                    if use_bump_base:
                        if macro:
                            off_expr = macro
                        else:
                            off_expr = str(activations_base + int(offset))
                        expr = ptr_expr("model->bump", off_expr, cast or "const float*")
                    else:
                        expr = ptr_expr("ACT", offset, cast or "const float*")

            elif src.startswith("output:"):
                key = src.split(":", 1)[1]
                info = select_from_dict(key, outputs, out_aliases)
                if not info:
                    op_errors.append(f"{func}.{name}: missing output '{key}'")
                    expr = "NULL"
                else:
                    offset = info.get("activation_offset", 0)
                    buf_name = info.get("buffer", key)
                    macro = act_define.get(buf_name)
                    if use_bump_base:
                        if macro:
                            off_expr = macro
                        else:
                            off_expr = str(activations_base + int(offset))
                        expr = ptr_expr("model->bump", off_expr, cast or "float*")
                    else:
                        expr = ptr_expr("ACT", offset, cast or "float*")

            elif src.startswith("scratch:"):
                key = src.split(":", 1)[1]
                info = scratch.get(key)
                if not info and len(scratch) == 1:
                    info = next(iter(scratch.values()))
                if not info:
                    op_errors.append(f"{func}.{name}: missing scratch '{key}'")
                    expr = "NULL"
                else:
                    offset = info.get("scratch_offset", 0)
                    buf_name = info.get("name", key)
                    macro = act_define.get(buf_name)
                    if info.get("force_offset"):
                        macro = None
                    if use_bump_base:
                        if macro:
                            off_expr = macro
                        else:
                            off_expr = str(activations_base + int(offset))
                        expr = ptr_expr("model->bump", off_expr, cast or "float*")
                    else:
                        expr = ptr_expr("ACT", offset, cast or "float*")

            elif src.startswith("weight_f:"):
                key = src.split(":", 1)[1]
                alt = param.get("alt", None)
                sel = select_weight(key, weights, alt)
                if not sel:
                    op_warnings.append(f"{func}.{name}: missing weight_f '{key}', using NULL")
                    expr = "NULL"
                else:
                    _, winfo = sel
                    offset = winfo.get("bump_offset", 0)
                    wname = winfo.get("name")
                    macro = weight_define.get(wname)
                    if use_bump_base:
                        off_expr = macro if macro else str(weights_base + int(offset))
                        expr = ptr_expr("model->bump", off_expr, cast or "float*")
                    else:
                        expr = ptr_expr("model->bump_weights", offset, cast or "float*")

            elif src.startswith("weight:"):
                key = src.split(":", 1)[1]
                sel = select_weight(key, weights)
                if not sel:
                    op_errors.append(f"{func}.{name}: missing weight '{key}'")
                    expr = "NULL"
                else:
                    _, winfo = sel
                    offset = winfo.get("bump_offset", 0)
                    wname = winfo.get("name")
                    macro = weight_define.get(wname)
                    if use_bump_base:
                        off_expr = macro if macro else str(weights_base + int(offset))
                        expr = ptr_expr("model->bump", off_expr, cast or "const void*")
                    else:
                        expr = ptr_expr("model->bump_weights", offset, cast or "const void*")

            elif src.startswith("dim:"):
                key = src.split(":", 1)[1]
                if key in params:
                    expr = str(params[key])
                elif key in config:
                    expr = str(config[key])
                elif key == "max_seq_len" and "context_length" in config:
                    expr = str(config["context_length"])
                elif key == "intermediate_size" and "intermediate_dim" in config:
                    expr = str(config["intermediate_dim"])
                else:
                    op_errors.append(f"{func}.{name}: missing dim '{key}'")
                    expr = "0"

            elif src.startswith("runtime:"):
                key = src.split(":", 1)[1]
                layer = op.get("layer", 0)
                if key in ("kv_cache_k_layer", "kv_k"):
                    expr = f"(model->kv_cache + ({layer}*2)*NUM_KV_HEADS*MAX_SEQ_LEN*HEAD_DIM)"
                elif key in ("kv_cache_v_layer", "kv_v"):
                    expr = f"(model->kv_cache + ({layer}*2+1)*NUM_KV_HEADS*MAX_SEQ_LEN*HEAD_DIM)"
                elif key == "rope_cos":
                    expr = "model->rope_cos"
                elif key == "rope_sin":
                    expr = "model->rope_sin"
                elif key == "pos":
                    expr = "model->pos"
                elif key == "seq_len":
                    if mode == "decode":
                        expr = "model->pos + 1"
                    else:
                        expr = str(params.get("seq_len", 1))
                elif key == "layer":
                    expr = str(layer)
                else:
                    op_errors.append(f"{func}.{name}: unknown runtime '{key}'")
                    expr = "0"
                if cast:
                    expr = f"({cast})({expr})"

            elif src.startswith("const:"):
                expr = src.split(":", 1)[1]

            elif src == "null":
                expr = "NULL"

            elif src.startswith("dtype_weight:"):
                key = src.split(":", 1)[1]
                sel = select_weight(key, weights)
                if not sel:
                    op_errors.append(f"{func}.{name}: missing dtype weight '{key}'")
                    expr = "0"
                else:
                    _, winfo = sel
                    dtype_str = str(winfo.get("dtype", "")).lower()
                    if dtype_str in dtype_map:
                        expr = dtype_map[dtype_str]
                    else:
                        op_errors.append(f"{func}.{name}: unknown weight dtype '{dtype_str}'")
                        expr = "0"

            elif src.startswith("dtype:"):
                key = src.split(":", 1)[1]
                if key in dtype_map:
                    expr = dtype_map[key]
                else:
                    op_errors.append(f"{func}.{name}: unknown dtype '{key}'")
                    expr = "0"

            else:
                op_errors.append(f"{func}.{name}: unknown source '{src}'")
                expr = "0"

            args.append({
                "name": name,
                "source": src,
                "expr": expr,
            })

        if op_errors:
            all_errors.append({
                "idx": op.get("idx", -1),
                "function": func,
                "errors": op_errors,
            })

        call_ops.append({
            "idx": op.get("idx", -1),
            "function": func,
            "op": op.get("op", ""),
            "layer": op.get("layer", -1),
            "section": op.get("section", ""),
            "args": args,
            "errors": op_errors,
            "warnings": op_warnings,
        })

    lowered_call = {
        "format": "lowered-ir-v3",
        "version": 3,
        "mode": mode,
        "config": lowered_ir.get("config", {}),
        "memory": lowered_ir.get("memory", {}),
        "operations": call_ops,
        "errors": all_errors,
    }

    return lowered_call


def generate_init_ir_lower_3(init_ir: Dict, layout: Dict) -> Dict:
    """
    IR Lower 3 for init ops: Emit call-ready ops with ordered args.

    Init ops are simpler than inference ops - they typically just have:
    - Output buffers (rope_cos, rope_sin)
    - Dimension params (max_seq_len, head_dim)
    - Config params (rope_theta)

    Codegen just reads this and emits the function calls sequentially.
    """
    if not init_ir:
        return {"format": "lowered-init-v3", "version": 1, "operations": [], "errors": []}

    ops = init_ir.get("ops", [])
    config = init_ir.get("config", {})
    memory = layout.get("memory", {}) if layout else {}
    act_buffers = {b.get("name"): b for b in memory.get("activations", {}).get("buffers", [])}

    call_ops = []
    all_errors = []

    for op in ops:
        func = op.get("kernel", "unknown")
        op_type = op.get("op", "")
        params = op.get("params", {})
        op_config = op.get("config", {})
        op_errors = []

        args = []

        # Handle rope_init specifically
        if op_type == "rope_init" and func == "rope_precompute_cache":
            # rope_precompute_cache(float *cos_cache, float *sin_cache, int max_seq_len, int head_dim, float base)

            # cos_cache output buffer
            rope_cache_buf = act_buffers.get("rope_cache", act_buffers.get("rope_cos_cache", {}))
            rope_cache_define = rope_cache_buf.get("define", "A_ROPE_CACHE")
            args.append({
                "name": "cos_cache",
                "source": "output:rope_cos",
                "expr": f"(float*)(g_model->bump + {rope_cache_define})",
            })

            # sin_cache output buffer (offset by half)
            args.append({
                "name": "sin_cache",
                "source": "output:rope_sin",
                "expr": f"(float*)(g_model->bump + {rope_cache_define}) + MAX_SEQ_LEN * HEAD_DIM / 2",
            })

            # max_seq_len from params or config
            max_seq = params.get("max_seq_len", {}).get("value", config.get("max_seq_len", 32768))
            args.append({
                "name": "max_seq_len",
                "source": "dim:max_seq_len",
                "expr": "MAX_SEQ_LEN",  # Use the #define for consistency
            })

            # head_dim from params or config
            head_dim = params.get("head_dim", {}).get("value", config.get("head_dim", 64))
            args.append({
                "name": "head_dim",
                "source": "dim:head_dim",
                "expr": "HEAD_DIM",  # Use the #define for consistency
            })

            # base (rope_theta) from params or config - THIS IS THE KEY VALUE
            rope_theta = params.get("base", {}).get("value", op_config.get("rope_theta", 10000.0))
            args.append({
                "name": "base",
                "source": "config:rope_theta",
                "expr": f"{rope_theta}f",  # Emit as float literal
            })

        elif op_type == "tokenizer_init":
            # Tokenizer init has explicit c_code - pass it through directly
            # Codegen will emit the c_code["init"] and c_code["free"] directly
            pass  # No args needed - c_code contains everything

        else:
            # Generic handling for future init ops
            op_errors.append(f"Unknown init op type: {op_type}")

        if op_errors:
            all_errors.append({
                "idx": op.get("op_id", -1),
                "function": func,
                "errors": op_errors,
            })

        call_op = {
            "idx": op.get("op_id", -1),
            "function": func,
            "op": op_type,
            "section": "init",
            "layer": -1,
            "args": args,
            "errors": op_errors,
            "notes": op.get("notes", ""),
        }
        # Pass through c_code for ops that have explicit C code (tokenizer_init, etc.)
        if "c_code" in op:
            call_op["c_code"] = op["c_code"]
        call_ops.append(call_op)

    # Pass through special_tokens from init_ir for code generation
    special_tokens = init_ir.get("special_tokens")

    return {
        "format": "lowered-init-v3",
        "version": 1,
        "config": config,
        # Special tokens (EOS, BOS, etc.) from GGUF - codegen generates stop token API
        "special_tokens": special_tokens,
        "operations": call_ops,
        "errors": all_errors,
        "stats": {
            "total_ops": len(call_ops),
            "errors": len(all_errors),
        }
    }


def main(args: List[str]) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build IR1: Direct template + quant → kernel IDs"
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to weights manifest JSON"
    )
    parser.add_argument(
        "--model",
        type=int,
        help="Use cached model by number (1, 2, ...)"
    )
    parser.add_argument(
        "--mode",
        choices=["decode", "prefill"],
        default="decode",
        help="Execution mode (default: decode)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output IR1 JSON file (just kernel list)"
    )
    parser.add_argument(
        "--layout-output",
        type=Path,
        help="Output memory layout JSON file (after fusion)"
    )
    parser.add_argument(
        "--layout-input",
        type=Path,
        help="Use an existing memory layout JSON instead of generating a new one"
    )
    parser.add_argument(
        "--lowered-output",
        type=Path,
        help="Output lowered IR JSON file (kernel maps stitched with memory layout)"
    )
    parser.add_argument(
        "--manifest-map-output",
        type=Path,
        help="Output weights_manifest.map (uses runtime offsets from layout)"
    )
    parser.add_argument(
        "--call-output",
        type=Path,
        help="Output call-ready IR JSON file (IR Lower 3)"
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Context length for buffer allocation (default: from model config)"
    )
    parser.add_argument(
        "--no-fusion",
        action="store_true",
        help="Disable kernel fusion pass (use unfused ops)"
    )
    parser.add_argument(
        "--layout-mode",
        choices=["region", "packed"],
        default="region",
        help="Memory layout mode (region=weights+activations, packed=single arena)"
    )
    parser.add_argument(
        "--layer-limit",
        type=int,
        default=None,
        help="Limit to first N layers (for packed layout prototypes)"
    )
    parser.add_argument(
        "--init-output",
        type=Path,
        help="Output init IR JSON file (one-time initialization ops like rope_init)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable OpenMP parallelization annotations in lowered IR"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-kernel profiling instrumentation in generated code"
    )

    parsed_args = parser.parse_args(args)

    # Load manifest
    if parsed_args.manifest:
        manifest_path = parsed_args.manifest
    elif parsed_args.model:
        # TODO: Find cached model
        print("Error: --model not implemented yet, use --manifest")
        return 1
    else:
        print("Error: Must specify --manifest or --model")
        parser.print_help()
        return 1

    print(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)

    # Build IR1
    registry = load_kernel_registry()
    ir1 = build_ir1_direct(manifest, manifest_path, mode=parsed_args.mode,
                           prefer_parallel=parsed_args.parallel)

    # Insert bias_add ops BEFORE fusion pass so fused kernels can match
    # [quantize + gemv + bias_add] sequences
    ir1_with_bias = insert_bias_add_ops(ir1, registry, manifest, parsed_args.mode)

    # Fusion pass: combine kernels (fused attention, fused MLP, fused GEMV+bias)
    fused_ops, fusion_stats = apply_fusion_pass(ir1_with_bias, registry, parsed_args.mode, no_fusion=parsed_args.no_fusion)

    # IR Lower 1: Stitch kernel maps with fused ops
    # This creates buffer requirements (inputs/outputs/scratch) for each kernel
    ir_lower_1 = generate_ir_lower_1(fused_ops, registry, manifest, parsed_args.mode)

    # Optional: limit to first N layers (keep header ops)
    if parsed_args.layer_limit:
        limit = int(parsed_args.layer_limit)
        filtered = []
        for op in ir_lower_1:
            layer = op.get("layer", -1)
            section = op.get("section", "")
            if section == "header":
                filtered.append(op)
            elif section == "body" and layer >= 0 and layer < limit:
                filtered.append(op)
            elif section == "footer" and limit <= 0:
                filtered.append(op)
        ir_lower_1 = filtered

    # Memory Planner: Plan memory layout using IR Lower 1 buffer requirements
    context_len = parsed_args.context_len  # May be None, will use model default
    if parsed_args.layout_input:
        with open(parsed_args.layout_input, "r") as f:
            layout = json.load(f)
        if not layout.get("memory"):
            raise RuntimeError(f"Invalid layout (missing 'memory'): {parsed_args.layout_input}")
        # Keep layout offsets, but update mode for clarity in per-mode outputs
        layout["mode"] = parsed_args.mode
        if parsed_args.layout_mode:
            arena_mode = layout.get("memory", {}).get("arena", {}).get("mode")
            if arena_mode and arena_mode != parsed_args.layout_mode:
                print(f"Warning: layout_input mode '{arena_mode}' != requested '{parsed_args.layout_mode}'")
    else:
        if parsed_args.layout_mode == "packed":
            layout = generate_memory_layout_packed(
                ir_lower_1, manifest, registry, parsed_args.mode, context_len, parsed_args.layer_limit
            )
        else:
            layout = generate_memory_layout(ir_lower_1, manifest, registry, parsed_args.mode, context_len)

    # IR Lower 2: Add concrete memory offsets to IR Lower 1
    # This produces the final lowered IR with explicit pointer expressions
    lowered_ir = generate_ir_lower_2(ir_lower_1, layout, manifest, registry, parsed_args.mode)

    # CRITICAL: Update context_length in lowered_ir to match layout
    # This ensures codegen uses the correct MAX_SEQ_LEN for KV cache strides
    # Use context_len from layout config if available, otherwise from context_len variable
    effective_context_len = context_len
    if layout and "config" in layout and "context_length" in layout["config"]:
        effective_context_len = layout["config"]["context_length"]
    elif layout and "config" in layout and "context_len" in layout["config"]:
        effective_context_len = layout["config"]["context_len"]

    if effective_context_len:
        if "config" not in lowered_ir:
            lowered_ir["config"] = {}
        lowered_ir["config"]["context_length"] = effective_context_len
        lowered_ir["config"]["context_len"] = effective_context_len

    lowered_call = None
    if parsed_args.call_output:
        lowered_call = generate_ir_lower_3(lowered_ir, parsed_args.mode)

        # NOTE: OpenMP parallel pass is SUPERSEDED by thread pool dispatch.
        #
        # parallel_pass.py annotates ops with #pragma omp parallel for, but
        # codegen_v6_6.py never reads these annotations. Actual parallelization
        # is handled by ck_parallel_decode.h / ck_parallel_prefill.h which use
        # persistent pthread thread pools via macro redirects:
        #   - Decode: gemv_q5_0_q8_0() → gemv_q5_0_q8_0_parallel_dispatch()
        #   - Prefill: gemm_nt_q5_0_q8_0() → gemm_nt_q5_0_q8_0_parallel_dispatch()
        #
        # Kept commented out for reference. The false-sharing and memory-bandwidth
        # analysis in parallel_pass.py remains useful for planning thread pool work
        # splitting strategies.
        #
        # if getattr(parsed_args, 'parallel', False) and lowered_call:
        #     try:
        #         from parallel_pass import run_parallel_pass
        #         lowered_call, parallel_stats = run_parallel_pass(
        #             lowered_call, parsed_args.mode
        #         )
        #         print(f"  [PARALLEL PASS] {parallel_stats['parallelized_ops']}/{parallel_stats['total_ops']} ops annotated")
        #         for strat, count in parallel_stats.get('strategies', {}).items():
        #             print(f"    - {strat}: {count} ops")
        #     except ImportError as e:
        #         print(f"  [PARALLEL PASS] Warning: parallel_pass.py not found ({e})")
        #     except Exception as e:
        #         print(f"  [PARALLEL PASS] Warning: parallelization failed: {e}")

    # Once we have the right memory and lowered graph then we can do codegen
    # codegen will read the lowered IR and memory layout to emit C code
    # it should just see memory and parse the memory layout - allocate bump
    # The code should then have a load weights and load then to the right bump offset.
    # Then read lowered graph and generate c code sequentially for prefill and decode with
    # right inputs and offset to weights read.
    # and then generate tokens. We have all this working in v5 and v6 and v6.5. but v6.6
    # is the first to have the full pipeline completely generated from template + quant summary
    # and no hardcoded logic for a specific family.

    # Output
    if parsed_args.output:
        output_data = {
            "format": "ir1-dataflow",
            "version": 3,
            "mode": parsed_args.mode,
            "ops": ir1  # Now a list of {kernel, op, section, layer, weights, dataflow}
        }
        with open(parsed_args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Wrote IR1 to: {parsed_args.output}")

    if parsed_args.layout_output:
        with open(parsed_args.layout_output, 'w') as f:
            json.dump(layout, f, indent=2)
        print(f"✓ Wrote memory layout to: {parsed_args.layout_output}")
    if parsed_args.manifest_map_output:
        write_manifest_map(layout, manifest, parsed_args.manifest_map_output)
        print(f"✓ Wrote manifest map to: {parsed_args.manifest_map_output}")

    if parsed_args.lowered_output:
        with open(parsed_args.lowered_output, 'w') as f:
            json.dump(lowered_ir, f, indent=2)
        print(f"✓ Wrote lowered IR to: {parsed_args.lowered_output}")

    if parsed_args.call_output and lowered_call is not None:
        with open(parsed_args.call_output, 'w') as f:
            json.dump(lowered_call, f, indent=2)
        print(f"✓ Wrote call-ready IR to: {parsed_args.call_output}")

    # Generate and write init IR (rope_init, etc.)
    if parsed_args.init_output:
        config = manifest.get("config", {})
        init_ir = generate_init_ir(manifest, config)
        with open(parsed_args.init_output, 'w') as f:
            json.dump(init_ir, f, indent=2)
        print(f"✓ Wrote init IR to: {parsed_args.init_output}")
        if init_ir["stats"]["has_rope_init"]:
            rope_theta = init_ir["config"].get("rope_theta", 10000.0)
            print(f"  - rope_init: theta={rope_theta}")

        # Also generate lowered init IR (init_call.json)
        init_call_path = parsed_args.init_output.parent / "init_call.json"
        init_call = generate_init_ir_lower_3(init_ir, layout)
        with open(init_call_path, 'w') as f:
            json.dump(init_call, f, indent=2)
        print(f"✓ Wrote init call IR to: {init_call_path}")

    if not parsed_args.output and not parsed_args.layout_output and not parsed_args.lowered_output:
        print(f"\nIR1 (first 10 ops):")
        for i, op in enumerate(ir1[:10]):
            kernel = op["kernel"]
            weights = len(op.get("weights", {}))
            print(f"  {i:3d}: {kernel} ({weights} weights)")
        if len(ir1) > 10:
            print(f"  ... ({len(ir1) - 10} more)")

        print(f"\nFused ops (first 10):")
        for i, op in enumerate(fused_ops[:10]):
            kernel = op["kernel"]
            weights = len(op.get("weights", {}))
            print(f"  {i:3d}: {kernel} ({weights} weights)")
        if len(fused_ops) > 10:
            print(f"  ... ({len(fused_ops) - 10} more)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
