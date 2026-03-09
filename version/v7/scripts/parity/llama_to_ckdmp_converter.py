#!/usr/bin/env python3
"""
llama_to_ckdmp_converter.py

Convert raw llama.cpp tensor dumps to CKDMP format for comparison with C-Kernel Engine.

Usage:
    # Convert all .bin files in llama_dump/ to CKDMP format
    python version/v7/scripts/parity/llama_to_ckdmp_converter.py --input llama_dump --output llama_parity_dumps/dump.bin

    # Convert with dtype inference from filename patterns
    python version/v7/scripts/parity/llama_to_ckdmp_converter.py --input llama_dump --output llama_parity_dumps/dump.bin --dtype-hint fp32

    # Generate index.json alongside dump
    python version/v7/scripts/parity/llama_to_ckdmp_converter.py --input llama_dump --output llama_parity_dumps --index
"""

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# =============================================================================
# CKDMP Format Definition (must match ck_parity_dump.h)
# =============================================================================

CKDMP_MAGIC = b"CKDMP\x00\x00\x00"
CKDMP_VERSION = 1
CKDMP_HEADER_SIZE = 128

HEADER_SPEC = struct.Struct(
    "<8s"     # magic (8)
    "I"       # version (4)
    "i"       # layer_id (4)
    "32s"     # op_name (32)
    "I"       # dtype (4)
    "I"       # rank (4)
    "4q"      # shape[4] (32)
    "I"       # elem_count (4)
    "i"       # token_id (4)
    "I"       # dump_type (4)
    "28x"     # reserved (28)
)

# =============================================================================
# Name Mapping Tables for Model Families
# =============================================================================

# Maps llama.cpp tensor names to C-Kernel Engine op names
# Format: (model_family, llama_name) -> ck_name
# Note: Includes common llama.cpp naming patterns (Qcur, Kcur, Vcur, etc.)
LLAMA_TO_CK_NAME_MAP = {
    # Gemma family
    "gemma": {
        # Standard names
        "token_embd": "token_embedding",
        "attn_norm": "attn_norm",
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_rot": "rotary_emb",
        "attn": "attn_output",
        "attn_out": "out_proj",
        "ffn_norm": "ffn_norm",
        "ffn_gate": "gate_proj",
        "ffn_up": "up_proj",
        "ffn_down": "down_proj",
        "ffn_act": "ffn_swiglu",
        "final_norm": "final_norm",
        "logits": "logits",
        "output": "output",
        # Llama.cpp internal naming patterns
        "Qcur": "q_proj",
        "Kcur": "k_proj",
        "Vcur": "v_proj",
        "attn_output": "attn_output",
        "ffn_gate_inp": "gate_proj",
        "ffn_down": "down_proj",
        "ffn_up": "up_proj",
        "token_embd": "token_embedding",
    },
    # Llama family
    "llama": {
        "token_embd": "token_embedding",
        "attn_norm": "attn_norm",
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_rot": "rotary_emb",
        "attn": "attn_output",
        "attn_out": "out_proj",
        "ffn_norm": "ffn_norm",
        "ffn_gate": "gate_proj",
        "ffn_up": "up_proj",
        "ffn_down": "down_proj",
        "ffn_act": "ffn_swiglu",
        "final_norm": "final_norm",
        "logits": "logits",
        "output": "output",
        # Llama.cpp internal naming patterns
        "Qcur": "q_proj",
        "Kcur": "k_proj",
        "Vcur": "v_proj",
        "attn_output": "attn_output",
        "ffn_gate_inp": "gate_proj",
    },
    # Qwen family
    "qwen": {
        "token_embd": "token_embedding",
        "attn_norm": "attn_norm",
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_rot": "rotary_emb",
        "attn": "attn_output",
        "attn_out": "out_proj",
        "ffn_norm": "ffn_norm",
        "ffn_gate": "gate_proj",
        "ffn_up": "up_proj",
        "ffn_down": "down_proj",
        "ffn_act": "ffn_swiglu",
        "final_norm": "final_norm",
        "logits": "logits",
        "output": "output",
        # Llama.cpp internal naming patterns
        "Qcur": "q_proj",
        "Kcur": "k_proj",
        "Vcur": "v_proj",
    },
    # Qwen2/Qwen3 family
    "qwen2": {
        "token_embd": "token_embedding",
        "attn_norm": "attn_norm",
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_rot": "rotary_emb",
        "attn": "attn_output",
        "attn_out": "out_proj",
        "ffn_norm": "ffn_norm",
        "ffn_gate": "gate_proj",
        "ffn_up": "up_proj",
        "ffn_down": "down_proj",
        "ffn_act": "ffn_swiglu",
        "final_norm": "final_norm",
        "logits": "logits",
        "output": "output",
        # Llama.cpp internal naming patterns
        "Qcur": "q_proj",
        "Kcur": "k_proj",
        "Vcur": "v_proj",
    },
    # Mistral family
    "mistral": {
        "token_embd": "token_embedding",
        "attn_norm": "attn_norm",
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_rot": "rotary_emb",
        "attn": "attn_output",
        "attn_out": "out_proj",
        "ffn_norm": "ffn_norm",
        "ffn_gate": "gate_proj",
        "ffn_up": "up_proj",
        "ffn_down": "down_proj",
        "ffn_act": "ffn_swiglu",
        "final_norm": "final_norm",
        "logits": "logits",
        "output": "output",
        # Llama.cpp internal naming patterns
        "Qcur": "q_proj",
        "Kcur": "k_proj",
        "Vcur": "v_proj",
    },
}

# Qwen3 currently shares parity tensor naming with qwen2 in this converter.
LLAMA_TO_CK_NAME_MAP["qwen3"] = dict(LLAMA_TO_CK_NAME_MAP["qwen2"])

# Reverse map for CK to llama name
CK_TO_LLAMA_NAME_MAP = {}
for family, mapping in LLAMA_TO_CK_NAME_MAP.items():
    for llama_name, ck_name in mapping.items():
        if ck_name not in CK_TO_LLAMA_NAME_MAP:
            CK_TO_LLAMA_NAME_MAP[ck_name] = {}
        CK_TO_LLAMA_NAME_MAP[ck_name][family] = llama_name


def extract_layer_id(op_name: str) -> int:
    """Extract layer ID from llama tensor name (e.g., 'attn_q-3' -> 3)."""
    # Common patterns: op-layer, op_layer_layerid, op.lN
    for sep in ['-', '_', '.']:
        parts = op_name.split(sep)
        if len(parts) >= 2:
            last_part = parts[-1]
            if last_part.isdigit():
                return int(last_part)
            # Handle patterns like "layer.3"
            if last_part.startswith('layer') or last_part.startswith('L'):
                try:
                    return int(last_part.replace('layer', '').replace('L', ''))
                except ValueError:
                    pass
    return -1


def infer_dtype_from_name(filename: str) -> int:
    """Infer dtype from filename patterns."""
    filename_lower = filename.lower()
    if 'fp16' in filename_lower or '_f16' in filename_lower:
        return 1  # fp16
    elif 'bf16' in filename_lower or '_bf16' in filename_lower:
        return 2  # bf16
    elif 'q8' in filename_lower or '_q8' in filename_lower:
        return 3  # int8
    elif 'q4' in filename_lower or '_q4' in filename_lower:
        return 4  # int4
    else:
        return 0  # default fp32


TOKEN_SUFFIX_PATTERNS = (
    re.compile(r"^(?P<base>.+?)-token-(?P<tok>\d+)(?:-occ-\d+)?$"),
    re.compile(r"^(?P<base>.+?)_token_(?P<tok>\d+)(?:_occ_\d+)?$"),
    re.compile(r"^(?P<base>.+?)-tok-(?P<tok>\d+)(?:-occ-\d+)?$"),
    re.compile(r"^(?P<base>.+?)_tok_(?P<tok>\d+)(?:_occ_\d+)?$"),
)


def split_token_suffix(op_name: str) -> Tuple[str, Optional[int]]:
    """Split optional token suffix from raw op name.

    Examples:
      Qcur-0-token-000012 -> (Qcur-0, 12)
      attn_out-0_tok_42   -> (attn_out-0, 42)
    """
    for pat in TOKEN_SUFFIX_PATTERNS:
        m = pat.match(op_name)
        if not m:
            continue
        base = str(m.group("base"))
        try:
            tok = int(m.group("tok"))
        except Exception:
            tok = None
        return base, tok
    return op_name, None


def infer_shape_from_data(data: np.ndarray) -> Tuple[int, List[int]]:
    """Infer proper shape from numpy array."""
    shape = list(data.shape)
    # Remove trailing 1s
    while len(shape) > 1 and shape[-1] == 1:
        shape.pop()
    rank = len(shape)
    if rank == 0:
        rank = 1
        shape = [1]
    return rank, shape


def parse_index_json(index_path: Path) -> Dict[str, Dict]:
    """Parse index.json to extract tensor metadata.

    Supports both JSON array format and JSONL (one object per line).
    """
    metadata = {}
    if not index_path.exists():
        return metadata

    try:
        with open(index_path, 'r') as f:
            content = f.read().strip()

            # Try JSON array format first
            if content.startswith('[') and content.endswith(']'):
                content = content[1:-1].strip()
                if content:
                    for line in content.split('\n'):
                        line = line.strip().rstrip(',')
                        if line.startswith('{') and line.endswith('}'):
                            try:
                                entry = json.loads(line)
                                if 'name' in entry:
                                    metadata[entry['name']] = entry
                            except json.JSONDecodeError:
                                pass
            # Try JSONL format
            else:
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if 'name' in entry:
                                metadata[entry['name']] = entry
                        except json.JSONDecodeError:
                            pass

    except Exception as e:
        print(f"[WARNING] Could not parse index.json: {e}")

    return metadata


def convert_raw_to_ckdmp(
    input_dir: Path,
    output_file: Path,
    model_family: str = "gemma",
    dtype_hint: Optional[int] = None,
    generate_index: bool = True,
) -> Tuple[int, int]:
    """
    Convert raw tensor dumps to CKDMP format.

    Args:
        input_dir: Directory containing raw .bin files
        output_file: Output CKDMP file path
        model_family: Model family for name mapping
        dtype_hint: Force dtype (0=fp32, 1=fp16, 2=bf16, 3=int8, 4=int4)
        generate_index: Also generate index.json

    Returns:
        (num_converted, num_errors)
    """
    converted = 0
    errors = 0

    # Get name mapping for model family
    name_map = LLAMA_TO_CK_NAME_MAP.get(model_family.lower(), LLAMA_TO_CK_NAME_MAP.get("gemma", {}))

    # Look for index.json with metadata (preferred over filename inference)
    index_metadata = parse_index_json(input_dir / "index.json")

    # Collect all .bin files
    bin_files = sorted(input_dir.glob("*.bin"))
    if not bin_files:
        print(f"[ERROR] No .bin files found in {input_dir}")
        return 0, 1

    print(f"[INFO] Found {len(bin_files)} tensor files in {input_dir}")

    # Prepare output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as out_f:
        # Write index entries if requested
        index_entries = []

        for bin_file in bin_files:
            op_name = bin_file.stem
            base_op_name, token_from_name = split_token_suffix(op_name)

            try:
                # Read raw data
                data = np.fromfile(bin_file, dtype=np.float32)

                if len(data) == 0:
                    print(f"[WARNING] Empty file: {bin_file}")
                    errors += 1
                    continue

                # Get metadata from index.json if available (preferred source)
                metadata = index_metadata.get(op_name, {})
                if not metadata:
                    metadata = index_metadata.get(base_op_name, {})

                raw_base_name = str(metadata.get("base_name") or base_op_name)
                if raw_base_name:
                    base_op_name = raw_base_name

                # Determine dtype
                if dtype_hint is not None:
                    dtype = dtype_hint
                elif 'dtype' in metadata:
                    dtype = int(metadata['dtype'])
                else:
                    dtype = infer_dtype_from_name(bin_file.name)

                # Get layer ID
                layer_id = extract_layer_id(base_op_name)
                if 'layer_id' in metadata:
                    layer_id = int(metadata['layer_id'])

                # Get token ID
                token_id = token_from_name if token_from_name is not None else 0
                if 'token_id' in metadata:
                    token_id = int(metadata['token_id'])
                else:
                    # Try to extract from filename
                    for sep in ['_token_', '-token-']:
                        if sep in op_name:
                            try:
                                token_id = int(op_name.split(sep)[1].split('-')[0].split('_')[0])
                                break
                            except (ValueError, IndexError):
                                pass

                # Get shape
                rank, shape = infer_shape_from_data(data)
                if 'shape' in metadata:
                    shape = list(metadata['shape'])
                    rank = len([s for s in shape if s > 0])
                if 'rank' in metadata:
                    rank = int(metadata['rank'])

                elem_count = len(data)
                if 'elem_count' in metadata:
                    elem_count = int(metadata['elem_count'])

                # Map llama name to CK name
                ck_name = name_map.get(base_op_name, base_op_name)

                # Determine dump type from context
                dump_type = 0  # prefill
                if token_id > 0:
                    dump_type = 1
                elif 'decode' in base_op_name.lower() or 'second' in base_op_name.lower():
                    dump_type = 1  # decode

                # Build header
                header = bytearray(CKDMP_HEADER_SIZE)
                # Keep magic assignment length-exact to avoid shrinking bytearray.
                header[0:8] = CKDMP_MAGIC
                struct.pack_into("<I", header, 8, CKDMP_VERSION)
                struct.pack_into("<i", header, 12, layer_id)
                name_bytes = ck_name.encode('utf-8')[:31]
                header[16:16 + len(name_bytes)] = name_bytes
                struct.pack_into("<I", header, 48, dtype)
                struct.pack_into("<I", header, 52, rank)
                for i in range(4):
                    offset = 56 + i * 8
                    val = shape[i] if i < len(shape) else 1
                    struct.pack_into("<q", header, offset, val)
                struct.pack_into("<I", header, 88, elem_count)
                struct.pack_into("<i", header, 92, token_id)
                struct.pack_into("<I", header, 96, dump_type)

                # Write to output
                out_f.write(header)
                out_f.write(data.astype(np.float32).tobytes())

                # Index entry
                if generate_index:
                    index_entries.append({
                        "name": ck_name,
                        "llama_name": base_op_name,
                        "llama_raw_name": op_name,
                        "original_file": bin_file.name,
                        "dtype": dtype,
                        "rank": rank,
                        "shape": shape,
                        "elem_count": elem_count,
                        "layer_id": layer_id,
                        "token_id": token_id,
                        "dump_type": dump_type,
                        "offset": out_f.tell() - CKDMP_HEADER_SIZE - elem_count * 4,
                    })

                print(
                    f"  [OK] {op_name} -> {ck_name} "
                    f"(layer={layer_id}, token={token_id}, dtype={dtype}, shape={shape})"
                )
                converted += 1

            except Exception as e:
                print(f"  [ERROR] {bin_file}: {e}")
                errors += 1

    # Write index.json
    if generate_index and index_entries:
        index_file = output_file.parent / "index.json"
        with open(index_file, 'w') as f:
            f.write('[\n')
            for i, entry in enumerate(index_entries):
                f.write("  " + json.dumps(entry))
                if i < len(index_entries) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("]\n")
        print(f"[INFO] Wrote index.json to {index_file}")

    print(f"\n[SUMMARY] Converted: {converted}, Errors: {errors}")
    return converted, errors


def merge_ckdmp_files(
    input_dir: Path,
    output_file: Path,
    model_family: str = "gemma",
) -> Tuple[int, int]:
    """
    Merge existing CKDMP files into a single dump.

    Args:
        input_dir: Directory containing .bin CKDMP files
        output_file: Output merged file
        model_family: Model family for name mapping

    Returns:
        (num_merged, num_errors)
    """
    return convert_raw_to_ckdmp(input_dir, output_file, model_family)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert llama.cpp tensor dumps to CKDMP format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert raw dumps to CKDMP
    python version/v7/scripts/parity/llama_to_ckdmp_converter.py -i llama_dump -o llama_parity_dumps/dump.bin

    # Convert with Gemma name mapping
    python version/v7/scripts/parity/llama_to_ckdmp_converter.py -i llama_dump -o llama_parity_dumps/dump.bin --model gemma

    # Force fp16 dtype
    python version/v7/scripts/parity/llama_to_ckdmp_converter.py -i llama_dump -o llama_parity_dumps/dump.bin --dtype fp16

    # Generate index.json for reference
    python version/v7/scripts/parity/llama_to_ckdmp_converter.py -i llama_dump -o llama_parity_dumps --index
        """
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                       help="Input directory containing .bin files")
    parser.add_argument("-o", "--output", type=Path, required=True,
                       help="Output CKDMP file path")
    parser.add_argument("--model", default="gemma",
                       choices=["gemma", "llama", "qwen", "qwen2", "qwen3", "mistral"],
                       help="Model family for name mapping")
    parser.add_argument("--dtype", default=None,
                       choices=["fp32", "fp16", "bf16", "q8", "q4"],
                       help="Force dtype for all tensors")
    parser.add_argument("--index", action="store_true",
                       help="Generate index.json alongside output")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Convert dtype option
    dtype_map = {"fp32": 0, "fp16": 1, "bf16": 2, "q8": 3, "q4": 4}
    dtype_hint = dtype_map.get(args.dtype)

    print(f"=" * 70)
    print("LLAMA.CPP to CKDMP Converter")
    print(f"=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")
    print(f"Dtype:   {'auto' if dtype_hint is None else dtype_map.get(dtype_hint, dtype_hint)}")
    print(f"=" * 70)

    converted, errors = convert_raw_to_ckdmp(
        args.input,
        args.output,
        args.model,
        dtype_hint,
        args.index,
    )

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
