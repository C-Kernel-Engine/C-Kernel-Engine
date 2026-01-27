#!/usr/bin/env python3
"""
Generate memory_layout.c from fused IR.

Calculates:
- Scratch buffer sizes
- Tensor strides
- Memory pool layout
- Alignment requirements

Usage:
    python scripts/gen_memory_layout.py --ir build/fused_ir/ --output src/memory_layout.c
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TensorLayout:
    """Layout information for a tensor."""
    name: str
    shape: List[int]
    dtype_size: int  # bytes
    strides: List[int]
    total_size: int
    alignment: int = 64


@dataclass
class ScratchBuffer:
    """Scratch buffer requirement."""
    name: str
    size_bytes: int
    alignment: int = 64
    purpose: str = ""


class MemoryLayoutGenerator:
    """Generate memory layout from IR."""

    # Size per dtype (bytes)
    DTYPE_SIZES = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp64": 8,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "uint8": 1,
        "uint16": 2,
        "q8_0": 1,  # 1 byte per value
        "q5_0": 0.5,  # 4 bits per value
        "q5_1": 0.5,
        "q4_0": 0.5,
        "q4_1": 0.5,
        "q4_k": 0.5,
        "q6_k": 0.75,
    }

    # Alignment requirements
    ALIGNMENTS = {
        "fp32": 16,
        "fp16": 16,
        "bf16": 16,
        "fp64": 32,
        "quantized": 32,
    }

    def __init__(self, max_tokens: int = 8192, max_embed_dim: int = 8192):
        self.max_tokens = max_tokens
        self.max_embed_dim = max_embed_dim
        self.scratch_buffers: List[ScratchBuffer] = []
        self.tensor_layouts: Dict[str, TensorLayout] = {}

    def dtype_size(self, dtype: str) -> int:
        """Get size in bytes for a dtype."""
        if dtype in self.DTYPE_SIZES:
            return int(self.DTYPE_SIZES[dtype] * 32)  # Handle q* types as int
        return 4  # Default to fp32

    def alignment_for_dtype(self, dtype: str) -> int:
        """Get alignment requirement for dtype."""
        if dtype.startswith("q"):
            return self.ALIGNMENTS["quantized"]
        return self.ALIGNMENTS.get(dtype, 64)

    def calculate_strides(self, shape: List[int]) -> List[int]:
        """Calculate row-major strides from shape."""
        if not shape:
            return [1]
        strides = [1]
        for dim in reversed(shape[:-1]):
            strides.insert(0, strides[0] * dim)
        return strides

    def calculate_size(self, shape: List[int], dtype: str) -> int:
        """Calculate total size in bytes."""
        size = 1
        for dim in shape:
            size *= dim
        return size * self.dtype_size(dtype)

    def round_to_alignment(self, size: int, alignment: int) -> int:
        """Round up to alignment boundary."""
        return (size + alignment - 1) // alignment * alignment

    def process_ir(self, ir: Dict) -> Tuple[Dict[str, TensorLayout], List[ScratchBuffer]]:
        """Process IR and extract memory requirements."""
        tensors = {}
        scratch = []

        # Process inputs
        for inp in ir.get("inputs", []):
            name = inp.get("name")
            dtype = inp.get("dtype", "fp32")
            shape = inp.get("shape", [])

            # Convert symbolic dims to actual sizes
            actual_shape = self._resolve_shape(shape)

            strides = self.calculate_strides(actual_shape)
            size = self.calculate_size(actual_shape, dtype)
            alignment = self.alignment_for_dtype(dtype)

            tensors[name] = TensorLayout(
                name=name,
                shape=actual_shape,
                dtype_size=self.dtype_size(dtype),
                strides=strides,
                total_size=self.round_to_alignment(size, alignment),
                alignment=alignment,
            )

        # Process outputs
        for out in ir.get("outputs", []):
            name = out.get("name")
            dtype = out.get("dtype", "fp32")
            shape = out.get("shape", [])

            actual_shape = self._resolve_shape(shape)

            strides = self.calculate_strides(actual_shape)
            size = self.calculate_size(actual_shape, dtype)
            alignment = self.alignment_for_dtype(dtype)

            tensors[name] = TensorLayout(
                name=name,
                shape=actual_shape,
                dtype_size=self.dtype_size(dtype),
                strides=strides,
                total_size=self.round_to_alignment(size, alignment),
                alignment=alignment,
            )

        # Process scratch buffers from ops
        for op in ir.get("ops", []):
            if isinstance(op, dict) and "scratch" in op:
                for scr in op["scratch"]:
                    name = scr.get("name", f"scratch_{len(scratch)}")
                    size_expr = scr.get("size_bytes", "0")
                    purpose = scr.get("desc", "")

                    # Parse size expression
                    if isinstance(size_expr, str) and size_expr.endswith("()"):
                        # Function call like "mega_fused_attention_prefill_scratch_size()"
                        # Use a default size for now
                        size = 1024 * 1024  # 1MB default
                    elif isinstance(size_expr, str):
                        # Try to parse expression
                        try:
                            size = int(size_expr)
                        except:
                            size = 1024
                    else:
                        size = int(size_expr)

                    scratch.append(ScratchBuffer(
                        name=name,
                        size_bytes=self.round_to_alignment(size, 64),
                        alignment=64,
                        purpose=purpose,
                    ))

        return tensors, scratch

    def _resolve_shape(self, shape: List) -> List[int]:
        """Resolve symbolic shape to actual integers."""
        resolved = []
        for dim in shape:
            if isinstance(dim, int):
                resolved.append(dim)
            elif isinstance(dim, str):
                # Map symbolic dims to actual sizes
                dim_map = {
                    "T": self.max_tokens,
                    "max_T": self.max_tokens,
                    "E": self.max_embed_dim,
                    "AE": self.max_embed_dim,
                    "H": 32,
                    "KV": 8,
                    "D": 128,
                    "AD": 128,
                }
                resolved.append(dim_map.get(dim, 1024))
            elif isinstance(dim, dict):
                # Handle {"dim": "T", "max": 8192}
                resolved.append(dim.get("max", 1024))
            else:
                resolved.append(1024)
        return resolved

    def generate_c_header(self, tensors: Dict[str, TensorLayout], scratch: List[ScratchBuffer]) -> str:
        """Generate memory_layout.c header."""
        lines = [
            "/* =========================================",
            " * Auto-generated Memory Layout",
            " * Generated by scripts/gen_memory_layout.py",
            " * =========================================",
            " */",
            "",
            "#ifndef CK_MEMORY_LAYOUT_H",
            "#define CK_MEMORY_LAYOUT_H",
            "",
            "#include <stddef.h>",
            "#include <stdint.h>",
            "",
            "/* =========================================",
            " * Configuration",
            " * =========================================",
            " */",
            f"#define CK_MAX_TOKENS {self.max_tokens}",
            f"#define CK_MAX_EMBED_DIM {self.max_embed_dim}",
            "",
            "/* =========================================",
            " * Tensor Layout Macros",
            " * =========================================",
        ]

        # Generate tensor macros
        for name, layout in sorted(tensors.items()):
            safe_name = name.upper().replace(" ", "_").replace("-", "_")
            lines.append(f"")
            lines.append(f"/* {name}: shape={layout.shape}, strides={layout.strides} */")
            lines.append(f"#define CK_TENSOR_{safe_name}_SIZE {layout.total_size}")
            lines.append(f"#define CK_TENSOR_{safe_name}_STRIDE_0 {layout.strides[0] if layout.strides else 1}")
            if len(layout.strides) > 1:
                lines.append(f"#define CK_TENSOR_{safe_name}_STRIDE_1 {layout.strides[1]}")

        # Generate scratch buffer constants
        lines.append("")
        lines.append("/* =========================================")
        lines.append(" * Scratch Buffer Sizes")
        lines.append(" * =========================================")
        lines.append("")

        total_scratch = 0
        for scr in scratch:
            safe_name = scr.name.upper().replace(" ", "_").replace("-", "_")
            lines.append(f"/* {scr.name}: {scr.purpose} */")
            lines.append(f"#define CK_SCRATCH_{safe_name}_SIZE {scr.size_bytes}")
            total_scratch += scr.size_bytes

        if scratch:
            lines.append("")
            lines.append(f"/* Total scratch requirement */")
            lines.append(f"#define CK_TOTAL_SCRATCH_SIZE {total_scratch}")

        # Generate offsets
        lines.append("")
        lines.append("/* =========================================")
        lines.append(" * Memory Pool Offsets")
        lines.append(" * =========================================")
        lines.append("")

        offset = 0
        for name, layout in sorted(tensors.items()):
            safe_name = name.upper().replace(" ", "_").replace("-", "_")
            aligned_offset = self.round_to_alignment(offset, layout.alignment)
            lines.append(f"#define CK_OFFSET_{safe_name} {aligned_offset}")
            offset = aligned_offset + layout.total_size

        lines.append("")
        lines.append(f"/* Total tensor memory */")
        lines.append(f"#define CK_TOTAL_TENSOR_SIZE {self.round_to_alignment(offset, 64)}")

        # Generate function declarations
        lines.append("")
        lines.append("/* =========================================")
        lines.append(" * Memory Allocation Functions")
        lines.append(" * =========================================")
        lines.append("")

        lines.append("// Initialize memory pool")
        lines.append("void* ck_alloc_memory_pool(size_t size);")
        lines.append("")
        lines.append("// Free memory pool")
        lines.append("void ck_free_memory_pool(void* pool);")
        lines.append("")
        lines.append("// Get scratch buffer pointer")
        lines.append(f"void* ck_get_scratch_{scr.name if scratch else 'main'}(void* pool);" if scratch else "")

        lines.append("")
        lines.append("#endif // CK_MEMORY_LAYOUT_H")

        return "\n".join(lines)

    def generate_c_source(self, tensors: Dict[str, TensorLayout], scratch: List[ScratchBuffer]) -> str:
        """Generate memory_layout.c source."""
        lines = [
            "/* =========================================",
            " * Auto-generated Memory Layout Implementation",
            " * Generated by scripts/gen_memory_layout.py",
            " * =========================================",
            " */",
            "",
            '#include "memory_layout.h"',
            '#include <stdlib.h>',
            '#include <string.h>',
            "",
            "/* =========================================",
            " * Memory Pool",
            " * =========================================",
            "",
            "static char* g_memory_pool = NULL;",
            "static size_t g_pool_capacity = 0;",
            "",
            "void* ck_alloc_memory_pool(size_t size) {",
            "    if (g_memory_pool) {",
            "        free(g_memory_pool);",
            "    }",
            "    g_pool_capacity = size;",
            "    g_memory_pool = (char*)aligned_alloc(64, size);",
            "    if (g_memory_pool) {",
            "        memset(g_memory_pool, 0, size);",
            "    }",
            "    return g_memory_pool;",
            "}",
            "",
            "void ck_free_memory_pool(void* pool) {",
            "    (void)pool;",
            "    if (g_memory_pool) {",
            "        free(g_memory_pool);",
            "        g_memory_pool = NULL;",
            "    }",
            "    g_pool_capacity = 0;",
            "}",
        ]

        # Generate scratch getters
        lines.append("")
        lines.append("/* =========================================")
        lines.append(" * Scratch Buffer Getters")
        lines.append(" * =========================================")
        lines.append("")

        offset = 0
        for scr in scratch:
            aligned_offset = self.round_to_alignment(offset, scr.alignment)
            lines.append(f"void* ck_get_scratch_{scr.name}(void* pool) {{")
            lines.append(f"    (void)pool;")
            lines.append(f"    return (char*)g_memory_pool + {aligned_offset};")
            lines.append(f"}}")
            offset = aligned_offset + scr.size_bytes

        lines.append("")
        lines.append("/* =========================================")
        lines.append(" * Tensor Pointer Getters")
        lines.append(" * =========================================")
        lines.append("")

        for name, layout in sorted(tensors.items()):
            safe_name = name.upper().replace(" ", "_").replace("-", "_")
            offset_macro = f"CK_OFFSET_{safe_name}"
            lines.append(f"float* ck_get_tensor_{name}(void* pool) {{")
            lines.append(f"    (void)pool;")
            lines.append(f"    return (float*)((char*)g_memory_pool + {offset_macro});")
            lines.append(f"}}")

        lines.append("")
        lines.append("#endif // CK_MEMORY_LAYOUT_C")

        return "\n".join(lines)


def process_ir_file(ir_path: Path, config: Dict) -> Tuple[Dict[str, TensorLayout], List[ScratchBuffer]]:
    """Process a single IR file."""
    ir = json.loads(ir_path.read_text())
    gen = MemoryLayoutGenerator(
        max_tokens=config.get("max_tokens", 8192),
        max_embed_dim=config.get("max_embed_dim", 8192),
    )
    return gen.process_ir(ir)


def main():
    parser = argparse.ArgumentParser(description="Generate memory layout from IR")
    parser.add_argument("--ir", help="Single IR file")
    parser.add_argument("--ir-dir", help="Directory of IR files")
    parser.add_argument("--output", "-o", default="src/memory_layout.c",
                        help="Output C file")
    parser.add_argument("--header", "-H", default="src/memory_layout.h",
                        help="Output header file")
    parser.add_argument("--max-tokens", "-T", type=int, default=8192,
                        help="Maximum sequence length")
    parser.add_argument("--max-embed", "-E", type=int, default=8192,
                        help="Maximum embedding dimension")
    args = parser.parse_args()

    config = {
        "max_tokens": args.max_tokens,
        "max_embed_dim": args.max_embed,
    }

    gen = MemoryLayoutGenerator(
        max_tokens=args.max_tokens,
        max_embed_dim=args.max_embed,
    )

    # Collect all tensors and scratch from all IR files
    all_tensors = {}
    all_scratch = []

    if args.ir:
        tensors, scratch = process_ir_file(Path(args.ir), config)
        all_tensors.update(tensors)
        all_scratch.extend(scratch)
    elif args.ir_dir:
        for ir_file in sorted(Path(args.ir_dir).glob("*.json")):
            tensors, scratch = process_ir_file(ir_file, config)
            all_tensors.update(tensors)
            all_scratch.extend(scratch)

    # Generate C files
    header_content = gen.generate_c_header(all_tensors, all_scratch)
    source_content = gen.generate_c_source(all_tensors, all_scratch)

    Path(args.header).write_text(header_content)
    Path(args.output).write_text(source_content)

    # Summary
    total_scratch = sum(s.size_bytes for s in all_scratch)
    total_tensor = sum(t.total_size for t in all_tensors.values())

    print(f"[ok] Generated memory layout")
    print(f"     Tensors: {len(all_tensors)} ({total_tensor:,} bytes)")
    print(f"     Scratch: {len(all_scratch)} buffers ({total_scratch:,} bytes)")
    print(f"     Written to {args.header} and {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
