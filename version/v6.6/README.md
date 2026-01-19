# C-Kernel-Engine v6.6

Self-contained version 6.6 of the C-Kernel-Engine inference system.

## Directory Structure

```
v6.6/
├── scripts/                    Python toolchain
│   ├── build_ir_v6_6.py       IR builder and lowering
│   ├── codegen_v6_6.py        C code generator
│   ├── convert_gguf_to_bump_v6_6.py   GGUF → bump converter
│   ├── v6_6_ir_lowering.py    Memory layout planner
│   ├── kernel_registry.py     Kernel I/O specifications (NEW)
│   ├── fusion_patterns.py     Fusion pattern definitions
│   ├── ir_types_v6_6.py       Core IR data structures
│   ├── compat_ir_v4_v6_6.py   V4 compatibility (HF conversion)
│   ├── compat_codegen_v4_v6_6.py  V4 codegen fallback
│   └── ...
├── src/                        C source code
│   ├── generated/             Codegen output
│   ├── ck_cli_v6.6.c          CLI entry point
│   ├── ckernel_codegen_v6.6.c Runtime codegen
│   └── ...
└── README.md
```

## Key Files

### Core Scripts
| File | Purpose |
|------|---------|
| `build_ir_v6_6.py` | Builds IR from model config, applies fusion |
| `codegen_v6_6.py` | Generates C code from IR |
| `convert_gguf_to_bump_v6_6.py` | Converts GGUF models to bump format |
| `v6_6_ir_lowering.py` | Memory layout and buffer allocation |
| `kernel_registry.py` | **NEW** - Kernel I/O specs for memory planning |
| `fusion_patterns.py` | Defines fusable operation patterns |

### Copied Dependencies (for v6.6 independence)
| File | Origin | Purpose |
|------|--------|---------|
| `ir_types_v6_6.py` | v3/build_ir_v3.py | Core IR types: Buffer, ModelLayout, BumpAllocator |
| `compat_ir_v4_v6_6.py` | v4/build_ir_v4.py | HF model conversion utilities |
| `compat_codegen_v4_v6_6.py` | v4/codegen_v4.py | Legacy codegen fallback |

## Usage

### Convert GGUF Model
```bash
cd version/v6.6/scripts
python convert_gguf_to_bump_v6_6.py /path/to/model.gguf --output /path/to/output
```

### Build IR
```bash
python build_ir_v6_6.py --model qwen2.5-1.5b --output generated/
```

### Generate C Code
```bash
python codegen_v6_6.py --layout generated/layout.json --output generated/inference.c
```

## Dependencies

This version is self-contained. No imports from `scripts/v3/` or `scripts/v4/` outside this folder.

### Shared Components (from root)
- `src/kernels/` - Fused kernels (mega_fused_attention_prefill, etc.)
- `include/` - Common headers (ckernel_engine.h, ckernel_quant.h)

## What's New in v6.6

1. **Kernel Registry** (`kernel_registry.py`)
   - Central definition of kernel I/O requirements
   - Enables automatic memory planning
   - Foundation for fusion pass

2. **Fusion Integration** (planned)
   - `mega_fused_attention_prefill` - 1.45x speedup
   - `mega_fused_outproj_mlp_prefill` - 1.1x speedup

3. **Unified Buffer Allocation** (planned)
   - Same buffers for prefill and decode
   - Allocate once for max_context_len

## Development Notes

- Files with `compat_` prefix are compatibility layers from older versions
- The `_v6_6` suffix indicates the file belongs to this version
- Generated code goes in `src/generated/`
