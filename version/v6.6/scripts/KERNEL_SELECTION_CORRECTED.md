# Kernel Selection: The Real Two-Level System

## The Mistake

Previously, I incorrectly described a three-level system:
```
Template Ops → Kernel Op Types → Kernel Functions  (WRONG!)
```

## The Correct Two-Level System

```
Template Ops → Kernel Registry (C functions + selection logic)
```

The kernel registry **already contains**:
1. C function names
2. Selection constraints (quant, mode, shape)

There's **no need** for an intermediate "kernel op type" abstraction!

## Real-World Example: matmul

### Template Level

```json
{
  "ops": [
    "qkv_proj",    // Matrix multiply
    "out_proj",    // Matrix multiply
    "mlp_down",    // Matrix multiply
    "logits"       // Matrix multiply
  ]
}
```

All four operations are matrix multiplications, but the kernel depends on:
- **Quantization**: q8_0, q4_k, fp32
- **Mode**: decode (M=1) vs prefill (M>1)
- **Shape**: Is M a scalar (GEMV) or matrix (GEMM)?

### Kernel Registry Level

```json
{
  "kernels": [
    {
      "op": "matmul",
      "c_function": "ck_gemv_q8_0_decode",
      "desc": "Vector-matrix multiply for decode",
      "selection": {
        "quant": {"weight": "q8_0"},
        "mode": "decode",
        "constraints": {
          "batch_size": 1,
          "seq_len": 1
        }
      },
      "signature": {
        "inputs": ["x", "weight"],
        "outputs": ["out"],
        "shapes": {
          "x": "[E]",           // Vector (1D)
          "weight": "[E, V]",   // Matrix (2D)
          "out": "[V]"          // Vector (1D)
        }
      }
    },
    {
      "op": "matmul",
      "c_function": "ck_gemm_q8_0_prefill",
      "desc": "Matrix-matrix multiply for prefill",
      "selection": {
        "quant": {"weight": "q8_0"},
        "mode": "prefill",
        "constraints": {
          "seq_len": "> 1"  // Batched
        }
      },
      "signature": {
        "inputs": ["x", "weight"],
        "outputs": ["out"],
        "shapes": {
          "x": "[T, E]",        // Matrix (2D)
          "weight": "[E, V]",   // Matrix (2D)
          "out": "[T, V]"       // Matrix (2D)
        }
      }
    },
    {
      "op": "matmul",
      "c_function": "ck_gemm_q4_k_prefill",
      "desc": "Matrix-matrix multiply for prefill (q4_k quant)",
      "selection": {
        "quant": {"weight": "q4_k"},
        "mode": "prefill",
        "constraints": {
          "seq_len": "> 1"
        }
      },
      "signature": {
        "inputs": ["x", "weight"],
        "outputs": ["out"],
        "shapes": {
          "x": "[T, E]",
          "weight": "[E, V]",  // q4_k quantized
          "out": "[T, V]"
        }
      }
    }
  ]
}
```

## Selection Algorithm

```python
def select_kernel(
    op_name: str,
    weight_dtype: str,
    input_shapes: Dict[str, tuple],
    mode: str  # "decode" or "prefill"
) -> str:
    """
    Select kernel from registry based on runtime constraints.

    Args:
        op_name: Template op name (e.g., "qkv_proj", "mlp_down")
        weight_dtype: Weight quantization (e.g., "q8_0", "q4_k")
        input_shapes: Actual input tensor shapes
        mode: Execution mode ("decode" or "prefill")

    Returns:
        C function name to call
    """
    # Map template op to kernel op
    kernel_op = TEMPLATE_TO_KERNEL_MAP[op_name]  # e.g., "qkv_proj" → "matmul"

    # Get all kernels for this operation
    candidates = [k for k in registry["kernels"] if k["op"] == kernel_op]

    # Filter by quantization
    candidates = [k for k in candidates
                  if k["selection"]["quant"]["weight"] == weight_dtype]

    # Filter by mode
    candidates = [k for k in candidates
                  if k["selection"]["mode"] == mode]

    # Filter by shape constraints
    for kernel in candidates:
        constraints = kernel["selection"]["constraints"]

        # Check batch size
        if "batch_size" in constraints:
            if not check_constraint(constraints["batch_size"],
                                   input_shapes["x"][0]):
                continue

        # Check sequence length
        if "seq_len" in constraints:
            seq_len = input_shapes["x"][1] if len(input_shapes["x"]) > 1 else 1
            if not check_constraint(constraints["seq_len"], seq_len):
                continue

        # Found matching kernel!
        return kernel["c_function"]

    raise ValueError(f"No kernel found for {op_name} with {weight_dtype}, {mode}")


def check_constraint(constraint_expr: str, value: int) -> bool:
    """Evaluate constraint expression."""
    if isinstance(constraint_expr, int):
        return value == constraint_expr
    elif constraint_expr == "> 1":
        return value > 1
    elif constraint_expr == "== 1":
        return value == 1
    # ... more constraint types
```

## Example: Runtime Selection

```python
# Template says: "out_proj"
# Runtime info:
#   - weight dtype: "q8_0"
#   - mode: "decode"
#   - input shape: [512] (vector, not matrix)

kernel = select_kernel(
    op_name="out_proj",
    weight_dtype="q8_0",
    input_shapes={"x": (512,)},  # 1D vector
    mode="decode"
)
# Result: "ck_gemv_q8_0_decode"

# Different runtime:
#   - weight dtype: "q8_0"
#   - mode: "prefill"
#   - input shape: [128, 512] (matrix)

kernel = select_kernel(
    op_name="out_proj",
    weight_dtype="q8_0",
    input_shapes={"x": (128, 512)},  # 2D matrix
    mode="prefill"
)
# Result: "ck_gemm_q8_0_prefill"
```

## Why This Is Better

### Before (3-level, wrong):
```
"out_proj" → "gemv" (intermediate) → "ck_gemv_q8_0_decode" (C function)
              ↑
              Unnecessary abstraction!
```

### After (2-level, correct):
```
"out_proj" → lookup in registry → "ck_gemv_q8_0_decode" (C function)
              ↑
              Direct! Selection logic is data-driven.
```

## Selection Logic Belongs in Registry

The kernel registry should encode **all selection logic**:

```json
{
  "op": "attention",
  "c_function": "ck_flash_attention_decode",
  "selection": {
    "quant": {"weight": "none"},
    "mode": "decode",
    "constraints": {
      "seq_len": "== 1",
      "use_flash": true
    },
    "hardware": {
      "min_simd_width": 256  // Requires AVX2
    }
  }
}
```

The selection algorithm reads these constraints and picks the right kernel **dynamically** based on:
1. **Quantization** (from manifest)
2. **Mode** (decode vs prefill)
3. **Shape** (from input tensors)
4. **Hardware** (detected at runtime)

## Fusion Is Also Selection Logic

```json
{
  "op": "attention_block",
  "c_function": "ck_fused_attn_block_prefill",
  "desc": "Fused: rmsnorm + qkv_proj + rope + attn + out_proj",
  "selection": {
    "quant": {"weight": "q8_0"},
    "mode": "prefill",
    "fusion": {
      "ops": ["rmsnorm", "qkv_proj", "rope", "attn", "out_proj"],
      "enabled": "prefill_only"  // Don't fuse in decode
    }
  }
}
```

Selection algorithm can choose:
- **Fused kernel** (prefill): Single call to `ck_fused_attn_block_prefill`
- **Separate kernels** (decode): Individual calls to each op

## Op Builders Should Generate GraphIR Directly

```python
# WRONG - intermediate "kernel op"
def build_out_proj_op(op_node, ctx, manifest):
    return Op(
        op="gemv",  # Intermediate abstraction (unnecessary!)
        kernel=None  # Filled later
    )

# RIGHT - template op with selection metadata
def build_out_proj_op(op_node, ctx, manifest):
    return Op(
        template_op="out_proj",  # From template
        kernel_op="matmul",      # Logical operation (for registry lookup)
        kernel=None,             # Selected at runtime based on quant/mode/shape
        selection_metadata={     # Metadata for kernel selection
            "weight_dtype": manifest.get_weight_dtype(op_node.weights[0]),
            "expected_shape_rank": 1  # Decode: vector
        }
    )
```

## The Correct Flow

```
┌──────────────────────────────────────────────────────────────┐
│ COMPILE TIME: Template → GraphIR                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ Template: "out_proj"                                         │
│     ↓                                                        │
│ Op Builder creates:                                          │
│   Op(                                                        │
│     template_op="out_proj",                                 │
│     kernel_op="matmul",  ← For registry lookup              │
│     kernel=None,         ← Selected at runtime              │
│     weights=["wo"],                                         │
│     selection_metadata={                                     │
│       "weight_dtype": "q8_0",  ← From manifest              │
│       "mode": "decode"          ← From execution context     │
│     }                                                        │
│   )                                                          │
│                                                              │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ RUNTIME: Kernel Selection                                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ For each op:                                                 │
│   1. Get op.kernel_op → "matmul"                            │
│   2. Get op.selection_metadata → {weight_dtype, mode}       │
│   3. Get input shapes → (512,) [vector]                     │
│   4. Query registry:                                         │
│        op="matmul" AND                                       │
│        quant="q8_0" AND                                      │
│        mode="decode" AND                                     │
│        matches_shape_constraints((512,))                     │
│   5. Result: kernel = "ck_gemv_q8_0_decode"                 │
│   6. Call: ck_gemv_q8_0_decode(x, weight, out)             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Summary

**You were right!** The system is actually two levels, not three:

1. **Template** (architecture): WHAT ops the model needs
2. **Kernel Registry** (implementations + selection logic): HOW to implement + WHEN to use

The "kernel op type" I introduced was an unnecessary abstraction. The kernel registry already contains:
- C function names
- Selection constraints (quant, mode, shape)
- Everything needed to pick the right kernel

Selection should be **data-driven** and **dynamic**, not a static mapping!
