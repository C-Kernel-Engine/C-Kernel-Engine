# IR-Level Logits Buffer Management Proposal

## Problem Statement

Currently, the `ck_model_decode` function in codegen has runtime logic:
```c
int token_pos = g_model->pos;
ck_decode(g_model, token);
if (token_pos > 0) {
    memmove(g_model->logits + token_pos * VOCAB_SIZE, g_model->logits, ...);
}
```

This violates the principle that **codegen should be dumb** - it should just emit
kernel calls from IR, not contain runtime logic.

## Proposed Solution: IR Operations for Buffer Management

### New IR Operations

1. **`logits_store_indexed`** - Store logits at a position-indexed location
   ```json
   {
     "op": "logits_store_indexed",
     "kernel": "logits_copy_to_position",
     "args": [
       {"name": "src", "source": "scratch:logits_scratch"},
       {"name": "dst", "source": "activation:logits"},
       {"name": "position", "source": "state:pos"},
       {"name": "vocab_size", "source": "config:vocab_size"}
     ]
   }
   ```

2. **`logits_compute`** - Compute logits for current token(s)
   ```json
   {
     "op": "logits_compute",
     "kernel": "gemm_nt_q8_0_q8_0",
     "output_mode": "scratch",  // Write to scratch, not final buffer
     "args": [...]
   }
   ```

### IR Pipeline Changes

#### IR1 (Build Phase)
- `logits` op generates `logits_compute` + `logits_store_indexed` for decode mode
- For prefill mode, `logits_compute` writes directly (batch mode)

#### IR2 (Layout Phase)
- Allocate `logits_scratch` buffer: [1, vocab_size] for single-token decode
- `logits` buffer: [max_seq_len, vocab_size] for position-indexed storage

#### IR3 (Lower Phase)
- `logits_store_indexed` → `logits_copy_to_position(src, dst, pos, size)`
- This kernel is a simple memcpy/memmove with position offset

### Kernel Definition

```json
// version/v6.6/kernel_maps/logits_copy_to_position.json
{
  "kernel": "logits_copy_to_position",
  "signature": "void logits_copy_to_position(const float *src, float *dst, int pos, int vocab_size)",
  "description": "Copy logits from scratch to position-indexed location in buffer",
  "implementation": "memmove(dst + pos * vocab_size, src, vocab_size * sizeof(float))"
}
```

### Benefits

1. **Codegen becomes truly dumb**: Just emit function calls from IR
2. **Mode-agnostic**: Same codegen for prefill/decode, IR handles the difference
3. **Testable**: Each IR operation can be unit-tested independently
4. **Extensible**: Easy to add new buffer management patterns

## Implementation Plan

### Phase 1: Add IR Operation
1. Define `logits_store_indexed` in KERNEL_REGISTRY.json
2. Add kernel map for `logits_copy_to_position`
3. Implement the kernel in `src/kernels/buffer_kernels.c`

### Phase 2: Update IR Builder
1. In `build_ir_v6_6.py`, emit `logits_store_indexed` after logits compute for decode
2. Keep prefill unchanged (writes directly to logits buffer)

### Phase 3: Update Codegen
1. Remove the memmove hack from `ck_model_decode`
2. Let IR-generated code handle the copy

### Phase 3: Remove Codegen Hack
1. Remove memmove logic from `ck_model_decode` in codegen
2. Codegen just calls `ck_decode` - all buffer management is in the IR

## Alternative: State Machine in IR

Instead of position-based indexing, use an IR state machine:

```json
{
  "states": {
    "pos": {"type": "int", "init": 0}
  },
  "ops": [
    {"op": "logits_compute", ...},
    {"op": "logits_store", "at": "state:pos"},
    {"op": "state_increment", "var": "pos"}
  ]
}
```

This makes the IR fully describe the runtime behavior.

## Conclusion

The key insight is: **If codegen needs if-statements or loops, that logic belongs in IR.**

Codegen should be:
```python
for op in ir_ops:
    emit(f"{op.kernel}({', '.join(op.args)})")
```

Nothing more.
