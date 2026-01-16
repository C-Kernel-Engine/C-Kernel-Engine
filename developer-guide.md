# Developer Guide

So you want to hack on the engine? Welcome!

This guide explains how to add a new kernel (operation) to the system, from the math implementation to the compiler integration.

## Workflow: Adding a New Kernel

Let's say you want to add a `GELU_TANH` op.

### 1. Define the Op in IR

Open `include/ckernel_ir.h` and add your op to the `CKOpType` enum:

```c
typedef enum {
    // ...
    CK_OP_SWIGLU,
    CK_OP_GELU_TANH, // <--- Add this
    // ...
} CKOpType;
```

### 2. Implement the Kernel

Create a new file `src/kernels/gelu_tanh.c` (or add to `src/kernels/gelu_kernels.c` if it fits):

```c
#include <math.h>

void gelu_tanh_forward(const float *x, float *y, int n) {
    // fast tanh approximation...
    for (int i = 0; i < n; ++i) {
        // ...
    }
}
```

Then expose it in `include/ckernel_engine.h`:

```c
void gelu_tanh_forward(const float *x, float *y, int n);
```

### 3. Register the Op Name

Open `src/ckernel_registry.c` and map the enum to a string name:

```c
static const char *ck_op_name(CKOpType op) {
    switch (op) {
        // ...
        case CK_OP_GELU_TANH: return "GELU_TANH";
        // ...
    }
}
```

And ensure `ck_op_supported` returns 1 for it.

### 4. Update the Code Generator

Open `src/ckernel_codegen.c`. You need to ensure the runtime knows how to emit code for this op.

First, update the local `op_name` helper if it duplicates the registry logic.

Then, if you are emitting a full runtime (in `emit_runtime_preamble` or the main loop inside `ck_codegen_emit_runtime`), you need to make sure your kernel's source file is included in the output `ai.c`.

For example, in `emit_runtime_preamble`:

```c
if (emit_source_filtered(out, "src/kernels/gelu_tanh.c") != 0) return -1;
```

### 5. Add a Test

We verify everything against PyTorch. Create `unittest/test_gelu_tanh.py`:

```python
import torch
import ctypes
import numpy as np

# Load your new library (you might need to add a target to Makefile first!)
lib = ctypes.CDLL("build/libckernel_gelu.so") 

def test_gelu_tanh():
    # ... setup input ...
    # ... call C function ...
    # ... compare with torch.nn.functional.gelu(..., approximate='tanh') ...
```

### 6. Build and Verify

```bash
make
make test
```

## Compiler Architecture

The "Compiler" is just `ckernel_ir_demo.c`. It doesn't generate machine code directly; it generates **C source code**. This is known as "Transpilation" or "Source-to-Source" compilation.

1. **Parser**: Reads `config.json` -> `CKModelConfig`.
2. **IR Builder**: Constructs `CKIRGraph` (a flat list of nodes).
3. **Codegen**: Walks the `CKIRGraph` and prints C code to a file (`ai.c`).

This generated `ai.c` is **standalone**. It contains all the kernel code (inlined/included) and a `main()` function. You can compile it with `gcc -O3 ai.c -o ai` and run it anywhere, without needing the original library.
