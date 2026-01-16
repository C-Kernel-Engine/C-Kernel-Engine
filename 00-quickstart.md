# Quickstart Guide

This guide will get you up and running with the C-Kernel-Engine in less than 5 minutes.

## Prerequisites

- **Linux** (tested on Ubuntu 20.04+)
- **GCC** (with OpenMP support)
- **Make**
- **Python 3** (for running tests and PyTorch comparisons)
- **PyTorch** (only for running comparison tests)

## 1. Build the Engine

The project uses a standard Makefile. To build the shared library and the IR demo tool:

```bash
# Core runtime (kernels + orchestration)
make

# IR + codegen tool (HF config.json -> IR -> generated C)
make build/ck_ir_demo

# Optional: build the orchestrator CLI ("ck")
make ck-cli
```

This will create:
- `build/libckernel_engine.so`: The main runtime library (kernels + orchestration).
- `build/ck_ir_demo`: The compiler tool that converts `config.json` -> generated C.
- `build/ck`: Optional CLI that wires download/convert/codegen for you.

## 2. Run the Compiler Demo

The "Hello World" of this engine is compiling a standard Llama-style configuration into a C runtime.

We have a default configuration file ready: `default.config.json`.

```bash
make ck
```

**What just happened?**
1. The tool parsed `default.config.json`.
2. It generated an **Intermediate Representation (IR)** of the model's compute graph.
3. It emitted a `generated_model.c` file (conceptually) or printed the skeleton to stdout.

You should see output like:

```text
=== Forward IR ===
CKIRGraph: layers=32, hidden_size=4096 ...
  L0 N0 RMSNORM       outputs=[L0:N0:0]              inputs=[IN]
  L0 N1 LINEAR_QKV    outputs=[L0:N1:0]              inputs=[L0:N0]
  ...
```

## 3. Run the Unit Tests

We use Python to verify that our C kernels match PyTorch's output exactly.

```bash
make test
```

This command will:
1. Build individual shared libraries for each kernel family (e.g., `libckernel_gelu.so`).
2. Run the Python scripts in `unittest/`.
3. Report `OK` if the C implementation matches PyTorch within floating-point tolerance.

## 4. Generate a Standalone Runtime

To generate a full C file `ai.c` that you could compile and run (this feature is currently in active development):

```bash
./build/ck_ir_demo default.config.json --emit build/ai.c
```

Open `build/ai.c` to see how the engine structures the forward pass using the "Header / Block / Footer" pattern.

## 5. Generate a `libmodel.so` With Prefill + Decode (KV Cache)

For inference you typically want:
- **Prefill** once (process the whole prompt).
- **Decode** many times (one token at a time) using a KV cache.

The codegen tool can emit a library-mode C file that exports a stable API:

```bash
./build/ck_ir_demo default.config.json --emit build/model.c --emit-lib
```

This produces:
- `build/model.c` (generated model runtime + exported API)
- `build/model.c.kernels` (one kernel source path per line to link into the shared object)

Compile it into a self-contained shared library:

```bash
cc -O3 -fPIC -fopenmp -shared -Iinclude -o build/libmodel.so build/model.c $(cat build/model.c.kernels) -lm
```

### 5.1 Inference Call Sequence

At runtime:

- Initialize: `ck_model_init(weights.bump)`
- Enable KV cache: `ck_model_kv_cache_enable(capacity)`
- **Prefill**: `ck_model_embed_tokens(prompt_tokens, n)` then `ck_model_forward(NULL)`
- **Decode**: repeatedly call `ck_model_decode(next_token, NULL)`

The helpers `ck_model_get_logits()` / `ck_model_get_active_tokens()` let you read logits for sampling.

### 5.2 Training / Backprop

Training uses the full forward+backward graph and does **not** use KV-cache decode:

- Enable training: set `CK_ENABLE_TRAINING=1` before `ck_model_init(...)`, or call `ck_model_enable_training(lr)`
- Run `ck_model_forward(NULL)` then `ck_model_backward(tokens, targets, &loss)`

KV-cache decode is explicitly **disabled when training is enabled**.
