# IR and Codegen Design for C-Kernel-Engine

This document captures how the IR (intermediate representation) and code
generation pipeline for C-Kernel-Engine is evolving. The goal is to go from a
HuggingFace `config.json` to a self-contained C runtime (inference + training)
without hand-writing a 15k-line `main.c` for every model.

The overall flow mirrors the way the `@antsand` HMVC website generator treated
sites as **header / body / footer**:

- **Header** – one-time setup before the main stack.
- **Body** – repeated structure (pages / modules / blocks).
- **Footer** – one-time finalization and wiring at the end.

For LLM decoders, we apply the same idea.

---

## 1. Big Picture

Pipeline (high level):

1. **HF config → canonical IR in memory**  
   - Parse `config.json` into a `CKModelConfig`.
   - Build a raw IR of kernels and wiring for the decoder stack.

2. **IR → IR file (map)**  
   - Serialize the IR (and later shapes) to a JSON map file.
   - This is the machine-readable, debuggable description of the model.

3. **IR file → structured IR program (header / block / footer)**  
   - Load the IR file and organize it as:
     - `header`: ops before the decoder stack.
     - `block`: per-layer ops (double array: layers × ops).
     - `footer`: ops after the decoder stack.

4. **IR program → memory layout (bump allocator)**  
   - Add shape metadata for all relevant tensors.
   - Compute activation and gradient buffer offsets.

5. **IR program + layout → generated C (`ai.c`)**  
   - Emit a `ModelConfig`, buffer sizes, per-layer forward & backward loops,
     and a `main` entrypoint that calls the kernels in order.

From there we add:

- **Fusion passes** (RMSNorm+Linear, QKV+Attention, etc.).
- **Backend selection** (plain C vs AVX vs GPU).
- **PyTorch parity tests** for forward and backward.

---

## 2. Canonical Node-Level IR (CKIRGraph)

At the core we have a primitive IR over **kernel invocations**, not big layers:

```c
typedef enum {
    CK_OP_RMSNORM = 0,
    CK_OP_LINEAR_QKV,
    CK_OP_ATTENTION,
    CK_OP_ADD,
    CK_OP_LINEAR,
    CK_OP_SPLIT,
    CK_OP_SWIGLU,
    // Backward ops:
    CK_OP_RMSNORM_BWD,
    CK_OP_LINEAR_QKV_BWD,
    CK_OP_ATTENTION_BWD,
    CK_OP_ADD_BWD,
    CK_OP_LINEAR_BWD,
    CK_OP_SPLIT_BWD,
    CK_OP_SWIGLU_BWD
} CKOpType;

typedef struct {
    uint16_t layer;  // decoder layer index
    uint16_t node;   // kernel index within the layer
} CKKernelId;

typedef struct {
    CKKernelId producer;  // which kernel produced this tensor
    uint8_t    out_index; // output slot (0 for most kernels)
} CKInputRef;

typedef struct {
    CKKernelId id;
    CKOpType   op;
    CKInputRef inputs[4];
    uint8_t    n_inputs;
    uint8_t    n_outputs;
} CKIRNode;

typedef struct {
    CKModelConfig config; // parsed from HF config
    int           num_nodes;
    CKIRNode     *nodes;
} CKIRGraph;
```

This is what the current `ck_ir_dump` prints in a human-friendly way:

```text
CKIRGraph: layers=40, hidden_size=5120, intermediate_size=32768, heads=32, kv_heads=8
  L28 N0 RMSNORM       outputs=[L28:N0:0]              inputs=[IN]
  L28 N1 LINEAR_QKV    outputs=[L28:N1:0]              inputs=[L28:N0]
  L28 N2 ATTENTION     outputs=[L28:N2:0]              inputs=[L28:N1]
  L28 N3 ADD           outputs=[L28:N3:0]              inputs=[IN,L28:N2]
  L28 N4 RMSNORM       outputs=[L28:N4:0]              inputs=[L28:N3]
  L28 N5 LINEAR        outputs=[L28:N5:0]              inputs=[L28:N4]
  L28 N6 SPLIT         outputs=[L28:N6:0,L28:N6:1]     inputs=[L28:N5]
  L28 N7 SWIGLU        outputs=[L28:N7:0]              inputs=[L28:N6,L28:N6]
  L28 N8 LINEAR        outputs=[L28:N8:0]              inputs=[L28:N7]
  L28 N9 ADD           outputs=[L28:N9:0]              inputs=[L28:N3,L28:N8]
```

Design choices:

- **Op granularity** is at the C-kernel level: RMSNorm, attention, SwiGLU, etc.
- **GQA vs MHA** is implicit via `num_heads` and `num_kv_heads`; the kernel
  sees shapes and head counts, not separate flags.
- **Residuals, splits, and merges** are explicit ADD and SPLIT nodes; kernels
  stay oblivious to higher-level wiring.

Backward IR mirrors this structure in reverse order, mapping each op to its
`*_BWD` counterpart. It is currently a skeleton, and will evolve to track
explicit gradient wiring.

---

## 3. Header / Block / Footer Layout

To make the execution structure explicit and avoid per-node flags, we adopt a
three-part program layout inspired by the HMVC website generator:

- **Header** – one-time ops before the decoder stack:
  - Token embedding.
  - Positional encoding (RoPE, ALiBi, etc.).
  - Any global setup work.

- **Block (Body)** – repeated per-layer ops, organized as a **double array**:

  ```c
  typedef struct {
      CKIRNode *ops;  // ops in this layer
      int       count;
  } CKIRLayer;

  typedef struct {
      CKModelConfig config;
      CKIRNode     *header;
      int           num_header;
      CKIRLayer    *layers;   // array of length config.num_layers
      int           num_layers;
      CKIRNode     *footer;
      int           num_footer;
  } CKIRProgram;
  ```

  In JSON form, the block becomes an array of layers, each an array of ops:

  ```json
  "block": [
    [
      { "op": "RMSNORM",    "inputs": ["IN"],      "outputs": ["L0:N0:0"] },
      { "op": "LINEAR_QKV", "inputs": ["L0:N0:0"], "outputs": ["L0:N1:0"] },
      ...
    ],
    [
      { "op": "RMSNORM",    "inputs": ["IN"],      "outputs": ["L1:N0:0"] },
      ...
    ]
  ]
  ```

- **Footer** – one-time ops after the decoder stack:
  - Final norm (e.g., RMSNorm on `h_last`).
  - LM head (with weight tying to embeddings).
  - Loss node for training (`CROSS_ENTROPY` or similar).

This layout maps cleanly to codegen:

```c
void run_model_forward(const CKIRProgram *p, ModelBuffers *buf, Weights *w) {
    run_header_forward(p, buf, w);
    for (int layer = 0; layer < p->num_layers; ++layer) {
        run_block_forward(p, layer, buf, w);
    }
    run_footer_forward(p, buf, w);
}
```

No `is_loop_body` flags are needed; the section (`header` vs `block` vs `footer`)
tells us how to execute each node.

---

## 4. Memory Planning and Bump Allocator

Once we have `CKIRProgram` (header/block/footer) we add tensor shape metadata
and run a memory planner:

- For each node output (and key intermediates) we track a tensor descriptor
  (rank + dims).
- A planner walks the IR and assigns activation and gradient offsets into a
  single contiguous `act` buffer (and optional `grad` buffer).

The planner eventually produces:

- `ACT_BYTES`, `GRAD_BYTES`.
- A mapping from `(section, layer, node, out_slot)` → byte offset.

Codegen uses this to emit:

```c
enum {
    OFF_L0_N0 = 0,
    OFF_L0_N1 = OFF_L0_N0 + ...,
    // ...
};
static const size_t ACT_BYTES  = ...;
static const size_t GRAD_BYTES = ...;
```

and then computes pointers inside the generated forward/backward loops using
these offsets.

---

## 5. Codegen: Forward & Backward Engines + Main

Using `CKIRProgram` and the layouts, codegen emits a file like `ai.c`:

- `ModelConfig` with `num_layers`, `hidden_size`, `intermediate_size`,
  `num_heads`, `num_kv_heads`, etc.
- `ModelBuffers` struct with `act` and `grad` pointers.
- Per-layer **forward engine**:

  ```c
  void run_block_forward(const CKIRProgram *p, int layer,
                         ModelBuffers *buf, Weights *w) {
      CKIRLayer *L = &p->layers[layer];
      for (int i = 0; i < L->count; ++i) {
          CKIRNode *n = &L->ops[i];
          // compute input/output pointers from offsets
          // switch(n->op) { call rmsnorm_forward, attention_forward, swiglu_forward, ... }
      }
  }
  ```

- Per-layer **backward engine** (similar, but reverse layer order and `_BWD` ops).
- A small `main`:

  ```c
  int main(void) {
      ModelBuffers buf = {0};
      buf.act  = malloc(ACT_BYTES);
      buf.grad = malloc(GRAD_BYTES);
      // Load weights, inputs
      run_model_forward(&program, &buf, &weights);
      // run_model_backward(&program, &buf, &weights, &grads); // for training
      return 0;
  }
  ```

The detailed `(Lx,Ny)` labeling survives as comments and/or enums in the
generated C, so we can still trace the exact path an activation or gradient
takes through the model.

---

## 6. Library Mode: Prefill + KV-Cache Decode + Backprop

In addition to emitting a standalone `main`, the codegen tool supports a
**library mode** (`--emit-lib`) that generates a `model.c` exporting a small,
stable ABI for `dlopen` / `ctypes` use:

```bash
./build/ck_ir_demo path/to/config.json --emit build/model.c --emit-lib
```

Alongside `build/model.c`, codegen writes a `build/model.c.kernels` manifest
(one kernel source path per line) so you can build a self-contained shared object:

```bash
cc -O3 -fPIC -fopenmp -shared -Iinclude -o build/libmodel.so build/model.c $(cat build/model.c.kernels) -lm
```

**Exported API (high level):**
- Common: `ck_model_init`, `ck_model_embed_tokens`, `ck_model_forward`, `ck_model_get_logits`, `ck_model_get_active_tokens`, `ck_model_free`
- Inference (KV cache): `ck_model_kv_cache_enable/reset/get_tokens`, `ck_model_decode`
- Training: `ck_model_enable_training`, `ck_model_disable_training`, `ck_model_backward`

**Prefill vs decode:**
- **Prefill** runs the full forward pass over `T` prompt tokens.
- **Decode** runs a *single-token* forward step using a KV cache (fast path).

Decode expects KV in a fixed-stride cache layout:
`[kv_head, cache_capacity, aligned_head_dim]`. Prefill kernels naturally write a
packed layout of `[kv_head, tokens, aligned_head_dim]`, so when KV-cache is
enabled the generated forward code repacks K/V into the cache layout once after
prefill.

KV-cache decode is explicitly **disabled when training is enabled** so the
training forward/backward graph stays consistent.

---

## 7. Relation to Existing Kernels

C-Kernel-Engine already provides optimized kernels in `src/kernels/*.c` and the
math backend interface in `ckernel_engine.h`:

- GEMMs for projections and MLPs.
- RMSNorm / LayerNorm forward/backward.
- Scaled dot-product attention with causal masking.
- SwiGLU and Sigmoid.

The IR + codegen layer is intentionally thin and mechanical: it wires these
small kernels together according to the model description, rather than
hand-writing one giant `main.c` per architecture.

This keeps:

- Kernels small, testable, and reusable.
- Model wiring data-driven (via config + IR), not tangled with math.
- The codegen pipeline free to add fusion and layout optimizations without
  touching each kernel.

---

## 8. Future Work

Short-term:

- Serialize IR to JSON and add an IR parser to reconstruct `CKIRProgram`.
- Add tensor shapes and a simple (non-reusing) memory planner.
- Generate real kernel calls for inference in `ai.c`, starting with a single
  block and then looping over all layers.
- Flesh out backward IR and generated backward engine.

Medium-term:

- Fusion passes (RMSNorm+Linear, QKV+Attention, etc.).
- Per-op backend selection (plain C vs AVX2/AVX-512, etc.).
- Test harnesses that compare IR-driven forward/backward to PyTorch for
  block-level and model-level tests.

Long-term:

- Support for encoder-decoders and vision+text models by extending the header
  and footer sections, while keeping the block design for the core decoder.
