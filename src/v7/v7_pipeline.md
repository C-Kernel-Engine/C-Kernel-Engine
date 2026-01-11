v7 Training Pipeline
====================

v7 builds on v6 inference and adds explicit training codegen.

Key principle: Same philosophy as v6 - generate explicit C code,
no runtime dispatch, model-specific binary.

Requirements
------------
- Start with FP32 weights (safetensors from HF)
- Quantized weights cannot be trained directly
- Memory planner determines checkpoint strategy

Build Flow
----------
  1. Download FP32 weights (safetensors)
  2. Convert → weights.bump (FP32)
  3. IR generation with training metadata
  4. Memory planner (activations, gradients, optimizer state)
  5. Generate: ck-kernel-backprop.c, ck-kernel-optimizer.c
  6. Compile: ck-engine-<model> --train
  7. Run: ./ck-engine-<model> --train weights.bump

Memory Layout (BUMP training)
-----------------------------
+---------------------------+  ← base pointer
| weights (FP32)            |  0
+---------------------------+
| activations (checkpoint)  |  after weights
+---------------------------+
| gradients                 |  after activations
+---------------------------+
| optimizer m (Adam)        |  after gradients
+---------------------------+
| optimizer v (Adam)        |  after optimizer m
+---------------------------+

Memory Planner (IR extension)
-----------------------------
{
  "model_config": { ... },
  "training_config": {
    "optimizer": "adam",
    "batch_size": 1,
    "seq_len": 512,
    "available_memory_gb": 4
  },
  "memory_plan": {
    "weights_bytes": 1000000000,
    "activations_per_layer": 50000000,
    "gradients_bytes": 1000000000,
    "optimizer_bytes": 2000000000,
    "checkpoint_every": 4,          // 0 = save all, N = checkpoint every N layers
    "total_required": 4500000000,
    "fits_in_memory": true
  }
}

Codegen Requirements
--------------------
1. Load IR
2. Check kernel_requirements.backward:
   For each op in forward_kernels:
     → Does backward kernel exist in src/kernels/?
3. If ANY missing:
   → ERROR: "Cannot generate training binary"
4. If ALL exist:
   → Generate ck-kernel-backprop.c
   → Stitch backward kernels in reverse layer order

Generated Files
---------------
- ck-kernel-backprop.c   # Gradient computation
- ck-kernel-optimizer.c  # Adam/SGD weight updates
- ck-kernel-sched.c      # LR scheduler
- ck-kernel-train.c      # Main training loop

Layer Backward Stitch Example
-----------------------------
Layer L forward:
  matmul → attention → layernorm → silu → matmul

Layer L backward (reverse):
  d_silu ← silu_backward
  d_matmul2 ← matmul_backward (input grad)
  d_ln ← layernorm_backward
  d_attn ← attention_backward
  d_matmul1 ← matmul_backward (weight grad)

Version Bootstrap (from v6)
---------------------------
Copy from v6:
- scripts/download_model_v6.py → scripts/download_model_v7.py
- scripts/convert_gguf_to_bump_v6.py → scripts/convert_to_bump_v7.py
- src/v6/build.sh → src/v7/build.sh
- src/v6/v6_inference.c → src/v7/v7_train.c

DO NOT copy:
- kernels (src/kernels/*) - these are version-agnostic
- generated artifacts - keep in ~/.cache/ck-engine-v7/models/*

Notes
-----
- v7 is TRAINING ONLY - inference remains v6
- Start with small model: SmolLM-135M or Qwen2-0.5B
- First phase: memory planner + checkpoint strategy
- Second phase: backprop codegen
- Third phase: optimizer kernels
