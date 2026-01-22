# v6.6 Build Pipeline (Inference-Complete)

## Quick Reference

| Phase | Input | Output | Script |
|-------|-------|--------|--------|
| 1. Convert Model | GGUF/HF safetensors | BUMPWGT5 + weights_manifest.json | `convert_gguf_to_bump_v6_6.py` |
| 2. Registry | kernel_maps/*.json | KERNEL_REGISTRY.json | `gen_kernel_registry_from_maps.py` |
| 3. Build IR | template + manifest + registry | graph.json | `build_ir_v6_6.py` |
| 4. Lower | graph.json | lowered_*.json | `v6_6_ir_lowering.py` |
| 5. Fuse | lowered_*.json | fused IR | `fusion_patterns.py` |
| 6. Plan Parallel | fused IR | parallel_*.json | `parallel_planner.py` |
| 7. Layout | parallel_*.json | layout_*.json + schedule_*.json | built into `build_ir_v6_6.py` |
| 8. Codegen | all above | ck-kernel-*.c/h | `codegen_v6_6.py` |

---

## Full Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL INPUT                                                                  │
│   GGUF file  OR  HuggingFace safetensors + config.json                       │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. CONVERT → BUMPWGT5 + weights_manifest.json                               │
│    - Extract weights with proper quantization                               │
│    - Store tokenizer (vocab_offsets, vocab_strings, vocab_merges)           │
│    - Write EOF metadata: template, config, quant_summary, manifest_hash     │
│                                                                             │
│    Output: model.bump (weights + EOF) + weights_manifest.json (sidecar)    │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. GEN_REGISTRY → KERNEL_REGISTRY.json                                      │
│    - Scan version/v6.6/kernel_maps/*.json                                   │
│    - Validate kernel IDs exist                                              │
│    - Extract quant support per kernel                                       │
│                                                                             │
│    Output: kernel_maps/KERNEL_REGISTRY.json (20 kernels)                    │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. BUILD_GRAPH_IR                                                           │
│    - Load template (e.g., "qwen2" = rmsnorm→qkv→rope→attn→out→res→mlp...)  │
│    - Load weights_manifest.json (weight dtypes, offsets)                    │
│    - Generate per-layer ops with symbolic buffer names                      │
│                                                                             │
│    Output: graph.json (ops only, no kernel IDs yet)                        │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────┐            ┌──────────────────────────┐
│ 4a. LOWER        │            │ 4b. FUSION               │
│ Map ops → kernels│            │ Match patterns           │
│ via registry     │            │ (manual patterns)        │
│                  │            │                          │
│ qkv_proj →       │            │ rmsnorm+qkv →            │
│   ck_qkv_project_│            │   fused_rmsnorm_qkv_     │
│   head_major_quant               prefill_head_major_quant │
│                  │            │                          │
│ attn_proj →      │            │ attn+out+res →           │
│   ck_attention_  │            │   mega_fused_attention_  │
│   project_head_  │            │   prefill                │
│   major_quant    │            │                          │
└────────┬─────────┘            └───────────┬──────────────┘
         │                                │
         └───────────────┬────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. PARALLEL PLANNING                                                        │
│    - Analyze GEMM shapes (M=tokens, N=features)                            │
│    - Assign strategy: M_parallel (tokens) or H_parallel (heads)            │
│                                                                             │
│    Output: parallel_prefill.json, parallel_decode.json                     │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. LAYOUT + SCHEDULE                                                        │
│    - Compute exact buffer offsets for all activations/weights              │
│    - Account for quantization block sizes (Q4_K=256, Q8_0=32)             │
│    - Generate kernel call sequence with buffer pointers                    │
│                                                                             │
│    Outputs:                                                                 │
│      - layout_prefill.json  (prefill memory offsets)                       │
│      - layout_decode.json   (decode memory offsets)                        │
│      - schedule_prefill.json (kernel invocation order)                     │
│      - schedule_decode.json  (kernel invocation order)                     │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. CODEGEN                                                                  │
│    - Read: lowered_*.json, layout_*.json, parallel_*.json, weights_manifest│
│    - Emit:                                                                    │
│      - ck-kernel-prefill.c/h  (many-token forward pass)                    │
│      - ck-kernel-inference.c/h (single-token decode)                       │
│      - weights_manifest.map    (binary index for O(1) lookup)              │
│                                                                             │
│    Generated C includes:                                                    │
│      - MagicHeader struct (64-byte metadata at file start)                 │
│      - Model struct with all buffer pointers                               │
│      - Per-layer kernel calls with correct strides/offsets                 │
│      - OpenMP pragmas from parallel planning                               │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. COMPILE + RUN                                                            │
│                                                                             │
│    Compile:                                                                 │
│      gcc -O3 -march=native -fopenmp ck-kernel-prefill.c ... -lckernel      │
│                                                                             │
│    Run:                                                                     │
│      ./ck-engine-<model> weights.bump --prompt "Hello" --max-tokens 30     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Manifest vs Map

| Term | Purpose | Format |
|------|---------|--------|
| **weights_manifest.json** | Weight metadata (dtype, offset, size) | JSON (human-readable) |
| **weights_manifest.map** | O(1) weight lookup index | Binary (mmap-able) |

### Prefill vs Decode

| Mode | Tokens | Memory | Kernels |
|------|--------|--------|---------|
| **Prefill** | Many (1 to max_seq_len) | Large activation buffers | GEMM (batch matrix multiply) |
| **Decode** | Single token | Minimal (reuse prefill) | GEMV (single vector multiply) |

### Quantization Support

| Format | Block Size | Use Case |
|--------|------------|----------|
| Q4_K | 256 | W2 (down projection) |
| Q5_0 | 32 | Q, K, O projections |
| Q6_K | 256 | W2 (alternative) |
| Q8_0 | 32 | V projection, activations |

---

## Output Artifacts (in ~/.cache/ck-engine-v6.6/models/<model>/bump/ir_out/)

```
ir_out/
├── ck-kernel-prefill.c/h    # Prefill kernel implementation
├── ck-kernel-inference.c/h  # Decode kernel implementation
├── graph.json               # Template ops (before lowering)
├── lowered_prefill.json     # Kernel-selected IR (prefill)
├── lowered_decode.json      # Kernel-selected IR (decode)
├── layout_prefill.json      # Memory offsets (prefill)
├── layout_decode.json       # Memory offsets (decode)
├── parallel_prefill.json    # Parallelization strategy
├── parallel_decode.json     # Parallelization strategy
├── schedule_prefill.json    # Kernel execution order
├── schedule_decode.json     # Kernel execution order
├── fusion_decode.json       # Fusion statistics
├── weights_manifest.json    # Weight metadata (sidecar)
└── weights_manifest.map     # Binary weight lookup index
```

---

## Scripts Quick Ref

| Script | Phase | What it does |
|--------|-------|--------------|
| `convert_gguf_to_bump_v6_6.py` | 1 | GGUF → BUMPWGT5 + manifest |
| `convert_hf_to_bump_v6_6.py` | 1 | HF safetensors → BUMPWGT5 + manifest |
| `gen_kernel_registry_from_maps.py` | 2 | kernel_maps/*.json → KERNEL_REGISTRY.json |
| `build_ir_v6_6.py` | 3-7 | Main pipeline (graph → lower → fuse → parallel → layout) |
| `v6_6_ir_lowering.py` | 4 | Op → kernel mapping |
| `fusion_patterns.py` | 5 | Fusion pattern definitions |
| `parallel_planner.py` | 5 | M_parallel vs H_parallel assignment |
| `codegen_v6_6.py` | 8 | IR → C code |

---

## Training (Future - Not Integrated)

v6.6 has scaffolding but training is incomplete:

| Needed | Status |
|--------|--------|
| Gradient buffers | ❌ Not started |
| Backward pass kernels | ⚠️ Scaffolding exists |
| Optimizer state (Adam/SGD) | ❌ Not started |
| ck-kernel-backprop.c | ❌ Not generated |

See: `version/v6.6/src/v6.6_pipeline.md` for full training roadmap.

---

*Updated: 2026-01-21*
