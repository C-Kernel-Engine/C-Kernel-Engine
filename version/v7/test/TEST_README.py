#!/usr/bin/env python3
"""
TEST_SUITE_README.md - C-Kernel-Engine v7 Test Suite Documentation

This document describes all test scripts in the version/v7/test/ directory,
organized by their purpose and usage scenario.

================================================================================
DIRECTORY STRUCTURE
================================================================================

version/v7/test/
├── TEST_README.py              # This documentation
├── ck_test_runner.py           # Convenience wrapper for common workflows
├── *.py                        # v7-specific tests (74 files)
├── integration/                # Integration tests (to be organized)
│   └── ...
├── debugging/                  # Debugging/tracing scripts (to be organized)
│   └── ...
└── unittest/                   # Generic unit tests (28 files)
    ├── test_attention.py
    ├── test_embedding.py
    ├── test_layernorm.py
    ├── test_gemm.py
    └── ...

unittest/ (at repo root)        # Empty - moved to version/v7/test/unittest/

================================================================================
TABLE OF CONTENTS
================================================================================
1. Quick Start Guide
2. Test Categories
3. Detailed Test Descriptions
4. Common Debugging Workflows
5. Exit Codes Reference
6. Environment Variables

================================================================================
1. QUICK START GUIDE
================================================================================

FIRST TIME SETUP:
    # Set model path
    export CK_MODEL_DIR=~/.cache/ck-engine-v7/models/Qwen--Qwen2-0.5B-Instruct-GGUF

RUN ALL TESTS:
    cd version/v7/test
    python run_all_v7_tests.py --model $CK_MODEL_DIR

QUICK PARITY CHECK:
    python ck_test_runner.py --quick

MEMORY LAYOUT VALIDATION:
    python test_memory_planner.py --layout=$CK_MODEL_DIR/layout_decode.json

WEIGHT OFFSET CHECK:
    python test_weight_offset_consistency.py --model-dir $CK_MODEL_DIR

================================================================================
2. TEST CATEGORIES
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ CATEGORY 1: NUMERICAL PARITY (9 tests)                                      │
│ Purpose: Compare v7 output against reference implementations              │
├─────────────────────────────────────────────────────────────────────────────┤
│ test_numerical_parity.py           | Layer-by-layer v7 internal comparison   │
│ test_layer_by_layer.py             | Full checkpoint-by-checkpoint diff    │
│ test_layer0_parity.py              | Focus on layer 0 validation           │
│ test_layer0_parity_prefill.py      | Prefill path layer 0                  │
│ test_layer0_parity_streamed.py     | Streaming mode layer 0                │
│ test_out_proj_parity.py            | Output projection specific            │
│ compare_attention_output.py        | Attention output comparison           │
│ compare_v7.py                 | v7 high-level compare       │
│ v7_comparison.py         | Detailed v7 analysis           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CATEGORY 2: MEMORY & LAYOUT VALIDATION (7 tests)                            │
│ Purpose: Verify memory layout, offsets, and buffer assignments              │
├─────────────────────────────────────────────────────────────────────────────┤
│ test_memory_planner.py            | Quick layout invariant checks          │
│ test_bump_layout_sync.py          | BUMP file vs layout sync check        │
│ test_scratch_buffer.py            | Scratch buffer validation             │
│ test_kv_cache.py                  | KV cache layout and access            │
│ test_weight_offset_consistency.py | Weight offset verification             │
│ advanced_memory_validator.py      | Deep memory validation                │
│ test_dtype_consistency.py         | Dtype consistency across pipeline     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CATEGORY 3: IR & CODE GENERATION (9 tests)                                  │
│ Purpose: Validate IR structure and generated code correctness               │
├─────────────────────────────────────────────────────────────────────────────┤
│ test_codegen_ir_builder.py        | IR builder correctness                 │
│ test_codegen_plumbing.py          | Code generation pipeline               │
│ test_codegen_respects_ir.py       | Generated code matches IR              │
│ test_ir_reverse_validator.py      | IR1 → IR3 reverse validation          │
│ test_ir_lower3.py                 | IR3 lowering validation               │
│ test_single_layer_completeness.py | Single layer IR completeness          │
│ test_pipeline_validation.py       | Full pipeline validation              │
│ test_pipeline_v2.py               | Pipeline v2 specific tests            │
│ test_op_naming_consistency.py     | Op naming consistency                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CATEGORY 4: KERNEL-SPECIFIC TESTS (6 tests)                                 │
│ Purpose: Validate individual kernel implementations                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ test_kernel_validation.py         | Kernel validation framework            │
│ test_kernel_direct.py             | Direct kernel testing                  │
│ test_gemv_q8_0_q8_0.py            | GEMV q8_0 kernel specific              │
│ test_q4k_packing.py               | Q4_K quantization packing              │
│ test_dimension_alignment.py       | Dimension alignment checks             │
│ test_io_chain.py                  | Input/output chain validation          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CATEGORY 5: TRACING & DEBUGGING (11 scripts)                                │
│ Purpose: Trace execution to find divergence points                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ trace_divergence.py               | Find v7 divergence point        │
│ v7_comprehensive_debug.py       | Step-by-step v7 debugging            │
│ trace_mlp_path.py                 | Trace MLP computation                  │
│ trace_nan_layer.py                | Find layer with NaN                    │
│ trace_nan_source.py               | Find source of NaN/Inf                 │
│ trace_nan_step.py                 | Find step with NaN                     │
│ trace_prefill_nan.py              | Prefill NaN tracing                    │
│ trace_residual.py                 | Residual flow tracing                  │
│ trace_detail.py                   | Detailed execution trace               │
│ trace_layers.py                   | Per-layer trace                        │
│ trace_narrow.py                   | Narrow down divergence region          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CATEGORY 6: UNITTEST/ (28 generic tests)                                    │
│ Purpose: Reusable tests for kernels and operations                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ test_attention.py                 | Attention kernel tests                 │
│ test_embedding.py                 | Embedding tests                        │
│ test_layernorm.py                 | Layer norm tests                       │
│ test_gemm.py                      | GEMM kernel tests                      │
│ test_softmax.py                   | Softmax tests                         │
│ test_swiglu.py                    | SwiGLU tests                          │
│ test_rope.py                      | RoPE tests                            │
│ test_quant_kernels.py             | Quantization kernel tests              │
│ ... and 20 more                                           │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
3. DETAILED TEST DESCRIPTIONS
================================================================================

TEST: test_layer_by_layer.py
┌──────────────────────────────────────────────────────────────────────────────┐
│ PURPOSE:                                                                      │
│   Layer-by-layer numerical validation comparing v7 against llama.cpp       │
│   or PyTorch reference. Finds the FIRST point of divergence.                 │
│                                                                              │
│ USAGE:                                                                        │
│   # Build reference (first time only)                                        │
│   python test_layer_by_layer.py --model $MODEL --build-reference            │
│                                                                              │
│   # Run comparison with PyTorch                                              │
│   python test_layer_by_layer.py --model $MODEL --token 25                   │
│                                                                              │
│   # Compare specific layer                                                   │
│   python test_layer_by_layer.py --model $MODEL --token 25 --layer 5         │
│                                                                              │
│   # Don't stop on first failure (show all)                                   │
│   python test_layer_by_layer.py --model $MODEL --token 25 --all-layers      │
│                                                                              │
│ CHECKPOINTS TESTED:                                                          │
│   - embedding → layer_{i}_ln1 → layer_{i}_q/k/v → layer_{i}_q_rope/k_rope   │
│   → layer_{i}_attn_out → layer_{i}_attn_residual → layer_{i}_ln2            │
│   → layer_{i}_mlp_out → layer_{i}_output → final_ln → logits                │
│                                                                              │
│ EXIT CODES:                                                                  │
│   0 = All checkpoints pass                                                   │
│   1 = First failure reported (check output for details)                      │
└──────────────────────────────────────────────────────────────────────────────┘

TEST: test_weight_offset_consistency.py
┌──────────────────────────────────────────────────────────────────────────────┐
│ PURPOSE:                                                                      │
│   Validates that BUMP file offsets are correct and data matches dtype.       │
│   Critical for catching converter bugs.                                      │
│                                                                              │
│ USAGE:                                                                        │
│   python test_weight_offset_consistency.py --model-dir $MODEL               │
│                                                                              │
│ WHAT IT CHECKS:                                                              │
│   1. Offsets follow expected layer stride pattern                            │
│   2. Actual BUMP data dtype matches manifest (e.g., not q5_0 vs q8_0)        │
│   3. Quantization block structure is valid                                    │
│   4. Offsets are monotonically increasing                                     │
│                                                                              │
│ EXIT CODES:                                                                  │
│   0 = All weights consistent                                                 │
│   1 = Inconsistencies found (likely offset or dtype bug)                     │
└──────────────────────────────────────────────────────────────────────────────┘

TEST: test_memory_planner.py
┌──────────────────────────────────────────────────────────────────────────────┐
│ PURPOSE:                                                                      │
│   Quick validation of memory layout without deep IR analysis.                │
│                                                                              │
│ USAGE:                                                                        │
│   python test_memory_planner.py --layout=$MODEL/layout_decode.json          │
│                                                                              │
│ QUICK TESTS:                                                                 │
│   1. No overlapping memory regions                                           │
│   2. Contiguity (report gaps between regions)                                │
│   3. Quantization size calculations                                          │
│   4. Layer stride consistency                                                │
│   5. All layers present                                                      │
│   6. Activation buffers exist                                                │
│   7. Canary markers (if present)                                             │
│   8. Data flow (no dangling tensors)                                        ────────────── │
└────────────────────────────────────────────────────────────────┘

TEST: trace_divergence.py
┌──────────────────────────────────────────────────────────────────────────────┐
│ PURPOSE:                                                                      │
│   Runs v7 probes across stop-op boundaries and compares activations by stage       │
│   to identify where divergence occurs.                                       │
│                                                                              │
│ USAGE:                                                                        │
│   python trace_divergence.py --token 25 --verbose                            │
│                                                                              │
│ OUTPUT:                                                                       │
│   Shows side-by-side comparison of activations from each layer, highlights   │
│   first layer where values differ significantly.                             │
│                                                                              │
│ REQUIRED:                                                                     │
│   A v7 model must be cached:                                  │
│   - ~/.cache/ck-engine-v7/models/Qwen--Qwen2-0.5B-Instruct-GGUF           │
│   - ~/.cache/ck-engine-v7/models/Qwen--Qwen2-0.5B-Instruct-GGUF           │
└──────────────────────────────────────────────────────────────────────────────┘

TEST: v7_comprehensive_debug.py
┌──────────────────────────────────────────────────────────────────────────────┐
│ PURPOSE:                                                                      │
│   Step-by-step debugging of v7. Can stop at specific operations and        │
│   inspect intermediate values.                                               │
│                                                                              │
│ USAGE:                                                                        │
│   # Test just embedding                                                      │
│   python v7_comprehensive_debug.py --stop-at 0 --verbose                   │
│                                                                              │
│   # Test through layer 0 attention                                           │
│   python v7_comprehensive_debug.py --stop-at 5 --verbose                   │
│                                                                              │
│   # Run normally (full decode)                                               │
│   python v7_comprehensive_debug.py                                         │
│                                                                              │
│ TIP: Use CK_STOP_OP environment variable to stop at specific op:             │
│   CK_STOP_OP=5 python v7_comprehensive_debug.py                            │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
4. COMMON DEBUGGING WORKFLOWS
================================================================================

WORKFLOW A: New Parity Issue Found
─────────────────────────────────
    1. Run quick layer-by-layer to find first failure:
       python test_layer_by_layer.py --model $MODEL --token 25

    2. Run comprehensive debug at that layer:
       python v7_comprehensive_debug.py --stop-at LAYER

    3. Trace v7 divergence:
       python trace_divergence.py --token 25

    4. Check memory layout:
       python test_memory_planner.py --layout=$MODEL/layout_decode.json

WORKFLOW B: NaN/Inf Detected
───────────────────────────
    1. Find which layer has NaN:
       python trace_nan_layer.py --token 25

    2. Find exact operation with NaN:
       python trace_nan_step.py --token 25

    3. Find source of NaN:
       python trace_nan_source.py --token 25 --layer X

    4. Validate weights haven't corrupted memory:
       python test_weight_offset_consistency.py --model-dir $MODEL

WORKFLOW C: Memory Corruption Suspected
───────────────────────────────────────
    1. Quick memory check:
       python test_memory_planner.py --layout=$MODEL/layout_decode.json

    2. Deep validation:
       python advanced_memory_validator.py --model $MODEL --deep

    3. Check BUMP file integrity:
       python test_weight_offset_consistency.py --model-dir $MODEL

    4. Verify KV cache layout:
       python test_kv.py --model $MODEL

WORKFLOW D: Converter Bug Found
───────────────────────────────
    1. Verify weight offsets:
       python test_weight_offset_consistency.py --model-dir $MODEL

    2. Check dtype consistency:
       python test_dtype_consistency.py --model $MODEL

    3. Validate memory layout sync:
       python test_bump_layout_sync.py --model $MODEL

    4. Run GEMV specific test:
       python test_gemv_q8_0_q8_0.py --model $MODEL

================================================================================
5. EXIT CODES REFERENCE
================================================================================

    ┌──────────┬─────────────────────────────────────────────────────────────┐
    │ CODE     │ MEANING                                                    │
    ├──────────┼─────────────────────────────────────────────────────────────┤
    │ 0        │ All tests passed                                           │
    │ 1        │ Test failed (check stdout for details)                     │
    │ 2        │ Missing dependencies (llama.cpp, PyTorch, etc.)            │
    │ 3        │ Model files not found                                      │
    │ 4        │ Invalid arguments                                          │
    │ 5        │ Internal error (assertion failure, unexpected state)       │
    └──────────┴─────────────────────────────────────────────────────────────┘

================================================================================
6. ENVIRONMENT VARIABLES
================================================================================

    CK_MODEL_DIR          │ Override default model cache path
    CK_STOP_OP            │ Stop execution at given operation index
    CK_DUMP_ACTIVATIONS   │ Dump all activations to files
    CK_LOG_LEVEL          │ 0=quiet, 1=info, 2=debug, 3=trace
    LD_LIBRARY_PATH       │ Add .so search paths

================================================================================
7. EXPECTED FILE LOCATIONS
================================================================================

    Model Cache: ~/.cache/ck-engine-v7/models/Qwen--Qwen2-0.5B-Instruct-GGUF/

    Expected files:
    ├── weights.bump              # Quantized weights
    ├── weights_manifest.json     # Weight metadata
    ├── layout_decode.json        # Memory layout (decode mode)
    ├── layout_prefill.json       # Memory layout (prefill mode)
    ├── lowered_decode.json       # Lowered IR (decode)
    ├── lowered_prefill.json      # Lowered IR (prefill)
    ├── libckernel_engine.so      # Kernel engine library
    ├── libmodel.so               # Generated model library
    └── ck-kernel-inference.so    # Alternative model library name

================================================================================
"""
if __name__ == "__main__":
    print(__doc__)
