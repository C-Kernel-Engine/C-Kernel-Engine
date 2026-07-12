# v8 DSL Codegen Migration Inventory

## Why This Ledger Exists

The v8 compiler must not rediscover model mathematics from C function names.
Circuits own topology and required semantics. Weight manifests describe stored
tensors. Kernel maps own available implementations, numerical capabilities,
execution policies, and call ABIs. Resolution records one exact provider before
memory planning, and code generation emits that decision without guessing.

This ledger records the remaining specialized code-generation sites after the
model-family cleanup and the map-owned call-ABI migration. It is migration debt,
not an allowlist for adding more function-name dispatch.

## Ownership And Target Stage

| Current specialized logic | Sites | Authoritative source | Resolution stage | Target IR representation | Why it must move |
|---|---:|---|---|---|---|
| Q8_0 versus Q8_K activation quantization | 11 | Circuit requirement plus kernel-map storage capability | IR1 to Lower 1 | `input_storage`, `output_storage`, `block_format`, exact kernel ID | Quantization format changes buffer size, layout, and downstream ABI. Codegen cannot infer it from function names. |
| Q4_K versus Q6_K projection | 13 | Weight manifest plus circuit numerical requirement plus kernel map | Lower 1 resolver | `weight_format`, `activation_format`, `accumulator`, exact provider | The manifest identifies stored weights; the circuit declares acceptable mathematics; the map identifies the compatible GEMM or GEMV. |
| FP32 debug/reference fallback | 5 | Kernel-map reference implementation plus parity profile | Lower 1 or diagnostic IR transformation | `production_kernel`, `reference_kernel`, `diagnostic_mode` | Codegen must not construct a reference function name from the production function name. |
| BF16 QKV/out-projection export | 2 | Circuit semantic checkpoints plus parity profile | IR1 checkpoint propagation | `checkpoint_id`, logical layout, storage boundary | Export behavior belongs to a semantic graph edge, not a `gemv_bf16` name check. |
| Q4 gate-up row layout | 2 | Kernel-map weight-layout capability | Lower 1 resolver | `weight_row_bytes`, `block_size`, `row_layout` | Constants such as 144 and 210 bytes are storage-format properties. |
| Q4/Q6 GEMM prefill overrides | 5 | Kernel-map phase and shape capabilities | Lower 1 resolver | `phase=prefill`, exact GEMM provider, fallback provider | Prefill codegen must receive an already selected batched implementation. |
| SwiGLU plus Q8_K fusion recognition | 5 | Circuit fusion permission plus kernel-map `fuses` capability | Fusion pass before Lower 1 | One resolved fused operation with exact inputs and outputs | Fusion changes graph topology and must be decided before memory planning. |
| Exact attention implementation recognition | 1 | Circuit reduction contract plus attention kernel map | Lower 1 resolver | Reduction, storage, threading, and exact attention provider | Attention function names currently encode numerical semantics that must be explicit. |
| Mamba projection/selective-scan recognition | 2 | Mamba circuit operations plus kernel maps | IR1 to Lower 1 | Exact recurrent operation and provider | The circuit already identifies these operations; codegen must not rediscover them from C symbols. |
| Prefill M-RoPE call rewriting | 1 | Circuit position-transform contract plus kernel map | IR1 and Lower 1 | Explicit positions input and exact M-RoPE provider | Generated-C replacement is late and invisible to graph and memory validation. |
| Decode M-RoPE wrapper rewriting | 1 | Bridge circuit/runtime contract plus kernel map | IR1 and Lower 1 | Explicit position source: cache offset or bridge positions | Runtime position selection must be a declared operation or parameter, not patched C source. |

These categories overlap when one site performs more than one responsibility.
The authoritative remaining inventory is 41 specialized emission sites: 39
exact-function predicate groups and two generated M-RoPE rewrite groups.

## Why Q8 And Q4/Q6 Logic Was Repeated

The 11 Q8 and 13 Q4/Q6 occurrences are not 24 distinct mathematical
requirements. They are the same missing resolved metadata rediscovered at many
emission boundaries:

- decode and prefill were implemented by separate emitters;
- Q, K, V, output projection, gate/up, down projection, and logits each added a
  local optimized path;
- debug parity added FP32 bypasses beside production quantized paths;
- multimodal embedded prefill added another emission loop;
- fusion detection inspected adjacent generated operations;
- row-byte calculations were reconstructed from Q4/Q6 function identities.

At the time those paths were added, call IR carried a C function and ordered
arguments but did not consistently carry storage format, reference provider,
row layout, fusion identity, or complete execution capabilities. Each emitter
therefore decoded the function name again. This worked locally but duplicated
policy and made a newly added kernel easy to miss in one path.

The target architecture resolves each fact once:

```text
weights + circuit requirements + kernel-map capabilities
    -> IR1 requirements
    -> fusion and Lower 1 exact provider resolution
    -> IR2 memory planning only
    -> Lower 3 map-owned ordered call ABI
    -> codegen prints the resolved call
```

IR2 must not select kernels. Codegen must not select quantization, reduction,
fusion, layout, or fallback behavior.

## Exact 41-Site Inventory

### Decode Codegen: 21

File: `version/v8/scripts/codegen_core_v8.py`

1. Line 748: Q8_K attention-input quantization emission.
2. Line 775: recurrent gate/alpha/beta quantized projection override.
3. Line 789: recurrent Q4 versus Q8 FP32 fallback selection.
4. Line 845: Q/K/V Q4_K versus Q6_K projection override.
5. Line 858: Q/K/V FP32 fallback-function selection.
6. Line 910: BF16 Q/K/V projection export.
7. Line 945: Q8_K output-projection input override.
8. Line 989: Q8_K final-output/LM-head quantization.
9. Line 1025: Q4/Q6 quantized logits override.
10. Line 1038: Q4/Q6 logits FP32 fallback selection.
11. Line 1076: BF16 output-projection export.
12. Line 1113: Q4/Q6 output-projection override.
13. Line 1126: output-projection FP32 fallback selection.
14. Line 1179: MLP-down input quantization selection.
15. Line 1212: Q4/Q6 MLP-down override.
16. Line 1225: MLP-down FP32 fallback selection.
17. Line 1279: Q4/Q6 MLP gate-up specialized emission.
18. Line 1294: Q4 gate-up row-layout selection.
19. Line 1296: Q6 gate-up row-layout selection.
20. Line 1398: Q8_0 versus Q8_K generic quantized-row emission.
21. Line 1444: Mamba2 selective-scan specialized emission.

### Prefill Codegen: 18

File: `version/v8/scripts/codegen_prefill_v8.py`

22. Line 498: Q8_0/Q8_K batch quantization path.
23. Line 531: Q4/Q6 prefill output-projection override.
24. Line 580: Q4/Q6 prefill gate-up override.
25. Line 664: Q5_0 Mamba input-projection path.
26. Line 769: exact attention-function emission.
27. Line 1152: fused SwiGLU-to-Q8_K quantization guard.
28. Line 1169: adjacent SwiGLU plus Q8_K fusion detection.
29. Line 1423: quantized-row helper eligibility.
30. Line 1457: quantization debug-override eligibility.
31. Line 1494: Q4/Q6 GEMM FP32-override eligibility.
32. Line 1815: embedded-prefill first-input quantization.
33. Line 1821: embedded-prefill secondary-input quantization.
34. Line 1830: embedded-prefill Q4/Q6 gate-up path.
35. Line 1833: Q4 gate-up/SwiGLU x16 fusion selection.
36. Line 1857: embedded-prefill MLP-down quantization.
37. Line 1858: SwiGLU/Q8_K fused execution selection.
38. Line 1885: multimodal Q4/Q6 MLP-down override.
39. Line 1905: adjacent SwiGLU/Q8_K fusion setup.

### Generated-C M-RoPE Rewrites: 2

40. `version/v8/scripts/codegen_prefill_v8.py:1897`: replace the resolved
    `mrope_qk_text` call with a multimodal prefill wrapper.
41. `version/v8/scripts/codegen_v8.py:461-480`: inject and rewrite the decode
    runtime M-RoPE wrapper.

Line numbers identify the baseline at commit `eea2d795`. Future migrations must
remove ledger entries and update tests rather than preserve line-number matches.

## Acceptance Rule For Each Migration

For every removed site:

1. Add or validate the circuit requirement.
2. Add the exact kernel-map capability and call ABI.
3. Resolve one provider or hard-fail zero/multiple matches.
4. Persist the requirement and provider through IR1, LoweredIR, and call IR.
5. Make codegen emit the resolved operation without function-name inspection.
6. Add negative resolver tests and generated-IR tests.
7. Run kernel parity, stitched parity, family regression, and applicable
   llama.cpp or PyTorch reference gates.
