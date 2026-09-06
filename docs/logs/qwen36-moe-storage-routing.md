# Qwen3.6 MoE Storage-Aware Provider Selection

## Failure

The exact `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` artifact converted, compiled and
loaded, but produced all-zero logits. Layer-boundary X-Ray showed finite values
through the final attention layer's post-attention normalization, followed by
NaNs in the routed MoE output.

The artifact is a mixed-format recipe. Most expert down-projections use Q5_K,
while three layers use Q6_K. The Qwen3.5 circuit declared a Q4_K/Q4_K/Q5_K
provider as its default, and lowering accepted that default before considering
the actual three-weight storage tuple. The Q5_K provider therefore interpreted
Q6_K rows with the wrong stride.

## Fix

`moe_swiglu_expert_mlp` now resolves its provider from all three manifest
storage roles before accepting the circuit default:

- `moe_expert_gate`
- `moe_expert_up`
- `moe_expert_down`

This precedence is intentionally limited to the routed SwiGLU operation. Other
composite operations retain their existing circuit-default precedence and
resolution paths. No quantized arithmetic, reduction order, RoPE, attention or
Gemma provider changed.

The recurrent text X-Ray also recognizes MoE models that do not declare hyper
connections, so it captures router and expert boundaries instead of dense MLP
boundaries.

## Evidence

- Focused Qwen/Gemma provider suite: 126 tests passed.
- Qwen3.6, dense Qwen3.8, Flash Next and Cohere/Laguna contract targets passed.
- Architecture contracts passed with existing WARN-only implicit-edge debt.
- Fast real-model regression passed Gemma3, Qwen2, Qwen3, Qwen3.5 and Nanbeige.
- Mixed Q4_K/Q6_K expert kernel: four serial/threadpool bit-exact tests passed.
- Exact 35B artifact on Ryzen generated coherent text after a fresh conversion,
  IR build and compile.
- Generated C contains 74 Q4_K/Q4_K/Q5_K and six Q4_K/Q4_K/Q6_K routed-MoE
  calls across the prefill and decode functions, matching the manifest's mixed
  layer inventory.

The first full-vocabulary row after the fix has matching top-1 token against
llama.cpp, cosine similarity 0.999339 and 18/20 top-token overlap. It is not
bit-exact, so long-trajectory and long-context numerical certification remain
open. This change fixes catastrophic storage misrouting; it does not claim
full Qwen3.6 parity.
