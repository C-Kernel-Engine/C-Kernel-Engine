# CK Model Compatibility Atlas

This is the reference-first workflow for adding new model families to C-Kernel-Engine.
GGUF and safetensors are source formats; CK runtime should converge on BUMP plus
explicit sidecar contracts.  New model work should start by proving the graph and
kernel contract before optimizing speed.

## Bring-up order

1. Inspect `config.json`.
2. Classify the layer pattern and required kernels.
3. Add safetensors-to-BUMP mapping with full source tensor coverage audit.
4. Run single-token and multi-token hidden/logit parity against PyTorch or a known
   reference.
5. Add GGUF conversion only after the BUMP graph contract is understood.
6. Optimize shared hot kernels after correctness is stable.

Use:

```bash
.venv/bin/python version/v8/scripts/inspect_model_contract_v8.py /path/to/config.json
```

The script is intentionally conservative: unsupported architectures should fail
closed with a list of missing kernels/templates.

## Current Families

| Family | Status | First target | Notes |
| --- | --- | --- | --- |
| Qwen2/Qwen3/Gemma3/Llama-style dense decoders | Supported at contract level | parity and perf | Uses standard attention + MLP path. |
| Qwen3.5 text | Bring-up active, usable | safetensors + GGUF parity | Hybrid DeltaNet/full-attention decoder. Safetensors require Qwen3.5 norm `+1` transform except `linear_attn.norm.weight`. |
| Gemma4 text | Bring-up active, usable | GGUF/safetensors parity | Hybrid full/sliding attention with per-layer embedding and per-layer RoPE control. |
| Whisper Tiny/Base/Small | Supported for FP32 transcription | safetensors + PyTorch parity | Generated audio frontend, encoder, and decoder. JFK tokens match exactly; Base has a 33-second long-form fixture. Longer PCM16 WAV recordings use sequential source windows. |
| Nematron-H | Supported (GGUF runtime lane) | GGUF Q4_K_M smoke | Hybrid Mamba2/attention decoder. Coherent Q4_K_M text E2E on the v8 GGUF lane; safetensors/PyTorch parity lane covers Mamba2 stitching and state-shape guardrails. |
| Cohere Command-style models | Supported as Cohere2 via GGUF (PR #401) | Q4_K_M bring-up | 8-token llama.cpp replay agreement on Command R7B; long-trajectory parity pending. |
| Qwen3.8 27B | Supported (qwen38 circuit) | Q4_K_M GGUF parity | 4,096-token trajectory bit-exact to llama.cpp; 262K context memory planner certified. |
| Instella-MoE 16B-A3B | Supported (instella_moe, BF16 safetensors) | PyTorch parity | BF16 safetensors certified; full-logit cosine 0.99998 at the 32-token checkpoint. Quantized GGUF and long trajectories pending. |
| Kimi-VL A3B text decoder | Supported (kimi_vl) | BF16 text certification | Text decoder certified post-#403 (repeatable on AVX2/AVX-512); MoonViT vision bridge pending. |
| Laguna-XS 2.1 | Supported (laguna, Q4_K_M text runtime) | GGUF Q4_K_M bring-up | Embedding and layer-0 RMSNorm bit-exact; long-trajectory parity not claimed (near-tied MoE route flips). |

## Nematron-H Gap

Observed public config properties:

- `model_type: nemotron_h`
- `architectures: ["NemotronHForCausalLM"]`
- `hybrid_override_pattern` as one character per layer. Current Nano uses `M` = Mamba2, `*` = attention, `E` = MoE, and `-` = dense MLP.
- Mamba parameters: `mamba_num_heads`, `mamba_head_dim`, `ssm_state_size`, `time_step_rank`, `conv_kernel`
- MLP activation: `relu2`
- MoE parameters: routed experts, shared expert, top-k routing, group-limited expert selection
- Attention parameters: normal GQA-style attention heads/KV heads

Required new CK contracts before full Nematron-H inference:

- `mamba_in_proj_split`
- `mamba_conv1d_state_update`
- `mamba_dt_softplus`
- `mamba_selective_scan`
- `mamba_rmsnorm_gate`
- `mamba_out_proj`
- `relu2_mlp` forward and backward
- group-limited top-k router and routed ReLU2 expert dispatch/combine are covered by scalar reference kernels; shared expert MLP wiring is still missing
- Nematron-H safetensors-to-BUMP mapping with strict all-weight coverage
- Nematron-H template/layer policy for `mamba` vs attention layers

The first implementation should be scalar FP32/BF16 parity-first, then optimized
with AVX/AVX-512/AMX only after hidden-stream parity is stable.

## Cohere Gap (closed by PR #401)

Cohere Command repositories were gated during the original survey, including
`config.json`, so tensor names could not be guessed safely. That gap is now
closed: declarative GGUF model-map support (#401) routes Command R artifacts to
the `cohere2` circuit automatically, with the shared pre-block LayerNorm,
parallel attention+MLP residual, sliding-only RoPE, and the model-declared
`logit_scale` footer expressed as contract data rather than a family branch.

Current evidence: Q4_K_M bring-up on Command R7B with tokenizer-free replay
matching llama.cpp for the first 8 greedy positions. Remaining work is
long-trajectory parity and performance sweeps.

## Why This Matters

CK should be able to say: given a kernel composition, embedding size, layer
pattern, and training contract, the DSL compiler can build a model independent
of PyTorch, llama.cpp, or Unsloth.  Compatibility with diverse model families is
evidence that the architecture is kernel/template-driven rather than hardcoded
to one lineage.
