# Mega-Fused Prefill Plan (v6.6)

Scope:
- Target v6.6 codegen and fused kernels under `src/kernels/fused/`.
- Leave legacy kernels under `src/kernels/` for older versions.
- Rule: kernels do not malloc/free and do not use large stack buffers. Scratch comes from bump allocator.

Cache summary (from lscpu):
- L1d: 128 KiB (4 instances) => 32 KiB per core => 512 lines @ 64 B.
- L2: 1 MiB (4 instances) => 256 KiB per core => 4096 lines.
- L3: 6 MiB shared => 98,304 lines.

Tile sizes chosen for this CPU:
- QKV projection token tile (Tproj): 32
- Attention query tile (Tq): 16
- Attention KV tile (Tk): 32

Prefill mega-fused attention sequence (AVX, quantized weights first):
1) RMSNorm per token row into Q8_0 scratch.
2) QKV projection (quantized weights):
   - Q stays in scratch.
   - K/V written directly to KV cache with stride = aligned_context_window.
3) RoPE on Q scratch and the new K row (only for current token positions).
4) Flash attention (online softmax):
   - Read K/V cache tiles, never materialize scores.
   - Attn output stays in scratch.
5) Output projection (quantized weights), residual add:
   - Write final output row (token-major) to DRAM.

DRAM vs cache:
- DRAM writes: K/V cache and final output (required).
- No DRAM writes for Q, scores, or attn_out.
- K/V tiles are streamed from DRAM/L3 into L2; Q stays hot in L1/L2 for each head.

Loop order (weight reuse + cache locality):
- Outer: token tiles (Tproj) so weights can be reused across tokens.
- Inner: head loops, then KV tiles (Tk) for flash attention.
- Process all query heads that share a kv-head before moving to next kv-head.

Variants to implement (one-by-one, with unit tests between each step):
1) Prefill fused RMSNorm+QKV (quantized).
2) Prefill fused MLP (quantized).
3) Prefill mega-fused attention (RMSNorm->QKV->RoPE->FlashAttn->OutProj+Residual).
4) Full prefill fusion wired into codegen.

Quantized MLP kernel (W1=gate+up, W2=down):
- W1 uses Q8_0 activations with Q5_0/Q8_0 weights (gemm_nt_q5_0_q8_0 or gemm_nt_q8_0_q8_0).
- W2 uses Q8_K activations with Q4_K/Q6_K weights (gemm_nt_q4_k_q8_k or gemm_nt_q6_k_q8_k).
- Cannot K-tile W2 because Q4_K/Q6_K GEMMs expect full-K row layout; instead tile tokens.
- Scratch layout per token tile: q8_0 input rows, gate tile, up tile, q8_k hidden rows.

Mega-fused attention kernel (prefill):
- No malloc/free; caller provides scratch.
- Scratch layout: Q buffer, attn_out buffer, token-major proj buffer, QKV scratch.
- QKV computed via fused_rmsnorm_qkv_prefill_head_major_quant (Q5_0/Q8_0 weights).
- Out-proj uses ck_gemm_nt_quant on flattened head-major attn_out (WO is Q5_0 for Qwen2).

Testing:
- Add unit tests per variant under `unittest/`.
- Parity vs PyTorch and llama.cpp for correctness.
- Microbench to compare speed across variants.
