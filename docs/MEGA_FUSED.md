That's the holy grail fusion - eliminating **all** intermediate DRAM traffic for the entire attention block. Let me create a detailed SVG showing exactly how this mega-fusion works through the cache hierarchy.Yes - this is exactly the Flash Attention insight extended to the full attention block. The numbers are staggering:

**Decode mode (single token, H=2048):**

| Metric | Unfused (10 kernels) | Mega-Fused | Reduction |
|--------|---------------------|------------|-----------|
| DRAM reads | ~400KB | 4KB | **100×** |
| DRAM writes | ~400KB | 4KB | **100×** |
| Arithmetic Intensity | ~0.5 FLOPs/byte | ~4000 FLOPs/byte | **8000×** |

**The key enablers for your C-Kernel-Engine:**

1. **RMSNorm → QKV**: Compute norm in registers, immediately stream normalized values into the GEMM. No intermediate buffer.

2. **QKV → RoPE**: Q and K are already in L2 from the GEMM. Apply rotation in-place using precomputed cos/sin (just 128 bytes for head_dim=64). Still in L2.

3. **RoPE → Flash Attention**: This is where it gets beautiful. The online softmax trick:
   - Keep running `m_i` (row max) and `l_i` (row sum) in **2 scalar registers per head**
   - Keep output accumulator `O_i` in **ZMM registers**
   - Stream K,V tiles from KV cache, update statistics, accumulate output
   - **Never materialize the S×S attention matrix**

4. **Flash → Out Proj**: O is already in registers. Feed directly into output projection GEMM. Add residual during the final store.

**Your working set per tile:**
```
Q_tile:  64 × 64 × 2B =   8KB  (in L1)
K_tile:  64 × 64 × 2B =   8KB  (stream from L2)
V_tile:  64 × 64 × 2B =   8KB  (stream from L2)
O_accum: 64 × 64 × 4B =  16KB  (in registers + L1)
Stats:   m, l per row  =  512B (in registers)
─────────────────────────────
Total:                   ~40KB  ✓ fits in L1+registers
```

Since you already have `attention_flash_true.c`, `rmsnorm_kernels.c`, `rope_kernels.c`, and `gemm_kernels.c` with PyTorch parity, the fusion is purely a matter of stitching them together while keeping intermediates in cache. The math doesn't change - only the memory traffic disappears.
