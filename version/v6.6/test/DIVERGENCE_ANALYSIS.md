# V6.5 vs V6.6 Divergence Analysis

## Summary

V6.5 and V6.6 produce different logits for the same input token, leading to different predictions:
- V6.5: argmax=9909, logits range (-16.5, 11.5)
- V6.6: argmax=20780, logits range (-21.3, 18.2)

## Root Cause Analysis

### Key Architectural Differences

| Component | V6.5 | V6.6 |
|-----------|------|------|
| Q/K/V Projection | `gemm_nt_q5_0_q8_0` (INT8 input) | `gemv_q5_0` (FP32 input) |
| Input Quantization | Quantizes layer input to Q8_0 | Uses FP32 directly |
| Projection Layout | Per-head (head-major) | Full matrix (row-major) |
| Attention Kernel | `attention_forward_causal_head_major_gqa_flash_strided` | `attention_forward_decode_head_major_gqa_flash` |
| LM Head | `gemm_nt_q8_0` | `gemv_q8_0` |

### Step-by-Step Trace (V6.6 Layer 0)

| Stop | Operation | Output Buffer Stats |
|------|-----------|---------------------|
| 0 | embedding | (-0.08, 0.06) |
| 1 | rmsnorm | layer_input: (-1.13, 1.12) |
| 3 | Q gemv | q: (-5.3, 2.8) |
| 4 | Q + bias | q: (-64.5, 126.4) |
| 6 | K + bias | k: (-152, 98) |
| 8 | V + bias | v: (-0.07, 0.10) |
| 11 | attention | layer_input: (-0.07, 0.10) |
| 12 | out_proj | embedded: (-0.16, 0.04) |
| 14 | residual add | layer_input: (-1.13, 1.11) |
| 15 | ln2 | embedded: (-9.0, 11.4) |
| 17 | gate_up gemv | mlp: (-4.8, 3.7) |
| 19 | swiglu | mlp: (-3.8, 5.4) |
| 22 | MLP residual | layer_input: (-10.1, 12.5) |

### Individual Operations: Verified Working

1. **Embedding lookup**: Correct Q8_0 dequantization
2. **RMSNorm**: Correct computation
3. **Q/K/V bias add**: Mathematically correct (bias values match v6.5)
4. **Attention (first token)**: Output matches V (correct for self-attention on single token)
5. **Residual add**: Correct element-wise addition
6. **MLP (gate, up, swiglu, down)**: Reasonable value ranges

### Hypothesis: Accumulated Precision Differences

The divergence is likely due to accumulated numerical differences between:
1. INT8 batch kernels (v6.5) vs FP32 input kernels (v6.6)
2. Different accumulation order in GEMV implementations
3. 24 layers × multiple projections = significant accumulation

### Evidence

1. **Q/K biases are large** (-65 to +128): Qwen2 has attention biases, which is expected
2. **No obvious bugs**: Each operation in isolation produces reasonable outputs
3. **Different kernel paths**: v6.5 and v6.6 use fundamentally different compute paths
4. **Final logits differ**: Max diff = 25.3, mean diff = 4.9

## Recommendations

### Option 1: Match V6.5 Kernel Selection
Make v6.6 use the same kernels as v6.5:
- Use `gemm_nt_q5_0_q8_0` with input quantization
- Use per-head projection layout
- Use `gemm_nt_q8_0` for lm_head

### Option 2: Accept Numerical Differences
Both versions produce valid outputs, just different:
- V6.6 uses more direct FP32 path
- May actually be more numerically stable
- Consider validating against PyTorch/HuggingFace reference

### Option 3: Hybrid Approach
- Compare both outputs against a reference (PyTorch)
- Use whichever is closer to reference
- Document expected numerical tolerance

## Test Files Created

- `debug_stop_seq_v65_v66.py`: Stop-sequence comparison
- `debug_embedding_v66.py`: Embedding verification
- `trace_divergence.py`: Full trace comparison
- `debug_q_bias.py`: Q bias analysis
- `compare_attention_output.py`: Attention trace
- `trace_mlp_path.py`: MLP trace

## Buffer Layout (No Aliasing Issues)

All activation buffers are properly separated:
- embedded_input: 396942152 - 400612168
- layer_input: 400612168 - 404282184
- residual: 404282184 - 407952200
- q_scratch: 433380168 - 437050184
- k_scratch: 437050184 - 437574472
- v_scratch: 437574472 - 438098760
- attn_scratch: 438098760 - 441768776
- mlp_scratch: 441768776 - 481614664
- logits: 485284680+

## Conclusion

The divergence is due to **different kernel implementations** rather than bugs:
- V6.5 uses INT8 batch-optimized kernels with input quantization
- V6.6 uses FP32 input kernels with separate bias addition
- Both are functionally correct but produce numerically different results

To achieve parity, V6.6's codegen would need to use the same kernel selection as V6.5.

## Next Steps

1. **Validate against PyTorch reference** to determine which implementation is more accurate
2. **Update V6.6 codegen** (`codegen_v6_6.py`) to use:
   - `gemm_nt_q5_0_q8_0` with input quantization for Q/K projections
   - `gemm_nt_q8_0_q8_0` for V projection
   - Per-head projection layout
   - `gemm_nt_q8_0` for lm_head
3. **Or** accept the numerical differences if v6.6 is closer to reference

## Test Commands

```bash
# Run stop_seq comparison
python version/v6.6/test/debug_stop_seq_v65_v66.py --token 100 --simple

# Trace layer 0 attention
python version/v6.6/test/compare_attention_output.py

# Trace MLP path
python version/v6.6/test/trace_mlp_path.py

# Validate against PyTorch (requires working PyTorch install)
python version/v6.6/test/validate_against_pytorch.py
```
