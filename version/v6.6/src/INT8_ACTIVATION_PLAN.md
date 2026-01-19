# INT8 Activation Optimization Plan (Q5_0 x Q8_0)

## Executive Summary

Replace FP32 activations with Q8_0 quantized activations for GEMM/GEMV operations. This enables use of integer dot product kernels (`vec_dot_q5_0_q8_0`) which are significantly faster than FP32.

## Current Architecture

```
Input (FP32) → GEMM_Q5_0 → Output (FP32)
              ↑
         Weights (Q5_0)

Kernel: gemm_nt_q5_0(float *A, void *B_q5_0, float *C, M, N, K)
```

- All activations are FP32 (4 bytes per element)
- GEMM kernels dequantize weights to FP32, then do FP32 matmul
- Memory bandwidth limited by FP32 activation reads/writes

## Target Architecture

```
Input (FP32) → Quantize → GEMM_Q5_0_Q8_0 → Output (FP32)
                 ↓              ↑
            Input (Q8_0)   Weights (Q5_0)

Kernels:
  quantize_fp32_to_q8_0(float *in, void *out_q8_0, n)
  gemv_q5_0_q8_0(float *y, void *W_q5_0, void *x_q8_0, M, K)
```

- Quantize activations to Q8_0 (1 byte per element + scales)
- Use integer dot product in GEMM kernel
- 4x memory savings on activation storage
- Faster integer arithmetic

## Performance Analysis

| Metric | FP32 Activations | Q8_0 Activations | Speedup |
|--------|------------------|------------------|---------|
| Activation size | 4 bytes/elem | 1.0625 bytes/elem | 3.8x smaller |
| Memory bandwidth | High | Low | ~2x faster |
| Compute | FP32 FMA | INT8 + scale | ~2x faster |
| **Expected overall** | 1.0x | **2-3x** | |

## Implementation Plan

### Phase 1: Kernel Infrastructure

#### 1.1 Verify Existing Kernels

File: `src/kernels/gemm_kernels_q5_0.c`

```c
// Already exists - verify correctness
void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy);
void gemv_q5_0_q8_0(float *y, const void *W, const void *x, int M, int K);
```

**Tasks:**
- [ ] Add parity test: `test_vec_dot_q5_0_q8_0_vs_llama.py`
- [ ] Add parity test: `test_gemv_q5_0_q8_0_vs_llama.py`
- [ ] Verify AVX2/AVX512 paths work correctly

#### 1.2 Add Quantization Kernels

File: `src/kernels/quant_kernels.c` (NEW)

```c
/**
 * Quantize FP32 vector to Q8_0 format.
 *
 * Q8_0 block format (34 bytes for 32 elements):
 *   - float16 scale (2 bytes)
 *   - int8_t quants[32] (32 bytes)
 *
 * @param src     Input FP32 vector
 * @param dst     Output Q8_0 buffer (must be n * 34/32 bytes)
 * @param n       Number of elements (must be multiple of 32)
 */
void quantize_row_q8_0(const float *src, void *dst, int n);

/**
 * Dequantize Q8_0 to FP32.
 */
void dequant_row_q8_0(const void *src, float *dst, int n);
```

**Implementation:**
```c
void quantize_row_q8_0(const float *src, void *dst, int n) {
    block_q8_0 *blocks = (block_q8_0 *)dst;
    int nb = n / QK8_0;  // QK8_0 = 32

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            amax = fmaxf(amax, fabsf(src[i*QK8_0 + j]));
        }
        float d = amax / 127.0f;
        blocks[i].d = GGML_FP32_TO_FP16(d);
        float id = d ? 1.0f / d : 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            blocks[i].qs[j] = (int8_t)roundf(src[i*QK8_0 + j] * id);
        }
    }
}
```

### Phase 2: IR and Layout Changes

#### 2.1 Add Q8_0 Activation Buffer Type

File: `scripts/v6.6/build_ir_v6.py`

```python
class BufferDType(Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    Q8_0 = "q8_0"  # NEW: Quantized activations

def bytes_per_element(dtype: BufferDType) -> float:
    if dtype == BufferDType.Q8_0:
        return 34.0 / 32.0  # 1.0625 bytes per element
    elif dtype == BufferDType.FP32:
        return 4.0
    # ...
```

#### 2.2 Add Q8_0 Scratch Buffers to Layout

For each layer, add Q8_0 versions of activation buffers:

```python
# In layout generation
if use_int8_activations:
    layer_buffers.append({
        "name": f"layer.{i}.ln1_out_q8",
        "shape": [max_seq_len, embed_dim],
        "dtype": "q8_0",
        "type": "activation",
        "bytes": int(max_seq_len * embed_dim * 34 / 32)
    })
```

#### 2.3 Memory Layout Comparison

| Buffer | FP32 Size | Q8_0 Size | Savings |
|--------|-----------|-----------|---------|
| ln1_out (32K x 896) | 117 MB | 31 MB | 73% |
| q (14 x 32K x 64) | 114 MB | 30 MB | 74% |
| k (2 x 32K x 64) | 16 MB | 4 MB | 75% |
| **Total activations** | ~1.5 TB | ~400 GB | 73% |

### Phase 3: Codegen Changes

#### 3.1 Add Kernel Selection for Q5_0 x Q8_0

File: `scripts/v6.6/codegen_v6.py`

```python
def select_matmul_kernel(weight_dtype: str, use_int8_act: bool, M: int) -> str:
    """Select optimal kernel based on types and dimensions."""
    if weight_dtype == "q5_0":
        if use_int8_act:
            if M == 1:
                return "gemv_q5_0_q8_0"  # GEMV for decode
            else:
                return "gemm_q5_0_q8_0"  # GEMM for prefill
        else:
            return "gemm_nt_q5_0"  # FP32 fallback
    # ... similar for q8_0, q4_k, q6_k
```

#### 3.2 Generate Quantization/Dequantization Calls

```python
def emit_int8_projection(layer_id, proj_name, in_buf, weight, bias, out_buf):
    """Emit projection with INT8 activations."""
    add(f"    // Quantize input to Q8_0")
    add(f"    quantize_row_q8_0({in_buf}, {in_buf}_q8, embed_dim);")
    add(f"")
    add(f"    // GEMV with Q5_0 weights x Q8_0 activations")
    add(f"    gemv_q5_0_q8_0({out_buf}, {weight}, {in_buf}_q8, M, K);")
    add(f"")
    add(f"    // Add bias (output is already FP32)")
    add(f"    for (int i = 0; i < M; i++) {out_buf}[i] += {bias}[i];")
```

#### 3.3 Full Layer Example

```c
// Layer N with INT8 activations
void layer_N_forward(Model *model, int num_tokens) {
    // 1. RMSNorm (FP32 → FP32, small enough to keep FP32)
    rmsnorm_forward(residual, ln1_gamma, ln1_out, num_tokens, embed_dim);

    // 2. Quantize normalized output to Q8_0
    quantize_row_q8_0(ln1_out, ln1_out_q8, num_tokens * embed_dim);

    // 3. Q/K/V projections with Q8_0 input
    for (int h = 0; h < H; h++) {
        gemv_q5_0_q8_0(q_h, wq_h, ln1_out_q8, head_dim, embed_dim);
        // ... add bias
    }
    for (int h = 0; h < H_kv; h++) {
        gemv_q5_0_q8_0(k_h, wk_h, ln1_out_q8, head_dim, embed_dim);
        gemv_q5_0_q8_0(v_h, wv_h, ln1_out_q8, head_dim, embed_dim);
    }

    // 4. RoPE (needs FP32) - q/k already FP32 from gemv output
    rope_forward_qk(q, k, rope_cos, rope_sin, ...);

    // 5. Attention (FP32)
    attention_forward_gqa_flash(q, k, v, attn_out, ...);

    // 6. Output projection - quantize attn_out first
    // Reshape attn_out to [num_tokens, H * head_dim]
    quantize_row_q8_0(attn_out, attn_out_q8, num_tokens * H * head_dim);
    gemv_q5_0_q8_0(proj_out, wo, attn_out_q8, embed_dim, H * head_dim);

    // 7. Residual add (FP32)
    add_inplace(residual, proj_out, num_tokens * embed_dim);

    // 8. MLP with INT8 activations
    rmsnorm_forward(residual, ln2_gamma, ln2_out, num_tokens, embed_dim);
    quantize_row_q8_0(ln2_out, ln2_out_q8, num_tokens * embed_dim);

    // Gate + Up projection
    gemv_q5_0_q8_0(fc1_out, w1, ln2_out_q8, 2 * intermediate_dim, embed_dim);

    // SwiGLU (FP32)
    swiglu_forward(fc1_out, swiglu_out, num_tokens, intermediate_dim);

    // Down projection
    quantize_row_q8_0(swiglu_out, swiglu_out_q8, num_tokens * intermediate_dim);
    gemv_q5_0_q8_0(mlp_out, w2, swiglu_out_q8, embed_dim, intermediate_dim);

    // Final residual
    add_inplace(residual, mlp_out, num_tokens * embed_dim);
}
```

### Phase 4: Testing and Validation

#### 4.1 Unit Tests

```bash
# New test files
unittest/
├── test_quantize_q8_0.py      # FP32 → Q8_0 → FP32 roundtrip
├── test_gemv_q5_0_q8_0.py     # GEMV parity vs llama.cpp
└── test_layer_int8_parity.py  # Full layer comparison
```

#### 4.2 Acceptance Criteria

| Test | Threshold | Notes |
|------|-----------|-------|
| Q8_0 roundtrip error | < 0.5% | Quantization fidelity |
| GEMV parity vs llama.cpp | max_diff < 1e-3 | Kernel correctness |
| Full layer parity | max_diff < 1e-2 | Accumulated error OK |
| End-to-end generation | Same output tokens | Functional equivalence |

### Phase 5: Codegen Flag

#### 5.1 Add CLI Flag

```python
# In ck_run_v6.py
parser.add_argument('--int8-activations', action='store_true',
                    help='Use Q8_0 quantized activations (faster)')
```

#### 5.2 Conditional Codegen

```python
def generate_layer_code(layer_id, config):
    if config.int8_activations and weight_dtype in ['q5_0', 'q8_0']:
        return emit_int8_layer(layer_id, ...)
    else:
        return emit_fp32_layer(layer_id, ...)
```

## File Changes Summary

| File | Change |
|------|--------|
| `src/kernels/quant_kernels.c` | NEW: quantize/dequant functions |
| `src/kernels/quant_kernels.h` | NEW: header |
| `src/kernels/gemm_kernels_q5_0.c` | Add/verify `gemv_q5_0_q8_0` |
| `scripts/v6.6/build_ir_v6.py` | Add Q8_0 buffer dtype |
| `scripts/v6.6/codegen_v6.py` | INT8 codegen path |
| `scripts/v6.6/ck_run_v6.py` | `--int8-activations` flag |
| `include/ckernel_quant.h` | Add Q8_0 block struct |
| `Makefile` | Add quant_kernels.c |
| `unittest/test_gemv_q5_0_q8_0.py` | NEW: parity test |
| `unittest/test_quantize_q8_0.py` | NEW: quant test |

## Rollout Plan

1. **Week 1**: Implement and test quantization kernels
2. **Week 2**: Implement and test `gemv_q5_0_q8_0` kernel
3. **Week 3**: Update IR/layout for Q8_0 buffers
4. **Week 4**: Update codegen with INT8 path
5. **Week 5**: Integration testing and benchmarking
6. **Week 6**: Performance tuning and documentation

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Decode speed | 1 tok/s | 2-3 tok/s | 2-3x |
| Activation memory | 1.5 TB | 400 GB | 73% reduction |
| Memory bandwidth | High | Low | ~2x |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Quantization error | Quality degradation | Keep FP32 fallback |
| Kernel bugs | Incorrect output | Extensive parity testing |
| Memory alignment | Crashes | Ensure 64-byte alignment |
| AVX2 compatibility | Slower on old CPUs | Scalar fallback |

## References

- llama.cpp quantization: `ggml-quants.c`
- Q8_0 format spec: block_q8_0 = { fp16 d, int8_t qs[32] }
- vec_dot_q5_0_q8_0: llama.cpp integer dot product

