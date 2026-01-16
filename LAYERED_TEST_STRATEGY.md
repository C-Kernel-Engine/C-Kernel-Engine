# CK-Engine Layered Test Strategy

## The Problem

When inference produces gibberish or fails, we need to pinpoint **exactly** where:
- Is it the GGUF parsing?
- Is it the bump conversion?
- Is it the IR generation?
- Is it the codegen?
- Is it the kernel implementation?
- Is it the tensor dimensions?

Currently, we jump to E2E and get "it works" or "gibberish" with no insight.

## The Solution: 6-Layer Test Pyramid

```
                    ┌─────────────────────────────┐
                    │      E2E INFERENCE          │ ← "Does it work?"
                    │   (prompt → response)       │
                    └─────────────────────────────┘
                              ▲
                    ┌─────────────────────────────┐
                    │   TENSOR FLOW VALIDATION    │ ← "Are shapes right?"
                    │ (input/output through IR)   │
                    └─────────────────────────────┘
                              ▲
                    ┌─────────────────────────────┐
                    │   GENERATED CODE TESTS      │ ← "Does code compile & run?"
                    │ (compile, link, basic exec) │
                    └─────────────────────────────┘
                              ▲
                    ┌─────────────────────────────┐
                    │   IR STRUCTURE VALIDATION   │ ← "Is the graph correct?"
                    │  (ops, dtypes, dimensions)  │
                    └─────────────────────────────┘
                              ▲
                    ┌─────────────────────────────┐
                    │   BUMP CONVERSION TESTS     │ ← "Are weights correct?"
                    │ (manifest, checksums, dtypes)│
                    └─────────────────────────────┘
                              ▲
                    ┌─────────────────────────────┐
                    │   KERNEL PARITY (llama.cpp) │ ← "Do kernels match?"
                    │   (Q4_K, Q6_K, RMSNorm...)  │
                    └─────────────────────────────┘
```

## Makefile Targets

```bash
# Layer 1: Kernel parity with llama.cpp
make test-kernels-parity          # Quick: 5 key kernels
make test-kernels-parity-full     # Full: all quant types + ops

# Layer 2: GGUF → Bump conversion
make test-bump-conversion         # Validate weights.bump + manifest

# Layer 3: IR generation
make test-ir-validation           # Validate lowered JSON structure

# Layer 4: Codegen
make test-codegen                 # Compile generated C code

# Layer 5: Tensor flow
make test-tensor-flow             # Validate dimensions through IR

# Layer 6: E2E
make e2e                          # Full inference test

# Run all layers in order (stops at first failure)
make test-full-pipeline
```

---

## Layer 1: Kernel Parity Tests

### Purpose
Verify CK kernels produce **identical output** to llama.cpp reference.

### What We Test
| Kernel | Input | Expected Output |
|--------|-------|-----------------|
| `dequant_q4_k` | Q4_K block (32 bytes) | 32 FP32 values |
| `dequant_q6_k` | Q6_K block (210 bytes) | 256 FP32 values |
| `dequant_q5_0` | Q5_0 block (22 bytes) | 32 FP32 values |
| `dequant_q8_0` | Q8_0 block (34 bytes) | 32 FP32 values |
| `gemv_q4_k` | Q4_K weights + FP32 vec | FP32 output |
| `gemm_q4_k` | Q4_K weights + FP32 mat | FP32 output |
| `rmsnorm` | FP32 input + weights | Normalized FP32 |
| `rope` | FP32 Q,K + positions | Rotated FP32 |
| `softmax` | FP32 scores | Probabilities |
| `swiglu` | FP32 gate + up | Activated FP32 |

### Tolerance
- Dequant: `max_diff < 1e-6` (exact)
- GEMV/GEMM: `max_diff < 1e-4` (FP32 accumulation)
- Softmax: `max_diff < 1e-5`

### Test Script: `scripts/test_kernel_parity_full.py`

```python
# Tests to implement:
1. test_dequant_all_types()    # Q4_K, Q6_K, Q5_0, Q8_0, Q4_0, Q4_1
2. test_gemv_all_types()       # Same quant types
3. test_gemm_all_types()       # Prefill path
4. test_rmsnorm_parity()       # With/without epsilon variants
5. test_rope_parity()          # Interleaved vs standard
6. test_softmax_parity()       # Full row softmax
7. test_swiglu_parity()        # gate * silu(up)
8. test_attention_parity()     # Full attention block
```

---

## Layer 2: Bump Conversion Tests

### Purpose
Verify GGUF → Bump conversion preserves **all tensor data correctly**.

### What We Test

1. **Tensor Count Match**
   ```
   GGUF tensors: 169
   Bump tensors: 169
   MATCH: ✓
   ```

2. **Per-Tensor Dtype Match**
   ```
   token_embd.weight: GGUF=Q8_0, Bump=Q8_0 ✓
   blk.0.attn_q.weight: GGUF=Q4_K, Bump=Q4_K ✓
   ...
   ```

3. **Per-Tensor Shape Match**
   ```
   token_embd.weight: GGUF=[151936,896], Bump=[151936,896] ✓
   ```

4. **Per-Tensor Checksum**
   ```
   token_embd.weight: SHA256=a1b2c3... MATCH ✓
   ```

5. **Manifest Consistency**
   ```
   weights_manifest.json entries: 169
   weights_manifest.map offsets: valid
   All checksums: verified
   ```

### Test Script: `scripts/test_bump_conversion.py`

```python
def test_bump_conversion(gguf_path, bump_dir):
    # 1. Parse GGUF
    gguf_tensors = parse_gguf(gguf_path)

    # 2. Parse bump manifest
    manifest = json.load(open(f"{bump_dir}/weights_manifest.json"))
    bump_map = parse_manifest_map(f"{bump_dir}/weights_manifest.map")

    # 3. Compare counts
    assert len(gguf_tensors) == len(manifest['tensors'])

    # 4. Per-tensor comparison
    for name, gguf_info in gguf_tensors.items():
        bump_info = manifest['tensors'][name]

        assert gguf_info['dtype'] == bump_info['dtype']
        assert gguf_info['shape'] == bump_info['shape']

        # 5. Data checksum
        gguf_data = extract_gguf_tensor(gguf_path, name)
        bump_data = extract_bump_tensor(bump_dir, name, bump_map)
        assert sha256(gguf_data) == sha256(bump_data)
```

---

## Layer 3: IR Validation Tests

### Purpose
Verify IR generation produces **correct computation graph**.

### What We Test

1. **IR Structure Completeness**
   ```
   ✓ version: 4
   ✓ kind: "lowered"
   ✓ mode: "decode" / "prefill"
   ✓ config: complete
   ✓ symbols: all defined
   ✓ sections: non-empty
   ```

2. **Operation Sequence**
   ```
   Layer 0:
     ✓ rmsnorm → expected input shape
     ✓ linear (qkv_proj) → weight dtype matches manifest
     ✓ rope → correct head_dim
     ✓ attention_decode → correct num_heads
     ✓ linear (o_proj) → output shape correct
     ✓ residual_add → shapes match
     ...
   ```

3. **Dimension Consistency**
   ```
   embed_dim (E) = 896, used consistently: ✓
   head_dim (D) = 64, E/num_heads = 64: ✓
   intermediate_dim (I) = 4864, correct: ✓
   ```

4. **Weight Dtype vs Manifest**
   ```
   IR says blk.0.attn_q uses Q4_K
   Manifest says blk.0.attn_q.weight is Q4_K
   MATCH: ✓
   ```

5. **Buffer Allocation**
   ```
   No overlapping buffers: ✓
   All activations have unique offsets: ✓
   KV cache properly sized: ✓
   ```

### Test Script: `scripts/test_ir_validation.py`

```python
def test_ir_validation(ir_path, manifest_path):
    ir = json.load(open(ir_path))
    manifest = json.load(open(manifest_path))

    # 1. Structure checks
    assert ir['version'] == 4
    assert ir['kind'] == 'lowered'
    assert ir['mode'] in ['decode', 'prefill']

    # 2. Symbol consistency
    symbols = ir['symbols']
    config = ir['config']
    assert symbols['E'] == config['embed_dim']
    assert symbols['H'] == config['num_heads']
    assert symbols['D'] == config['head_dim']

    # 3. Per-layer validation
    for section in ir['sections']:
        for layer in section['layers']:
            validate_layer_ops(layer, manifest, symbols)

    # 4. Buffer non-overlap
    validate_buffer_layout(ir)
```

---

## Layer 4: Codegen Tests

### Purpose
Verify generated C code is **correct and compilable**.

### What We Test

1. **Compilation Success**
   ```
   gcc -c ck-kernel-inference.c → SUCCESS ✓
   No errors, warnings acceptable
   ```

2. **Header Consistency**
   ```
   Header defines match source usage:
   - VOCAB_SIZE in header = VOCAB_SIZE in source ✓
   - MAX_SEQ_LEN in header = MAX_SEQ_LEN in source ✓
   ```

3. **Kernel Calls Match IR**
   ```
   IR: layer 0, qkv_proj, dtype=Q4_K
   Generated: gemm_nt_q4_k(L0_input, L0_WQ, ...)
   MATCH: ✓
   ```

4. **Buffer References Valid**
   ```
   All QWEN2_DECODE_PTR(&model, ...) references valid offsets ✓
   No undefined buffer references ✓
   ```

5. **Canary Verification Present**
   ```
   qwen2_decode_verify_canaries() function exists ✓
   Checks all layer canaries ✓
   ```

### Test Script: `scripts/test_codegen_validation.py`

```python
def test_codegen(c_file, h_file, ir_path):
    # 1. Parse generated code
    c_code = open(c_file).read()
    h_code = open(h_file).read()

    # 2. Compile test
    result = subprocess.run([
        'gcc', '-c', '-O2', '-fPIC', '-fopenmp',
        '-I', 'include', c_file, '-o', '/tmp/test.o'
    ])
    assert result.returncode == 0, "Compilation failed"

    # 3. Extract kernel calls from code
    kernel_calls = extract_kernel_calls(c_code)

    # 4. Compare to IR
    ir = json.load(open(ir_path))
    ir_ops = extract_ir_ops(ir)

    # 5. Verify 1:1 correspondence
    for layer_id, ir_op in ir_ops.items():
        code_call = kernel_calls[layer_id]
        assert ir_op['kernel'] == code_call['function']
        assert ir_op['dtype'] == code_call['inferred_dtype']
```

---

## Layer 5: Tensor Flow Validation

### Purpose
Verify **input/output tensor shapes** are correct throughout inference.

This is the **most critical** layer for catching subtle bugs.

### What We Test

1. **Embedding Output Shape**
   ```
   Input: tokens [S] (S=1 for decode, S=N for prefill)
   Output: hidden [S, E] where E=896
   ✓ Shape matches
   ```

2. **Per-Layer Shape Flow**
   ```
   Layer 0:
     RMSNorm input: [S, E]        → output: [S, E]       ✓
     QKV proj input: [S, E]       → output: [S, 3*H*D]   ✓
     Q reshape: [S, 3*H*D]        → [S, H, D]            ✓
     RoPE input: [S, H, D]        → output: [S, H, D]    ✓
     K cache write: [H, D]        → cache[H, T, D]       ✓
     Attention: Q[S,H,D], K[H,t,D], V[H,t,D] → [S,H,D]  ✓
     O proj: [S, H*D]             → [S, E]               ✓
     ...
   ```

3. **Prefill vs Decode Equivalence**
   ```
   Prefill token 0 output ≈ Decode token 0 output (same KV)
   Max diff: 1.2e-5 ✓
   ```

4. **KV Cache Shape**
   ```
   K cache: [num_kv_heads, max_seq_len, head_dim]
   V cache: [num_kv_heads, max_seq_len, head_dim]
   ✓ Shapes match config
   ```

5. **Final Logits Shape**
   ```
   Output: [S, vocab_size] where vocab_size=151936
   ✓ Shape matches
   ```

### Test Script: `scripts/test_tensor_flow.py`

```python
def test_tensor_flow(model_dir, mode='decode'):
    """
    Instrument the generated code to capture tensor shapes at each step.
    Compare against expected shapes from IR.
    """
    ir = load_ir(model_dir, mode)

    # 1. Load model
    lib = ctypes.CDLL(f"{model_dir}/libmodel.so")

    # 2. Instrument for shape capture
    shape_log = []

    # 3. Run single forward pass
    tokens = [1]  # BOS token
    lib.ck_model_embed_tokens(tokens, len(tokens))
    lib.ck_model_forward(None)

    # 4. Extract shapes from instrumented code
    actual_shapes = parse_shape_log(shape_log)

    # 5. Compare to IR expected shapes
    expected_shapes = extract_expected_shapes(ir)

    for op_name, expected in expected_shapes.items():
        actual = actual_shapes[op_name]
        assert actual == expected, f"{op_name}: expected {expected}, got {actual}"
```

### Alternative: Static Analysis

If instrumentation is complex, we can do **static analysis** of the generated code:

```python
def test_tensor_flow_static(c_file, ir_path):
    """
    Parse generated C code and verify buffer sizes match IR.
    """
    c_code = open(c_file).read()
    ir = json.load(open(ir_path))

    # Extract all buffer size definitions
    buffer_sizes = extract_buffer_sizes(c_code)

    # Compare to IR
    for buf in ir['buffers']:
        expected_size = compute_size(buf['shape'], buf['dtype'])
        actual_size = buffer_sizes[buf['name']]
        assert actual_size == expected_size
```

---

## Layer 6: E2E Inference (Already Implemented)

See `scripts/full_integration_testing.sh`.

---

## Combined Pipeline Test

```bash
#!/bin/bash
# scripts/test_full_pipeline.sh

set -e  # Stop on first failure

echo "=== CK-Engine Full Pipeline Validation ==="
echo ""

# Layer 1: Kernel parity
echo "[1/6] Testing kernel parity with llama.cpp..."
python3 scripts/test_kernel_parity_full.py || {
    echo "FAILED at Layer 1: Kernel parity"
    echo "  → Check kernel implementations in src/kernels/"
    exit 1
}

# Layer 2: Bump conversion
echo "[2/6] Testing GGUF → Bump conversion..."
python3 scripts/test_bump_conversion.py || {
    echo "FAILED at Layer 2: Bump conversion"
    echo "  → Check scripts/v6/convert_gguf_to_bump_v6.py"
    exit 2
}

# Layer 3: IR validation
echo "[3/6] Testing IR generation..."
python3 scripts/test_ir_validation.py || {
    echo "FAILED at Layer 3: IR generation"
    echo "  → Check scripts/v6/build_ir_v6.py"
    exit 3
}

# Layer 4: Codegen
echo "[4/6] Testing code generation..."
python3 scripts/test_codegen_validation.py || {
    echo "FAILED at Layer 4: Code generation"
    echo "  → Check scripts/v6/codegen_v6.py"
    exit 4
}

# Layer 5: Tensor flow
echo "[5/6] Testing tensor flow dimensions..."
python3 scripts/test_tensor_flow.py || {
    echo "FAILED at Layer 5: Tensor flow"
    echo "  → Check IR shapes vs generated code"
    exit 5
}

# Layer 6: E2E
echo "[6/6] Testing end-to-end inference..."
bash scripts/full_integration_testing.sh || {
    echo "FAILED at Layer 6: E2E inference"
    echo "  → All lower layers passed, check runtime"
    exit 6
}

echo ""
echo "=== ALL 6 LAYERS PASSED ==="
echo "Pipeline is healthy!"
```

---

## Diagnostic Output Example

When something fails, you get:

```
=== CK-Engine Full Pipeline Validation ===

[1/6] Testing kernel parity with llama.cpp...
  ✓ dequant_q4_k: max_diff=2.3e-7
  ✓ dequant_q6_k: max_diff=1.1e-7
  ✓ dequant_q5_0: max_diff=0.0
  ✓ gemv_q4_k: max_diff=3.2e-5
  ✓ rmsnorm: max_diff=1.8e-6
  ✓ rope: max_diff=2.1e-6
  ✓ softmax: max_diff=4.5e-7
  ✓ swiglu: max_diff=1.9e-6
  Layer 1 PASSED

[2/6] Testing GGUF → Bump conversion...
  ✓ Tensor count: 169/169
  ✓ Dtype match: 169/169
  ✓ Shape match: 169/169
  ✓ Checksum match: 169/169
  Layer 2 PASSED

[3/6] Testing IR generation...
  ✓ Structure valid
  ✓ Symbols consistent
  ✓ 24 layers validated
  ✓ Buffer layout non-overlapping
  Layer 3 PASSED

[4/6] Testing code generation...
  ✓ Compilation successful
  ✓ 24 layers × 12 ops = 288 kernel calls
  ✓ All kernel calls match IR
  Layer 4 PASSED

[5/6] Testing tensor flow dimensions...
  ✗ Layer 5, MLP down_proj:
    Expected output: [1, 896]
    Actual output: [1, 4864]

FAILED at Layer 5: Tensor flow
  → Check IR shapes vs generated code
  → Specifically: layer 5 MLP down projection
```

Now you know **exactly** where to look: Layer 5 MLP down projection has wrong output shape.

---

## Model-Agnostic Design

These tests work for **any model** by reading config from:
1. `config.json` - Model architecture
2. `weights_manifest.json` - Tensor dtypes/shapes
3. `lowered_*.json` - IR structure

No hardcoded Qwen assumptions. Works for:
- Qwen2-0.5B
- SmolLM-360M
- Llama-7B
- Any GGUF model

---

## Implementation Priority

1. **Layer 1** (Kernel Parity) - Already mostly exists, needs consolidation
2. **Layer 2** (Bump Conversion) - Quick to implement, high value
3. **Layer 3** (IR Validation) - Critical for catching graph bugs
4. **Layer 5** (Tensor Flow) - Most important for shape bugs
5. **Layer 4** (Codegen) - Less critical if 3 and 5 pass
6. **Layer 6** (E2E) - Already implemented

---

## Future: Per-Version Regression

```bash
# Run same tests on v6, v6.5, v6.6
for version in v6 v6.5 v6.6; do
    echo "Testing $version..."
    MODEL_VERSION=$version make test-full-pipeline
done
```

This catches when a v6.6 change breaks v6 compatibility.
