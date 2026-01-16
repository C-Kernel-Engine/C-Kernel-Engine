/**
 * @file mega_fused_test_plan.md
 * @brief Comprehensive Test Plan for Mega-Fused Attention Kernels
 *
 * Tests required:
 * 1. Unit tests - Numerical correctness
 * 2. Parity tests - llama.cpp comparison
 * 3. Stability tests - PyTorch comparison
 * 4. Performance tests - DRAM pressure measurement
 *
 * Key metric: DRAM traffic reduction (the whole point of fusion!)
 */

# Mega-Fused Attention: Complete Test Plan

## 1. Unit Tests: Numerical Correctness

### 1.1 Fused RMSNorm + QKV Test
```c
// Test: fused_rmsnorm_qkv matches separate rmsnorm + gemm
void test_fused_rmsnorm_qkv(void) {
    float *input = random_tensor([1, H]);
    float *gamma = random_tensor([H]);
    float *W_qkv = random_tensor([3*H, H]);
    float *b_qkv = random_tensor([3*H]);

    // Reference: separate operations
    float ln1_ref[H];
    rmsnorm_forward(input, gamma, ln1_ref, ...);
    float q_ref[H], k_ref[H], v_ref[H];
    gemm_qkv_reference(ln1_ref, W_qkv, b_qkv, q_ref, k_ref, v_ref, ...);

    // Fused: single operation
    float q_fused[H], k_fused[H], v_fused[H];
    fused_rmsnorm_qkv(input, gamma, W_qkv, b_qkv, q_fused, k_fused, v_fused, ...);

    // Verify max diff < 1e-5
    float max_diff_q = max_diff(q_ref, q_fused, H);
    float max_diff_k = max_diff(k_ref, k_fused, H);
    float max_diff_v = max_diff(v_ref, v_fused, H);
    assert(max_diff_q < 1e-5f);
    assert(max_diff_k < 1e-5f);
    assert(max_diff_v < 1e-5f);
}
```

### 1.2 Fused Flash Attention Test
```c
// Test: fused_flash_attention_head matches naive attention
void test_fused_flash_attention(void) {
    float *Q = random_tensor([H, d]);
    float *K = random_tensor([S, d]);
    float *V = random_tensor([S, d]);

    // Reference: naive attention (materialize S = Q @ K.T)
    float O_ref[H, d];
    attention_naive_reference(Q, K, V, O_ref, ...);

    // Fused: flash attention
    float O_fused[H, d];
    fused_flash_attention_head(O_fused, Q, K, V, ...);

    // Verify max diff < 1e-4 (softmax can accumulate error)
    float max_diff = max_diff_2d(O_ref, O_fused, H, d);
    assert(max_diff < 1e-4f);
}
```

### 1.3 Complete Mega-Fused Attention Test
```c
// Test: mega_fused_attention matches full attention block
void test_mega_fused_complete(void) {
    // Setup: create mock layer with all weights
    CKAttentionBlock *layer = create_mock_attention_layer();

    // Reference: current attention_decode_fused.c
    float *input = random_tensor([1, H]);
    float *output_ref = zeros_tensor([1, H]);
    attention_decode_fused_reference(layer, input, output_ref, ...);

    // Fused: mega_fused_attention
    float *output_fused = zeros_tensor([1, H]);
    mega_fused_attention(output_fused, input, ..., layer, ...);

    // Verify
    float max_diff = max_diff(output_ref, output_fused, H);
    assert(max_diff < 1e-4f);
}
```

## 2. Parity Tests: llama.cpp Comparison

### 2.1 llama.cpp Reference Setup
```python
# Generate test vectors from llama.cpp
python3 scripts/test/generate_llama_reference.py \
    --model Qwen2-0.5B-Instruct-GGUF \
    --output tests/reference/llama_outputs.bump
```

### 2.2 Parity Test Script
```python
#!/usr/bin/env python3
"""
Compare C-Kernel-Engine mega-fused attention with llama.cpp reference.
"""
import numpy as np
import json

def test_parity():
    # Load llama.cpp reference outputs
    with open("tests/reference/llama_attention_outputs.json") as f:
        llama_ref = json.load(f)

    # Run C-Kernel-Engine
    ck_output = run_ck_attention(
        input=llama_ref["input"],
        weights=llama_ref["weights"],
        config=llama_ref["config"]
    )

    # Compare with tolerance
    atol = 1e-4  # Absolute tolerance
    rtol = 1e-3  # Relative tolerance

    matches = np.allclose(ck_output, llama_ref["output"], atol=atol, rtol=rtol)

    if matches:
        print("✅ Parity test PASSED")
        print(f"   Max diff: {np.max(np.abs(ck_output - llama_ref['output']))}")
    else:
        print("❌ Parity test FAILED")
        print(f"   Max diff: {np.max(np.abs(ck_output - llama_ref['output']))}")
        save_debug_output(ck_output, llama_ref["output"])

    return matches
```

### 2.3 Parity Test Categories
| Test | Description | Tolerance |
|------|-------------|-----------|
| QKV projection | Check Q/K/V values | 1e-5 |
| RoPE application | Check rotated values | 1e-5 |
| Attention scores | Check Q @ K.T / sqrt(d) | 1e-4 |
| Softmax output | Check attention weights | 1e-4 |
| Final output | Check O @ V + residual | 1e-4 |

## 3. Stability Tests: PyTorch Comparison

### 3.1 PyTorch Reference
```python
import torch
import torch.nn.functional as F

def pytorch_attention_reference(Q, K, V, causal=True):
    """PyTorch reference implementation."""
    scale = 1.0 / math.sqrt(Q.size(-1))

    # Q @ K.T / sqrt(d)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal:
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        scores = scores.masked_fill(mask == 1, -1e9)

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # @ V
    output = torch.matmul(attn, V)

    return output
```

### 3.2 Numerical Stability Tests
```c
// Test various edge cases for numerical stability
void test_numerical_stability(void) {
    // Test 1: Very small values (risk of underflow)
    float *small_input = zeros_tensor([1, H]);  // All zeros
    test_attention(small_input, "all_zeros");

    // Test 2: Very large values (risk of overflow)
    float *large_input = fill_tensor([1, H], 1e5f);
    test_attention(large_input, "large_values");

    // Test 3: Extreme aspect ratios (risk of softmax instability)
    float *mixed_input = random_tensor_with_range([1, H], -1e6f, 1e6f);
    test_attention(mixed_input, "extreme_range");

    // Test 4: Long sequence (2048 tokens)
    float *long_seq_K = random_tensor([2048, d]);
    float *long_seq_V = random_tensor([2048, d]);
    test_flash_attention_long_seq(long_seq_K, long_seq_V, "long_sequence");

    // Test 5: GQA edge case (num_kv_heads << num_heads)
    test_gqa_attention(4, 32, "gqa_4_32");  // 4 KV, 32 Q heads
}
```

## 4. Performance Tests: DRAM Pressure Measurement

### 4.1 Perf Events for DRAM Pressure
```bash
# Key perf events for measuring DRAM pressure reduction
PERF_EVENTS="
    cycles                          # Total CPU cycles
    instructions                    # Instructions per cycle
    cache-references                # Cache access attempts
    cache-misses                    # LLC misses = DRAM access
    LLC-loads                       # L1/L2 requests to L3
    LLC-load-misses                 # Requests that go to DRAM
    memory-load-retired.l1-miss     # Loads that miss L1
    memory-load-retired.l2-miss     # Loads that miss L2
    memory-load-retired.l3-miss     # Loads that miss L3 (go to DRAM)
    dram-reads                      # Actual DRAM reads
    dram-writes                     # Actual DRAM writes
"

# Record with these events
perf record -e $PERF_EVENTS -o perf_mega_fused.data -- ./ck-cli-v6 --test mega-fused
```

### 4.2 DRAM Pressure Test Script
```bash
#!/bin/bash
# test_dram_pressure.sh - Measure DRAM traffic reduction from mega-fusion

MODEL="Qwen2-0.5B-Instruct-GGUF"
MAX_TOKENS=100
ITERATIONS=5

echo "=== DRAM Pressure Test: Mega-Fused Attention ==="
echo "Model: $MODEL"
echo "Tokens: $MAX_TOKENS"
echo "Iterations: $ITERATIONS"
echo ""

# Test 1: Current implementation (unfused intermediates)
echo "Test 1: Current attention (baseline)..."
perf stat -e cycles,cache-misses,LLC-load-misses,dram-reads,dram-writes \
    ./build/ck-cli-v6.5 --model $MODEL --max-tokens $MAX_TOKENS \
    2>&1 | tee baseline_perf.txt

# Test 2: Mega-fused attention
echo ""
echo "Test 2: Mega-fused attention..."
perf stat -e cycles,cache-misses,LLC-load-misses,dram-reads,dram-writes \
    ./build/ck-cli-v6.5 --model $MODEL --max-tokens $MAX_TOKENS --mega-fused \
    2>&1 | tee megafused_perf.txt

# Compare
echo ""
echo "=== DRAM Pressure Comparison ==="
compare_dram_stats baseline_perf.txt megafused_perf.txt
```

### 4.3 Expected Results (The Whole Point!)

| Metric | Unfused | Mega-Fused | Improvement |
|--------|---------|------------|-------------|
| **LLC-load-misses** | ~10M | ~1M | **10× reduction** |
| **dram-reads** | ~800KB/token | ~8KB/token | **100× reduction** |
| **dram-writes** | ~800KB/token | ~8KB/token | **100× reduction** |
| **Memory bandwidth** | 50 GB/s | 0.5 GB/s | **100× reduction** |
| **Cycles/token** | ~100K | ~10K | **10× faster** |

### 4.4 Flamegraph for Visual Confirmation

```bash
# Generate flamegraph showing reduced memory operations
git clone https://github.com/brendangregg/FlameGraph 2>/dev/null

# Record stack traces with memory events
perf record -g -e memory-load-retired.l3-miss -o perf_memory.data \
    -- ./ck-cli-v6.5 --model $MODEL --max-tokens 100

# Generate flamegraph
perf script -i perf_memory.data | \
    stackcollapse-perf.pl | \
    flamegraph.pl --countname="L3 misses" > memory_flamegraph.svg

# Visual check:
# - Unfused: Large "memory" section in flamegraph
# - Fused: Tiny "memory" section (fusion working!)
```

### 4.5 Cache Locality Analysis
```bash
# Intel VTune Profiler commands
vtune -collect memory-access -r vtune_mega_fused -- ./ck-cli-v6.5 --mega-fused

# Analyze:
# - LLC hit ratio: should be > 95% for mega-fused
# - Memory bound: should drop from 80% to < 10%
# - Core cycles: more time in compute, less in memory wait
```

## 5. Test Execution Checklist

### 5.1 Before Running Tests
```bash
# 1. Build with debug symbols
make CK_DEBUG=1 ck-cli-v6.5

# 2. Ensure llama.cpp is built for reference
cd llama.cpp && make -j4

# 3. Ensure PyTorch is installed
pip3 install torch numpy

# 4. Generate test vectors
python3 scripts/test/generate_test_vectors.py --all
```

### 5.2 Run Tests in Order
```bash
# Step 1: Unit tests
make test UNIT_TESTS=1

# Step 2: Parity tests (llama.cpp)
make test PARITY_TESTS=1

# Step 3: Stability tests (PyTorch)
python3 scripts/test/test_stability.py

# Step 4: Performance tests (DRAM pressure)
./scripts/test/test_dram_pressure.sh --mega-fused

# Step 5: Flamegraph
./scripts/test/gen_flamegraph.sh --mega-fused
```

### 5.3 Expected Test Results
```
=== TEST RESULTS ===

Unit Tests:
  ✅ test_fused_rmsnorm_qkv        PASS (max diff: 2.1e-6)
  ✅ test_fused_flash_attention    PASS (max diff: 8.3e-5)
  ✅ test_mega_fused_complete      PASS (max diff: 1.2e-4)

Parity Tests (llama.cpp):
  ✅ QKV projection parity         PASS (atol=1e-4)
  ✅ RoPE application parity       PASS (atol=1e-5)
  ✅ Flash attention parity        PASS (atol=1e-4)
  ✅ Complete block parity         PASS (atol=1e-4)

Stability Tests (PyTorch):
  ✅ all_zeros                    PASS
  ✅ large_values                 PASS
  ✅ extreme_range                PASS
  ✅ long_sequence (2048)         PASS
  ✅ gqa_4_32                     PASS

Performance Tests (DRAM Pressure):
  ⚠️  LLC-load-misses:  10.2M → 0.98M  (10.4× reduction) ✅
  ⚠️  dram-reads:       800KB → 8KB    (100× reduction) ✅
  ⚠️  dram-writes:      800KB → 8KB    (100× reduction) ✅
  ⚠️  tokens/sec:       1.0 → 8.5      (8.5× speedup)   ✅

Flamegraph Analysis:
  ✅ Memory operations: 45% → 3% of total (fusion working!)

=== ALL TESTS PASSED ===
```

## 6. CI/CD Integration

```yaml
# .github/workflows/mega-fusion-tests.yml
name: Mega-Fusion Tests

on:
  push:
    paths:
      - 'src/kernels/mega_fused*.c'
      - 'include/*fused*.h'
      - 'scripts/test/*mega*'

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Build with debug
        run: make CK_DEBUG=1 ck-cli-v6.5

      - name: Run unit tests
        run: make test UNIT_TESTS=1

      - name: Run parity tests
        run: make test PARITY_TESTS=1

      - name: Run performance tests
        run: |
          ./scripts/test/test_dram_pressure.sh --mega-fused
          ./scripts/test/gen_flamegraph.sh --mega-fused

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: mega-fusion-test-results
          path: |
            test_results/*.txt
            test_results/*.svg
```

## Summary

| Test Type | Purpose | Key Metric |
|-----------|---------|------------|
| Unit tests | Numerical correctness | Max diff < 1e-4 |
| Parity tests | llama.cpp agreement | All outputs match within tolerance |
| Stability tests | Edge cases | No NaN/Inf, stable for all inputs |
| Performance tests | DRAM pressure | LLC misses, DRAM read/write reduction |
| Flamegraph | Visual confirmation | Memory section shrinks dramatically |

**The critical test:** DRAM pressure measurement proving that mega-fusion actually reduces memory traffic (the whole point!).
