# Kernel Development Workflow

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      KERNEL DEVELOPMENT PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │  1. WRITE    │     │  2. TEST     │     │  3. GENERATE │               │
│   │   KERNEL     │────▶│   KERNEL     │────▶│  KERNEL MAP  │               │
│   └──────────────┘     └──────────────┘     └──────────────┘               │
│                                                    │                         │
│                                                    ▼                         │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │  6. USE IN   │     │  5. SYNC     │     │  4. VALIDATE │               │
│   │  IR/CODEGEN  │◀────│   CHECK      │◀────│  KERNEL MAP  │               │
│   └──────────────┘     └──────────────┘     └──────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Workflow

### Step 1: Write Kernel

```
src/kernels/my_new_kernel.c
```

Follow kernel rules:
- NO malloc/free (bump allocator)
- NO OpenMP (parallelization at orchestrator layer)
- Pure computation, deterministic
- Document inputs/outputs in header comment

### Step 2: Test Kernel

```
unittest/test_my_new_kernel.py
```

Tests must cover:
1. **Numerical parity** with PyTorch/NumPy reference
2. **SIMD variants** (ref, SSE, AVX, AVX2, AVX512, VNNI, AMX)
3. **Edge cases** (small sizes, alignment, etc.)
4. **llama.cpp parity** (if applicable)

### Step 3: Generate Kernel Map

```bash
python3 version/v6.6/scripts/gen_kernel_map.py --kernel my_new_kernel
```

This script:
1. Finds kernel in `src/kernels/`
2. Parses function signature
3. Finds unit test in `unittest/`
4. Runs unit test to discover SIMD variants
5. Generates `kernel_maps/my_new_kernel.json` template
6. Developer fills in: parallelization, constraints, scratch

### Step 4: Validate Kernel Map

```bash
python3 version/v6.6/scripts/validate_kernel_maps.py --strict
```

### Step 5: Sync Check

```bash
python3 version/v6.6/scripts/check_kernel_map_sync.py
```

Ensures:
- Kernel map `impl.function` matches actual function name
- Kernel map `impl.sources` points to correct file
- All SIMD variants listed in map exist in source

### Step 6: Regenerate Registry

```bash
python3 version/v6.6/scripts/gen_kernel_registry_from_maps.py
```

## The gen_kernel_map.py Script

### What It Does

```python
def gen_kernel_map(kernel_name: str) -> dict:
    """
    Generate kernel map template from source and tests.

    1. Find kernel source file
    2. Parse function signature(s)
    3. Find unit test
    4. Run unit test to discover:
       - SIMD variants that pass
       - Input/output shapes
       - Tolerance levels
    5. Generate JSON template
    """
```

### Example Run

```bash
$ python3 scripts/gen_kernel_map.py --kernel gemv_q5_0_q8_0

[search] looking for gemv_q5_0_q8_0 in src/kernels/...
[found]  src/kernels/gemm_kernels_q5_0.c:1273

[parse]  signature: void gemv_q5_0_q8_0(float *y, const void *W, const void *x, int M, int K)
[infer]  op: gemv
[infer]  quant: weight=q5_0, activation=q8_0, output=fp32

[search] looking for unit test...
[found]  unittest/test_gemm_q5_0_q8_0.py

[test]   running SIMD variant discovery...
         ✓ ref      (tolerance: 1e-5)
         ✓ sse      (tolerance: 1e-5)
         ✓ avx      (tolerance: 1e-5)
         ✓ avx2     (tolerance: 1e-5)
         ✓ avx512   (tolerance: 1e-5)

[test]   running llama.cpp parity...
         ✓ parity   (tolerance: 1e-4)

[generate] kernel_maps/gemv_q5_0_q8_0.json

[TODO] Please fill in manually:
       - parallelization.strategies
       - constraints.alignment
       - scratch (if any)
```

### Generated Template

```json
{
  "id": "gemv_q5_0_q8_0",
  "op": "gemv",
  "variant": "q5_0_w_q8_0_a_fp32_out",
  "quant": {
    "weight": "q5_0",
    "activation": "q8_0",
    "output": "fp32"
  },
  "inputs": [
    {"name": "W", "dtype": "q5_0", "shape": ["M", "K"], "desc": "TODO: describe"},
    {"name": "x", "dtype": "q8_0", "shape": ["K"], "desc": "TODO: describe"}
  ],
  "outputs": [
    {"name": "y", "dtype": "fp32", "shape": ["M"], "desc": "TODO: describe"}
  ],
  "scratch": [],
  "dims": ["M", "K"],
  "parallelization": {
    "supported": ["TODO"],
    "preferred": {"prefill": "TODO", "decode": "TODO"},
    "strategies": [
      {
        "name": "TODO",
        "partition_dim": "TODO",
        "param_style": "TODO",
        "notes": "TODO"
      }
    ]
  },
  "constraints": {
    "alignment": {"K": "TODO: e.g., 256 for Q5_0"},
    "notes": "TODO"
  },
  "impl": {
    "function": "gemv_q5_0_q8_0",
    "sources": ["src/kernels/gemm_kernels_q5_0.c"],
    "variants": [
      {"name": "ref", "requires": []},
      {"name": "sse", "requires": ["sse4.1"], "compile_flags": ["-msse4.1"]},
      {"name": "avx", "requires": ["avx"], "compile_flags": ["-mavx"]},
      {"name": "avx2", "requires": ["avx2", "fma"], "compile_flags": ["-mavx2", "-mfma"]},
      {"name": "avx512", "requires": ["avx512f"], "compile_flags": ["-mavx512f"]}
    ]
  },
  "tests": {
    "unit": ["unittest/test_gemm_q5_0_q8_0.py"],
    "parity": [
      {
        "kind": "numpy",
        "command": "python unittest/test_gemm_q5_0_q8_0.py --compare-numpy",
        "tolerance": "1e-5"
      },
      {
        "kind": "llamacpp",
        "command": "make llamacpp-parity-gemv-q5-0-q8-0",
        "tolerance": "1e-4"
      }
    ]
  },
  "_generated": {
    "by": "gen_kernel_map.py",
    "date": "2026-01-19",
    "needs_review": ["parallelization", "constraints", "scratch"]
  }
}
```

## Unit Test Requirements

For `gen_kernel_map.py` to work, unit tests should follow this pattern:

```python
# unittest/test_my_kernel.py

import numpy as np
import ctypes

# Required: list of SIMD variants to test
VARIANTS = ["ref", "sse", "avx", "avx2", "avx512"]

# Required: test function that accepts variant name
def test_kernel(variant: str = "ref"):
    """Test kernel with specified SIMD variant."""
    # Set up test data
    M, K = 256, 512
    W = np.random.randn(M, K).astype(np.float32)
    x = np.random.randn(K).astype(np.float32)

    # Call C kernel
    y = call_kernel(variant, W, x, M, K)

    # Reference (numpy)
    y_ref = W @ x

    # Check
    np.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-5)

# Required: discover available variants
def get_available_variants() -> list:
    """Return list of variants that compile/run on this machine."""
    available = []
    for v in VARIANTS:
        try:
            # Try to load the variant-specific function
            if can_run_variant(v):
                available.append(v)
        except:
            pass
    return available

# Optional: llama.cpp parity test
def test_llamacpp_parity():
    """Compare against llama.cpp reference."""
    ...
```

## Directory Structure

```
version/v6.6/
├── scripts/
│   ├── gen_kernel_map.py          # Generate single kernel map
│   ├── gen_kernel_maps_batch.py   # Generate maps for all kernels
│   ├── gen_kernel_registry_from_maps.py  # Build registry from maps
│   ├── validate_kernel_maps.py    # Validate map JSON
│   ├── check_kernel_map_sync.py   # Sync maps ↔ sources
│   └── run_kernel_tests.py        # Run all kernel tests
├── kernel_maps/
│   ├── KERNEL_REGISTRY.json       # Generated from maps
│   ├── gemv_q5_0_q8_0.json        # Manual (reviewed)
│   ├── mega_fused_attention_prefill.json
│   └── ...
└── docs/
    └── KERNEL_DEV_WORKFLOW.md     # This document
```

## Makefile Integration

```makefile
# Kernel development targets

# Generate kernel map for a specific kernel
kernel-map-%:
	python3 version/v6.6/scripts/gen_kernel_map.py --kernel $*

# Validate all kernel maps
validate-maps:
	python3 version/v6.6/scripts/validate_kernel_maps.py --strict

# Sync check (maps vs sources)
sync-check:
	python3 version/v6.6/scripts/check_kernel_map_sync.py

# Regenerate registry from maps
registry:
	python3 version/v6.6/scripts/gen_kernel_registry_from_maps.py

# Full kernel pipeline
kernel-pipeline: validate-maps sync-check registry
	@echo "[ok] kernel pipeline complete"

# Test a specific kernel (all variants)
test-kernel-%:
	python3 unittest/test_$*.py --all-variants

# Add new kernel (interactive)
new-kernel:
	@echo "1. Create src/kernels/NEW_KERNEL.c"
	@echo "2. Create unittest/test_NEW_KERNEL.py"
	@echo "3. Run: make kernel-map-NEW_KERNEL"
	@echo "4. Edit kernel_maps/NEW_KERNEL.json (fill TODOs)"
	@echo "5. Run: make kernel-pipeline"
```

## CI Integration

```yaml
# .github/workflows/kernel-ci.yml

name: Kernel CI

on:
  push:
    paths:
      - 'src/kernels/**'
      - 'unittest/test_*.py'
      - 'version/v6.6/kernel_maps/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate kernel maps
        run: python3 version/v6.6/scripts/validate_kernel_maps.py --strict

      - name: Sync check
        run: python3 version/v6.6/scripts/check_kernel_map_sync.py --strict

      - name: Run kernel tests
        run: make test-kernels

      - name: Regenerate registry (check for drift)
        run: |
          python3 version/v6.6/scripts/gen_kernel_registry_from_maps.py
          git diff --exit-code version/v6.6/kernel_maps/KERNEL_REGISTRY.json
```

## Summary

| Step | Script | Input | Output |
|------|--------|-------|--------|
| Write | (manual) | - | `src/kernels/foo.c` |
| Test | (manual) | - | `unittest/test_foo.py` |
| Generate | `gen_kernel_map.py` | kernel name | `kernel_maps/foo.json` (template) |
| Review | (manual) | template | `kernel_maps/foo.json` (complete) |
| Validate | `validate_kernel_maps.py` | maps | pass/fail |
| Sync | `check_kernel_map_sync.py` | maps + sources | pass/fail |
| Registry | `gen_kernel_registry_from_maps.py` | maps | `KERNEL_REGISTRY.json` |
| Use | `build_ir_v6_6.py` | registry | IR with kernels |
