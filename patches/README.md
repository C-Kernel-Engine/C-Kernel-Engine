# Patches for llama.cpp Parity Testing

This directory contains patches and files to apply to the llama.cpp submodule for parity testing.

## Files

| File | Purpose |
|------|---------|
| `llama.patch` | Tensor dump hack for ggml-cpu.c (dumps layer 0 intermediates) |
| `test-kernel-parity.cpp` | Kernel test library exposing GGML internals for parity testing |

## Setup

### 1. Add llama.cpp as submodule (one-time)

```bash
git submodule add https://github.com/ggerganov/llama.cpp.git llama.cpp
git submodule update --init
cd llama.cpp && git checkout b4876  # Pin to known-good version
cd ..
```

### 2. Apply patches

```bash
cd llama.cpp
git apply ../patches/llama.patch
cp ../patches/test-kernel-parity.cpp tests/
cd ..
```

### 3. Build llama.cpp

```bash
cd llama.cpp
cmake -B build -DGGML_CPU=ON
cmake --build build -j$(nproc)
cd ..
```

### 4. Run parity tests

```bash
# Quick kernel tests
make llamacpp-parity

# Full parity test
make llamacpp-parity-full
```

## Patches

### llama_tensor_dump.patch (optional)

Adds tensor dump functionality to llama.cpp for layer-by-layer comparison.
Creates `.bin` files in `llama_dump/` directory with intermediate activations:
- `attn_norm-{layer}.bin` - RMSNorm output before attention
- `Qcur-{layer}.bin` - Q projection output
- `Kcur-{layer}.bin` - K projection output
- etc.

To create this patch after modifying llama.cpp:
```bash
cd llama.cpp
git diff > ../patches/llama_tensor_dump.patch
```

## CI Integration

The nightly workflow will:
1. Initialize submodules automatically
2. Build llama.cpp if present
3. Run parity tests via `scripts/run_parity_smoketest.sh`

To trigger manually:
```bash
# Via GitHub Actions
gh workflow run nightly.yml -f test_category=llamacpp

# Locally
./scripts/run_parity_smoketest.sh --quick
```
