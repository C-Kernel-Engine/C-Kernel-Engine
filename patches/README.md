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
cd llama.cpp && git checkout 07fbe19f1fbcfa09abca7cccc62eaf82c1567b7e
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

The parity smoketest script (`scripts/run_parity_smoketest.sh`) will automatically:
1. Clone llama.cpp if not present (or init submodule)
2. Checkout the repo's pinned `llama.cpp` gitlink for reproducibility by default
3. Apply patches from this directory
4. Build llama.cpp with CPU support
5. Build the kernel test library
6. Run parity tests

### GitHub Actions / CI

```yaml
# In your workflow:
- name: Run llama.cpp parity tests
  run: |
    # Script handles clone, patch, build automatically
    ./scripts/run_parity_smoketest.sh --quick
```

### Local Usage

```bash
# Quick kernel tests (clones/patches if needed)
./scripts/run_parity_smoketest.sh --quick

# Full parity test
./scripts/run_parity_smoketest.sh

# Force rebuild (clean build)
./scripts/run_parity_smoketest.sh --force-rebuild

# Use specific llama.cpp commit
LLAMA_CPP_COMMIT=abc1234 ./scripts/run_parity_smoketest.sh

# Via make
make llamacpp-parity-full
```

### Pinned Version

The script defaults to the current `llama.cpp` gitlink in the superproject for reproducible testing.
To update the pinned version:

1. Test with new commit: `LLAMA_CPP_COMMIT=<new_commit> ./scripts/run_parity_smoketest.sh`
2. If tests pass, update the submodule gitlink: `cd llama.cpp && git checkout <new_commit> && cd .. && git add llama.cpp`
3. Commit the gitlink update together with any patch/workflow changes that depend on it

### Manual Trigger

```bash
# Via GitHub Actions
gh workflow run nightly.yml -f test_category=llamacpp

# Locally with fresh clone
rm -rf llama.cpp && ./scripts/run_parity_smoketest.sh
```
