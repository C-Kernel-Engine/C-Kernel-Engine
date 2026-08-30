# Training Fix - 2025-12-26

## Problem

Training loop showed loss staying constant at exactly `10.802672` for all steps:
```
Step 1: loss = 10.802672
Step 2: loss = 10.802672
Step 3: loss = 10.802672
Warning: Loss didn't decrease
```

## Root Cause Analysis

### Issue 1: Gradient buffers not allocated

In `src/ckernel_codegen.c`, gradient buffer offsets were conditionally allocated:
```c
m->d_token_emb_offset = m->training_enabled ? bump_bytes(...) : 0;
```

The problem:
1. Model is initialized with `training_enabled = false` (default)
2. Gradient buffer offsets are set to `0` during layout
3. User calls `enable_training()` AFTER init
4. But gradient buffers were never allocated - offsets are still 0
5. `sgd_update()` checks `if (offset)` before updating - fails silently
6. No weights get updated -> Loss stays constant

### Issue 2: Duplicate SGD update

`run_model_backward()` was calling `sgd_update()` at the end, then user's code called `optimizer_step()` which called `sgd_update()` again. Double update per iteration.

## Fixes Applied

### Fix 1: Environment variable for training mode

Added check for `CK_ENABLE_TRAINING=1` at model init time (before layout):

```c
// In ck_model_init():
const char *train_env = getenv("CK_ENABLE_TRAINING");
if (train_env && (train_env[0] == '1' || train_env[0] == 'y' || train_env[0] == 'Y')) {
    g_model.training_enabled = true;
    g_model.learning_rate = 1e-4f;
}
if (layout_model(&g_model) != 0) return -1;
```

This pre-enables training BEFORE layout, so gradient buffers get allocated.

**File**: `src/ckernel_codegen.c` lines 820-825

### Fix 2: Remove duplicate SGD from backward

Removed `sgd_update(m, m->learning_rate);` from end of `run_model_backward()`.
SGD now only happens when user explicitly calls `optimizer_step()`.

**File**: `src/ckernel_codegen.c` line 1268

## Files Modified

- `src/ckernel_codegen.c` - Both fixes
- `scripts/test_training_fix.sh` - Test script (new)

## How to Test

```bash
./scripts/test_training_fix.sh
```

Or manually:
```bash
make clean && make && make ck-cli build/ck_ir_demo

./build/ck_ir_demo ~/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M/config.json \
    --emit ~/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M/model.c --emit-lib

cd ~/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M
icx -O3 -fPIC -shared -I$PROJECT/include -o libmodel.so model.c \
    -L$PROJECT/build -lckernel_engine -lm

CK_ENABLE_TRAINING=1 LD_LIBRARY_PATH=$PROJECT/build \
    python3 scripts/test_training.py --model-dir ~/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M --steps 5
```

## Expected Result

Loss should DECREASE across training steps:
```
Step 1: loss = 10.8xxx
Step 2: loss = 10.7xxx  <- should be lower
Step 3: loss = 10.6xxx  <- should keep decreasing
...
Loss decreased: 10.8xxx -> 10.xxxx
```

## Memory Note

With `CK_ENABLE_TRAINING=1`, the model allocates ~2x memory (weights + gradients).
For SmolLM-135M, expect ~2-3GB RAM usage during training.

## Next Steps

- [ ] Test on machine with more memory
- [ ] Create PyTorch parity test to verify gradient correctness
- [ ] Consider adding `ck_model_init_for_training()` API as alternative to env var
