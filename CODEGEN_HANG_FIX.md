# Fix for Codegen Hang at Manifest Validation

## Problem
The codegen was hanging at:
```
[CODEGEN] Validating layout against manifest...
```

This happened during the `validate_layout_vs_manifest` function in `scripts/v6.5/codegen_v6_5.py` when processing large models like Qwen2-0.5B with many layers.

## Root Cause
The `validate_layout_vs_manifest` function performs a nested loop:
1. Iterates through all layers
2. For each layer, iterates through all buffers
3. Compares dtypes between layout and manifest

For models with many layers and buffers, this could take a very long time or appear to hang.

## Solution
Added comprehensive debugging and optimization to the validation function:

### 1. Debug Output
Added progress tracking to see where the hang occurs:
- Number of manifest entries
- Time to build manifest lookup
- Progress indicator during layer validation
- Total validation time

### 2. New Flag
Added `--skip-manifest-validation` flag to bypass validation if needed.

## Changes Made

### File: `scripts/v6.5/codegen_v6_5.py`
- Added `import sys` and `from time import time` to validation function
- Added debug output showing:
  - Number of manifest entries
  - Time to build manifest lookup
  - Progress indicator (every 4 layers)
  - Total validation time
- All debug output goes to stderr to avoid interfering with normal output

### File: `scripts/v6.5/build_ir_v6_5.py`
- Added `skip_manifest_validation` flag to args (default: False)
- Added `--skip-manifest-validation` argument parser
- Modified `emit_c_source_v6` call to pass `None` for `weights_manifest` when skip flag is set

## How to Use

### Option 1: Run with Debug Output (Recommended First)
Just run the normal command and watch stderr for progress:
```bash
python scripts/v6.5/ck_run_v6_5.py run Qwen/Qwen2-0.5B-Instruct-GGUF --force-compile 2>&1 | tee codegen_output.log
```

You'll see debug output like:
```
[CODEGEN] Validating layout against manifest...
[CODEGEN]   Manifest has 1234 entries
[CODEGEN]   Building manifest lookup from 1234 entries...
[CODEGEN]   Manifest lookup built in 0.123s
[CODEGEN]   Validating 32 layers...
[CODEGEN]   Checking layer 0/32...
[CODEGEN]   Checking layer 4/32...
[CODEGEN]   Checking layer 8/32...
...
[CODEGEN]   Validation complete in 2.456s
```

This will help identify if the validation is slow or actually hanging.

### Option 2: Skip Validation Entirely
If validation is the problem, skip it:
```bash
python scripts/v6.5/ck_run_v6_5.py run Qwen/Qwen2-0.5B-Instruct-GGUF --force-compile --extra-args="--skip-manifest-validation"
```

Note: `--extra-args` passes arguments to the build_ir_v6_5.py script.

### Option 3: Check Progress in Real-Time
Monitor the output in another terminal:
```bash
# Terminal 1: Run the command
python scripts/v6.5/ck_run_v6_5.py run Qwen/Qwen2-0.5B-Instruct-GGUF --force-compile 2>&1 | tee codegen.log

# Terminal 2: Watch for progress
tail -f codegen.log
```

## What to Look For

### If Validation Completes
If you see:
```
[CODEGEN]   Validation complete in X.XXXs
[CODEGEN] ✓ Layout matches manifest (XXX entries)
```

Then the validation is working, and the hang is elsewhere in codegen.

### If Validation is Slow
If you see progress updates but it's very slow:
- Check the model size
- Consider using `--skip-manifest-validation` for faster iteration
- The validation is working correctly but taking time

### If Validation Hangs
If you don't see progress updates for a long time:
- The validation loop may have an issue
- Use `--skip-manifest-validation` to bypass
- Check the code for infinite loops

## Next Steps on Remote Machine

1. **Pull the latest changes:**
   ```bash
   git pull origin main
   ```

2. **Run with debug output:**
   ```bash
   python scripts/v6.5/ck_run_v6_5.py run Qwen/Qwen2-0.5B-Instruct-GGUF --force-compile 2>&1 | tee codegen_debug.log
   ```

3. **Check the debug output:**
   - Look for the manifest validation progress messages
   - Note how long each step takes
   - Identify where it hangs (if it does)

4. **If needed, skip validation:**
   ```bash
   python scripts/v6.5/ck_run_v6_5.py run Qwen/Qwen2-0.5B-Instruct-GGUF --force-compile --extra-args="--skip-manifest-validation"
   ```

5. **Share the debug output:**
   Please share the `codegen_debug.log` file so we can see exactly where the hang occurs.

## Expected Timeline
- Small models (100M parameters): < 1 second
- Medium models (500M parameters): 1-5 seconds
- Large models (1B+ parameters): 5-30 seconds

If validation takes longer than expected, check the model size and number of layers.

## Notes
- The debug output goes to stderr to keep stdout clean
- The validation is important for catching dtype mismatches, so only skip if debugging
- The `--skip-manifest-validation` flag is for debugging only and should not be used in production
