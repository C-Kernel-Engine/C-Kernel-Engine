# Parity Testing CI/CD System

Layer-by-layer comparison between llama.cpp and C-Kernel-Engine to catch model compatibility issues automatically.

## Quick Start

```bash
# 1. Check CKE for NaN/Inf (no llama.cpp needed)
python version/v6.6/scripts/parity_report.py --ck-dump ck_parity_dumps/dump.bin

# 2. View in browser with pretty UI
python version/v6.6/scripts/parity_report.py --ck-dump ck_parity_dumps/dump.bin --view

# 3. Generate terminal-only report
python version/v6.6/scripts/parity_report.py --ck-dump ck_parity_dumps/dump.bin --output report.json
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. RUN CKE WITH DUMPING                                             │
│  ├─ Compile with -DCK_PARITY_DUMP                                    │
│  ├─ Run inference                                                     │
│  └─ Dumps each op's output to ck_parity_dumps/dump.bin                 │
├─────────────────────────────────────────────────────────────────────┤
│  2. RUN PARITY_REPORT.PY                                              │
│  ├─ Reads dump.bin                                                    │
│  ├─ Checks each tensor for NaN/Inf                                     │
│  ├─ Compares with reference (if available)                            │
│  └─ Generates report.json + terminal output                            │
├─────────────────────────────────────────────────────────────────────┤
│  3. VIEW IN PARITY_VIEWER.HTML                                        │
│  ├─ Load report.json                                                  │
│  ├─ Shows summary cards, expandable layer groups                      │
│  ├─ Color-coded status (PASS/FAIL/ERROR)                               │
│  └─ Interactive filtering                                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Files Created

| File | Purpose |
|------|---------|
| `version/v6.6/scripts/ck_parity_dump.h` | Dump API header (included in generated C when `--parity-dump` is on) |
| `version/v6.6/scripts/parity_test.py` | Basic dump reader + comparison (CLI tool) |
| `version/v6.6/scripts/parity_report.py` | Full report generator (JSON + terminal + viewer) |
| `version/v6.6/tools/parity_viewer.html` | Interactive viewer with expand/collapse |

## How to Enable Dumping

### Method 1: Using codegen (when regenerating model)

```bash
python version/v6.6/scripts/codegen_v6_6.py \
    --decode lowered_decode_call.json \
    --init init_call.json \
    -o model_v6_6.c \
    --parity-dump

# Compile
gcc -o model_parity model_v6_6.c -lckernel_engine -lm -DCK_PARITY_DUMP
```

### Method 2: Using ck-cli (built-in parity mode)

```bash
./build/ck-cli-v6.6 \
    --model Qwen3-0.6B-GGUF \
    --parity-dump \
    --prompt "Hello" \
    --max-tokens 10
```

This automatically:
1. Runs with `CK_PARITY_DUMP` defined
2. Creates `ck_parity_dumps/dump.bin`
3. Generates `parity_report.json`

### Method 3: Manual in C code

```c
#define CK_PARITY_DUMP 1
#include "ck_parity_dump.h"

// In your inference loop:
ck_dump_init("ck_parity_dumps");

for each_token:
    // ... run inference ...
    ck_dump_set_token(token_id);
    // ... inference happens ...
    // (dumps happen automatically via instrumentation)

ck_dump_close();
```

## Understanding the Report

### Status Types

| Status | Meaning | Action |
|--------|---------|--------|
| **PASS** | Values match within tolerance | No action needed |
| **FAIL** | Diverges from llama.cpp | Check weight offsets, kernel implementation |
| **ERROR** | Contains NaN or Inf | Critical issue — fix immediately |
| **WARN** | No reference available | Only self-check performed |

### Example Output

```
============================================================================================================================================
PARITY REPORT - Qwen3-0.6B-GGUF
Generated: 2025-02-01T15:30:00
============================================================================================================================================

Operations:            451
  ✓ PASS:               441
  ✗ FAIL:                8
  ⚠ ERROR:               2
  ⚠ WARN:               0

Model Info
--------------------------
  model                  Qwen3-0.6B
  num_hidden_layers       28
  hidden_size            1024
  num_attention_heads    16
  num_key_value_heads   8
  vocab_size             151936

============================================================================================================================================
Layer  Op                   Status   Max Abs      Mean Abs     Max Rel    Ref Range             CKE Range
--------------------------------------------------------------------------------------------------------------------------------------------
0      embed_out           PASS     1.23e-06     3.45e-07     0.00%     [-2.34e+00, 3.45e+00]  [-2.34e+00, 3.45e+00]
0      attn_norm           PASS     2.12e-06     5.67e-08     0.01%     [-1.23e+00, 2.34e+00]  [-1.23e+00, 2.34e+00]
0      q_proj              FAIL     1.45e-01     3.21e-02     15.2%     [-3.45e-01, 4.56e-01]  [-2.12e-01, 5.67e-01]  @ idx 1234
0      qk_norm             ERROR    NaN/Inf      ...          ...        ...                   ...                   NaN=True, Inf=False
...
```

## Debugging a Failed Operation

If `q_proj` FAILS at layer 0:

1. **Check weight offset in manifest**:
   ```bash
   python -c "
   import json
   with open('weights_manifest.json') as f:
       m = json.load(f)
   for e in m['entries']:
           if 'layer.0.wq' in e['name']:
               print(f\"offset: {e['file_offset']}, size: {e['size']}\")
   "
   ```

2. **Compare with llama.cpp**:
   - Run llama.cpp with same input
   - Dump its q_proj tensor
   - Compare values to find exact divergence point

3. **Check kernel implementation**:
   - Verify gemv_q8_0_q8_0 is correct
   - Check quantization format matches

4. **Check model config**:
   - head_dim matches?
   - num_heads correct?

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/parity.yml
name: Model Parity Test

on: [push, pull_request]

jobs:
  parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          pip install numpy

      - name: Convert GGUF to BUMP
        run: |
          python version/v6.6/scripts/convert_gguf_to_bump_v6_6.py \
            --gguf ${{ env.MODEL_PATH }} \
            --output weights.bump

      - name: Generate IR with parity dump
        run: |
          python version/v6.6/scripts/build_ir_v6_6.py \
            --manifest weights_manifest.json \
            --parity-dump \
            --output ir1_decode.json

      - name: Compile with dump
        run: |
          gcc -o model model_v6_6.c -lckernel_engine -lm -DCK_PARITY_DUMP

      - name: Run inference with dump
        run: |
          ./model weights.gguf "Hello" 5 || true

      - name: Generate parity report
        run: |
          python version/v6.6/scripts/parity_report.py \
            --ck-dump ck_parity_dumps/dump.bin \
            --output parity_report.json

      - name: Check for errors
        run: |
          python -c "
          import json
          with open('parity_report.json') as f:
              data = json.load(f)
          assert data['summary']['errors'] == 0, f\"Parity errors: {data['summary']['errors']}\"
          "
```

## Troubleshooting

### "No CKE dumps found"

- Make sure you compiled with `-DCK_PARITY_DUMP`
- Make sure `ck_dump_init()` is called before inference
- Check file permissions

### "Invalid magic number"

- The dump file is corrupted or from an old version
- Delete `ck_parity_dumps/dump.bin` and regenerate

### All ops show "WARN - No reference"

- This is expected when running without llama.cpp reference
- Still checks for NaN/Inf in CKE outputs

## Tips

1. **Start with check-only mode** to verify CKE has no NaN/Inf before adding llama.cpp comparison
2. **Use small token counts** (1-5 tokens) when debugging
3. **Filter to "ERROR" status** to focus on critical issues first
4. **Expand layer groups** to see all operations in a layer
5. **Export JSON** for offline analysis or sharing

## See Also

- `unittest/test_qk_norm.py` - Kernel-level unit tests
- `version/v6.6/scripts/build_ir_v6_6.py` - IR generation
- `version/v6.6/tools/ir_visualizer.html` - IR pipeline visualizer
