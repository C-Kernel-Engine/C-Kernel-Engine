# Detailed Parity Testing Implementation Plan

## Target Command
```bash
python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
    --context-len 1024 \
    --force-compile \
    --detailed-llamacpp-parity
```

## Key Requirements
- Llama reference dumps must include dtype and shape. Raw bytes alone are not enough.
- Prefer CKDMP output directly from llama.cpp. If raw dumps are used, also emit an index file.
- Run prefill and decode in separate passes to avoid overwriting filenames.
- Compare by `(layer_id, op_name, token_id)` after applying model-family mapping.

## First-Mismatch Workflow (Minimal Dumps)
Goal: find the first divergence quickly without dumping everything.

1. End-to-end check (logits only)

```bash
python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --context-len 1024 \
  --force-compile \
  --detailed-llamacpp-parity \
  --max-tokens 1 \
  --llama-filter logits \
  --llama-stop-after 1
```

2. Layer 0 check (core ops only)

```bash
python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --context-len 1024 \
  --force-compile \
  --detailed-llamacpp-parity \
  --max-tokens 1 \
  --llama-layer 0 \
  --llama-include-global \
  --llama-filter attn_norm,Qcur,Kcur,Vcur,attn_out,ffn_norm,ffn_gate,ffn_up,ffn_down
```

3. If llama.cpp still times out, make it even smaller

```bash
python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --context-len 1024 \
  --force-compile \
  --detailed-llamacpp-parity \
  --max-tokens 1 \
  --llama-layer 0 \
  --llama-include-global \
  --llama-filter attn_norm,Qcur,Kcur,Vcur \
  --llama-stop-after 32
```

4. If you truly need full-layer dumps, bump the timeout

```bash
python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --context-len 1024 \
  --force-compile \
  --detailed-llamacpp-parity \
  --max-tokens 1 \
  --llama-timeout 900
```

5. Optional: limit CKE dumps with CK_STOP_OP

```bash
CK_STOP_OP=26 python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --context-len 1024 \
  --force-compile \
  --parity-dump \
  --max-tokens 1
```

Notes:
`CK_STOP_OP` stops the CKE graph at a specific op index. Use `ir1_prefill.json` in the model build directory to identify op ordering if you want a clean stop right after a specific layer op.
For ck-cli parity, use `--max-tokens 1` even if you only care about prefill. `--max-tokens 0` is treated as the default (256) and can cause prompt truncation when `--context-len 256`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              DETAILED LLAMACPP PARITY TESTING PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ PHASE 1: SETUP                                                         │
  │  • Check llama.cpp exists at pinned commit                             │
  │  • Apply CKDMP dump patch if needed                                    │
  │  • Build llama.cpp                                                     │
  │  • Detect model family from GGUF metadata                              │
  │  • Load model-family mappings                                          │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ PHASE 2: LLAMA.CPP REFERENCE RUN                                       │
  │  • Run llama.cpp with same prompt/settings                             │
  │  • Dump tensors as CKDMP to llama_parity_dumps/dump.bin                │
  │  • Separate runs for prefill and decode                                │
  │  • If raw dumps are used, also write llama_dump/index.json             │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ PHASE 3: CK-ENGINE INFERENCE                                           │
  │  • Run CK-engine with same prompt/settings                             │
  │  • Dump tensors to ck_parity_dumps/dump.bin (CKDMP)                     │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ PHASE 4: MATCH + COMPARE                                                │
  │  • Normalize op names using model-family mapping                        │
  │  • Match by (layer_id, op_name, token_id)                               │
  │  • Compare with tolerance and find FIRST divergence                      │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ PHASE 5: REPORT GENERATION                                             │
  │  • Generate detailed HTML/terminal report                               │
  │  • Show first divergence with layer, op, shape, index, values           │
  └─────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
├── patches/
│   ├── llama.patch                  # Existing raw dump patch (layer 0 only)
│   └── llama_tensor_dump_ckdmp.patch   # Planned: CKDMP dump for all layers
│
├── scripts/
│   ├── llama_to_ck_converter.py     # Optional: raw dump → CKDMP using index.json
│   └── model_family_detector.py     # Optional: GGUF-based family detection
│
└── version/v6.6/scripts/
    └── ck_run_v6_6.py               # Uses ck-cli-v6.6 for detailed parity
```

## Model-Family Registry Format

```json
{
  "families": {
    "gemma": {
      "architectures": ["gemma"],
      "tensor_mapping": {
        "Qcur": "q_proj",
        "Kcur": "k_proj",
        "Vcur": "v_proj",
        "attn_norm": "attn_norm",
        "ffn_norm": "ffn_norm",
        "ffn_up": "ffn_up",
        "ffn_down": "ffn_down",
        "ffn_gate": "ffn_gate",
        "token_embd": "token_embd",
        "output_norm": "output_norm"
      },
      "expected_ops": [
        "attn_norm", "q_proj", "k_proj", "v_proj",
        "attn_out",
        "ffn_norm", "ffn_up", "ffn_gate", "ffn_down",
        "output_norm", "logits"
      ]
    },
    "qwen": {
      "architectures": ["qwen2", "qwen3"],
      "tensor_mapping": {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "attn_norm": "attn_norm",
        "ffn_norm": "ffn_norm",
        "ffn_up": "ffn_up",
        "ffn_gate": "ffn_gate",
        "ffn_down": "ffn_down"
      },
      "expected_ops": [
        "attn_norm", "q_proj", "k_proj", "v_proj",
        "attn_out",
        "ffn_norm", "ffn_up", "ffn_gate", "ffn_down"
      ]
    },
    "llama": {
      "architectures": ["llama"],
      "tensor_mapping": {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
        "w1": "ffn_gate",
        "w2": "ffn_down",
        "w3": "ffn_up"
      },
      "expected_ops": [
        "attn_norm", "q_proj", "k_proj", "v_proj",
        "attn_out",
        "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"
      ]
    },
    "mistral": {
      "architectures": ["mistral"],
      "tensor_mapping": {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "attn_norm": "attn_norm",
        "ffn_norm": "ffn_norm",
        "w1": "ffn_gate",
        "w2": "ffn_down",
        "w3": "ffn_up"
      },
      "expected_ops": [
        "attn_norm", "q_proj", "k_proj", "v_proj",
        "attn_out",
        "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"
      ]
    }
  }
}
```

## Name-Based Matching Logic

```python
def normalize_name(raw_name, mapping):
    """
    Normalize llama.cpp tensor name to CK op name.
    Examples:
      "Qcur-0" -> "q_proj"
      "attn_norm-12" -> "attn_norm"
    """
    base = raw_name.split("-")[0]
    return mapping.get(base, base)

def match_tensors(ck_tensors, llama_tensors, model_family):
    """
    Match by (layer_id, op_name, token_id) after applying model-family mapping.
    Prefill and decode should run as separate passes to avoid collisions.
    """
    mapping = MODEL_REGISTRY["families"][model_family]["tensor_mapping"]

    ck_index = {}
    for t in ck_tensors:
        key = (t.layer_id, t.op_name, t.token_id)
        ck_index[key] = t

    matched = []
    unmatched_llama = []

    for lt in llama_tensors:
        op = normalize_name(lt.op_name, mapping)
        key = (lt.layer_id, op, lt.token_id)
        ck_match = ck_index.get(key)
        if ck_match:
            matched.append((ck_match, lt, lt.layer_id, op))
        else:
            unmatched_llama.append(lt)

    return matched, unmatched_llama
```

## Divergence Detection

```python
def find_first_divergence(matched_pairs, atol=1e-4):
    """
    Find the FIRST point of divergence.

    Returns:
        divergence: {
            'layer': int,
            'op_name': str,  # e.g., "q_proj"
            'ck_max_diff': float,
            'llama_max_diff': float,
            'first_idx': int,
            'ck_value': float,
            'llama_value': float,
            'diagnosis': str,  # e.g., "GEMV truncated output"
        }
    """

    for ck, llama, layer, op_name in sorted(matched_pairs, key=lambda x: (x[2], x[3])):
        diffs = np.abs(ck.data - llama.data)
        max_diff = np.max(diffs)

        if max_diff > atol:
            first_idx = np.argmax(diffs)

            # Auto-diagnose
            diagnosis = diagnose_divergence(
                layer=layer,
                op_name=op_name,
                max_diff=max_diff,
                shape=ck.shape,
                ck_data=ck.data,
                llama_data=llama.data
            )

            return {
                'layer': layer,
                'op_name': op_name,
                'first_idx': int(first_idx),
                'ck_value': float(ck.data.flat[first_idx]),
                'llama_value': float(llama.data.flat[first_idx]),
                'max_diff': float(max_diff),
                'diagnosis': diagnosis
            }

    return None  # No divergence
```

## Auto-Diagnosis Rules

| Pattern | Diagnosis | Suggested Fix |
|---------|-----------|--------------|
| First 128/256/512 elements differ, rest are zero | GEMV with wrong K/stride | Check GEMV K parameter or sliding window |
| All elements zero | Wrong buffer/wiring | Check op input buffer |
| NaN/Inf values | Kernel bug or uninitialized | Check kernel implementation |
| Small random diffs (< 1e-3) | Quantization difference | Accept if within tolerance |
| Large systematic offset | Missing bias or offset | Check op parameters |
| Pattern repeats every N | RoPE phase issue | Check RoPE implementation |
| Gradual drift across layer | RMSNorm epsilon diff | Check norm epsilon value |
| Attention only | Attention kernel bug | Check softmax/sliding window |

## CK_RUN_V6_6.PY Integration

### Existing Flag: --detailed-llamacpp-parity

```python
parser.add_argument(
    "--detailed-llamacpp-parity",
    action="store_true",
    help="Run detailed llama.cpp parity test (CKDMP reference + report)."
)
```

Optional future flags if we want finer control:
- `--llamacpp-dir` to override the llama.cpp path
- `--parity-atol` to override tolerance

### New Method: run_detailed_parity()

```python
def run_detailed_parity(args):
    """
    Main entry point for --detailed-llamacpp-parity flag.

    1. Verify llama.cpp setup
    2. Run llama.cpp reference inference
    3. Run CK-engine inference with dumps
    4. Convert and compare
    5. Generate report
    """

    # Step 1: Check llama.cpp
    llama_dir = PROJECT_ROOT / "llama.cpp"
    check_llama_setup(llama_dir)

    # Step 2: Detect model family
    model_family = detect_model_family(args.model_path)
    print(f"[PARITY] Detected model family: {model_family}")

    # Step 3: Run llama.cpp reference
    llama_ref_path = run_llama_reference(llama_dir, args)

    # Step 4: Run CK with dumps
    ck_dump_path = run_ck_with_dumps(args)

    # Step 5: Convert llama dumps if needed
    if llama_ref_path.is_dir():
        ref_path = convert_llama_to_ck(llama_ref_path)
    else:
        ref_path = llama_ref_path

    # Step 6: Compare
    report = compare_parity(ck_dump_path, ref_path, model_family, atol=1e-4)

    # Step 7: Output report
    print_report(report)

    if args.view_html:
        open_html_report(report)
```

## Output Example

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   LLAMACPP PARITY TEST RESULTS                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Model: gemma-3-270m-it-Q5_K_M.gguf                                             ║
║ Family: gemma                                                                  ║
║ Prompt: "Hello!"                                                               ║
║ Tokens: 1                                                                      ║
║ Tolerance: atol=1e-4                                                           ║
╠═══════════════════════════════════════════════════════════════════════════════╣

LAYER 0
─────────────────────────────────────────────────────────────────────────────────
  Op          llama.cpp           CK Engine          Diff      Status
─────────────────────────────────────────────────────────────────────────────────
  token_embd   [-0.42, 0.38]       [-0.42, 0.38]      1.2e-06   PASS
  attn_norm    [-2.1, 2.1]         [-2.1, 2.1]        8.5e-06   PASS
  Qcur         [-3.2, 3.2]        [-3.2, 3.2]        3.2e-02   FAIL ⚠
  Kcur         [-2.8, 2.8]        [-2.8, 2.8]        2.1e-02   FAIL ⚠
  Vcur         [-3.1, 3.1]        [-3.1, 3.1]        4.5e-02   FAIL ⚠
  attn_out     [-2.5, 2.5]        [-2.5, 2.5]        2.8e-02   FAIL ⚠
  ffn_norm     [-2.2, 2.2]        [-2.2, 2.2]        9.1e-06   PASS
  ffn_up       [-1.9, 1.9]        [-1.9, 1.9]        1.2e-05   PASS
  ffn_gate     [-2.0, 2.0]        [-2.0, 2.0]        8.7e-06   PASS
  ffn_down     [-2.3, 2.3]        [-2.3, 2.3]        1.1e-05   PASS

LAYER 1
─────────────────────────────────────────────────────────────────────────────────
  Op          llama.cpp           CK Engine          Diff      Status
─────────────────────────────────────────────────────────────────────────────────
  attn_norm    [-2.0, 2.0]         [-2.0, 2.0]        7.2e-06   PASS
  ...                                                                         ║

╠═══════════════════════════════════════════════════════════════════════════════╣
║ FIRST DIVERGENCE                                                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ Layer: 0                                                                      ║
║ Op: Qcur (Query projection)                                                   ║
║ Shape: [1, 1024, 16]                                                         ║
║ First divergence @ index: 127                                                  ║
║                                                                                ║
║ Element 127:                                                                   ║
║   llama.cpp:  0.123456                                                         ║
║   CK Engine:   0.000000                                                       ║
║                                                                                ║
║ Max diff: 3.2e-02                                                             ║
║                                                                                ║
║ DIAGNOSIS: GEMV TRUNCATED OUTPUT                                               ║
║                                                                                ║
║ The Q projection output is truncated after 128 elements.                     ║
║ Expected: 1024 elements (16 heads × 64 dim each)                              ║
║ Actual: Only first 128 elements computed correctly                            ║
║                                                                                ║
║ LIKELY CAUSES:                                                                ║
║   1. sliding_window=512 but being treated as 128 somewhere                    ║
║   2. GEMV kernel with incorrect K/stride parameter                           ║
║   3. Attention buffer sized incorrectly for Gemma-3                         ║
║                                                                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ RECOMMENDED FIXES                                                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ 1. Check ck_engine for sliding_window parameter in attention                  ║
║    └─► Gemma-3 uses 512, not 0 like Llama                                     ║
║                                                                                ║
║ 2. Verify Q projection GEMV kernel:                                            ║
║    └─► Check n_cols, k stride, and output size                               ║
║                                                                                ║
║ 3. Cross-reference ggml/src/ggml-cpu/ggml-cpu.c for tensor shape              ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## Implementation Order

1. Add llama.cpp CKDMP dump patch (preferred). If raw dumps are used, add index.json emission.
2. Add model-family mapping and normalization utility.
3. Update parity_test.py to match by `(layer_id, op_name, token_id)` after mapping.
4. Optional: add raw→CKDMP converter using index.json.
5. Run prefill-only parity, then decode-only parity, on Gemma-3-270M.

---

## Questions/Clarifications Needed

1. **Reference dump format**: CKDMP direct from llama.cpp, or raw dumps + index.json + converter?
2. **Pass order**: Start with prefill-only, then decode-only?
3. **Output location**: Keep `ck_parity_dumps/` and `llama_parity_dumps/` under `ck_build/`?

Let me know if this plan looks complete or if you'd like any changes before I start implementing.
