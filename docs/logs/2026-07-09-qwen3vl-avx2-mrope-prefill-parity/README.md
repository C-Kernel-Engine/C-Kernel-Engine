# Qwen3-VL AVX2 M-RoPE and Prefill Parity Handoff

Date: 2026-07-09

## Scope

This log records the AVX2 Qwen3-VL OCR parity work on the canonical SDPR OCR image.

The goal was correctness, not speed. VTune/Advisor work should stay paused for this path until the parity split is closed.

## Canonical Inputs

- Image: `ocr/1 81.jpg`
- JPEG sha256: `eb3becd507063f184d417a6baccfdb22d86277308f21a679aac4e029ed8f25ff`
- Canonical PPM sha256: `81ab4a6dabf3fc41ef86b620602a28dbbba59aea89414301b604df355afda182`
- Decoder GGUF: `Qwen3VL-8B-Instruct-Q4_K_M.gguf`
- Decoder GGUF sha256: `67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2`
- mmproj GGUF: `mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`
- mmproj sha256: `c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd`
- Prompt: `Extract visible form fields as compact JSON.`
- Context: `4096`
- Threads: `20`
- Visual prefix: `1008 x 16384`
- Grid: `36 x 28`
- `prefix_text_pos`: `41`

## Fix 1: M-RoPE Width

Root cause:

CK derived `mrope_n_dims` from `sum(mrope_sections) == 64`. For Qwen3-VL llama.cpp passes `n_rot == head_dim == 128`. The section values select the time/height/width axis pattern; they are not the rotary width.

Changed:

- `version/v8/scripts/convert_gguf_to_bump_v8.py`
  - Emits full `mrope_n_dims` from rotary/head width.
- `version/v8/scripts/build_ir_v8.py`
  - Uses head width as stale-config fallback, not `sum(mrope_sections)`.

Result:

- Before: fresh AVX2 bridge failed at generated step 4.
  - CK token `262` (`'   '`)
  - llama token `220` (`' '`)
  - cosine `0.797069`
  - top-k overlap `8/16`
- After: persistent decode matched much further and first mismatch moved to step 45.
  - CK token `5500` (`'_number'`)
  - llama token `82427` (`'_instructions'`)
  - cosine `0.998952`
  - top-k overlap `15/16`

## Fix 2: Qwen3-VL Prefill Safety Gate

After the M-RoPE fix, full replay at step 45 matched llama, but persistent decode still flipped top-1. Layer-0 hidden-state attribution showed:

| Boundary | Max abs | RMSE | Cosine |
| --- | ---: | ---: | ---: |
| `attn_out` | `5.60e-05` | `3.88e-06` | `0.99999988` |
| `out_proj` | `2.61e-04` | `7.25e-05` | `0.99999905` |
| `after_attn` | `2.61e-04` | `7.25e-05` | `0.99999946` |
| `ffn_norm` | `6.83e-04` | `1.51e-04` | `0.99999785` |
| `layer_out` | `1.18e-02` | `1.12e-03` | `0.99997771` |

This placed the material persistent/full-replay split inside layer-0 MLP.

Diagnostic result:

- With `CK_DISABLE_Q4K_GATEUP_SWIGLU_X16=1`, persistent decode matched llama through 64 tokens.
- Generated text begins with coherent OCR JSON:
  - `{"form_title": "Monthly Report", ...}`
- With the fused path enabled, the same run failed at step 45.

Production hardening:

- Qwen3-VL now defaults `ck_enable_q4k_gateup_swiglu_x16 = 0`.
- Other models keep the previous default.
- `CK_ENABLE_Q4K_GATEUP_SWIGLU_X16=1` can still explicitly enable the path for performance experiments.
- `CK_DISABLE_Q4K_GATEUP_SWIGLU_X16=1` remains a hard off switch.

## Validation Commands

Syntax:

```bash
/home/antshiv/Workspace/C-Kernel-Engine/.venv/bin/python -m py_compile \
  version/v8/scripts/codegen_prefill_v8.py \
  version/v8/scripts/codegen_core_v8.py \
  version/v8/scripts/build_ir_v8.py \
  version/v8/scripts/convert_gguf_to_bump_v8.py \
  version/v8/scripts/compare_multimodal_multitoken_logits_v8.py \
  version/v8/scripts/decoder_first_token_parity_v8.py
```

Default-safe Qwen3-VL AVX2 persistent parity:

```bash
CK_NUM_THREADS=20 OMP_NUM_THREADS=1 \
/home/antshiv/Workspace/C-Kernel-Engine/.venv/bin/python \
  version/v8/scripts/compare_multimodal_multitoken_logits_v8.py \
  --bridge-report /home/antshiv/Workspace/C-Kernel-Engine/build/qwen3vl_avx2_mrope128_bridge_20260709/bridge/bridge_report.json \
  --prefix-f32 /home/antshiv/Workspace/C-Kernel-Engine/build/qwen3vl_avx2_mrope128_bridge_20260709/prefix.f32 \
  --prefix-grid-x 36 \
  --prefix-grid-y 28 \
  --prefix-text-pos 41 \
  --workdir /home/antshiv/Workspace/C-Kernel-Engine/build/qwen3vl_avx2_mrope128_default_safe_20260709 \
  --ctx-len 4096 \
  --threads 20 \
  --top-k 16 \
  --max-new-tokens 64 \
  --append-on-divergence stop \
  --json-out /home/antshiv/Workspace/C-Kernel-Engine/build/qwen3vl_avx2_mrope128_default_safe_20260709/persistent64.json \
  --summary
```

Result:

```text
status=pass steps=64
first_mismatch=None
```

Generated C check:

```bash
rg -n "ck_enable_q4k_gateup_swiglu_x16 =" \
  /home/antshiv/Workspace/C-Kernel-Engine/build/qwen3vl_avx2_mrope128_default_safe_20260709/decoder/decoder_v8.c \
  /home/antshiv/Workspace/C-Kernel-Engine/build/qwen3vl_avx2_mrope128_default_safe_20260709/decoder/decoder_v8_prefill.c
```

Result:

```text
ck_enable_q4k_gateup_swiglu_x16 = 0
```

## Remaining Work

The fused Q4 gate-up/SwiGLU x16 prefill kernel still needs a real numerical parity fix. Keep it opt-in for Qwen3-VL until:

1. Persistent decode and full replay both match llama on the canonical OCR image.
2. The same result holds for at least the 5-image OCR smoke set.
3. Full OCR output quality does not regress.
4. The performance win is remeasured after correctness is restored.

