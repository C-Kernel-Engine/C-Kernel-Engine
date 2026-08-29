fix(v8/qwen3vl): align mrope width and gate fused prefill path

Why: Qwen3-VL AVX2 OCR decode diverged from llama.cpp on the canonical image after a correct-looking visual bridge. The failure was caused by an M-RoPE width mismatch first, then by Qwen3-VL taking a fused Q4 gate-up/SwiGLU prefill path whose live persistent decode parity was worse than the unfused path.

Validation: py_compile for modified v8 scripts, `make -B build/libckernel_engine.so`, canonical Qwen3-VL OCR persistent parity `status=pass steps=64`, and pre-commit `v8-regression-fast`.

Qwen3-VL M-RoPE was using `sum(mrope_sections)` as the rotary width.
That gives 64 for the 24/20/20/0 section layout, but llama.cpp passes
`n_rot == head_dim == 128`. The sections select the M-RoPE axis pattern;
they are not the rotary width.

This changes GGUF conversion and IR fallback defaults so Qwen3-VL emits
and consumes the full rotary width.

The AVX2 parity run then moved from a hard early failure to a late
numeric drift:

| Case | First mismatch | CK token | llama token | cosine | top-k overlap |
| --- | ---: | --- | --- | ---: | ---: |
| before M-RoPE fix | 4 | `262` (`'   '`) | `220` (`' '`) | `0.797069` | `8/16` |
| after M-RoPE fix | 45 | `5500` (`'_number'`) | `82427` (`'_instructions'`) | `0.998952` | `15/16` |

Layer-0 persistent-vs-full-replay attribution after the M-RoPE fix:

| Boundary | Max abs | RMSE | Cosine |
| --- | ---: | ---: | ---: |
| `attn_out` | `5.60e-05` | `3.88e-06` | `0.99999988` |
| `out_proj` | `2.61e-04` | `7.25e-05` | `0.99999905` |
| `after_attn` | `2.61e-04` | `7.25e-05` | `0.99999946` |
| `ffn_norm` | `6.83e-04` | `1.51e-04` | `0.99999785` |
| `layer_out` | `1.18e-02` | `1.12e-03` | `0.99997771` |

That places the remaining split inside layer-0 MLP. A diagnostic run with
`CK_DISABLE_Q4K_GATEUP_SWIGLU_X16=1` made the actual persistent OCR decode
match llama through 64 tokens. Therefore Qwen3-VL now defaults the fused
Q4 gate-up/SwiGLU x16 prefill path off until the fused kernel is fixed.
Other models retain the previous default, and the path can still be
explicitly enabled with `CK_ENABLE_Q4K_GATEUP_SWIGLU_X16=1`.

Validation:

```
.venv/bin/python -m py_compile \
  version/v8/scripts/codegen_prefill_v8.py \
  version/v8/scripts/codegen_core_v8.py \
  version/v8/scripts/build_ir_v8.py \
  version/v8/scripts/convert_gguf_to_bump_v8.py \
  version/v8/scripts/compare_multimodal_multitoken_logits_v8.py \
  version/v8/scripts/decoder_first_token_parity_v8.py
```

Canonical Qwen3-VL OCR parity:

```
CK_NUM_THREADS=20 OMP_NUM_THREADS=1 \
.venv/bin/python version/v8/scripts/compare_multimodal_multitoken_logits_v8.py \
  --bridge-report build/qwen3vl_avx2_mrope128_bridge_20260709/bridge/bridge_report.json \
  --prefix-f32 build/qwen3vl_avx2_mrope128_bridge_20260709/prefix.f32 \
  --prefix-grid-x 36 \
  --prefix-grid-y 28 \
  --prefix-text-pos 41 \
  --workdir build/qwen3vl_avx2_mrope128_default_safe_20260709 \
  --ctx-len 4096 \
  --threads 20 \
  --top-k 16 \
  --max-new-tokens 64 \
  --append-on-divergence stop \
  --json-out build/qwen3vl_avx2_mrope128_default_safe_20260709/persistent64.json \
  --summary
```

Result:

```
status=pass steps=64
first_mismatch=None
```

Research log:

`logs/2026-07-09-qwen3vl-avx2-mrope-prefill-parity/README.md`
