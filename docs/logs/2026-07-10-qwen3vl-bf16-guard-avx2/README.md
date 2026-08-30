# Qwen3-VL BF16 Guard Patch and AVX2 GGUF Recheck

Date: 2026-07-10

This log records the AVX2-side validation of the Xeon BF16/Qwen3-VL parity guard patch.
The patch is useful as BF16 safetensors lowering/tooling coverage, but it does not close
the AVX2 GGUF Qwen3-VL encoder parity issue.

## Patch Scope

- Add Qwen3-VL BF16 safetensors lowering guardrails.
- Add/extend Qwen3-VL BF16 and prefix parity tooling.
- Extend safetensors mapping for Qwen3-VL vision tensors.
- Adjust shared M-RoPE vision handling in `src/kernels/rope_kernels.c`.

This should be described as a BF16 parity guard/tooling patch, not as full Qwen3-VL
parity closure.

## AVX2 Validation

Ran in a clean worktree from latest `origin/main` with the patch applied.

Syntax and diff checks:

```bash
.venv/bin/python -m py_compile \
  version/v8/scripts/bf16_safetensors_lowering_guard_v8.py \
  version/v8/scripts/compare_qwen3vl_bf16_vision_hidden_v8.py \
  version/v8/scripts/qwen3vl_encoder_prefix_parity_suite_v8.py \
  version/v8/scripts/numeric_parity_qwen3vl_mmproj_v8.py \
  version/v8/scripts/convert_safetensors_to_bump_v8.py \
  version/v8/scripts/replay_attention_from_dumps_v8.py \
  version/v8/scripts/stitched_parity_v8.py

git diff --check
```

Kernel and BF16 guards:

```bash
make test-v8-vision-kernels
make test-bf16
make nightly-bf16
```

Results:

- `make test-v8-vision-kernels`: pass
- `make test-bf16`: pass on AVX2 host; AVX-512 BF16-only tests skip as expected
- `make nightly-bf16`: 12/12 pass after BF16 libraries are built

## GGUF Stitched Parity Recheck

The shared M-RoPE code is used outside the BF16 lane, so Qwen3-VL GGUF was rechecked
against llama.cpp/mtmd with the canonical OCR image.

Configuration:

- decoder: `Qwen3VL-8B-Instruct-Q4_K_M.gguf`
- mmproj: `mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`
- image: canonical PPM derived from `ocr/1 81.jpg`
- image SHA: `81ab4a6dabf3fc41ef86b620602a28dbbba59aea89414301b604df355afda182`
- prompt: `Extract visible form fields as compact JSON.`
- ctx: 4096
- threads: 20
- image max tokens: 1024

Command shape:

```bash
CK_NUM_THREADS=20 OMP_NUM_THREADS=1 \
.venv/bin/python version/v8/scripts/stitched_parity_v8.py \
  --template qwen3vl \
  --mode fast \
  --decoder-gguf <Qwen3VL-8B-Instruct-Q4_K_M.gguf> \
  --mmproj-gguf <mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf> \
  --image-path <1_81_canonical.ppm> \
  --prompt 'Extract visible form fields as compact JSON.' \
  --ctx-len 4096 \
  --threads 20 \
  --top-k 16 \
  --max-new-tokens 64 \
  --image-max-tokens 1024 \
  --phase-timeout-sec 1800 \
  --log-byte-limit 1048576
```

Result:

```text
status=fail
stage=encoder_numeric
cosine=0.9995497810308069
rmse=0.0048769261972719455
max_abs=0.35991716384887695
granular_layer=0
granular_op=attn_output
```

Bridge geometry stayed correct:

```text
prefix_tokens=1008
prefix_embed_dim=16384
prefix_grid_x=36
prefix_grid_y=28
prefix_text_pos=41
prompt_tokens_before_image=5
prompt_tokens_after_image=14
```

First granular issue:

```text
layer=0
op=attn_output
max_abs_diff=0.000552058219909668
mean_abs_diff=7.441822447162849e-08
diverge_idx=112909
ref_shape=[4644864]
test_shape=[4644864]
```

## Interpretation

This patch does not regress AVX2 GGUF Qwen3-VL parity, but it also does not fix it.
The remaining AVX2 encoder issue is still in the vision/mmproj path, first reported by
the stitched harness at layer-0 `attn_output`.

Earlier fixes made the visual geometry and several early boundaries tight enough:
position embedding order, patch/bias/ln1/qkv/RoPE/kqv-style boundaries were improved or
guarded. The current failure is later than the original position-embedding bug and is
not evidence that those fixes regressed.

Next target:

- continue granular attribution inside layer-0 attention output semantics;
- compare CK and llama dump boundaries for attention context/output projection;
- do not resume Qwen3-VL speed work until this GGUF encoder parity issue is closed.
