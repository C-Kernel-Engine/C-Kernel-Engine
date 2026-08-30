# Qwen3-VL OCR Parity Handoff - 2026-07-08

This note captures the current Qwen3-VL OCR parity investigation so the next agent can continue without repeating the same search. The target is CK v8 Qwen3-VL OCR against llama.cpp on real SDPR form images.

## Scope

Runtime under test:

- CK v8 Qwen3-VL decoder and mmproj path.
- llama.cpp `llama-mtmd-cli` / local mtmd shim as reference.
- Real OCR form image used for focused debugging: `build/sdpr_ck_ocr_report_full/ppm/1_81.ppm`.
- Decoder GGUF: `.cache/ck-engine-v8/models/Qwen--Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf`.
- MMProj GGUF: `.cache/ck-engine-v8/models/Qwen--Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`.

Do not commit model files, OCR images, generated dumps, or build outputs.

## Current Diagnosis

The original OCR failure looked like weak visual grounding: CK often emitted parseable JSON but filled fewer real fields than llama.cpp. The focused parity work has narrowed that substantially.

What is no longer the likely root cause:

- OCR image patch count or geometry.
- PPM/JPEG loader as the main problem.
- Tokenizer or JSON formatting.
- Immediate decoder collapse.
- Qwen3-VL visual position ID shape.

Current root is much narrower:

- CK's Qwen3-VL image prefix path was numerically drifting in mmproj layer 0.
- Position embedding interpolation was one concrete mismatch. After matching llama.cpp-style float evaluation order, `inp_pos_emb` is bit-exact on the bad OCR image.
- Layer-0 `Qcur`, `Kcur`, `Vcur`, `Qcur_rope`, and `Kcur_rope` are now tight, around 1e-6.
- Layer-0 attention pre-output projection tensor `kqv_out` passes with max diff about `1.19e-6`.
- The remaining visible mismatch appears after q8 output projection: the tiny `kqv_out` difference can amplify to about `5e-4`.

So the remaining OCR drift is not a broad bridge failure. It is a small visual-prefix numerical parity issue around Q8 attention output projection / activation quantization.

## Code Changes In This Patch

### `src/kernels/vision_kernels.c`

`position_embeddings_add_tiled_2d` now forces llama.cpp-compatible float32 evaluation order for Qwen3-VL position embedding interpolation:

- Adds `CK_VISION_NOINLINE` and `CK_VISION_OPTNONE`.
- Adds `ck_force_f32`.
- Forces intermediate scale/support/index arithmetic through float32.
- Uses `fmaf(sample, weight, val)` in the interpolation accumulation.

This changed the bad OCR image result from a small but significant `inp_pos_emb` mismatch to bit-exact parity against llama.cpp.

### `src/kernels/attention_oracle_ggml.c`

Strict ggml attention oracle debug path no longer feeds a contiguous copy into softmax just because score dumping is enabled. The contiguous `kq_scores_dump` tensor is now only for optional dumps/meta; the softmax consumes the original score tensor.

This is primarily parity-debug hygiene. In the current focused test it did not change `kqv_out`, but it avoids a debug-only graph perturbation when score dumps are enabled.

## Key Commands

All commands were run from repo root.

### Position embedding parity

```bash
env CK_STOP_OP=8 \
  CK_PARITY_LAYER_FILTER=-1 \
  CK_PARITY_OP_FILTER=patch_bias,inp_pos_emb \
  CK_LLAMA_CPP_ROOT=/opt/app-root/src/Software/llama.cpp \
  .venv/bin/python -u version/v8/scripts/activation_parity_qwen3vl_mmproj_v8.py \
  --gguf /opt/app-root/src/.cache/ck-engine-v8/models/Qwen--Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
  --output-dir build/qwen3vl_1_81_activation_layer4_v1 \
  --image-path build/sdpr_ck_ocr_report_full/ppm/1_81.ppm \
  --image-min-tokens 1024 --image-max-tokens 1024 \
  --threads 20 --ck-threads 20 \
  --activation-pref qkv_packed_proj=q8_0 \
  --activation-pref out_proj=q8_0 \
  --activation-pref mlp_up=q8_0 \
  --prefer-q8-0-contract \
  --llama-dump-names patch_bias,inp_pos_emb \
  --llama-dump-layer -1 --quiet \
  --report build/qwen3vl_1_81_activation_patch_pos_fmaf_v1.json
```

Result:

- `patch_bias`: max diff `0`.
- `inp_pos_emb`: max diff `0`.

### Layer-0 Q/K/V parity

```bash
env CK_STOP_OP=11 \
  CK_PARITY_LAYER_FILTER=0 \
  CK_PARITY_OP_FILTER=ln1,Qcur,Kcur,Vcur \
  CK_LLAMA_CPP_ROOT=/opt/app-root/src/Software/llama.cpp \
  .venv/bin/python -u version/v8/scripts/activation_parity_qwen3vl_mmproj_v8.py \
  --gguf /opt/app-root/src/.cache/ck-engine-v8/models/Qwen--Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
  --output-dir build/qwen3vl_1_81_activation_layer4_v1 \
  --image-path build/sdpr_ck_ocr_report_full/ppm/1_81.ppm \
  --image-min-tokens 1024 --image-max-tokens 1024 \
  --threads 20 --ck-threads 20 \
  --activation-pref qkv_packed_proj=q8_0 \
  --activation-pref out_proj=q8_0 \
  --activation-pref mlp_up=q8_0 \
  --prefer-q8-0-contract \
  --strict-parity \
  --llama-dump-names ln1,Qcur,Kcur,Vcur \
  --llama-dump-layer 0 --quiet \
  --report build/qwen3vl_1_81_activation_layer0_qkv_fmaf_v1.json
```

Observed max diffs:

- `ln1`: about `9.54e-7`.
- `q_proj`: about `1.43e-6`.
- `k_proj`: about `1.67e-6`.
- `v_proj`: about `1.19e-6`.

Forcing `qkv_packed_proj=fp32` was worse, so Q8 activation contract remains the llama-compatible direction.

### Layer-0 attention boundary

```bash
env CK_STOP_OP=15 \
  CK_PARITY_LAYER_FILTER=0 \
  CK_PARITY_OP_FILTER=Qcur_rope,Kcur_rope,attn_out,ffn_inp \
  CK_LLAMA_CPP_ROOT=/opt/app-root/src/Software/llama.cpp \
  .venv/bin/python -u version/v8/scripts/activation_parity_qwen3vl_mmproj_v8.py \
  --gguf /opt/app-root/src/.cache/ck-engine-v8/models/Qwen--Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
  --output-dir build/qwen3vl_1_81_activation_layer4_v1 \
  --image-path build/sdpr_ck_ocr_report_full/ppm/1_81.ppm \
  --image-min-tokens 1024 --image-max-tokens 1024 \
  --threads 20 --ck-threads 20 \
  --activation-pref qkv_packed_proj=q8_0 \
  --activation-pref out_proj=q8_0 \
  --activation-pref mlp_up=q8_0 \
  --prefer-q8-0-contract \
  --strict-parity \
  --llama-dump-names Qcur_rope,Kcur_rope,attn_out,ffn_inp \
  --llama-dump-layer 0 --quiet \
  --report build/qwen3vl_1_81_activation_layer0_attn_fmaf_strict_v1.json
```

Observed:

- `Qcur_rope`: pass, bit-exact in strict run.
- `Kcur_rope`: pass, bit-exact in strict run.
- `attn_output`: fail, max diff about `5.52e-4`.
- `ffn_inp`: fail, same boundary.

### Isolated pre-output-projection attention tensor

```bash
env CK_STOP_OP=13 \
  CK_PARITY_LAYER_FILTER=0 \
  CK_PARITY_OP_FILTER=kqv_out \
  CK_LLAMA_CPP_ROOT=/opt/app-root/src/Software/llama.cpp \
  .venv/bin/python -u version/v8/scripts/activation_parity_qwen3vl_mmproj_v8.py \
  --gguf /opt/app-root/src/.cache/ck-engine-v8/models/Qwen--Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
  --output-dir build/qwen3vl_1_81_activation_layer4_v1 \
  --image-path build/sdpr_ck_ocr_report_full/ppm/1_81.ppm \
  --image-min-tokens 1024 --image-max-tokens 1024 \
  --threads 20 --ck-threads 20 \
  --activation-pref qkv_packed_proj=q8_0 \
  --activation-pref out_proj=q8_0 \
  --activation-pref mlp_up=q8_0 \
  --prefer-q8-0-contract \
  --strict-parity \
  --llama-dump-names kqv_out \
  --llama-dump-layer 0 --quiet \
  --report build/qwen3vl_1_81_activation_layer0_kqv_fmaf_strict_v1.json
```

Observed:

- `kqv_out`: pass, max diff about `1.19e-6`, mean diff about `5.04e-9`.

Disabling the multihead strict oracle produced the same `kqv_out` result, so the current issue is not specific to the packed multihead oracle.

## Current Open Issue

The current bottleneck is correctness, not speed:

1. `kqv_out` is close but not bit-exact.
2. The following q8 output projection amplifies that small difference.
3. Long OCR decode then flips around later tokens even when early tokens match llama.cpp.

Next diagnostic target:

- Dump or compare the exact Q8 activation quantization input/output around layer-0 `out_proj`.
- Confirm whether llama.cpp quantizes `kqv_out` with the same row boundaries, row order, and rounding as CK.
- If row quantization matches, compare q8_0 x q8_0 output projection math itself.
- Keep dumps limited to one layer or one row; disk is tight and full mixed-prefix dumps can be multi-GB.

## How To Continue

1. Run the focused parity commands above after applying this patch.
2. Run 3 known bad OCR images end-to-end only after position embedding parity is confirmed.
3. If accuracy still lags llama.cpp, instrument only layer-0 `out_proj` Q8 quantization and matmul.
4. Avoid broad `CK_PARITY_ALL` or full mixed-prefix dumps unless there is at least 10 GB free.
5. Once the layer-0 visual prefix is tighter, rerun the 40-image OCR comparison against the existing llama.cpp baseline.

## AVX2 Local Application Note

Applied and checked on the AVX2 host in an isolated clone on top of
`origin/main` commit `48db6ead`.

Local validation passed:

- `git apply --check qwen3vl-ocr-parity-handoff-2026-07-08.patch`
- `make -B build/libckernel_engine.so`
- `git diff --check`
- `.venv/bin/python -m py_compile version/v8/scripts/activation_parity_qwen3vl_mmproj_v8.py version/v8/scripts/compare_multimodal_multitoken_logits_v8.py`
- `.venv/bin/python unittest/test_vision.py`
- `make test-kernels` with the existing local llama.cpp helper checkout: `58/58`
- Pre-commit staged snapshot: `v8-regression-fast` passed.

The full pre-commit hook was not accepted because `v7-regression-fast` failed.
That failure was reproduced on an untouched `48db6ead` baseline clone with the
same summary: Gemma and Qwen2 first-token parity failed, Nanbeige first-token
parity failed, and Nanbeige coherence failed. This is a pre-existing v7 gate
issue, not introduced by the Qwen3-VL vision patch.

Focused Qwen3-VL parity was not rerun on this AVX2 host because the required
local artifacts were not present:

- `build/sdpr_ck_ocr_report_full/ppm/1_81.ppm`
- `~/.cache/ck-engine-v8/models/Qwen--Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`
- `~/.cache/ck-engine-v8/models/Qwen--Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf`

## Expected Patch Review

The vision interpolation change is the main correctness fix. The attention oracle change is debug hygiene and should be reviewed as parity scaffolding, not a performance change.

Suggested commit title:

```text
fix(v8): tighten qwen3-vl vision position parity
```
