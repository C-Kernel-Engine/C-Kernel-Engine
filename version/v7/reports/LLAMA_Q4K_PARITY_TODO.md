# Llama Q4_K Parity TODO (Nanbeige 4.1 3B)

Status date: 2026-02-26
Scope: `hf://mradermacher/Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf`

## Update: 2026-03-09 (f16kv attention contract tightened)

### What changed

1. `attention_forward_*_flash_f16kv` now matches llama.cpp more closely.
- `src/kernels/attention_kernels.c` now mirrors three GGML flash-attention details that the older CK path skipped:
  - round `Q` through FP16 before the KQ dot when using the `f16kv` path
  - use the online softmax update shape
  - round the weighted `V` accumulator through FP16 on each update

### What improved

1. The old layer-0 out-proj amplification path is materially smaller on the checked `"Hello!"` sequence.
- Fresh direct probe replay on the rebuilt runtime shows:
  - token step 1 `__fattn__-0`: max diff dropped from about `1.17e-4` to about `3.31e-5`
  - token step 1 `attn_out-0`: max diff dropped from about `4.60e-4` to about `1.69e-5`
  - token step 1 `ffn_norm-0`: now passes at about `9.51e-6`
  - token step 2 `attn_out-0`: now essentially exact at about `2.38e-7`
  - token step 2 `ffn_norm-0`: now essentially exact at about `2.98e-7`

2. Layer-0 Q/K/V remain clean.
- Fresh direct probe replay still shows token steps 1 and 2 passing for:
  - `Qcur-0`
  - `Kcur-0`
  - `Vcur-0`

### What is still not fixed

1. Nanbeige is still not footer/logits clean.
- Despite the layer-0 improvement, footer probes still drift heavily on the same checked sequence:
  - token step 1 `result_norm`: max diff about `0.74`
  - token step 1 `result_output`: max diff about `0.90`
  - token step 2 `result_norm`: max diff about `0.86`
  - token step 2 `result_output`: max diff about `1.01`

### Updated interpretation

- This patch fixed a real layer-0 `f16kv` attention contract bug.
- But it did **not** finish Nanbeige parity end-to-end.
- The remaining problem is now better described as later-model cumulative drift:
  - layer 0 is much cleaner
  - footer/logits still diverge strongly
  - so the next bug is no longer best explained as only the old
    `__fattn__ -> q8_k -> wo -> ffn_norm` boundary at layer 0
  - the next investigation should isolate the first later decoder layer where the
    cleaned layer-0 path stops being enough

## Update: 2026-03-09 (first later-layer failure isolated)

### What changed

1. The local llama raw-dump hook can now dump selectable or all layers.
- `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` still supports `LLAMA_DUMP_LAYER0=1`.
- It now also supports:
  - `LLAMA_DUMP_LAYERS=all`
  - `LLAMA_DUMP_LAYERS=0,1,4-7`
- The dump-name table was also expanded so all-layer dumps do not silently truncate.

### New isolation result

1. The first stable later-layer failure is **not** footer-only.
- On the explicit token sequence for raw `"Hello!"` (`166100,23877,152518`), layer-0 remains clean.
- The first later-layer failure shows up immediately at **layer 1, token step 0**.

2. Layer-1 token-0 `attn_norm` is still essentially clean, but `q/k/v` already drift.
- Direct replay vs llama raw dumps shows:
  - `attn_norm-1`: still near-pass at about `9.01e-5`
  - `Qcur-1`: fail at about `2.05e-3`
  - `Kcur-1`: fail at about `2.52e-3`
  - `Vcur-1`: fail at about `6.66e-4`

3. That layer-1 token-0 `q/k/v` drift immediately poisons the first cache slot and then the later attention steps.
- Right after `kv_cache_store` for layer 1 on token step 0, cached values already differ from llama raw dumps by about:
  - `K cache[token 0]`: `~2.52e-3`
  - `V cache[token 0]`: `~6.66e-4`
- By token steps 1 and 2, current layer-1 `q/k/v` are close again, but `__fattn__-1` / `attn_out-1` still fail because the earlier cached token is already wrong.

4. Downstream layers then inherit the bad state quickly.
- Layer-1 `l_out` already fails at token step 0 (`~9.19e-3` max diff).
- On token step 2:
  - `l_out-0`: still passes (`~5.72e-6`)
  - `l_out-1`: first fail (`~9.57e-3`)
  - `l_out-2`: much worse (`~8.59e-2`)

### Updated interpretation

- The next real bug is earlier than the footer and earlier than the layer-1 attention combine itself.
- The first stable break is:
  - layer 1
  - token step 0
  - after `attn_norm`
  - in `q_proj / k_proj / v_proj`
- The most likely remaining root-cause class is now:
  - later-layer decode projection path on the first token
  - specifically the layer-1 token-0 `attn_norm -> quantize_row_q8_k -> q/k/v gemv` boundary
- `kv_cache_store` is still part of the observed failure chain, but the latest replay shows it is downstream of already-wrong layer-1 token-0 `q/k/v`.
- This means the next fix should focus on the **layer-1 token-0 qkv projection boundary**, not on footer/logits code first.

## Update: 2026-03-08 (root cause found and fixed)

### Major fixes landed

1. llama raw-dump parity hook had a thread-sync bug.
- In `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`, tensor dumps were taken by thread 0 immediately after `ggml_compute_forward(...)` and before the per-node barrier.
- That allowed raw dumps to capture partially-written tensors for multi-threaded nodes.
- After adding a pre-dump barrier under `LLAMA_DUMP_LAYER0=1`, repeated raw dumps became byte-identical for:
  - `__fattn__-0`
  - `attn_out-0`
  - `ffn_inp-0`
- Practical rule:
  - do not trust raw llama dumps until the dump path is synchronized.

2. Nanbeige/Llama was taking the wrong MLP kernel path.
- `build_ir_v7.py` had a Llama/Mistral-specific `prefer_fp32_mlp_matmuls` override.
- Built-in `templates/llama.json` set that override to `true`.
- That forced:
  - `mlp_gate_up` off `gemv_q4_k_q8_k`
  - `mlp_down` off `gemv_q4_k_q8_k` / `gemv_q6_k_q8_k`
- Direct replay on token 0 with the exact llama `ffn_norm-0` input showed:
  - FP32 `w1` path vs llama `ffn_gate-0`: max diff `~1.80e-2`
  - Q8_K `w1` path vs llama `ffn_gate-0`: max diff `~4.77e-7`
  - FP32 `w2` path vs llama `ffn_out-0`: max diff `~2.13e-2`
  - Q8_K `w2` path vs llama `ffn_out-0`: max diff `~4.77e-7`
- This was a real wrong-kernel-selection bug, not a generic numeric drift.

3. The Llama MLP override is now opt-in only.
- `build_ir_v7.py` no longer honors the template flag by default for Llama/Mistral families.
- `templates/llama.json` no longer sets `prefer_fp32_mlp_matmuls`.
- Fresh Nanbeige rebuild now lowers:
  - `quantize_input_1 -> quantize_row_q8_k`
  - `mlp_gate_up -> gemv_q4_k_q8_k`
  - `quantize_mlp_down_input -> quantize_row_q8_k`
  - `mlp_down -> gemv_q4_k_q8_k` or `gemv_q6_k_q8_k` per layer

### What improved after the fix

1. Layer-0 token-0 parity is now clean through the full block.
- Fresh `"Hello!"` token sequence:
  - `[166100, 23877, 152518]`
- Token 0 now passes all of:
  - `inp_embd`
  - `attn_norm-0`
  - `Qcur/Kcur/Vcur`
  - `__fattn__-0`
  - `attn_out-0`
  - `ffn_inp-0`
  - `ffn_norm-0`
  - `ffn_gate-0`
  - `ffn_up-0`
  - `ffn_swiglu-0`
  - `ffn_out-0`
  - `l_out-0`

2. End-to-end logits parity improved materially.
- `compare_first_token_logits.py` on `"Hello!"` tokens now reports:
  - `top1_match=true`
  - `topk_overlap=15/16`
  - `cosine≈0.9978`
- CK runtime output for `"Hello!"` is no longer degenerate punctuation and now starts with coherent text.

3. Non-Llama families remain green.
- `validate_model_matrix_v7.py --context-len 1024` still reports:
  - `qwen2-0.5b: PASS`
  - `qwen3-0.6b: PASS`
  - `gemma3-270m: PASS`

### Current remaining drift

1. The first stable failure is no longer layer-0 MLP on token 0.
- It moved to footer/output accumulation:
  - token 0 first stable fail: `result_norm`
  - token 0 `result_output` still differs, but much less than before

2. Multi-token layer-0 drift still exists, but it is smaller.
- On token steps 1 and 2:
  - `__fattn__-0` stays within pass tolerance
  - `attn_out-0` stays within pass tolerance
  - `ffn_norm-0` and downstream MLP activations still drift mildly
- Interpretation:
  - the major Llama-family stitching bug was the MLP kernel path
  - the remaining issue is now a later/deeper accumulation problem, not the original layer-0 kernel contract bug

## Update: 2026-03-08

### What is now cleared

1. Decode attention kernel math is not the main culprit.
- Fresh focused decode checks on layer 0 pass for:
  - tokens `[1,2]`
  - tokens `[1,2,3]`
  - tokens `[1,2,3,4]`
  - tokens `[1,2,3,4,5]`
- Verified boundaries:
  - post-RoPE `q`
  - post-RoPE `k`
  - KV cache store
  - decode attention output
- Conclusion:
  - flash-attention decode path is internally correct on the checked Nanbeige decode steps.
  - this does not support a "KV cache corruption" or "RoPE kernel still wrong" hypothesis.

2. RMSNorm parameters are correct and RMSNorm math is not the main culprit.
- `config.json` uses `rms_norm_eps=1e-5`, and generated layer-0 LN1/LN2 both receive that exact value.
- Direct scalar recompute from CK `ffn_inp-0`, `layer.0.ln2_gamma`, and `eps=1e-5` matches CK `ffn_norm-0` to about `1e-7`.
- Conclusion:
  - LN2 epsilon/gamma wiring is correct.
  - the RMSNorm kernel is reproducing CK math exactly.
  - when `ffn_norm` differs from llama.cpp, it is because its input already differs, not because RMSNorm itself is wrong.

3. `out_proj` kernel is also not the main kernel bug.
- Layer-0 `attn_out` uses:
  - `quantize_row_q8_k`
  - `gemv_q4_k_q8_k` on `layer.0.wo`
- Direct scalar/reference replay with `gemv_q4_k_q8_k_ref` matches CK `attn_out` to about `1e-7`.
- Conclusion:
  - the `wo` matvec kernel itself is behaving correctly.

4. Layer-0 QKV projections are clean on real `"Hello!"` tokens too.
- CK tokenization for `"Hello!"` produced:
  - `[166100, 23877, 152518]`
- Focused contract checks passed for:
  - decode token `152518` with prefill `[166100, 23877]`
  - prefill `[166100, 23877]`
- Verified boundaries:
  - `q_proj`
  - `k_proj`
  - `v_proj`
- Conclusion:
  - the remaining Nanbeige drift is not an embedding lookup bug and not a late-token projection bug.

5. Temporarily bypassing the `q8_k` activation path for `wo` is not a fix.
- Added a debug-only env gate:
  - `CK_V7_DEBUG_OUTPROJ_FP32=1`
- Result on the `"Hello!"` sequence:
  - `attn_out-0` got worse, not better
  - the focused boundary checker reported `ref_vs_ck out_proj` mismatch under the FP32 bypass
- Interpretation:
  - llama.cpp is expected to use the quantized `q8_k` activation path for `Q4_K/Q6_K` matmuls here
  - the `q8_k` `wo` path is not a mistaken kernel choice
  - a raw FP32-input `wo` path should be treated only as a debugging ablation, not a candidate fix

6. A separate Python-tokenizer contract bug can masquerade as runtime parity drift.
- `scripts/ck_chat.py --python-tokenizer` was loading legacy `vocab.json` without
  reapplying the SentencePiece `add_space_prefix` contract.
- That made the Python fallback encode bare `"Hello!"` as:
  - `[166100, 17782, 152518]`
  instead of the correct llama.cpp / C-tokenizer path:
  - `[166100, 23877, 152518]`
- Consequence:
  - explicit-token parity checks and live `ck_chat.py` runs were not evaluating the same prefix
  - user-visible output looked more inconsistent than the generated runtime actually was
- Fix:
  - `scripts/gguf_tokenizer.py` now persists and honors `add_space_prefix`
  - `scripts/ck_chat.py` reapplies tokenizer contract fields from nearby
    `weights_manifest.json` / `config.json` so old cached `vocab.json` files
    still tokenize correctly

### What is still true

1. Fresh llama.cpp sequence dumps still show a stable small drift at/after `attn_out`, then a larger visible drift at `ffn_norm`, then amplification through the MLP.
- On fresh layer-0 sequence replay, `__fattn__-0` stays very close to llama.cpp.
- The first consistently useful divergence is:
  - tiny at `attn_out-0`
  - identical at `ffn_inp-0`
  - larger and stable at `ffn_norm-0`
  - then amplified in `ffn_gate/up/down`

2. Some raw llama dump probes are not trustworthy enough to drive first-failure attribution by themselves.
- `Qcur-0` occurrence mapping was not stable across dump sets for token-step 1.
- Older and newer dump directories disagreed on which `Qcur` occurrence aligned with CK pre/post-RoPE tensors.
- The sequence parity report should track:
  - earliest raw failure
  - earliest stable non-duplicate failure
- Practical rule:
  - do not treat `Qcur/Kcur` duplicate-occurrence failures as root cause until they are cross-checked with the focused decode checker.

### Current best explanation

- The main Nanbeige path is now:
  - attention math very close to llama.cpp
  - tiny pre-`out_proj` drift at `__fattn__`
  - `quantize_row_q8_k + gemv_q4_k_q8_k` on `wo` magnifies that tiny drift into a visible `attn_out` mismatch
  - `ffn_norm` reflects that mismatch exactly
  - MLP then amplifies it enough to break final logits parity

- So the next major culprit is not:
  - flash attention
  - KV cache store
  - RoPE kernel
  - RMSNorm epsilon
  - RMSNorm kernel math
  - `wo` gemv kernel correctness

- The next major culprit is the sensitive boundary:
  - `__fattn__ -> quantize_row_q8_k(out_proj input) -> wo q4_k x q8_k -> ffn_norm -> MLP`

### Next best investigation

1. Treat llama.cpp-vs-CK attention-output numeric drift as the next highest-value target.
- `Q/K/V` are now cleared.
- `wo` q4_k x q8_k is the intended path.
- The remaining question is why `__fattn__` is still slightly different from llama.cpp before `wo` amplification.

2. Do not over-interpret raw llama dump internals until dump semantics are tighter.
- A naive scalar replay from llama raw `Qcur/Kcur/Vcur` did not reconstruct llama `__fattn__` exactly.
- That suggests at least one of these is still ambiguous in the raw dump path:
  - occurrence selection
  - tensor layout interpretation
  - exact internal checkpoint/debug dump semantics

### Fast bringup workflow for future model families

1. Regenerate runtime artifacts first.
- Do not trust old `.ck_build` reports after template/rope/head-contract changes.

2. Run the focused decode checker before trusting raw sequence dumps.
- If decode checker passes on short histories (`[1,2]`, `[1,2,3]`, `[1,2,3,4]`), do not start by blaming flash attention or KV cache.

3. Re-run fresh llama raw dumps before using occurrence-based probes.
- Duplicate raw names (`Qcur/Kcur/norm`) are useful but not authoritative on their own.

4. When `ffn_norm` is first bad, scalar-recompute it immediately.
- If scalar RMSNorm matches CK exactly, stop debugging epsilon/gamma and move upstream/downstream.

5. When `attn_out` is first visibly bad but `__fattn__` is clean, run scalar `wo` replay.
- Use `quantize_row_q8_k` + `gemv_q4_k_q8_k_ref`.
- If that matches CK exactly, the problem is not the output-projection kernel; it is sensitivity to tiny upstream differences.

6. Treat the first major actionable boundary as the first place tiny drift becomes materially amplified, not necessarily the first raw probe with a FAIL label.

## Current Findings

1. Llama parity report is not actionable yet (WARN-only path).
- Evidence: `.ck_build/detailed_parity_analysis_latest.json` contains:
  - `"Skipped ck_run; used existing dumps."`
  - `"Missing llama dump: .../llama_parity_dumps/dump.bin"`
- Impact: We do not have CK-vs-llama tensor diffs for prefill/decode.

2. ~~Projection contract checker fails before numeric diff on Q4_K.~~ **FIXED**
- `check_layer0_qkv_contract.py` now includes `q4_k` in the supported dtype list alongside `q6_k`, `q5_0`, `q8_0`.
- Script runs without `unsupported dtype q4_k` on Nanbeige.

3. Existing autopsy points to projection checks as blocker.
- Evidence: `.ck_build/parity_autopsy_latest.md`:
  - `q_proj rc=1 pass_contract=False`
  - `k_proj rc=1 pass_contract=False`
  - `v_proj rc=1 pass_contract=False`
- Impact: attention/rope/debugging is premature until projection contract path is fixed.

4. Llama default context can cause local OOM kill in llama.cpp smoke runs.
- Evidence: `llama-cli --model ...` without `-c` gets `-Killed` on this machine.
- Impact: false "runtime crash" signal if operator does not cap context.

## Priority TODO

## P0: Make parity run produce real llama reference dumps every time — **FIXED**
- [x] Update `ck_run_v7.py` detailed parity flow to fail-fast if llama dump file is absent, not WARN-only.
  - `ck_run_v7.py` line ~9020: changed from `log(..., C_ORANGE)` to `log_error(...)` + `sys.exit(1)`.
- [x] Add a post-check in `detailed_parity_analysis.py`: returns exit code 2 (not 0) when either the llama or CK reference dump is absent, and prints `[analysis] INCOMPLETE: missing ...` before exiting.
- [ ] Ensure `llama_parity_dumps/dump.bin` is created in run dir on every `--detailed-llamacpp-parity` invocation.
  - Remaining: investigate why `_run_llamacpp_parity()` returns False on this model (binary path, GGUF path, or parity binary crash).

Pass criteria:
- `detailed_parity_analysis_latest.json` contains non-empty `prefill_summary`/`decode_summary`.
- `parity_autopsy_latest.md` no longer says "llama reference dump missing/empty".
- `ck_run_v7.py --detailed-llamacpp-parity` exits non-zero when llama dump cannot be produced.

## P0: Add Q4_K support to layer0 QKV contract checker — **FIXED**
- [x] Extended `version/v7/scripts/parity/check_layer0_qkv_contract.py` to handle `dtype=q4_k`.
- [ ] Wire CK and llama parity C symbols for Q4_K projection path (numeric diff still TBD).
- [ ] Emit `max_diff`/`mean_diff` for q_proj/k_proj/v_proj instead of `None`.

Pass criteria:
- Script runs on Nanbeige without `unsupported dtype q4_k`. ✓
- Contract report returns numeric diffs and pass/fail gates for all three projections.

## P1: Lock operator commands to safe context defaults for llama.cpp
- [ ] In runbook snippets, always include `-c 1024` (or explicit bounded `-c`).
- [ ] Add note: model-default context may exceed local RAM and trigger OS kill.

Pass criteria:
- Operator smoke commands do not OOM on default laptop configuration.

## P1: Metadata sanity checks in detailed parity output
- [ ] Ensure `weights`/`ck_coverage` sections are populated (not null) when dumps exist.
- [ ] If null, fail report generation with explicit reason and remediation.

Pass criteria:
- `detailed_parity_analysis_latest.json` has populated `weights` and coverage fields.

## P2: Tokenizer verification for llama-family GGUF runs
- [ ] Add tokenizer sanity step in `template-audit` for llama-family:
  - compare prompt token IDs CK vs llama.cpp on 3 fixed prompts.
- [ ] Emit a compact mismatch report in JSON.

Pass criteria:
- Tokenization mismatches are visible before full decode parity run.

## Suggested Validation Commands

1. Template + structure audit:
```bash
python3 version/v7/scripts/ck_run_v7.py template-audit \
  ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF \
  --run /tmp/v7_nanbeige_template_audit \
  --context-len 1024 \
  --force-compile
```

2. Full detailed parity (must produce llama dump; now fails-fast if not):
```bash
python3 version/v7/scripts/ck_run_v7.py run \
  ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF \
  --context-len 1024 \
  --detailed-llamacpp-parity \
  --max-tokens 32
```

3. QKV contract check (Q4_K now supported):
```bash
.venv/bin/python version/v7/scripts/parity/check_layer0_qkv_contract.py \
  --model-dir ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF/.ck_build \
  --layer 0 --skip-prefill --tol 1e-3
```

4. llama.cpp bounded-memory smoke:
```bash
cd llama.cpp
build/bin/llama-cli \
  --model ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf \
  -c 1024 -n 64 -p "Hello" --temp 0.0
```

5. Standalone detailed_parity_analysis (now returns exit 2 when dump absent):
```bash
python3 version/v7/scripts/detailed_parity_analysis.py \
  --model-dir ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF/.ck_build \
  --family llama
echo "exit: $?"
```
